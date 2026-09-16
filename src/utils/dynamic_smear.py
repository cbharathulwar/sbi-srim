"""On-the-fly, per-batch, CONTINUOUS smearing for domain-randomized training.

Unlike smearing_resolution's approach (a handful of precomputed discrete
sigma tiers, used for the one-shot warm-started fine-tune), this module
draws a genuinely continuous sigma per TRACK, per BATCH, every training
step, and does the smear -> center -> physics-descriptors -> normalize ->
kNN pipeline live -- the exact same per-track algorithm as
preprocess_egnn() in data_utils.py (replicated faithfully so train-time
corruption matches eval-time corruption exactly), just computed on demand
instead of cached for a fixed set of sigma values.

Why continuous instead of a bigger discrete grid: with a 3-4 day A100
budget, the CPU cost of this (~0.1s/batch, see smoke-test timing) is a
small fraction of total step time, and a continuous sigma distribution is
a stronger, harder-to-argue-with generalization claim than "we tested N
fixed noise levels" -- a reviewer can't ask "but what about the resolution
between your tiers" when there's no tiers.

Two-part API:
  load_raw_tracks(csv_path)         -- ONE-TIME: load unsmeared per-track
                                        point clouds + theta into RAM.
  build_batch_dynamic(...)          -- EVERY BATCH: draw one sigma per
                                        track, smear+prepare, assemble x_flat.
"""
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree


def load_raw_tracks(csv_path, k_neighbors=16, max_points="auto"):
    """Replicates preprocess_egnn's Pass 1 (track boundaries, n_vac>=3 filter,
    max_points/n_max determination) but WITHOUT smearing/centering/kNN --
    those happen per-batch in build_batch_dynamic. Returns raw, unsmeared
    per-track point arrays (variable length, float64, Angstrom) plus theta.

    Returns:
        raw_points: list of (n_i, 3) float64 arrays, one per valid track
        theta: (N, 4) float32 tensor [energy_keV, Vx, Vy, Vz]
        n_max: int, padding length (95th-percentile track length, or as given)
        max_points_used: int, truncation length actually used
    """
    import pandas as pd
    csv_path = Path(csv_path)
    use_cols = ['x', 'y', 'z', 'ion_number', 'energy_keV',
                'target_vx', 'target_vy', 'target_vz']
    print(f"[DYNAMIC] Loading raw tracks from: {csv_path}")
    df = pd.read_csv(csv_path, usecols=use_cols)
    df.sort_values('ion_number', inplace=True)

    if max_points == "auto":
        track_lengths = df.groupby('ion_number').size()
        max_points = int(track_lengths.quantile(0.95))
        print(f"[DYNAMIC] Track lengths -> max={track_lengths.max()}, "
              f"95%={max_points}, tracks={len(track_lengths):,}")
        del track_lengths

    ion_ids = df['ion_number'].values
    xyz = df[['x', 'y', 'z']].values.astype(np.float64)
    energies = df['energy_keV'].values
    vx = df['target_vx'].values
    vy = df['target_vy'].values
    vz = df['target_vz'].values
    del df
    import gc; gc.collect()

    boundaries = np.where(np.diff(ion_ids) != 0)[0] + 1
    track_starts = np.concatenate([[0], boundaries])
    track_ends = np.concatenate([boundaries, [len(ion_ids)]])

    raw_points = []
    theta_list = []
    lengths = []
    for t_idx in range(len(track_starts)):
        s, e = track_starts[t_idx], track_ends[t_idx]
        n = e - s
        if n < 3:
            continue
        # IMPORTANT: keep the FULL untruncated track here. preprocess_egnn
        # computes the physics descriptors (log n_vac, log extent, log R_gyration,
        # elongation) on the FULL centered track BEFORE truncating -- truncating
        # at load time (as an earlier version of this function did) silently
        # corrupts those descriptors for any track longer than max_points
        # (caught by a direct sigma=0 cross-check against preprocess_egnn: one
        # mismatch out of 5 spot-checked tracks, exactly the one at the
        # truncation boundary). Truncation happens later, per-call, in
        # _smear_and_prepare_one, in the same order preprocess_egnn uses.
        pts = xyz[s:e]
        raw_points.append(pts)
        lengths.append(min(n, max_points))
        theta_list.append([energies[s], vx[s], vy[s], vz[s]])

    n_max = max(lengths)
    theta = torch.tensor(theta_list, dtype=torch.float32)
    print(f"[DYNAMIC] {len(raw_points):,} valid tracks, N_max={n_max}, "
          f"max_points={max_points}")
    return raw_points, theta, n_max, max_points


def _smear_and_prepare_one(points, sigma_A, k_neighbors, n_max, rng):
    """Exact replica of preprocess_egnn's Pass-2 per-track body (data_utils.py
    ~lines 1138-1188): smear -> center -> physics descriptors (pre-normalize)
    -> normalize by max_extent -> pad -> kNN via cKDTree.

    Returns:
        coords: (n_max, 3) float32, zero-padded
        knn: (n_max, k_neighbors) int16, -1 padded
        phys: (5,) float32
    """
    if sigma_A > 0:
        points = points + rng.normal(0.0, sigma_A, size=points.shape)

    centroid = points.mean(axis=0)
    centered = points - centroid

    n_vac_full = centered.shape[0]
    max_extent = np.abs(centered).max()
    sq_radii = (centered ** 2).sum(axis=1)
    r_gyration = np.sqrt(sq_radii.mean())
    cov = (centered.T @ centered) / n_vac_full
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    lam_sum = float(evals.sum()) + 1e-12
    elong1 = float(evals[2]) / lam_sum
    elong2 = float(evals[1]) / lam_sum
    phys = np.array([
        np.log(n_vac_full + 1.0),
        np.log(max_extent + 1e-6),
        np.log(r_gyration + 1e-6),
        elong1, elong2,
    ], dtype=np.float32)

    if max_extent > 0:
        centered = centered / max_extent

    # Truncate AFTER physics descriptors are computed on the full track --
    # matches preprocess_egnn's order exactly (see load_raw_tracks note).
    if len(centered) > n_max:
        centered = centered[:n_max]

    n = len(centered)
    coords = np.zeros((n_max, 3), dtype=np.float32)
    coords[:n] = centered.astype(np.float32)

    knn = np.full((n_max, k_neighbors), -1, dtype=np.int16)
    if n > 1:
        k_use = min(k_neighbors, n - 1)
        tree = cKDTree(coords[:n])
        _, indices = tree.query(coords[:n], k=k_use + 1)
        neighbors = indices[:, 1:]
        knn[:n, :k_use] = neighbors.astype(np.int16)

    return coords, knn, phys


def sample_sigma_continuous(batch_size, rng, p_zero=0.3,
                             min_sigma_A=1.0, max_sigma_A=1000.0):
    """Draw one sigma (Angstrom) per track: point mass at exactly 0 with
    probability p_zero (protects sigma=0 headline performance -- same role
    as the discrete pipeline's 30% sigma=0 weight), else log-uniform over
    [min_sigma_A, max_sigma_A] = [0.1nm, 100nm] -- covers the full range
    continuously rather than a handful of fixed points."""
    sigmas = np.zeros(batch_size, dtype=np.float64)
    is_nonzero = rng.random(batch_size) >= p_zero
    n_nonzero = int(is_nonzero.sum())
    if n_nonzero > 0:
        log_lo, log_hi = np.log(min_sigma_A), np.log(max_sigma_A)
        sigmas[is_nonzero] = np.exp(rng.uniform(log_lo, log_hi, size=n_nonzero))
    return sigmas


def build_batch_dynamic(raw_points, theta, batch_idx, k_neighbors, n_max,
                         device, rng, p_zero=0.3, min_sigma_A=1.0, max_sigma_A=1000.0):
    """Draw a continuous sigma per track in the batch, smear+prepare each,
    and assemble x_flat = [coords | knn | phys] on the fly. This is the
    per-BATCH, EVERY-STEP hot path -- called once per training step."""
    B = len(batch_idx)
    sigmas = sample_sigma_continuous(B, rng, p_zero, min_sigma_A, max_sigma_A)

    flat_dim = n_max * (3 + k_neighbors) + 5
    x_flat = torch.empty(B, flat_dim, dtype=torch.float32)
    knn_end = n_max * (3 + k_neighbors)

    for j, idx in enumerate(batch_idx):
        coords, knn, phys = _smear_and_prepare_one(
            raw_points[idx], sigmas[j], k_neighbors, n_max, rng)
        x_flat[j, :n_max * 3] = torch.from_numpy(coords.reshape(-1))
        x_flat[j, n_max * 3:knn_end] = torch.from_numpy(knn.astype(np.float32).reshape(-1))
        x_flat[j, knn_end:] = torch.from_numpy(phys)

    theta_batch = theta[batch_idx].to(device)
    x_batch = x_flat.to(device)
    return theta_batch, x_batch, sigmas
