"""a100_final: new smear / normalize / graph-building pipeline.

Two changes vs. src/utils/dynamic_smear.py::_smear_and_prepare_one (which is
left untouched and still importable for A/B):

  (2) NORMALIZER.  The old pipeline divided the centered cloud by
      ``max_extent = |centered|.max()`` -- an L-infinity EXTREME ORDER
      STATISTIC.  Under isotropic Gaussian blur of width sigma it grows like
      the max of ~3N half-normal draws, i.e. roughly sigma*sqrt(2 ln(3N)) once
      blur dominates, so the normalizer depends on BOTH the blur level and the
      point count in a way the network cannot undo.  We replace it with a
      blur-deconvolved radius of gyration (derivation in README S2):

          R2_obs = mean_i |y_i - ybar|^2
          s2     = sigma_A^2 * (n-1)/n          # exact centered noise variance
          scale  = sqrt( max( R2_obs - 3*s2 , floor ) )
          floor  = max(FLOOR_ABS, (FLOOR_FRAC*sigma_A)^2)

      E[R2_obs] = R2_true + 3*sigma^2*(n-1)/n EXACTLY (no approximation), so
      ``scale`` is an unbiased-in-the-square estimator of the true R_g.

  (5) GRAPH.  Fixed k=16 kNN is replaced by a blur-adaptive hybrid: in
      NORMALIZED coordinates, each point connects to every neighbour inside
      radius ``r = sqrt(h0^2 + sigma_n^2)`` (sigma_n = sigma_A/scale) OR to its
      K_MIN nearest neighbours, whichever yields MORE edges, capped at K_MAX.
      Because kNN indices come back distance-sorted, the radius set is exactly
      a prefix of the kNN list, so this needs no second tree query.

Extra scalars carried in the phys block (see PHYS_* indices below): the two
sigma-conditioning features consumed by the direction head's trunk, and the
raw normalized sigma consumed by the deterministic 4th-moment blur correction.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import cKDTree

# re-exported unchanged from the existing repo
from src.utils.dynamic_smear import load_raw_tracks, sample_sigma_continuous  # noqa: F401

# ---------------------------------------------------------------- constants
FLOOR_ABS = 1e-4        # Angstrom^2; keeps scale>0 for a degenerate cloud
FLOOR_FRAC = 0.1        # deconvolution is capped at 10% of sigma (see README)
SIGMA0_A = 1.0          # Angstrom; log-floor for the sigma-conditioning feature
SIGMA_N_FLOOR = 1e-3    # log-floor for the normalized-sigma feature

# blur-adaptive graph (see README S5 for how H0 was chosen from the data)
H0_DEFAULT = 0.05       # base bandwidth in R_g-normalized units
K_MIN_DEFAULT = 8
K_MAX_DEFAULT = 24

# phys-block layout
N_PHYS = 8
PHYS_LOG_NVAC = 0
PHYS_LOG_EXTENT = 1
PHYS_LOG_RG = 2
PHYS_ELONG1 = 3
PHYS_ELONG2 = 4
PHYS_LOG_SIGMA_A = 5     # log(sigma_A + SIGMA0_A)      -> trunk (zero-init'd)
PHYS_LOG_SIGMA_N = 6     # log(sigma_n + SIGMA_N_FLOOR) -> trunk (zero-init'd)
PHYS_SIGMA_N = 7         # sigma_n, raw                 -> moment correction


def deconvolved_scale(centered, sigma_A, floor_abs=FLOOR_ABS, floor_frac=FLOOR_FRAC):
    """Blur-deconvolved radius of gyration of an already-centered cloud."""
    n = centered.shape[0]
    R2_obs = float((centered ** 2).sum(axis=1).mean())
    s2 = float(sigma_A) ** 2 * (n - 1) / max(n, 1)
    floor = max(floor_abs, (floor_frac * float(sigma_A)) ** 2)
    return float(np.sqrt(max(R2_obs - 3.0 * s2, floor)))


def legacy_scale(centered):
    """The winner's normalizer, kept for the A/B validation test."""
    return float(np.abs(centered).max())


def hybrid_graph(coords_n, sigma_n, h0=H0_DEFAULT, k_min=K_MIN_DEFAULT,
                 k_max=K_MAX_DEFAULT):
    """Blur-adaptive neighbour indices for one (n,3) normalized cloud.

    Returns (n, k_max) int array, -1 padded.
    """
    n = coords_n.shape[0]
    out = np.full((n, k_max), -1, dtype=np.int64)
    if n < 2:
        return out
    k_use = min(k_max, n - 1)
    tree = cKDTree(coords_n)
    d, idx = tree.query(coords_n, k=k_use + 1)
    d = np.atleast_2d(d)[:, 1:]
    idx = np.atleast_2d(idx)[:, 1:]
    radius = np.sqrt(h0 * h0 + float(sigma_n) ** 2)
    # kNN output is distance-sorted -> the radius set is a prefix
    n_rad = (d <= radius).sum(axis=1)
    n_keep = np.clip(np.maximum(n_rad, k_min), 0, k_use)
    col = np.arange(k_use)[None, :]
    keep = col < n_keep[:, None]
    out[:, :k_use] = np.where(keep, idx, -1)
    return out


def _smear_and_prepare_one_v2(points, sigma_A, n_max, rng,
                              h0=H0_DEFAULT, k_min=K_MIN_DEFAULT,
                              k_max=K_MAX_DEFAULT, pad=True):
    """smear -> center -> physics descriptors -> deconvolved-Rg normalize ->
    truncate -> pad -> blur-adaptive graph.

    Returns coords (n_max,3) f32, knn (n_max,k_max) i32, phys (N_PHYS,) f32.

    pad=False returns COMPACT (n,3)/(n,k_max) arrays, deferring padding to the
    caller.  Worker PROCESSES use this: mean track length is 91 against
    n_max=600, so shipping pre-padded arrays through the DataLoader's shared
    memory would waste ~6.5x the IPC bandwidth for nothing.
    """
    if sigma_A > 0:
        points = points + rng.normal(0.0, sigma_A, size=points.shape)

    centered = points - points.mean(axis=0)
    n_vac_full = centered.shape[0]

    max_extent = np.abs(centered).max()
    sq_radii = (centered ** 2).sum(axis=1)
    r_gyration = np.sqrt(sq_radii.mean())
    cov = (centered.T @ centered) / n_vac_full
    evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    lam_sum = float(evals.sum()) + 1e-12

    scale = deconvolved_scale(centered, sigma_A)
    sigma_n = float(sigma_A) / scale

    phys = np.zeros(N_PHYS, dtype=np.float32)
    phys[PHYS_LOG_NVAC] = np.log(n_vac_full + 1.0)
    phys[PHYS_LOG_EXTENT] = np.log(max_extent + 1e-6)
    phys[PHYS_LOG_RG] = np.log(r_gyration + 1e-6)
    phys[PHYS_ELONG1] = float(evals[2]) / lam_sum
    phys[PHYS_ELONG2] = float(evals[1]) / lam_sum
    phys[PHYS_LOG_SIGMA_A] = np.log(float(sigma_A) + SIGMA0_A)
    phys[PHYS_LOG_SIGMA_N] = np.log(sigma_n + SIGMA_N_FLOOR)
    phys[PHYS_SIGMA_N] = sigma_n

    centered = centered / scale

    # truncate AFTER the descriptors, matching the winner's ordering
    if len(centered) > n_max:
        centered = centered[:n_max]
    n = len(centered)

    cc = centered.astype(np.float32)
    kk = hybrid_graph(cc.astype(np.float64), sigma_n,
                      h0=h0, k_min=k_min, k_max=k_max).astype(np.int32)
    if not pad:
        return cc, kk, phys

    coords = np.zeros((n_max, 3), dtype=np.float32)
    coords[:n] = cc
    knn = np.full((n_max, k_max), -1, dtype=np.int32)
    knn[:n] = kk
    return coords, knn, phys


class TrackPool:
    """Fork-safe container for the raw track point clouds.

    `load_raw_tracks` returns a PYTHON LIST of ~293k small numpy arrays.  Under
    the DataLoader's fork-based workers that is a trap: copy-on-write shares
    pages initially, but CPython writes to each object's refcount the moment a
    worker touches it, so the per-object headers get copied into every worker
    and RSS grows by roughly the pool size per worker.

    Storing one concatenated (M,3) array plus an offsets array reduces the
    whole pool to TWO Python objects, so a worker slicing it takes a view and
    touches no per-track refcount.  Memory stays genuinely shared.
    """

    __slots__ = ('points', 'offsets')

    def __init__(self, raw_points):
        self.points = np.concatenate(raw_points, axis=0).astype(np.float64)
        lens = np.fromiter((len(p) for p in raw_points), dtype=np.int64,
                           count=len(raw_points))
        self.offsets = np.zeros(len(raw_points) + 1, dtype=np.int64)
        np.cumsum(lens, out=self.offsets[1:])

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.points[self.offsets[i]:self.offsets[i + 1]]


def flat_dim(n_max, k_max):
    return n_max * (3 + k_max) + N_PHYS


def pack_one(x_flat_row, coords, knn, phys, n_max, k_max):
    knn_end = n_max * (3 + k_max)
    x_flat_row[:n_max * 3] = torch.from_numpy(coords.reshape(-1))
    x_flat_row[n_max * 3:knn_end] = torch.from_numpy(
        knn.astype(np.float32).reshape(-1))
    x_flat_row[knn_end:] = torch.from_numpy(phys)


def build_batch_v2(raw_points, theta, batch_idx, n_max, device, rng,
                   p_zero=0.30, min_sigma_A=1.0, max_sigma_A=1000.0,
                   h0=H0_DEFAULT, k_min=K_MIN_DEFAULT, k_max=K_MAX_DEFAULT,
                   sigmas=None):
    """Per-batch hot path. Returns (theta_batch, x_batch, sigmas)."""
    B = len(batch_idx)
    if sigmas is None:
        sigmas = sample_sigma_continuous(B, rng, p_zero, min_sigma_A, max_sigma_A)
    x_flat = torch.empty(B, flat_dim(n_max, k_max), dtype=torch.float32)
    for j, idx in enumerate(batch_idx):
        coords, knn, phys = _smear_and_prepare_one_v2(
            raw_points[idx], sigmas[j], n_max, rng, h0, k_min, k_max)
        pack_one(x_flat[j], coords, knn, phys, n_max, k_max)
    return theta[batch_idx].to(device), x_flat.to(device), sigmas


def build_eval_batch(raw_points, idx_list, sigma_A, n_max, rng,
                     h0=H0_DEFAULT, k_min=K_MIN_DEFAULT, k_max=K_MAX_DEFAULT):
    """Fixed-sigma cohort assembly for eval (CPU tensor, caller moves it)."""
    x_flat = torch.zeros(len(idx_list), flat_dim(n_max, k_max), dtype=torch.float32)
    for j, i in enumerate(idx_list):
        coords, knn, phys = _smear_and_prepare_one_v2(
            raw_points[i], sigma_A, n_max, rng, h0, k_min, k_max)
        pack_one(x_flat[j], coords, knn, phys, n_max, k_max)
    return x_flat
