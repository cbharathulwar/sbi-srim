"""v23_final: smear / normalize / higher-moment / graph pipeline.

Ported from a100_final/data_pipeline.py (already validated there -- see that
folder's README S2/S3/S5 and validate_math.py) with the v23 additions from
reports/v23_architecture_plan.tex.  Nothing outside v23_final/ is modified.

What is carried over VERBATIM from a100_final
---------------------------------------------
  * deconvolved_scale()  -- S{normalizer}.  scale = sqrt(max(R2_obs - 3 s2, floor)),
    s2 = sigma_A^2 (n-1)/n, floor = max(FLOOR_ABS, (FLOOR_FRAC sigma_A)^2).
    E[R2_obs] = R2_true + 3 sigma^2 (n-1)/n EXACTLY (no approximation).
    Measured: flat at ratio 1.000 across four decades of blur, vs the legacy
    L-infinity max_extent normalizer's 2.268 -> 7.768 drift.
  * hybrid_graph()       -- S{graph}.  r = sqrt(H0^2 + sigma_n^2) OR K_MIN
    nearest neighbours, whichever gives MORE edges, capped at K_MAX.  kNN output
    is distance-sorted so the radius set is a prefix -- no second tree query.
  * TrackPool            -- 2 numpy objects instead of a 1M-element python list,
    so fork-based DataLoader workers share it copy-on-write instead of copying
    per-object refcounts.
  * TrackDataset/Collate/make_loader (in this file) -- the epoch-permutation
    shuffled sampling + worker-process pattern.

What is NEW in v23
------------------
  * moment_features()    -- S{physfeat}.  The blur-corrected higher-moment
    (kurtosis-type) tensor A_hat and its uncorrected counterpart M_obs, reduced
    to FOUR scalars using the SIGNED COSINE (|u . eig1|), not a raw angle
    (corrected per external review: an eigenvector's sign is arbitrary and a
    flip near degeneracy would discontinuously jump a raw angle between ~0 and
    ~180 deg).  Both corrected and uncorrected are fed -- a100_final's TEST B2
    showed the UNCORRECTED estimator's accidental shrinkage toward the PCA axis
    wins by 11 deg at sigma/R_g = 2.37, so the spec says "Do not pick one".
  * the phys block grows 5 -> 11: the 4 moment scalars plus the two RAW sigma
    values (sigma_A, sigma_n) that model.py turns into the 16-dim sinusoidal
    embedding.
  * load_raw_tracks_v23() also returns ion_number, needed for the Pool A /
    Pool B split that Stage 2 uses.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from scipy.spatial import cKDTree

from smearing_resolution.architecture_experiments.v23_final import config as C

# ---------------------------------------------------------------- constants
# [a100] read directly from a100_final/data_pipeline.py lines 42-50 so v23
# matches the already-validated constants rather than a plausible-sounding guess.
FLOOR_ABS = C.FLOOR_ABS          # 1e-4 Angstrom^2
FLOOR_FRAC = C.FLOOR_FRAC        # 0.1
SIGMA0_A = C.SIGMA0_A            # 1.0 Angstrom
SIGMA_N_FLOOR = C.SIGMA_N_FLOOR  # 1e-3
H0_DEFAULT = C.H0                # 0.05 (R_g-normalized units)
K_MIN_DEFAULT = C.K_MIN          # 8
K_MAX_DEFAULT = C.K_MAX          # 24

# ------------------------------------------------------- phys-block layout
N_PHYS = C.N_PHYS                # 11
PHYS_LOG_NVAC = 0                # [v22] log(n_vac + 1)
PHYS_LOG_EXTENT = 1              # [v22] log(max_extent + 1e-6)   (L-inf, FEATURE only)
PHYS_LOG_RG = 2                  # [v22] log(r_gyration + 1e-6)   (observed, blurred)
PHYS_ELONG1 = 3                  # [v22] lambda_1 / sum lambda
PHYS_ELONG2 = 4                  # [v22] lambda_2 / sum lambda
PHYS_COS_CORR = 5                # [v23] |u . eig1(A_hat)|
PHYS_COS_UNCORR = 6              # [v23] |u . eig1(M_obs)|
PHYS_RATIO_CORR = 7              # [v23] lam_max(A_hat)  / sum|lam(A_hat)|
PHYS_RATIO_UNCORR = 8            # [v23] lam_max(M_obs)  / sum|lam(M_obs)|
PHYS_SIGMA_A = 9                 # [v23] raw sigma_A  -> sinusoidal embedding
PHYS_SIGMA_N = 10                # [v23] raw sigma_n  -> sinusoidal embedding
# columns 0..8 are standardized with training-set stats; 9..10 are NOT (they are
# consumed only by the fixed sinusoidal embedding in model.py).
N_PHYS_COND = C.N_PHYS_COND      # 9
MOMENT_SLICE = slice(PHYS_COS_CORR, PHYS_RATIO_UNCORR + 1)   # 5:9
SIGMA_RAW_SLICE = slice(PHYS_SIGMA_A, PHYS_SIGMA_N + 1)      # 9:11

_EPS = 1e-12


# =========================================================== S{normalizer}
def deconvolved_scale(centered, sigma_A, floor_abs=FLOOR_ABS,
                      floor_frac=FLOOR_FRAC):
    """[a100, VERBATIM] Blur-deconvolved radius of gyration of a centered cloud.

    E[R2_obs] = R2_true + 3 sigma_A^2 (n-1)/n  EXACTLY, so R2_obs - 3 s2 is an
    unbiased estimator of R2_true (the sqrt is then mildly biased low; that is
    a100_final's documented, accepted behaviour and is NOT corrected here).

    The floor serves two distinct failure modes:
      (FLOOR_FRAC*sigma_A)^2 -> once sigma >> R_g, noise can drive the
          subtraction negative; clamping there bounds scale >= 0.1*sigma_A, i.e.
          sigma_n <= 10, so the normalized coordinates cannot diverge.
      FLOOR_ABS              -> the sigma=0 branch, where the fractional floor
          vanishes identically and a degenerate/collinear cloud gives R2_obs=0.
    """
    n = centered.shape[0]
    R2_obs = float((centered ** 2).sum(axis=1).mean())
    s2 = float(sigma_A) ** 2 * (n - 1) / max(n, 1)
    floor = max(floor_abs, (floor_frac * float(sigma_A)) ** 2)
    return float(np.sqrt(max(R2_obs - 3.0 * s2, floor)))


def legacy_scale(centered):
    """[a100, VERBATIM] v22's normalizer (L-infinity extreme order statistic),
    kept only for the A/B validation test.  NOT used in the v23 pipeline."""
    return float(np.abs(centered).max())


# ================================================================ S{graph}
def hybrid_graph(coords_n, sigma_n, h0=H0_DEFAULT, k_min=K_MIN_DEFAULT,
                 k_max=K_MAX_DEFAULT):
    """[a100, VERBATIM] Blur-adaptive neighbour indices for one (n,3) normalized
    cloud.  Returns (n, k_max) int64, -1 padded.

    K_MAX is a real, documented approximation: the flat tensor layout is
    fixed-width, so a point whose radius ball holds more than k_max neighbours
    keeps only its k_max nearest.
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
    n_rad = (d <= radius).sum(axis=1)
    n_keep = np.clip(np.maximum(n_rad, k_min), 0, k_use)
    col = np.arange(k_use)[None, :]
    keep = col < n_keep[:, None]
    out[:, :k_use] = np.where(keep, idx, -1)
    return out


# ============================================================ S{physfeat}
def pca_axis_signed(centered):
    """Raw-PCA axis u with the CLASSICAL BASELINE's sign convention.

    Identical to a100_final/eval.py::pca_axis_and_signed (itself identical to
    eval_r7_full_sweep.py:211): top eigenvector of the n-normalized coordinate
    covariance via eigh, then flipped when the third moment of the projections
    is POSITIVE.  Using the baseline's own convention (rather than
    a100_final/model.py::_skew_sign_fix, which flips on the opposite sign) is
    what makes the S{fallback} residual reduce EXACTLY to the reported PCA
    baseline -- including its head/tail decision -- at zero-init.

    Scale-invariant, so it gives the same axis on physical or normalized coords.
    """
    cov = (centered.T @ centered) / len(centered)
    axis = np.linalg.eigh(cov)[1][:, -1]
    if ((centered @ axis) ** 3).sum() > 0:
        axis = -axis
    return axis, cov


def moment_features(coords_n, sigma_n, u=None, cov=None):
    """[v23, S{physfeat}] Four scalars from the blur-corrected higher-moment
    (kurtosis-type) tensor and its uncorrected counterpart.

    Tensor (a100_final README S3b):
        M = (1/n) sum_i p_i^2 outer(x_i, x_i),   p_i = x_i . u,  u = raw-PCA axis
    Exact isotropic-Gaussian blur correction:
        A_hat = M_obs - s2 (<q^2>_hat I + 4 <q^2>_hat u u^T + C_hat)
                      - s2^2 (I + 2 u u^T)
        s2 = sigma_n^2 (n-1)/n,  <q^2>_hat = <p^2>_obs - s2,  C_hat = C_obs - s2 I

    COORDINATE UNITS MATTER: the tensor is computed on the NORMALIZED cloud and
    the correction therefore uses sigma_n = sigma_A / scale.  Feeding physical
    coordinates with sigma_n (or normalized coordinates with sigma_A) is wrong
    by a factor of scale^2.

    Assumptions carried over (a100_final README S3b, unchanged, not re-derived):
      * noise is additive iid N(0, sigma^2 I_3), independent of the signal;
      * Gaussianity is load-bearing (the 4-noise-factor term uses Isserlis);
      * the cloud is centered, and s2 is the CENTERED per-coordinate variance;
      * u is treated as fixed though it is estimated from the same blurred
        cloud -- an unmodelled O(1/n) correlation, numerically far below the
        retained terms (TEST B1);
      * u is exactly the top eigenvector of C, which is what collapses
        b = C u to <q^2> u and produces the factor 4.

    Reduction to scalars uses the SIGNED COSINE |u . eig1|, per external review,
    NOT a raw angle: an eigenvector's sign is arbitrary, and a flip near a
    degenerate case would discontinuously jump a raw angle between values near
    0 and 180 deg, forcing the network to learn around a self-inflicted
    discontinuity.  Absolute value because axis direction, not sign, is what
    these features should convey (head/tail is handled separately, S{loss}).

    The eigenvalue "ratio" confidence proxy is lam_max / sum|lam| -- see README
    (the spec asks for "the corresponding top eigenvalue ratios" without pinning
    the normalization; A_hat can be indefinite, so a plain lam_1/sum(lam) is
    ill-defined and the absolute-value denominator is the well-defined choice).
    """
    n = coords_n.shape[0]
    if u is None or cov is None:
        u, cov = pca_axis_signed(coords_n)
    p = coords_n @ u
    p2 = p * p
    # M_obs = (1/n) sum_i p_i^2 x_i x_i^T
    M_obs = np.einsum('i,ij,ik->jk', p2, coords_n, coords_n) / n
    s2 = float(sigma_n) ** 2 * (n - 1) / max(n, 1)
    q2 = max(float(p2.mean()) - s2, 0.0)          # clamped at 0, as in a100_final
    I3 = np.eye(3)
    uu = np.outer(u, u)
    C_hat = cov - s2 * I3                          # NOT clamped (may go indefinite)
    A = (M_obs - s2 * (q2 * I3 + 4.0 * q2 * uu + C_hat)
         - (s2 ** 2) * (I3 + 2.0 * uu))

    out = np.zeros(4, dtype=np.float64)
    for j, T in enumerate((A, M_obs)):
        T = 0.5 * (T + T.T)                        # symmetrize
        lam, vec = np.linalg.eigh(T)               # ascending eigenvalues
        # eigh gives the ALGEBRAICALLY-largest eigenvalue at index -1, which is
        # exactly what a100_final's shifted power iteration converges to (the
        # correction can leave A indefinite; an unshifted power iteration would
        # find the most-negative eigenvector instead).
        v = vec[:, -1]
        out[j] = abs(float(u @ v))                                  # signed cosine
        out[2 + j] = float(lam[-1]) / (float(np.abs(lam).sum()) + _EPS)
    return out, u


# ======================================================= per-track pipeline
def _smear_and_prepare_one_v23(points, sigma_A, n_max, rng,
                               h0=H0_DEFAULT, k_min=K_MIN_DEFAULT,
                               k_max=K_MAX_DEFAULT, pad=True):
    """smear -> center -> v22 physics descriptors -> deconvolved-Rg normalize ->
    higher-moment features -> truncate -> pad -> blur-adaptive graph.

    Returns coords (n,3) f32, knn (n,k_max) i32, phys (N_PHYS,) f32.
    pad=False returns COMPACT arrays (mean track length is 91 against n_max=600,
    so shipping pre-padded arrays through the DataLoader's shared memory would
    waste ~6.5x the IPC bandwidth for nothing).

    Ordering note: the v22 descriptors AND the normalizer AND the moment
    features are all computed on the FULL (pre-truncation) cloud, matching
    preprocess_egnn's and a100_final's ordering; only the graph is built on the
    truncated cloud.  With MAX_POINTS=600 > the longest track (587 train / 545
    eval) nothing is truncated at all, so the distinction is moot in practice --
    but it is preserved so the code stays correct if that ever changes.
    """
    if sigma_A > 0:
        points = points + rng.normal(0.0, sigma_A, size=points.shape)

    centered = points - points.mean(axis=0)
    n_vac_full = centered.shape[0]

    # ---- v22's 5 physics features, UNCHANGED (data_utils.py lines 1153-1167).
    # NOTE: index 1 ("extent") is the same L-infinity statistic the NORMALIZER
    # replaces.  It is kept here as a FEATURE unchanged -- diagnostic
    # information, not a scale to divide by -- so its known blur-drift is not a
    # bug in this role.  Do not conflate the two roles.
    max_extent = np.abs(centered).max()
    sq_radii = (centered ** 2).sum(axis=1)
    r_gyration = np.sqrt(sq_radii.mean())
    cov_phys = (centered.T @ centered) / n_vac_full
    evals = np.clip(np.linalg.eigvalsh(cov_phys), 0.0, None)
    lam_sum = float(evals.sum()) + _EPS

    # ---- S{normalizer}: blur-deconvolved R_g replaces max_extent
    scale = deconvolved_scale(centered, sigma_A)
    sigma_n = float(sigma_A) / scale
    coords_n = centered / scale

    # ---- S{physfeat}: higher-moment features on the NORMALIZED cloud
    mom, _u = moment_features(coords_n, sigma_n)

    phys = np.zeros(N_PHYS, dtype=np.float32)
    phys[PHYS_LOG_NVAC] = np.log(n_vac_full + 1.0)
    phys[PHYS_LOG_EXTENT] = np.log(max_extent + 1e-6)
    phys[PHYS_LOG_RG] = np.log(r_gyration + 1e-6)
    phys[PHYS_ELONG1] = float(evals[2]) / lam_sum
    phys[PHYS_ELONG2] = float(evals[1]) / lam_sum
    phys[PHYS_COS_CORR] = mom[0]
    phys[PHYS_COS_UNCORR] = mom[1]
    phys[PHYS_RATIO_CORR] = mom[2]
    phys[PHYS_RATIO_UNCORR] = mom[3]
    phys[PHYS_SIGMA_A] = float(sigma_A)
    phys[PHYS_SIGMA_N] = sigma_n

    # truncate AFTER the descriptors, matching v22/a100_final ordering
    if len(coords_n) > n_max:
        coords_n = coords_n[:n_max]
    n = len(coords_n)

    cc = coords_n.astype(np.float32)
    kk = hybrid_graph(cc.astype(np.float64), sigma_n,
                      h0=h0, k_min=k_min, k_max=k_max).astype(np.int32)
    if not pad:
        return cc, kk, phys

    coords = np.zeros((n_max, 3), dtype=np.float32)
    coords[:n] = cc
    knn = np.full((n_max, k_max), -1, dtype=np.int32)
    knn[:n] = kk
    return coords, knn, phys


# ============================================================= sigma sampling
def sample_sigma_continuous(batch_size, rng, p_zero=None, min_sigma_A=None,
                            max_sigma_A=None):
    """[a100/R7, VERBATIM] One sigma (Angstrom) per track: a point mass at
    EXACTLY 0 with probability p_zero (protects sigma=0 headline performance),
    else log-uniform over [min_sigma_A, max_sigma_A] = [0.1nm, 100nm].

    S{training} item 2: this is the convention Stage 1 AND Stage 2 now both use.
    v22 used neither (smear_sigma defaulted to 0.0 and was never overridden).
    """
    p_zero = C.P_ZERO if p_zero is None else p_zero
    min_sigma_A = C.MIN_SIGMA_A if min_sigma_A is None else min_sigma_A
    max_sigma_A = C.MAX_SIGMA_A if max_sigma_A is None else max_sigma_A
    sigmas = np.zeros(batch_size, dtype=np.float64)
    is_nonzero = rng.random(batch_size) >= p_zero
    n_nonzero = int(is_nonzero.sum())
    if n_nonzero > 0:
        log_lo, log_hi = np.log(min_sigma_A), np.log(max_sigma_A)
        sigmas[is_nonzero] = np.exp(rng.uniform(log_lo, log_hi, size=n_nonzero))
    return sigmas


# ================================================================ track pool
class TrackPool:
    """[a100, VERBATIM] Fork-safe container for the raw track point clouds.

    A python LIST of ~1M small numpy arrays is a trap under fork-based workers:
    copy-on-write shares pages initially, but CPython writes each object's
    refcount the moment a worker touches it, so per-object headers get copied
    into every worker and RSS grows by roughly the pool size per worker.  One
    concatenated (M,3) array plus an offsets array is TWO python objects, so a
    worker slicing it takes a view and touches no per-track refcount.
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


# ================================================================= CSV load
def load_raw_tracks_v23(csv_path, max_points=None, nrows=0, verbose=True):
    """Adapted from src/utils/dynamic_smear.py::load_raw_tracks, with the
    ion_number column ALSO returned (Stage 2 needs it for the Pool A split).

    Keeps the FULL untruncated track: the physics descriptors are computed on the
    full centered cloud before truncation, and truncating at load time silently
    corrupts them for any track longer than max_points.

    Returns (raw_points, theta, ion_numbers, n_max, max_points_used).
    """
    import pandas as pd
    max_points = C.MAX_POINTS if max_points is None else max_points
    use_cols = ['x', 'y', 'z', 'ion_number', 'energy_keV',
                'target_vx', 'target_vy', 'target_vz']
    if verbose:
        print(f"[LOAD] {csv_path}" + (f"  (nrows={nrows:,})" if nrows else ""))
    kw = dict(usecols=use_cols)
    if nrows:
        kw['nrows'] = nrows
    df = pd.read_csv(csv_path, **kw)
    df.sort_values('ion_number', inplace=True, kind='mergesort')

    ion_ids = df['ion_number'].values
    xyz = df[['x', 'y', 'z']].values.astype(np.float64)
    energies = df['energy_keV'].values
    vx, vy, vz = (df['target_vx'].values, df['target_vy'].values,
                  df['target_vz'].values)
    del df
    import gc
    gc.collect()

    boundaries = np.where(np.diff(ion_ids) != 0)[0] + 1
    track_starts = np.concatenate([[0], boundaries])
    track_ends = np.concatenate([boundaries, [len(ion_ids)]])

    raw_points, theta_list, ions, lengths = [], [], [], []
    observed_max = 0
    for t_idx in range(len(track_starts)):
        s, e = track_starts[t_idx], track_ends[t_idx]
        n = e - s
        if n < 3:                       # same filter as preprocess_egnn
            continue
        raw_points.append(xyz[s:e])
        lengths.append(min(n, max_points))
        observed_max = max(observed_max, n)
        theta_list.append([energies[s], vx[s], vy[s], vz[s]])
        ions.append(ion_ids[s])

    n_max = max(lengths)
    theta = torch.tensor(np.asarray(theta_list, dtype=np.float32))
    ion_numbers = np.asarray(ions, dtype=np.int64)
    if verbose:
        print(f"[LOAD] {len(raw_points):,} valid tracks, n_max={n_max}, "
              f"longest track={observed_max}, max_points={max_points}")
    # a100_final README S8b hazard: load_raw_tracks CAPS n_max at max_points, so
    # a corpus with a 650-point track would silently truncate and still pass an
    # `n_max <= max_points` assert.  Guard on the OBSERVED length instead.
    if observed_max > max_points:
        raise SystemExit(
            f"[LOAD] FATAL: longest track is {observed_max} points but "
            f"MAX_POINTS={max_points}: this would SILENTLY TRUNCATE. Raise "
            f"MAX_POINTS (a100_final README S8b).")
    if observed_max > 0.95 * max_points:
        print(f"[LOAD] WARNING: longest track {observed_max} is within 5% of "
              f"MAX_POINTS={max_points}; raise MAX_POINTS before regenerating.")
    return raw_points, theta, ion_numbers, n_max, max_points


def pool_a_mask(ion_numbers, verbose=True, expect_full_corpus=True):
    """Pool A membership for the v2 corpus.  See config.py for the full
    derivation of POOL_B_LO / POOL_B_HI and why v22's plain
    `ion_number < 300_000` rule is WRONG here (it would mislabel every
    isotropic v2 tier-extra track as channeling-enriched Pool B)."""
    is_b = (ion_numbers >= C.POOL_B_LO) & (ion_numbers <= C.POOL_B_HI)
    n_b = int(is_b.sum())
    if verbose:
        print(f"[POOL] Pool A: {len(ion_numbers) - n_b:,} tracks   "
              f"Pool B: {n_b:,} tracks   "
              f"(Pool B = ion_number in [{C.POOL_B_LO:,}, {C.POOL_B_HI:,}])")
    exp, tol = C.POOL_B_EXPECTED, C.POOL_B_TOLERANCE
    if not expect_full_corpus:
        if verbose:
            print(f"[POOL] (partial CSV read -- Pool B count is NOT expected to "
                  f"match gen_meta.json's {exp:,}; check skipped)")
    elif exp > 0 and abs(n_b - exp) > tol * exp:
        print(f"[POOL] WARNING: Pool B count {n_b:,} differs from the expected "
              f"{exp:,} (gen_meta.json) by more than {100*tol:.0f}%. The corpus "
              f"or the id convention may have changed -- CHECK BEFORE TRAINING.")
    return ~is_b


# ============================================================ flat packing
def flat_dim(n_max, k_max):
    return n_max * (3 + k_max) + N_PHYS


def pack_one(x_flat_row, coords, knn, phys, n_max, k_max):
    knn_end = n_max * (3 + k_max)
    x_flat_row[:n_max * 3] = torch.from_numpy(coords.reshape(-1))
    x_flat_row[n_max * 3:knn_end] = torch.from_numpy(
        knn.astype(np.float32).reshape(-1))
    x_flat_row[knn_end:] = torch.from_numpy(phys)


def build_eval_batch(raw_points, idx_list, sigma_A, n_max, rng,
                     h0=H0_DEFAULT, k_min=K_MIN_DEFAULT, k_max=K_MAX_DEFAULT):
    """[a100] Fixed-sigma cohort assembly for eval (CPU tensor; caller moves)."""
    x_flat = torch.zeros(len(idx_list), flat_dim(n_max, k_max),
                         dtype=torch.float32)
    for j, i in enumerate(idx_list):
        coords, knn, phys = _smear_and_prepare_one_v23(
            raw_points[i], sigma_A, n_max, rng, h0, k_min, k_max)
        pack_one(x_flat[j], coords, knn, phys, n_max, k_max)
    return x_flat


def build_batch_serial(pool, theta, batch_idx, n_max, device, rng,
                       h0=H0_DEFAULT, k_min=K_MIN_DEFAULT, k_max=K_MAX_DEFAULT,
                       sigmas=None):
    """Synchronous per-batch hot path (used by the validation pass and the fast
    correctness check; training uses the worker-process loader below)."""
    B = len(batch_idx)
    if sigmas is None:
        sigmas = sample_sigma_continuous(B, rng)
    x_flat = torch.empty(B, flat_dim(n_max, k_max), dtype=torch.float32)
    for j, idx in enumerate(batch_idx):
        coords, knn, phys = _smear_and_prepare_one_v23(
            pool[idx], sigmas[j], n_max, rng, h0, k_min, k_max)
        pack_one(x_flat[j], coords, knn, phys, n_max, k_max)
    return theta[batch_idx].to(device), x_flat.to(device), sigmas


# ====================================================== worker-process loader
class TrackDataset(torch.utils.data.Dataset):
    """[a100, adapted] Map-style dataset over GLOBAL SAMPLE INDEX
    g = (step-1)*B + j.  Paired with a sequential sampler this reproduces the
    epoch-permutation batch composition EXACTLY and delivers batches in order.

    EPOCH-BASED SHUFFLED ITERATION: each epoch is one random permutation of the
    index pool consumed in batch-sized chunks (replacing per-step
    rng.choice(replace=False), which samples WITH replacement ACROSS steps and
    so never touches ~exp(-steps*B/pool) of the pool).  The last partial batch
    of a permutation is dropped; a different remainder is dropped each epoch.

    Sigma is a fresh continuous draw every time a track is used -- precomputing
    blurred clouds would force a finite discrete sigma grid and risk
    reintroducing a milder form of the sigma-blindness v23 exists to fix.  The
    one semantic difference from a serial per-batch draw: sigma comes from a
    per-SAMPLE generator seeded by (seed, g) so any worker can produce any
    sample independently.  The draws stay iid with the identical marginal.

    NOTE on `steps_per_epoch`: this is the SCHEDULER's epoch (config
    SUB_EPOCH_STEPS), which may be shorter than a full pass over the pool.  The
    PERMUTATION is over the whole pool and advances across scheduler epochs, so
    every track is still visited -- see config.py SUB_EPOCH_STEPS.
    """

    def __init__(self, pool, theta, idx_arr, steps_per_epoch, batch_size, n_max,
                 start_step, total_steps, seed, h0, k_min, k_max,
                 p_zero, min_sigma_A, max_sigma_A):
        self.pool, self.theta, self.idx_arr = pool, theta, idx_arr
        self.spe = max(1, len(idx_arr) // batch_size)   # permutation length
        self.sched_spe = steps_per_epoch
        self.B, self.n_max = batch_size, n_max
        self.seed = seed
        self.h0, self.k_min, self.k_max = h0, k_min, k_max
        self.p_zero, self.min_s, self.max_s = p_zero, min_sigma_A, max_sigma_A
        self.start = start_step * batch_size
        self.n = max(0, total_steps - start_step) * batch_size
        self._perm_cache = {}

    def __len__(self):
        return self.n

    def _permutation(self, cycle):
        p = self._perm_cache.get(cycle)
        if p is None:
            p = np.random.default_rng(
                (self.seed + 1) * 7_919 + cycle).permutation(self.idx_arr)
            self._perm_cache[cycle] = p
            for k in [k for k in self._perm_cache if k < cycle - 1]:
                del self._perm_cache[k]
        return p

    def __getitem__(self, i):
        g = self.start + i
        step = g // self.B + 1
        j = g % self.B
        cycle = (step - 1) // self.spe          # which full-pass permutation
        pos = (step - 1) % self.spe
        t_idx = int(self._permutation(cycle)[pos * self.B + j])
        rng = np.random.default_rng((self.seed + 1) * 2_000_003 + g)
        sig = float(sample_sigma_continuous(1, rng, self.p_zero, self.min_s,
                                            self.max_s)[0])
        co, kn, ph = _smear_and_prepare_one_v23(
            self.pool[t_idx], sig, self.n_max, rng,
            self.h0, self.k_min, self.k_max, pad=False)
        return co, kn, ph, self.theta[t_idx], step


class Collate:
    """[a100, VERBATIM in structure] Pad the compact per-track arrays into the
    flat model input.  A class, not a closure, so it stays picklable for
    spawn-based workers."""

    def __init__(self, n_max, k_max):
        self.n_max, self.k_max = n_max, k_max
        self.flat = flat_dim(n_max, k_max)

    def __call__(self, items):
        B = len(items)
        n_max, k_max = self.n_max, self.k_max
        x = torch.zeros(B, self.flat, dtype=torch.float32)
        th = torch.empty(B, 4, dtype=torch.float32)
        c_end, k_end = n_max * 3, n_max * (3 + k_max)
        for b, (co, kn, ph, theta, _s) in enumerate(items):
            n = co.shape[0]
            x[b, :c_end].view(n_max, 3)[:n] = torch.from_numpy(co)
            xk = x[b, c_end:k_end].view(n_max, k_max)
            xk.fill_(-1.0)
            xk[:n] = torch.from_numpy(kn.astype(np.float32))
            x[b, k_end:] = torch.from_numpy(ph)
            th[b] = theta
        return th, x, items[0][4]


def resolve_workers(requested, verbose=True):
    """[a100, VERBATIM] Windows uses spawn, not fork: every worker would
    re-import the training module and reload the CSVs.  Force synchronous
    loading there."""
    if requested > 0 and os.name == 'nt':
        if verbose:
            print(f"[LOADER] WARNING: N_WORKERS={requested} ignored on Windows "
                  f"(spawn would reload the dataset per worker); using 0. The "
                  f"A100 cluster is Linux, where workers are enabled.")
        return 0
    return requested


def make_loader(dataset, batch_size, n_workers, prefetch, n_max, k_max,
                pin_memory=False, verbose=True):
    n_workers = resolve_workers(n_workers, verbose)
    kw = (dict(persistent_workers=True, prefetch_factor=prefetch)
          if n_workers > 0 else {})
    if verbose:
        print(f"[LOADER] {n_workers} worker process(es), "
              f"prefetch_factor={kw.get('prefetch_factor', 'n/a')}, "
              f"{len(dataset) // max(batch_size, 1)} steps queued")
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
        collate_fn=Collate(n_max, k_max), drop_last=True,
        pin_memory=pin_memory, **kw)
