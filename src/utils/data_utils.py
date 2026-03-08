# src/utils/data_utils.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree, ConvexHull, QhullError
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.neighbors import NearestNeighbors
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# SHARED HELPERS
# ============================================================

def infer_relative_bin_edges(n_bins=6, r_min=1e-3, r_max=1.0):
    edges = np.geomspace(r_min, r_max, n_bins)
    edges = np.insert(edges, 0, 0.0)
    return edges

def relative_bin_fractions_from_events(depths, norm_depth, r_edges):
    x = np.asarray(depths, float)
    if x.size == 0 or norm_depth <= 0: return np.zeros(len(r_edges)-1)
    r = x / (norm_depth + 1e-12)
    hist, _ = np.histogram(r, bins=r_edges)
    hist[-1] += np.sum(r > r_edges[-1])
    total = hist.sum()
    return hist / total if total > 0 else np.zeros_like(hist, float)

def compute_centered_track_asymmetry(z):
    z = np.asarray(z, float)
    n_left = np.sum(z < 0)
    n_right = np.sum(z > 0)
    total = n_left + n_right
    return (n_right - n_left) / (total + 1e-12) if total > 0 else 0.0

def compute_centered_nn_asymmetry(z, x):
    if z.size < 2: return 0.0
    pts = np.column_stack((x, z))
    # Check if we have points on both sides to compare
    if not (np.any(z < 0) and np.any(z > 0)): return 0.0

    def mean_nn(p):
        if p.shape[0] < 2: return 0.0
        nn = NearestNeighbors(n_neighbors=2).fit(p)
        dists, _ = nn.kneighbors(p)
        return np.mean(dists[:, 1])

    d_left = mean_nn(pts[z < 0])
    d_right = mean_nn(pts[z > 0])
    return (d_right - d_left) / (d_right + d_left + 1e-12)

def compute_rajendran_asymmetry(z):
    """
    Exact Rajendran Figure 3 logic:
    Ratio of vacancies in the end third vs the beginning third.
    """
    z = np.asarray(z, float)
    z_sorted = np.sort(z) # Tracks start at min(z) and end at max(z)

    # Define boundaries for thirds
    z_min, z_max = z_sorted[0], z_sorted[-1]
    one_third = z_min + (z_max - z_min) / 3.0
    two_thirds = z_min + 2.0 * (z_max - z_min) / 3.0

    # Count defects in each bucket
    n_first_third = np.sum(z <= one_third)
    n_last_third = np.sum(z >= two_thirds)

    # Rajendran defines asymmetry as the ratio (End / Start)
    # They expect a ratio of ~2:1 for success[cite: 122].
    return n_last_third / (n_first_third + 1e-12)


# ============================================================
# NPE PREPROCESSING (ENERGY ONLY)
# ============================================================

def preprocess_npe(data_path: str | Path, n_bins: int = 6):
    """
    Preprocess SRIM collision CSV for NPE (energy-only) model.

    Features per track:
      [mean_depth, max_depth, n_vac, rbin_frac_1..n_bins]
    θ per track:
      [energy_keV] (rounded)
    """
    df = pd.read_csv(data_path)

    # Normalize column names if they exist
    rename_map = {"x_ang": "x", "y_ang": "y", "z_ang": "z"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Required columns
    for c in ["x", "y", "z", "ion_number"]:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in {data_path}")

    # Decide energy source → always end up with energy_keV
    if "energy_keV" in df.columns:
        energy_keV = pd.to_numeric(df["energy_keV"], errors="coerce")
    elif "energy_eV" in df.columns:
        energy_keV = pd.to_numeric(df["energy_eV"], errors="coerce") / 1e3
    elif "energy" in df.columns:
        e_raw = pd.to_numeric(df["energy"], errors="coerce")
        energy_keV = e_raw / 1e3 if np.nanmax(e_raw) > 3000 else e_raw
    else:
        raise KeyError("No energy column found: expected one of ['energy_keV', 'energy_eV', 'energy'].")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["ion_number"] = pd.to_numeric(df["ion_number"], errors="coerce")
    df["energy_keV"] = pd.to_numeric(energy_keV, errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV"]).reset_index(drop=True)

    # Relative-depth bin edges
    bin_edges = infer_relative_bin_edges(n_bins=n_bins)

    rows = []
    x_obs_list = []
    theta_list = []
    track_ids = []

    # Group by rounded keV + ion_number
    df["energy_int"] = df["energy_keV"].round().astype(int)

    for (E_int, ion_no), g in df.groupby(["energy_int", "ion_number"], sort=False):
        x = np.abs(g["x"].to_numpy(float))
        if x.size == 0:
            continue

        mean_depth = float(x.mean())
        max_depth = float(np.max(x))
        norm_depth = float(np.percentile(x, 95))
        n_vac = int(x.size)

        rbin_fracs = relative_bin_fractions_from_events(x, norm_depth, bin_edges)
        if rbin_fracs.sum() > 0:
            rbin_fracs = rbin_fracs / (rbin_fracs.sum() + 1e-12)

        E_keV = float(E_int)
        tid = f"C_{int(E_keV)}keV_ion{int(ion_no)}"

        rows.append(
            {
                "track_id": tid,
                "ion": "C",
                "energy_keV": E_keV,
                "mean_depth_A": mean_depth,
                "max_depth_A": max_depth,
                "vacancies_per_ion": n_vac,
                **{f"rbin_frac_{i+1}": float(v) for i, v in enumerate(rbin_fracs)},
            }
        )

        x_obs_list.append([mean_depth, max_depth, n_vac, *rbin_fracs])
        theta_list.append([E_keV])
        track_ids.append(tid)

    df_summary = pd.DataFrame(rows)

    print("[DEBUG] energy_keV range:", df_summary["energy_keV"].min(), "→", df_summary["energy_keV"].max())
    print("[DEBUG] example energies (keV):", df_summary["energy_keV"].unique()[:10])

    x_obs = torch.tensor(np.asarray(x_obs_list, dtype=np.float32))
    theta = torch.tensor(np.asarray(theta_list, dtype=np.float32))

    return x_obs, theta, track_ids, {"rel_bin_edges": bin_edges}, df_summary


# ============================================================
# MNPE PREPROCESSING (ENERGY + PARITY)
# ============================================================
def preprocess_mnpe(data_path: str | Path, n_bins: int = 6, norm_stats=None):
    """
    Preprocess SRIM collision CSV for MNPE (energy + parity) model.

    Args:
        norm_stats: If provided, dict with 'feat_mean' and 'feat_std' from training.
                    If None, computes normalization from this dataset.
    """
    df = pd.read_csv(data_path)

    rename_map = {"x_ang": "x", "y_ang": "y", "z_ang": "z"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_cols = ["x", "y", "z", "ion_number", "parity"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {data_path}")

    if "energy_keV" in df.columns:
        df["energy_keV"] = pd.to_numeric(df["energy_keV"], errors="coerce")
    elif "energy_eV" in df.columns:
        df["energy_keV"] = pd.to_numeric(df["energy_eV"], errors="coerce") / 1e3
    elif "energy" in df.columns:
        e_raw = pd.to_numeric(df["energy"], errors="coerce")
        df["energy_keV"] = e_raw / 1e3 if e_raw.max() > 3000 else e_raw
    else:
        raise KeyError("No energy column found for MNPE preprocessing.")

    for col in ["x", "y", "z", "ion_number", "energy_keV", "parity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV", "parity"]).reset_index(drop=True)
    df["energy_norm"] = df["energy_keV"]

    r_edges = infer_relative_bin_edges(n_bins=n_bins)

    rows = []
    x_obs_list = []
    theta_list = []
    track_ids = []

    for (E_val, ion_no, par), g in df.groupby(["energy_norm", "ion_number", "parity"], sort=False):
        x = np.asarray(g["x"], float)
        z = np.asarray(g["z"], float)
        if x.size == 0:
            continue

        x = x - np.mean(x)
        z = z - np.mean(z)

        abs_x = np.abs(x)
        mean_depth = float(np.mean(abs_x))
        max_depth = float(np.max(abs_x))
        n_vac = int(len(x))

        depth_coord = x - np.min(x)
        norm_depth_bins = float(np.percentile(depth_coord, 95))

        rbin_fracs = relative_bin_fractions_from_events(
            depth_coord, norm_depth_bins, r_edges
        )
        asym_count = compute_centered_track_asymmetry(x)
        asym_nn = compute_centered_nn_asymmetry(x, z)

        try:
            skew_x = float(skew(x)) if x.size > 2 else 0.0
        except Exception:
            skew_x = 0.0
        if not np.isfinite(skew_x):
            skew_x = 0.0

        left_x = x[x < 0]
        right_x = x[x > 0]

        var_left = np.var(left_x) if left_x.size > 2 else 0.0
        var_right = np.var(right_x) if right_x.size > 2 else 0.0
        var_diff_x = float(np.log1p(var_right) - np.log1p(var_left))
        if not np.isfinite(var_diff_x):
            var_diff_x = 0.0

        abs_left = np.abs(left_x)
        abs_right = np.abs(right_x)
        mean_abs_left = abs_left.mean() if abs_left.size > 0 else 1e-9
        mean_abs_right = abs_right.mean() if abs_right.size > 0 else 1e-9

        mean_abs_ratio_lr = float(mean_abs_right / (mean_abs_left + 1e-9))
        if not np.isfinite(mean_abs_ratio_lr):
            mean_abs_ratio_lr = 1.0

        tid = f"E{E_val:.3f}_ion{int(ion_no)}_p{int(par)}"

        rows.append({
            "track_id": tid,
            "energy_keV": float(E_val),
            "parity": int(par),
            "mean_depth_A": mean_depth,
            "max_depth_A": max_depth,
            "vacancies_per_ion": n_vac,
            **{f"rbin_frac_{i+1}": rbin_fracs[i] for i in range(n_bins)},
            "asym_count_centered": asym_count,
            "asym_nn_centered": asym_nn,
            "skew_x": skew_x,
            "var_diff_x": var_diff_x,
            "mean_abs_ratio_lr": mean_abs_ratio_lr,
        })

        x_obs_list.append([
            mean_depth, max_depth, n_vac,
            *rbin_fracs,
            asym_count, asym_nn,
            skew_x, var_diff_x, mean_abs_ratio_lr,
        ])

        theta_list.append([float(E_val), float(int(par))])
        track_ids.append(tid)

    if len(x_obs_list) == 0:
        print("[WARN] preprocess_mnpe found NO valid tracks.")
        return (
            torch.zeros((0, 14)),
            torch.zeros((0, 2)),
            [],
            {"rel_bin_edges": r_edges},
            pd.DataFrame(rows),
        )

    x_obs = torch.tensor(np.asarray(x_obs_list, dtype=np.float32))
    theta = torch.tensor(np.asarray(theta_list, dtype=np.float32))
    theta[:, 1] = (theta[:, 1] == 1).float()

    # Normalization: use provided stats or compute from this data
    if norm_stats is not None:
        mean = norm_stats['feat_mean']
        std = norm_stats['feat_std']
    else:
        mean = x_obs.mean(dim=0)
        std = x_obs.std(dim=0) + 1e-6

    x_obs = (x_obs - mean) / std

    df_summary = pd.DataFrame(rows)

    print(f"[DEBUG] Tracks: {len(track_ids)}")
    print(f"[DEBUG] x_obs shape: {x_obs.shape}")
    print(f"[DEBUG] theta shape: {theta.shape}")

    return x_obs, theta, track_ids, {"rel_bin_edges": r_edges, "feat_mean": mean, "feat_std": std}, df_summary


# ============================================================
# MNPE v2 PREPROCESSING (WITH RAJENDRAN FEATURE)
# ============================================================

def preprocess_2mnpe(data_path, n_bins=6):
    df = pd.read_csv(data_path)

    # 1. Standard Cleanup
    df.columns = [c.strip() for c in df.columns]
    rename = {"x_ang": "x", "y_ang": "y", "z_ang": "z"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    # Ensure energy is in keV
    if "energy_keV" not in df.columns and "energy" in df.columns:
        df["energy_keV"] = df["energy"]/1e3

    # Ensure parity column exists (0 = Up, 1 = Down)
    if "parity" not in df.columns:
        df["parity"] = 0

    x_obs, theta_list, tids, rows = [], [], [], []
    r_edges = infer_relative_bin_edges(n_bins)

    # 2. Feature Extraction Loop
    for (E, ion, par), g in df.groupby(["energy_keV", "ion_number", "parity"], sort=False):
        raw_z = g["z"].values.astype(float)
        x_coords = g["x"].values.astype(float)

        if len(raw_z) < 5:
            continue

        # --- THE FIX: CENTERING ONLY ---
        z = raw_z - np.mean(raw_z)

        # --- PHYSICS FEATURES (Z-AXIS) ---
        abs_z = np.abs(z)
        mean_depth = np.mean(abs_z)
        max_depth = np.max(abs_z)

        # Longitudinal density profile
        depth_coord = z - np.min(z)
        norm = np.percentile(depth_coord, 95)
        r_fracs = relative_bin_fractions_from_events(depth_coord, norm, r_edges)

        # Directional Asymmetry Metrics
        asym_c = compute_centered_track_asymmetry(z)
        asym_nn = compute_centered_nn_asymmetry(z, x_coords)

        # --- RAJENDRAN ADDITION ---
        # This explicitly uses the 1/3 vs 1/3 ratio from Figure 3 [cite: 179, 181]
        asym_raj = compute_rajendran_asymmetry(z)

        # Skewness is the key discriminator for Bragg peaks [cite: 83]
        try:
            skew_z = float(skew(z))
        except:
            skew_z = 0.0

        # Log-Variance Difference
        var_up = np.var(z[z < 0]) if np.any(z < 0) else 0.0
        var_down = np.var(z[z > 0]) if np.any(z > 0) else 0.0
        var_diff = np.log1p(var_down) - np.log1p(var_up)

        # Assemble summary feature vector (Now 9D + histogram bins)
        # Added asym_raj to the feature vector so the model can learn from it
        row = [mean_depth, max_depth, len(z), *r_fracs, asym_c, asym_nn, asym_raj, skew_z, var_diff]

        x_obs.append(row)
        theta_list.append([float(E), float(par)])
        tids.append(f"E{E:.0f}_ion{int(ion)}")

        rows.append({
            "track_id": tids[-1],
            "energy": E,
            "parity": par,
            "skew_z": skew_z,
            "var_diff": var_diff,
            "asym_count": asym_c,
            "asym_rajendran": asym_raj  # Saved for post-eval analysis
        })

    # 3. Tensor Conversion
    x_t = torch.tensor(x_obs, dtype=torch.float32)
    theta_t = torch.tensor(theta_list, dtype=torch.float32)

    # 4. ROBUST NORMALIZATION
    if len(x_t) > 1:
        mean = x_t.mean(dim=0)
        std = x_t.std(dim=0) + 1e-6
        x_t = (x_t - mean) / std
        print(f"[PREPROCESS] Normalized {len(x_t)} tracks. Feature Max: {x_t.max().item():.2f}")

    return x_t, theta_t, tids, {"edges": r_edges}, pd.DataFrame(rows)


# ============================================================
# MDPE PREPROCESSING (28D, DIRECTIONAL 8-CLASS)
# ============================================================

def preprocess_mdpe(data_path: str | Path, n_radial_bins: int = 6, n_phi_bins: int = 8):
    """
    ULTRA-MAX 28D Preprocessing (Updated for Head/Tail Fix).
    Added: Skewness & Kurtosis to detect 'Fake Bragg Peaks'.
    """
    # =======================================================
    # 0. Load + Surgical Column Flattening
    # =======================================================
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {
        "ion #": "ion_number", "ion": "ion_number",
        "energy": "energy_keV", "x_ang": "x", "y_ang": "y", "z_ang": "z"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["energy_keV", "ion_number"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                val_1d = df[col].iloc[:, 0]
                df = df.drop(columns=[col])
                df[col] = val_1d
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ANGLE_BINS = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    df["angle_class"] = df["theta_deg"].apply(lambda a: int(np.argmin(np.abs(ANGLE_BINS - a))))
    df = df.dropna(subset=["x", "y", "z", "energy_keV", "ion_number", "angle_class"]).reset_index(drop=True)

    x_obs_list, theta_list, track_ids = [], [], []

    # =======================================================
    # 1. Physics Feature Extraction
    # =======================================================
    for (E, ion, cls), g in df.groupby(["energy_keV", "ion_number", "angle_class"], sort=False):
        pts = g[["x", "y", "z"]].to_numpy(float)
        if pts.shape[0] < 10: continue

        # --- A. CENTER & ANCHOR ---
        cm = np.mean(pts, axis=0)
        centered_pts = pts - cm
        dists = np.linalg.norm(centered_pts, axis=1)

        # Bragg Peak Anchor (Tail Estimation)
        tail_mask = dists >= np.percentile(dists, 80)
        tail_pts = pts[tail_mask]
        tail_anchor = np.mean(tail_pts, axis=0)
        growth_vec = cm - tail_anchor

        # --- B. PCA & ORIENTATION ---
        cov = np.cov(centered_pts.T)
        evals, evecs = np.linalg.eig(cov)
        idx = evals.argsort()[::-1]
        lam, vec = np.maximum(evals[idx], 1e-12), evecs[:, idx]

        # Force PC1 Direction (Head -> Tail)
        # Note: If this logic fails, Skewness will catch it!
        if np.dot(vec[:, 0], growth_vec) < 0: vec[:, 0] *= -1

        # --- C. NEW: HEAD/TAIL DISCRIMINATORS ---
        # Project points onto the spine
        proj = centered_pts @ vec[:, 0]

        # 1. Skewness (The Golden Feature)
        # Negative = Mass at End (Correct), Positive = Mass at Start (Flipped)
        track_skew = float(skew(proj))

        # 2. Kurtosis (Peak Sharpness)
        # Bragg peaks are sharper than random scattering blobs
        track_kurt = float(kurtosis(proj))

        # 3. Bragg Ratio (Contrast)
        min_p, max_p = np.min(proj), np.max(proj)
        n_head = np.sum(proj < (min_p + 0.2*(max_p-min_p)))
        n_tail = np.sum(proj > (max_p - 0.2*(max_p-min_p)))
        bragg_ratio = float(n_tail / (n_head + 1))

        # 4. Z-Depth Dive (3D Context)
        z_spread = np.std(centered_pts[:, 2])

        # 5. Simple Curvature Proxy
        curvature_proxy = float(lam[1] / lam[0])

        # --- D. STANDARD FEATURES ---
        total_var = float(np.sum(lam))
        lin = float((lam[0] - lam[1]) / lam[0])
        ent = float(-np.sum((lam/total_var) * np.log(lam/total_var + 1e-12)))
        internal_offset = float(np.linalg.norm(cm - tail_anchor))

        norm_r = dists / (np.percentile(dists, 95) + 1e-12)
        r_fracs = np.histogram(norm_r, bins=n_radial_bins, range=(0,1))[0] / len(pts)

        phi = np.degrees(np.arctan2(centered_pts[:, 1], centered_pts[:, 0])) % 360.0
        phi_fracs = np.histogram(phi, bins=n_phi_bins, range=(0, 360))[0] / len(pts)

        # --- E. ASSEMBLE 28D VECTOR ---
        features = [
            len(pts), np.max(dists),                    # Scale (2)
            *r_fracs,                                   # Radial (6)
            *phi_fracs,                                 # Angular (8)
            lin, ent, curvature_proxy,                  # Shape (3)
            bragg_ratio, z_spread,                      # Context (2)
            track_skew, track_kurt,                     # **NEW: Head/Tail Fixers (2)**
            float(vec[0, 0]), float(vec[1, 0]),         # Directed Spine (2)
            total_var, internal_offset,                 # Global stats (2)
            float(vec[2, 0])                            # Z-component (1)
        ]

        x_obs_list.append(features)
        theta_list.append([E, cls])
        track_ids.append(f"E{E:.1f}_ion{int(ion)}_c{int(cls)}")

    print(f"[MDPE] Preprocessed into {len(x_obs_list[0])}D vectors. Total Tracks: {len(x_obs_list)}")

    return (
        torch.tensor(np.asarray(x_obs_list, np.float32)),
        torch.tensor(np.asarray(theta_list, np.float32)),
        track_ids,
        {"r_edges": np.linspace(0, 1, n_radial_bins+1)},
        pd.DataFrame()
    )


# ============================================================
# MDPE EMBEDDING (RAW POINT CLOUDS)
# ============================================================

def create_embedding_dataset(track_list, max_points=1000):
    """
    Reformats tracks into NaN-padded tensor (N, max_points, 4).
    Channels: (x, y, z, local_density)
    """
    SCALE_FACTOR = 2000.0
    DENSITY_RADIUS = 20.0 # Angstroms (Look for neighbors within this range)

    num_tracks = len(track_list)
    # CHANGE: 4 channels instead of 3
    x_data = np.full((num_tracks, max_points, 4), np.nan, dtype=np.float32)

    for i, vacancies in enumerate(track_list):
        coords = np.array(vacancies)
        if len(coords) == 0: continue

        # 1. Center the track
        center = np.mean(coords, axis=0)
        centered_coords = coords - center

        # 2. Scale Spatial Coords (0-3)
        scaled_xyz = centered_coords / SCALE_FACTOR

        # 3. Calculate Local Density (Channel 4)
        # We compute this on the unscaled coords (real Angstroms)
        tree = cKDTree(coords)
        # Count how many neighbors are within 20 Angstroms
        densities = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
        densities = np.array(densities, dtype=np.float32)

        # Normalize density to 0-1 range (relative to this track's max)
        # This highlights the "Bragg Peak" regardless of total energy
        densities = densities / (densities.max() + 1e-9)

        # 4. Stack (x, y, z, density)
        track_4d = np.column_stack((scaled_xyz, densities))

        # 5. Store
        num_to_copy = min(len(track_4d), max_points)
        x_data[i, :num_to_copy, :] = track_4d[:num_to_copy]

    return torch.tensor(x_data, dtype=torch.float32)

def load_raw_vacancies(csv_path):
    """
    Loads CSV, groups by track, and returns raw vacancy clouds.
    """
    df = pd.read_csv(csv_path)

    # 1. Clean columns
    df.columns = [c.strip() for c in df.columns]

    # 2. Rename columns
    rename_map = {
        "ion #": "ion_number", "ion": "ion_number",
        "energy": "energy_keV", "x (ang)": "x", "y (ang)": "y", "z (ang)": "z",
        "x_ang": "x", "y_ang": "y", "z_ang": "z"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- CRITICAL FIX: Remove Duplicate Columns ---
    # This keeps the first occurrence and drops subsequent duplicates
    df = df.loc[:, ~df.columns.duplicated()]
    # ----------------------------------------------

    # 3. Ensure numeric
    cols = ["x", "y", "z", "energy_keV", "ion_number", "angle_class"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=cols)

    vacancies_list = []
    theta_list = []

    # 4. Group by track
    for (E, ion, cls), g in df.groupby(["energy_keV", "ion_number", "angle_class"], sort=False):
        xyz = g[["x", "y", "z"]].values.astype(np.float32)
        vacancies_list.append(xyz)
        theta_list.append([E, cls])

    return vacancies_list, torch.tensor(np.array(theta_list), dtype=torch.float32)


# ============================================================
# MCPE PREPROCESSING (28D, CONTINUOUS ANGLE)
# ============================================================

def preprocess_mcpe(data_path: str, n_radial_bins: int = 6, n_phi_bins: int = 8):
    """
    MCPE 28D Preprocessing (Continuous Version).
    Target: [Energy, Sin(theta), Cos(theta)]
    """
    print(f"[INFO] Loading {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Standardize Columns
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "ion #": "ion_number", "ion": "ion_number",
        "energy": "energy_keV",
        "x (ang)": "x", "y (ang)": "y", "z (ang)": "z",
        "x_ang": "x", "y_ang": "y", "z_ang": "z"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- BUG FIX: REMOVE DUPLICATE COLUMNS IMMEDIATELY ---
    # If 'energy' became 'energy_keV' and 'energy_keV' already existed, we now have two.
    # This line keeps only the first occurrence of every column name.
    df = df.loc[:, ~df.columns.duplicated()]
    # -----------------------------------------------------

    # 2. Filter Valid Rows
    # Ensure columns exist before trying to numeric-ize them
    for col in ["energy_keV", "ion_number", "theta_deg"]:
        if col in df.columns:
            # This will now work because df[col] is guaranteed to be a Series (1 column)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    req_cols = ["x", "y", "z", "energy_keV", "ion_number", "theta_deg"]
    existing_cols = [c for c in req_cols if c in df.columns]
    df = df.dropna(subset=existing_cols).reset_index(drop=True)

    x_obs_list, target_list, track_ids = [], [], []

    # 3. Group by Unique Track
    print(f"[INFO] Extracting features...")

    # Ensure grouping columns are valid
    if "energy_keV" not in df.columns or "ion_number" not in df.columns:
        raise KeyError(f"Missing required columns in {data_path}. Found: {df.columns.tolist()}")

    for (E, ion), g in df.groupby(["energy_keV", "ion_number"], sort=False):
        pts = g[["x", "y", "z"]].to_numpy(float)
        if pts.shape[0] < 5: continue

        # --- A. Targets (Continuous) ---
        if "theta_deg" in g.columns:
            theta_deg = float(g["theta_deg"].iloc[0])
            theta_rad = np.radians(theta_deg)
            target = [float(E), np.sin(theta_rad), np.cos(theta_rad)]
        else:
            # Fallback if testing on data without angles (shouldn't happen in training)
            target = [float(E), 0.0, 0.0]

        # --- B. Feature Extraction (28D) ---
        # 1. Center & Anchor
        cm = np.mean(pts, axis=0)
        centered_pts = pts - cm
        dists = np.linalg.norm(centered_pts, axis=1)

        tail_mask = dists >= np.percentile(dists, 80)
        tail_pts = pts[tail_mask]
        tail_anchor = np.mean(tail_pts, axis=0)
        growth_vec = cm - tail_anchor

        # 2. PCA & Orientation
        cov = np.cov(centered_pts.T)
        evals, evecs = np.linalg.eig(cov)
        idx = evals.argsort()[::-1]
        lam, vec = np.maximum(evals[idx], 1e-12), evecs[:, idx]

        if np.dot(vec[:, 0], growth_vec) < 0: vec[:, 0] *= -1

        # 3. Projection Features
        proj = centered_pts @ vec[:, 0]
        track_skew = float(skew(proj))
        track_kurt = float(kurtosis(proj))

        min_p, max_p = np.min(proj), np.max(proj)
        n_head = np.sum(proj < (min_p + 0.2*(max_p-min_p)))
        n_tail = np.sum(proj > (max_p - 0.2*(max_p-min_p)))
        bragg_ratio = float(n_tail / (n_head + 1))

        # 4. Standard Stats
        z_spread = np.std(centered_pts[:, 2])
        curvature_proxy = float(lam[1] / lam[0])
        total_var = float(np.sum(lam))
        lin = float((lam[0] - lam[1]) / lam[0])
        ent = float(-np.sum((lam/total_var) * np.log(lam/total_var + 1e-12)))
        internal_offset = float(np.linalg.norm(cm - tail_anchor))

        # 5. Histograms
        norm_r = dists / (np.percentile(dists, 95) + 1e-12)
        r_fracs = np.histogram(norm_r, bins=n_radial_bins, range=(0,1))[0] / len(pts)

        phi = np.degrees(np.arctan2(centered_pts[:, 1], centered_pts[:, 0])) % 360.0
        phi_fracs = np.histogram(phi, bins=n_phi_bins, range=(0, 360))[0] / len(pts)

        features = [
            len(pts), np.max(dists),
            *r_fracs,
            *phi_fracs,
            lin, ent, curvature_proxy,
            bragg_ratio, z_spread,
            track_skew, track_kurt,
            float(vec[0, 0]), float(vec[1, 0]),
            total_var, internal_offset,
            float(vec[2, 0])
        ]

        x_obs_list.append(features)
        target_list.append(target)
        track_ids.append(f"E{E:.1f}_ion{int(ion)}")

    print(f"[MCPE] Processed {len(x_obs_list)} tracks. Feature Dim: {len(x_obs_list[0])}")

    return (
        torch.tensor(np.asarray(x_obs_list, np.float32)),
        torch.tensor(np.asarray(target_list, np.float32)),
        track_ids,
        {"r_edges": np.linspace(0, 1, n_radial_bins+1)},
        pd.DataFrame()
    )


# ============================================================
# MCPE-3D PREPROCESSING (OVERKILL FEATURES)
# ============================================================

def _get_overkill_features(group):
    # 1. SETUP
    pts = group[["x", "y", "z"]].values
    n_vacancies = len(pts)

    # Center the cloud
    center = pts.mean(axis=0)
    pts_centered = pts - center

    # 2. ADVANCED PCA & TENSOR ANALYSIS
    cov = np.cov(pts_centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    # Sort L1 (largest) to L3 (smallest)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    l1, l2, l3 = evals

    # Shape Factors (Westin Indices) - Robust 3D shape descriptors
    # Avoid div/0
    denom = l1 + 1e-9
    linearity = (l1 - l2) / denom
    planarity = (l2 - l3) / denom
    sphericity = l3 / denom
    anisotropy = (l1 - l3) / denom

    # 3. CONVEX HULL (Volume & Density)
    # This measures the "tightness" of the damage cloud.
    # Higher energy tracks are often "fluffier" (straggling).
    try:
        if len(pts) >= 4: # Need 4 points for 3D hull
            hull = ConvexHull(pts_centered)
            hull_vol = hull.volume
            hull_area = hull.area
            # "Solidity": Ratio of vacancies to the volume they occupy
            solidity = n_vacancies / (hull_vol + 1e-9)
        else:
            hull_vol, hull_area, solidity = 0.0, 0.0, 0.0
    except QhullError:
        # Fallback for perfectly planar/linear tracks that break Hull
        hull_vol, hull_area, solidity = 0.0, 0.0, 0.0

    # 4. LONGITUDINAL PROFILE (The "dE/dx" Curve)
    # We project points onto the main axis (PC1) and bin them.
    # This gives the network a 10-pixel "image" of the energy deposition.
    v1 = evecs[:, 0] # Main axis
    projections = pts_centered @ v1

    # Create 10 bins along the track length
    # range ensures we capture the full span
    hist, _ = np.histogram(projections, bins=10, density=False)

    # Normalize histogram so it sums to 1 (shape only)
    # OR keep raw counts (density info). Let's keep raw counts for Energy info.
    # We label them "seg_0" to "seg_9".

    # 5. RADIAL PROFILE (Core vs. Halo)
    # Distance from the principal axis (cylindrical radius)
    # r = sqrt(d^2 - proj^2)
    dist_sq = np.sum(pts_centered**2, axis=1)
    proj_sq = projections**2
    cyl_radii = np.sqrt(np.maximum(0, dist_sq - proj_sq))

    mean_cyl_radius = np.mean(cyl_radii)
    max_cyl_radius = np.max(cyl_radii)

    # 6. HEAD-TAIL LOGIC
    sk = skew(projections)
    kurt = kurtosis(projections)

    # Front-Back Ratio (Simple split)
    n_pos = np.sum(projections > 0)
    n_neg = np.sum(projections < 0)
    fb_ratio = np.log((n_pos + 1) / (n_neg + 1))

    # 7. CONSTRUCT OUTPUT
    features = {
        # --- Targets ---
        "energy_true": group["energy_keV"].iloc[0],
        "vx_true": group["target_vx"].iloc[0],
        "vy_true": group["target_vy"].iloc[0],
        "vz_true": group["target_vz"].iloc[0],

        # --- Global Counts ---
        "n_vacancies": n_vacancies,

        # --- PCA Eigenvalues ---
        "eigen_l1": l1, "eigen_l2": l2, "eigen_l3": l3,

        # --- Shape Factors (Westin) ---
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "anisotropy": anisotropy,

        # --- Hull / Volumetric ---
        "hull_volume": hull_vol,
        "hull_area": hull_area,
        "solidity": solidity,

        # --- Directional Stats ---
        "skewness": sk,
        "kurtosis": kurt,
        "front_back_ratio": fb_ratio,
        "centroid_dist": np.linalg.norm(center), # Distance from (0,0,0)

        # --- Cylindrical/Radial Stats ---
        "mean_cyl_radius": mean_cyl_radius,
        "max_cyl_radius": max_cyl_radius,

        # --- Raw Vectors (PC1) ---
        "pc1_x": v1[0], "pc1_y": v1[1], "pc1_z": v1[2]
    }

    # Add the 10 Histogram Bins dynamically
    for i, count in enumerate(hist):
        features[f"profile_bin_{i}"] = count

    return pd.Series(features)

def preprocess_mcpe_3d(df):
    """
    Runs the Overkill feature extraction.
    """
    print(f"[Overkill] Processing {len(df)} raw points...")

    if 'run_id' in df.columns:
        grouped = df.groupby(["run_id", "energy_keV", "ion_number"], sort=False)
    else:
        grouped = df.groupby(["energy_keV", "ion_number"], sort=False)

    print(f"[Overkill] Extracting rich features for {len(grouped)} tracks...")

    features_df = grouped.apply(_get_overkill_features)
    return features_df.reset_index(drop=True)


# ============================================================
# MCPE-3D RAW POINT CLOUD LOADING
# ============================================================

def load_and_pad_3d_tracks(csv_path, max_points="auto"):
    """
    Loads raw 3D tracks (x, y, z) and pads them to a uniform tensor shape.
    """
    print(f"[DATA] Loading raw point cloud data from: {csv_path}")
    df = pd.read_csv(csv_path)

    grouped = df.groupby('ion_number')

    # --- DYNAMIC MAX POINTS CALCULATION ---
    if max_points == "auto":
        track_lengths = grouped.size()
        # 95th percentile balances coverage vs memory/speed
        max_points = int(track_lengths.quantile(0.95))
        absolute_max = track_lengths.max()
        print(f"[DATA] Track lengths -> Max: {absolute_max}, 95%: {max_points}")
        print(f"[DATA] Auto-setting max_points to {max_points}")
    # --------------------------------------

    track_tensors = []
    theta_list = []

    for ion_idx, group in grouped:
        # Extract purely Spatial Coordinates (x, y, z)
        points = group[['x', 'y', 'z']].values

        # --- NEW: MEAN CENTERING ---
        # Find the center of mass of the track
        centroid = np.mean(points, axis=0)
        # Shift the entire track so its center is exactly at (0, 0, 0)
        points = points - centroid
        # ---------------------------


        # Truncate to max_points
        if len(points) > max_points:
            points = points[:max_points]

        track_tensors.append(torch.tensor(points, dtype=torch.float32))

        # Extract Ground Truth Targets [E, Vx, Vy, Vz]
        true_E = group['energy_keV'].iloc[0]
        true_vx = group['target_vx'].iloc[0]
        true_vy = group['target_vy'].iloc[0]
        true_vz = group['target_vz'].iloc[0]

        theta_list.append([true_E, true_vx, true_vy, true_vz])

    print("[DATA] Padding tracks to uniform dimensions...")
    # Pad sequences with NaN so the network knows which points are fake
    x_padded = pad_sequence(track_tensors, batch_first=True, padding_value=float('nan'))
    theta_tensor = torch.tensor(theta_list, dtype=torch.float32)

    print(f"[DATA] Final X shape:     {x_padded.shape} -> (Batch, Points, 3)")
    print(f"[DATA] Final Theta shape: {theta_tensor.shape} -> [E, Vx, Vy, Vz]")

    return x_padded, theta_tensor


# ============================================================
# EGNN PREPROCESSING (Pipeline C)
# ============================================================

def preprocess_egnn(csv_path, max_points="auto", k_neighbors=8):
    """
    Preprocess 3D tracks for EGNN Pipeline C.

    Per track:
      1. Center (subtract centroid)
      2. PCA-align (SVD so principal axis = x-axis)
      3. Normalize (divide by max spatial extent -> coords in [-1, 1])
      4. Zero-pad to N_max, create boolean mask

    CRITICAL: Do NOT flip/orient tracks by density.
    The network must learn head-tail discrimination itself.

    Results are cached to disk as .egnn_cache.pt for instant reloading.
    Cache auto-invalidates when k_neighbors or max_points change.

    Args:
        csv_path: path to CSV with columns [x, y, z, ion_number, energy_keV,
                  target_vx, target_vy, target_vz]
        max_points: "auto" to use 95th percentile, or an integer
        k_neighbors: number of kNN neighbors to precompute

    Returns:
        x_padded: (N_tracks, N_max, 3) zero-padded, PCA-aligned coordinates
        mask:     (N_tracks, N_max) boolean mask (True = real point)
        theta:    (N_tracks, 4) ground truth [E, Vx, Vy, Vz]
        n_max:    int, the padding length (needed by EGNNEmbedding)
        knn_idx:  (N_tracks, N_max, k) precomputed kNN neighbor indices (-1 = pad)
    """
    csv_path = Path(csv_path)

    # --- Check disk cache ---
    cache_path = csv_path.parent / f"{csv_path.stem}_k{k_neighbors}_mp{max_points}.egnn_cache.pt"
    if cache_path.exists():
        print(f"[EGNN] Loading cached preprocessed data from {cache_path.name}")
        cached = torch.load(cache_path, weights_only=False)
        print(f"[EGNN] Loaded {cached['x_padded'].shape[0]} tracks, "
              f"N_max={cached['n_max']}, k={k_neighbors}")
        return (cached['x_padded'], cached['mask'], cached['theta'],
                cached['n_max'], cached['knn_idx'])

    # --- No cache: run full preprocessing ---
    # Only load columns we need (saves ~50% RAM on large CSVs)
    use_cols = ['x', 'y', 'z', 'ion_number', 'energy_keV',
                'target_vx', 'target_vy', 'target_vz']
    print(f"[EGNN] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, usecols=use_cols)
    print(f"[EGNN] Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # Sort by ion_number for efficient groupby (avoids building hash table)
    df.sort_values('ion_number', inplace=True)

    # Compute track lengths for auto max_points BEFORE groupby iteration
    if max_points == "auto":
        track_lengths = df.groupby('ion_number').size()
        max_points = int(track_lengths.quantile(0.95))
        print(f"[EGNN] Track lengths -> Max: {track_lengths.max()}, "
              f"95%: {max_points}, Tracks: {len(track_lengths):,}")
        n_tracks_est = len(track_lengths)
        del track_lengths
    else:
        n_tracks_est = df['ion_number'].nunique()

    # Extract numpy arrays and free the DataFrame immediately
    ion_ids = df['ion_number'].values
    xyz = df[['x', 'y', 'z']].values.astype(np.float64)
    energies = df['energy_keV'].values
    vx = df['target_vx'].values
    vy = df['target_vy'].values
    vz = df['target_vz'].values
    del df  # Free ~1 GB of RAM
    import gc; gc.collect()
    print(f"[EGNN] DataFrame freed, processing tracks via numpy...")

    # Find track boundaries (data is sorted by ion_number)
    boundaries = np.where(np.diff(ion_ids) != 0)[0] + 1
    track_starts = np.concatenate([[0], boundaries])
    track_ends = np.concatenate([boundaries, [len(ion_ids)]])

    coords_list = []
    theta_list = []
    lengths = []

    for t_idx in tqdm(range(len(track_starts)),
                      desc="[EGNN] Processing tracks"):
        s, e = track_starts[t_idx], track_ends[t_idx]
        points = xyz[s:e]  # (n, 3) — no copy, just a view

        if len(points) < 3:
            continue  # Need at least 3 points for PCA

        # 1. Center
        centroid = points.mean(axis=0)
        centered = points - centroid

        # 2. PCA-align via SVD
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue  # Skip degenerate tracks

        # Rotate so principal axis = x, secondary = y, tertiary = z
        aligned = centered @ Vt.T  # (N, 3)

        # 3. Normalize by max spatial extent -> coords in ~[-1, 1]
        max_extent = np.abs(aligned).max()
        if max_extent > 0:
            aligned = aligned / max_extent

        # Truncate if too long
        if len(aligned) > max_points:
            aligned = aligned[:max_points]

        coords_list.append(aligned.astype(np.float32))
        lengths.append(len(aligned))

        # Extract ground truth (constant within a track, take first row)
        theta_list.append([energies[s], vx[s], vy[s], vz[s]])

    # Free the raw arrays
    del ion_ids, xyz, energies, vx, vy, vz
    gc.collect()

    # 4. Zero-pad to N_max, create mask, compute kNN neighbors
    N_tracks = len(coords_list)
    N_max = max(lengths)
    print(f"[EGNN] {N_tracks:,} tracks, N_max = {N_max}")

    x_padded = np.zeros((N_tracks, N_max, 3), dtype=np.float32)
    mask = np.zeros((N_tracks, N_max), dtype=bool)
    knn_idx = np.full((N_tracks, N_max, k_neighbors), -1, dtype=np.int64)

    for i, coords in tqdm(enumerate(coords_list),
                          desc="[EGNN] Building kNN graphs",
                          total=N_tracks):
        n = len(coords)
        x_padded[i, :n, :] = coords
        mask[i, :n] = True

        # Precompute kNN with scipy cKDTree (O(N log N), much faster than cdist)
        if n > 1:
            k_use = min(k_neighbors, n - 1)
            tree = cKDTree(coords)
            _, indices = tree.query(coords, k=k_use + 1)  # +1 for self
            neighbors = indices[:, 1:]  # (n, k_use) — skip self
            knn_idx[i, :n, :k_use] = neighbors

    # Free coords_list before creating tensors
    del coords_list
    gc.collect()

    x_padded = torch.tensor(x_padded, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    theta = torch.tensor(theta_list, dtype=torch.float32)
    knn_idx = torch.tensor(knn_idx, dtype=torch.long)

    print(f"[EGNN] x_padded: {x_padded.shape}  mask: {mask.shape}  "
          f"theta: {theta.shape}  knn_idx: {knn_idx.shape}")

    # --- Save cache to disk ---
    torch.save({'x_padded': x_padded, 'mask': mask, 'theta': theta,
                'n_max': N_max, 'knn_idx': knn_idx}, cache_path)
    print(f"[EGNN] Cached preprocessed data -> {cache_path.name}")

    return x_padded, mask, theta, N_max, knn_idx
