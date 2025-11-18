# src/utils/data_utils.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SHARED HELPERS
# ============================================================

def infer_relative_bin_edges(n_bins: int = 6, r_min: float = 1e-3, r_max: float = 1.0) -> np.ndarray:
    """Log-spaced relative-depth bin edges in [0, r_max], with an explicit 0.0 edge."""
    edges = np.geomspace(r_min, r_max, n_bins)
    edges = np.insert(edges, 0, 0.0)
    return edges


def relative_bin_fractions_from_events(depths_A, norm_depth_A, r_edges: np.ndarray) -> np.ndarray:
    """Compute relative depth histogram fractions given absolute depths and a normalization scale."""
    x = np.asarray(depths_A, float)
    x = x[np.isfinite(x)]
    if x.size == 0 or norm_depth_A <= 0:
        return np.zeros(len(r_edges) - 1, float)

    r = x / (norm_depth_A + 1e-12)
    hist, _ = np.histogram(r, bins=r_edges)
    hist[-1] += np.sum(r > r_edges[-1])

    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=float)

    return hist / total


def compute_centered_track_asymmetry(x, eps: float = 1e-12) -> float:
    """Simple left/right count asymmetry after centering."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0

    n_left = np.sum(x < 0)
    n_right = np.sum(x > 0)
    total = n_left + n_right
    if total == 0:
        return 0.0

    return (n_right - n_left) / (total + eps)


def compute_centered_nn_asymmetry(x, z, eps: float = 1e-12) -> float:
    """
    NN asymmetry using two regions:
      - left side (x < 0)
      - right side (x > 0)

    Returns (d_right - d_left) / (d_right + d_left).
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    if x.size < 2:
        return 0.0

    left_mask = x < 0
    right_mask = x > 0

    left_pts = np.column_stack((x[left_mask], z[left_mask]))
    right_pts = np.column_stack((x[right_mask], z[right_mask]))

    def mean_nn(points: np.ndarray) -> float:
        if points.shape[0] < 2:
            return np.nan
        nn = NearestNeighbors(n_neighbors=2).fit(points)
        dists, _ = nn.kneighbors(points)
        return float(np.mean(dists[:, 1]))

    d_left = mean_nn(left_pts)
    d_right = mean_nn(right_pts)

    if np.isnan(d_left) or np.isnan(d_right):
        return 0.0

    return (d_right - d_left) / (d_right + d_left + eps)


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

def preprocess_mnpe(data_path: str | Path, n_bins: int = 6):
    """
    Preprocess SRIM collision CSV for MNPE (energy + parity) model.

    Features per track:
      [mean_depth, max_depth, n_vac,
       rbin_frac_1..n_bins,
       asym_count_centered, asym_nn_centered,
       skew_x, var_diff_x, mean_abs_ratio_lr]

    θ per track:
      [energy_keV (continuous), parity ∈ {0,1}]
    """
    df = pd.read_csv(data_path)

    # Standardize column names
    rename_map = {"x_ang": "x", "y_ang": "y", "z_ang": "z"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_cols = ["x", "y", "z", "ion_number", "parity"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {data_path}")

    # Energy handling → energy_keV
    if "energy_keV" in df.columns:
        df["energy_keV"] = pd.to_numeric(df["energy_keV"], errors="coerce")
    elif "energy_eV" in df.columns:
        df["energy_keV"] = pd.to_numeric(df["energy_eV"], errors="coerce") / 1e3
    elif "energy" in df.columns:
        e_raw = pd.to_numeric(df["energy"], errors="coerce")
        df["energy_keV"] = e_raw / 1e3 if e_raw.max() > 3000 else e_raw
    else:
        raise KeyError("No energy column found for MNPE preprocessing.")

    # Ensure numeric
    for col in ["x", "y", "z", "ion_number", "energy_keV", "parity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV", "parity"]).reset_index(drop=True)

    # Continuous energy
    df["energy_norm"] = df["energy_keV"]

    # Relative bin edges (shared helper)
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

        # Base features
        abs_x = np.abs(x)
        mean_depth = float(np.mean(abs_x))
        max_depth = float(np.max(abs_x))
        norm_depth = float(np.percentile(abs_x, 95))
        n_vac = int(len(x))

        rbin_fracs = relative_bin_fractions_from_events(abs_x, norm_depth, r_edges)
        asym_count = compute_centered_track_asymmetry(x)
        asym_nn = compute_centered_nn_asymmetry(x, z)

        # Skewness
        try:
            skew_x = float(skew(x)) if x.size > 2 else 0.0
        except Exception:
            skew_x = 0.0
        if not np.isfinite(skew_x):
            skew_x = 0.0

        # Left/right variance difference
        left_x = x[x < 0]
        right_x = x[x > 0]

        var_left = np.var(left_x) if left_x.size > 2 else 0.0
        var_right = np.var(right_x) if right_x.size > 2 else 0.0
        var_diff_x = float(var_right - var_left)
        if not np.isfinite(var_diff_x):
            var_diff_x = 0.0

        # Left/right mean absolute ratio
        abs_left = np.abs(left_x)
        abs_right = np.abs(right_x)
        mean_abs_left = abs_left.mean() if abs_left.size > 0 else 1e-9
        mean_abs_right = abs_right.mean() if abs_right.size > 0 else 1e-9

        mean_abs_ratio_lr = float(mean_abs_right / (mean_abs_left + 1e-9))
        if not np.isfinite(mean_abs_ratio_lr):
            mean_abs_ratio_lr = 1.0

        tid = f"E{E_val:.3f}_ion{int(ion_no)}_p{int(par)}"

        rows.append(
            {
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
            }
        )

        x_obs_list.append(
            [
                mean_depth,
                max_depth,
                n_vac,
                *rbin_fracs,
                asym_count,
                asym_nn,
                skew_x,
                var_diff_x,
                mean_abs_ratio_lr,
            ]
        )

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

    df_summary = pd.DataFrame(rows)

    print(f"[DEBUG] Tracks: {len(track_ids)}")
    print(f"[DEBUG] x_obs shape: {x_obs.shape}")
    print(f"[DEBUG] theta shape: {theta.shape}")

    return x_obs, theta, track_ids, {"rel_bin_edges": r_edges}, df_summary