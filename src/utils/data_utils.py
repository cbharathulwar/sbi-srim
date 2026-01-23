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

# def preprocess_mnpe(data_path: str | Path, n_bins: int = 6):
#     """
#     Preprocess SRIM collision CSV for MNPE (energy + parity) model.

#     Features per track:
#       [mean_depth, max_depth, n_vac,
#        rbin_frac_1..n_bins,
#        asym_count_centered, asym_nn_centered,
#        skew_x, var_diff_x, mean_abs_ratio_lr]

#     θ per track:
#       [energy_keV (continuous), parity ∈ {0,1}]
#     """
#     df = pd.read_csv(data_path)
    

#     # Standardize column names
#     rename_map = {"x_ang": "x", "y_ang": "y", "z_ang": "z"}
#     df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

#     required_cols = ["x", "y", "z", "ion_number", "parity"]
#     for c in required_cols:
#         if c not in df.columns:
#             raise KeyError(f"Missing column '{c}' in {data_path}")

#     # Energy handling → energy_keV
#     if "energy_keV" in df.columns:
#         df["energy_keV"] = pd.to_numeric(df["energy_keV"], errors="coerce")
#     elif "energy_eV" in df.columns:
#         df["energy_keV"] = pd.to_numeric(df["energy_eV"], errors="coerce") / 1e3
#     elif "energy" in df.columns:
#         e_raw = pd.to_numeric(df["energy"], errors="coerce")
#         df["energy_keV"] = e_raw / 1e3 if e_raw.max() > 3000 else e_raw
#     else:
#         raise KeyError("No energy column found for MNPE preprocessing.")

#     # Ensure numeric
#     for col in ["x", "y", "z", "ion_number", "energy_keV", "parity"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Drop bad rows
#     df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV", "parity"]).reset_index(drop=True)

#     # Continuous energy
#     df["energy_norm"] = df["energy_keV"]

#     # Relative bin edges (shared helper)
#     r_edges = infer_relative_bin_edges(n_bins=n_bins)

#     rows = []
#     x_obs_list = []
#     theta_list = []
#     track_ids = []

#     for (E_val, ion_no, par), g in df.groupby(["energy_norm", "ion_number", "parity"], sort=False):
#         x = np.asarray(g["x"], float)
#         z = np.asarray(g["z"], float)
#         if x.size == 0:
#             continue

#         # Base features
#         abs_x = np.abs(x)
#         mean_depth = float(np.mean(abs_x))
#         max_depth = float(np.max(abs_x))
#         norm_depth = float(np.percentile(abs_x, 95))
#         n_vac = int(len(x))

#                 # FIXED RBIN DEFINITION — USE TRACK EXTENT, NOT abs(x)
#         depth_coord = x - np.min(x)                  # 0 → track end
#         norm_depth_bins = float(np.percentile(depth_coord, 95))

#         rbin_fracs = relative_bin_fractions_from_events(
#             depth_coord, norm_depth_bins, r_edges
#         )
#         asym_count = compute_centered_track_asymmetry(x)
#         asym_nn = compute_centered_nn_asymmetry(x, z)

#         # Skewness
#         try:
#             skew_x = float(skew(x)) if x.size > 2 else 0.0
#         except Exception:
#             skew_x = 0.0
#         if not np.isfinite(skew_x):
#             skew_x = 0.0

#         # Left/right variance difference
#         left_x = x[x < 0]
#         right_x = x[x > 0]

#         var_left = np.var(left_x) if left_x.size > 2 else 0.0
#         var_right = np.var(right_x) if right_x.size > 2 else 0.0
#         var_diff_x = float(var_right - var_left)
#         if not np.isfinite(var_diff_x):
#             var_diff_x = 0.0

#         # Left/right mean absolute ratio
#         abs_left = np.abs(left_x)
#         abs_right = np.abs(right_x)
#         mean_abs_left = abs_left.mean() if abs_left.size > 0 else 1e-9
#         mean_abs_right = abs_right.mean() if abs_right.size > 0 else 1e-9

#         mean_abs_ratio_lr = float(mean_abs_right / (mean_abs_left + 1e-9))
#         if not np.isfinite(mean_abs_ratio_lr):
#             mean_abs_ratio_lr = 1.0

#         tid = f"E{E_val:.3f}_ion{int(ion_no)}_p{int(par)}"

#         rows.append(
#             {
#                 "track_id": tid,
#                 "energy_keV": float(E_val),
#                 "parity": int(par),
#                 "mean_depth_A": mean_depth,
#                 "max_depth_A": max_depth,
#                 "vacancies_per_ion": n_vac,
#                 **{f"rbin_frac_{i+1}": rbin_fracs[i] for i in range(n_bins)},
#                 "asym_count_centered": asym_count,
#                 "asym_nn_centered": asym_nn,
#                 "skew_x": skew_x,
#                 "var_diff_x": var_diff_x,
#                 "mean_abs_ratio_lr": mean_abs_ratio_lr,
#             }
#         )

#         x_obs_list.append(
#             [
#                 mean_depth,
#                 max_depth,
#                 n_vac,
#                 *rbin_fracs,
#                 asym_count,
#                 asym_nn,
#                 skew_x,
#                 var_diff_x,
#                 mean_abs_ratio_lr,
#             ]
#         )

#         theta_list.append([float(E_val), float(int(par))])
#         track_ids.append(tid)

#     if len(x_obs_list) == 0:
#         print("[WARN] preprocess_mnpe found NO valid tracks.")
#         return (
#             torch.zeros((0, 14)),
#             torch.zeros((0, 2)),
#             [],
#             {"rel_bin_edges": r_edges},
#             pd.DataFrame(rows),
#         )

#     x_obs = torch.tensor(np.asarray(x_obs_list, dtype=np.float32))
#     theta = torch.tensor(np.asarray(theta_list, dtype=np.float32))
#     theta[:, 1] = (theta[:, 1] == 1).float()

#     df_summary = pd.DataFrame(rows)

#     print(f"[DEBUG] Tracks: {len(track_ids)}")
#     print(f"[DEBUG] x_obs shape: {x_obs.shape}")
#     print(f"[DEBUG] theta shape: {theta.shape}")

#     return x_obs, theta, track_ids, {"rel_bin_edges": r_edges}, df_summary


import numpy as np
import pandas as pd
import torch
from scipy.stats import skew

# You likely have these helpers defined elsewhere, but ensuring they are imported
# from src.utils.data_utils import infer_relative_bin_edges, relative_bin_fractions_from_events, ...

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew

def preprocess_mnpe(data_path, n_bins=6):
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
    # MNPE estimates p(theta|x) for mixed continuous/discrete types [cite: 5, 34]
    for (E, ion, par), g in df.groupby(["energy_keV", "ion_number", "parity"], sort=False):
        raw_z = g["z"].values.astype(float)
        x_coords = g["x"].values.astype(float)
        
        if len(raw_z) < 5: 
            continue
        
        # --- THE FIX: CENTERING ONLY ---
        # Centering removes the 'detector position' bias so the model 
        # must learn the internal track directionality (Bragg Peak).
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
        
        # Skewness is the key discriminator for Bragg peaks
        try: 
            skew_z = float(skew(z))
        except: 
            skew_z = 0.0
            
        # Log-Variance Difference to handle the 10^11 scale mismatch
        var_up = np.var(z[z < 0]) if np.any(z < 0) else 0.0
        var_down = np.var(z[z > 0]) if np.any(z > 0) else 0.0
        var_diff = np.log1p(var_down) - np.log1p(var_up)

        # Assemble summary feature vector
        row = [mean_depth, max_depth, len(z), *r_fracs, asym_c, asym_nn, skew_z, var_diff]
        
        x_obs.append(row)
        # MNPE requires continuous (Energy) then discrete (Parity) 
        theta_list.append([float(E), float(par)])
        tids.append(f"E{E:.0f}_ion{int(ion)}")
        
        rows.append({
            "track_id": tids[-1], "energy": E, "parity": par, 
            "skew_z": skew_z, "var_diff": var_diff, "asym_count": asym_c
        })

    # 3. Tensor Conversion
    x_t = torch.tensor(x_obs, dtype=torch.float32)
    theta_t = torch.tensor(theta_list, dtype=torch.float32)

    # 4. ROBUST NORMALIZATION
    # Prevents large features (like 10^11 variance) from drowning out skewness.
    if len(x_t) > 1:
        mean = x_t.mean(dim=0)
        std = x_t.std(dim=0) + 1e-6
        x_t = (x_t - mean) / std
        print(f"[PREPROCESS] Normalized {len(x_t)} tracks. Feature Max: {x_t.max().item():.2f}")

    return x_t, theta_t, tids, {"edges": r_edges}, pd.DataFrame(rows)

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.utils.data_utils import infer_relative_bin_edges, relative_bin_fractions_from_events
import pandas as pd
import numpy as np
import torch
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew, kurtosis
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import skew
from pathlib import Path

import pandas as pd
import numpy as np
import torch

# def preprocess_mdpe(data_path: str | Path, n_radial_bins: int = 6, n_phi_bins: int = 8):
#     """
#     Optimized 22D Preprocessing for Diamond DM Detector.
#     Focuses on high-signal directional features and full-resolution histograms.
#     """
#     # =======================================================
#     # 0. Load + Surgical Column Flattening
#     # =======================================================
#     df = pd.read_csv(data_path)
#     df.columns = [c.strip() for c in df.columns]
#     df = df.loc[:, ~df.columns.duplicated()].copy()

#     rename_map = {
#         "ion #": "ion_number", "ion": "ion_number",
#         "energy": "energy_keV", "x_ang": "x", "y_ang": "y", "z_ang": "z"
#     }
#     df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

#     for col in ["energy_keV", "ion_number"]:
#         if col in df.columns:
#             if isinstance(df[col], pd.DataFrame):
#                 val_1d = df[col].iloc[:, 0]
#                 df = df.drop(columns=[col])
#                 df[col] = val_1d
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Map angle classes
#     ANGLE_BINS = np.array([0, 45, 90, 135, 180, 225, 270, 315])
#     df["angle_class"] = df["theta_deg"].apply(lambda a: int(np.argmin(np.abs(ANGLE_BINS - a))))
#     df = df.dropna(subset=["x", "y", "z", "energy_keV", "ion_number", "angle_class"]).reset_index(drop=True)

#     # --- FIX 1: Initialize track_ids list ---
#     x_obs_list, theta_list, track_ids = [], [], []

#     # =======================================================
#     # 1. High-Signal Processing (22D)
#     # =======================================================
#     for (E, ion, cls), g in df.groupby(["energy_keV", "ion_number", "angle_class"], sort=False):
#         pts = g[["x", "y", "z"]].to_numpy(float)
#         if pts.shape[0] < 10: continue 

#         # --- A. INTERNAL DENSITY ANCHOR ---
#         cm = np.mean(pts, axis=0)
#         centered_pts = pts - cm
#         dists = np.linalg.norm(centered_pts, axis=1)
        
#         tail_threshold = np.percentile(dists, 80)
#         tail_pts = pts[dists >= tail_threshold]
#         tail_anchor = np.mean(tail_pts, axis=0)
#         growth_vec = cm - tail_anchor 

#         # --- B. PCA SPINE ORIENTATION ---
#         cov = np.cov(centered_pts.T)
#         evals, evecs = np.linalg.eig(cov)
#         idx = evals.argsort()[::-1]
#         lam = np.maximum(evals[idx], 1e-12)
#         vec = evecs[:, idx]

#         if np.dot(vec[:, 0], growth_vec) < 0:
#             vec[:, 0] *= -1

#         # --- C. CORE FEATURES ---
#         total_var = float(np.sum(lam))
#         lin = float((lam[0] - lam[1]) / lam[0])
#         ent = float(-np.sum((lam/total_var) * np.log(lam/total_var + 1e-12)))
#         internal_offset = float(np.linalg.norm(cm - tail_anchor))

#         norm_r = dists / (np.percentile(dists, 95) + 1e-12)
#         r_fracs = np.histogram(norm_r, bins=n_radial_bins, range=(0,1))[0] / len(pts)

#         phi = np.degrees(np.arctan2(centered_pts[:, 1], centered_pts[:, 0])) % 360.0
#         phi_fracs = np.histogram(phi, bins=n_phi_bins, range=(0, 360))[0] / len(pts)

#         features = [
#             len(pts), np.max(dists),                    
#             *r_fracs,                                   
#             *phi_fracs,                                 
#             lin, ent,                                   
#             float(vec[0, 0]), float(vec[1, 0]),         
#             total_var, internal_offset                  
#         ]

#         x_obs_list.append(features)
#         theta_list.append([E, cls])
#         # --- FIX 2: Append the ID string ---
#         track_ids.append(f"E{E:.1f}_ion{int(ion)}_c{int(cls)}")

#     print(f"[MDPE] Preprocessed into {len(x_obs_list[0])}D vectors. Total Tracks: {len(x_obs_list)}")
    
#     # --- FIX 3: Return all 5 values ---
#     return (
#         torch.tensor(np.asarray(x_obs_list, np.float32)), 
#         torch.tensor(np.asarray(theta_list, np.float32)), 
#         track_ids,                         
#         {"r_edges": np.linspace(0, 1, n_radial_bins+1)}, 
#         pd.DataFrame()                     
#     )


import pandas as pd
import numpy as np
import torch
from scipy.optimize import curve_fit

from scipy.stats import skew, kurtosis

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

import numpy as np
import torch
import numpy as np
import torch

from scipy.spatial import cKDTree

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




