

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def infer_relative_bin_edges(n_bins=6, r_min=1e-3, r_max=1.0):
    edges = np.geomspace(r_min, r_max, n_bins) #geomspace functoin returns numbers spaced evenly on a log scale
    edges = np.insert(edges, 0, 0.0) # you need to start at 0 but cannot with geom space because log 0 is -inf so you add it here (6 bins requires 7 edges)
    return edges

def relative_bin_fractions_from_events(depths_A, norm_depth_A, r_edges):
    x = np.asarray(depths_A, float)
    #make sure nothing funky is going on
    x = x[np.isfinite(x)]
    if x.size == 0 or norm_depth_A <= 0:
        return np.zeros(len(r_edges) - 1, float)

    
    #add the 1e-12 to avoid the divide by 0 error just in case 
    r = x / (norm_depth_A + 1e-12) 
    hist, _ = np.histogram(r, bins=r_edges) #build histogram
    hist[-1] += np.sum(r > r_edges[-1])  # overflow, finds all depts grater than last bin top edge tgeb adds to the last bin

    total = hist.sum() #adds all values in each bin of hist 
    if total == 0:
        return np.zeros_like(hist, dtype=float)

    return hist / total #final fraction


def preprocess(data_path: str | Path, n_bins: int = 6):
    # 1) Load with headers (your split CSVs have headers)
    df = pd.read_csv(data_path)

    # 2) Normalize column names if they exist
    rename_map = {
        "x_ang": "x",
        "y_ang": "y",
        "z_ang": "z",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3) Coerce numerics
    #    Figure out which energy column is present and convert ONCE to keV
    for c in ["x", "y", "z", "ion_number"]:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in {data_path}")

    # Decide energy source
    energy_keV = None
    if "energy_keV" in df.columns:
        # Already keV
        energy_keV = pd.to_numeric(df["energy_keV"], errors="coerce")
    elif "energy_eV" in df.columns:
        # eV → keV
        energy_keV = pd.to_numeric(df["energy_eV"], errors="coerce") / 1e3
    elif "energy" in df.columns:
        # Detect units by magnitude
        e_raw = pd.to_numeric(df["energy"], errors="coerce")
        if np.nanmax(e_raw) > 3000:   # almost surely eV
            energy_keV = e_raw / 1e3
        else:                         # already keV
            energy_keV = e_raw
    else:
        raise KeyError("No energy column found: expected one of ['energy_keV','energy_eV','energy'].")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["ion_number"] = pd.to_numeric(df["ion_number"], errors="coerce")
    df["energy_keV"] = pd.to_numeric(energy_keV, errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV"]).reset_index(drop=True)

    # 4) Relative-depth bin edges (unchanged)
    bin_edges = infer_relative_bin_edges(n_bins=n_bins)

    rows, x_obs_list, theta_list, track_ids = [], [], [], []

    # Group by rounded keV + ion_number
    df["energy_int"] = df["energy_keV"].round().astype(int)

    for (E_int, ion_no), g in df.groupby(["energy_int", "ion_number"], sort=False):
        x = np.abs(g["x"].to_numpy(float))
        if x.size == 0:
            continue

        mean_depth = float(x.mean())
        max_depth  = float(np.max(x))
        norm_depth = float(np.percentile(x, 95))
        n_vac      = int(x.size)

        rbin_fracs = relative_bin_fractions_from_events(x, norm_depth, bin_edges)
        if rbin_fracs.sum() > 0:
            rbin_fracs = rbin_fracs / (rbin_fracs.sum() + 1e-12)

        E_keV = float(E_int)  # θ in keV (rounded bin label)
        tid   = f"C_{int(E_keV)}keV_ion{int(ion_no)}"

        rows.append({
            "track_id": tid,
            "ion": "C",
            "energy_keV": E_keV,
            "mean_depth_A": mean_depth,
            "max_depth_A": max_depth,
            "vacancies_per_ion": n_vac,
            **{f"rbin_frac_{i+1}": float(v) for i, v in enumerate(rbin_fracs)}
        })

        x_obs_list.append([mean_depth, max_depth, n_vac, *rbin_fracs])
        theta_list.append([E_keV])
        track_ids.append(tid)

    df_summary = pd.DataFrame(rows)

    # Debug sanity
    print("[DEBUG] energy_keV range:", df_summary["energy_keV"].min(), "→", df_summary["energy_keV"].max())
    print("[DEBUG] example energies (keV):", df_summary["energy_keV"].unique()[:10])

    x_obs = torch.tensor(np.asarray(x_obs_list, dtype=np.float32))
    theta = torch.tensor(np.asarray(theta_list, dtype=np.float32))

    return x_obs, theta, track_ids, {"rel_bin_edges": bin_edges}, df_summary




def make_x_test(df_summary, n_per_energy=1, random_state=42):
    """
    Pick a few sample tracks per energy level for testing.

    Tracks are grouped by energy (ion type is already unique from preprocess()).
    """

    # round energy to avoid small differences
    df_summary = df_summary.copy()
    df_summary["energy_int"] = df_summary["energy_keV"].round().astype(int)

    sampled_rows = []
    for energy, group in df_summary.groupby("energy_int"):
        pick = group.sample(n=min(n_per_energy, len(group)), random_state=random_state)
        sampled_rows.append(pick)

    x_test = pd.concat(sampled_rows).reset_index(drop=True)

    # quick summary
    print(f"[INFO] Picked {len(x_test)} test tracks across {x_test['energy_int'].nunique()} energies.")
    print(f"[DEBUG] Tracks per energy:\n{x_test.groupby('energy_int').size()}")

    x_test_ids = x_test["track_id"].tolist()
    return x_test, x_test_ids