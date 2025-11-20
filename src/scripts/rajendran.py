import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.srim_utils import parse_collisions


# ==========================================================
# CONFIG
# ==========================================================

BASE = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/nov19run5")

ENERGIES = [1, 3, 10, 20, 30, 50, 70, 100]
OUTPUT_RESULTS = BASE / "rajendran_results.csv"
OUTPUT_CURVES  = BASE / "rajendran_asymmetry_curves.csv"


# ==========================================================
# COMPUTE TRACK ASYMMETRY
# ==========================================================

def compute_asymmetry(track_df):
    """
    Rajendran asymmetry A = N_end / N_begin
    where N_begin = first 1/3 of track length,
          N_end   = last 1/3
    """
    s = track_df["x"].values
    if len(s) < 3:
        return np.nan

    s_min, s_max = s.min(), s.max()
    L = s_max - s_min
    if L <= 0:
        return np.nan

    b_end = s_min + L/3
    e_start = s_min + 2*L/3

    N_begin = np.sum((s >= s_min) & (s < b_end))
    N_end   = np.sum((s >= e_start) & (s <= s_max))

    if N_begin == 0:
        return np.nan

    return N_end / N_begin


# ==========================================================
# FIND THRESHOLD T FOR 5% FALSE POSITIVES
# ==========================================================

def compute_threshold_and_efficiency(A_vals, fp_target=0.05):
    A_vals = np.array(A_vals)
    A_vals = A_vals[np.isfinite(A_vals)]

    # A_mag = fold backward + forward
    A_mag = np.where(A_vals >= 1, A_vals, 1/A_vals)

    # candidates for T = unique A_mag
    cand = np.sort(np.unique(A_mag[A_mag > 1]))

    best_T, best_eff = None, 0.0

    for T in cand:
        # false positives = A < 1/T
        fp = np.mean(A_vals < (1.0 / T))
        if fp <= fp_target:
            eff = np.mean(A_vals > T)
            if eff > best_eff:
                best_eff = eff
                best_T = T

    return best_T, best_eff


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

all_rows = []
curve_rows = []

x_grid = np.logspace(-1, 1.2, 300)

for E in ENERGIES:
    folder = BASE / f"{E:.1f}keV"

    print(f"\n[INFO] Processing {E} keV in {folder}")

    df = parse_collisions(folder, E)
    if df.empty:
        print(f"[WARN] No data for {E} keV")
        continue

    asym_vals = []

    # --- group by ion track ---
    for ion, ion_df in df.groupby("ion_number"):
        A = compute_asymmetry(ion_df)
        if np.isfinite(A):
            asym_vals.append(A)

    if len(asym_vals) == 0:
        continue

    # Compute threshold & efficiency
    T, eff = compute_threshold_and_efficiency(asym_vals)

    print(f"   → T = {T:.3f}, Efficiency = {eff:.3f} using {len(asym_vals)} tracks")

    # Save summary row
    all_rows.append({
        "energy_keV": E,
        "n_tracks": len(asym_vals),
        "T_threshold": T,
        "raj_efficiency": eff
    })

    # Compute cumulative curves for plotting
    A = np.array(asym_vals)
    frac = np.array([(A >= x).mean() for x in x_grid])

    for x, f in zip(x_grid, frac):
        curve_rows.append({"energy_keV": E, "x": x, "fraction": f})


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

pd.DataFrame(all_rows).to_csv(OUTPUT_RESULTS, index=False)
pd.DataFrame(curve_rows).to_csv(OUTPUT_CURVES, index=False)

print("\nSaved:")
print(f" - {OUTPUT_RESULTS}")
print(f" - {OUTPUT_CURVES}")