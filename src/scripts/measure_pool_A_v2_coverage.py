"""Measure Pool A v2 channeling coverage using ROTATED lab-frame axes.

Uses channeling_axes.AXES_LAB (literal <hkl> rotated by R_y(35 deg) and filtered
to upper hemisphere) to correctly count channeled tracks in lab frame.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channeling_axes import AXES_LAB, D, PREFAC_eV_A

E_BINS = [(1.0, 4.66, "Low"), (4.66, 21.7, "Mid"), (21.7, 105.0001, "High")]


def measure(csv_path: Path, out_path: Path, n_target: int, label: str):
    print(f"Reading {csv_path} ...", flush=True)
    df = pd.read_csv(
        csv_path,
        usecols=["ion_number", "energy_keV", "target_vx", "target_vy", "target_vz"],
    )
    tracks = df.drop_duplicates(subset="ion_number").reset_index(drop=True)
    print(f"Tracks: {len(tracks):,}", flush=True)

    v = tracks[["target_vx", "target_vy", "target_vz"]].to_numpy()
    E_keV = tracks["energy_keV"].to_numpy()
    E_eV = E_keV * 1000.0

    print(f"\n{label} coverage (rotated lab-frame axes, N_target={n_target})")
    print(f"{'Axis':<8} {'E-bin':<6} {'N':>9} {'Within psi_c':>13} {'Frac':>8} "
          f"{'Deficit':>9}")
    print("-" * 60)

    family_within = {}
    for fam, axs in AXES_LAB.items():
        cos_max = np.clip(np.max(np.abs(v @ axs.T), axis=1), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_max))
        psi_c = np.degrees(np.sqrt(PREFAC_eV_A / (E_eV * D[fam])))
        family_within[fam] = angle_deg <= psi_c

    table = []
    for fam in AXES_LAB:
        for lo, hi, name in E_BINS:
            m = (E_keV >= lo) & (E_keV < hi)
            n_bin = int(m.sum())
            n_within = int((family_within[fam] & m).sum())
            deficit = max(0, n_target - n_within)
            table.append({
                "axis": fam, "e_bin": name, "e_lo": lo, "e_hi": hi,
                "n_pool_A_in_bin": n_bin, "n_within_psi_c": n_within,
                "fraction_channeled": n_within / n_bin if n_bin else 0.0,
                "n_target": n_target, "pool_B_deficit": deficit,
            })
            print(f"{fam:<8} {name:<6} {n_bin:>9,} {n_within:>13,} "
                  f"{n_within/n_bin*100:>7.2f}% {deficit:>9,}")

    total_def = sum(r["pool_B_deficit"] for r in table)
    print("-" * 60)
    print(f"Total Pool B deficit: {total_def:,}")

    out = {
        "label": label,
        "n_pool_A_tracks": len(tracks),
        "n_target_per_bin": n_target,
        "table": table,
        "total_pool_B_deficit": total_def,
        "psi_c_constants": {"prefactor_eV_A": PREFAC_eV_A, "row_spacing_A": D},
        "axes_lab_frame": {k: v.tolist() for k, v in AXES_LAB.items()},
        "energy_bins": [{"name": n, "lo": lo, "hi": hi} for lo, hi, n in E_BINS],
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-target", type=int, required=True)
    ap.add_argument("--label", default="POOL A")
    args = ap.parse_args()
    measure(args.csv, args.out, args.n_target, args.label)


if __name__ == "__main__":
    main()
