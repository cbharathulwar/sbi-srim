"""Finalize SIIMPL v2 dataset: concatenate Pool A + Pool B, audit uniqueness,
verify per-bin coverage at rotated axes, and rename to canonical filenames.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channeling_axes import AXES_LAB, D, PREFAC_eV_A

DATA = Path(r"C:/Users/walsworthlab/Inverse ML/data/siimpl")
E_BINS = [(1.0, 4.66, "Low"), (4.66, 21.7, "Mid"), (21.7, 105.001, "High")]


def audit_uniqueness(df):
    fps = {}
    for ion, grp in df.groupby("ion_number"):
        fps[ion] = hashlib.md5(grp[["x","y","z"]].to_numpy().tobytes()).hexdigest()
    return len(fps), len(set(fps.values()))


def verify_coverage(df, n_target):
    tracks = df.drop_duplicates(subset="ion_number").reset_index(drop=True)
    v = tracks[["target_vx","target_vy","target_vz"]].to_numpy()
    E_keV = tracks["energy_keV"].to_numpy()
    E_eV = E_keV * 1000.0
    print(f"{'Axis':<8} {'E-bin':<6} {'N':>9} {'Within ψ_c':>13} "
          f"{'Status':>10}")
    print("-" * 60)
    all_pass = True
    rows = []
    for fam, axs in AXES_LAB.items():
        cos_max = np.clip(np.max(np.abs(v @ axs.T), axis=1), -1, 1)
        angle = np.degrees(np.arccos(cos_max))
        psi_c = np.degrees(np.sqrt(PREFAC_eV_A / (E_eV * D[fam])))
        within = angle <= psi_c
        for lo, hi, name in E_BINS:
            m = (E_keV >= lo) & (E_keV < hi)
            n = int(m.sum())
            w = int((within & m).sum())
            ok = w >= n_target
            if not ok:
                all_pass = False
            rows.append({"axis": fam, "e_bin": name, "n": n, "within": w,
                         "target": n_target, "pass": ok})
            print(f"{fam:<8} {name:<6} {n:>9,} {w:>13,} "
                  f"{('PASS' if ok else f'FAIL(-{n_target-w})'):>10}")
    return all_pass, rows


def process(split: str, n_target: int):
    print(f"\n{'='*72}")
    print(f"FINALIZE {split.upper()}  (target {n_target}/bin)")
    print(f"{'='*72}")
    pa = pd.read_csv(DATA / f"siimpl_{split}_poolA_v2.csv")
    pb = pd.read_csv(DATA / f"siimpl_{split}_poolB_v2.csv")
    combined = pd.concat([pa, pb], ignore_index=True)
    print(f"Pool A: {pa['ion_number'].nunique():,} tracks, {len(pa):,} rows")
    print(f"Pool B: {pb['ion_number'].nunique():,} tracks, {len(pb):,} rows")
    print(f"Combined: {combined['ion_number'].nunique():,} tracks, "
          f"{len(combined):,} rows")

    print("\nUniqueness audit (this may take a minute for train)...", flush=True)
    n_ions, n_unique = audit_uniqueness(combined)
    pct = n_unique / n_ions * 100
    print(f"  {n_unique:,} / {n_ions:,} unique cascade fingerprints "
          f"({pct:.4f}%)")

    print("\nCoverage verification:")
    all_pass, rows = verify_coverage(combined, n_target)

    canonical = DATA / f"siimpl_{split}.csv"
    print(f"\nWriting canonical: {canonical}")
    combined.to_csv(canonical, index=False)

    summary = {
        "split": split,
        "n_tracks": int(combined["ion_number"].nunique()),
        "n_rows": len(combined),
        "n_unique_cascades": n_unique,
        "uniqueness_pct": pct,
        "n_target_per_bin": n_target,
        "all_bins_pass": all_pass,
        "coverage_rows": rows,
        "pool_A_tracks": int(pa["ion_number"].nunique()),
        "pool_B_tracks": int(pb["ion_number"].nunique()),
    }
    (DATA / f"siimpl_{split}.summary.json").write_text(json.dumps(summary, indent=2))
    return all_pass


def main():
    train_ok = process("train", 3000)
    eval_ok  = process("eval", 600)
    print(f"\n{'='*72}")
    print(f"OVERALL: TRAIN={'PASS' if train_ok else 'FAIL'}, "
          f"EVAL={'PASS' if eval_ok else 'FAIL'}")
    print(f"{'='*72}")
    print("\nCanonical files:")
    print(f"  {DATA / 'siimpl_train.csv'}")
    print(f"  {DATA / 'siimpl_eval.csv'}")


if __name__ == "__main__":
    main()
