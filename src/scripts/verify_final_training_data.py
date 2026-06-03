"""End-to-end verification of the canonical siimpl_train.csv and siimpl_eval.csv.

Checks every requirement we set when designing the dataset:
  1. Track scale (~250k train, ~50k eval)
  2. Schema matches v2 SRIM exactly
  3. Cascade uniqueness (target: > 99.9%)
  4. Per (axis x energy) bin coverage hits target (3000 train / 600 eval)
  5. Channeling axes targeted are the ROTATED lab-frame axes (not literal)
  6. Energy distribution is log-uniform on [1, 105] keV
  7. Direction distribution is upper-hemisphere uniform (vz >= 0)
  8. Label tuples (E, vx, vy, vz) are valid unit vectors with correct E range
  9. Empty / NaN / out-of-range sanity
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channeling_axes import AXES_LAB, D, PREFAC_eV_A, DIAMOND_PREROT_DEG

DATA = Path(r"C:/Users/walsworthlab/Inverse ML/data/siimpl")
EXPECTED_SCHEMA = ["x", "y", "z", "ion_number", "energy_keV", "theta_deg",
                   "target_vx", "target_vy", "target_vz"]
E_BINS = [(1.0, 4.66, "Low"), (4.66, 21.7, "Mid"), (21.7, 105.001, "High")]


def check_schema(df):
    cols = list(df.columns)
    ok = cols == EXPECTED_SCHEMA
    return ok, cols


def check_no_nan(df):
    n_nan = df.isna().sum().sum()
    return n_nan == 0, int(n_nan)


def check_label_unit_vectors(tracks):
    v = tracks[["target_vx","target_vy","target_vz"]].to_numpy()
    norms = np.linalg.norm(v, axis=1)
    min_n, max_n = norms.min(), norms.max()
    ok = (abs(min_n - 1.0) < 1e-3) and (abs(max_n - 1.0) < 1e-3)
    return ok, (min_n, max_n)


def check_upper_hemisphere(tracks):
    vz = tracks["target_vz"].to_numpy()
    n_below = (vz < -1e-6).sum()
    return n_below == 0, int(n_below), vz.min()


def check_energy_range(tracks, lo=1.0, hi=105.0):
    E = tracks["energy_keV"].to_numpy()
    n_below = (E < lo - 1e-6).sum()
    n_above = (E > hi + 1e-6).sum()
    return (n_below == 0 and n_above == 0), int(n_below), int(n_above), (E.min(), E.max())


def check_log_uniform(tracks):
    """Crude check: log(E) should be roughly uniform on [log(1), log(105)]."""
    E = tracks["energy_keV"].to_numpy()
    logE = np.log(E)
    expected_mean = (np.log(1) + np.log(105)) / 2
    actual_mean = logE.mean()
    bias_pct = (actual_mean - expected_mean) / expected_mean * 100 if expected_mean else 0
    return abs(bias_pct) < 5, bias_pct, actual_mean, expected_mean


def check_upper_hemisphere_uniform(tracks):
    """Crude: |vz| should average 0.5 for uniform-on-upper-S^2 (since |vz|
    in [0, 1] is uniform for hemisphere sampling)."""
    vz = tracks["target_vz"].to_numpy()
    mean_vz = vz.mean()
    return abs(mean_vz - 0.5) < 0.03, mean_vz


def check_uniqueness(df):
    fps = {}
    for ion, grp in df.groupby("ion_number"):
        fps[ion] = hashlib.md5(grp[["x","y","z"]].to_numpy().tobytes()).hexdigest()
    n = len(fps)
    u = len(set(fps.values()))
    return (u / n > 0.999), n, u


def check_coverage(tracks, n_target):
    v = tracks[["target_vx","target_vy","target_vz"]].to_numpy()
    E_keV = tracks["energy_keV"].to_numpy()
    E_eV = E_keV * 1000.0
    rows = []
    all_pass = True
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
            if not ok: all_pass = False
            rows.append((fam, name, n, w, ok))
    return all_pass, rows


def run(split: str, n_target: int):
    print(f"\n{'='*72}")
    print(f"VERIFY {split.upper()}")
    print(f"{'='*72}")
    path = DATA / f"siimpl_{split}.csv"
    print(f"Reading {path} ({path.stat().st_size/1e9:.2f} GB)...", flush=True)
    df = pd.read_csv(path)
    tracks = df.drop_duplicates(subset="ion_number").reset_index(drop=True)
    n_ions = len(tracks)
    n_rows = len(df)

    results = []

    print(f"\n1. Scale: {n_ions:,} unique ions, {n_rows:,} vacancy rows")
    results.append(("Scale", n_ions >= (250_000 if split=='train' else 50_000)))

    print(f"\n2. Schema check:")
    ok, cols = check_schema(df)
    print(f"   {'PASS' if ok else 'FAIL'}: columns = {cols}")
    results.append(("Schema", ok))

    print(f"\n3. No NaN values:")
    ok, n = check_no_nan(df)
    print(f"   {'PASS' if ok else 'FAIL'}: {n} NaN values")
    results.append(("No NaN", ok))

    print(f"\n4. Labels are unit vectors:")
    ok, (mn, mx) = check_label_unit_vectors(tracks)
    print(f"   {'PASS' if ok else 'FAIL'}: |v| in [{mn:.6f}, {mx:.6f}]")
    results.append(("Unit vectors", ok))

    print(f"\n5. All labels in upper hemisphere (vz >= 0):")
    ok, n_below, vz_min = check_upper_hemisphere(tracks)
    print(f"   {'PASS' if ok else 'FAIL'}: {n_below} tracks with vz < 0, vz_min = {vz_min:.6f}")
    results.append(("Upper hemisphere", ok))

    print(f"\n6. Energy in [1.0, 105.0] keV:")
    ok, nb, na, (e_min, e_max) = check_energy_range(tracks)
    print(f"   {'PASS' if ok else 'FAIL'}: {nb} below + {na} above, E range "
          f"[{e_min:.4f}, {e_max:.4f}]")
    results.append(("Energy range", ok))

    print(f"\n7. Log-uniform energy distribution:")
    ok, bias, actual, expected = check_log_uniform(tracks)
    print(f"   {'PASS' if ok else 'FAIL'}: mean log E = {actual:.4f} (expected "
          f"{expected:.4f}, bias {bias:+.2f}%)")
    results.append(("Log-uniform E", ok))

    print(f"\n8. Upper-hemisphere uniform directions (mean vz ≈ 0.5):")
    ok, mean_vz = check_upper_hemisphere_uniform(tracks)
    print(f"   {'PASS' if ok else 'FAIL'}: mean vz = {mean_vz:.4f}")
    results.append(("Direction uniform", ok))

    print(f"\n9. Cascade uniqueness (target > 99.9%):")
    print(f"   Hashing all {n_ions:,} cascades...", flush=True)
    ok, n, u = check_uniqueness(df)
    print(f"   {'PASS' if ok else 'FAIL'}: {u:,}/{n:,} unique ({u/n*100:.4f}%)")
    results.append(("Uniqueness", ok))

    print(f"\n10. Per-bin coverage at ROTATED lab-frame axes "
          f"(target ≥ {n_target}/bin, R_y({DIAMOND_PREROT_DEG}°) pre-rotation):")
    ok, rows = check_coverage(tracks, n_target)
    print(f"    {'Axis':<8} {'E-bin':<6} {'N':>9} {'Within ψ_c':>13} {'Status':>10}")
    for fam, name, n, w, p in rows:
        print(f"    {fam:<8} {name:<6} {n:>9,} {w:>13,} "
              f"{'PASS' if p else 'FAIL':>10}")
    print(f"   {'PASS' if ok else 'FAIL'} overall")
    results.append(("Coverage", ok))

    # Final summary
    all_pass = all(p for _, p in results)
    print(f"\n{'-'*72}")
    print(f"SUMMARY FOR {split.upper()}:")
    for name, p in results:
        print(f"   [{'✓' if p else '✗'}] {name}")
    print(f"\n   {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    return all_pass


def main():
    train_ok = run("train", 3000)
    eval_ok  = run("eval", 600)
    print(f"\n{'='*72}")
    print(f"OVERALL VERDICT")
    print(f"{'='*72}")
    print(f"  TRAIN: {'PASS' if train_ok else 'FAIL'}")
    print(f"  EVAL:  {'PASS' if eval_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
