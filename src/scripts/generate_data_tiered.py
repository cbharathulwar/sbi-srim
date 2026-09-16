"""Generate additional crystal-rotation tracks restricted to one energy tier,
to grow the train/eval corpus to 1M/100k with a deliberate 25/30/45 low/mid/high
weighting (up-weighting mid/high energy for the A100 rebuild's target regime).

Reuses the exact same SIIMPL call, rotation/frame convention, and CSV schema
as generate_data.py (rotate_crystal_mode=True, crystal-frame storage) so the
output is a drop-in append to the existing siimpl_train.csv / siimpl_eval_merged.csv.

Usage:
    python src/scripts/generate_data_tiered.py --split train_low --e_lo 1.0 --e_hi 5.0 \
        --n_cfg 3000 --k 50 --seed 90001 --out_suffix _extra_low
"""
import os, sys, csv, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SIIMPL = os.environ.get("SIIMPL_PYTHON")
if _SIIMPL:
    sys.path.insert(0, _SIIMPL)

import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.spatial.transform import Rotation
from siimpl import Siimpl

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(_ROOT, "data", "siimpl_rot")
os.makedirs(OUT, exist_ok=True)

EDISPL = 43.0


def invert(g):
    th = float(np.degrees(np.arccos(np.clip(g[2], -1, 1))))
    ph = float((180.0 - np.degrees(np.arctan2(g[1], g[0]))) % 360.0)
    return th, ph


def sample_iso(rng):
    g = rng.standard_normal(3)
    g /= np.linalg.norm(g)
    g[2] = abs(g[2])
    return g


def gen(out_path, n_cfg, k, seed, e_lo, e_hi):
    rng = np.random.default_rng(seed)
    f = open(out_path, "w", newline=""); w = csv.writer(f)
    w.writerow(["x", "y", "z", "ion_number", "energy_keV", "theta_deg", "phi_deg",
                "target_vx", "target_vy", "target_vz"])
    t0 = time.perf_counter(); nid = 0; nemp = 0
    for i in range(n_cfg):
        E = float(np.exp(rng.uniform(np.log(e_lo), np.log(e_hi))))
        g = sample_iso(rng)
        th, ph = invert(g)
        R = Rotation.from_euler("YZ", [th, ph], degrees=True)
        try:
            sim = Siimpl.diamond(ion="C", energy_keV=E, theta=th, phi=ph, n_ions=k,
                                 crystalline=True, full_cascade=True,
                                 rotate_crystal_mode=True, Edispl_eV=EDISPL)
            sim.run()
            for tr in sim.tracks:
                if tr.n_vacancies == 0:
                    nemp += 1; continue
                cl = R.inv().apply(np.asarray(tr.vacancies, float))
                for (x, y, z) in cl:
                    w.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", nid, f"{E:.4f}",
                                f"{th:.4f}", f"{ph:.4f}",
                                f"{g[0]:.6f}", f"{g[1]:.6f}", f"{g[2]:.6f}"])
                nid += 1
            sim.close()
        except Exception as e:
            print(f"  [warn] cfg {i}: {e}", flush=True); continue
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  [{out_path}] cfg {i+1}/{n_cfg} tracks={nid:,} "
                  f"{(i+1)*k/el:.0f} ion/s ETA {(n_cfg-i-1)*k/max((i+1)*k/el,1)/60:.1f}min", flush=True)
            f.flush()
    f.close()
    print(f"[{out_path}] DONE {nid:,} tracks, {nemp} empty, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return nid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--e_lo", type=float, required=True)
    ap.add_argument("--e_hi", type=float, required=True)
    ap.add_argument("--n_cfg", type=int, required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out_path = os.path.join(OUT, a.out_name)
    print(f"=== {a.out_name}: {a.n_cfg} cfg x {a.k} ions, E in [{a.e_lo},{a.e_hi}) seed={a.seed} ===", flush=True)
    n = gen(out_path, a.n_cfg, a.k, a.seed, a.e_lo, a.e_hi)
    print(f"ALL DONE {a.out_name}: {n} tracks", flush=True)
