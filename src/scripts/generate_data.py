"""Generate the crystal-rotation (rotate_crystal_mode=True) training/eval datasets.

Verified recipe (crystal-frame storage, grazing-free):
  recoil direction g  ->  theta = arccos(g_z),  phi = (180 - atan2(g_y, g_x)) % 360
  SIIMPL call: rotate_crystal_mode=True  (beam stays normal; crystal is rotated)
  cloud_crystal = R^-1 . cloud_lab       (R = scipy Rotation.from_euler("YZ",[theta,phi]))
  target        = g                       (unit recoil direction, crystal frame)

Splits produced in data/siimpl_rot/:
  siimpl_train_poolA.csv   isotropic recoils (S^2-uniform prior)
  siimpl_train_poolB.csv   channeling-enriched (near the 26 low-index axes)
  siimpl_eval.csv          isotropic held-out evaluation set

Requires SIIMPL (ion-implantation BCA Monte-Carlo). Point SIIMPL_PYTHON at its
python package dir, or install it on PYTHONPATH.

    python src/scripts/generate_data.py --only all
"""
import os, sys, csv, json, time, argparse, itertools
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- locate SIIMPL (env override, else assume importable) ---
_SIIMPL = os.environ.get("SIIMPL_PYTHON")
if _SIIMPL:
    sys.path.insert(0, _SIIMPL)

import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.spatial.transform import Rotation
from siimpl import Siimpl

# --- output relative to repo root (../.. from this file) ---
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(_ROOT, "data", "siimpl_rot")
os.makedirs(OUT, exist_ok=True)

EDISPL = 43.0            # carbon displacement energy in diamond (eV)
E_LO, E_HI = 1.0, 105.0  # recoil energy range (keV), log-uniform


def invert(g):
    """Recoil direction g -> (theta, phi) beam angles for rotate_crystal_mode."""
    th = float(np.degrees(np.arccos(np.clip(g[2], -1, 1))))
    ph = float((180.0 - np.degrees(np.arctan2(g[1], g[0]))) % 360.0)
    return th, ph


def fam(v):
    """All symmetry-equivalent unit axes of a Miller family (permutations + sign flips)."""
    s = set()
    for p in itertools.permutations(v):
        for sg in itertools.product([1, -1], repeat=3):
            w = tuple(a * b for a, b in zip(p, sg))
            if any(w):
                s.add(w)
    A = np.array(sorted(s), float)
    return A / np.linalg.norm(A, axis=1, keepdims=True)


AXES = np.vstack([fam((1, 0, 0)), fam((1, 1, 0)), fam((1, 1, 1))])   # 26 low-index axes


def sample_iso(rng):
    """S^2-uniform recoil direction (upper hemisphere by convention)."""
    g = rng.standard_normal(3)
    g /= np.linalg.norm(g)
    g[2] = abs(g[2])
    return g


def sample_near_axis(rng, half_deg=10.0):
    """Recoil direction within `half_deg` of a randomly chosen low-index axis."""
    ax = AXES[rng.integers(len(AXES))]
    ca = np.cos(np.radians(half_deg))
    z = rng.uniform(ca, 1); ph = rng.uniform(0, 2 * np.pi)
    s = np.sqrt(max(0, 1 - z * z))
    loc = np.array([s * np.cos(ph), s * np.sin(ph), z])
    a = np.array([0, 0, 1.]); v = np.cross(a, ax); c = a @ ax
    if np.linalg.norm(v) < 1e-8:
        g = ax * np.sign(c)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
        g = Rm @ loc
    g /= np.linalg.norm(g); g[2] = abs(g[2])
    return g


def gen(split, n_cfg, k, seed, mode):
    rng = np.random.default_rng(seed)
    tp = os.path.join(OUT, f"siimpl_{split}.csv")
    f = open(tp, "w", newline=""); w = csv.writer(f)
    w.writerow(["x", "y", "z", "ion_number", "energy_keV", "theta_deg", "phi_deg",
                "target_vx", "target_vy", "target_vz"])
    t0 = time.perf_counter(); nid = 0; nemp = 0
    for i in range(n_cfg):
        E = float(np.exp(rng.uniform(np.log(E_LO), np.log(E_HI))))
        g = sample_iso(rng) if mode == "iso" else sample_near_axis(rng)
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
                cl = R.inv().apply(np.asarray(tr.vacancies, float))   # lab -> crystal frame
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
            print(f"  [{split}] cfg {i+1}/{n_cfg} tracks={nid:,} {(i+1)*k/el:.0f} ion/s "
                  f"ETA {(n_cfg-i-1)*k/max((i+1)*k/el,1)/60:.0f}min", flush=True); f.flush()
    f.close()
    print(f"[{split}] DONE {nid:,} tracks, {nemp} empty, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return nid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="one of: train_poolA train_poolB eval  (default: all)")
    a = ap.parse_args()
    # (split, n_configs, ions_per_config, seed, sampling_mode)
    plan = [("train_poolA", 5000, 50, 42,   "iso"),
            ("train_poolB", 1000, 50, 4242, "axis"),
            ("eval",        1000, 50, 1337, "iso")]
    meta = {}
    for name, nc, k, sd, md in plan:
        if a.only != "all" and a.only != name:
            continue
        print(f"\n=== {name}: {nc} cfg x {k} ions ({md}) seed={sd} ===", flush=True)
        meta[name] = gen(name, nc, k, sd, md)
    json.dump({"mode": "rotate_crystal_mode=True (crystal frame)", "edispl": EDISPL,
               "energy_keV": [E_LO, E_HI], "tracks": meta,
               "transform": "cloud=R^-1.cloud_lab, target=g, R=from_euler(YZ,[th,ph])"},
              open(os.path.join(OUT, "gen_meta.json"), "w"), indent=2)
    print("\nALL DONE", flush=True)
