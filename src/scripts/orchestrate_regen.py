"""One-shot regeneration of the SIIMPL v2.1 dataset with the 0-degree (unrotated)
crystal and an EXPANDED Pool B channeling fill.

Plan (Option B): Pool A unchanged (5,000 cfg x 50 ions train; 1,000 x 50 eval),
Pool B per-bin target raised 3000 -> 6000 for train (deficit logic auto-targets
the thin high-energy channeling bins). Eval Pool B target stays 600.

Runs EVAL first and validates the 0-degree geometry before committing the long
train generation. Backs up the existing 35-degree data first; clears stale EGNN
caches at the end so training rebuilds the v3 (physics-feature) cache.

Run from repo root:
    set SIIMPL_DIR=C:/siimpl/python
    python -m src.scripts.orchestrate_regen
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "src" / "scripts"
DATA = REPO / "data" / "siimpl"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))

# Make sure the generation modules find the local SIIMPL install on import.
os.environ.setdefault("SIIMPL_DIR", r"C:/siimpl/python")
os.environ.setdefault("SIIMPL_DATA_DIR", str(DATA))

import generate_pool_A_v2 as gpa          # noqa: E402
import generate_pool_B_v2 as gpb          # noqa: E402
import measure_pool_A_v2_coverage as mc   # noqa: E402
import finalize_siimpl_v2 as fin          # noqa: E402
import channeling_axes as cax             # noqa: E402

fin.DATA = DATA  # ensure finalize writes to the local repo data dir

EDISPL = 43.0

# Pool A (unchanged plan, now 0-degree crystal)
A_TRAIN = dict(n_configs=5000, n_ions=50, seed=42)
A_EVAL  = dict(n_configs=1000, n_ions=50, seed=1337)

# Pool B (EXPANDED for train; deficit logic concentrates fill in thin hi-E bins).
# target 4500 (was 3000) -> ~2x Pool B size in the thin high-E channeling bins,
# keeping Pool A+B total under the training memory cap. Eval target stays 600.
B_TRAIN = dict(target=4500, ions=25, seed=4242, start=300_000)
B_EVAL  = dict(target=600,  ions=25, seed=13337, start=100_000)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def backup_old():
    bdir = DATA / "_backup_35deg"
    bdir.mkdir(exist_ok=True)
    patterns = [
        "siimpl_train.csv", "siimpl_eval.csv",
        "siimpl_train.summary.json", "siimpl_eval.summary.json",
        "siimpl_train_poolA_v2.csv", "siimpl_train_poolB_v2.csv",
        "siimpl_eval_poolA_v2.csv", "siimpl_eval_poolB_v2.csv",
        "siimpl_train_poolA_v2.meta.json", "siimpl_train_poolB_v2.meta.json",
        "siimpl_eval_poolA_v2.meta.json", "siimpl_eval_poolB_v2.meta.json",
        "pool_A_v2_train_coverage.json", "pool_A_v2_eval_coverage.json",
    ]
    moved = 0
    for name in patterns:
        p = DATA / name
        if p.exists():
            shutil.move(str(p), str(bdir / name))
            moved += 1
    # EGNN preprocessing caches (stale; v3 will rebuild from new data)
    for c in DATA.glob("*egnn_cache.pt"):
        shutil.move(str(c), str(bdir / c.name))
        moved += 1
    log(f"Backed up {moved} existing files -> {bdir}")


# ---------------------------------------------------------------------------
def gen_split(split, A, B):
    pa_csv  = DATA / f"siimpl_{split}_poolA_v2.csv"
    pa_meta = DATA / f"siimpl_{split}_poolA_v2.meta.json"
    cov     = DATA / f"pool_A_v2_{split}_coverage.json"
    pb_csv  = DATA / f"siimpl_{split}_poolB_v2.csv"
    pb_meta = DATA / f"siimpl_{split}_poolB_v2.meta.json"

    log(f"=== {split.upper()} Pool A: {A['n_configs']} cfg x {A['n_ions']} ions ===")
    gpa.generate(n_configs=A["n_configs"], n_ions_per_config=A["n_ions"],
                 seed=A["seed"], out_path=pa_csv, meta_path=pa_meta,
                 edispl_eV=EDISPL)

    log(f"=== {split.upper()} coverage (target {B['target']}/bin) ===")
    mc.measure(pa_csv, cov, B["target"], f"POOL A {split.upper()}")

    log(f"=== {split.upper()} Pool B (target {B['target']}, start ion {B['start']}) ===")
    gpb.generate(coverage_json=cov, out_path=pb_csv, meta_path=pb_meta,
                 seed=B["seed"], ion_number_start=B["start"],
                 n_ions_per_config=B["ions"], n_target_override=B["target"],
                 edispl_eV=EDISPL)

    log(f"=== {split.upper()} finalize/merge ===")
    fin.process(split, B["target"])


# ---------------------------------------------------------------------------
def validate_eval():
    """Sanity-check the regenerated eval set reflects the 0-degree geometry."""
    log("Validating regenerated EVAL geometry...")
    assert abs(cax.DIAMOND_PREROT_DEG) < 1e-9, \
        f"channeling_axes pre-rotation is {cax.DIAMOND_PREROT_DEG}, expected 0"

    # <100> lab axes should be canonical (contain a ~[0,0,1] vector at 0 deg)
    a100 = cax.AXES_LAB["<100>"]
    has_z = np.any(np.all(np.abs(a100 - np.array([0, 0, 1.0])) < 1e-6, axis=1))
    assert has_z, f"<100> lab axes not canonical at 0 deg: {a100.tolist()}"

    df = pd.read_csv(DATA / "siimpl_eval.csv",
                     usecols=["ion_number", "energy_keV",
                              "target_vx", "target_vy", "target_vz"])
    tr = df.drop_duplicates("ion_number").reset_index(drop=True)
    v = tr[["target_vx", "target_vy", "target_vz"]].to_numpy()

    # Upper-hemisphere generation: all target_vz >= 0
    assert v[:, 2].min() > -1e-6, f"found vz < 0 (min {v[:,2].min():.3f})"

    # Channeling tracks should exist near a literal <100> axis
    cosmax = np.clip(np.max(np.abs(v @ a100.T), axis=1), -1, 1)
    ang = np.degrees(np.arccos(cosmax))
    n_near = int((ang < 5.0).sum())

    nvac = df.groupby("ion_number").size()
    log(f"  eval tracks={len(tr):,}  near<100>(<5deg)={n_near:,}  "
        f"median_vac={nvac.median():.0f}  max_vac={nvac.max()}")
    assert n_near > 0, "no tracks near <100> axis — channeling/geometry suspect"
    assert nvac.median() > 5, f"median vac/track too low ({nvac.median()})"
    log("  EVAL geometry OK.")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log(f"REPO={REPO}  DATA={DATA}")
    log(f"SIIMPL_DIR={os.environ.get('SIIMPL_DIR')}  prerot={cax.DIAMOND_PREROT_DEG}")

    backup_old()

    # --- EVAL first (fast), then validate before the long train run ---
    gen_split("eval", A_EVAL, B_EVAL)
    validate_eval()

    # --- TRAIN (long) ---
    gen_split("train", A_TRAIN, B_TRAIN)

    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min.")
    log("Canonical files:")
    for f in ("siimpl_train.csv", "siimpl_eval.csv"):
        p = DATA / f
        log(f"  {p}  ({p.stat().st_size/1e6:.0f} MB)" if p.exists() else f"  MISSING {p}")
    log("EGNN caches cleared; training will rebuild the v3 (physics) cache.")


if __name__ == "__main__":
    main()
