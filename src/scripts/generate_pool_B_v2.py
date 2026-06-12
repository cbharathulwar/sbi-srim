"""Pool B v2 — deficit-driven channeling fill, batched and rotated-axis aware.

Two fixes vs v1:
  1. Uses batched n_ions per call to guarantee per-cascade uniqueness.
  2. Targets the ROTATED lab-frame channeling axes (literal <hkl> rotated by
     R_y(35 deg) — see channeling_axes.py) instead of literal <hkl>.

Reads pool_A_v2_coverage.json (per-bin deficit table). For each (axis, energy
bin) with deficit > 0, samples beam directions within 2*psi_c cones of the
appropriate rotated lab-frame axes, distributing energies uniformly within the
bin. Each Pool B "config" gets K ions (batched) for cascade uniqueness.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import os
# Resolve the SIIMPL python package across machines (see generate_pool_A_v2).
_SIIMPL_CANDIDATES = [os.environ.get("SIIMPL_DIR"), r"C:/siimpl/python",
                      r"C:/Users/walsworthlab/SIIMPL/siimpl/python"]
SIIMPL_DIR = next((Path(p) for p in _SIIMPL_CANDIDATES if p and Path(p).exists()),
                  Path(r"C:/siimpl/python"))
sys.path.insert(0, str(SIIMPL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for channeling_axes

import warnings
warnings.filterwarnings("ignore")
from siimpl import Siimpl
from channeling_axes import (
    AXES_LAB, D, psi_c_deg, angle_to_nearest_axis, sample_in_cone,
)


def sample_one_target(family: str, e_lo: float, e_hi: float,
                      rng: np.random.Generator):
    """Pick (E, beam direction) for one Pool B 'config' targeting this family/bin."""
    E = float(rng.uniform(e_lo, e_hi))
    axs = AXES_LAB[family]
    ax = axs[rng.integers(len(axs))]
    psi_deg = psi_c_deg(E, family)
    v = sample_in_cone(ax, np.radians(2 * psi_deg), rng)
    if v[2] < 0:
        v = -v
    # convert to (theta, phi) for SIIMPL
    theta_deg = float(np.degrees(np.arccos(np.clip(v[2], -1, 1))))
    phi_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    return E, v, theta_deg, phi_deg


def generate(
    coverage_json: Path,
    out_path: Path,
    meta_path: Path,
    seed: int,
    ion_number_start: int,
    n_ions_per_config: int,
    n_target_override: int = None,
    edispl_eV: float = 43.0,
):
    cov = json.loads(coverage_json.read_text())
    rng = np.random.default_rng(seed)

    # Build deficit-bin list
    bins = []
    for row in cov["table"]:
        target = n_target_override if n_target_override is not None else row["n_target"]
        deficit = max(0, target - row["n_within_psi_c"])
        if deficit > 0:
            bins.append({
                "axis": row["axis"], "e_lo": row["e_lo"], "e_hi": row["e_hi"],
                "deficit": deficit, "filled": 0,
            })
    if not bins:
        print("No deficits to fill. Nothing to do.", flush=True)
        out_path.write_text("x,y,z,ion_number,energy_keV,theta_deg,target_vx,target_vy,target_vz\n")
        meta_path.write_text(json.dumps({"deficit_total": 0, "n_done": 0}, indent=2))
        return

    total_deficit = sum(b["deficit"] for b in bins)
    print(f"Filling {total_deficit:,} channeled-track deficit across "
          f"{len(bins)} bins (batched, rotated axes)", flush=True)
    for b in bins:
        print(f"  {b['axis']:6s} E[{b['e_lo']:.2f},{b['e_hi']:.2f}) keV  "
              f"need {b['deficit']:,}", flush=True)

    fields = ["x", "y", "z", "ion_number", "energy_keV", "theta_deg",
              "target_vx", "target_vy", "target_vz"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_empty = 0
    n_backscatter = 0
    n_within_psi_c = 0
    total_vacancies = 0
    ion_id = ion_number_start
    t0 = time.perf_counter()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        while bins:
            weights = np.array([max(0, b["deficit"] - b["filled"]) for b in bins], float)
            if weights.sum() <= 0:
                break
            weights /= weights.sum()
            bidx = int(rng.choice(len(bins), p=weights))
            b = bins[bidx]

            E, v, theta_deg, phi_deg = sample_one_target(
                b["axis"], b["e_lo"], b["e_hi"], rng,
            )
            vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

            try:
                sim = Siimpl.diamond(
                    ion="C", energy_keV=E,
                    theta=theta_deg, phi=phi_deg,
                    n_ions=n_ions_per_config,
                    crystalline=True, full_cascade=True,
                    rotate_crystal_mode=False,
                    Edispl_eV=edispl_eV,
                )
                sim.run()

                psi = psi_c_deg(E, b["axis"])
                for track in sim.tracks:
                    if track.backscattered: n_backscatter += 1
                    if track.n_vacancies == 0:
                        n_empty += 1
                        continue
                    for (x, y, z) in track.vacancies:
                        writer.writerow([
                            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                            ion_id, f"{E:.4f}", f"{theta_deg:.4f}",
                            f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                        ])
                    total_vacancies += track.n_vacancies
                    n_done += 1
                    ion_id += 1

                # All ions in this batch share the same beam direction, so they
                # contribute identically to the channeled-count for this bin.
                angle = angle_to_nearest_axis(np.array([vx, vy, vz]), b["axis"])
                if angle <= psi:
                    contrib = sum(1 for t in sim.tracks if t.n_vacancies > 0)
                    b["filled"] += contrib
                    n_within_psi_c += contrib

                sim.close()
            except Exception as e:
                print(f"  [warn] sim failed: {e}", flush=True)
                continue

            # Remove finished bins
            bins = [bb for bb in bins if bb["filled"] < bb["deficit"]]

            # Progress
            if n_done % 1000 < n_ions_per_config:
                elapsed = time.perf_counter() - t0
                remaining = sum(max(0, bb["deficit"] - bb["filled"]) for bb in bins)
                print(f"  raw={n_done:,}  within_psi_c={n_within_psi_c:,}  "
                      f"remaining={remaining:,}  ({n_done/max(elapsed,0.1):.1f} tr/s)",
                      flush=True)

    elapsed = time.perf_counter() - t0
    meta = {
        "n_raw_tracks": n_done,
        "n_within_psi_c": n_within_psi_c,
        "n_empty_skipped": n_empty,
        "n_backscatter": n_backscatter,
        "total_vacancies": total_vacancies,
        "mean_vacancies_per_track": total_vacancies / max(n_done, 1),
        "elapsed_seconds": elapsed,
        "tracks_per_second": n_done / max(elapsed, 0.1),
        "edispl_eV": edispl_eV,
        "seed": seed,
        "n_ions_per_config": n_ions_per_config,
        "ion_number_start": ion_number_start,
        "ion_number_end": ion_id - 1,
        "method": "Pool B v2: batched n_ions, rotated lab-frame axes",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. raw={n_done:,}  within_psi_c={n_within_psi_c:,}  "
          f"in {elapsed/60:.2f} min.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--coverage-json", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ion-number-start", type=int, required=True)
    ap.add_argument("--n-ions-per-config", type=int, default=25)
    ap.add_argument("--n-target-override", type=int, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(r"C:/Users/walsworthlab/Inverse ML/data/siimpl"))
    args = ap.parse_args()

    seed_defaults = {"train": 4242, "eval": 13337}
    seed = args.seed if args.seed is not None else seed_defaults[args.split]

    out_path  = args.out_dir / f"siimpl_{args.split}_poolB_v2.csv"
    meta_path = args.out_dir / f"siimpl_{args.split}_poolB_v2.meta.json"

    print(f"=== Pool B v2 {args.split} ===")
    print(f"  coverage:  {args.coverage_json}")
    print(f"  output:    {out_path}")
    print(f"  n_ions/cfg:{args.n_ions_per_config}")
    generate(
        coverage_json=args.coverage_json, out_path=out_path,
        meta_path=meta_path, seed=seed,
        ion_number_start=args.ion_number_start,
        n_ions_per_config=args.n_ions_per_config,
        n_target_override=args.n_target_override,
    )


if __name__ == "__main__":
    main()
