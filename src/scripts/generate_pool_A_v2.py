"""Pool A v2 — batched generation with guaranteed per-track cascade uniqueness.

Each SIIMPL call uses n_ions=K so the internal RNG evolves per ion (the DLL
seed-lock cannot be circumvented across calls, but WITHIN one call cascades
are genuinely distinct). We sample N_CONFIGS unique (E, theta, phi)
configurations and run K ions at each.

Coverage:
    N_CONFIGS  uniform-on-upper-hemisphere directions x log-uniform[1,105] keV
    K          replications per config (within-call RNG variation)
    Total      N_CONFIGS * K = 250,000 tracks (10,000 configs * 25 ions)

Schema, generation parameters, and theta/phi convention identical to Pool A v1.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SIIMPL_DIR = Path(r"C:/Users/walsworthlab/SIIMPL/siimpl/python")
sys.path.insert(0, str(SIIMPL_DIR))

import warnings
warnings.filterwarnings("ignore")
from siimpl import Siimpl


def generate(
    n_configs: int,
    n_ions_per_config: int,
    seed: int,
    out_path: Path,
    meta_path: Path,
    energy_range_keV: tuple = (1.0, 105.0),
    edispl_eV: float = 43.0,
    progress_every_configs: int = 50,
):
    rng = np.random.default_rng(seed)
    ln_lo, ln_hi = np.log(energy_range_keV[0]), np.log(energy_range_keV[1])

    # Pre-sample all configs
    energies = np.exp(rng.uniform(ln_lo, ln_hi, n_configs))
    g = rng.standard_normal((n_configs, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g[:, 2] = np.abs(g[:, 2])  # upper hemisphere
    thetas = np.degrees(np.arccos(g[:, 2]))
    phis   = np.degrees(np.arctan2(g[:, 1], g[:, 0]))

    fields = ["x", "y", "z", "ion_number", "energy_keV", "theta_deg",
              "target_vx", "target_vy", "target_vz"]
    n_written = 0
    n_empty = 0
    n_backscatter = 0
    total_vacancies = 0
    t0 = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        for i in range(n_configs):
            E = float(energies[i])
            theta_deg = float(thetas[i])
            phi_deg = float(phis[i])
            vx, vy, vz = float(g[i, 0]), float(g[i, 1]), float(g[i, 2])

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
                for track in sim.tracks:
                    if track.backscattered:
                        n_backscatter += 1
                    if track.n_vacancies == 0:
                        n_empty += 1
                        continue
                    for (x, y, z) in track.vacancies:
                        writer.writerow([
                            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                            n_written, f"{E:.4f}", f"{theta_deg:.4f}",
                            f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                        ])
                    total_vacancies += track.n_vacancies
                    n_written += 1
                sim.close()
            except Exception as e:
                print(f"  [warn] sim failed at config {i}: {e}", flush=True)
                continue

            if (i + 1) % progress_every_configs == 0:
                elapsed = time.perf_counter() - t0
                rate_cfg = (i + 1) / elapsed
                rate_ion = n_written / elapsed
                pct = (i + 1) / n_configs * 100
                eta = (n_configs - i - 1) / rate_cfg
                print(f"  cfg {i+1}/{n_configs} ({pct:.1f}%)  "
                      f"tracks={n_written:,}  rate={rate_ion:.1f} tr/s  "
                      f"ETA {eta/60:.1f} min  empty={n_empty}", flush=True)

    elapsed = time.perf_counter() - t0
    meta = {
        "n_configs": n_configs,
        "n_ions_per_config": n_ions_per_config,
        "n_target": n_configs * n_ions_per_config,
        "n_done": n_written,
        "n_empty_skipped": n_empty,
        "n_backscatter": n_backscatter,
        "total_vacancies": total_vacancies,
        "mean_vacancies_per_track": total_vacancies / max(n_written, 1),
        "elapsed_seconds": elapsed,
        "tracks_per_second": n_written / elapsed,
        "edispl_eV": edispl_eV,
        "energy_range_keV": list(energy_range_keV),
        "seed": seed,
        "method": "batched n_ions per call for cascade uniqueness; "
                  "upper-hemisphere sampling + Oh aug at training",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. {n_written:,} tracks, {total_vacancies:,} vacancies "
          f"in {elapsed/60:.1f} min.  Meta: {meta_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--n-configs", type=int, default=None)
    ap.add_argument("--n-ions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(r"C:/Users/walsworthlab/Inverse ML/data/siimpl"))
    args = ap.parse_args()

    # Defaults: train = 5000 cfg x 50 ions = 250k tracks
    #           eval  = 1000 cfg x 50 ions =  50k tracks
    # Matches v1 scale. Pool B fills channeling deficits separately.
    defaults = {"train": (5_000, 50, 42), "eval": (1_000, 50, 1337)}
    nc_default, ni_default, seed_default = defaults[args.split]
    n_configs = args.n_configs or nc_default
    n_ions    = args.n_ions    or ni_default
    seed      = args.seed if args.seed is not None else seed_default

    out_path  = args.out_dir / f"siimpl_{args.split}_poolA_v2.csv"
    meta_path = args.out_dir / f"siimpl_{args.split}_poolA_v2.meta.json"

    print(f"=== Pool A v2 {args.split} ===")
    print(f"  configs    : {n_configs:,}")
    print(f"  ions/config: {n_ions}")
    print(f"  total      : {n_configs * n_ions:,}")
    print(f"  seed       : {seed}")
    print(f"  output     : {out_path}")
    generate(n_configs=n_configs, n_ions_per_config=n_ions,
             seed=seed, out_path=out_path, meta_path=meta_path)


if __name__ == "__main__":
    main()
