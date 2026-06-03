"""Generate a vacancy-only Sandia-geometry dataset at a single energy.

Sandia 3rd ion-implantation settings: theta=11, phi=0, E_d=43 eV, crystalline,
single ion per simulation. Direction is identical for every track (no sampling).

Usage:
    python generate_vacancy_sandia.py --energy-keV 1.0 --n 5000 --name 1kev_train --seed 1001
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

OUT_DIR = Path(r"C:/Users/walsworthlab/Inverse ML/data/vacancy only data")
THETA_DEG = 11.0
PHI_DEG = 0.0
EDISPL_eV = 43.0


def generate(energy_keV: float, n_tracks: int, name: str, seed: int,
             progress_every: int = 500):
    """Generate `n_tracks` cascades by running ONE SIIMPL call with n_ions=n_tracks.

    Single-call batching is essential here because the SIIMPL DLL bundled with
    this install does not support seed setting; consecutive single-ion calls at
    identical (E, theta, phi) all use seed 314159 and produce the SAME track.
    Within ONE call, however, the internal RNG evolves per ion and each cascade
    is unique.
    """
    out_path  = OUT_DIR / f"vacancy_{name}.csv"
    meta_path = OUT_DIR / f"vacancy_{name}.meta.json"

    theta_rad = np.radians(THETA_DEG); phi_rad = np.radians(PHI_DEG)
    vx = float(np.sin(theta_rad) * np.cos(phi_rad))
    vy = float(np.sin(theta_rad) * np.sin(phi_rad))
    vz = float(np.cos(theta_rad))

    fields = ["x", "y", "z", "ion_number", "energy_keV", "theta_deg",
              "target_vx", "target_vy", "target_vz"]
    n_empty = 0; n_backscatter = 0; total_vacancies = 0
    t0 = time.perf_counter()

    print(f"=== vacancy_{name}: E={energy_keV} keV, theta={THETA_DEG}, "
          f"phi={PHI_DEG}, E_d={EDISPL_eV} eV, n={n_tracks} ===", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sim = Siimpl.diamond(
        ion="C", energy_keV=energy_keV,
        theta=THETA_DEG, phi=PHI_DEG,
        n_ions=n_tracks,
        crystalline=True, full_cascade=True,
        rotate_crystal_mode=True,   # Sandia/lablog convention
        Edispl_eV=EDISPL_eV,
    )
    def progress_cb(i, total):
        if (i + 1) % progress_every == 0 or i == total - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{total} ({rate:.1f} ions/s)", flush=True)
    sim.run(progress_callback=progress_cb)
    tracks = sim.tracks
    elapsed_sim = time.perf_counter() - t0

    n_done = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for t in tracks:
            if t.backscattered: n_backscatter += 1
            if t.n_vacancies == 0:
                n_empty += 1
                continue
            for (x, y, z) in t.vacancies:
                writer.writerow([
                    f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                    n_done, f"{energy_keV:.4f}", f"{THETA_DEG:.4f}",
                    f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                ])
            total_vacancies += t.n_vacancies
            n_done += 1
    sim.close()

    elapsed = time.perf_counter() - t0
    meta = {
        "energy_keV": energy_keV, "theta_deg": THETA_DEG, "phi_deg": PHI_DEG,
        "edispl_eV": EDISPL_eV,
        "n_target_tracks": n_tracks, "n_done": n_done,
        "n_empty_skipped": n_empty, "n_backscatter": n_backscatter,
        "total_vacancies": total_vacancies,
        "mean_vacancies_per_track": total_vacancies / max(n_done, 1),
        "elapsed_seconds": elapsed, "tracks_per_second": n_done / elapsed,
        "seed": seed,
        "configuration": "Sandia 3rd ion-implantation, crystalline mode (lablog)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Done {name}: {n_done} tracks, {total_vacancies:,} vacancies "
          f"in {elapsed:.1f} s. -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy-keV", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--name", required=True,
                    help="suffix for output, e.g. '1kev_train' -> vacancy_1kev_train.csv")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    generate(args.energy_keV, args.n, args.name, args.seed)


if __name__ == "__main__":
    main()
