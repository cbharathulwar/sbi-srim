# generate_data.py
import os
import shutil
import time
from srim import TRIM, Ion, Layer, Target
from srim.output import Collision
def run_srim_for_energy(energy_keV, srim_dir, output_root, number_ions=200):
    """
    Runs SRIM for a given energy, saves output into a folder:
        <output_root>/<energy>keV/
    Always ensures COLLISON.txt (PySRIM-expected filename) exists.
    """
    print(f"[INFO] Running SRIM for {energy_keV:.1f} keV")
    # Create output directory for this energy
    run_dir = os.path.join(output_root, f"{energy_keV:.1f}keV")
    os.makedirs(run_dir, exist_ok=True)
    # SRIM output folder
    outputs_dir = os.path.join(srim_dir, "SRIM Outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    # Clean SRIM Outputs folder
    for f in os.listdir(outputs_dir):
        fp = os.path.join(outputs_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
        else:
            shutil.rmtree(fp, ignore_errors=True)
    # Configure ion + target
    ion = Ion("C", energy=energy_keV * 1e3)  # convert keV → eV
    layer = Layer({"C": 1.0}, density=3.51, width=20000.0)
    target = Target([layer])
    # Run in SRIM directory
    cwd = os.getcwd()
    os.chdir(srim_dir)
    try:
        trim = TRIM(target=target,
                    ion=ion,
                    number_ions=number_ions,
                    calculation=2,
                    collisions=2)  # IMPORTANT: dump collision cascades
        trim.run(srim_directory=srim_dir)
    finally:
        os.chdir(cwd)
    # Wait for SRIM to produce collision file
    collisions_file = None
    for _ in range(20):  # up to 10 seconds
        for name in ["COLLISIONS.txt", "COLLISON.txt"]:
            candidate = os.path.join(outputs_dir, name)
            if os.path.exists(candidate):
                collisions_file = candidate
                break
        if collisions_file:
            break
        time.sleep(0.5)
    if not collisions_file:
        raise FileNotFoundError("No COLLISIONS.txt or COLLISON.txt found from SRIM!")
    # :white_check_mark: Normalize filename → PySRIM expects *exactly* COLLISON.txt (no 'S')
    correct_file = os.path.join(outputs_dir, "COLLISON.txt")
    if collisions_file.endswith("COLLISIONS.txt"):
        os.rename(collisions_file, correct_file)
    else:
        correct_file = collisions_file
    # Copy all generated SRIM output files to this energy folder
    for f in os.listdir(outputs_dir):
        shutil.copy(os.path.join(outputs_dir, f), run_dir)
    print(f"[SAVED] SRIM outputs stored → {run_dir}")
    return run_dir









# generate_data.py  (continued)
import pandas as pd
import re

from srim.output import Collision
import numpy as np
import pandas as pd
def parse_collisions(folder_path: str, energy_keV: float):
    """
    Parse SRIM COLLISON.txt into a flat event table expected by preprocess():
      columns: x, y, z, ion_number, energy_keV
    """
    coll = Collision(folder_path)
    rows = []

    for ion_idx in range(len(coll)):
        ion_data = coll[ion_idx]
        collisions = ion_data.get('collisions', None)
        if collisions is None:
            continue

        for primary in collisions:
            for recoil in primary.get('cascade', []):
                pos = recoil.get('position', None)
                if pos is None:
                    continue
                rows.append({
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "ion_number": ion_idx,
                    "energy_keV": energy_keV
                })

    df = pd.DataFrame(rows, columns=["x", "y", "z", "ion_number", "energy_keV"])

    # Enforce numeric dtypes and drop bad rows
    for c in ["x", "y", "z", "energy_keV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ion_number"] = pd.to_numeric(df["ion_number"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV"]).reset_index(drop=True)
    df["ion_number"] = df["ion_number"].astype(int)

    return df


























# generate_data.py
import pandas as pd

def run_and_parse_energy(energy_keV, srim_dir, output_root, number_ions=200):
    print(f"\nStarting SRIM run for {energy_keV:.1f} keV")
    # Run SRIM and get back folder path like .../output/5.0keV/
    folder_path = run_srim_for_energy(
        energy_keV=energy_keV,
        srim_dir=srim_dir,
        output_root=output_root,
        number_ions=number_ions
    )
    # Parse collisions using Max-style logic
    df = parse_collisions(folder_path, energy_keV)
    # :white_check_mark: Save *inside* the energy folder now
    save_csv = os.path.join(folder_path, f"{energy_keV:.1f}_with_energy.csv")
    df.to_csv(save_csv, index=False)
    print(f":white_check_mark: Saved parsed data for {energy_keV:.1f} keV → {save_csv}")
    return df











