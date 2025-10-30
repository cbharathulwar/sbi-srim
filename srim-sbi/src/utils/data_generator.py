# generate_data.py
import os
from srim import TRIM, Ion, Layer, Target
import shutil
import time

import os, shutil
from srim import TRIM, Ion, Layer, Target

def run_srim_for_energy(energy_keV, srim_dir, output_root, number_ions=200):
    ion = Ion("C", energy=energy_keV * 1e3)
    layer = Layer({"C": 1.0}, density=3.51, width=2e4)
    target = Target([layer])

    # Ensure SRIM Outputs exists (prevents error 76)
    outputs_dir = os.path.join(srim_dir, "SRIM Outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Optional: start each run clean
    for name in os.listdir(outputs_dir):
        p = os.path.join(outputs_dir, name)
        if os.path.isfile(p):
            os.remove(p)
        else:
            shutil.rmtree(p)

    # Always run from SRIM's directory so VB relative paths work
    cwd = os.getcwd()
    os.chdir(srim_dir)
    try:
        trim = TRIM(target=target, ion=ion, calculation=2, number_ions=number_ions)
        trim.run(srim_directory=srim_dir)
    finally:
        os.chdir(cwd)

        # --- Wait for SRIM output file (handle both COLLISON(S).txt variants)
    possible_names = ["COLLISIONS.txt", "COLLISON.txt"]
    collisions_path = None

    # Wait up to 10 seconds for SRIM to finish writing
    for _ in range(20):
        for name in possible_names:
            candidate = os.path.join(outputs_dir, name)
            if os.path.exists(candidate):
                collisions_path = candidate
                break
        if collisions_path:
            break
        time.sleep(0.5)

    if not collisions_path:
        raise FileNotFoundError(f"No COLLISIONS/COLLISON file found in {outputs_dir}")

    # Normalize filename to COLLISIONS.txt (so downstream parsing is consistent)
    if collisions_path.endswith("COLLISON.txt"):
        fixed_path = os.path.join(outputs_dir, "COLLISIONS.txt")
        os.rename(collisions_path, fixed_path)
        collisions_path = fixed_path

    # --- Copy to organized results folder
    run_dir = os.path.join(srim_dir, "results", f"{energy_keV:.1f}keV")
    os.makedirs(run_dir, exist_ok=True)
    dest = os.path.join(run_dir, "COLLISIONS.txt")

    shutil.copy(collisions_path, dest)
    print(f" Copied SRIM results to {dest}")

    return dest

# generate_data.py  (continued)
import pandas as pd
import re

def parse_collisions(collisions_path: str, energy_keV: float):
    """
    Parse SRIM COLLISIONS.txt file to extract vacancy (x, y, z) coordinates.

    Parameters
    ----------
    collisions_path : str
        Path to the COLLISIONS.txt file produced by SRIM.
    energy_keV : float
        The incident energy (keV) used for this run.

    Returns
    -------
    df : pd.DataFrame
        Columns: [x (ang), y (ang), z (ang), ion #, energy]
    """
    rows = []
    ion_id = -1  # increment whenever a new ion is detected

    # Pattern:   Recoil #, Z, E_recoil, X, Y, Z, Vac, Repl
    line_pattern = re.compile(
        r"^\s*\d+\s+\d+\s+[0-9Ee+.-]+\s+([0-9Ee+.-]+)\s+([0-9Ee+.-]+)\s+([0-9Ee+.-]+)\s+(\d)\s+(\d)"
    )

    with open(collisions_path, "r", errors="ignore") as f:
        for line in f:
            # Detect start of new ion cascade (often "Ion #" or "CASCADE")
            if "Ion" in line and "Recoil" in line or "CASCADE" in line.upper():
                ion_id += 1
                continue

            m = line_pattern.match(line)
            if not m:
                continue

            x, y, z, vac, repl = map(float, m.groups())
            if int(vac) == 1:  # keep only vacancies
                rows.append({
                    "x (ang)": x,
                    "y (ang)": y,
                    "z (ang)": z,
                    "ion #": ion_id,
                    "energy": energy_keV
                })

    df = pd.DataFrame(rows)
    return df


# generate_data.py
import pandas as pd

def run_and_parse_energy(energy_keV: float, srim_dir: str, output_root: str, number_ions: int = 200):
    """
    Runs a full SRIM simulation for a given energy and parses the results.

    Parameters
    ----------
    energy_keV : float
        Incident ion energy in keV (e.g., 50.0)
    srim_dir : str
        Path to SRIM installation directory.
    output_root : str
        Directory where SRIM output folders will be stored.
    number_ions : int
        Number of ions to simulate per energy.

    Returns
    -------
    df : pd.DataFrame
        Parsed results DataFrame with columns:
        [x (ang), y (ang), z (ang), ion #, energy]
    """
    print(f"\n Starting SRIM run for {energy_keV:.1f} keV")

    # 1. Run SRIM simulation
    from src.utils.data_generator import run_srim_for_energy, parse_collisions
    collisions_path = run_srim_for_energy(
        energy_keV=energy_keV,
        srim_dir=srim_dir,
        output_root=output_root,
        number_ions=number_ions
    )

    # 2. Parse the COLLISIONS.txt
    df = parse_collisions(collisions_path, energy_keV)

    # 3. Save per-energy CSV
    save_path = os.path.join(output_root, f"{energy_keV:.1f}_with_energy.csv")
    df.to_csv(save_path, index=False)
    print(f" Saved parsed data for {energy_keV:.1f} keV → {save_path}")

    return df