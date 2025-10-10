"""
utils/srim_utils.py

Handles running SRIM simulations for given ion energies (thetas)
and saving the results to structured output folders.
"""



import torch
from pathlib import Path
import os
import json
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
from srim import TRIM, Ion, Layer, Target
from srim.output import Results


import shutil
from pathlib import Path

import io
import srim.output

def _patched_read_ion(self, output):
    """
    Fix for SRIM 2013 outputs where 'Ion = C' appears.
    Handles both path and in-memory text cases.
    """
    # Case 1: `output` is already file contents (string or bytes)
    if isinstance(output, (str, bytes)):
        text = output.decode() if isinstance(output, bytes) else output
        stream = io.StringIO(text)
    else:
        # Case 2: `output` is a path (Path or str)
        stream = open(output, "r")

    with stream as f:
        for line in f:
            if "Ion" in line:
                # Normalize both "=" and ":" formats
                parts = line.replace("=", " ").replace(":", " ").split()
                for token in parts:
                    if token.isalpha() and len(token) <= 2:  # e.g. C, Si, He
                        return token
    raise srim.output.SRIMOutputParseError("unable to extract ion from file")

# Apply once
srim.output.Ioniz._read_ion = _patched_read_ion
print("[patch] Applied SRIM Ioniz._read_ion() parser fix (handles text or path).")

import srim.output

import srim.output

# Patch all known SRIM output parsers that use _read_ion
for cls_name in [
    "Ioniz",
    "Vacancy",
    "Phonon", "Phonons",   # singular and plural (depends on PySRIM version)
    "E2RECOIL",
    "NOVAC",
    "EnergyToRecoils"
]:
    if hasattr(srim.output, cls_name):
        cls = getattr(srim.output, cls_name)
        cls._read_ion = _patched_read_ion

print("[patch] Applied SRIM parser fix to all SRIM output classes.")


# All SRIM output files typically produced by TRIM
KNOWN_TXT_OUTPUTS = [
    "TDATA.txt", "RANGE.txt", "VACANCY.txt", "IONIZ.txt",
    "PHONON.txt", "E2RECOIL.txt", "NOVAC.txt", "LATERAL.txt"
]
ALSO_COPY = ["TRIM.IN", "TRIMAUTO"]  # metadata and audit files


def run_srim_for_theta(theta_eV,
                       srim_directory,
                       output_base,
                       ion_symbol="C",
                       number_ions=200,
                       calculation=1,
                       layer_spec=None,
                       density_g_cm3=3.51,
                       width_A=15000.0,
                       overwrite=False):
    """
    Run SRIM at a given energy and save results to a unique folder.

    Parameters
    ----------
    theta_eV : float
        Ion energy in eV (e.g., 1e3 for 1 keV).
    srim_directory : str | Path
        Path to the SRIM-2013 directory (where TRIM.exe lives).
    output_base : str | Path
        Base directory under which a unique folder per theta will be created.
    ion_symbol : str, default="C"
        Element symbol for the ion.
    number_ions : int, default=200
        Number of ions to simulate.
    calculation : int, default=1
        TRIM calculation mode (1 = Quick KP).
    layer_spec : dict | None
        Layer specification; default is pure Carbon.
    density_g_cm3 : float, default=3.51
        Target layer density.
    width_A : float, default=15000.0
        Layer thickness in Å (1.5 μm).
    overwrite : bool, default=False
        If True, overwrite any existing folder for this theta.

    Returns
    -------
    output_dir : str
        Path to the unique folder containing this run’s outputs.
    """
    srim_dir = Path(srim_directory)
    out_base = Path(output_base)

    # Unique folder for this energy
    theta_tag = f"theta_{int(round(float(theta_eV)))}"
    out_dir = out_base / theta_tag

    # Optional overwrite
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old SRIM outputs
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        p = srim_dir / name
        if p.exists():
            p.unlink()

    # Default target layer (pure carbon)
    if layer_spec is None:
        layer_spec = {
            'C': {'stoich': 1.0, 'E_d': 30.0, 'lattice': 0.0, 'surface': 3.0}
        }

    # Build SRIM simulation components
    ion = Ion(ion_symbol, energy=float(theta_eV))
    layer = Layer(layer_spec, density=density_g_cm3, width=width_A)
    target = Target([layer])

    # Run TRIM simulation
    trim = TRIM(target, ion, number_ions=number_ions, calculation=calculation)
    trim.run(str(srim_dir))

    # Copy outputs to unique directory
    copied = []
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        src = srim_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)

    # Sanity check
    if "TDATA.txt" not in copied:
        raise RuntimeError(
            f"SRIM run at {theta_eV} eV produced no TDATA.txt in {srim_dir}. "
            f"Copied files: {copied}"
        )

    return str(out_dir)


def run_srim_batch(thetas_eV,
                   srim_directory,
                   output_base,
                   **kwargs):
    """
    Run SRIM for multiple energies (thetas) and save outputs in separate folders.

    Parameters
    ----------
    thetas_eV : list[float]
        List of ion energies in eV.
    srim_directory : str | Path
        Path to SRIM installation.
    output_base : str | Path
        Output directory where results will be stored.

    Returns
    -------
    results : list[tuple(float, str)]
        List of (theta_eV, output_folder) pairs.
    """
    results = []
    for theta in thetas_eV:
        print(f"[SRIM] Running theta = {theta:.2f} eV ...")
        folder = run_srim_for_theta(theta, srim_directory, output_base, **kwargs)
        print(f"[SRIM]   → results saved to: {folder}")
        results.append((theta, folder))
    return results


from pathlib import Path
from datetime import datetime
import json
import torch
import numpy as np

def run_srim_multi_track(
    samples_dict,
    x_test,
    track_ids,
    srim_directory,   # shared path where TRIM.exe lives
    output_base,
    ion_symbol="C",
    number_ions=200,
    calculation=1,
    density_g_cm3=3.51,
    width_A=15000.0,
    overwrite=False,
    df_summary=None,  # ✅ dataframe mapping track_id → ion, energy_keV
    n_jobs=None       # (not used)
):
    """
    Run SRIM for multiple test tracks, one at a time, using a shared SRIM installation.
    Each track corresponds to one observed x_test entry and one posterior theta set.
    Creates human-readable output folders like:
        track_<ion>_<energy>keV_<short_hash>/
    """

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if track_ids is None:
        track_ids = list(range(len(x_test)))

    if len(track_ids) != len(x_test):
        raise ValueError("track_ids must match x_test length")

    print(f"[PPC] Running SRIM serially for {len(track_ids)} tracks ...")

    results = []

    for i, (x, track_id) in enumerate(zip(x_test, track_ids)):
        track_id = str(track_id)
        print(f"\n[PPC] === Running SRIM for Track ID {track_id} (Index {i}) ===")

        # ---------------------------------------------------------------------
        # ✅ 1. Recover ion, energy, and composite_key from df_summary if available
        # ---------------------------------------------------------------------
        ion = ion_symbol
        energy_keV = None
        composite_key = None

        if df_summary is not None and "track_id" in df_summary.columns:
            match = df_summary[df_summary["track_id"] == track_id]
            if not match.empty:
                row = match.iloc[0]
                ion = row.get("ion", ion_symbol)
                energy_keV = row.get("energy_keV", row.get("energy", None))
                composite_key = row.get("composite_key", f"{ion}_unknown")

        # ---------------------------------------------------------------------
        # ✅ 2. Build descriptive, human-readable track folder name
        # ---------------------------------------------------------------------
        short_hash = track_id[:6]
        if energy_keV is not None:
            try:
                energy_int = int(float(energy_keV))
                folder_name = f"track_{ion}_{energy_int}keV_{short_hash}"
            except Exception:
                folder_name = f"track_{ion}_unknown_{short_hash}"
        else:
            folder_name = f"track_{ion}_unknown_{short_hash}"

        track_dir = output_base / folder_name
        track_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # ✅ 3. Convert posterior θ samples safely
        # ---------------------------------------------------------------------
        if track_id in samples_dict:
            theta_samples = samples_dict[track_id]
        elif track_id.isdigit() and int(track_id) in samples_dict:
            theta_samples = samples_dict[int(track_id)]
        else:
            raise KeyError(f"Track ID {track_id} not found in samples_dict")

        if isinstance(theta_samples, torch.Tensor):
            theta_samples = theta_samples.detach().cpu().numpy().flatten().tolist()
        elif isinstance(theta_samples, np.ndarray):
            theta_samples = theta_samples.flatten().tolist()
        else:
            theta_samples = list(theta_samples)

        theta_samples = sorted(set(round(float(t), 2) for t in theta_samples))

        # ---------------------------------------------------------------------
        # ✅ 4. Convert x_test row → list (float)
        # ---------------------------------------------------------------------
        if isinstance(x, torch.Tensor):
            x_values = x.detach().cpu().numpy().tolist()
        elif isinstance(x, np.ndarray):
            x_values = x.tolist()
        elif isinstance(x, (pd.Series, dict)):
            x_values = [x["mean_depth_A"], x["std_depth_A"], x["vacancies_per_ion"]]
        else:
            x_values = list(x)

        # ---------------------------------------------------------------------
        # ✅ 5. Write metadata.json
        # ---------------------------------------------------------------------
        metadata = {
            "track_index": i,
            "track_id": track_id,
            "ion": ion,
            "energy_keV": energy_keV,
            "composite_key": composite_key,
            "track_folder": str(track_dir),
            "x_test": x_values,
            "num_samples": len(theta_samples),
            "theta_samples_eV": theta_samples,
            "timestamp": datetime.now().isoformat(),
        }

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ---------------------------------------------------------------------
        # ✅ 6. Run SRIM for this track
        # ---------------------------------------------------------------------
        run_srim_batch(
            thetas_eV=theta_samples,
            srim_directory=srim_directory,
            output_base=track_dir,
            ion_symbol=ion_symbol,
            number_ions=number_ions,
            calculation=calculation,
            density_g_cm3=density_g_cm3,
            width_A=width_A,
            overwrite=overwrite,
        )

        print(f"[PPC] Track {track_id} complete → results in {track_dir}")
        results.append(str(track_dir))

    print(f"\n[PPC] ✅ All SRIM runs complete ({len(results)} tracks).")
    return results