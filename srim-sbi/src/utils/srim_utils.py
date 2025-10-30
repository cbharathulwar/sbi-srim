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
import pandas as pd
import numpy as np


import shutil
from pathlib import Path

import io
import srim.output


def _patched_read_ion(self, output):
    """
    Patch for SRIM outputs where 'Ion = X' appears in mixed formats.
    Works for both file paths and in-memory text.
    """
    if isinstance(output, (str, bytes)):
        text = output.decode() if isinstance(output, bytes) else output
        f = io.StringIO(text)
    else:
        f = open(output, "r")

    with f:
        for line in f:
            if "Ion" in line:
                parts = line.replace("=", " ").replace(":", " ").split()
                for token in parts:
                    if token.isalpha() and len(token) <= 2:
                        return token
    raise srim.output.SRIMOutputParseError("Could not extract ion.")

# Apply once
srim.output.Ioniz._read_ion = _patched_read_ion
print("[patch] Applied SRIM Ioniz._read_ion() parser fix (handles text or path).")





# Patch all known SRIM output parsers that use _read_ion
# Patch SRIM output readers to use the fixed ion parser
for name in ["Ioniz", "Vacancy", "Phonon", "Phonons", "E2RECOIL", "NOVAC", "EnergyToRecoils"]:
    if hasattr(srim.output, name):
        getattr(srim.output, name)._read_ion = _patched_read_ion
print("[patch] Applied SRIM parser fix to all SRIM output classes.")


# All SRIM output files typically produced by TRIM
KNOWN_TXT_OUTPUTS = [
    "TDATA.txt",
    "RANGE.txt",
    "VACANCY.txt",
    "IONIZ.txt",
    "PHONON.txt",
    "E2RECOIL.txt",
    "NOVAC.txt",
    "LATERAL.txt",
]
ALSO_COPY = ["TRIM.IN", "TRIMAUTO"]  # metadata and audit files


def run_srim_for_theta(
    theta_eV,
    srim_directory,
    output_base,
    ion_symbol="C",
    number_ions=200,
    calculation=1,
    layer_spec=None,
    density_g_cm3=3.51,
    width_A=15000.0,
    overwrite=False,
):
    """
    Run SRIM at a given ion energy and save results to a unique folder.

    Parameters
    ----------
    theta_eV : float
        Ion energy in eV.
    srim_directory : str | Path
        Path to SRIM installation (where TRIM.exe is located).
    output_base : str | Path
        Base directory where results are saved.
    ion_symbol : str, default="C"
        Element symbol for the ion.
    number_ions : int, default=200
        Number of ions to simulate.
    calculation : int, default=1
        TRIM calculation mode (1 = Quick KP).
    layer_spec : dict | None
        Target layer specification (defaults to pure carbon).
    density_g_cm3 : float, default=3.51
        Target layer density.
    width_A : float, default=15000.0
        Target layer thickness in Å.
    overwrite : bool, default=False
        If True, overwrite existing run folder.

    Returns
    -------
    str
        Path to the folder containing this run’s outputs.
    """
    srim_dir = Path(srim_directory)
    out_base = Path(output_base)

    theta_eV = float(theta_eV)
    theta_int = int(round(theta_eV))
    theta_tag = f"theta_{theta_int}"
    out_dir = out_base / theta_tag

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old SRIM output files
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        path = srim_dir / name
        if path.exists():
            path.unlink()

    # Default pure carbon layer
    if layer_spec is None:
        layer_spec = {"C": {"stoich": 1.0, "E_d": 30.0, "lattice": 0.0, "surface": 3.0}}

    # Allow numeric atomic numbers (e.g., 6 → C)
    ELEMENT_MAP = {"6": "C", "14": "Si", "13": "Al", "8": "O", "1": "H", "79": "Au"}
    if str(ion_symbol).isdigit():
        ion_symbol = ELEMENT_MAP.get(str(ion_symbol), "C")

    ion = Ion(ion_symbol, energy=theta_eV)
    layer = Layer(layer_spec, density=density_g_cm3, width=width_A)
    target = Target([layer])

    # Run TRIM
    trim = TRIM(target, ion, number_ions=number_ions, calculation=calculation)
    trim.run(str(srim_dir))

    # Copy SRIM outputs
    copied = []
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        src = srim_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)

    if "TDATA.txt" not in copied:
        raise RuntimeError(
            f"No TDATA.txt found after SRIM run at {theta_eV} eV. Copied: {copied}"
        )

    return str(out_dir)



def run_srim_batch(thetas_eV, srim_directory, output_base, overwrite=False, **kwargs):
    """
    Run SRIM for a list of energies and save each run in its own folder.

    Parameters
    ----------
    thetas_eV : list[float]
        Ion energies in eV.
    srim_directory : str | Path
        Path to SRIM installation.
    output_base : str | Path
        Directory where all runs are saved.
    overwrite : bool, default=False
        If True, reruns and replaces existing folders.
    kwargs :
        Extra options passed to run_srim_for_theta().

    Returns
    -------
    pd.DataFrame
        Summary of each run with columns:
        [theta_eV, output_folder, status, timestamp].
    """
    srim_dir = Path(srim_directory)
    out_base = Path(output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[INFO] Running SRIM batch for {len(thetas_eV)} energies → {out_base}")

    for theta in sorted(thetas_eV):
        try:
            folder = run_srim_for_theta(
                theta_eV=theta,
                srim_directory=srim_dir,
                output_base=out_base,
                overwrite=overwrite,
                **kwargs,
            )
            results.append((theta, str(folder), "OK", timestamp))
        except Exception as e:
            print(f"[WARN] SRIM failed at {theta:.2f} eV: {e}")
            results.append((theta, None, f"ERROR: {e}", timestamp))

    results_df = pd.DataFrame(
        results, columns=["theta_eV", "output_folder", "status", "timestamp"]
    )

    manifest_path = out_base / f"srim_batch_manifest_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results_df.to_csv(manifest_path, index=False)
    print(f"[INFO] Batch manifest saved → {manifest_path}")

    return results_df

import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def run_srim_multi_track(
    samples_dict,
    x_test,
    track_ids,
    srim_directory,
    output_base,
    ion_symbol="C",
    number_ions=200,
    calculation=1,
    density_g_cm3=3.51,
    width_A=15000.0,
    overwrite=False,
    df_summary=None,
    bin_edges_path=None,
    n_jobs=None,
):
    """
    Run SRIM for multiple test tracks, one at a time.
    Each track corresponds to one observed x_test entry and a set of posterior θ samples.

    Folder structure:
        track_<ion>_<energy>keV_<ion####>/
            metadata.json
            posterior_samples.csv
            srim_runs/theta_<val>eV/
    """
    from src.utils.srim_utils import run_srim_for_theta  # avoid circular imports

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if len(track_ids) != len(x_test):
        raise ValueError("track_ids and x_test must have the same length.")

    print(f"[INFO] Running SRIM for {len(track_ids)} tracks...")

    results = []

    for i, (x, track_id) in enumerate(zip(x_test.itertuples(index=False), track_ids)):
        track_id = str(track_id)
        print(f"[INFO] Track {i+1}/{len(track_ids)}: {track_id}")

        # --- Retrieve ion and energy info ---
        ion, energy_keV, composite_key = ion_symbol, None, None
        if df_summary is not None and "track_id" in df_summary.columns:
            row = df_summary[df_summary["track_id"] == track_id]
            if not row.empty:
                row = row.iloc[0]
                ion = str(row.get("ion", ion_symbol))
                energy_keV = row.get("energy_keV", None)
                composite_key = row.get("composite_key", f"{ion}_{energy_keV}keV")

        # Guard against numeric ion labels
        if str(ion).isdigit():
            ion = ion_symbol

        energy_int = int(round(float(energy_keV or 0)))
        suffix = re.search(r"(ion\d+)", track_id.lower())
        suffix = suffix.group(1) if suffix else re.sub(r"[^a-zA-Z0-9]+", "", track_id)[-8:]
        track_dir = Path(output_base) / f"track_{ion}_{energy_int}keV_{suffix}"
        srim_runs_dir = track_dir / "srim_runs"
        srim_runs_dir.mkdir(parents=True, exist_ok=True)

        # --- Retrieve posterior θ samples (in keV) ---
        if track_id not in samples_dict:
            raise KeyError(f"Track ID {track_id} not in samples_dict")

        theta_samples_keV = samples_dict[track_id]
        if isinstance(theta_samples_keV, torch.Tensor):
            theta_samples_keV = theta_samples_keV.cpu().numpy().flatten().tolist()
        elif isinstance(theta_samples_keV, np.ndarray):
            theta_samples_keV = theta_samples_keV.flatten().tolist()
        else:
            theta_samples_keV = list(theta_samples_keV)

        theta_samples_keV = sorted(set(round(float(t), 2) for t in theta_samples_keV))

        # --- Convert keV → eV for SRIM ---
        theta_samples_eV = [float(t) * 1_000.0 for t in theta_samples_keV]

        # Sanity checks to prevent silent unit bugs
        for val_keV, val_eV in zip(theta_samples_keV, theta_samples_eV):
            assert 1 <= val_keV <= 2000, f"Posterior θ {val_keV} keV out of prior range."
            assert 1_000 <= val_eV <= 2_000_000, f"Converted θ {val_eV} eV out of expected range."

        print(f"[DEBUG] keV→eV conversion: {theta_samples_keV[:3]} → {theta_samples_eV[:3]} ...")

        # Save posterior samples (eV) for traceability
        pd.DataFrame({"theta_eV": theta_samples_eV}).to_csv(
            track_dir / "posterior_samples.csv", index=False
        )

        # --- Prepare metadata ---
        if isinstance(x, torch.Tensor):
            x_values = x.cpu().numpy().tolist()
        elif isinstance(x, np.ndarray):
            x_values = x.tolist()
        elif isinstance(x, (pd.Series, dict)):
            x_values = list(x.values())
        else:
            x_values = list(x)

        metadata = {
            "track_index": i,
            "track_id": track_id,
            "ion": ion,
            "energy_keV": energy_int,
            "composite_key": composite_key,
            "track_folder": str(track_dir),
            "num_samples": len(theta_samples_keV),
            "theta_samples_keV": theta_samples_keV,
            "theta_samples_eV": theta_samples_eV,
            "x_test": x_values,
            "timestamp": datetime.now().isoformat(),
        }

        def _json_safe(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, (list, tuple)):
                return [_json_safe(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            return obj

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(_json_safe(metadata), f, indent=2)

        # --- Run SRIM for each θ sample (in eV) ---
        for theta_eV in theta_samples_eV:
            try:
                run_srim_for_theta(
                    theta_eV=theta_eV,
                    srim_directory=srim_directory,
                    output_base=srim_runs_dir,
                    ion_symbol=ion,
                    number_ions=number_ions,
                    calculation=calculation,
                    density_g_cm3=density_g_cm3,
                    width_A=width_A,
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[WARN] SRIM failed for {track_id} @ {theta_eV} eV: {e}")

        results.append(str(track_dir))
        print(f"[INFO] Completed {track_id}")

    print(f"[INFO] SRIM finished for all {len(results)} tracks.")
    return results