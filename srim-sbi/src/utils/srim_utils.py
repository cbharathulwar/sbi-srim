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

    # --- Compute once ---
    theta_eV = float(theta_eV)
    theta_int = int(round(theta_eV))
    theta_tag = f"theta_{theta_int}"
    out_dir = out_base / theta_tag

    # --- DEBUG: trace SRIM call ---
    print(
        f"[DEBUG run_srim_for_theta] ion={ion_symbol} | "
        f"θ_eV={theta_eV} (rounded={theta_int}) | out_dir={out_dir}"
    )

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

    # --- ensure element symbol is string ---
    if not isinstance(ion_symbol, str):
      ion_symbol = str(ion_symbol)


# --- DEBUG CHECKPOINT ---------------------------------------------
    print(f"[DEBUG: run_srim_for_theta] Called with:")
    print(f"   ion_symbol = {ion_symbol}")
    print(f"   theta_eV   = {theta_eV} (rounded → {int(round(float(theta_eV)))})")
    print(f"   out_dir    = {out_dir}")
    # sanity check folder name
    if '.0keV' in str(out_dir):
        print(f"⚠️  [WARN] Non-integer folder name detected → {out_dir}")
    # ---------------------------------------------------------------
    # Safety: handle numeric element codes accidentally passed in
    if ion_symbol.isdigit():
        # basic mapping fallback if someone passes atomic number
        ELEMENT_MAP = {"6": "C", "14": "Si", "13": "Al", "8": "O", "1": "H", "79": "Au" }
    ion_symbol = ELEMENT_MAP.get(ion_symbol, "C")
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


from pathlib import Path
import pandas as pd
from datetime import datetime

def run_srim_batch(
    thetas_eV,
    srim_directory,
    output_base,
    overwrite=False,
    **kwargs
):
    """
    Run SRIM for multiple energies (thetas) and save outputs in separate folders.

    Parameters
    ----------
    thetas_eV : list[float]
        List of ion energies in eV.
    srim_directory : str | Path
        Path to SRIM installation directory (where TRIM.exe lives).
    output_base : str | Path
        Output directory where per-theta results will be stored.
    overwrite : bool, default=False
        If True, clears existing subfolders before rerunning.
    kwargs :
        Extra args passed to run_srim_for_theta().

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns [theta_eV, output_folder, status, timestamp].
    """

    srim_dir = Path(srim_directory)
    out_base = Path(output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[SRIM-BATCH] Starting batch of {len(thetas_eV)} energies → {out_base}")

    for theta in sorted(thetas_eV):
        try:
            print(f"[SRIM] Running θ = {theta:.2f} eV ...")
            folder = run_srim_for_theta(
                theta_eV=theta,
                srim_directory=srim_dir,
                output_base=out_base,
                overwrite=overwrite,
                **kwargs
            )
            print(f"[SRIM] ✅ Saved outputs → {folder}")
            results.append((theta, str(folder), "OK", timestamp))
        except Exception as e:
            print(f"[SRIM] ❌ Failed for θ={theta:.2f} eV → {e}")
            results.append((theta, None, f"ERROR: {e}", timestamp))
            continue

    results_df = pd.DataFrame(results, columns=["theta_eV", "output_folder", "status", "timestamp"])

    # optional — save manifest to disk for traceability
    manifest_path = out_base / f"srim_batch_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(manifest_path, index=False)
    print(f"[SRIM-BATCH] Manifest saved → {manifest_path}")

    return results_df

def run_srim_multi_track(
    samples_dict,
    x_test,
    track_ids,
    srim_directory,      # shared path where TRIM.exe lives
    output_base,
    ion_symbol="C",
    number_ions=200,
    calculation=1,
    density_g_cm3=3.51,
    width_A=15000.0,
    overwrite=False,
    df_summary=None,     # dataframe mapping track_id → ion, energy_keV
    n_jobs=None          # (not used)
):
    """
    Run SRIM for multiple test tracks, one at a time, using a shared SRIM installation.
    Each track corresponds to one observed x_test entry and a posterior θ sample set.
    Creates a clean structure like:
        track_<ion>_<energy>keV_<id>/
            metadata.json
            srim_runs/theta_<val>eV/
    """

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    

    if len(track_ids) != len(x_test):
        raise ValueError("track_ids must match x_test length")

    print(f"[PPC] Running SRIM serially for {len(track_ids)} tracks ...")

    results = []

    for i, (x, track_id) in enumerate(zip(x_test, track_ids)):
        track_id = str(track_id)
        print(f"\n[PPC] === Running SRIM for Track ID {track_id} (Index {i}) ===")

        # ---------------------------------------------------------------
        # 1️⃣ Retrieve ion and energy info from df_summary
        # ---------------------------------------------------------------
        ion, energy_keV, composite_key = ion_symbol, None, None

        if df_summary is not None and "track_id" in df_summary.columns:
            row = df_summary[df_summary["track_id"] == track_id]
            if not row.empty:
                row = row.iloc[0]
                ion = row.get("ion", ion_symbol)
                energy_keV = row.get("energy_keV", None)
                composite_key = row.get("composite_key", f"{ion}_{energy_keV}keV")

        # ---------------------------------------------------------------
        # 2️⃣ Track folder naming — readable & deterministic
        # ---------------------------------------------------------------
        if energy_keV is None:
            energy_keV = 0.0
        energy_int = int(round(float(energy_keV)))
        folder_name = f"track_{ion}_{energy_int}keV_{track_id.split('_')[-1]}"

        track_dir = output_base / folder_name
        srim_runs_dir = track_dir / "srim_runs"
        srim_runs_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # 3️⃣ Extract posterior θ samples
        # ---------------------------------------------------------------
        if track_id not in samples_dict:
            raise KeyError(f"Track ID {track_id} not found in samples_dict")

        theta_samples = samples_dict[track_id]
        if isinstance(theta_samples, torch.Tensor):
            theta_samples = theta_samples.detach().cpu().numpy().flatten().tolist()
        elif isinstance(theta_samples, np.ndarray):
            theta_samples = theta_samples.flatten().tolist()
        else:
            theta_samples = list(theta_samples)

        theta_samples = sorted(set(round(float(t), 2) for t in theta_samples))

        # Save the posterior sample list
        pd.DataFrame({"theta_eV": theta_samples}).to_csv(track_dir / "posterior_samples.csv", index=False)

        # ---------------------------------------------------------------
        # 4️⃣ Prepare x_test row
        # ---------------------------------------------------------------
        if isinstance(x, torch.Tensor):
            x_values = x.detach().cpu().numpy().tolist()
        elif isinstance(x, np.ndarray):
            x_values = x.tolist()
        elif isinstance(x, (pd.Series, dict)):
            x_values = [x["mean_depth_A"], x["std_depth_A"], x["vacancies_per_ion"]]
        else:
            x_values = list(x)

        # ---------------------------------------------------------------
        # 5️⃣ Write metadata.json
        # ---------------------------------------------------------------
        metadata = {
            "track_index": i,
            "track_id": track_id,
            "ion": ion,
            "energy_keV": energy_int,
            "composite_key": composite_key,
            "track_folder": str(track_dir),
            "num_samples": len(theta_samples),
            "theta_samples_eV": theta_samples,
            "x_test": x_values,
            "timestamp": datetime.now().isoformat()
        }

        def _to_json_safe(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            elif isinstance(obj, (torch.Tensor,)):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, (list, tuple)):
                return [_to_json_safe(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            else:
              return obj

        metadata_safe = _to_json_safe(metadata)

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata_safe, f, indent=2)

# ---------------------------------------------------------------
# 6️⃣ Run SRIM for each θ sample into srim_runs/
# ---------------------------------------------------------------

        print(
            f"[DEBUG multi_track] track={track_id} | E_true_keV={energy_keV} | "
            f"n_theta={len(theta_samples)} | first5={theta_samples[:5]}"
        )

        for theta_eV in theta_samples:
            try:
                print(f"[DEBUG: run_srim_multi_track] → Track {track_id}, running SRIM at θ={theta_eV} eV")
                folder = run_srim_for_theta(
                    theta_eV=theta_eV,
                    srim_directory=srim_directory,
                    output_base=srim_runs_dir,  # store under srim_runs
                    ion_symbol=ion,
                    number_ions=number_ions,
                    calculation=calculation,
                    density_g_cm3=density_g_cm3,
                    width_A=width_A,
                    overwrite=overwrite
                )
                print(f"[DEBUG: run_srim_multi_track] ✅ SRIM output → {folder}")
            except Exception as e:
                print(f"[DEBUG: run_srim_multi_track] ❌ SRIM failed for {track_id} @ θ={theta_eV} eV → {e}")
        
        # for theta_eV in theta_samples:
        #     run_srim_for_theta(
        #         theta_eV=theta_eV,
        #         srim_directory=srim_directory,
        #         output_base=srim_runs_dir,  # store under srim_runs
        #         ion_symbol=ion,
        #         number_ions=number_ions,
        #         calculation=calculation,
        #         density_g_cm3=density_g_cm3,
        #         width_A=width_A,
        #         overwrite=overwrite
        #     )

        print(f"[PPC] Track {track_id} complete → results in {track_dir}")
        results.append(str(track_dir))


        manifest = run_srim_batch(
            thetas_eV=theta_samples,
            srim_directory=srim_directory,
            output_base=srim_runs_dir,
            ion_symbol=ion,
            number_ions=number_ions,
            calculation=calculation,
            density_g_cm3=density_g_cm3,
            width_A=width_A,
            overwrite=overwrite,
        )

        bad = manifest[manifest["status"] != "OK"]
        print(f"[MANIFEST] track={track_id} total={len(manifest)} ok={len(manifest)-len(bad)} failed={len(bad)}")
        if not bad.empty:
            print(bad.sort_values('theta_eV').to_string(index=False))

    # ---------------------------------------------------------------
    print(f"[PPC] Track {track_id} complete → results in {track_dir}")
    results.append(str(track_dir))

    print(f"\n[PPC] ✅ All SRIM runs complete ({len(results)} tracks).")
    return results