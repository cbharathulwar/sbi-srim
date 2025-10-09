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


def pick_random_tracks(x_obs, n=10, seed=None):
    """
    Randomly select n tracks from x_obs and return both
    the selected rows and their indices (for reproducibility).

    Parameters
    ----------
    x_obs : torch.Tensor
        Full observation tensor, shape (n_tracks, n_features)
    n : int
        Number of tracks to sample
    seed : int | None
        Optional random seed for reproducibility

    Returns
    -------
    x_test : torch.Tensor
        Sampled observation rows
    idx : torch.Tensor
        Indices of sampled tracks in x_obs
    """
    if seed is not None:
        torch.manual_seed(seed)

    total_tracks = x_obs.shape[0]
    n = min(n, total_tracks)

    idx = torch.randperm(total_tracks)[:n]
    x_test = x_obs[idx]

    return x_test, idx

import torch

import torch







import shutil
from pathlib import Path
def sample_posterior_bulk(posterior, x_obs, num_samples=100, track_ids=None):
    """
    Sample posterior for multiple tracks and store results per track ID.

    Parameters
    ----------
    posterior : trained sbi posterior
        Trained posterior object (e.g., DirectPosterior)
    x_obs : torch.Tensor of shape (n_tracks, n_features)
        Observation tensor
    num_samples : int
        Number of posterior samples per track
    track_ids : list[int] | None
        Optional original indices for each observation in x_obs

    Returns
    -------
    samples_by_track : dict[int, torch.Tensor]
        Mapping track_id -> samples [num_samples, n_params]
    samples_tensor : torch.Tensor
        Combined samples [n_tracks, num_samples, n_params]
    """
    samples_by_track = {}
    all_tensors = []

    # Iterate over each observation and its associated track ID
    for i, x in enumerate(x_obs):
        with torch.no_grad():
            theta_samples = posterior.sample((num_samples,), x=x)

        # Determine the correct track ID
        track_id = int(track_ids[i]) if track_ids is not None else i

        samples_by_track[track_id] = theta_samples
        all_tensors.append(theta_samples)

    samples_tensor = torch.stack(all_tensors)
    return samples_by_track, samples_tensor



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
    n_jobs=None
):
    """
    Run SRIM for multiple test tracks, each with its own posterior-sampled energies.
    Parallelized across tracks for faster execution.
    """

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if track_ids is None:
        track_ids = list(range(len(x_test)))

    if len(track_ids) != len(x_test):
        raise ValueError("track_ids must match x_test length")

    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    print(f"[PPC] Parallel SRIM run using {n_jobs} cores ...")

    # --------------------------------------------------
    # INNER FUNCTION: runs one track at a time
    # --------------------------------------------------
    def process_track(i, x, track_id):
        print(f"\n[PPC] === Running SRIM for Track ID {track_id} (Index {i}) ===")

        track_id = int(track_id)  # ensure it's an int for dict keys
        track_dir = output_base / f"track_{int(track_id):04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        srim_sandbox = track_dir / "srim_run"
        srim_sandbox.mkdir(parents=True, exist_ok=True)


        theta_samples = samples_dict[int(track_id)].detach().cpu().numpy().flatten().tolist()
        theta_samples = sorted(set(round(t, 2) for t in theta_samples))

        metadata = {
            "track_index": int(i),
            "track_id": int(track_id),
            "x_test": x.detach().cpu().numpy().tolist(),
            "num_samples": len(theta_samples),
            "theta_samples_eV": theta_samples,
            "timestamp": datetime.now().isoformat(), 
        }
        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        run_srim_batch(
            thetas_eV=theta_samples,
            srim_directory=srim_sandbox,
            output_base=track_dir,
            ion_symbol=ion_symbol,
            number_ions=number_ions,
            calculation=calculation,
            density_g_cm3=density_g_cm3,
            width_A=width_A,
            overwrite=overwrite,
        )

        print(f"[PPC] Track {track_id} complete → results in {track_dir}")
        return str(track_dir)

    # --------------------------------------------------
    # RUN ALL TRACKS IN PARALLEL
    # --------------------------------------------------
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_track)(i, x, track_id)
        for i, (x, track_id) in enumerate(zip(x_test, track_ids))
    )

    print(f"\n[PPC] All SRIM PPC runs complete ({len(results)} tracks).")
    return results