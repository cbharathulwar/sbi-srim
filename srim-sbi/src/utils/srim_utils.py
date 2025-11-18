"""
srim_utils.py
----------------
All SRIM-related utilities in one place:

  • Running SRIM:
      - run_srim_for_theta(theta_eV, ...)
      - run_srim_batch(thetas_eV, ...)
      - run_srim_for_energy(energy_keV, srim_dir, output_root, ...)
      - run_and_parse_energy(...)
      - run_srim_multi_track(...)

  • Parsing + summarizing SRIM outputs:
      - parse_tdata(TDATA.txt)
      - parse_vacancy(VACANCY.txt)
      - summarize_srim_output(theta_folder)
      - summarize_all_runs(output_base)
      - parse_collisions(folder_path, energy_keV)
"""

import os
import re
import io
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

from srim import TRIM, Ion, Layer, Target
import srim.output
from srim.output import Collision


# ============================================================
# SRIM PARSER PATCH — REQUIRED FOR STABILITY
# ============================================================

def _patched_read_ion(self, output):
    """
    Fix SRIM readers that choke on 'Ion = C' formatting.
    Works for both file paths and in-memory text buffers.
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


# Apply patch to all relevant SRIM output classes
for _name in ["Ioniz", "Vacancy", "Phonon", "Phonons", "E2RECOIL", "NOVAC", "EnergyToRecoils"]:
    if hasattr(srim.output, _name):
        setattr(getattr(srim.output, _name), "_read_ion", _patched_read_ion)

print("[patch] Applied SRIM parser fix.")


# ============================================================
# CONSTANTS
# ============================================================

# Files normally produced by TRIM
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

# Extra "metadata" / control files worth copying
ALSO_COPY = ["TRIM.IN", "TRIMAUTO"]

# Allow numeric atomic numbers → symbols
ELEMENT_MAP = {"6": "C", "14": "Si", "13": "Al", "8": "O", "1": "H", "79": "Au"}


# ============================================================
# LOW-LEVEL: RUN SRIM FOR A SINGLE ENERGY (eV) → theta_XXXX
# ============================================================

def run_srim_for_theta(
    theta_eV: float,
    srim_directory,
    output_base,
    ion_symbol: str = "C",
    number_ions: int = 200,
    calculation: int = 1,
    layer_spec: Optional[dict] = None,
    density_g_cm3: float = 3.51,
    width_A: float = 15000.0,
    overwrite: bool = False,
) -> str:
    """
    Run SRIM at energy theta_eV (eV).
    Creates folder: <output_base>/theta_<integer eV>/ with SRIM outputs.

    This is the generic runner used by:
      - run_srim_batch(...)
      - run_srim_multi_track(...)
    """

    srim_dir = Path(srim_directory)
    out_base = Path(output_base)

    theta_eV = float(theta_eV)
    theta_int = int(round(theta_eV))
    out_dir = out_base / f"theta_{theta_int}"

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old output files in SRIM directory
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        path = srim_dir / name
        if path.exists():
            path.unlink()

    # Default target: Carbon layer
    if layer_spec is None:
        layer_spec = {
            "C": {"stoich": 1.0, "E_d": 30.0, "lattice": 0.0, "surface": 3.0}
        }

    # Map numeric atomic numbers → symbols
    if str(ion_symbol).isdigit():
        ion_symbol = ELEMENT_MAP.get(str(ion_symbol), "C")

    ion = Ion(ion_symbol, energy=theta_eV)
    layer = Layer(layer_spec, density=density_g_cm3, width=width_A)
    target = Target([layer])

    trim = TRIM(target, ion, number_ions=number_ions, calculation=calculation)
    trim.run(str(srim_dir))

    # Copy output files to run folder
    copied = []
    for name in KNOWN_TXT_OUTPUTS + ALSO_COPY:
        src = srim_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)

    if "TDATA.txt" not in copied:
        raise RuntimeError(f"No TDATA.txt found after SRIM run @ {theta_eV} eV. Copied: {copied}")

    return str(out_dir)


# ============================================================
# MID-LEVEL: RUN SRIM FOR MANY THETA VALUES (eV)
# ============================================================

def run_srim_batch(
    thetas_eV,
    srim_directory,
    output_base,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Run SRIM for a list of energies in eV.
    Each run is stored under: <output_base>/theta_<integer eV>/.

    Returns
    -------
    pd.DataFrame
        Columns: [theta_eV, output_folder, status, timestamp]
    """
    out_base = Path(output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    results = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[INFO] Running SRIM batch ({len(thetas_eV)} energies) → {out_base}")

    for theta in sorted(thetas_eV):
        try:
            folder = run_srim_for_theta(
                theta_eV=theta,
                srim_directory=srim_directory,
                output_base=out_base,
                overwrite=overwrite,
                **kwargs,
            )
            results.append((theta, folder, "OK", stamp))
        except Exception as e:
            print(f"[WARN] SRIM failed @ {theta:.2f} eV: {e}")
            results.append((theta, None, f"ERROR: {e}", stamp))

    df = pd.DataFrame(results, columns=["theta_eV", "output_folder", "status", "timestamp"])

    manifest_path = out_base / f"srim_batch_manifest_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(manifest_path, index=False)
    print(f"[INFO] Batch manifest saved → {manifest_path}")

    return df


# ============================================================
# ALT RUNNER: RUN SRIM IN keV & DUMP COLLISIONS.TXT → folder
# (for raw event-level data / PySRIM Collision parsing)
# ============================================================

def run_srim_for_energy(
    energy_keV: float,
    srim_dir,
    output_root,
    number_ions: int = 200,
) -> str:
    """
    Runs SRIM for a given energy (keV) and saves outputs into:

        <output_root>/<energy_keV:.1f>keV/

    Ensures that a COLLISON.txt (PySRIM-expected filename) exists
    in the SRIM Outputs directory before copying.
    """
    print(f"[INFO] Running SRIM for {energy_keV:.1f} keV")

    # Output directory for this energy
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
    ion = Ion("C", energy=energy_keV * 1e3)  # keV → eV
    layer = Layer({"C": 1.0}, density=3.51, width=20000.0)
    target = Target([layer])

    # Run in SRIM directory
    cwd = os.getcwd()
    os.chdir(srim_dir)
    try:
        trim = TRIM(
            target=target,
            ion=ion,
            number_ions=number_ions,
            calculation=2,
            collisions=2,  # IMPORTANT: dump collision cascades
        )
        trim.run(srim_directory=srim_dir)
    finally:
        os.chdir(cwd)

    # Wait for SRIM to produce a COLLISIONS*/COLLISON* file
    collisions_file = None
    for _ in range(20):  # up to ~10 seconds
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

    # Normalize filename → PySRIM expects COLLISON.txt (no 'S')
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


# ============================================================
# COLLISION PARSER (PySRIM Collision → flat table)
# ============================================================

def parse_collisions(folder_path: str, energy_keV: float) -> pd.DataFrame:
    """
    Parse SRIM COLLISON.txt into a flat event table:

        columns: x, y, z, ion_number, energy_keV

    Parameters
    ----------
    folder_path : str
        Path to folder containing COLLISON.txt (or SRIM Outputs dir)
    energy_keV : float
        Energy label to attach to each row

    Returns
    -------
    pd.DataFrame
    """
    coll = Collision(folder_path)
    rows = []

    for ion_idx in range(len(coll)):
        ion_data = coll[ion_idx]
        collisions = ion_data.get("collisions", None)
        if collisions is None:
            continue

        for primary in collisions:
            for recoil in primary.get("cascade", []):
                pos = recoil.get("position", None)
                if pos is None:
                    continue
                rows.append(
                    {
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2],
                        "ion_number": ion_idx,
                        "energy_keV": energy_keV,
                    }
                )

    df = pd.DataFrame(rows, columns=["x", "y", "z", "ion_number", "energy_keV"])

    # Enforce numeric dtypes and drop bad rows
    for c in ["x", "y", "z", "energy_keV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ion_number"] = pd.to_numeric(df["ion_number"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["x", "y", "z", "ion_number", "energy_keV"]).reset_index(drop=True)
    df["ion_number"] = df["ion_number"].astype(int)

    return df


def run_and_parse_energy(
    energy_keV: float,
    srim_dir,
    output_root,
    number_ions: int = 200,
) -> pd.DataFrame:
    """
    Convenience wrapper:

      1. run_srim_for_energy(energy_keV, ...)
      2. parse_collisions(...)
      3. save <energy>_with_energy.csv inside that folder
    """
    print(f"\nStarting SRIM run for {energy_keV:.1f} keV")

    folder_path = run_srim_for_energy(
        energy_keV=energy_keV,
        srim_dir=srim_dir,
        output_root=output_root,
        number_ions=number_ions,
    )

    df = parse_collisions(folder_path, energy_keV)

    save_csv = os.path.join(folder_path, f"{energy_keV:.1f}_with_energy.csv")
    df.to_csv(save_csv, index=False)
    print(f"[OK] Saved parsed data for {energy_keV:.1f} keV → {save_csv}")

    return df


# ============================================================
# HELPER FOR TEXT-PARSING (TDATA / VACANCY)
# ============================================================

_FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"


def _find_file(folder: str | Path, name: str) -> str:
    """
    Case-insensitive search for <name> in folder or folder/SRIM Outputs.
    """
    folder = Path(folder)
    target = name.lower()

    # direct match
    if folder.is_file() and folder.name.lower() == target:
        return str(folder)

    if folder.is_dir():
        for f in folder.iterdir():
            if f.name.lower() == target:
                return str(f)

        so = folder / "SRIM Outputs"
        if so.is_dir():
            for f in so.iterdir():
                if f.name.lower() == target:
                    return str(f)

    raise FileNotFoundError(f"{name} not found under {folder}")


# ============================================================
# PARSE TDATA.TXT
# ============================================================

def parse_tdata(path: str) -> Dict[str, float]:
    """
    Parse TDATA.txt:
      • Energy (keV)
      • Mean range / depth (Å)
      • Vacancies per ion
      • Max depth (Å)
    """
    energy_keV = None
    avg_range_A = None
    avg_vac = None
    max_depth_A = None

    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            # energy in keV
            if energy_keV is None and "Energy" in line and "keV" in line:
                m = re.search(_FLOAT, line)
                if m:
                    energy_keV = float(m.group(1))

            elif line.startswith("Average Range"):
                m = re.search(_FLOAT, line)
                if m:
                    avg_range_A = float(m.group(1))

            elif line.startswith("Average Vacancy/Ion"):
                m = re.search(_FLOAT, line)
                if m:
                    avg_vac = float(m.group(1))

            elif "Depth Range of Tabulated Data" in line:
                m = re.findall(_FLOAT, line)
                if m and len(m) >= 2:
                    try:
                        max_depth_A = float(m[-1])
                    except Exception:
                        pass

    if energy_keV is None:
        raise ValueError(f"Energy not found in {path}")

    return {
        "energy_keV": energy_keV,
        "theta_eV": energy_keV * 1_000.0,
        "mean_depth_A_tdata": avg_range_A,
        "vacancies_per_ion_tdata": avg_vac,
        "max_depth_A": max_depth_A,
    }


# ============================================================
# RELATIVE BINNING
# ============================================================

def infer_relative_bin_edges(n_bins: int = 6, r_min: float = 1e-3, r_max: float = 1.0) -> np.ndarray:
    """
    Relative depth bins r = depth / characteristic_depth, log-spaced.
    Includes r=0 as first edge.
    """
    edges = np.geomspace(r_min, r_max, n_bins)
    edges = np.insert(edges, 0, 0.0)
    return edges


# ============================================================
# PARSE VACANCY.TXT
# ============================================================

def parse_vacancy(path: str, n_bins: int = 6) -> Dict[str, float]:
    """
    Extract vacancy profile from VACANCY.txt:

      • total vacancies per ion (integrated)
      • mean vacancy depth
      • relative depth fractions rbin_frac_1..n_bins (using 95th percentile depth as scale)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    depths, ions, recoils = [], [], []
    total_header = None

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    # header: total vacancies
    for line in lines:
        if "Total Target Vacancies" in line and "/Ion" in line:
            m = re.search(r"=\s*" + _FLOAT, line)
            if m:
                total_header = float(m.group(1))
            break

    # numeric rows: depth, ions, recoils
    numrow = re.compile(r"^\s*" + _FLOAT + r"\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s*$")
    for line in lines:
        m = numrow.match(line)
        if m:
            depths.append(float(m.group(1)))
            ions.append(float(m.group(2)))
            recoils.append(float(m.group(3)))

    if len(depths) == 0:
        return {}

    depths = np.asarray(depths, float)
    ions = np.asarray(ions, float)
    recoils = np.asarray(recoils, float)
    rho = ions + recoils

    # approximate bin widths
    if len(depths) > 1:
        widths = np.diff(np.append(depths, depths[-1] + (depths[-1] - depths[-2])))
    else:
        widths = np.array([1.0])

    counts = rho * widths
    total_integrated = float(np.sum(counts))

    if total_integrated <= 0:
        return {}

    # weighted mean depth
    mean_depth = float(np.sum(depths * counts) / total_integrated)

    # weighted quantile helper
    def w_q(x, w, q):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            return np.nan
        x, w = x[mask], w[mask]
        idx = np.argsort(x)
        x, w = x[idx], w[idx]
        cw = np.cumsum(w)
        cutoff = q * cw[-1]
        k = np.searchsorted(cw, cutoff, side="left")
        return float(x[min(max(k, 0), len(x) - 1)])

    p95 = w_q(depths, counts, 0.95)
    if not np.isfinite(p95) or p95 <= 0:
        return {}

    # relative depth r = depth / p95
    r_edges = infer_relative_bin_edges(n_bins=n_bins)
    r = depths / (p95 + 1e-12)

    hist, _ = np.histogram(r, bins=r_edges, weights=counts)
    if np.any(r > r_edges[-1]):
        hist[-1] += float(np.sum(counts[r > r_edges[-1]]))

    if hist.sum() > 0:
        hist = hist / hist.sum()

    out = {
        "vacancies_per_ion": total_integrated,
        "vacancy_depth_mean_A": mean_depth,
        "vacancy_integral_mismatch_pct": (
            100.0 * (total_integrated - total_header) / total_header
            if (total_header and total_header != 0)
            else np.nan
        ),
    }

    for i, v in enumerate(hist, start=1):
        out[f"rbin_frac_{i}"] = float(v)

    return out


# ============================================================
# SUMMARIZE A SINGLE SRIM RUN (one theta_XXXX folder)
# ============================================================

def summarize_srim_output(theta_folder: str | Path, n_bins: int = 6) -> Dict[str, Any]:
    """
    Summarize a single SRIM output folder.

    Returns dict with:
      energy_keV, theta_eV, max_depth_A,
      mean_depth_A, vacancies_per_ion, vacancy_integral_mismatch_pct,
      rbin_frac_*
    """
    tdata_path = _find_file(theta_folder, "TDATA.txt")
    vac_path = _find_file(theta_folder, "VACANCY.txt")

    tdata = parse_tdata(tdata_path)
    vac = parse_vacancy(vac_path, n_bins=n_bins)

    if not vac:
        return {}

    out = {
        "energy_keV": tdata.get("energy_keV", np.nan),
        "theta_eV": tdata.get("theta_eV", np.nan),
        "max_depth_A": tdata.get("max_depth_A", np.nan),
        "mean_depth_A": vac.get("vacancy_depth_mean_A", np.nan),
        "vacancies_per_ion": vac.get("vacancies_per_ion", np.nan),
        "vacancy_integral_mismatch_pct": vac.get("vacancy_integral_mismatch_pct", np.nan),
    }

    for k, v in vac.items():
        if k.startswith("rbin_frac_"):
            out[k] = float(v)

    return out


# ============================================================
# SUMMARIZE ALL RUNS UNDER BASE DIRECTORY
# ============================================================

def summarize_all_runs(
    output_base: str | Path,
    *,
    label: Optional[str] = None,
    strict: bool = True,
    n_bins: int = 6,
) -> pd.DataFrame:
    """
    Walks:

        <output_base>/track_*/srim_runs/theta_*/

    Reads metadata.json (if present) in each track_*/ folder to get
    track_id, ion, etc.

    Returns a DataFrame where each row is a single SRIM run (one theta).
    """
    output_base = Path(output_base)
    if not output_base.exists():
        raise FileNotFoundError(output_base)

    rows = []
    track_dirs = sorted(output_base.glob("track_*"))

    for track_dir in track_dirs:
        if not track_dir.is_dir():
            continue

        track_id_default = track_dir.name.replace("track_", "")
        srim_runs_dir = track_dir / "srim_runs"

        # metadata.json (optional)
        meta_path = track_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                pass

        if not srim_runs_dir.exists():
            continue

        theta_dirs = sorted(srim_runs_dir.glob("theta_*"))
        if strict and not theta_dirs:
            raise RuntimeError(f"No theta_* runs in {track_dir}")

        for theta_dir in theta_dirs:
            try:
                summary = summarize_srim_output(theta_dir, n_bins=n_bins)
            except Exception as e:
                print(f"[WARN] Skipping bad SRIM run {theta_dir}: {e}")
                continue

            if not summary:
                continue

            row = {
                "track_folder": str(track_dir),
                "theta_folder": str(theta_dir),
                "track_id": meta.get("track_id", track_id_default),
                "ion": meta.get("ion", "C"),
                "energy_keV": summary.get("energy_keV"),
                "theta_eV": summary.get("theta_eV"),
                "mean_depth_A": summary.get("mean_depth_A"),
                "max_depth_A": summary.get("max_depth_A"),
                "vacancies_per_ion": summary.get("vacancies_per_ion"),
                "vacancy_integral_mismatch_pct": summary.get("vacancy_integral_mismatch_pct"),
            }

            for k, v in summary.items():
                if k.startswith("rbin_frac_"):
                    row[k] = v

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid SRIM outputs under {output_base}")

    df = pd.DataFrame(rows).sort_values(["track_id", "theta_eV"]).reset_index(drop=True)

    # Save summary CSV
    stamp = label or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_base / f"srim_summary_{stamp}.csv"
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Summarized {len(df)} SRIM runs → {out_csv}")
    return df


# ============================================================
# HIGH-LEVEL: RUN SRIM FOR MULTIPLE TRACKS (POSTERIOR SAMPLES)
# ============================================================

def run_srim_multi_track(
    samples_dict: Dict[str, Any],
    x_test: pd.DataFrame,
    track_ids,
    srim_directory,
    output_base,
    ion_symbol: str = "C",
    number_ions: int = 200,
    calculation: int = 1,
    density_g_cm3: float = 3.51,
    width_A: float = 15000.0,
    overwrite: bool = False,
    df_summary: Optional[pd.DataFrame] = None,
) -> list[str]:
    """
    For each track:
      • Make folder: track_<ion>_<E>keV_<suffix>/
      • Save metadata.json + posterior_samples.csv
      • Run SRIM for each posterior sample under srim_runs/theta_XXXX/

    Parameters
    ----------
    samples_dict : dict
        { track_id: samples } where samples are energies in keV (or tensor).
    x_test : pd.DataFrame
        Test set (only used for sanity metadata; can be ignored here).
    track_ids : list
        Track identifiers, must match len(x_test).
    srim_directory : str | Path
        SRIM installation directory.
    output_base : str | Path
        Base directory for all track_* folders.
    df_summary : pd.DataFrame, optional
        If provided, used to extract ion + energy_keV for naming.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if len(track_ids) != len(x_test):
        raise ValueError("track_ids and x_test must match length")

    print(f"[INFO] Running SRIM for {len(track_ids)} tracks.")
    results = []

    for i, (row, tid) in enumerate(zip(x_test.itertuples(index=False), track_ids)):
        tid = str(tid)
        print(f"[INFO] Track {i+1}/{len(track_ids)}: {tid}")

        ion = ion_symbol
        energy_keV = None

        if df_summary is not None and "track_id" in df_summary.columns:
            match = df_summary[df_summary["track_id"] == tid]
            if not match.empty:
                rec = match.iloc[0]
                ion = rec.get("ion", ion_symbol)
                energy_keV = rec.get("energy_keV", None)

        if str(ion).isdigit():
            ion = ion_symbol

        energy_int = int(round(float(energy_keV or 0.0)))

        # Unique suffix from track_id
        suffix = re.sub(r"[^a-zA-Z0-9]+", "", tid)[-8:]
        track_dir = output_base / f"track_{ion}_{energy_int}keV_{suffix}"
        srim_runs_dir = track_dir / "srim_runs"
        srim_runs_dir.mkdir(parents=True, exist_ok=True)

        # Grab posterior samples in keV
        if tid not in samples_dict:
            raise KeyError(f"Missing posterior samples for {tid}")

        samples_keV = samples_dict[tid]
        if isinstance(samples_keV, torch.Tensor):
            samples_keV = samples_keV.cpu().numpy().flatten().tolist()
        else:
            samples_keV = list(np.array(samples_keV).flatten())

        # Deduplicate & round
        samples_keV = sorted(set(round(float(t), 2) for t in samples_keV))
        samples_eV = [t * 1_000.0 for t in samples_keV]

        # sanity check on units
        for val_keV, val_eV in zip(samples_keV, samples_eV):
            assert 1 <= val_keV <= 2000, f"Posterior θ {val_keV} keV out of range"
            assert 1_000 <= val_eV <= 2_000_000, f"Converted θ {val_eV} eV out of range"

        # Save posterior samples
        pd.DataFrame({"theta_eV": samples_eV}).to_csv(
            track_dir / "posterior_samples.csv", index=False
        )

        # Metadata
        metadata = {
            "track_index": i,
            "track_id": tid,
            "ion": ion,
            "energy_keV": energy_int,
            "num_samples": len(samples_eV),
            "theta_samples_keV": samples_keV,
            "theta_samples_eV": samples_eV,
            "timestamp": datetime.now().isoformat(),
        }

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Run SRIM for each posterior sample (in eV)
        for theta_eV in samples_eV:
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
                print(f"[WARN] SRIM failed for {tid} @ {theta_eV} eV: {e}")

        results.append(str(track_dir))
        print(f"[INFO] Completed track {tid}")

    print(f"[INFO] SRIM finished for all tracks.")
    return results