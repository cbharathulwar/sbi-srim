#define functions for parsing through SRIM data files and summarize data 
import os
import pandas as pd
import torch
import numpy
from pathlib import Path
from typing import Dict, List
import re, math

_FLOAT = r'([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)'

def _find_file(folder: str, filename: str) -> str:
    """Find a file by exact name (case-insensitive) in folder or 'SRIM Outputs'."""
    target = filename.lower()
    if os.path.isfile(folder) and os.path.basename(folder).lower() == target:
        return folder
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            if fn.lower() == target:
                return os.path.join(folder, fn)
        so = os.path.join(folder, "SRIM Outputs")
        if os.path.isdir(so):
            for fn in os.listdir(so):
                if fn.lower() == target:
                    return os.path.join(so, fn)
    raise FileNotFoundError(f"{filename} not found under '{folder}'.")

def _parse_tdata(path: str) -> Dict[str, float]:
    energy_keV = avg_range_A = avg_straggling_A = avg_vac_per_ion = None
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if energy_keV is None and "Energy" in line and "keV" in line:
                m = re.search(_FLOAT, line);    energy_keV = float(m.group(1)) if m else None
            elif line.startswith("Average Range"):
                m = re.search(_FLOAT, line);    avg_range_A = float(m.group(1)) if m else None
            elif line.startswith("Average Straggling"):
                m = re.search(_FLOAT, line);    avg_straggling_A = float(m.group(1)) if m else None
            elif line.startswith("Average Vacancy/Ion"):
                m = re.search(_FLOAT, line);    avg_vac_per_ion = float(m.group(1)) if m else None
    if None in (energy_keV, avg_range_A, avg_straggling_A, avg_vac_per_ion):
        raise ValueError("TDATA.txt missing required fields.")
    return {
        "theta_eV": energy_keV * 1_000.0,
        "mean_depth_A": avg_range_A,
        "std_depth_A": avg_straggling_A,
        "vacancies_per_ion_tdata": avg_vac_per_ion,
    }

def _parse_vacancy(path: str) -> Dict[str, float]:
    total_header = None
    depths: List[float] = []; ions: List[float] = []; recoils: List[float] = []
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    # header total
    for line in lines:
        if "Total Target Vacancies" in line and "/Ion" in line:
            m = re.search(r'=\s*' + _FLOAT + r'\s*/Ion', line)
            if m: total_header = float(m.group(1))
            break
    # numeric rows: depth, by ions, by recoils
    numrow = re.compile(r'^\s*' + _FLOAT + r'\s+' + _FLOAT + r'\s+' + _FLOAT + r'\s*$')
    for line in lines:
        m = numrow.match(line)
        if m:
            depths.append(float(m.group(1)))
            ions.append(float(m.group(2)))
            recoils.append(float(m.group(3)))
    total_integrated = float("nan"); mean_depth = float("nan"); std_depth = float("nan")
    if len(depths) >= 2:
        rho = [a + b for a, b in zip(ions, recoils)]  # Vac/(Å·Ion)
        widths = [depths[i+1] - depths[i] for i in range(len(depths)-1)]
        widths.append(widths[-1])                     # last bin width ~= previous
        counts = [r * w for r, w in zip(rho, widths)] # Vac/Ion in each bin
        total_integrated = sum(counts)
        if total_integrated > 0:
            mean_depth = sum(x*c for x,c in zip(depths, counts)) / total_integrated
            var = sum(((x-mean_depth)**2)*c for x,c in zip(depths, counts)) / total_integrated
            std_depth = math.sqrt(var)
    mismatch_pct = None
    if total_header not in (None, 0) and not math.isnan(total_integrated):
        mismatch_pct = 100.0 * (total_integrated - total_header) / total_header
    return {
        "vacancies_per_ion_vacancy_header": total_header,
        "vacancies_per_ion_vacancy_integrated": total_integrated,
        "vacancy_depth_mean_from_table_A": mean_depth,
        "vacancy_depth_std_from_table_A": std_depth,
        "vacancy_integral_mismatch_pct": mismatch_pct,
    }

def summarize_srim_output(folder: str) -> Dict[str, float]:
    """Return the 3 core quantities + sanity checks from TDATA.txt and VACANCY.txt."""
    tdata = _parse_tdata(_find_file(folder, "TDATA.txt"))
    vac   = _parse_vacancy(_find_file(folder, "VACANCY.txt"))
    return {
        # core outputs you should use in PPC + modeling
        "theta_eV": tdata["theta_eV"],
        "mean_depth_A": tdata["mean_depth_A"],
        "std_depth_A": tdata["std_depth_A"],
        "vacancies_per_ion": tdata["vacancies_per_ion_tdata"],
        # extra fields for debugging / audits
        **tdata,
        **vac,
    }


import pandas as pd
from pathlib import Path

def summarize_all_runs(output_base: str) -> pd.DataFrame:
    """
    Loop through all tracks and their theta_* folders under output_base,
    summarize each SRIM run, and return a DataFrame with hierarchical metadata.

    Structure expected:
        output_base/
        ├── track_00/
        │   ├── metadata.json
        │   ├── theta_335/
        │   ├── theta_396/
        │   └── theta_466/
        ├── track_01/
        │   ├── theta_...
        │   └── metadata.json
        ...
    """
    base = Path(output_base)
    summaries = []

    # loop over all track directories
    for track_dir in sorted(base.glob("track_*")):
        if not track_dir.is_dir():
            continue

        track_name = track_dir.name
        track_index = int(track_name.replace("track_", ""))

        # loop over theta directories within each track
        for theta_dir in sorted(track_dir.glob("theta_*")):
            if not theta_dir.is_dir():
                continue

            theta_str = theta_dir.name.replace("theta_", "")
            try:
                theta_val = float(theta_str)
            except ValueError:
                theta_val = None

            try:
                summary = summarize_srim_output(theta_dir)
                summary["theta_eV"] = theta_val
                summary["track_index"] = track_index
                summary["track_folder"] = str(track_dir)
                summary["theta_folder"] = str(theta_dir)
                summaries.append(summary)
            except Exception as e:
                print(f"[WARN] Failed to summarize {theta_dir}: {e}")

    if not summaries:
        raise RuntimeError(f"No SRIM outputs found under {output_base}")

    df = pd.DataFrame(summaries)
    df.sort_values(["track_index", "theta_eV"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df