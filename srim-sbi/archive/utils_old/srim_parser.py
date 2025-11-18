# define functions for parsing through SRIM data files and summarize data
import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re, math
import json

_FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"


def _find_file(folder: str, filename: str) -> str:
    """
    Look for a file (case-insensitive) inside a folder.

    Checks both the main folder and a subfolder called 'SRIM Outputs'.
    Returns the full path if found, otherwise raises FileNotFoundError.
    """
    target = filename.lower()

    # direct file match
    if os.path.isfile(folder) and os.path.basename(folder).lower() == target:
        return folder

    # look inside folder and optional SRIM Outputs/
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
    """
    Read TDATA.txt for metadata:
      - ion energy (keV → eV)
      - mean range (Å)
      - vacancies per ion
      - max simulation depth (Å)
    """
    energy_keV = avg_range_A = avg_vac_per_ion = None
    max_depth_A = None

    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            # energy in keV
            if energy_keV is None and "Energy" in line and "keV" in line:
                m = re.search(_FLOAT, line)
                energy_keV = float(m.group(1)) if m else None

            # average range
            elif line.startswith("Average Range"):
                m = re.search(_FLOAT, line)
                avg_range_A = float(m.group(1)) if m else None

            # average vacancies per ion
            elif line.startswith("Average Vacancy/Ion"):
                m = re.search(_FLOAT, line)
                avg_vac_per_ion = float(m.group(1)) if m else None

            # depth range upper limit
            elif "Depth Range of Tabulated Data" in line:
                m = re.findall(_FLOAT, line)
                if m and len(m) >= 2:
                    try:
                        max_depth_A = float(m[-1])
                    except Exception:
                        pass

    if None in (energy_keV, avg_range_A, avg_vac_per_ion):
        raise ValueError(f"[ERROR] Missing required fields in TDATA.txt at {path}")

    return {
        "energy_keV": energy_keV,
        "theta_eV": energy_keV * 1_000.0,
        "mean_depth_A": avg_range_A,
        "vacancies_per_ion": avg_vac_per_ion,
        "max_depth_A": max_depth_A,
    }


def infer_relative_bin_edges(
    n_bins: int = 6,
    r_min: float = 1e-3,
    r_max: float = 1.0,
) -> np.ndarray:
    """
    Create relative depth bin edges for r = depth / mean_depth.

    Uses log spacing to give finer resolution near the surface (r ≈ 0).
    Any r > r_max values are added to the last bin.
    """
    import numpy as np

    # basic input safety
    r_min = max(float(r_min), 1e-6)
    r_max = float(r_max)
    if r_max <= r_min:
        r_max = r_min * 10.0

    # log-spaced bins with r=0 included
    edges = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_bins + 1))
    edges[0] = 0.0

    # make sure edges increase properly
    edges = np.asarray(edges, float)
    if not np.all(np.diff(edges) > 0):
        edges = np.unique(edges)
        if edges.size != n_bins + 1 or not np.all(np.diff(edges) > 0):
            raise ValueError("Bad relative bin edges.")

    return edges

def _parse_vacancy(path: str, n_bins: int = 6) -> Dict[str, float]:
    """
    Read SRIM VACANCY.txt and extract key values:
      - total vacancies per ion
      - mean vacancy depth (Å)
      - relative depth fractions (r = depth / 95th percentile depth)
    """
    import numpy as np, re, os

    depths, ions, recoils = [], [], []
    total_header = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"VACANCY.txt not found: {path}")

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    # find total vacancies from header
    for line in lines:
        if "Total Target Vacancies" in line and "/Ion" in line:
            m = re.search(r"=\s*" + _FLOAT + r"\s*/Ion", line)
            if m:
                total_header = float(m.group(1))
            break

    # read the main data table (depth, ions, recoils)
    numrow = re.compile(r"^\s*" + _FLOAT + r"\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s*$")
    for line in lines:
        m = numrow.match(line)
        if m:
            depths.append(float(m.group(1)))
            ions.append(float(m.group(2)))
            recoils.append(float(m.group(3)))

    if len(depths) == 0:
        print(f"[WARN] No numeric VACANCY data in {path}")
        return {
            "vacancies_per_ion_vacancy_header": total_header,
            "vacancies_per_ion_vacancy_integrated": np.nan,
            "vacancy_depth_mean_from_table_A": np.nan,
            "vacancy_integral_mismatch_pct": np.nan,
        }

    depths, ions, recoils = map(np.asarray, [depths, ions, recoils], [float] * 3)

    # compute total vacancies and mean depth
    rho = ions + recoils
    widths = np.diff(np.append(depths, depths[-1] + (depths[-1] - depths[-2]))) if len(depths) > 1 else np.array([1.0])
    counts = rho * widths
    total_integrated = float(np.sum(counts))
    mean_depth = (
        float(np.sum(depths * counts) / total_integrated) if total_integrated > 0 else np.nan
    )

    # find weighted 95th percentile depth
    def _weighted_quantile(x, w, q):
        x, w = np.asarray(x, float), np.asarray(w, float)
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

    p95_depth = _weighted_quantile(depths, counts, 0.95)

    # compute relative bin fractions
    rbin_fracs = {}
    try:
        if np.isfinite(p95_depth) and p95_depth > 0 and total_integrated > 0:
            r_edges = infer_relative_bin_edges(n_bins=n_bins)
            r = depths / (p95_depth + 1e-12)
            hist, _ = np.histogram(r, bins=r_edges, weights=counts)
            if np.any(r > r_edges[-1]):
                hist[-1] += float(np.sum(counts[r > r_edges[-1]]))
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else np.full_like(hist, np.nan)
            for i, v in enumerate(hist, start=1):
                rbin_fracs[f"rbin_frac_{i}"] = float(v)
        else:
            for i in range(1, n_bins + 1):
                rbin_fracs[f"rbin_frac_{i}"] = np.nan
    except Exception as e:
        print(f"[WARN] Relative bin fraction computation failed for {path}: {e}")
        for i in range(1, n_bins + 1):
            rbin_fracs[f"rbin_frac_{i}"] = np.nan

    mismatch_pct = (
        100.0 * (total_integrated - total_header) / total_header
        if (total_header and total_header != 0)
        else np.nan
    )

    return {
        "vacancies_per_ion_vacancy_header": total_header,
        "vacancies_per_ion_vacancy_integrated": total_integrated,
        "vacancy_depth_mean_from_table_A": mean_depth,
        "vacancy_integral_mismatch_pct": mismatch_pct,
        **rbin_fracs,
    }
def summarize_srim_output(folder: str, n_bins: int = 6) -> Dict[str, float]:
    """
    Summarize one SRIM run into a feature dictionary.

    Pulls energy info from TDATA.txt and all physical values from VACANCY.txt.
    No fallback to TDATA for physics values to avoid unit mismatches.
    """
    import numpy as np

    # read SRIM component files
    tdata = _parse_tdata(_find_file(folder, "TDATA.txt"))
    vac = _parse_vacancy(_find_file(folder, "VACANCY.txt"), n_bins=n_bins)

    # combine results (VACANCY data is the physics truth)
    out = {
        "theta_eV": tdata.get("theta_eV", np.nan),
        "energy_keV": tdata.get("energy_keV", np.nan),
        "mean_depth_A": vac.get("vacancy_depth_mean_from_table_A", np.nan),
        "vacancies_per_ion": vac.get("vacancies_per_ion_vacancy_integrated", np.nan),
        "vacancy_integral_mismatch_pct": vac.get("vacancy_integral_mismatch_pct", np.nan),
        "max_depth_A": tdata.get("max_depth_A", np.nan),  # metadata only
    }

    # add relative depth fractions (rbin_frac_*)
    for k, v in vac.items():
        if k.startswith("rbin_frac_"):
            out[k] = float(v) if np.isfinite(v) else np.nan

    # skip if data invalid
    if not np.isfinite(out["mean_depth_A"]) or not np.isfinite(out["vacancies_per_ion"]):
        print(f"[WARN] Skipping {folder} — missing or bad VACANCY data.")
        return {}

    return out

import os, re, json, math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


def _parse_theta_folder(theta_dir: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Parse folder name 'theta_*' into:
      - theta_id (int) if suffix is pure digits, e.g. 'theta_335' -> 335
      - theta_folder_val (float) if suffix parses as float, e.g. 'theta_20000.0' -> 20000.0
    If neither pattern fits, returns (None, None).
    """
    m = re.match(r"^theta_(.+)$", theta_dir.name)
    if not m:
        return None, None
    token = m.group(1)

    if re.fullmatch(r"\d+", token):
        return int(token), None

    try:
        return None, float(token)
    except ValueError:
        return None, None

def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _make_composite_key(ion: Any, energy_keV: Any) -> Optional[str]:
    ion_ok = ion is not None
    e = _to_float(energy_keV)
    if (not ion_ok) or (e is None) or (math.isnan(e)):
        return None
    return f"ion={ion}|E={int(round(e))}keV"


from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import Optional, Dict, Any


def summarize_all_runs(
    output_base: str,
    *,
    label: Optional[str] = None,
    strict: bool = True,
    n_bins: int = 6,
) -> pd.DataFrame:
    """
    Go through all SRIM runs under output_base and collect results into one DataFrame.

    Each row is a single SRIM run (one theta sample).
    Output columns match the preprocess format: energy, depth stats, and rbin fractions.
    """
    base = Path(output_base)
    if not base.exists():
        raise FileNotFoundError(f"output_base not found: {output_base}")

    rows = []
    n_tracks, n_theta_runs = 0, 0

    # loop over each track directory
    for track_dir in sorted(base.glob("track_*")):
        if not track_dir.is_dir():
            continue

        track_name = track_dir.name
        if not track_name.startswith("track_"):
            continue

        # --- basic folder parsing ---
        parts = track_name.replace("track_", "").split("_")
        ion, energy_keV, track_id = None, None, None

        for p in parts:
            if p.lower().endswith("kev"):
                try:
                    energy_keV = float(p.lower().replace("kev", ""))
                except ValueError:
                    pass
            elif p.isalpha():
                ion = p
            else:
                track_id = p

        ion = ion or "C"
        track_id = track_id or parts[-1]
        track_index = None

        # --- metadata override ---
        meta_path = track_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                ion = metadata.get("ion", ion)
                energy_keV = metadata.get("energy_keV", energy_keV)
                track_id = str(metadata.get("track_id", track_id))
                track_index = metadata.get("track_index", track_index)
            except Exception as e:
                msg = f"[WARN] Failed to read {meta_path}: {e}"
                if strict:
                    raise RuntimeError(msg) from e
                print(msg)

        energy_keV = _to_float(energy_keV)
        composite_key = _make_composite_key(ion, energy_keV)

        # --- find theta_* folders ---
        srim_runs_dir = track_dir / "srim_runs"
        theta_dirs = (
            sorted(srim_runs_dir.glob("theta_*"))
            if srim_runs_dir.exists()
            else sorted(track_dir.glob("theta_*"))
        )
        if strict and not theta_dirs:
            raise RuntimeError(f"No theta_* subfolders under {track_dir}")

        # optional batch manifest
        manifest_status = {}
        manifest_path = track_dir / "srim_batch_manifest.csv"
        if manifest_path.exists():
            try:
                df_manifest = pd.read_csv(manifest_path)
                manifest_status = dict(
                    zip(df_manifest["theta_eV"].astype(float), df_manifest["status"])
                )
            except Exception:
                print(f"[WARN] Failed to parse manifest for {track_dir}")

        # --- summarize each theta folder ---
        for theta_dir in theta_dirs:
            if not theta_dir.is_dir():
                continue

            theta_id, theta_folder_val = _parse_theta_folder(theta_dir)

            try:
                s = summarize_srim_output(str(theta_dir), n_bins=n_bins)
            except Exception as e:
                print(f"[WARN] Skipping {theta_dir}: {e}")
                continue

            if not isinstance(s, dict) or not s:
                continue

            if not np.isfinite(s.get("mean_depth_A", np.nan)):
                print(f"[WARN] Invalid SRIM summary in {theta_dir}, skipping.")
                continue

            theta_val = s.get("theta_eV", None)
            status = manifest_status.get(theta_val, "OK" if theta_val else "UNKNOWN")

            # one summary row per run
            row = {
                "track_folder": str(track_dir),
                "theta_folder": str(theta_dir),
                "track_index": track_index,
                "track_id": track_id,
                "ion": ion,
                "energy_keV": s.get("energy_keV", energy_keV),
                "theta_eV": s.get("theta_eV", None),
                "mean_depth_A": s.get("mean_depth_A", np.nan),
                "max_depth_A": s.get("max_depth_A", np.nan),
                "vacancies_per_ion": s.get("vacancies_per_ion", np.nan),
                "vacancy_integral_mismatch_pct": s.get(
                    "vacancy_integral_mismatch_pct", np.nan
                ),
                "composite_key": composite_key,
                "theta_id": theta_id,
                "theta_folder_val": theta_folder_val,
                "status": status,
            }

            # add rbin fractions
            for k, v in s.items():
                if k.startswith("rbin_frac_"):
                    row[k] = float(v) if np.isfinite(v) else np.nan

            rows.append(row)
            n_theta_runs += 1
        n_tracks += 1

    if not rows:
        raise RuntimeError(f"No SRIM outputs found under {output_base}")

    # --- build dataframe ---
    df = (
        pd.DataFrame(rows)
        .sort_values(["ion", "energy_keV", "track_id", "theta_id"], na_position="last")
        .reset_index(drop=True)
    )

    # check required columns
    for c in ["mean_depth_A", "vacancies_per_ion", "track_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    # make sure all rbin_frac_* are floats
    rbin_cols = sorted([c for c in df.columns if c.startswith("rbin_frac_")])
    if rbin_cols:
        df[rbin_cols] = df[rbin_cols].astype(float)

    # --- save summary ---
    stamp = label or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(output_base) / f"srim_summary_{stamp}.csv"
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Summarized {n_tracks} tracks, {n_theta_runs} theta runs.")
    print(f"[INFO] Saved → {out_csv}")

    return df