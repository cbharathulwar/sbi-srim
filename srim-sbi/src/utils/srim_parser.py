#define functions for parsing through SRIM data files and summarize data 
import os
import pandas as pd
import torch
import numpy
from pathlib import Path
from typing import Dict, List
import re, math
import json

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


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


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
) -> pd.DataFrame:
    """
    Walk output_base/track_*/srim_runs/theta_*/ and summarize each SRIM run into a PPC-ready DataFrame.

    Final columns:
      - ion, energy_keV, composite_key
      - track_id, track_index, track_folder, theta_folder
      - theta_id, theta_folder_val, theta_file_eV
      - mean_depth_A, std_depth_A, vacancies_per_ion
      - status (optional, if srim_batch_manifest exists)
    """

    base = Path(output_base)
    if not base.exists():
        raise FileNotFoundError(f"output_base not found: {output_base}")

    rows = []
    n_tracks, n_theta_runs = 0, 0

    for track_dir in sorted(base.glob("track_*")):
        if not track_dir.is_dir():
            continue

        # ------------------------------------------------------------
        # 1️⃣ Parse folder name
        # ------------------------------------------------------------
        track_name = track_dir.name
        if not track_name.startswith("track_"):
            continue

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

        if ion is None:
            ion = "C"
        if track_id is None:
            track_id = parts[-1]
        track_index = None

        # ------------------------------------------------------------
        # 2️⃣ Load metadata.json (overrides folder info)
        # ------------------------------------------------------------
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
                if strict:
                    raise RuntimeError(f"Failed to read {meta_path}: {e}") from e
                print(f"[WARN] Failed to read {meta_path}: {e}")

        energy_keV = _to_float(energy_keV)
        composite_key = _make_composite_key(ion, energy_keV)

        # ------------------------------------------------------------
        # 3️⃣ Detect theta directories (new structure)
        # ------------------------------------------------------------
        srim_runs_dir = track_dir / "srim_runs"
        if srim_runs_dir.exists():
            theta_dirs = sorted(srim_runs_dir.glob("theta_*"))
        else:
            # backward compatibility
            theta_dirs = sorted(track_dir.glob("theta_*"))

        if strict and not theta_dirs:
            raise RuntimeError(f"No theta_* subfolders under {track_dir}")

        # optional: merge srim_batch_manifest for status
        manifest_status = {}
        manifest_path = track_dir / "srim_batch_manifest.csv"
        if manifest_path.exists():
            try:
                df_manifest = pd.read_csv(manifest_path)
                manifest_status = dict(zip(df_manifest["theta_eV"].astype(float), df_manifest["status"]))
            except Exception:
                print(f"[WARN] Failed to parse manifest for {track_dir}")

        # ------------------------------------------------------------
        # 4️⃣ Loop through theta_* subfolders
        # ------------------------------------------------------------
        for theta_dir in theta_dirs:
            if not theta_dir.is_dir():
                continue

            theta_id, theta_folder_val = _parse_theta_folder(theta_dir)
            try:
                s = summarize_srim_output(str(theta_dir))
            except:
                print(f"[WARN] Missing SRIM outputs under {theta_dir}, skipping.")
                continue


            # get manifest status if available
            theta_val = s.get("theta_eV", None)
            status = manifest_status.get(theta_val, "OK" if theta_val else "UNKNOWN")

            rows.append({
                "track_folder": str(track_dir),
                "theta_folder": str(theta_dir),
                "track_index": track_index,
                "track_id": track_id,
                "ion": ion,
                "energy_keV": energy_keV,
                "composite_key": composite_key,
                "theta_id": theta_id,
                "theta_folder_val": theta_folder_val,
                "theta_file_eV": theta_val,
                "mean_depth_A": s["mean_depth_A"],
                "std_depth_A": s["std_depth_A"],
                "vacancies_per_ion": s["vacancies_per_ion"],
                "status": status,
            })
            n_theta_runs += 1

        n_tracks += 1

    if not rows:
        raise RuntimeError(f"No SRIM outputs found under {output_base}")

    # ------------------------------------------------------------
    # 5️⃣ Build summary DataFrame
    # ------------------------------------------------------------
    x_check = (
        pd.DataFrame(rows)
        .sort_values(["ion", "energy_keV", "track_id", "theta_id"], na_position="last")
        .reset_index(drop=True)
    )

    required = ["mean_depth_A", "std_depth_A", "vacancies_per_ion", "track_id"]
    missing = [c for c in required if c not in x_check.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if "composite_key" not in x_check.columns or x_check["composite_key"].isna().any():
        x_check["composite_key"] = x_check.apply(
            lambda r: _make_composite_key(r.get("ion"), r.get("energy_keV")), axis=1
        )

    if x_check["ion"].isna().any() or x_check["energy_keV"].isna().any():
        bad = x_check.loc[x_check["ion"].isna() | x_check["energy_keV"].isna(),
                          ["track_folder", "track_id", "ion", "energy_keV"]]
        msg = "[ERROR] Missing ion/energy_keV; fix metadata.json files."
        if strict:
            raise RuntimeError(f"{msg}\n{bad.to_string(index=False)}")
        else:
            print(msg)

    # ------------------------------------------------------------
    # 6️⃣ Save to CSV
    # ------------------------------------------------------------
    stamp = label or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(output_base) / f"srim_summary_{stamp}.csv"
    x_check.to_csv(out_csv, index=False)

    print(f"[INFO] Summarized {n_tracks} track(s), {n_theta_runs} theta run(s).")
    print(f"[INFO] Saved SRIM summary DataFrame → {out_csv}")

    return x_check