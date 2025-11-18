# ======================================================
#  DETERMINISTIC POSTERIOR PREDICTIVE CHECK (PPC) PIPELINE
# ======================================================

import torch
import pandas as pd
from pathlib import Path
import os

# --- Import helper functions ---
from src.utils.data_utils import preprocess
from src.utils.srim_utils import pick_tracks_deterministic
from src.utils.srim_parser import summarize_all_runs
from src.utils.analysis_utils import clean_summary_data
from src.utils.analysis_utils import plot_ppc_histograms_per_track
from src.utils.data_utils import plot_ppc_histograms
from src.utils.data_utils import tensor_to_observed_dict

# ------------------------------------------------------
#  STEP 1. SETUP & CONFIG
# ------------------------------------------------------

BASE_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth")
DATA_PATH = BASE_DIR / "Code/SBI/srim-sbi/data/all_vacancies.csv"
OUTPUT_BASE = "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs"

# Where to store PPC results
OUTPUT_DIR_PER_TRACK = BASE_DIR / "Code/SBI/srim-sbi/data/ppc-results/per-track"
OUTPUT_DIR_GLOBAL = BASE_DIR / "Code/SBI/srim-sbi/data/ppc-results/global"

os.makedirs(OUTPUT_DIR_PER_TRACK, exist_ok=True)
os.makedirs(OUTPUT_DIR_GLOBAL, exist_ok=True)

# ------------------------------------------------------
#  STEP 2. PREPROCESS — deterministic track IDs
# ------------------------------------------------------

x_obs, theta, track_ids, grouped, df_summary = preprocess(DATA_PATH)
print(f"[INFO] Preprocessed {len(track_ids)} deterministic tracks.")

# ------------------------------------------------------
#  STEP 3. SELECT TRACKS DETERMINISTICALLY
# ------------------------------------------------------

# Instead of random sampling, just pick first 10 for reproducibility
x_test, x_test_ids, idx = pick_tracks_deterministic(x_obs, track_ids, n=10)
x_test_ids = [331, 976, 2575, 16075, 16708, 21135, 21550, 26559, 29780, 39544]

print(f"[INFO] Deterministically selected tracks: {x_test_ids}")

# ------------------------------------------------------
#  STEP 4. SUMMARIZE SRIM RUNS
# ------------------------------------------------------

df_summary_full = summarize_all_runs(str(OUTPUT_BASE))
df_clean = clean_summary_data(df_summary_full)  # Keep only necessary columns
print(f"[INFO] Cleaned SRIM summary shape: {df_clean.shape}")

print("[DEBUG] Selected x_test_ids:", x_test_ids)
print("[DEBUG] SRIM df_clean track_ids:", sorted(df_clean["track_id"].unique())[:15])
# Verify track_id consistency
available_tracks = sorted(df_clean["track_id"].unique())
print(f"[INFO] Available track_ids in SRIM summary: {available_tracks}")

# ------------------------------------------------------
#  STEP 5. CREATE OBSERVED DICTIONARY
# ------------------------------------------------------
observed_dict = tensor_to_observed_dict(x_test, x_test_ids)

print(f"[INFO] Created observed_dict for {len(observed_dict)} tracks.")
print(list(observed_dict.items())[:2])  # sanity print

# ------------------------------------------------------
#  STEP 6. PER-TRACK PPC
# ------------------------------------------------------

metrics_per_track = plot_ppc_histograms_per_track(
    df=df_clean,
    observed=observed_dict,
    output_dir=str(OUTPUT_DIR_PER_TRACK),
    bins=40,
    save_plots=True,
    return_metrics=True,
)

# ------------------------------------------------------
#  STEP 7. GLOBAL PPC (aggregated across all tracks)
# ------------------------------------------------------

metrics_global = plot_ppc_histograms(
    df=df_clean,
    x_test=x_test,
    x_test_ids=x_test_ids,
    output_dir=str(OUTPUT_DIR_GLOBAL),
    bins=40,
    save_plots=True,
    return_metrics=True,
)

# ------------------------------------------------------
#  STEP 8. SAVE MANIFEST (reproducibility)
# ------------------------------------------------------

manifest = {
    "n_tracks": len(x_test_ids),
    "deterministic": True,
    "bins": 40,
    "data_path": str(DATA_PATH),
    "output_base": str(OUTPUT_BASE),
    "results_dir": str(OUTPUT_DIR_PER_TRACK),
}
manifest_path = BASE_DIR / "Code/SBI/srim-sbi/data/ppc-results/manifest.json"
pd.Series(manifest).to_json(manifest_path)
print(f"[INFO] Saved PPC manifest → {manifest_path}")

print("\n✅ PPC pipeline completed successfully.")
