import os
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# --- your modules ---
from src.utils.data_utils import preprocess, make_x_test
from src.utils.analysis_utils import plot_ppc_histograms_per_track
from src.utils.sbi_runner import make_prior, make_inference, train_posterior, sample_posterior_bulk
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import summarize_all_runs


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Outputs6"
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
PPC_DIR = RESULTS_DIR / "ppc-results4"


# --------------------------------------------------------------------------
# STEP 1 — Preprocess data
# --------------------------------------------------------------------------
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, grouped, df_summary = preprocess(RAW_CSV)
    print(f"[INFO] x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"[INFO] {len(track_ids)} unique (ion, energy) tracks summarized.")

    # Save raw summary for inspection
    df_summary.to_csv(RESULTS_DIR / "srim_summary_preprocessed.csv", index=False)

    print("\n[DEBUG] ---- Energy Distribution in df_summary ----")
    print(df_summary.groupby("energy_keV").size())
    print(f"[DEBUG] Total tracks: {len(df_summary)}")
    print(f"[DEBUG] Unique energies: {sorted(df_summary['energy_keV'].unique())}")
    print("----------------------------------------------------")

    return x_obs, theta, df_summary


# --------------------------------------------------------------------------
# STEP 2 — Test track selection
# --------------------------------------------------------------------------
def debug_make_x_test(df_summary, n_per_energy=3):
    print("\n[STEP 2] Debugging make_x_test() ...")

    # Before running make_x_test
    print("[DEBUG] Number of unique (ion, energy_keV) pairs before sampling:",
          len(df_summary.groupby(['ion', 'energy_keV'])))

    # Preview the first few groups
    for (ion, energy), group in df_summary.groupby(['ion', 'energy_keV']):
        print(f"[DEBUG] Group: {ion}_{energy}keV, rows={len(group)}")
        if len(group) == 0:
            print(f"  ⚠️ WARNING: Empty group found for {ion}_{energy}keV")

    # Run the sampler
    x_test, x_test_ids = make_x_test(df_summary, n_per_energy=n_per_energy, random_state=42)

    # After sampling
    print("\n[DEBUG] ---- Results from make_x_test() ----")
    print(f"[DEBUG] x_test shape: {x_test.shape}")
    print(f"[DEBUG] Unique energies in x_test: {sorted(x_test['energy_keV'].unique())}")
    print("[DEBUG] Counts per energy in x_test:")
    print(x_test.groupby("energy_keV").size())
    print("----------------------------------------------------")

    # Check for duplicate or missing track IDs
    dup_ids = x_test['track_id'][x_test['track_id'].duplicated()].unique()
    if len(dup_ids) > 0:
        print(f"⚠️ WARNING: Found {len(dup_ids)} duplicate track_ids → {dup_ids}")
    else:
        print("[DEBUG] ✅ No duplicate track_ids detected.")

    # Save debug output
    x_test.to_csv(RESULTS_DIR / "x_test_debug.csv", index=False)
    print(f"[INFO] x_test debug saved → {RESULTS_DIR / 'x_test_debug.csv'}")

    # ----------------------------------------------------------------------
    # 🧩 NEW SECTION: Verify track_id consistency between x_test and df_summary
    # ----------------------------------------------------------------------
    print("\n[STEP 2.1] Checking track_id consistency between x_test and df_summary ...")

    missing_in_summary = set(x_test["track_id"]) - set(df_summary["track_id"])
    missing_in_xtest = set(df_summary["track_id"]) - set(x_test["track_id"])

    print(f"[DEBUG] Tracks in x_test but NOT in df_summary: {len(missing_in_summary)}")
    if missing_in_summary:
        print("  Example missing (first 10):", list(missing_in_summary)[:10])

    print(f"[DEBUG] Tracks in df_summary but NOT in x_test: {len(missing_in_xtest)}")
    if missing_in_xtest:
        print("  Example missing (first 10):", list(missing_in_xtest)[:10])

    if len(missing_in_summary) == 0:
        print("[DEBUG] ✅ All x_test track_ids match df_summary entries perfectly.")
    else:
        print("❌ [ERROR] Some x_test IDs don't exist in df_summary — SRIM will skip these tracks!")

    print("----------------------------------------------------")

    return x_test, x_test_ids


# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == "__main__":
    x_obs, theta, df_summary = preprocess_data()
    x_test, x_test_ids = debug_make_x_test(df_summary, n_per_energy=3)
    run_srim_multi_track(
        samples_dict=samplesdict,
        x_test=x_test,
        track_ids=x_test_ids,
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_summary,   # ✅ critical for correct folder naming
        overwrite=True            # ✅ clean reruns
    )

    print("[DEBUG] Unique energies in df_summary:", sorted(df_summary["energy_keV"].unique()))
    print("[DEBUG] Total energy groups:", df_summary["energy_keV"].nunique())