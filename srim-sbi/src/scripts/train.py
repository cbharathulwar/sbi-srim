#!/usr/bin/env python3
"""
SRIM–SBI Full Pipeline Runner (Optimized)
=========================================
End-to-end workflow:
    1. Preprocess SRIM raw CSV → x_obs
    2. Train SBI posterior (NSF)
    3. Sample posterior for x_test subset (balanced across energies)
    4. Run SRIM for predicted θ samples
    5. Summarize SRIM outputs → x_check
    6. Perform PPC (global + per-track)
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# --- Project modules ---
from src.utils.data_utils import preprocess, make_x_test
from src.utils.analysis_utils import plot_ppc_histograms_per_track
from src.utils.sbi_runner import make_prior, make_inference, train_posterior, sample_posterior_bulk
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import summarize_all_runs

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Outputs_Final"
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
PPC_DIR = RESULTS_DIR / "ppc-results-final"

ION_SYMBOL = "C"
N_IONS = 50                # increase for realism
SAMPLES_PER_TRACK = 40     # posterior samples per track
N_PER_ENERGY = 3           # test tracks per energy

PRIOR_LOW, PRIOR_HIGH = [1_000], [2_000_000]
POSTERIOR_FILE = RESULTS_DIR / "trained_posterior.pt"

# --------------------------------------------------------------------------
# 1. Preprocess SRIM CSV data
# --------------------------------------------------------------------------
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, grouped, df_summary = preprocess(RAW_CSV)
    print(f"[INFO] x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"[INFO] {len(track_ids)} unique (ion, energy) tracks summarized.")

    df_summary.to_csv(RESULTS_DIR / "srim_summary_preprocessed.csv", index=False)
    print(f"[INFO] Saved summary → {RESULTS_DIR / 'srim_summary_preprocessed.csv'}")

    print("[DEBUG] ---- Energy Distribution ----")
    print(df_summary.groupby(["ion", "energy_keV"]).size())
    print(f"[DEBUG] Total tracks: {len(df_summary)}")
    print("----------------------------------------------------")

    return x_obs, theta, df_summary

# --------------------------------------------------------------------------
# 2. Train or load posterior
# --------------------------------------------------------------------------
def train_or_load_posterior(x_obs, theta, save_path=POSTERIOR_FILE):
    if save_path.exists():
        posterior = torch.load(save_path, map_location="cpu")
        print(f"[STEP 2] ✅ Loaded existing posterior from {save_path}")
    else:
        print("[STEP 2] Training new SBI posterior ...")
        prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)
        inference = make_inference(prior, density_estimator="nsf")
        posterior = train_posterior(inference, theta, x_obs)
        torch.save(posterior, save_path)
        print(f"[INFO] Posterior saved → {save_path}")
    return posterior

# --------------------------------------------------------------------------
# 3. Sample posterior & run SRIM
# --------------------------------------------------------------------------
def sample_and_run_srim(posterior, df_summary):
    print("\n[STEP 3] Selecting test tracks & running SRIM ...")

    # Balanced selection
    x_test, _ = make_x_test(df_summary, n_per_energy=N_PER_ENERGY, random_state=42)
    x_test.to_csv(RESULTS_DIR / "x_test_selection.csv", index=False)
    track_ids = x_test["track_id"].tolist()

    print(f"[INFO] Selected {len(track_ids)} tracks ({N_PER_ENERGY} per energy level).")
    print("[DEBUG] Energies represented:", sorted(x_test['energy_keV'].unique()))

    # Sample posterior
    feats = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    x_tensor = torch.tensor(x_test[feats].values, dtype=torch.float32)

    print(f"[DEBUG] Sampling posterior ({SAMPLES_PER_TRACK} samples per track) ...")
    samples_dict, _ = sample_posterior_bulk(
        posterior=posterior,
        x_obs=x_tensor,
        num_samples=SAMPLES_PER_TRACK,
        track_ids=track_ids
    )

    # Verify coverage
    missing = set(track_ids) - set(samples_dict.keys())
    assert not missing, f"[ASSERT] Missing posterior samples for tracks: {missing}"

    # Run SRIM (single pass, no double batch)
    print(f"\n[INFO] Running SRIM for {len(track_ids)} tracks × {SAMPLES_PER_TRACK} θ samples ...")
    run_srim_multi_track(
        samples_dict=samples_dict,
        x_test=x_test,
        track_ids=track_ids,
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_summary,
        overwrite=True
    )

    print("[STEP 3] ✅ SRIM runs complete.")
    return x_test, track_ids

# --------------------------------------------------------------------------
# 4. Summarize SRIM outputs
# --------------------------------------------------------------------------
def summarize_outputs():
    print("\n[STEP 4] Summarizing SRIM outputs ...")
    x_check = summarize_all_runs(str(OUTPUT_BASE), label=datetime.now().strftime("%Y%m%d_%H%M%S"))
    x_check.to_csv(RESULTS_DIR / "x_check_summary.csv", index=False)
    print(f"[INFO] x_check summary saved → {RESULTS_DIR / 'x_check_summary.csv'}")
    return x_check

# --------------------------------------------------------------------------
# 5. Posterior Predictive Check (PPC)
# --------------------------------------------------------------------------
def run_ppc(x_check, x_test):
    print("\n[STEP 5] Running Posterior Predictive Check (PPC) ...")
    os.makedirs(PPC_DIR, exist_ok=True)

    observed = {
        str(row.track_id): {
            "mean_depth_A": float(row.mean_depth_A),
            "std_depth_A": float(row.std_depth_A),
            "vacancies_per_ion": float(row.vacancies_per_ion)
        }
        for _, row in x_test.iterrows()
    }

    plot_ppc_histograms_per_track(
        df=x_check,
        observed=observed,
        output_dir=str(PPC_DIR / "per_track"),
        bins=30,
        save_plots=True,
        return_metrics=True,
    )
    print("[INFO] ✅ PPC complete.")

# --------------------------------------------------------------------------
# 6. Master run
# --------------------------------------------------------------------------
def run_pipeline():
    start_time = datetime.now()
    print(f"[INIT] SRIM–SBI pipeline started @ {start_time:%Y-%m-%d %H:%M:%S}")

    # Steps
    x_obs, theta, df_summary = preprocess_data()
    posterior = train_or_load_posterior(x_obs, theta)
    x_test, track_ids = sample_and_run_srim(posterior, df_summary)
    x_check = summarize_outputs()
    run_ppc(x_check, x_test)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[DONE] ✅ Full pipeline completed in {elapsed/60:.1f} min")
    print(f"[DEBUG] Tracks processed: {len(track_ids)}")

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()