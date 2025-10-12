#!/usr/bin/env python3
"""
SRIM–SBI Full Pipeline Runner
=============================
End-to-end workflow:
    1. Preprocess SRIM raw CSV → x_obs
    2. Train SBI posterior (NSF)
    3. Sample posterior for x_test subset
    4. Run SRIM for predicted thetas
    5. Summarize SRIM outputs → x_check
    6. Perform PPC (global + per-track)
"""

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
# Configuration
# --------------------------------------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Outputs8"
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
PPC_DIR = RESULTS_DIR / "ppc-results6"

ION_SYMBOL = "C"
N_IONS = 20
N_XTEST = 1         # number of unique test tracks (per energy)
SAMPLES_PER_TRACK = 25
PRIOR_LOW, PRIOR_HIGH = [1_000], [2_000_000]
POSTERIOR_FILE = RESULTS_DIR / "trained_posterior.pt"

# --------------------------------------------------------------------------
# 1. Data preprocessing
# --------------------------------------------------------------------------
def preprocess_data():
    print("[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, grouped, df_summary = preprocess(RAW_CSV)
    print(f"[INFO] x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"[INFO] {len(track_ids)} unique (ion, energy) tracks summarized.")
    df_summary.to_csv(RESULTS_DIR / "srim_summary_preprocessed.csv", index=False)
    return x_obs, theta, df_summary

# --------------------------------------------------------------------------
# 2. Train SBI posterior
# --------------------------------------------------------------------------
def train_sbi(x_obs, theta, save_path=POSTERIOR_FILE):
    print("[STEP 2] Training SBI posterior ...")
    prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)
    inference = make_inference(prior, density_estimator="nsf")
    posterior = train_posterior(inference, theta, x_obs)
    torch.save(posterior, save_path)
    print(f"[INFO] Posterior saved → {save_path}")
    return posterior

# --------------------------------------------------------------------------
# 3. Sample posterior and run SRIM
# --------------------------------------------------------------------------
def run_srim_for_posterior(posterior, df_summary):
    print("[STEP 3] Selecting test tracks and running SRIM ...")

    # Choose test subset (balanced per energy)
    x_test, _ = make_x_test(df_summary, n_per_energy=5)
    x_test.to_csv(RESULTS_DIR / "x_test_selection.csv", index=False)
    track_ids = x_test["track_id"].tolist()

    print(f"[INFO] Selected {len(track_ids)} tracks for SRIM inference:")
    for _, row in x_test.iterrows():
        print(f"    → Track {row.track_id} ({row.ion}, {row.energy_keV:.1f} keV)")

    # Sample posterior for each track
    x_tensor = torch.tensor(
        x_test[["mean_depth_A", "std_depth_A", "vacancies_per_ion"]].values,
        dtype=torch.float32
    )
    samples_dict, _ = sample_posterior_bulk(
        posterior=posterior,
        x_obs=x_tensor,
        num_samples=SAMPLES_PER_TRACK,
        track_ids=track_ids
    )

    # Run SRIM
    run_srim_multi_track(
        samples_dict=samples_dict,
        x_test=x_test,
        track_ids=track_ids,
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_summary,   # ✅ critical for correct folder naming
        overwrite=True            # ✅ clean reruns
    )

    print("[INFO] SRIM batch runs complete.")
    return x_test, track_ids

# --------------------------------------------------------------------------
# 4. Summarize SRIM outputs
# --------------------------------------------------------------------------
def summarize_srim_outputs():
    print("[STEP 4] Summarizing SRIM outputs ...")
    x_check = summarize_all_runs(str(OUTPUT_BASE), label=datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[INFO] x_check shape: {x_check.shape}")
    x_check.to_csv(RESULTS_DIR / "x_check_summary.csv", index=False)
    return x_check

# --------------------------------------------------------------------------
# 5. Posterior Predictive Check (PPC)
# --------------------------------------------------------------------------
def run_ppc(x_check, x_test):
    print("[STEP 5] Running PPC ...")
    os.makedirs(PPC_DIR, exist_ok=True)

    # Build observed dictionary
    observed = {
        str(row.track_id): {
            "mean_depth_A": float(row.mean_depth_A),
            "std_depth_A": float(row.std_depth_A),
            "vacancies_per_ion": float(row.vacancies_per_ion)
        }
        for _, row in x_test.iterrows()
    }

    # Per-track PPC
    plot_ppc_histograms_per_track(
        df=x_check,
        observed=observed,
        output_dir=str(PPC_DIR / "per_track"),
        bins=30,
        save_plots=True,
        return_metrics=True,
    )

    print("[INFO] PPC complete.")

# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------
def train_pipeline():
    t0 = datetime.now()
    print(f"[INIT] SRIM–SBI training pipeline started @ {t0:%Y-%m-%d %H:%M:%S}")

    x_obs, theta, df_summary = preprocess_data()
    print(df_summary.groupby(["ion", "energy_keV"]).size())

    if POSTERIOR_FILE.exists():
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        print(f"[INFO] Loaded existing posterior from {POSTERIOR_FILE}")
    else:
        posterior = train_sbi(x_obs, theta, save_path=POSTERIOR_FILE)

    x_test, track_ids = run_srim_for_posterior(posterior, df_summary)
    x_check = summarize_srim_outputs()
    run_ppc(x_check, x_test)

    print(f"[DONE] Pipeline finished in {(datetime.now()-t0).total_seconds():.1f}s")
    print(df_summary.groupby(["ion", "energy_keV"]).size())

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    train_pipeline()