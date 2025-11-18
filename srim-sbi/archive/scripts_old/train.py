#!/usr/bin/env python3
"""
SRIM–SBI Random Evaluation Pipeline
===================================

This pipeline does:
    ✅ Preprocess existing SRIM CSV (for training data)
    ✅ Train or load SBI posterior
    ✅ (Optional) Evaluate on EXISTING tracks (old method)
    ✅ (Optional & Recommended) Evaluate on RANDOM SRIM energies (Daniel's method)

You can toggle behavior using:
    USE_EXISTING_DATA = True / False
    USE_RANDOM_SRIM = True / False
"""

import os
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# Toggle modes
USE_EXISTING_DATA = False   # old way using sample_posterior_only()
USE_RANDOM_SRIM   = True    # new random-energy SRIM pipeline

# ----------------------------------------------------------------------------
# Import project modules
# ----------------------------------------------------------------------------
from src.utils.data_utils import preprocess
from src.utils.sbi_runner import (
    make_prior,
    make_inference,
    train_posterior,
)
from src.utils.random_eval import evaluate_multiple_random_energies   # ✅ you added this!
from src.utils.data_utils import make_x_test                         # old evaluation
from src.utils.sbi_runner import sample_posterior_bulk
from src.utils.hist import run_scalar_ppc, run_shape_ppc

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nov3srim/vacancies.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")

POSTERIOR_FILE = RESULTS_DIR / "trained_posterior.pt"
SAMPLES_PER_TRACK = 5000
N_BINS = 6
PRIOR_LOW, PRIOR_HIGH = [0.5], [105]  # keV energy range

# ----------------------------------------------------------------------------
# STEP 1: Preprocess Training Data
# ----------------------------------------------------------------------------
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, _, df_summary = preprocess(RAW_CSV, n_bins=N_BINS)
    out_csv = RESULTS_DIR / "srim_summary_preprocessed.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"[INFO] Saved summary → {out_csv}")
    return x_obs, theta, df_summary

# ----------------------------------------------------------------------------
# STEP 2: Train or Load Posterior
# ----------------------------------------------------------------------------
def train_or_load_posterior(x_obs, theta):
    if POSTERIOR_FILE.exists():
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        print(f"[STEP 2] ✅ Loaded existing posterior from {POSTERIOR_FILE}")
    else:
        print("[STEP 2] Training new posterior ...")
        prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)
        inference = make_inference(prior, density_estimator="nsf")
        posterior = train_posterior(inference, theta, x_obs)
        torch.save(posterior, POSTERIOR_FILE)
        print(f"[INFO] Posterior saved → {POSTERIOR_FILE}")
    return posterior

# ----------------------------------------------------------------------------
# OLD EVALUATION (OPTIONAL)
# ----------------------------------------------------------------------------
def evaluate_existing_data(posterior, df_summary, n_per_energy=3):
    """
    Uses old method: sample from existing SRIM data (no new simulations).
    """
    from src.utils.sbi_runner import sample_posterior_bulk
    import numpy as np

    print("\n[OLD MODE] Evaluating using existing SRIM CSV ...")

    x_test, _ = make_x_test(df_summary, n_per_energy=n_per_energy, random_state=42)
    feats = ["mean_depth_A", "max_depth_A", "vacancies_per_ion"] + [f"rbin_frac_{i}" for i in range(1, N_BINS + 1)]
    feats = [f for f in feats if f in x_test.columns]

    x_tensor = torch.tensor(x_test[feats].values, dtype=torch.float32)
    track_ids = x_test["track_id"].tolist()

    samples_dict, _ = sample_posterior_bulk(posterior, x_tensor, num_samples=SAMPLES_PER_TRACK, track_ids=track_ids)

    rows = []
    for tid in track_ids:
        true_e = x_test.loc[x_test["track_id"] == tid, "energy_keV"].values[0]
        samples = samples_dict[tid]
        mean_e = samples.mean()
        std_e = samples.std()
        rows.append({"track_id": tid, "true_energy_keV": true_e, "posterior_mean_keV": mean_e, "posterior_std_keV": std_e})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS_DIR / "existing_eval.csv", index=False)
    print(f"[INFO] Saved → {RESULTS_DIR / 'existing_eval.csv'}")
    return df_out

# ----------------------------------------------------------------------------
# Master Run
# ----------------------------------------------------------------------------
def run_pipeline():
    start = datetime.now()
    print(f"[INIT] Pipeline started @ {start:%Y-%m-%d %H:%M:%S}")

    # Step 1
    x_obs, theta, df_summary = preprocess_data()

    # Step 2
    posterior = train_or_load_posterior(x_obs, theta)

    # Step 3: OLD vs NEW Evaluation
    if USE_EXISTING_DATA:
        evaluate_existing_data(posterior, df_summary, n_per_energy=3)

    if USE_RANDOM_SRIM:
        print("\n[STEP 3B] Running RANDOM ENERGY SRIM evaluation ...")
        evaluate_multiple_random_energies(
            posterior=posterior,
            srim_dir=SRIM_DIR,
            output_root=RESULTS_DIR / "random_eval_outputs_big",
            n_random=200,          # how many random energies to test
            n_ions=200,           # how many SRIM ions per energy
            n_post_samples=5000,  # posterior samples per track
            n_bins=N_BINS,
            save_csv=RESULTS_DIR / "random_energy_eval_big.csv"
        )

    end = datetime.now()
    print(f"\n[DONE] Pipeline finished in {(end - start).total_seconds()/60:.1f} min")

# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()