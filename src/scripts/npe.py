#!/usr/bin/env python3
"""
NPE Full Pipeline (Energy-Only)
-------------------------------
1. Preprocess NPE training data
2. Train or load NPE posterior
3. Run random-energy SRIM evaluation (energy only)
"""

import torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------
# Imports from your new structured modules
# ---------------------------------------------
from src.utils.data_utils import preprocess_npe
from src.utils.sbi_runner import (
    make_npe_prior,
    make_npe_inference,
    train_npe_posterior,
)
from src.evaluation.random_eval_npe import evaluate_random_srims_npe

# ---------------------------------------------
# Config
# ---------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nov3srim/centered_tracks.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")

POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_npe.pt"

PRIOR_LOW = 1.0     # keV
PRIOR_HIGH = 100.0  # keV
N_BINS = 6


# ============================================================
# STEP 1 — Preprocess SRIM training data (energy-only)
# ============================================================
def preprocess_training_data():
    print("\n[STEP 1] Preprocessing training CSV for NPE…")
    x_obs, theta, _, _, df_summary = preprocess_npe(RAW_CSV, n_bins=N_BINS)
    print(f"[INFO] Loaded {len(theta)} training tracks.")
    return x_obs, theta


# ============================================================
# STEP 2 — Train or load NPE posterior
# ============================================================
def train_or_load_posterior(x_obs, theta):
    if POSTERIOR_FILE.exists():
        print(f"[STEP 2] Loading existing NPE posterior → {POSTERIOR_FILE}")
        return torch.load(POSTERIOR_FILE, map_location="cpu")

    print("[STEP 2] Training new NPE posterior…")

    # Build prior (energy only)
    prior = make_npe_prior(low=[PRIOR_LOW], high=[PRIOR_HIGH])
    inf = make_npe_inference(prior, density_estimator="nsf")

    posterior = train_npe_posterior(inf, theta, x_obs)

    # Save
    torch.save(posterior, POSTERIOR_FILE)
    print(f"[INFO] Saved NPE posterior → {POSTERIOR_FILE}")

    return posterior


# ============================================================
# MASTER PIPELINE
# ============================================================
def run_pipeline():
    start = datetime.now()
    print(f"[INIT] NPE pipeline started @ {start:%Y-%m-%d %H:%M:%S}")

    # 1. Preprocess training data
    x_obs, theta = preprocess_training_data()

    # 2. Train/load posterior
    posterior = train_or_load_posterior(x_obs, theta)

    # 3. Random SRIM evaluation
    output_root = RESULTS_DIR / "random_eval_npe"
    output_csv  = RESULTS_DIR / "random_eval_results_npe.csv"

    evaluate_random_srims_npe(
        posterior=posterior,
        srim_dir=SRIM_DIR,
        output_root=output_root,
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        n_random=100,
        n_ions=200,
        n_post_samples=500,
        n_bins=N_BINS,
        save_csv=output_csv,
    )

    end = datetime.now()
    print(f"\n[DONE] NPE pipeline finished in {(end - start).total_seconds()/60:.2f} min")


# ============================================================
if __name__ == "__main__":
    run_pipeline()