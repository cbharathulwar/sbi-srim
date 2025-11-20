#!/usr/bin/env python3
"""
MNPE Full Pipeline
------------------
1. Preprocess MNPE training data
2. Train or load posterior
3. Run random-energy SRIM evaluation (energy + parity)
"""

import torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------
# Imports from your new structured modules
# ---------------------------------------------
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import (
    make_mnpe_prior,
    make_mnpe_inference,
    train_mnpe_posterior,
)
from src.evaluation.eval_mnpe import evaluate_random_srims_mnpe

# ---------------------------------------------
# Config
# ---------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nov3srim/centered_tracks.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results")

POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_mnpe.pt"

PRIOR_LOW = 1     # keV
PRIOR_HIGH = 100.0  # keV
N_BINS = 6


# ============================================================
# STEP 1 — Preprocess SRIM training data
# ============================================================
def preprocess_training_data():
    print("\n[STEP 1] Preprocessing training CSV…")
    x_obs, theta, _, _, df_summary = preprocess_mnpe(RAW_CSV, n_bins=N_BINS)
    print(f"[INFO] Loaded {len(theta)} training tracks.")
    return x_obs, theta


# ============================================================
# STEP 2 — Train or load MNPE posterior
# ============================================================
def train_or_load_posterior(x_obs, theta):
    if POSTERIOR_FILE.exists():
        print(f"[STEP 2] Loading existing posterior → {POSTERIOR_FILE}")
        return torch.load(POSTERIOR_FILE, map_location="cpu")

    print("[STEP 2] Training MNPE posterior…")

    # Build prior + inference object
    prior = make_mnpe_prior(PRIOR_LOW, PRIOR_HIGH)
    inf = make_mnpe_inference(prior)

    # Train MNPE (energy + parity)
    posterior = train_mnpe_posterior(inf, theta, x_obs)

    # Save
    torch.save(posterior, POSTERIOR_FILE)
    print(f"[INFO] Saved posterior → {POSTERIOR_FILE}")

    return posterior


# ============================================================
# MASTER PIPELINE
# ============================================================
def run_pipeline():
    start = datetime.now()
    print(f"[INIT] MNPE pipeline started @ {start:%Y-%m-%d %H:%M:%S}")

    # 1. Preprocess training data
    x_obs, theta = preprocess_training_data()

    # 2. Train/load posterior
    posterior = train_or_load_posterior(x_obs, theta)

    # 3. Random SRIM evaluation
    output_root = RESULTS_DIR / "Nov19Run1"
    output_csv  = RESULTS_DIR / "Nov19_mnpe.csv"

    evaluate_random_srims_mnpe(
        posterior=posterior,
        srim_dir=SRIM_DIR,
        output_root=output_root,
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        n_random=250,
        n_ions=200,
        n_post_samples=500,
        n_bins=N_BINS,
        save_csv=output_csv,
        sample_mode="continuous",   # or "grid" or "biased_low"
        step=0.1,                   # only needed for grid mode
    )

    end = datetime.now()
    print(f"\n[DONE] Finished in {(end - start).total_seconds()/60:.2f} min")


# ============================================================
if __name__ == "__main__":
    run_pipeline()