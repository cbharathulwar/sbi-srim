#!/usr/bin/env python3
"""
MNPE Full Pipeline
------------------
1. Preprocess MNPE training data
2. Train or load posterior
3. (OPTIONAL) Run FIXED-energy SRIM evaluation (generate eval dataset)
4. Evaluate posterior from SAVED per-track CSVs (no SRIM)
"""

import torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------
# Module imports
# ---------------------------------------------
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import (
    make_mnpe_prior,
    make_mnpe_inference,
    train_mnpe_posterior,
)

from src.evaluation.eval_mnpe import (
    evaluate_fixed_energies_mnpe,
    set_confidence_threshold,
)

from src.evaluation.eval_mnpe import (
    evaluate_from_saved_tracks_mnpe,
)

# ---------------------------------------------
# Config
# ---------------------------------------------
RAW_CSV = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/centered_tracks_plus_failed_all.csv"
)

SRIM_DIR = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013"
)

RESULTS_DIR = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results"
)

POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_mnpe.pt"

PRIOR_LOW = 1.0
PRIOR_HIGH = 120.0
N_BINS = 6

# Energies
FIXED_ENERGIES = list(range(1, 101))

# ---------------------------------------------
# TOGGLES (THIS IS THE KEY)
# ---------------------------------------------
RUN_SRIM_EVAL = False        # set False once eval dataset exists
RUN_SAVED_EVAL = True      # evaluate from saved tracks

CONFIDENCE_THRESHOLD = 0.8

# ---------------------------------------------
# Paths
# ---------------------------------------------
SRIM_EVAL_ROOT = RESULTS_DIR / "Dec11_HNMOvernightFixedEnergyEval"
# SRIM_EVAL_CSV  = RESULTS_DIR / "Dec12_HNMAllTracksfixed_energy_eval.csv"

SAVED_EVAL_CSV = RESULTS_DIR / "Dec12_HNMFull_eval_from_saved.csv"


# ============================================================
# STEP 1 — Preprocess MNPE training data
# ============================================================
def preprocess_training_data():
    print("\n[STEP 1] Preprocessing training CSV…")
    x_obs, theta, _, _, _ = preprocess_mnpe(RAW_CSV, n_bins=N_BINS)
    print(f"[INFO] Loaded {len(theta)} training tracks.")
    return x_obs, theta


# ============================================================
# STEP 2 — Train or load posterior
# ============================================================
def train_or_load_posterior(x_obs, theta):
    if POSTERIOR_FILE.exists():
        print(f"[STEP 2] Loading posterior → {POSTERIOR_FILE}")
        return torch.load(POSTERIOR_FILE, map_location="cpu")

    print("[STEP 2] Training MNPE posterior…")

    prior = make_mnpe_prior(PRIOR_LOW, PRIOR_HIGH)
    inf = make_mnpe_inference(prior)

    posterior = train_mnpe_posterior(inf, theta, x_obs)

    torch.save(posterior, POSTERIOR_FILE)
    print(f"[INFO] Saved posterior → {POSTERIOR_FILE}")

    return posterior


# ============================================================
# MASTER PIPELINE
# ============================================================
def run_pipeline():
    start = datetime.now()
    print(f"[INIT] MNPE pipeline started @ {start:%Y-%m-%d %H:%M:%S}")

    # --------------------------------------------------------
    # Step 1 — Training data
    # --------------------------------------------------------
    x_obs, theta = preprocess_training_data()

    # --------------------------------------------------------
    # Step 2 — Posterior
    # --------------------------------------------------------
    posterior = train_or_load_posterior(x_obs, theta)

    set_confidence_threshold(CONFIDENCE_THRESHOLD)

    # --------------------------------------------------------
    # Step 3 — Generate evaluation dataset (SRIM)
    # --------------------------------------------------------
    if RUN_SRIM_EVAL:
        print("\n[STEP 3] Generating SRIM evaluation dataset...")

        evaluate_fixed_energies_mnpe(
            posterior=posterior,
            srim_dir=SRIM_DIR,
            output_root=SRIM_EVAL_ROOT,
            energies_keV=FIXED_ENERGIES,
            n_ions=600,
            n_post_samples=500,
            n_bins=N_BINS,
            save_csv=SRIM_EVAL_CSV,
        )

    else:
        print("\n[STEP 3] Skipped SRIM eval (dataset already exists)")

    # --------------------------------------------------------
    # Step 4 — Evaluate from saved tracks (NO SRIM)
    # --------------------------------------------------------
    if RUN_SAVED_EVAL:
        print("\n[STEP 4] Evaluating from saved per-track CSVs...")

        evaluate_from_saved_tracks_mnpe(
            posterior=posterior,
            saved_tracks_root=SRIM_EVAL_ROOT,
            output_csv=SAVED_EVAL_CSV,
            n_bins=N_BINS,
            n_post_samples=500,
            use_observer=False,
        )

    end = datetime.now()
    print(f"\n[DONE] Finished in {(end - start).total_seconds()/60:.2f} min")


# ============================================================
if __name__ == "__main__":
    run_pipeline()