#!/usr/bin/env python3
"""
Pipeline A: Summary Features (Physics Model)
--------------------------------------------
1. Preprocesses tracks into 26D physics vectors (Energy, Skewness, Bragg Ratio, etc.)
2. Trains an MNPE (Mixed Neural Posterior) for Continuous Energy + Discrete Angle.
3. Evaluates on the rotated test set.
"""

from pathlib import Path
from time import time
import torch
import os

# Specific Imports for Pipeline A
from src.utils.data_utils import preprocess_mdpe
from src.utils.sbi_runner import (
    make_prior,                 # Shared Prior
    make_inference_summary,     # MNPE Engine (Fixed!)
    train_summary_posterior     # Training Loop (Fixed!)
)
from src.evaluation.eval_mdpe import evaluate_from_saved_tracks_mdpe

# Mac Optimization
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================= CONFIGURATION =================
BASE_DIR     = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi")
TRAIN_CSV    = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_TRAIN_200k_ROTATED.csv"
EVAL_CSV     =  "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_EVAL_5k_ROTATED.csv"
RESULTS_DIR  = BASE_DIR / "data/results"

# Filenames specific to Pipeline A
POSTERIOR_FILE = RESULTS_DIR / "mdpe_summary_posterior.pt"
EVAL_RESULTS   = RESULTS_DIR / "Jan21_SummaryFeatures_eval.csv"

ENERGY_MIN, ENERGY_MAX = 1.0, 101.0
N_POST_SAMPLES = 500
# =================================================

def main():
    t_start = time()
    print(f"\n[START] Pipeline A: SUMMARY FEATURES (Physics 26D)")

    # 1. LOAD / TRAIN
    if POSTERIOR_FILE.exists():
        print(f"\n[STEP 1] Loading existing posterior -> {POSTERIOR_FILE}")
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        train_duration = 0.0
    else:
        print("\n[STEP 1] Training New Posterior...")
        t_train_start = time()

        # Calculate Summary Features
        print("[DATA] Preprocessing training data (26D Vectors)...")
        # preprocess_mdpe returns (x_obs, theta, ...)
        x_train, theta_train, _, _, _ = preprocess_mdpe(TRAIN_CSV)
        print(f"[DATA] Feature Vector Size: {x_train.shape[1]}")

        # Train MNPE
        prior = make_prior(ENERGY_MIN, ENERGY_MAX)
        inference = make_inference_summary(prior)
        posterior = train_summary_posterior(inference, theta_train, x_train)

        train_duration = (time() - t_train_start) / 60
        
        POSTERIOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(posterior, POSTERIOR_FILE)
        print(f"[SAVE] Posterior saved -> {POSTERIOR_FILE}")

    # 2. EVALUATE
    print("\n[STEP 2] Evaluating...")
    # max_points=None triggers the "Summary Feature" logic in the evaluator
    df_eval = evaluate_from_saved_tracks_mdpe(
        posterior=posterior,
        eval_csv=EVAL_CSV,
        output_csv=EVAL_RESULTS,
        max_points=None, 
        n_post_samples=N_POST_SAMPLES
    )

    # 3. REPORT
    print("\n==================== RESULTS (Pipeline A) ====================")
    if len(df_eval) > 0:
        acc = df_eval["class_correct"].mean() * 100
        err = df_eval["energy_error_pct"].median()
        print(f"Direction Accuracy:       {acc:.2f}%")
        print(f"Median Energy Error:      {err:.2f}%")
    
    print(f"Training Time:            {train_duration:.2f} min")
    print(f"Saved to:                 {EVAL_RESULTS}")
    print("============================================================\n")

if __name__ == "__main__":
    main()