#!/usr/bin/env python3
"""
Master Controller: Learning Curve Analysis (Pipeline A)
-----------------------------------------------------
Automates the training and evaluation of MNPE models across 
logarithmic dataset sizes to determine the optimal data requirement.

LOGIC:
1. Loads full 200k training set ONCE.
2. Loops through sizes [1k, 5k, 10k, 50k, 100k, 200k].
3. Creates BALANCED subsets (equal number of tracks per angle class).
4. Trains, Saves, Evaluates, and Logs metrics for each size.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import shutil

# --- PROJECT IMPORTS ---
from src.utils.data_utils import preprocess_mdpe
from src.utils.sbi_runner import (
    make_prior,
    make_inference_summary,
    train_summary_posterior
)
from src.evaluation.eval_mdpe import evaluate_from_saved_tracks_mdpe

# ============================================================
# CONFIGURATION
# ============================================================
# Windows-safe paths
BASE_DIR     = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi")
TRAIN_CSV    = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_TRAIN_200k_ROTATED.csv"
EVAL_CSV     =  "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_EVAL_5k_ROTATED.csv"

# Output Directory Structure
EXPERIMENT_NAME = "LearningCurve_Jan29"
RESULTS_ROOT    = BASE_DIR / "data/results" / EXPERIMENT_NAME
POSTERIOR_DIR   = RESULTS_ROOT / "posteriors"
EVAL_DIR        = RESULTS_ROOT / "eval_csvs"

# Metric Logging File
MASTER_LOG_FILE = RESULTS_ROOT / "learning_curve_master_report.csv"

# Training Sizes (Total Samples)
# The script will divide these by 8 to get per-class samples.
SIZES_TO_TEST = [
    1000,    # Tiny
    5000,    # Small
    10000,   # Medium
    50000,   # Large
    100000,  # V. Large
    200000   # Max
]

ENERGY_MIN, ENERGY_MAX = 1.0, 101.0
EVAL_SAMPLES = 500  # Number of posterior samples during evaluation

# ============================================================
# HELPER: BALANCED STRATIFIED SAMPLER
# ============================================================
def get_balanced_subset(x_full, theta_full, total_size):
    """
    Returns a subset of x and theta with exactly (total_size / 8) 
    samples per angular class.
    """
    n_per_class = total_size // 8
    
    # Identify classes (Column 1 of theta is the class index)
    classes = theta_full[:, 1].long()
    
    all_indices = []
    
    print(f"   -> Slicing balanced subset: {total_size} total ({n_per_class} per class)...")
    
    for class_idx in range(8):
        # Find all indices where class is 'class_idx'
        indices = torch.where(classes == class_idx)[0]
        
        # Check if we have enough data
        if len(indices) < n_per_class:
            print(f"[WARN] Class {class_idx} has {len(indices)} samples, but requested {n_per_class}.")
            # Fallback: take all available (this shouldn't happen with synthetic data)
            current_selection = indices
        else:
            # Randomly shuffle and pick first N
            perm = torch.randperm(len(indices))
            current_selection = indices[perm[:n_per_class]]
        
        all_indices.append(current_selection)
    
    # Flatten and convert to tensor
    final_indices = torch.cat(all_indices)
    
    # Shuffle the final batch so classes are mixed during training
    final_indices = final_indices[torch.randperm(len(final_indices))]
    
    return x_full[final_indices], theta_full[final_indices]

# ============================================================
# MAIN LOOP
# ============================================================
# ============================================================
# MAIN LOOP
# ============================================================
def main():
    # 0. SETUP
    t_global_start = time()
    
    # Create directories
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[START] Learning Curve Experiment: {EXPERIMENT_NAME}")
    print(f"Results will be saved to: {RESULTS_ROOT}")

    # 1. LOAD DATA ONCE (The Global Pool)
    print("\n[STEP 1] Loading Full Training Data (200k)...")
    x_full, theta_full, _, _, _ = preprocess_mdpe(TRAIN_CSV)
    print(f"   -> Loaded X: {x_full.shape}, Theta: {theta_full.shape}")

    # Master list to store results
    experiment_results = []

    # 2. EXPERIMENT LOOP
    for size in SIZES_TO_TEST:
        print(f"\n" + "="*60)
        print(f"   CHECKING TRIAL: SIZE {size}")
        print("="*60)
        
        # Define filenames early to check for existence
        post_filename = f"posterior_{size}.pt"
        eval_filename = f"eval_results_{size}.csv"
        
        save_path = POSTERIOR_DIR / post_filename
        eval_path = EVAL_DIR / eval_filename
        
        # --- A. CHECK IF ALREADY DONE ---
        if save_path.exists() and eval_path.exists():
            print(f"   [SKIP] Found existing results for Size {size}. Skipping training.")
            print(f"          Posterior: {post_filename}")
            print(f"          Eval Data: {eval_filename}")
            
            # Load old results so the final report is complete
            try:
                old_df = pd.read_csv(eval_path)
                acc = old_df["class_correct"].mean() * 100
                err = old_df["energy_error_pct"].median()
                
                experiment_results.append({
                    "train_size": size,
                    "accuracy_percent": acc,
                    "median_energy_error": err,
                    "training_time_min": 0.0, # Placeholder
                    "posterior_file": post_filename,
                    "eval_file": eval_filename
                })
            except Exception as e:
                print(f"   [WARN] Could not read old results ({e}).")
            
            # JUMP TO NEXT SIZE
            continue

        # --- B. PREPARE DATA (If not skipped) ---
        if size == len(x_full):
            print("   -> Using Full Dataset.")
            x_train, theta_train = x_full, theta_full
        else:
            x_train, theta_train = get_balanced_subset(x_full, theta_full, size)
        
        # --- C. TRAIN ---
        prior = make_prior(ENERGY_MIN, ENERGY_MAX)
        inference = make_inference_summary(prior)
        
        t0 = time()
        print(f"   -> Training MNPE on {len(x_train)} samples...")
        posterior = train_summary_posterior(inference, theta_train, x_train)
        duration_mins = (time() - t0) / 60.0
        print(f"   -> Training finished in {duration_mins:.2f} min.")
        
        # --- D. SAVE POSTERIOR ---
        torch.save(posterior, save_path)
        print(f"   -> Model saved to: {post_filename}")
        
        # --- E. EVALUATE ---
        print(f"   -> Evaluating on fixed 5k set...")
        
        # Run standard evaluation
        df_eval = evaluate_from_saved_tracks_mdpe(
            posterior=posterior,
            eval_csv=EVAL_CSV,
            output_csv=eval_path,
            max_points=None, # None triggers Summary Feature logic
            n_post_samples=EVAL_SAMPLES
        )
        
        # --- F. LOG METRICS ---
        accuracy = df_eval["class_correct"].mean() * 100
        energy_error = df_eval["energy_error_pct"].median()
        
        print(f"   -> RESULTS: Acc={accuracy:.2f}%, EnergyErr={energy_error:.2f}%")
        
        experiment_results.append({
            "train_size": size,
            "accuracy_percent": accuracy,
            "median_energy_error": energy_error,
            "training_time_min": duration_mins,
            "posterior_file": post_filename,
            "eval_file": eval_filename
        })
        
        # Save intermediate backup of the master log
        pd.DataFrame(experiment_results).to_csv(MASTER_LOG_FILE, index=False)

    # 3. FINISH
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total Runtime: {(time() - t_global_start)/60:.2f} min")
    print(f"Master Report Saved: {MASTER_LOG_FILE}")
    
    # Print Final Table
    if len(experiment_results) > 0:
        print("\nFINAL SUMMARY:")
        print(pd.DataFrame(experiment_results)[["train_size", "accuracy_percent", "median_energy_error", "training_time_min"]])

if __name__ == "__main__":
    main()