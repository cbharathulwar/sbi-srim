#!/usr/bin/env python3
"""
Pipeline B: Raw 3D Point Clouds -> PointNet -> Continuous SNPE
--------------------------------------------------------------
1. Loads variable-length 3D tracks (x, y, z), mean-centers them, and zero-pads them.
2. Trains an SNPE (Neural Spline Flow) using a Pure 3D PointNet embedding network.
3. Evaluates inference speed, angular resolution, and energy accuracy.
"""

import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from sbi.utils import BoxUniform

# Ensure 'src' is importable from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- IMPORT PIPELINE B COMPONENTS ---
# 1. Data Loader (Centers and Pads)
from src.utils.data_utils import load_and_pad_3d_tracks

# 2. Architecture & Training Setup
from src.utils.sbi_runner import make_inference_3d_embedding, train_3d_embedding_posterior

# 3. Evaluator
from src.evaluation.eval_mcpe3d import ContinuousEvaluator3D

# Mac/Apple Silicon Optimization
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR      = BASE_DIR / "data/mcpe-3d"
TRAIN_CSV     = DATA_DIR / "mcpe_3d_train.csv"
EVAL_CSV      = DATA_DIR / "mcpe_3d_eval_10k.csv" # Using the 10k subset for faster eval
RESULTS_DIR   = BASE_DIR / "results/pipeline_b"

# Filenames
POSTERIOR_FILE = RESULTS_DIR / "pointnet_3d_posterior.pt"
EVAL_RESULTS   = RESULTS_DIR / "pointnet_3d_eval_results.csv"

# Hyperparameters
FORCE_RETRAIN  = False  # Set to True to overwrite an existing trained model
MAX_POINTS     = "auto" # Automatically calculates 99.5th percentile length
# =================================================

def main():
    print("\n" + "="*50)
    print("🚀 STARTING PIPELINE B: PURE 3D POINT CLOUDS")
    print("="*50)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start_total = time.time()

    # ==========================================
    # STEP 1: LOAD OR TRAIN THE POSTERIOR
    # ==========================================
    if POSTERIOR_FILE.exists() and not FORCE_RETRAIN:
        print(f"\n[STEP 1] Found existing model. Loading -> {POSTERIOR_FILE.name}")
        # map_location ensures we don't crash if moving from GPU to CPU/MPS
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        train_duration = 0.0
    else:
        print("\n[STEP 1] Training New PointNet Posterior...")
        t_train_start = time.time()

        # A. Load and Pad Training Data
        print("\n--- Preparing Training Data ---")
        x_train, theta_train = load_and_pad_3d_tracks(TRAIN_CSV, max_points=MAX_POINTS)

        # B. Define Physics Prior
        # Detect the correct device
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Order must match theta_train: [Energy, Vx, Vy, Vz]
        # ADD device=device to force these tensors onto the GPU
        prior_min = torch.tensor([0.0, -1.0, -1.0, -1.0], device=device)
        prior_max = torch.tensor([105.0, 1.0, 1.0, 1.0], device=device)
        
        prior = BoxUniform(low=prior_min, high=prior_max)
        
        # C. Initialize SBI Inference Pipeline with PointNet
        print("\n--- Initializing PointNet + Neural Spline Flow ---")
        inference = make_inference_3d_embedding(prior, latent_dim=64)

        # D. Train
        print("\n--- Starting SBI Training Loop ---")
        # Pass the raw padded tensors directly into our training wrapper
        posterior = train_3d_embedding_posterior(inference, theta_train, x_train)
        
        # E. Save Model
        train_duration = (time.time() - t_train_start) / 60
        torch.save(posterior, POSTERIOR_FILE)
        print(f"\n[SAVE] Pipeline B Posterior saved to -> {POSTERIOR_FILE}")

    # ==========================================
    # STEP 2: EVALUATION
    # ==========================================
    print("\n[STEP 2] Running Evaluation on Test Set...")
    
    # Instantiate the Continuous Evaluator for 3D
    evaluator = ContinuousEvaluator3D(posterior)
    
    # Run evaluation (loads eval_csv, pads to match, runs inference, calcs physics metrics)
    df_eval_results = evaluator.run_eval(EVAL_CSV, num_samples=100, batch_size=250)
    
    # Save raw results for the plotting script
    if not df_eval_results.empty:
        df_eval_results.to_csv(EVAL_RESULTS, index=False)
        print(f"[SAVE] Evaluation Results saved to -> {EVAL_RESULTS}")
    
    # ==========================================
    # STEP 3: FINAL REPORT
    # ==========================================
    total_time = (time.time() - t_start_total) / 60
    print("\n" + "="*50)
    print("🏆 PIPELINE B FINAL REPORT")
    print("="*50)
    
    if not df_eval_results.empty:
        med_E_err = df_eval_results["energy_error"].median()
        med_A_err = df_eval_results["angular_error_deg"].median()
        r68_A_err = np.percentile(df_eval_results["angular_error_deg"], 68)
        flip_rate = (df_eval_results["angular_error_deg"] > 90).mean() * 100
        
        print(f"Median Energy Error:      {med_E_err:.2f} keV")
        print(f"Median Angular Error:     {med_A_err:.2f}°")
        print(f"68% Containment (R68):    {r68_A_err:.2f}°")
        print(f"Head-Tail Flip Rate:      {flip_rate:.2f}%")
        
    print(f"Training Time:            {train_duration if 'train_duration' in locals() else 0.0:.2f} min")
    print(f"Total Pipeline Time:      {total_time:.2f} min")
    print("="*50 + "\n")
    print("Next step: Run `src/scripts/plot_analysis.py` to generate your paper plots!")

if __name__ == "__main__":
    main()