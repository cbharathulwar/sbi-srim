#!/usr/bin/env python3
"""
Pipeline: MCPE-3D (Continuous Physics Model)
--------------------------------------------
1. Preprocesses 3D tracks into "Overkill" feature vectors.
2. Trains an SNPE (Sequential Neural Posterior Estimator) to predict [Energy, Vx, Vy, Vz].
3. Evaluates angular resolution and energy accuracy.
"""

from pathlib import Path
from time import time
import torch
import numpy as np
import pandas as pd
import os
import sys

# Ensure 'src' is importable
# Assumes this script is in src/scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Specific Imports for MCPE-3D
from src.utils.data_utils import preprocess_mcpe_3d
from src.evaluation.eval_mcpe import ContinuousEvaluator3D

# SBI Imports
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi import utils as utils

# Mac Optimization
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR      = BASE_DIR / "data/mcpe3d"
TRAIN_CSV     = DATA_DIR / "mcpe_3d_train.csv"
EVAL_CSV      = DATA_DIR / "mcpe_3d_eval_10k.csv"
RESULTS_DIR   = BASE_DIR / "results/mcpe_3d"

# Filenames
POSTERIOR_FILE = RESULTS_DIR / "mcpe_3d_posterior.pt"
EVAL_RESULTS   = RESULTS_DIR / "mcpe_3d_eval_results2.csv"

# Hyperparameters
FORCE_RETRAIN  = False  # Set True to overwrite existing model
# =================================================

def main():
    t_start = time()
    print(f"\n[START] Pipeline: MCPE-3D (Directional Regression)")
    print(f"        Train Data: {TRAIN_CSV.name}")
    print(f"        Eval Data:  {EVAL_CSV.name}")
    
    # Create Results Directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD / TRAIN
    if POSTERIOR_FILE.exists() and not FORCE_RETRAIN:
        print(f"\n[STEP 1] Loading existing posterior -> {POSTERIOR_FILE.name}")
        # map_location ensures CPU compatibility if moved from GPU
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        train_duration = 0.0
    else:
        print("\n[STEP 1] Training New Posterior (SBI Auto-Tuning)...")
        t_train_start = time()

        # A. Preprocessing
        print("[DATA] Preprocessing training data (Extracting 3D Features)...")
        train_raw = pd.read_csv(TRAIN_CSV)
        df_features = preprocess_mcpe_3d(train_raw)
        
        # B. Split into Theta (Targets) and X (Context)
        # Targets: [Energy, Vx, Vy, Vz]
        target_cols = ["energy_true", "vx_true", "vy_true", "vz_true"]
        drop_cols = target_cols + ["run_id", "ion_number"] # Don't train on IDs
        
        # Filter columns that exist
        feature_cols = [c for c in df_features.columns if c not in drop_cols]
        
        # Create Tensors
        theta_train = torch.tensor(df_features[target_cols].values, dtype=torch.float32)
        x_train = torch.tensor(df_features[feature_cols].values, dtype=torch.float32)

        print(f"[DATA] Context (Features) Size: {x_train.shape[1]} dims")
        print(f"[DATA] Theta (Params) Size:    {theta_train.shape[1]} dims [E, Vx, Vy, Vz]")

        # C. Define Prior
        # Energy: 0-100 keV (Soft bounds)
        # Vectors: -1 to 1 (Hard bounds)
        # Order must match target_cols: [E, Vx, Vy, Vz]
        prior_min = torch.tensor([0.0, -1.0, -1.0, -1.0])
        prior_max = torch.tensor([105.0, 1.0, 1.0, 1.0]) # Slight buffer on energy
        
        prior = BoxUniform(low=prior_min, high=prior_max)

        # D. Configure Network & Inference
        print("[MODEL] Initializing SNPE with Neural Spline Flow (NSF)...")
        # We use Neural Spline Flow (NSF) as it handles complex multimodal posteriors well
        inference = SNPE(prior=prior, density_estimator="nsf")
        
        # Append Data
        inference.append_simulations(theta_train, x_train)
        
        # E. Train
        print(f"[TRAIN] Starting SBI training loop...")
        density_estimator = inference.train(
            training_batch_size=512, 
            learning_rate=5e-4,
            validation_fraction=0.1,
            stop_after_epochs=20,
            show_train_summary=True
        )

        # F. Build & Save Posterior
        posterior = inference.build_posterior(density_estimator)
        train_duration = (time() - t_train_start) / 60
        
        torch.save(posterior, POSTERIOR_FILE)
        print(f"[SAVE] Posterior saved -> {POSTERIOR_FILE}")

    # 2. EVALUATE
    print("\n[STEP 2] Evaluating...")
    
    # Instantiate the 3D evaluator
    # (Assumes GPU/MPS availability if possible)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    evaluator = ContinuousEvaluator3D(posterior, device=device)
    
    # Run evaluation logic (100 samples per track is a good balance)
    df_eval_results = evaluator.run_eval(EVAL_CSV, num_samples=100)
    
    # Save CSV Results
    df_eval_results.to_csv(EVAL_RESULTS, index=False)
    
    # Generate Plots
    print("[PLOTS] Generating Summary Plots...")
    evaluator.plot_results(df_eval_results, save_dir=RESULTS_DIR)
    
    # 3. REPORT
    print("\n==================== RESULTS (MCPE-3D) ====================")
    if len(df_eval_results) > 0:
        med_E_err = df_eval_results["energy_error"].median()
        med_A_err = df_eval_results["angular_error_deg"].median()
        flip_rate = (df_eval_results["angular_error_deg"] > 90).mean() * 100
        
        print(f"Median Energy Error:      {med_E_err:.2f} keV")
        print(f"Median Angular Error:     {med_A_err:.2f} deg")
        print(f"Head-Tail Flip Rate:      {flip_rate:.2f} %")
    
    print(f"Training Time:            {train_duration if 'train_duration' in locals() else 0.0:.2f} min")
    print(f"Saved Results:            {EVAL_RESULTS}")
    print("===========================================================\n")

if __name__ == "__main__":
    main()