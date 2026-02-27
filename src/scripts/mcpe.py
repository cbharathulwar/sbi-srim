#!/usr/bin/env python3
"""
Pipeline: Continuous MCPE (Physics Model)
-----------------------------------------
1. Preprocesses tracks into 28D physics vectors.
2. Trains an SNPE (Sequential Neural Posterior) using SBI defaults.
3. Evaluates and saves a CSV report (No plots).
"""

from pathlib import Path
from time import time
import torch
import os
import sys

# Ensure 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Specific Imports for Continuous Pipeline
from src.utils.data_utils import preprocess_mcpe
from src.evaluation.eval_mcpe import ContinuousEvaluator

# SBI Imports
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

# Mac Optimization
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
TRAIN_CSV     = BASE_DIR / "data/MCPE_TRAIN_CONTINUOUS_FIXED_250k.csv"
EVAL_CSV      = BASE_DIR / "data/MCPE_EVAL_CONTINUOUS_FIXED_250k.csv"
RESULTS_DIR   = BASE_DIR / "results"

# Filenames
POSTERIOR_FILE = RESULTS_DIR / "mcpe2_continuous_posterior.pt"
EVAL_RESULTS   = RESULTS_DIR / "mcpe3_continuous250k_eval_results.csv"

# Hyperparameters
ENERGY_MIN, ENERGY_MAX = 1.0, 100.0
FORCE_RETRAIN  = False  # Set True to overwrite existing model
# =================================================

def main():
    t_start = time()
    print(f"\n[START] Pipeline: CONTINUOUS MCPE (Physics 28D)")
    print(f"        Train Data: {TRAIN_CSV.name}")
    print(f"        Eval Data:  {EVAL_CSV.name}")

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
        print("[DATA] Preprocessing training data (28D Vectors)...")
        # Returns: features (N, 28), targets (N, 3) [E, sin, cos]
        x_train, theta_train, _, _, _ = preprocess_mcpe(TRAIN_CSV)
        print(f"[DATA] Feature Vector Size: {x_train.shape[1]}")
        print(f"[DATA] Target Vector Size:  {theta_train.shape[1]} (Energy, Sin, Cos)")

        # B. Define Prior
        # Energy: 1-100, Sin/Cos: -1 to 1
        prior = BoxUniform(
            low=torch.tensor([ENERGY_MIN, -1.0, -1.0]), 
            high=torch.tensor([ENERGY_MAX, 1.0, 1.0])
        )

        # C. Configure Network (NSF with Normalization)
        print("[MODEL] Initializing Neural Spline Flow (NSF)...")
        # z_score_x='independent' handles normalization automatically
        density_estimator = posterior_nn(
            model="nsf", 
            hidden_features=64, 
            num_transforms=5,
            z_score_x='independent' 
        )

        # D. Train SNPE
        inference = SNPE(prior=prior, density_estimator=density_estimator)
        # SBI expects (theta, x) -> (targets, features)
        inference.append_simulations(theta_train, x_train)
        
        print(f"[TRAIN] Starting SBI training loop (Default parameters)...")
        # Letting SBI handle batch size, learning rate, and stopping
        density_estimator = inference.train(show_train_summary=True)

        # E. Save
        posterior = inference.build_posterior(density_estimator)
        train_duration = (time() - t_train_start) / 60
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(posterior, POSTERIOR_FILE)
        print(f"[SAVE] Posterior saved -> {POSTERIOR_FILE}")

    # 2. EVALUATE
    print("\n[STEP 2] Evaluating...")
    
    # Instantiate the evaluator wrapper
    evaluator = ContinuousEvaluator(posterior)
    
    # Run evaluation logic (No Plots)
    df_eval = evaluator.run_eval(EVAL_CSV, num_samples=1000)
    
    # Save CSV Results
    df_eval.to_csv(EVAL_RESULTS, index=False)
    
    # 3. REPORT
    print("\n==================== RESULTS (Continuous MCPE) ====================")
    if len(df_eval) > 0:
        med_E_err = df_eval["energy_error"].median()
        med_A_err = df_eval["angular_error"].median()
        
        print(f"Median Energy Error:      {med_E_err:.2f} keV")
        print(f"Median Angular Error:     {med_A_err:.2f} deg")
    
    print(f"Training Time:            {train_duration:.2f} min")
    print(f"Saved Results:            {EVAL_RESULTS}")
    print("===================================================================\n")

if __name__ == "__main__":
    main()