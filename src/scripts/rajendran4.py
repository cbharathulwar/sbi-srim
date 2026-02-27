import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_DISABLE"] = "1"


import pandas as pd
import numpy as np
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
from pathlib import Path

# --- MNPE Imports ---
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import ( 
    make_mnpe_inference_old,
    make_mnpe_prior_old, 
    make_mnpe_posterior_old
)
from src.evaluation.eval_mnpe import (
    evaluate_mnpe_with_confidence, 
    generate_curve_data_ranked,
    generate_mnpe_curve_data
)

# --- Rajendran Imports ---
from src.evaluation.eval_rajendran import (
    batch_process_raj_features, 
    calibrate_raj_thresholds, 
    generate_raj_curve_data
)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = Path("data/processed_splits")
RESULTS_DIR = Path("results/apples_to_apples")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input Datasets (FIXED: Using the single Shared Eval set)
MNPE_TRAIN_CSV  = DATA_DIR / "mnpe_train.csv"
RAJ_CALIB_CSV   = DATA_DIR / "raj_calib.csv"
SHARED_EVAL_CSV = DATA_DIR / "shared_eval.csv"

# Output Files
POSTERIOR_PATH = RESULTS_DIR / "mnpe_raj_posterior6.pt"
MNPE_RAW_EVAL  = RESULTS_DIR / "mnpe_raw_eval_results8.csv"
RAJ_RAW_EVAL   = RESULTS_DIR / "raj_raw_eval_results8.csv"
RAJ_RAW_CALIB  = RESULTS_DIR / "raj_raw_calib_results8.csv"
1
FINAL_CURVE_DATA = RESULTS_DIR / "efficiency_vs_accuracy_curves8.csv"
# The discrete energies we want to compare (matching the paper)
ENERGIES_TO_PLOT = [1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]

def main():
    print("==================================================")
    print("  MNPE vs. RAJENDRAN (2017) MASTER PIPELINE")
    print("==================================================")

    # ---------------------------------------------------------
    # PART 1: MNPE Neural Pipeline
    # ---------------------------------------------------------
    print("\n[PART 1] Executing MNPE Neural Pipeline...")
    
    # Always preprocess training data to get norm_stats
    # (even if posterior exists, we need the stats for eval)
    print(" -> Preprocessing MNPE Training Data...")
    x_train, theta_train, _, meta_train, _ = preprocess_mnpe(MNPE_TRAIN_CSV, n_bins=6)
    norm_stats = {'feat_mean': meta_train['feat_mean'], 'feat_std': meta_train['feat_std']}
    
    if not POSTERIOR_PATH.exists():
        print(" -> Training MNPE Posterior...")
        inference, _ = make_mnpe_inference_old(
            theta_train.to("cpu"), x_train.to("cpu"), prior=None, device="cpu"
        )
        posterior = inference.build_posterior()
        torch.save(posterior, POSTERIOR_PATH)
    else:
        print(f" -> Loading existing posterior from {POSTERIOR_PATH}")
        posterior = torch.load(POSTERIOR_PATH, map_location="cpu")

    print(" -> Evaluating MNPE Confidence on SHARED Eval Set...")
    df_mnpe_results = evaluate_mnpe_with_confidence(
        posterior=posterior,
        eval_csv_path=SHARED_EVAL_CSV,
        output_csv=MNPE_RAW_EVAL,
        n_post_samples=500,
        norm_stats=norm_stats  # <-- PASS TRAINING STATS
    )

    # ... rest of pipeline unchanged ...

    # ---------------------------------------------------------
    # PART 2: Rajendran Baseline Pipeline
    # ---------------------------------------------------------
    print("\n[PART 2] Executing Rajendran (2017) Baseline Pipeline...")
    
    print(" -> Processing Calibration Set (Extracting Asymmetries)...")
    df_raj_calib = pd.read_csv(RAJ_CALIB_CSV)
    df_calib_features = batch_process_raj_features(df_raj_calib)
    df_calib_features.to_csv(RAJ_RAW_CALIB, index=False)
    
    # Prove we can hit 95% accuracy (5% FPR) and log the required thresholds
    print(" -> Calibrating Rajendran Thresholds for 95% Accuracy...")
    raj_thresholds = calibrate_raj_thresholds(df_calib_features, target_accuracy=0.95)

    print(" -> Processing SHARED Evaluation Set (Extracting Asymmetries)...")
    df_raj_eval = pd.read_csv(SHARED_EVAL_CSV) # <-- FIXED: Now uses shared eval
    df_raj_results = batch_process_raj_features(df_raj_eval)
    df_raj_results.to_csv(RAJ_RAW_EVAL, index=False)

    # ---------------------------------------------------------
    # PART 3: Curve Generation (The Apples-to-Apples Sweep)
    # ---------------------------------------------------------
    print("\n[PART 3] Generating Performance Curves...")
    
    all_curve_data = []
    
    df_raw_shared = pd.read_csv(SHARED_EVAL_CSV)
    
    for energy in ENERGIES_TO_PLOT:
        e_min = energy - 0.5
        e_max = energy + 0.5
        
        true_tracks_in_bin = df_raw_shared[
            (df_raw_shared['energy_keV'] >= e_min) & 
            (df_raw_shared['energy_keV'] < e_max)
        ]['ion_number'].nunique()
        
        if true_tracks_in_bin == 0:
            print(f" [Warning] No tracks found in raw data for {energy} keV. Skipping.")
            continue
        
        # 1. MNPE curve (ranked by confidence)
        mnpe_bin = df_mnpe_results[
            (df_mnpe_results['true_energy'] >= e_min) & 
            (df_mnpe_results['true_energy'] < e_max)
        ]
        if len(mnpe_bin) > 0:
            mnpe_curve = generate_curve_data_ranked(mnpe_bin, 'confidence', true_tracks_in_bin)
            mnpe_curve['model'] = 'MNPE'
            mnpe_curve['energy_keV'] = energy
            all_curve_data.append(mnpe_curve)
        
        # 2. Rajendran curve (ranked by asymmetry)
        raj_bin = df_raj_results[
            (df_raj_results['energy_keV'] >= e_min) & 
            (df_raj_results['energy_keV'] < e_max)
        ]
        if len(raj_bin) > 0:
            raj_curve = generate_curve_data_ranked(raj_bin, 'asymmetry', true_tracks_in_bin)
            raj_curve['model'] = 'Rajendran_2017'
            raj_curve['energy_keV'] = energy
            all_curve_data.append(raj_curve)

    # Combine all points into a single master dataframe
    if len(all_curve_data) > 0:
        df_final_curves = pd.concat(all_curve_data, ignore_index=True)
        df_final_curves.to_csv(FINAL_CURVE_DATA, index=False)
        
        # Print a quick summary comparing both models at 95% Accuracy
        print("\n[QUICK SUMMARY] Efficiency at ~95% Accuracy:")
        print(f"{'Energy':<10} | {'MNPE Eff':<10} | {'Rajendran Eff':<15}")
        print("-" * 40)
        for energy in ENERGIES_TO_PLOT:
            # Grab the point on the curve closest to 95% accuracy
            e_data = df_final_curves[(df_final_curves['energy_keV'] == energy) & (df_final_curves['accuracy'] >= 0.945)]
            if e_data.empty: continue
            
            mnpe_eff = e_data[e_data['model'] == 'MNPE']['efficiency'].max()
            raj_eff = e_data[e_data['model'] == 'Rajendran_2017']['efficiency'].max()
            
            m_str = f"{mnpe_eff:.2%}" if pd.notna(mnpe_eff) else "N/A"
            r_str = f"{raj_eff:.2%}" if pd.notna(raj_eff) else "N/A"
            
            print(f"{energy:<10} | {m_str:<10} | {r_str:<15}")
            
        print("\n==================================================")
        print("  PIPELINE COMPLETE!")
        print(f"  Final plotting data saved to:\n  {FINAL_CURVE_DATA}")
        print("==================================================")
    else:
        print("\n[WARNING] No curve data was generated.")

if __name__ == "__main__":
    main()





