#!/usr/bin/env python3
"""
MDPE Dual Pipeline Runner (Updated for MNPE)
--------------------------------------------
Toggle `USE_EMBEDDING_NET` to switch between:
 1. Summary Features (Standard NPE)
 2. MNPE + PointNet (Mixed Neural Posterior Estimation)
"""

from pathlib import Path
import torch
from time import time
import numpy as np
import os

# --- IMPORTS ---
from src.utils.data_utils import (
    load_raw_vacancies, 
    create_embedding_dataset, 
    preprocess_mdpe
)
from src.utils.sbi_runner import (
    make_prior,
    # Pipeline A: Summary Features (Standard NPE)
    make_inference_summary, 
    train_summary_posterior,
    # Pipeline B: Embedding Network (NEW MNPE FUNCTIONS)
    make_inference_mnpe_embedding,   # <--- Updated
    train_mnpe_embedding_posterior   # <--- Updated
)
from src.evaluation.eval_mdpe import evaluate_from_saved_tracks_mdpe

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
t_start = time()

# ============================================================
# CONFIGURATION
# ============================================================
# --- TOGGLE THIS TO SWITCH PIPELINES ---
USE_EMBEDDING_NET = True  
# ---------------------------------------

BASE_DIR     = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi")
TRAIN_CSV    = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_TRAIN_200k_ROTATED.csv"
EVAL_CSV     =  "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/MDPE_EVAL_5k_ROTATED.csv"
RESULTS_DIR  = BASE_DIR / "data/results"

# Naming based on method
if USE_EMBEDDING_NET:
    # Changed filename to reflect MNPE usage
    POSTERIOR_FILE = Path("results/mdpe_mnpe_pointnet_posterior.pt")
    EVAL_RESULTS   = RESULTS_DIR / "Jan23_MNPE_PointNet_eval.csv"
else:
    POSTERIOR_FILE = Path("results/mdpe_summary_posterior.pt")
    EVAL_RESULTS   = RESULTS_DIR / "Jan22_SummaryFeatures_eval.csv"

ENERGY_MIN, ENERGY_MAX = 1.0, 101.0
LATENT_DIM = 64

# ============================================================
# STEP 1: SETUP & LOAD DATA
# ============================================================
print(f"\n[START] Running Pipeline: {'MNPE + EMBEDDING' if USE_EMBEDDING_NET else 'SUMMARY FEATURES'}")

# Ensure the prior is suitable for MNPE (Energy=Continuous, Class=Discrete)
# make_prior should handle the BoxUniform bounds. 
# MNPE will automatically interpret the data types based on the prior or inputs.
prior = make_prior(ENERGY_MIN, ENERGY_MAX)

train_duration_min = 0.0
MAX_POINTS = None 

if POSTERIOR_FILE.exists():
    print(f"\n[STEP 1] Loading existing posterior -> {POSTERIOR_FILE}")
    # Warning: MNPE posteriors require 'weights_only=False' usually, or safe globals
    posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
    
    if USE_EMBEDDING_NET:
        print("[INFO] Recalculating MAX_POINTS from training data to ensure eval consistency...")
        vacancies_raw, _ = load_raw_vacancies(TRAIN_CSV)
        MAX_POINTS = max(len(t) for t in vacancies_raw)

else:
    print("\n[STEP 1] Training New Posterior...")
    t_train_start = time()

    if USE_EMBEDDING_NET:
        # --- PIPELINE B: MNPE + POINTNET ---
        print("[DATA] Loading Raw Point Clouds...")
        vacancies_raw, theta_train = load_raw_vacancies(TRAIN_CSV)
        
        # Dynamic Padding Size
        MAX_POINTS = max(len(t) for t in vacancies_raw)
        print(f"[DATA] Max Track Length found: {MAX_POINTS}. Padding dataset...")
        
        # Format Tensor (N, MAX_POINTS, 4) -> [x, y, z, mask]
        x_train = create_embedding_dataset(vacancies_raw, max_points=MAX_POINTS)
        
        # --- NEW MNPE LOGIC ---
        # We no longer manually create 'emb_net'. 
        # The 'make_inference_mnpe_embedding' function creates it internally 
        # to ensure it is properly registered with the Mixed Density Network.
        
        inference = make_inference_mnpe_embedding(prior, latent_dim=LATENT_DIM)
        
        # Train using the MNPE-specific trainer
        posterior = train_mnpe_embedding_posterior(inference, theta_train, x_train)

    else:
        # --- PIPELINE A: SUMMARY FEATURES ---
        print("[DATA] Calculating Summary Features...")
        x_train, theta_train, _, _, _ = preprocess_mdpe(TRAIN_CSV)
        print(f"[DATA] Feature Vector Size: {x_train.shape[1]}")
        
        # Build & Train (Standard NPE)
        inference = make_inference_summary(prior)
        posterior = train_summary_posterior(inference, theta_train, x_train)

    t_train_end = time()
    train_duration_min = (t_train_end - t_train_start) / 60
    
    # Save
    POSTERIOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(posterior, POSTERIOR_FILE)
    print(f"[SAVE] Posterior saved -> {POSTERIOR_FILE}")


# ============================================================
# STEP 2: EVALUATE
# ============================================================
print("\n[STEP 2] Evaluating...")

# Note: The evaluation script must simply call `posterior.sample()`.
# MNPE supports this API transparently, so no changes needed in the eval call.
df_eval = evaluate_from_saved_tracks_mdpe(
    posterior=posterior,
    eval_csv=EVAL_CSV,
    output_csv=EVAL_RESULTS,
    max_points=MAX_POINTS if USE_EMBEDDING_NET else None, 
    n_post_samples=500
)

# ============================================================
# RESULTS
# ============================================================
print("\n==================== RESULTS ====================")
accuracy = df_eval["class_correct"].mean() * 100
median_energy_error = df_eval["energy_error_pct"].median()

print(f"Method:                   {'MNPE + DeepSets' if USE_EMBEDDING_NET else 'Summary Features'}")
if USE_EMBEDDING_NET:
    print(f"Max Points (Padding):     {MAX_POINTS}")
print(f"Direction Accuracy:       {accuracy:.2f}%")
print(f"Median Energy Error:      {median_energy_error:.2f}%")
print(f"Training Time:            {train_duration_min:.2f} min")
print("================================================\n")

print(f"TOTAL RUNTIME: {(time() - t_start)/60:.2f} min")