#!/usr/bin/env python3
import torch
from pathlib import Path
from datetime import datetime

# Import utils
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import (
    make_mnpe_prior,
    make_mnpe_inference,
    train_mnpe_posterior,
)
# Import our new evaluator
from src.evaluation.eval_mnpe import evaluate_siimpl_batch

# ============================================================
# CONFIGURATION
# ============================================================

# 1. INPUT DATA (Train & Eval Split)
TRAIN_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/siimpl_800keV_train.csv")
EVAL_CSV  = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nv_800keV2_eval.csv")

# 2. OUTPUTS
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results")
POSTERIOR_FILE = RESULTS_DIR / "posterior_NVsiimpl6_800keV.pt"
EVAL_RESULTS_CSV = RESULTS_DIR / "results_NVsiimpl6_800keV.csv"

# 3. PHYSICS PRIORS (Critical Change!)
# We must cover the 800 keV range.
PRIOR_LOW = 100.0   # Lower bound
PRIOR_HIGH = 1000.0 # Upper bound (Safety margin above 800)
N_BINS = 6

# ============================================================
# PIPELINE STEPS
# ============================================================

PRIOR_LOW = 100.0
PRIOR_HIGH = 1000.0
N_BINS = 6

def run_pipeline():
    start = datetime.now()
    print(f"[INIT] SIIMPL Pipeline started @ {start:%Y-%m-%d %H:%M:%S}")

    # 1. LOAD DATA
    print(f"\n[STEP 1] Loading Training Data: {TRAIN_CSV}")
    x_train, theta_train, _, _, _ = preprocess_mnpe(TRAIN_CSV, n_bins=N_BINS)
    print(f"   Loaded {len(x_train)} training tracks.")
 
    # --- THE MAGIC FIX: ENERGY JITTER ---
    # Since all tracks are exactly 800.0 keV, SBI thinks it's a Discrete Category.
    # We add tiny noise (+/- 0.1 keV) so SBI treats it as Continuous.
    print("   [FIX] Applying micro-jitter to Energy values...")
    jitter = torch.randn_like(theta_train[:, 0]) * 0.1
    theta_train[:, 0] += jitter

    # 2. TRAIN
    if POSTERIOR_FILE.exists():
        print(f"\n[STEP 2] Loading existing posterior: {POSTERIOR_FILE}")
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
    else:
        print(f"\n[STEP 2] Training NEW Posterior...")
        prior = make_mnpe_prior(PRIOR_LOW, PRIOR_HIGH)
        inf = make_mnpe_inference(prior)
        posterior = train_mnpe_posterior(inf, theta_train, x_train)
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(posterior, POSTERIOR_FILE)

    # 3. EVALUATE
    if EVAL_CSV.exists():
        print("\n[STEP 3] Running Evaluation...")
        evaluate_siimpl_batch(
            posterior=posterior,
            eval_csv_path=EVAL_CSV,
            output_csv_path=EVAL_RESULTS_CSV,
            n_bins=N_BINS
        )

    print(f"\n[DONE] Finished.")

if __name__ == "__main__":
    run_pipeline()