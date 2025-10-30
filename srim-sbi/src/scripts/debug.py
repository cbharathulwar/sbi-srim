#!/usr/bin/env python3
"""
SRIM–SBI Debug: Inspect Summary Vectors
=======================================
Run full pipeline up to SRIM summarization (no PPC),
then print example summary feature vectors to verify
bin_frac_* structure and content.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# --- Imports from your modules ---
from src.utils.data_utils import preprocess, make_x_test
from src.utils.sbi_runner import (
    make_prior,
    make_inference,
    train_posterior,
    sample_posterior_bulk,
)
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import summarize_all_runs

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RAW_CSV = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv"
)
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Outputs_debug_vec"
RESULTS_DIR = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data"
)

ION_SYMBOL = "C"
N_IONS = 5
SAMPLES_PER_TRACK = 2
N_PER_ENERGY = 2

PRIOR_LOW, PRIOR_HIGH = [1_000], [2_000_000]
POSTERIOR_FILE = RESULTS_DIR / "trained_posterior.pt"
N_BINS = 6
BIN_EDGES_FILE = RESULTS_DIR / "bin_edges.json"


# --------------------------------------------------------------------------
def run_vector_debug():
    print("\n[INIT] Starting debug run to inspect summary vectors...")

    # 1️⃣ Preprocess raw CSV → summary features
    x_obs, theta, track_ids, _, df_pre = preprocess(
        RAW_CSV, n_bins=N_BINS, bin_edges_path=BIN_EDGES_FILE
    )
    df_pre.to_csv(RESULTS_DIR / "debug_pre_summary_vec.csv", index=False)
    print(
        f"[INFO] Preprocess summary saved → {RESULTS_DIR / 'debug_pre_summary_vec.csv'}"
    )
    print(f"[INFO] Shape: {df_pre.shape}, Columns: {list(df_pre.columns)}")

    # Print a few example rows
    print("\n=== Example PREPROCESS summary vectors (first 3 tracks) ===")
    print(df_pre.head(3).to_string(index=False))

    # 2️⃣ Train/load posterior
    if POSTERIOR_FILE.exists():
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        print(f"[INFO] Loaded posterior from {POSTERIOR_FILE}")
    else:
        prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)
        inference = make_inference(prior, density_estimator="nsf")
        posterior = train_posterior(inference, theta, x_obs)
        torch.save(posterior, POSTERIOR_FILE)
        print(f"[INFO] Trained and saved new posterior → {POSTERIOR_FILE}")

    # 3️⃣ Select test tracks & run SRIM
    x_test, _ = make_x_test(df_pre, n_per_energy=N_PER_ENERGY, random_state=42)
    x_test.to_csv(RESULTS_DIR / "debug_x_test_vec.csv", index=False)

    feats = ["mean_depth_A", "vacancies_per_ion"] + [
        f"bin_frac_{i}" for i in range(1, N_BINS + 1)
    ]
    feats = [f for f in feats if f in x_test.columns]
    x_tensor = torch.tensor(x_test[feats].values, dtype=torch.float32)

    samples_dict, _ = sample_posterior_bulk(
        posterior,
        x_obs=x_tensor,
        num_samples=SAMPLES_PER_TRACK,
        track_ids=x_test["track_id"].tolist(),
    )

    run_srim_multi_track(
        samples_dict=samples_dict,
        x_test=x_test,
        track_ids=x_test["track_id"].tolist(),
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_pre,
        bin_edges_path=BIN_EDGES_FILE,
        overwrite=True,
    )

    # 4️⃣ Summarize SRIM outputs
    df_post = summarize_all_runs(
        str(OUTPUT_BASE),
        label="debug_vec",
        n_bins=N_BINS,
        bin_edges_path=BIN_EDGES_FILE,
        strict=False,
    )
    df_post.to_csv(RESULTS_DIR / "debug_post_summary_vec.csv", index=False)
    print(
        f"[INFO] Postprocess summary saved → {RESULTS_DIR / 'debug_post_summary_vec.csv'}"
    )
    print(f"[INFO] Shape: {df_post.shape}, Columns: {list(df_post.columns)}")

    # Print a few example SRIM summaries
    print("\n=== Example POSTPROCESS SRIM summary vectors (first 5 rows) ===")
    print(df_post.head(5).to_string(index=False))

    # 5️⃣ Optional quick bin sanity check
    bin_cols = [c for c in df_post.columns if c.startswith("bin_frac_")]
    if bin_cols:
        sums = df_post[bin_cols].sum(axis=1)
        print(
            f"\n[CHECK] bin_frac_* sums → mean={sums.mean():.3f}, std={sums.std():.3f}"
        )
        print(f"Non-NaN fraction: {(sums.notna().mean()*100):.1f}%")

    print("\n✅ Debug complete — inspect printed vectors above.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_vector_debug()
