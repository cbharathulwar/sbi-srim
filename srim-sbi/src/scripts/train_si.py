#!/usr/bin/env python3
"""
SRIM–SBI Simple Pipeline Runner (Finalized)
=========================================
End-to-end workflow:
    1. Preprocess SRIM raw CSV → x_obs
    2. Train SBI posterior (NSF)
    3. Sample posterior for x_test subset (balanced across energies)
    4. Do analysis on this 
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# --------------------------------------------------------------------------
# --- Project modules ---
# --------------------------------------------------------------------------
from src.utils.data_utils import preprocess, make_x_test
from src.utils.sbi_runner import (
    make_prior,
    make_inference,
    train_posterior,
    sample_posterior_bulk,
)
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import summarize_all_runs

# Updated PPC imports
from src.utils.hist import run_scalar_ppc, run_shape_ppc

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
RAW_CSV = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/low_1_3_10.csv"
)
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Oct28Run1"
RESULTS_DIR = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data"
)
PPC_DIR = RESULTS_DIR / "ppc-results-Oct28Run1"

ION_SYMBOL = "C"
SAMPLES_PER_TRACK = 5000  # posterior samples per track
N_PER_ENERGY = 50  # test tracks per energy

# Use keV consistently for SBI training and conditioning
PRIOR_LOW, PRIOR_HIGH = [0.5], [12]  # keV
POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_low.pt"

N_BINS = 6  # number of depth bins for rbin_frac_*
BIN_EDGES_FILE = RESULTS_DIR / "bin_edges.json"  # used by preprocess() and SRIM runner


# --------------------------------------------------------------------------
# 1. Preprocess SRIM CSV data
# --------------------------------------------------------------------------
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, _, df_summary = preprocess(
        RAW_CSV, n_bins=N_BINS
    )
    print(f"[INFO] x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"[INFO] {len(track_ids)} unique (ion, energy) tracks summarized.")

    out_csv = RESULTS_DIR / "srim_summary_preprocessed.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"[INFO] Saved summary → {out_csv}")

    print("[DEBUG] ---- Energy Distribution ----")
    print(df_summary.groupby(["ion", "energy_keV"]).size())
    print(f"[DEBUG] Total tracks: {len(df_summary)}")
    print("----------------------------------------------------")

    return x_obs, theta, df_summary


# --------------------------------------------------------------------------
# 2. Train or load posterior
# --------------------------------------------------------------------------
def train_or_load_posterior(x_obs, theta, save_path=POSTERIOR_FILE):
    if save_path.exists():
        posterior = torch.load(save_path, map_location="cpu")
        print(f"[STEP 2] ✅ Loaded existing posterior from {save_path}")
    else:
        print("[STEP 2] Training new SBI posterior ...")
        prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)  # keV bounds
        inference = make_inference(prior, density_estimator="nsf")
        posterior = train_posterior(inference, theta, x_obs)
        torch.save(posterior, save_path)
        print(f"[INFO] Posterior saved → {save_path}")
    return posterior


# --------------------------------------------------------------------------
# 3. Sample posterior & run SRIM
# --------------------------------------------------------------------------
import torch

def sample_posterior_only(posterior, df_summary, n_per_energy=3, n_samples=5000, n_bins=6):
    """
    Sample theta values from the posterior for selected test tracks.
    Does NOT run SRIM or compute any stats — only returns raw samples.
    """

    print("\n[STEP 3] Sampling posterior")

    # --- Select representative test tracks ---
    x_test, _ = make_x_test(df_summary, n_per_energy=n_per_energy, random_state=42)
    track_ids = x_test["track_id"].tolist()

    print(f"[INFO] Selected {len(track_ids)} test tracks ({n_per_energy} per energy).")

    # --- Prepare conditioning features ---
    feats = ["mean_depth_A", "max_depth_A", "vacancies_per_ion"] + [
        f"rbin_frac_{i}" for i in range(1, n_bins + 1)
    ]
    feats = [f for f in feats if f in x_test.columns]
    print(f"[INFO] Using features: {feats}")

    x_tensor = torch.tensor(x_test[feats].values, dtype=torch.float32)

    # --- Sample posterior ---
    print(f"[INFO] Drawing {n_samples} samples per track ...")
    samples_dict, _ = sample_posterior_bulk(
        posterior=posterior,
        x_obs=x_tensor,
        num_samples=n_samples,
        track_ids=track_ids,
    )

    print(f"[STEP 3] Sampling complete ({len(samples_dict)} tracks).")

    return samples_dict, x_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_and_plot(samples_dict, x_test, output_dir="posterior_eval"):
    """
    Compute percent error and posterior std per track,
    save CSV summary, and generate distribution plots.

    Args:
        samples_dict: dict of {track_id: [posterior samples]}
        x_test: DataFrame with 'track_id' and 'energy_keV' columns
        output_dir: folder to save plots and CSV
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    rows = []

    for tid in x_test["track_id"]:
        true_e = x_test.loc[x_test["track_id"] == tid, "energy_keV"].values[0]
        samples = np.array(samples_dict[tid])

        mean_e = np.mean(samples)
        std_e = np.std(samples)
        percent_error = ((mean_e - true_e) / true_e) * 100.0

        rows.append({
            "track_id": tid,
            "true_energy_keV": true_e,
            "posterior_mean_keV": mean_e,
            "posterior_std_keV": std_e,
            "percent_error": percent_error
        })

        # --- Plot ---
        plt.figure(figsize=(6, 4))
        plt.hist(samples, bins=40, color="skyblue", alpha=0.7, density=True, label="Posterior samples")

        # True energy line
        plt.axvline(true_e, color="red", linestyle="--", label=f"True = {true_e:.1f} keV")

        # Posterior mean line
        plt.axvline(mean_e, color="black", linestyle="-", label=f"Mean = {mean_e:.1f} keV")

        # Annotation box
        textstr = f"Δ = {percent_error:.2f}%\nσ = {std_e:.2f} keV"
        plt.text(
            0.97, 0.97, textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        plt.xlabel("Energy (keV)")
        plt.ylabel("Density")
        plt.title(f"Posterior Distribution — {tid}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_dir / f"{tid}_posterior.png", dpi=150)
        plt.close()

    # --- Summary CSV ---
    df_eval = pd.DataFrame(rows)
    csv_path = output_dir / "posterior_summary.csv"
    df_eval.to_csv(csv_path, index=False)

    # --- Summary prints ---
    mean_err = df_eval["percent_error"].mean()
    mean_std = df_eval["posterior_std_keV"].mean()
    print(f"[INFO] Saved posterior plots and summary to: {output_dir}")
    print(f"[SUMMARY] Mean percent error: {mean_err:.2f}%")
    print(f"[SUMMARY] Mean posterior std: {mean_std:.2f} keV")

    return df_eval


# --------------------------------------------------------------------------
# 6. Master run
# --------------------------------------------------------------------------
from datetime import datetime

def run_pipeline():
    start_time = datetime.now()
    print(f"[INIT] SRIM–SBI pipeline started @ {start_time:%Y-%m-%d %H:%M:%S}")

    try:
        # --- STEP 1: Preprocess ---
        print("\n[STEP 1] Preprocessing SRIM data ...")
        x_obs, theta, df_summary = preprocess_data()

        # --- STEP 2: Train or load posterior model ---
        print("\n[STEP 2] Training/loading SBI posterior ...")
        posterior = train_or_load_posterior(x_obs, theta)

        # --- STEP 3: Sample posterior only ---
        print("\n[STEP 3] Sampling posterior ...")
        samples_dict, x_test = sample_posterior_only(posterior, df_summary,50)

        # --- STEP 4: Evaluate posterior & visualize ---
        print("\n[STEP 4] Evaluating posterior predictions ...")
        df_eval = evaluate_and_plot(samples_dict, x_test, output_dir="posterior_eval")

        # --- STEP 5: Summary ---
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n[DONE] ✅ Full simplified pipeline completed in {elapsed/60:.1f} min")
        print(f"[INFO] Tracks processed: {len(x_test)}")
        print(f"[INFO] Results saved to: posterior_eval/")

        return df_eval

    except Exception as e:
        print(f"[FATAL] Pipeline failed: {e}")
        raise


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
