#!/usr/bin/env python3
"""
SRIM–SBI Full Pipeline Runner (Finalized)
=========================================
End-to-end workflow:
    1. Preprocess SRIM raw CSV → x_obs
    2. Train SBI posterior (NSF)
    3. Sample posterior for x_test subset (balanced across energies)
    4. Run SRIM for predicted θ samples
    5. Summarize SRIM outputs → x_check
    6. Perform Posterior Predictive Check (PPC):
         - Scalars: Mean depth + Vacancies per ion (+ max depth if present)
         - Shape: rbin_frac_* (EMD, Cosine, Z)
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
# Modules
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
# Stuff
# --------------------------------------------------------------------------
RAW_CSV = Path(
    "//Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/low_1_3_10.csv"
)
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Oct24Run1"
RESULTS_DIR = Path(
    "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data"
)
PPC_DIR = RESULTS_DIR / "ppc-results-Oct24Run1"

ION_SYMBOL = "C"
N_IONS = 150  # increase for realism
SAMPLES_PER_TRACK = 20  # posterior samples per track
N_PER_ENERGY = 2  # test tracks per energy

# Use keV consistently for SBI training and conditioning
PRIOR_LOW, PRIOR_HIGH = [0.5], [12]  # keV
POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_low.pt"

N_BINS = 6  # number of depth bins for rbin_frac_*
BIN_EDGES_FILE = RESULTS_DIR / "bin_edges.json"  # used by preprocess() and SRIM runner


# --------------------------------------------------------------------------
# 1. Preprocess SRIM CSV data
# --------------------------------------------------------------------------
def preprocess_data():
    print("\n [Step 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, _, df_summary = preprocess(
        RAW_CSV, n_bins=N_BINS, bin_edges_path=BIN_EDGES_FILE
    )
    print(f"x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"{len(track_ids)} unique (ion, energy) tracks summarized.")

    out_csv = RESULTS_DIR / "srim_summary_preprocessed.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")

    print(df_summary.groupby(["ion", "energy_keV"]).size())
    return x_obs, theta, df_summary


# --------------------------------------------------------------------------
# 2. Train or load posterior
# --------------------------------------------------------------------------
def train_or_load_posterior(x_obs, theta, save_path=POSTERIOR_FILE):
    if save_path.exists():
        posterior = torch.load(save_path, map_location="cpu")
        print(f"[Step 2] Loaded existing posterior from {save_path}")
    else:
        print("[Step[] 2] Training SBI posterior ...")
        prior = make_prior(low=PRIOR_LOW, high=PRIOR_HIGH)  # keV bounds
        inference = make_inference(prior, density_estimator="nsf")
        posterior = train_posterior(inference, theta, x_obs)
        torch.save(posterior, save_path)
        print(f"Posterior saved → {save_path}")
    return posterior




# --------------------------------------------------------------------------
# 3. Sample posterior & run SRIM
# --------------------------------------------------------------------------
def sample_and_run_srim(posterior, df_summary):
    print("\n[STEP 3] Selecting test tracks & running SRIM ...")

    # Balanced selection per energy
    x_test, _ = make_x_test(df_summary, n_per_energy=N_PER_ENERGY, random_state=42)
    x_test.to_csv(RESULTS_DIR / "x_test_selection.csv", index=False)
    track_ids = x_test["track_id"].tolist()

    print(f"[INFO] Selected {len(track_ids)} tracks ({N_PER_ENERGY} per energy level).")
    print("[DEBUG] Energies represented:", sorted(x_test["energy_keV"].unique()))

    # --- Prepare feature list for posterior conditioning ---
    # Use rbin_frac_* (NOT bin_frac_*)
    feats = ["mean_depth_A", "max_depth_A", "vacancies_per_ion"] + [f"rbin_frac_{i}" for i in range(1, N_BINS + 1)]
    available_feats = [f for f in feats if f in x_test.columns]
    print(f"[INFO] Using features for posterior sampling: {available_feats}")

    x_tensor = torch.tensor(x_test[available_feats].values, dtype=torch.float32)

    print(f"[DEBUG] Sampling posterior ({SAMPLES_PER_TRACK} samples per track) ...")
    samples_dict, _ = sample_posterior_bulk(
        posterior=posterior,
        x_obs=x_tensor,
        num_samples=SAMPLES_PER_TRACK,
        track_ids=track_ids,
    )

    missing = set(track_ids) - set(samples_dict.keys())
    assert not missing, f"[ASSERT] Missing posterior samples for tracks: {missing}"

    print(
        f"\n[INFO] Running SRIM for {len(track_ids)} tracks × {SAMPLES_PER_TRACK} θ samples ..."
    )
    run_srim_multi_track(
        samples_dict=samples_dict,
        x_test=x_test,
        track_ids=track_ids,
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_summary,
        bin_edges_path=BIN_EDGES_FILE,  # keep edges consistent
        overwrite=True,
    )

    print("[STEP 3] ✅ SRIM runs complete.")
    return x_test, track_ids


# --------------------------------------------------------------------------
# 4. Summarize SRIM outputs
# --------------------------------------------------------------------------
def summarize_outputs():
    print("\n[STEP 4] Summarizing SRIM outputs ...")
    x_check = summarize_all_runs(
        str(OUTPUT_BASE), label=datetime.now().strftime("%Y%m%d_%H%M%S"), n_bins=N_BINS
    )
    out_csv = RESULTS_DIR / "x_check_summary.csv"
    x_check.to_csv(out_csv, index=False)
    print(f"[INFO] x_check summary saved → {out_csv}")
    return x_check


# --------------------------------------------------------------------------
# 5. Posterior Predictive Check (PPC) — scalars + shapes
# --------------------------------------------------------------------------
def run_ppc(x_check: pd.DataFrame, x_test: pd.DataFrame):
    """
    Posterior Predictive Check (PPC):
      - Scalars: mean_depth_A, vacancies_per_ion (and max_depth_A if present)
      - Shape: rbin_frac_* distributions (EMD, cosine, Z)
    Saves all results to PPC_DIR.
    """
    print("\n[STEP 5] Running Posterior Predictive Check (PPC) ...")
    os.makedirs(PPC_DIR, exist_ok=True)

    # --- Align simulated and observed sets by track_id ---
    common_ids = sorted(
        set(x_check["track_id"].astype(str)).intersection(
            set(x_test["track_id"].astype(str))
        )
    )
    if not common_ids:
        raise RuntimeError(
            "No overlapping track_ids between x_check (sim) and x_test (obs)."
        )

    aligned_sim = x_check[x_check["track_id"].astype(str).isin(common_ids)].copy()
    aligned_obs = x_test[x_test["track_id"].astype(str).isin(common_ids)].copy()
    aligned_sim = aligned_sim.sort_values("track_id").reset_index(drop=True)
    aligned_obs = aligned_obs.sort_values("track_id").reset_index(drop=True)
    print(f"[INFO] Aligned {len(common_ids)} tracks for PPC.")

    # --- Scalar PPC ---
    print("\n[STEP 5A] Scalar PPC (mean depth + vacancies per ion) ...")
    scalar_results = run_scalar_ppc(
        aligned_sim, aligned_obs, out_dir=PPC_DIR / "scalars"
    )

    # --- Shape PPC ---
    print("\n[STEP 5B] Shape PPC (rbin_frac_* bins) ...")
    if not any(c.startswith("rbin_frac_") for c in aligned_sim.columns):
        raise RuntimeError("No rbin_frac_ columns found — cannot compute shape PPC.")

    shape_results = run_shape_ppc(
        aligned_sim, aligned_obs, out_dir=PPC_DIR / "shapes", prefix="rbin_frac_"
    )
    print(f"[INFO] Saved shape PPC results → {PPC_DIR / 'shapes'}")

    print(f"\n[DONE] ✅ PPC completed successfully for {len(common_ids)} tracks.")
    print(f"[INFO] Scalar + shape metrics saved in {PPC_DIR}")


# --------------------------------------------------------------------------
# 6. Master run
# --------------------------------------------------------------------------
def run_pipeline():
    start_time = datetime.now()
    print(f"[INIT] SRIM–SBI pipeline started @ {start_time:%Y-%m-%d %H:%M:%S}")

    try:
        x_obs, theta, df_summary = preprocess_data()
        posterior = train_or_load_posterior(x_obs, theta)
        x_test, track_ids = sample_and_run_srim(posterior, df_summary)
        x_check = summarize_outputs()
        run_ppc(x_check, x_test)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n[DONE] ✅ Full pipeline completed in {elapsed/60:.1f} min")
        print(f"[DEBUG] Tracks processed: {len(track_ids)}")

    except Exception as e:
        print(f"[FATAL] Pipeline failed: {e}")
        raise


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
