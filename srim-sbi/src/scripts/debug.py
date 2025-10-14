import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*DataFrame.swapaxes.*")

# --- your modules ---
from src.utils.data_utils import preprocess, make_x_test
from src.utils.analysis_utils import plot_ppc_histograms_per_track
from src.utils.sbi_runner import (
    make_prior, make_inference, train_posterior, sample_posterior_bulk
)
from src.utils.srim_utils import run_srim_multi_track
from src.utils.srim_parser import summarize_all_runs

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
OUTPUT_BASE = SRIM_DIR / "Outputs21"
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
PPC_DIR = RESULTS_DIR / "ppc-results14"

ION_SYMBOL = "C"     # keep explicit symbol (avoid numeric -> symbol pitfalls)
N_IONS = 10

# --------------------------------------------------------------------------
# STEP 1 — Preprocess data
# --------------------------------------------------------------------------
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM CSV data ...")
    x_obs, theta, track_ids, grouped, df_summary = preprocess(RAW_CSV)
    print(f"[INFO] x_obs shape: {x_obs.shape}, theta shape: {theta.shape}")
    print(f"[INFO] {len(track_ids)} unique (ion, energy) tracks summarized.")

    df_summary.to_csv(RESULTS_DIR / "srim_summary_preprocessed.csv", index=False)

    print("\n[DEBUG] ---- Energy Distribution in df_summary ----")
    print(df_summary.groupby("energy_keV").size())
    print(f"[DEBUG] Total rows: {len(df_summary)}")
    print(f"[DEBUG] Unique energies: {sorted(df_summary['energy_keV'].unique())}")
    print("----------------------------------------------------")
    return x_obs, theta, df_summary

# --------------------------------------------------------------------------
# STEP 3 — Train posterior and sample for ALL test tracks (no slicing)
# --------------------------------------------------------------------------
def train_and_sample(x_obs, theta, x_test, x_test_ids):
    print("\n[STEP 3] Training NPE posterior ...")
    prior = make_prior(low=[1_000.0], high=[2_000_000.0])  # eV
    inference = make_inference(prior=prior, density_estimator="nsf")
    posterior = train_posterior(inference, theta=theta, x_obs=x_obs)

    # Convert x_test (DataFrame) → tensor expected by sampler
    feats = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    assert all(f in x_test.columns for f in feats), f"[ASSERT] Missing features in x_test: {feats}"
    x_test_tensor = torch.tensor(x_test[feats].values, dtype=torch.float32)

    print(f"[DEBUG] Sampling posterior for {len(x_test_ids)} tracks ...")
    samples_by_track, samples_tensor = sample_posterior_bulk(
        posterior,
        x_obs=x_test_tensor,
        num_samples=20,
        track_ids=x_test_ids
    )

    # Invariants: IDs line up and energies truly span 9 levels
    assert len(samples_by_track) == len(x_test_ids), "[ASSERT] samples_by_track size mismatch"

    print("\n[DEBUG] Posterior theta ranges per track (first 12):")
    for tid in x_test_ids[:12]:
        s = samples_by_track[tid]
        s = s.detach().cpu().numpy().flatten() if hasattr(s, "detach") else np.asarray(s).flatten()
        print(f"  [RANGE] {tid}: {s.min():.2f} eV — {s.max():.2f} eV")

    return samples_by_track

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Preprocess → x_obs, theta, df_summary
    x_obs, theta, df_summary = preprocess_data()

    # 2) Build x_test with all 9 energies represented
    x_test = df_summary.copy()
    x_test_ids = x_test["track_id"].tolist()

    print(f"[DEBUG] len(x_test): {len(x_test)}")
    print(f"[DEBUG] Unique energies in x_test: {sorted(x_test['energy_keV'].unique())}")

    # 3) Train posterior + sample for ALL test tracks
    samples_by_track = train_and_sample(x_obs, theta, x_test, x_test_ids)

    # 4) Final pre-SRIM invariants
    print("\n[STEP 4] Verifying invariants before SRIM runs ...")
    assert len(x_test_ids) == len(x_test), "[ASSERT] track_ids and x_test length mismatch"

    missing = set(x_test_ids) - set(samples_by_track.keys())
    assert not missing, f"[ASSERT] sample dict missing track ids: {sorted(list(missing))[:10]}"

    sel_energies = df_summary[df_summary["track_id"].isin(x_test_ids)]["energy_keV"].round().astype(int).unique()
    print("[DEBUG] energies in current test set (int keV, pre-SRIM):", sorted(sel_energies))
    assert len(sel_energies) == 9, f"[ASSERT] expected 9 energies, got {len(sel_energies)}: {sorted(sel_energies)}"

    # 5) Run SRIM for ALL tracks with their posterior samples
    print("\n[STEP 5] Running SRIM for all test tracks ...")
    run_srim_multi_track(
        samples_dict=samples_by_track,
        x_test=x_test,
        track_ids=x_test_ids,
        srim_directory=str(SRIM_DIR),
        output_base=OUTPUT_BASE,
        ion_symbol=ION_SYMBOL,
        number_ions=N_IONS,
        df_summary=df_summary,
        overwrite=True
    )

    print("\n[DONE] Debug run complete.")