# src/evaluation/random_eval_npe.py

"""
Random SRIM Evaluation for NPE (energy-only model)
--------------------------------------------------
This evaluates your ENERGY-only posterior:

        θ = [energy_keV]

Steps:
    1. Sample a random energy from prior range
    2. Run SRIM at that true energy
    3. Parse collisions → center → preprocess_npe
    4. Sample posterior → compare predicted mean to true
"""

import os
import time
import random
import torch
import pandas as pd
from pathlib import Path

from src.utils.data_utils import preprocess_npe   # << your existing feature extractor
from src.utils.srim_utils import run_srim_for_energy, parse_collisions
from src.utils.sbi_runner import guarded_posterior_sample


def evaluate_one_random_npe(
    posterior,
    prior_low,
    prior_high,
    srim_dir,
    out_root,
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
):
    """Evaluate ONE randomly generated SRIM track (NPE only)."""

    # -----------------------------
    # 1. Draw true energy
    # -----------------------------
    E_true = random.uniform(prior_low, prior_high)
    print(f"[INFO][NPE] True energy = {E_true:.2f} keV")

    # -----------------------------
    # 2. Run SRIM
    # -----------------------------
    run_dir = run_srim_for_energy(E_true, srim_dir, out_root, number_ions=n_ions)
    df = parse_collisions(run_dir, E_true)
    if df.empty:
        print("[WARN] Empty collision data, skipping.")
        return None

    # -----------------------------
    # 3. Center track (match training)
    # -----------------------------
    df["x"] -= df["x"].mean()
    df["y"] -= df["y"].mean()
    df["z"] -= df["z"].mean()

    # save centered track
    csv_path = Path(run_dir) / f"{E_true:.2f}keV_centered.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # 4. Preprocess → feature vector
    # -----------------------------
    x_obs_new, _, track_ids, _, _ = preprocess_npe(csv_path, n_bins=n_bins)
    x = x_obs_new[0].unsqueeze(0)

    # -----------------------------
    # 5. Sample posterior
    # -----------------------------
    samples = guarded_posterior_sample(
        posterior, x, n_samples=n_post_samples, hard_timeout_sec=180
    )

    if samples is None:
        return {"track_id": track_ids[0], "true_energy": E_true, "status": "SKIPPED"}

    samples = samples.cpu()

    # -----------------------------
    # 6. Summary stats
    # -----------------------------
    mean_pred = samples[:, 0].mean().item()
    std_pred  = samples[:, 0].std().item()

    return {
        "track_id": track_ids[0],
        "true_energy_keV": E_true,
        "pred_energy_mean": mean_pred,
        "pred_energy_std": std_pred,
        "percent_error_abs": 100 * abs(mean_pred - E_true) / E_true,
        "status": "OK",
    }


def evaluate_random_srims_npe(
    posterior,
    srim_dir,
    output_root,
    prior_low,
    prior_high,
    n_random=100,
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
    save_csv=None,
):
    """Run many random NPE evaluations."""

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    results = []
    for i in range(n_random):
        print(f"\n[NPE TEST {i+1}/{n_random}]")
        r = evaluate_one_random_npe(
            posterior,
            prior_low,
            prior_high,
            srim_dir,
            output_root,
            n_ions=n_ions,
            n_post_samples=n_post_samples,
            n_bins=n_bins,
        )
        results.append(r)

    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"[NPE] Saved → {save_csv}")

    return df