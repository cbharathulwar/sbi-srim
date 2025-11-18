# src/evaluation/random_eval_mnpe.py

"""
Random SRIM Evaluation for MNPE (energy + parity)
--------------------------------------------------
This evaluates your 2D posterior:

        θ = [energy_keV, parity]

Steps:
    1. Sample energy & parity
    2. Run SRIM
    3. Inject parity flip into X coords
    4. Center track
    5. Preprocess with preprocess_mnpe
    6. Posterior sampling
"""

import os
import time
import random
import torch
import pandas as pd
from pathlib import Path

from src.utils.data_utils import preprocess_mnpe
from src.utils.srim_utils import run_srim_for_energy, parse_collisions
from src.utils.sbi_runner import guarded_posterior_sample




from src.utils.sbi_runner import sample_energy, guarded_posterior_sample

def evaluate_one_random_mnpe(
    posterior,
    prior_low,
    prior_high,
    srim_dir,
    out_root,
    *,
    sample_mode="continuous",   # NEW
    step=None,                  # NEW
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
):
    """Evaluate one random SRIM sample (MNPE: energy + parity)."""

    # ------------------------------------------------------
    # 1. Draw true parameters (now controlled by pipeline)
    # ------------------------------------------------------
    E_true = sample_energy(
        low=prior_low,
        high=prior_high,
        mode=sample_mode,
        step=step,
    )
    P_true = random.choice([0, 1])
    print(f"[INFO][MNPE] True energy = {E_true:.3f} keV, parity = {P_true}")

    # ------------------------------------------------------
    # 2. Run SRIM
    # ------------------------------------------------------
    run_dir = run_srim_for_energy(E_true, srim_dir, out_root, number_ions=n_ions)
    df = parse_collisions(run_dir, E_true)
    if df.empty:
        return None

    # ------------------------------------------------------
    # 3. Inject parity flip
    # ------------------------------------------------------
    if P_true == 0:
        df["x"] *= -1
    df["parity"] = P_true

    # ------------------------------------------------------
    # 4. Center track (match training)
    # ------------------------------------------------------
    df["x"] -= df["x"].mean()
    df["y"] -= df["y"].mean()
    df["z"] -= df["z"].mean()

    csv_path = Path(run_dir) / f"{E_true:.3f}keV_p{P_true}.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------
    # 5. Extract MNPE features
    # ------------------------------------------------------
    x_obs_new, _, track_ids, _, _ = preprocess_mnpe(csv_path, n_bins=n_bins)
    x = x_obs_new[0].unsqueeze(0)

    # ------------------------------------------------------
    # 6. Guarded posterior sampling
    # ------------------------------------------------------
    samples = guarded_posterior_sample(
        posterior,
        x,
        n_samples=n_post_samples,
        hard_timeout_sec=180,
    )

    if samples is None:
        return {
            "track_id": track_ids[0],
            "true_energy": E_true,
            "true_parity": P_true,
            "status": "SKIPPED",
        }

    samples = samples.cpu()

    # ------------------------------------------------------
    # 7. Decode posterior
    # ------------------------------------------------------
    E_pred = samples[:, 0].mean().item()
    E_std  = samples[:, 0].std().item()

    P_pred = int(samples[:, 1].round().mode()[0].item())
    parity_correct = int(P_pred == P_true)

    return {
        "track_id": track_ids[0],
        "true_energy_keV": E_true,
        "true_parity": P_true,
        "pred_energy_mean": E_pred,
        "pred_energy_std": E_std,
        "pred_parity": P_pred,
        "parity_correct": parity_correct,
        "percent_error_abs": 100 * abs(E_pred - E_true) / E_true,
        "status": "OK",
    }


def evaluate_random_srims_mnpe(
    posterior,
    srim_dir,
    output_root,
    prior_low,
    prior_high,
    n_random=100,
    *,
    sample_mode="continuous",   # NEW
    step=None,                  # NEW
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
    save_csv=None,
):
    """Run many random MNPE evaluations with controlled sampling."""

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    results = []
    for i in range(n_random):
        print(f"\n[MNPE TEST {i+1}/{n_random}]")

        r = evaluate_one_random_mnpe(
            posterior=posterior,
            prior_low=prior_low,
            prior_high=prior_high,
            srim_dir=srim_dir,
            out_root=output_root,
            sample_mode=sample_mode,   # NEW
            step=step,                 # NEW
            n_ions=n_ions,
            n_post_samples=n_post_samples,
            n_bins=n_bins,
        )
        results.append(r)

    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"[MNPE] Saved → {save_csv}")

    return df