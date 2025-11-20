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
    sample_mode="continuous",
    step=None,
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
):
    """
    Evaluate MNPE for ONE randomly sampled energy + 200 tracks.
    Parity is assigned PER ION (correct), tracks are centered PER ION.
    """

    # ------------------------------------------------------
    # 1. Draw true parameters
    # ------------------------------------------------------
    E_true = sample_energy(
        low=prior_low,
        high=prior_high,
        mode=sample_mode,
        step=step,
    )

    print(f"[INFO][MNPE] True energy = {E_true:.3f} keV")

    # ------------------------------------------------------
    # 2. Run SRIM for this energy
    # ------------------------------------------------------
    run_dir = run_srim_for_energy(E_true, srim_dir, out_root, number_ions=n_ions)
    df = parse_collisions(run_dir, E_true)

    if df.empty:
        print("[WARN] Empty SRIM output.")
        return None

    # ------------------------------------------------------
    # 3. Assign parity PER ION (correct)
    # ------------------------------------------------------
    ion_ids = sorted(df["ion_number"].unique().tolist())
    par_assign = {ion: random.choice([0, 1]) for ion in ion_ids}

    df["parity"] = df["ion_number"].map(par_assign)

    # Apply flip for parity 0 ions
    df.loc[df["parity"] == 0, "x"] *= -1

    # ------------------------------------------------------
    # 4. Center EACH ION SEPARATELY
    # ------------------------------------------------------
    centered_tracks = []
    for ion in ion_ids:
        g = df[df["ion_number"] == ion].copy()
        g["x"] -= g["x"].mean()
        g["y"] -= g["y"].mean()
        g["z"] -= g["z"].mean()
        centered_tracks.append(g)

    df_centered = pd.concat(centered_tracks, ignore_index=True)

    # Save combined CSV for preprocessing
    csv_path = Path(run_dir) / f"{E_true:.3f}keV_random.csv"
    df_centered.to_csv(csv_path, index=False)

    # ------------------------------------------------------
    # 5. Preprocess ALL 200 tracks
    # ------------------------------------------------------
    x_obs_all, _, track_ids, _, _ = preprocess_mnpe(csv_path, n_bins=n_bins)

    if len(x_obs_all) != n_ions:
        print(f"[WARN] Expected {n_ions} tracks but found {len(x_obs_all)}")

    # ------------------------------------------------------
    # 6. Run posterior for each track
    # ------------------------------------------------------
    all_results = []

    for i in range(len(x_obs_all)):
        x = x_obs_all[i].unsqueeze(0)
        true_par = par_assign[ion_ids[i]]

        samples = guarded_posterior_sample(
            posterior,
            x,
            n_samples=n_post_samples,
            hard_timeout_sec=180,
        )

        if samples is None:
            all_results.append({
                "track_id": track_ids[i],
                "true_energy_keV": E_true,
                "true_parity": true_par,
                "status": "SKIPPED",
            })
            continue

        # Convert samples back to tensor if numpy
        E_pred = samples[:, 0].mean().item()
        E_std  = samples[:, 0].std().item()
        P_pred = int(samples[:, 1].round().mode()[0].item())

        all_results.append({
            "track_id": track_ids[i],
            "true_energy_keV": E_true,
            "true_parity": true_par,
            "pred_energy_mean": E_pred,
            "pred_energy_std": E_std,
            "pred_parity": P_pred,
            "parity_correct": int(P_pred == true_par),
            "percent_error_abs": 100 * abs(E_pred - E_true) / E_true,
            "status": "OK",
            "csv_path": str(csv_path),
        })

    return all_results

def evaluate_random_srims_mnpe(
    posterior,
    srim_dir,
    output_root,
    prior_low,
    prior_high,
    n_random=100,
    *,
    sample_mode="continuous",
    step=None,
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
    save_csv=None,
):
    """Run many random MNPE evaluations with controlled sampling (ALL ions)."""

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    all_results = []   # <-- FLATTENED LIST OF DICTS

    for i in range(n_random):
        print(f"\n[MNPE TEST {i+1}/{n_random}]")

        batch_results = evaluate_one_random_mnpe(
            posterior=posterior,
            prior_low=prior_low,
            prior_high=prior_high,
            srim_dir=srim_dir,
            out_root=output_root,
            sample_mode=sample_mode,
            step=step,
            n_ions=n_ions,
            n_post_samples=n_post_samples,
            n_bins=n_bins,
        )

        if batch_results is None:
            continue

        # batch_results is a LIST of ~200 track dicts
        all_results.extend(batch_results)

    # Build DataFrame
    df = pd.DataFrame(all_results)

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"[MNPE] Saved → {save_csv}")

    return df

# ===============================================================
# FIXED-ENERGY MNPE EVALUATION (correct scientific evaluation)
# ===============================================================
import os
import random
import pandas as pd
from pathlib import Path
import torch

from src.utils.srim_utils import run_srim_for_energy, parse_collisions
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import guarded_posterior_sample


def evaluate_fixed_energies_mnpe(
    posterior,
    srim_dir,
    output_root,
    energies_keV=[1, 3, 10, 30, 100],
    *,
    n_ions=200,
    n_post_samples=500,
    n_bins=6,
    save_csv=None,
):
    """
    Evaluate MNPE model over FIXED energies — NOT random sampling.
    
    Steps:
        • For each E in energies_keV:
            - Run SRIM once → 200 ions
            - Parse collisions → 200 tracks
            - Assign parity: flip x for 50% of tracks
            - Preprocess each track
            - Evaluate posterior for each track
            - Store predictions
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    # -----------------------------------------------------------
    # MAIN LOOP OVER ENERGIES
    # -----------------------------------------------------------
    for E in energies_keV:

        print(f"\n===========================================")
        print(f"[MNPE] Running SRIM for fixed energy {E} keV")
        print(f"===========================================\n")

        # Run SRIM (ONE RUN) with 200 ions
        run_dir = run_srim_for_energy(E, srim_dir, output_root, number_ions=n_ions)

        # Parse all collision cascades
        df = parse_collisions(run_dir, E)
        if df.empty:
            print(f"[WARN] No SRIM output for {E} keV")
            continue

        # -----------------------------------------------------------
        # CREATE PARITY LABELS
        # -----------------------------------------------------------
        ion_ids = sorted(df["ion_number"].unique().tolist())
        random.shuffle(ion_ids)

        half = len(ion_ids) // 2
        flipped = set(ion_ids[:half])     # 50% flipped = parity 0
        unflipped = set(ion_ids[half:])   # 50% not flipped = parity 1

        df["parity"] = df["ion_number"].apply(lambda t: 0 if t in flipped else 1)

        # Apply parity flip
        df.loc[df["parity"] == 0, "x"] *= -1

        # Center each ion’s track (important)
        centered = []
        for ion in ion_ids:
            part = df[df["ion_number"] == ion].copy()
            part["x"] -= part["x"].mean()
            part["y"] -= part["y"].mean()
            part["z"] -= part["z"].mean()
            centered.append(part)

        df = pd.concat(centered, ignore_index=True)

        # Save combined CSV (optional)
        combined_csv = Path(run_dir) / f"{E:.1f}keV_alltracks.csv"
        df.to_csv(combined_csv, index=False)

        # -----------------------------------------------------------
        # Preprocess for MNPE model
        # -----------------------------------------------------------
        x_obs_all, _, track_ids, _, _ = preprocess_mnpe(combined_csv, n_bins=n_bins)

        # -----------------------------------------------------------
        # POSTERIOR EVALUATION PER TRACK
        # -----------------------------------------------------------
        for i in range(len(x_obs_all)):
            print(f"[Track {i+1}/{len(x_obs_all)} @ {E} keV]")

            x = x_obs_all[i].unsqueeze(0)
            true_parity = int(df[df["ion_number"] == i]["parity"].iloc[0])

            samples = guarded_posterior_sample(
                posterior,
                x,
                n_samples=n_post_samples,
                hard_timeout_sec=120,
            )

            if samples is None:
                all_results.append({
                    "energy_keV": E,
                    "ion_number": i,
                    "true_parity": true_parity,
                    "status": "SKIPPED",
                })
                continue

            samples = samples.cpu()

            E_pred = samples[:, 0].mean().item()
            E_std  = samples[:, 0].std().item()

            P_pred = int(samples[:, 1].round().mode()[0].item())
            correct = int(P_pred == true_parity)

            all_results.append({
                "energy_keV": E,
                "ion_number": i,
                "true_parity": true_parity,
                "pred_parity": P_pred,
                "parity_correct": correct,
                "pred_energy_mean": E_pred,
                "pred_energy_std": E_std,
                "csv_path": str(combined_csv),
                "status": "OK",
            })

    # -----------------------------------------------------------
    # FINAL SAVE
    # -----------------------------------------------------------
    df_out = pd.DataFrame(all_results)

    if save_csv:
        df_out.to_csv(save_csv, index=False)
        print(f"[MNPE] Final results saved → {save_csv}")

    print("\n[MNPE] Completed fixed-energy evaluation.\n")
    return df_out