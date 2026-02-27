"""
MNPE Evaluation Utilities
-------------------------
- SIIMPL batch evaluation (standard)
- Confidence-based evaluation with threshold sweeping
- Efficiency vs. Accuracy curve generation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import guarded_posterior_sample, get_device


# ===============================================================
# SIIMPL BATCH EVALUATION (The New Standard)
# ===============================================================

def evaluate_siimpl_batch(posterior, eval_csv_path, output_csv_path, n_bins=6):
    print(f"[EVAL] Loading eval dataset: {eval_csv_path}")
    device = get_device()

    # 1. Get tensors AND the physics summary dataframe
    x_eval, theta_eval, tids, _, df_physics = preprocess_mnpe(eval_csv_path, n_bins=n_bins)

    predictions = []
    confidences = []
    pred_energies = []

    print(f"[EVAL] Evaluating {len(x_eval)} tracks...")

    # 2. SAMPLING LOOP
    for i in range(len(x_eval)):
        # Draw 100 samples from the posterior for this track
        samples = guarded_posterior_sample(posterior, x_eval[i].unsqueeze(0), n_samples=100)

        if samples is not None:
            # MNPE Order: [Energy (0), Parity (1)]
            energy_samples = samples[:, 0].numpy()
            parity_samples = samples[:, 1].numpy().astype(int)

            # Use mode for Parity (Discrete)
            vals, counts = np.unique(parity_samples, return_counts=True)
            mode_idx = np.argmax(counts)

            predictions.append(vals[mode_idx])
            confidences.append(counts[mode_idx] / 100.0) # Fraction of samples in the mode
            pred_energies.append(np.mean(energy_samples))
        else:
            # Fallback for failed samples
            predictions.append(-1)
            confidences.append(0.0)
            pred_energies.append(0.0)

        if (i + 1) % 100 == 0:
            print(f"   Evaluated {i+1}/{len(x_eval)}...")

    # 3. Build the prediction dataframe
    results_df = pd.DataFrame({
        'track_id': tids,
        'true_energy': theta_eval[:, 0].numpy(),
        'true_parity': theta_eval[:, 1].numpy().astype(int),
        'pred_parity': predictions,
        'confidence': confidences,
        'pred_energy': pred_energies
    })

    # 4. MERGE: Add physics features (skew, var, etc.) to the results
    # This prevents the KeyError: 'skew_z' in your notebook
    final_df = pd.merge(results_df, df_physics, on='track_id')

    final_df.to_csv(output_csv_path, index=False)

    # Calculate and print final accuracy for the console
    acc = (final_df['true_parity'] == final_df['pred_parity']).mean() * 100
    print(f"Final Accuracy: {acc:.2f}%")
    print(f"Saved merged results to: {output_csv_path}")


# ===============================================================
# CONFIDENCE-BASED EVALUATION
# ===============================================================

def compute_mnpe_confidence(samples):
    """
    Calculates the confidence score based on parity sample agreement.
    """
    # Round samples to ensure they are discrete 0 or 1
    parity_samples = torch.round(samples[:, 1])

    # Calculate the proportion of samples that voted for Parity 1
    p1_fraction = parity_samples.mean().item()

    # Majority vote determines the prediction
    if p1_fraction >= 0.5:
        pred_parity = 1
        confidence = p1_fraction
    else:
        pred_parity = 0
        confidence = 1.0 - p1_fraction

    return pred_parity, confidence

def evaluate_mnpe_with_confidence(
    posterior,
    eval_csv_path,
    output_csv,
    n_post_samples=500,
    n_bins=6,
    norm_stats=None
):
    x_obs, theta, track_ids, _, _ = preprocess_mnpe(
        eval_csv_path, n_bins=n_bins, norm_stats=norm_stats
    )
    results = []

    print(f"[INFO] Running inference on {len(x_obs)} tracks...")

    with torch.no_grad():
        for i in tqdm(range(len(x_obs)), desc="Sampling Posteriors", unit="track"):
            x = x_obs[i].unsqueeze(0).to("cpu")
            true_energy, true_parity = theta[i].numpy()

            samples = guarded_posterior_sample(
                posterior, x, n_samples=n_post_samples
            )

            if samples is None:
                continue

            samples = samples.cpu()

            pred_parity, confidence = compute_mnpe_confidence(samples)
            pred_energy_mean = samples[:, 0].mean().item()
            energy_error_abs = abs(pred_energy_mean - true_energy)

            results.append({
                "track_id": track_ids[i],
                "true_energy": true_energy,
                "true_parity": int(true_parity),
                "pred_parity": pred_parity,
                "confidence": confidence,
                "is_correct": int(pred_parity == true_parity),
                "pred_energy_mean": pred_energy_mean,
                "energy_error": energy_error_abs
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\n[DONE] Results with confidence saved to {output_csv}")
    return df_results


# ===============================================================
# EFFICIENCY vs. ACCURACY CURVE GENERATION
# ===============================================================

def generate_mnpe_curve_data(results_df, energy_min, energy_max, total_events):
    """
    Generates Efficiency vs. Accuracy data points by sweeping confidence.
    """
    bin_df = results_df[
        (results_df['true_energy'] >= energy_min) &
        (results_df['true_energy'] < energy_max)
    ]

    curve_points = []

    # Sweep confidence threshold from 0.5 (keep all) to 0.99 (super strict)
    for threshold in np.linspace(0.5, 0.99, 50):
        kept_tracks = bin_df[bin_df['confidence'] >= threshold]

        if len(kept_tracks) == 0: continue

        # Calculate X and Y for your plot (using absolute total_events to prevent survivorship bias)
        efficiency = len(kept_tracks) / total_events
        accuracy = kept_tracks['is_correct'].mean()

        curve_points.append({
            "threshold": threshold,
            "efficiency": efficiency,
            "accuracy": accuracy
        })

    return pd.DataFrame(curve_points)


def generate_curve_data_ranked(bin_df, score_col, total_events):
    """Works for both MNPE (score_col='confidence') and
    Rajendran (score_col='asymmetry')."""
    bin_df = bin_df.sort_values(score_col, ascending=False)

    correct = bin_df['is_correct'].values
    cumsum = np.cumsum(correct)

    curve_points = pd.DataFrame({
        "efficiency": np.arange(1, len(bin_df) + 1) / total_events,
        "accuracy": cumsum / np.arange(1, len(bin_df) + 1),
    })

    return curve_points
