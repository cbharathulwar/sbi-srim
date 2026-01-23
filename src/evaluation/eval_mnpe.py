# """
# MNPE Evaluation Utilities
# -------------------------
# - Fixed-energy SRIM evaluation
# - Evaluation from saved per-track CSVs
# - Passive observer for hard negative mining
# """

# # ===============================================================
# # Imports
# # ===============================================================
# import random
# import torch
# import pandas as pd
# import numpy as np
# from pathlib import Path

# from src.utils.data_utils import preprocess_mnpe
# from src.utils.srim_utils import run_srim_for_energy, parse_collisions
# from src.utils.sbi_runner import guarded_posterior_sample

# # ===============================================================
# # GLOBAL STATE (HNM BUFFERS)
# # ===============================================================

# # ALL failed tracks
# failed_raw_rows_all = []
# failed_metadata_all = []
# seen_failed_all = set()

# # CONFIDENT failed tracks
# failed_raw_rows_confident = []
# failed_metadata_confident = []
# seen_failed_confident = set()

# # Confidence threshold (set by runner)
# CONFIDENCE_THRESHOLD = None


# # ===============================================================
# # CONFIG HELPERS
# # ===============================================================
# def set_confidence_threshold(threshold):
#     global CONFIDENCE_THRESHOLD
#     CONFIDENCE_THRESHOLD = threshold


# # ===============================================================
# # OBSERVER (PASSIVE)
# # ===============================================================
# def observe_and_collect_failed_track(
#     df_centered,
#     ion_number,
#     energy,
#     true_parity,
#     posterior_samples,
# ):
#     """
#     Passive observer:
#     - detects parity failure
#     - collects ALL failed tracks
#     - optionally collects CONFIDENT failed tracks
#     """

#     parity_samples = posterior_samples[:, 1].detach().cpu().numpy()
#     votes = parity_samples > 0.5
#     p_hat = votes.mean()

#     pred_parity = 1 if p_hat >= 0.5 else 0
#     if pred_parity == true_parity:
#         return

#     confidence = max(p_hat, 1.0 - p_hat)

#     raw_rows = df_centered[df_centered["ion_number"] == ion_number].copy()
#     if raw_rows.empty:
#         return

#     track_key = (float(energy), int(ion_number), int(true_parity))

#     # ---- ALL failed ----
#     if track_key not in seen_failed_all:
#         failed_raw_rows_all.append(raw_rows)
#         failed_metadata_all.append({
#             "energy": float(energy),
#             "ion_number": int(ion_number),
#             "true_parity": int(true_parity),
#             "pred_parity": int(pred_parity),
#             "confidence": float(confidence),
#             "p_hat": float(p_hat),
#             "n_events": len(raw_rows),
#             "bucket": "all_failed",
#         })
#         seen_failed_all.add(track_key)

#     # ---- CONFIDENT failed ----
#     if CONFIDENCE_THRESHOLD is not None and confidence >= CONFIDENCE_THRESHOLD:
#         if track_key not in seen_failed_confident:
#             failed_raw_rows_confident.append(raw_rows)
#             failed_metadata_confident.append({
#                 "energy": float(energy),
#                 "ion_number": int(ion_number),
#                 "true_parity": int(true_parity),
#                 "pred_parity": int(pred_parity),
#                 "confidence": float(confidence),
#                 "p_hat": float(p_hat),
#                 "n_events": len(raw_rows),
#                 "confidence_threshold": CONFIDENCE_THRESHOLD,
#                 "bucket": "confident_failed",
#             })
#             seen_failed_confident.add(track_key)


# # ===============================================================
# # FIXED-ENERGY SRIM EVALUATION
# # ===============================================================
# def evaluate_fixed_energies_mnpe(
#     posterior,
#     srim_dir,
#     output_root,
#     energies_keV,
#     *,
#     n_ions=200,
#     n_post_samples=500,
#     n_bins=6,
#     save_csv=None,
# ):
#     output_root = Path(output_root)
#     output_root.mkdir(parents=True, exist_ok=True)

#     all_results = []

#     for E in energies_keV:
#         print(f"\n[MNPE] Running SRIM for {E} keV")

#         run_dir = run_srim_for_energy(E, srim_dir, output_root, number_ions=n_ions)
#         df = parse_collisions(run_dir, E)
#         if df.empty:
#             continue

#         ion_ids = sorted(df["ion_number"].unique())
#         random.shuffle(ion_ids)

#         half = len(ion_ids) // 2
#         flipped = set(ion_ids[:half])
#         df["parity"] = df["ion_number"].apply(lambda i: 0 if i in flipped else 1)
#         df.loc[df["parity"] == 0, "x"] *= -1

#         centered = []
#         for ion in ion_ids:
#             part = df[df["ion_number"] == ion].copy()
#             for c in ["x", "y", "z"]:
#                 part[c] -= part[c].mean()
#             centered.append(part)

#         df_centered = pd.concat(centered, ignore_index=True)

#         ion_csvs = {}
#         for ion in ion_ids:
#             out_csv = Path(run_dir) / f"{E:.1f}keV_ion{ion}.csv"
#             df_centered[df_centered["ion_number"] == ion].to_csv(out_csv, index=False)
#             ion_csvs[ion] = out_csv

#         for ion, csv_path in ion_csvs.items():
#             x_obs, _, tids, _, df_sum = preprocess_mnpe(csv_path, n_bins=n_bins)
#             if len(x_obs) != 1:
#                 continue

#             x = x_obs[0].unsqueeze(0)
#             tid = tids[0]
#             true_parity = int(df_sum["parity"].iloc[0])

#             samples = guarded_posterior_sample(posterior, x, n_samples=n_post_samples)
#             if samples is None:
#                 continue

#             observe_and_collect_failed_track(
#                 df_centered, ion, E, true_parity, samples
#             )

#             E_pred = samples[:, 0].mean().item()
#             P_pred = int(samples[:, 1].round().mode()[0].item())

#             all_results.append({
#                 "energy_keV": E,
#                 "ion_number": ion,
#                 "track_id": tid,
#                 "true_parity": true_parity,
#                 "pred_parity": P_pred,
#                 "parity_correct": int(P_pred == true_parity),
#                 "pred_energy_mean": E_pred,
#                 "pred_energy_std": samples[:, 0].std().item(),
#                 "percent_error_abs": 100 * abs(E_pred - E) / E,
#                 "csv_path": str(csv_path),
#                 "status": "OK",
#             })

#     df_out = pd.DataFrame(all_results)
#     if save_csv:
#         df_out.to_csv(save_csv, index=False)

#     return df_out


# # ===============================================================
# # EVALUATION FROM SAVED TRACKS (NO SRIM)
# # ===============================================================
# def evaluate_from_saved_tracks_mnpe(
#     *,
#     posterior,
#     saved_tracks_root,
#     output_csv,
#     n_bins=6,
#     n_post_samples=500,
#     use_observer=False,
# ):
#     saved_tracks_root = Path(saved_tracks_root)
#     all_results = []

#     for csv_path in saved_tracks_root.rglob("*_ion*.csv"):
#         df = pd.read_csv(csv_path)
#         if "energy_keV" in df.columns:
#             df = df.rename(columns={"energy_keV": "energy"})

#         tmp = csv_path.with_suffix(".tmp.csv")
#         df.to_csv(tmp, index=False)

#         x_obs, _, tids, _, df_sum = preprocess_mnpe(tmp, n_bins=n_bins)
#         tmp.unlink(missing_ok=True)

#         if len(x_obs) != 1:
#             continue

#         x = x_obs[0].unsqueeze(0)
#         tid = tids[0]
#         if "energy" in df_sum.columns:
#             true_energy = float(df_sum["energy"].iloc[0])
#         elif "energy_keV" in df_sum.columns:
#             true_energy = float(df_sum["energy_keV"].iloc[0])
#         else:
#             raise KeyError(f"No energy column found in df_summary: {df_sum.columns}")
#         true_parity = int(df_sum["parity"].iloc[0])
#         # Extract ion_number from filename: *_ion{N}.csv
#         stem = csv_path.stem          # e.g. "4.0keV_ion0"
#         if "_ion" in stem:
#             ion_number = int(stem.split("_ion")[-1])
#         else:
#             ion_number = -1  # fallback, should never happen

#         samples = guarded_posterior_sample(posterior, x, n_samples=n_post_samples, hard_timeout_sec=120)
#         if samples is None:
#             continue

#         if use_observer:
#             observe_and_collect_failed_track(df, ion_number, true_energy, true_parity, samples)

#         E_pred = samples[:, 0].mean().item()
#         P_pred = int(samples[:, 1].round().mode()[0].item())

#         all_results.append({
#             "energy_keV": true_energy,
#             "ion_number": ion_number,
#             "track_id": tid,
#             "true_parity": true_parity,
#             "pred_parity": P_pred,
#             "parity_correct": int(P_pred == true_parity),
#             "pred_energy_mean": E_pred,
#             "pred_energy_std": samples[:, 0].std().item(),
#             "percent_error_abs": 100 * abs(E_pred - true_energy) / true_energy,
#             "csv_path": str(csv_path),
#             "status": "OK",
#         })

#     df_out = pd.DataFrame(all_results)
#     df_out.to_csv(output_csv, index=False)
#     return df_out






import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import guarded_posterior_sample

# ===============================================================
# SIIMPL BATCH EVALUATION (The New Standard)
# ===============================================================
import pandas as pd
import torch
from src.utils.data_utils import preprocess_mnpe

import pandas as pd
import torch
import numpy as np
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import guarded_posterior_sample, get_device

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
    print(f"📊 Final Accuracy: {acc:.2f}%")
    print(f"✅ Saved merged results to: {output_csv_path}")