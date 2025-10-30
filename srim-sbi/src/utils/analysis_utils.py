import pandas as pd
import torch


def create_observed_dataframe(x_test):
    """
    Convert selected test tracks (torch.Tensor) into a DataFrame
    of observed (ground truth) summary features.

    Parameters
    ----------
    x_test : torch.Tensor
        Selected test tracks of shape (n_tracks, n_features)

    Returns
    -------
    observed_df : pd.DataFrame
        DataFrame with columns:
        ['track_index', 'mean_range', 'long_straggling', 'total_vacancies']
    """
    x_test_np = x_test.numpy()  # Convert to numpy for easier handling

    observed_list = []
    for i, obs in enumerate(x_test_np):
        observed_list.append({
            "track_index": i,
            "mean_range": float(obs[0]),
            "long_straggling": float(obs[1]),
            "total_vacancies": float(obs[2])
        })

    observed_df = pd.DataFrame(observed_list)
    return observed_df


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tqdm import tqdm

def plot_ppc_histograms_per_track(
    df,
    observed,
    output_dir=None,
    bins=30,
    save_plots=True,
    return_metrics=True,
):
    """
    Per-track Posterior Predictive Check (PPC).
    Compares simulated SRIM distributions vs observed summary features per track.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from datetime import datetime
    from tqdm import tqdm

    if "track_id" not in df.columns:
        raise ValueError("DataFrame must include 'track_id' column.")

    features = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    titles   = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for track_id in tqdm(sorted(df["track_id"].unique()), desc="Generating per-track PPC plots"):
        df_track = df[df["track_id"] == track_id]
        obs_dict = observed.get(track_id, None)
        if obs_dict is None:
            print(f"[WARN] No observed values for track_id={track_id}. Skipping.")
            continue

        track_dir = None
        if output_dir:
          # ✅ handle both integer and string-based track IDs (e.g. hashes)
            if isinstance(track_id, (int, np.integer)):
                track_dir = os.path.join(output_dir, f"track_{track_id:04d}")
            else:
               track_dir = os.path.join(output_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)

        for feat, title in zip(features, titles):
            if feat not in df_track.columns:
                continue

            vals = df_track[feat].dropna().to_numpy()
            if vals.size == 0:
                continue

            obs_val = obs_dict.get(feat)
            mu, sigma = np.mean(vals), np.std(vals)
            if not np.isfinite(mu) or not np.isfinite(sigma) or obs_val is None:
                continue

            delta_percent = abs(obs_val - mu) / (mu if mu != 0 else 1) * 100
            z_score = np.nan if sigma == 0 else (obs_val - mu) / sigma

            all_metrics.append({
                "track_id": track_id,
                "feature": feat,
                "mu_sim": mu,
                "sigma_sim": sigma,
                "obs_val": obs_val,
                "delta_percent": delta_percent,
                "z_score": z_score,
            })

            if save_plots and track_dir:
                plt.figure(figsize=(6, 4))
                plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k",
                         density=True, label="Simulated (SRIM)")
                plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
                plt.axvline(obs_val, color="red", linestyle="--", label=f"Obs = {obs_val:.2f}")
                plt.axvline(mu + sigma, color="gray", linestyle=":", alpha=0.4)
                plt.axvline(mu - sigma, color="gray", linestyle=":", alpha=0.4)
                plt.title(f"Track {track_id} — {title}")
                plt.xlabel(title); plt.ylabel("Density")
                plt.text(0.95, 0.90, f"Δ={delta_percent:.1f}% | Z={z_score:.2f}",
                         transform=plt.gca().transAxes, ha="right", va="top",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6))
                plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(track_dir, f"PPC_{feat}_{stamp}.png"))
                plt.close()

    # Aggregate metrics
    if return_metrics and all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        if output_dir:
            out_path = os.path.join(output_dir, f"PPC_metrics_all_tracks_{stamp}.csv")
            metrics_df.to_csv(out_path, index=False)
            print(f"[INFO] Saved per-track metrics → {out_path}")
        print("\n[PPC] Per-track Summary Metrics:")
        print(metrics_df.to_string(index=False))
        return metrics_df
    return None


import pandas as pd

def clean_summary_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean SRIM summary DataFrame for PPC or downstream analysis.
    
    Keeps only the key numeric summary columns:
        ['theta_eV', 'mean_depth_A', 'std_depth_A']

    Performs:
        - Column name normalization (stripped + lowercase)
        - Safe column selection (skips missing ones)
        - Drops rows with NaN in key columns
        - Returns a fresh DataFrame copy
    """
    # --- Normalize column names to lowercase, strip whitespace ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Define the essential columns we care about ---
    keep_cols = ['track_id', 'mean_depth_a', 'std_depth_a', 'vacancies_per_ion']

    # --- Keep only existing columns ---
    existing_cols = [col for col in keep_cols if col in df.columns]
    if not existing_cols:
        raise ValueError(f"None of the expected columns {keep_cols} found in DataFrame.")

    # --- Select + drop NaN rows ---
    df_clean = df[existing_cols].dropna(subset=existing_cols).copy()

    # --- Print a quick summary ---
    print(f"[INFO] Cleaned SRIM summary data: {df_clean.shape[0]} rows, {len(existing_cols)} columns kept.")
    print(f"[INFO] Columns retained: {existing_cols}")

    return df_clean



import numpy as np
import pandas as pd
