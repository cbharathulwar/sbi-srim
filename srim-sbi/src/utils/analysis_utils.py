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

def plot_ppc_histograms_per_track(df, observed, output_dir=None, bins=30, save_plots=True, return_metrics=True):
    """
    Posterior Predictive Check (PPC) — per-track comparison between SRIM-simulated 
    and observed summary features.

    Parameters
    ----------
    df : pd.DataFrame
        SRIM summary with columns ['track_id', 'mean_depth_a', 'std_depth_a', 'vacancies_per_ion'].
    observed : dict[int → dict[str → float]]
        Observed features per track, e.g. from x_test tensor converted via tensor_to_observed_dict().
    output_dir : str | None
        Folder to save per-track plots and metrics.
    bins : int
        Number of bins for histograms.
    save_plots : bool
        Whether to save plots.
    return_metrics : bool
        Whether to return a DataFrame of metrics.

    Returns
    -------
    pd.DataFrame | None
        Per-track metrics (Δ%, Z-score, μ_sim, σ_sim, obs_val).
    """

    if "track_id" not in df.columns:
        raise ValueError("DataFrame must include 'track_id' column.")

    features = ["mean_depth_a", "std_depth_a", "vacancies_per_ion"]
    titles = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    all_metrics = []

    for track_id in tqdm(sorted(df["track_id"].unique()), desc="Generating per-track PPC plots"):
        df_track = df[df["track_id"] == track_id]
        obs_dict = observed.get(track_id, None)
        if obs_dict is None:
            print(f"[WARN] No observed values for track_id={track_id}. Skipping.")
            continue

        # Create per-track output folder
        track_dir = os.path.join(output_dir, f"track_{track_id:04d}")
        os.makedirs(track_dir, exist_ok=True)

        track_metrics = []

        for feat, title in zip(features, titles):
            if feat not in df_track.columns:
                print(f"[WARN] Missing column '{feat}' in SRIM DataFrame for track {track_id}. Skipping.")
                continue

            vals = df_track[feat].dropna().values
            if len(vals) == 0:
                continue

            obs_val = obs_dict.get(feat, None)
            mu, sigma = np.mean(vals), np.std(vals)

            if obs_val is not None and np.isfinite(mu) and np.isfinite(sigma):
                delta_percent = abs(obs_val - mu) / mu * 100
                z_score = (obs_val - mu) / sigma

                track_metrics.append({
                    "track_id": track_id,
                    "feature": feat,
                    "μ_sim": mu,
                    "σ_sim": sigma,
                    "obs_val": obs_val,
                    "Δ%": delta_percent,
                    "Z": z_score
                })

            # --- Plot histogram ---
            if save_plots:
                plt.figure(figsize=(6, 4))
                plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k", label="Simulated (SRIM)")
                plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
                plt.axvline(mu + sigma, color="gray", linestyle=":", alpha=0.5)
                plt.axvline(mu - sigma, color="gray", linestyle=":", alpha=0.5)
                if obs_val is not None:
                    plt.axvline(obs_val, color="red", linestyle="--", label=f"Observed = {obs_val:.2f}")

                plt.title(f"Track {track_id} — {title}")
                plt.xlabel(title)
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(track_dir, f"PPC_{feat}.png"))
                plt.close()

        # Save per-track metrics CSV
        if track_metrics:
            df_track_metrics = pd.DataFrame(track_metrics)
            df_track_metrics.to_csv(os.path.join(track_dir, "metrics_per_track.csv"), index=False)
            all_metrics.extend(track_metrics)

    # Combine metrics across all tracks
    if return_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        if output_dir:
            metrics_df.to_csv(os.path.join(output_dir, "PPC_metrics_all_tracks.csv"), index=False)
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

