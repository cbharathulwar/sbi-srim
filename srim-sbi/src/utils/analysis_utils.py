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


def plot_ppc_histograms_per_track(df, observed, output_dir=None, bins=30, save_plots=True, return_metrics=True):
    """
    Posterior Predictive Check (PPC) — compare SRIM-simulated vs observed summary features,
    per track (not globally).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SRIM summaries. Must include `track_id`, `mean_depth_A`, etc.
    observed : dict[int → dict[str → float]]
        Mapping from track_id to dict of observed summary features (e.g. from x_obs).
    output_dir : str | None
        Folder to save plots and metrics. If None, plots are not saved.
    bins : int
        Number of bins for histograms.
    save_plots : bool
        Whether to save plots as PNGs.
    return_metrics : bool
        Whether to return a DataFrame of Δ%, Z-score per feature per track.

    Returns
    -------
    pd.DataFrame | None
        Per-track summary metrics (Δ%, Z) if return_metrics=True
    """

    if "track_id" not in df.columns:
        raise ValueError("DataFrame must include 'track_id' column for per-track PPC.")

    key_map = {
        "mean_range": "mean_depth_A",
        "long_straggling": "std_depth_A",
        "total_vacancies": "vacancies_per_ion",
    }

    features = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    titles = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    all_metrics = []

    for track_id in sorted(df["track_id"].unique()):
        df_track = df[df["track_id"] == track_id]
        obs_dict = observed.get(track_id, None)
        if obs_dict is None:
            print(f"[WARN] No observed values for track_id={track_id}. Skipping.")
            continue

        obs_mapped = {key_map.get(k, k): v for k, v in obs_dict.items()}
        track_results = []

        # Optional: per-track output folder
        track_dir = os.path.join(output_dir, f"track_{track_id:04d}") if output_dir else None
        if track_dir:
            os.makedirs(track_dir, exist_ok=True)

        for feat, title in zip(features, titles):
            vals = df_track[feat].dropna().values
            if len(vals) == 0:
                continue

            obs_val = obs_mapped.get(feat, None)
            mu, sigma = np.mean(vals), np.std(vals)

            if obs_val is not None and np.isfinite(obs_val) and np.isfinite(mu) and np.isfinite(sigma):
                percent_diff = abs(obs_val - mu) / mu * 100
                z_score = (obs_val - mu) / sigma
                track_results.append({
                    "track_id": track_id,
                    "feature": feat,
                    "Δ%": percent_diff,
                    "Z": z_score,
                    "μ_sim": mu,
                    "σ_sim": sigma,
                    "obs_val": obs_val,
                })

            # Plot histogram with observed and simulated overlays
            if save_plots:
                plt.figure(figsize=(6, 4))
                plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k")
                if obs_val is not None:
                    plt.axvline(obs_val, color="red", linestyle="--", label=f"Observed = {obs_val:.2f}")
                plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
                plt.axvline(mu + sigma, color="gray", linestyle=":", alpha=0.5)
                plt.axvline(mu - sigma, color="gray", linestyle=":", alpha=0.5)

                plt.title(f"Track {track_id} — {title}")
                plt.xlabel(title)
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()

                if track_dir:
                    plt.savefig(os.path.join(track_dir, f"PPC_{feat}.png"))
                plt.close()

        all_metrics.extend(track_results)

    if return_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        if output_dir:
            metrics_df.to_csv(os.path.join(output_dir, "PPC_metrics_all_tracks.csv"), index=False)
        print("\n[PPC] Per-track Summary Metrics:")
        print(metrics_df.to_string(index=False))
        return metrics_df

    return None