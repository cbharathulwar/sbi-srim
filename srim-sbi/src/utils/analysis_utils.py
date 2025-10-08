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
import matplotlib.pyplot as plt
import pandas as pd

def plot_ppc_histograms(df, observed, output_dir=None, bins=30, save_plots=True, return_metrics=True):
    """
    Posterior Predictive Check (PPC) — compare SRIM-simulated vs observed summary features.
    """

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    key_map = {
        "mean_range": "mean_depth_A",
        "long_straggling": "std_depth_A",
        "total_vacancies": "vacancies_per_ion",
    }

    observed_mapped = {key_map.get(k, k): v for k, v in observed.items()}
    features = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    titles = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    results = []

    for feat, title in zip(features, titles):
        vals = df[feat].dropna().values
        if len(vals) == 0:
            continue

        obs_val = observed_mapped.get(feat, None)
        mu, sigma = np.mean(vals), np.std(vals)

        if obs_val is not None and np.isfinite(obs_val) and np.isfinite(mu) and np.isfinite(sigma):
            percent_diff = abs(obs_val - mu) / mu * 100
            z_score = (obs_val - mu) / sigma
            results.append({"Feature": feat, "Δ%": percent_diff, "Z": z_score})

        if save_plots:
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k")
            if obs_val is not None:
                plt.axvline(obs_val, color="red", linestyle="--", label=f"Observed = {obs_val:.2f}")
            plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
            plt.axvline(mu + sigma, color="gray", linestyle=":", alpha=0.5)
            plt.axvline(mu - sigma, color="gray", linestyle=":", alpha=0.5)

            plt.title(f"PPC for {title}")
            plt.xlabel(title)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f"PPC_{feat}.png"))
            plt.close()

    if return_metrics:
        metrics_df = pd.DataFrame(results)
        print("\n[PPC] Summary Metrics:")
        print(metrics_df.to_string(index=False))

        if output_dir:
            metrics_df.to_csv(os.path.join(output_dir, "PPC_metrics.csv"), index=False)

        return metrics_df
    return None