import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
import json
from datetime import datetime
import matplotlib as plt

def preprocess(data_path):
    """
    Load SRIM data, group by ion and energy, and find summary statistics
    for each (ion, energy) combination.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the CSV file containing SRIM data.
        Expected columns: ["x", "y", "z", "ion", "energy"]
    
    Returns
    -------
    x_obs : torch.Tensor
        Tensor of summary features [mean_x,  std_x, num_vacancies] for each group.
    theta : torch.Tensor
        Tensor of corresponding energy (and optionally other parameters).
    grouped : pd.core.groupby.DataFrameGroupBy
        Grouped DataFrame (for debugging or inspection).
    """
    
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df.columns = ["x", "y", "z", "ion", "energy"]
    grouped = df.groupby(["ion", "energy"])

    x_obs = []
    theta = []

    for (ion, energy), group in grouped:
        # group → subset of all rows with this ion-energy pair
        track = group[["x", "y", "z"]].values

        mean_x = np.mean(track[:, 0])
        std_x = np.std(track[:, 0])
        num_vac = track.shape[0]

        x_obs.append([mean_x, std_x, num_vac])
        theta.append([energy])

    # Convert to torch tensors
    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)

    return x_obs, theta, grouped

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