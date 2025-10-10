import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
import json
from datetime import datetime
import matplotlib as plt

from pathlib import Path
import pandas as pd
import numpy as np
import torch


def preprocess(data_path):
    """
    Load SRIM data, group by ion and energy, and compute summary statistics
    for each (ion, energy) combination.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV file containing SRIM data.
        Expected columns: ["x", "y", "z", "ion", "energy"]

    Returns
    -------
    x_obs : torch.Tensor
        Tensor of summary features [mean_depth_a, std_depth_a, vacancies_per_ion].
    theta : torch.Tensor
        Tensor of corresponding energies.
    track_ids : list[int]
        Deterministic list of unique track identifiers (one per group).
    grouped : pd.core.groupby.DataFrameGroupBy
        Grouped DataFrame (for inspection).
    df_summary : pd.DataFrame
        Summary DataFrame of all tracks with feature columns.
    """

    # --- Load data ---
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df.columns = ["x", "y", "z", "ion", "energy"]

    grouped = df.groupby(["ion", "energy"], sort=True)

    x_obs, theta, track_ids = [], [], []

    for (ion, energy), group in grouped:
        track = group[["x", "y", "z"]].values
        if len(track) == 0 or np.isnan(track).any():
            print(f"[WARN] Skipping invalid track for ion={ion}, energy={energy}")
            continue

        mean_x = np.mean(track[:, 0])
        std_x = np.std(track[:, 0])
        num_vac = track.shape[0]

        x_obs.append([mean_x, std_x, num_vac])
        theta.append([energy])
        track_ids.append(len(track_ids))  # deterministic id

    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)

    # Optional sanity check
    if torch.max(x_obs[:, 0]) > 1e4:
        print("[WARN] Large depth values detected (>10,000). Check SRIM unit scale (Å vs nm).")

    # Create human-readable summary
    df_summary = pd.DataFrame({
        "track_id": track_ids,
        "mean_depth_a": x_obs[:, 0].numpy(),
        "std_depth_a": x_obs[:, 1].numpy(),
        "vacancies_per_ion": x_obs[:, 2].numpy(),
        "energy": theta[:, 0].numpy()
    })

    print(f"[INFO] Preprocessed {len(track_ids)} tracks successfully.")
    print(f"[INFO] Features computed: mean_depth_a, std_depth_a, vacancies_per_ion")

    return x_obs, theta, track_ids, grouped, df_summary

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_ppc_histograms(df, x_test, x_test_ids, output_dir=None, bins=30, save_plots=True, return_metrics=True):
    """
    Posterior Predictive Check (PPC) — GLOBAL comparison between SRIM-simulated
    and observed summary features (aggregated across all selected tracks).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned SRIM summary containing columns ['mean_depth_a', 'std_depth_a', 'vacancies_per_ion'].
    x_test : torch.Tensor
        Observed tensor, shape (n_tracks, 3) → [mean_x, std_x, num_vacancies].
    x_test_ids : list[int]
        Track IDs corresponding to x_test rows (for logging/reference).
    output_dir : str | None
        Folder to save plots and metrics.
    bins : int
        Number of bins for histograms.
    save_plots : bool
        Whether to save plots as PNGs.
    return_metrics : bool
        Whether to return a DataFrame of Δ%, Z-score per feature.

    Returns
    -------
    pd.DataFrame | None
        Global metrics (Δ%, Z-score per feature).
    """

    # ---- Setup ----
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("`df` is empty.")
    if x_test.ndim != 2 or x_test.shape[1] != 3:
        raise ValueError("`x_test` must be (n_tracks, 3).")

    # ---- Convert x_test tensor to numpy ----
    x_test_np = x_test.detach().cpu().numpy()

    # Compute mean across all observed tracks → global "observed mean"
    observed_mean = np.mean(x_test_np, axis=0)
    feature_map = {
        "mean_depth_a": observed_mean[0],
        "std_depth_a": observed_mean[1],
        "vacancies_per_ion": observed_mean[2],
    }

    features = ["mean_depth_a", "std_depth_a", "vacancies_per_ion"]
    titles = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    print(f"[INFO] Performing GLOBAL PPC across {len(x_test_ids)} tracks.")
    print(f"[INFO] Using global observed means: {feature_map}")

    # ---- Compute metrics and plot ----
    all_metrics = []
    for feat, title in zip(features, titles):
        if feat not in df.columns:
            print(f"[WARN] Missing column '{feat}' in SRIM summary. Skipping.")
            continue

        vals = df[feat].dropna().values
        if len(vals) == 0:
            print(f"[WARN] No SRIM values for '{feat}'. Skipping.")
            continue

        obs_val = feature_map[feat]
        mu, sigma = np.mean(vals), np.std(vals)

        # Compute metrics
        delta_percent = abs(obs_val - mu) / mu * 100
        z_score = (obs_val - mu) / sigma

        all_metrics.append({
            "Feature": feat,
            "μ_sim": mu,
            "σ_sim": sigma,
            "obs_mean": obs_val,
            "Δ%": delta_percent,
            "Z": z_score,
        })

        # ---- Plot global histogram ----
        if save_plots:
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k", label="Simulated (SRIM)")
            plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
            plt.axvline(mu + sigma, color="gray", linestyle=":", alpha=0.5)
            plt.axvline(mu - sigma, color="gray", linestyle=":", alpha=0.5)
            plt.axvline(obs_val, color="red", linestyle="--", label=f"Observed Mean = {obs_val:.2f}")

            plt.title(f"Global PPC — {title}")
            plt.xlabel(title)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f"PPC_global_{feat}.png"))
            plt.close()

    # ---- Save metrics ----
    if return_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        if output_dir:
            metrics_path = os.path.join(output_dir, "PPC_metrics_global.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"[INFO] Global PPC metrics saved → {metrics_path}")

        print("\n[PPC] Global Summary Metrics:")
        print(metrics_df.to_string(index=False))
        return metrics_df

    return None

def tensor_to_observed_dict(x_test, x_test_ids):
    """
    Convert x_test tensor + corresponding track IDs into dict format
    expected by PPC plotting functions.
    """
    observed = {}
    for track_id, values in zip(x_test_ids, x_test.tolist()):
        observed[track_id] = {
            "mean_depth_a": float(values[0]),
            "std_depth_a": float(values[1]),
            "vacancies_per_ion": float(values[2])
        }
    return observed