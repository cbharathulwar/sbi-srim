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


import pandas as pd
import numpy as np
import torch
from pathlib import Path
import hashlib

def preprocess(data_path):
    """
    Load SRIM data, group by ion and energy, and compute summary statistics
    for each (ion, energy) combination.
    """
    import hashlib
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path

    # --- Load data ---
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df.columns = ["x", "y", "z", "ion", "energy"]

    # ✅ Convert numeric columns safely
    for c in ["x", "y", "z", "energy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows that are fully invalid
    df = df.dropna(subset=["x", "y", "z", "energy"]).reset_index(drop=True)

    grouped = df.groupby(["ion", "energy"], sort=True)
    x_obs, theta, track_ids, ions, energies = [], [], [], [], []

    def make_track_id(ion, energy):
        key = f"{ion}_{energy}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    for (ion, energy), group in grouped:
        track = group[["x", "y", "z"]].values.astype(float)  # ✅ force float
        if track.size == 0 or np.isnan(track).any():
            print(f"[WARN] Skipping invalid track for ion={ion}, energy={energy}")
            continue

        mean_x = np.mean(track[:, 0])
        std_x = np.std(track[:, 0])
        num_vac = track.shape[0]

        x_obs.append([mean_x, std_x, num_vac])
        theta.append([energy])
        ions.append(ion)
        energies.append(energy)
        track_ids.append(make_track_id(ion, energy))

    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    composite_keys = [f"{ion}_{int(energy)}keV" for ion, energy in zip(ions, energies)]

    df_summary = pd.DataFrame({
        "track_id": track_ids,
        "ion": ions,
        "energy_keV": energies,
        "composite_key": composite_keys,
        "mean_depth_A": x_obs[:, 0].numpy(),
        "std_depth_A": x_obs[:, 1].numpy(),
        "vacancies_per_ion": x_obs[:, 2].numpy()
    })

    print(f"[INFO] Preprocessed {len(track_ids)} tracks successfully.")
    print(f"[INFO] Unique track IDs generated via MD5 hash of (ion, energy).")
    return x_obs, theta, track_ids, grouped, df_summary

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_x_test(df_summary, n_per_energy=1):
    """
    Select representative test tracks from the summarized SRIM data.

    Parameters
    ----------
    df_summary : pd.DataFrame
        DataFrame from preprocess(), containing:
        ['ion', 'energy_keV', 'mean_depth_A', 'std_depth_A',
         'vacancies_per_ion', 'track_id', 'composite_key']
    n_per_energy : int
        Number of tracks to sample per unique (ion, energy_keV) group.

    Returns
    -------
    x_test : pd.DataFrame
        Subset of df_summary ready for SRIM simulation and PPC.
    x_test_ids : list[str]
        Ordered list of track IDs corresponding to x_test rows.
    """

    required = {
        "ion", "energy_keV",
        "mean_depth_A", "std_depth_A", "vacancies_per_ion"
    }
    if not required.issubset(df_summary.columns):
        raise ValueError(f"df_summary must contain {required}")

    df_sorted = df_summary.sort_values(["ion", "energy_keV"]).reset_index(drop=True)

    sampled_rows = []
    for (ion, energy), group in df_sorted.groupby(["ion", "energy_keV"]):
        # pick one or up to n_per_energy samples per energy level
        pick = group.sample(n=min(n_per_energy, len(group)), random_state=42)
        sampled_rows.append(pick)

    x_test = pd.concat(sampled_rows).reset_index(drop=True)

    # ensure correct composite_key and track_id exist
    if "composite_key" not in x_test.columns:
        x_test["composite_key"] = x_test.apply(
            lambda r: f"{r['ion']}_{int(r['energy_keV'])}keV", axis=1
        )

    if "track_id" not in x_test.columns:
        import hashlib
        x_test["track_id"] = x_test["composite_key"].apply(
            lambda s: hashlib.md5(s.encode()).hexdigest()[:8]
        )

    x_test_ids = x_test["track_id"].tolist()

    print(f"[INFO] Selected {len(x_test)} test tracks "
          f"({len(x_test['energy_keV'].unique())} unique energies).")

    return x_test, x_test_ids









def plot_ppc_histograms(
    df,
    x_test,
    x_test_ids,
    output_dir=None,
    bins=30,
    save_plots=True,
    return_metrics=True,
):
    """
    Global distribution check between simulated SRIM summaries and observed means.
    (Not per-track PPC, but an aggregated sanity check.)
    """
    import numpy as np, matplotlib.pyplot as plt, os
    from datetime import datetime

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("`df` must be a non-empty DataFrame.")

    # Accept torch or numpy input
    if hasattr(x_test, "detach"):
        x_test_np = x_test.detach().cpu().numpy()
    else:
        x_test_np = np.asarray(x_test)
    if x_test_np.ndim != 2 or x_test_np.shape[1] != 3:
        raise ValueError("`x_test` must have shape (n_tracks, 3).")

    observed_mean = np.mean(x_test_np, axis=0)
    feature_map = {
        "mean_depth_A": observed_mean[0],
        "std_depth_A": observed_mean[1],
        "vacancies_per_ion": observed_mean[2],
    }

    features = ["mean_depth_A", "std_depth_A", "vacancies_per_ion"]
    titles = ["Mean Depth (Å)", "Std. Depth (Å)", "Vacancies per Ion"]

    print(f"[INFO] Global SRIM–Observed Comparison across {len(x_test_ids)} tracks.")
    print(f"[INFO] Using observed global means: {feature_map}")

    all_metrics = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for feat, title in zip(features, titles):
        if feat not in df.columns:
            print(f"[WARN] Missing '{feat}' in SRIM summary; skipping.")
            continue

        vals = df[feat].dropna().values
        if vals.size == 0:
            print(f"[WARN] No data for '{feat}'. Skipping.")
            continue

        mu, sigma = np.mean(vals), np.std(vals)
        obs_val = feature_map[feat]
        delta_percent = abs(obs_val - mu) / (mu if mu != 0 else 1) * 100
        z_score = np.nan if sigma == 0 else (obs_val - mu) / sigma

        all_metrics.append({
            "feature": feat,
            "mu_sim": mu,
            "sigma_sim": sigma,
            "obs_mean": obs_val,
            "delta_percent": delta_percent,
            "z_score": z_score,
            "timestamp": stamp,
        })

        if save_plots:
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, alpha=0.7, edgecolor="k", density=True, label="Simulated")
            plt.axvline(mu, color="blue", linestyle=":", label=f"μ = {mu:.2f}")
            plt.axvline(obs_val, color="red", linestyle="--", label=f"Observed = {obs_val:.2f}")
            plt.title(f"Global Distribution — {title}")
            plt.xlabel(title)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"PPC_global_{feat}_{stamp}.png"))
            plt.close()

    if not return_metrics:
        return None

    metrics_df = pd.DataFrame(all_metrics)
    if output_dir:
        metrics_path = os.path.join(output_dir, f"PPC_metrics_global_{stamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[INFO] Saved metrics → {metrics_path}")

    print("\n[PPC] Global Summary Metrics:")
    print(metrics_df.to_string(index=False))
    return metrics_df

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