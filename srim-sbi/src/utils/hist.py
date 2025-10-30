# =============================================
# scalar_ppc.py — Posterior Predictive Check (Scalar)
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

# ---------------------------
# 1️⃣ Data Preparation
# ---------------------------


def group_simulated_scalars(df_post: pd.DataFrame) -> dict:
    """
    Group SRIM simulation results into per-track scalar lists.

    Parameters
    ----------
    df_post : pd.DataFrame
        Output from summarize_all_runs(), with multiple SRIM runs per track.

    Returns
    -------
    dict
        { track_id: { 'mean_depth_A': [...], 'max_depth_A': [...], 'vacancies_per_ion': [...] } }
    """
    required = ["track_id", "mean_depth_A", "vacancies_per_ion", "max_depth_A"]
    for c in required:
        if c not in df_post.columns:
            raise ValueError(f"Missing required column: {c}")

    sim_dict = {}
    for track_id, g in df_post.groupby("track_id", sort=False):
        sim_dict[track_id] = {
            "mean_depth_A": g["mean_depth_A"].dropna().astype(float).tolist(),
            "max_depth_A": g["max_depth_A"].dropna().astype(float).tolist()
            if "max_depth_A" in g
            else [],
            "vacancies_per_ion": g["vacancies_per_ion"].dropna().astype(float).tolist(),
        }

    print(f"[INFO] Aggregated scalar results for {len(sim_dict)} tracks.")
    return sim_dict



def load_observed_scalars(df_obs: pd.DataFrame) -> dict:
    """
    Extract true observed scalar values for each track.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Output from preprocess() or similar observed dataset summary.

    Returns
    -------
    obs_dict : dict
        { track_id: { 'mean_depth_A': val, 'max_depth_A': val, 'vacancies_per_ion': val } }
    """
    required_cols = ["track_id", "mean_depth_A", "vacancies_per_ion"]
    for col in required_cols:
        if col not in df_obs.columns:
            raise ValueError(f"Missing required column '{col}' in df_obs")

    obs_dict = {}

    for _, row in df_obs.iterrows():
        track_id = str(row["track_id"]).strip()

        mean_val = (
            float(row["mean_depth_A"]) if pd.notna(row["mean_depth_A"]) else np.nan
        )
        vac_val = (
            float(row["vacancies_per_ion"])
            if pd.notna(row["vacancies_per_ion"])
            else np.nan
        )

        if "max_depth_A" in df_obs.columns:
            max_val = (
                float(row["max_depth_A"]) if pd.notna(row["max_depth_A"]) else np.nan
            )
        else:
            max_val = np.nan

        obs_dict[track_id] = {
            "mean_depth_A": mean_val,
            "max_depth_A": max_val,
            "vacancies_per_ion": vac_val,
        }

    print(f"[INFO] Loaded observed scalar values for {len(obs_dict)} tracks.")
    return obs_dict



# Metric Computation


def compute_scalar_ppc_per_track(obs_dict: dict, sim_dict: dict) -> pd.DataFrame:
    """
    Compute posterior predictive check (PPC) metrics for scalar quantities per track.

    Returns
    -------
    pd.DataFrame with columns:
        track_id, feature, n_samples, obs, mu_sim, sigma_sim,
        delta_percent, z_score, ppp_one_sided, ppp_two_sided, status
    """
    import numpy as np
    import pandas as pd

    features = ["mean_depth_A", "max_depth_A", "vacancies_per_ion"]
    rows = []

    for track_id, obs_vals in obs_dict.items():
        sim_vals = sim_dict.get(track_id)

        for f in features:
            obs = obs_vals.get(f, np.nan)

            if not sim_vals or f not in sim_vals:
                rows.append({
                    "track_id": track_id, "feature": f, "n_samples": 0,
                    "obs": obs, "mu_sim": np.nan, "sigma_sim": np.nan,
                    "delta_percent": np.nan, "z_score": np.nan,
                    "ppp_one_sided": np.nan, "ppp_two_sided": np.nan,
                    "status": "MISSING_SIM",
                })
                continue

            vals = np.asarray(sim_vals[f], float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)

            if n == 0:
                status = "MISSING_SIM"
                mu, sigma = np.nan, np.nan
            else:
                mu = float(np.mean(vals))
                sigma = float(np.std(vals, ddof=1)) if n > 1 else np.nan
                status = "OK"

            delta = 100 * (obs - mu) / (mu + 1e-12) if np.isfinite(mu) and np.isfinite(obs) else np.nan

            # handle edge cases
            if not np.isfinite(obs):
                status = "BAD_OBS"
                z = p1 = p2 = np.nan
            elif n < 2:
                status = "FEW_SAMPLES"
                z = p1 = p2 = np.nan
            elif np.isfinite(sigma) and sigma == 0:
                status = "CONST_SIM"
                z = p1 = p2 = np.nan
            elif np.isfinite(sigma) and sigma > 0:
                z = (obs - mu) / (sigma + 1e-12)
                p1 = np.mean(vals >= obs)
                p2 = float(np.clip(2 * min(p1, 1 - p1), 0, 1))
            else:
                status = "BAD_SIM"
                z = p1 = p2 = np.nan

            rows.append({
                "track_id": track_id,
                "feature": f,
                "n_samples": n,
                "obs": float(obs),
                "mu_sim": mu,
                "sigma_sim": sigma,
                "delta_percent": delta,
                "z_score": z,
                "ppp_one_sided": p1,
                "ppp_two_sided": p2,
                "status": status,
            })

    df = pd.DataFrame(rows)
    print(f"[INFO] Computed scalar PPC metrics for {len(df)} track-feature pairs.")
    return df

def summarize_scalar_ppc(per_track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per-track PPC results for each scalar feature.
    Only includes tracks with status "OK" or "CONST_SIM".
    """
    import numpy as np
    import pandas as pd

    if not isinstance(per_track_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    needed = ["feature", "status", "delta_percent", "z_score"]
    for c in needed:
        if c not in per_track_df.columns:
            raise ValueError(f"Missing required column: {c}")

    valid = {"OK", "CONST_SIM"}
    rows = []

    for f, g in per_track_df.groupby("feature", sort=False):
        g = g[g["status"].isin(valid)]
        if g.empty:
            rows.append({
                "feature": f,
                "n_tracks_ok": 0,
                "mean_delta_pct": np.nan,
                "median_delta_pct": np.nan,
                "median_abs_z": np.nan,
                "std_z": np.nan,
                "frac_outliers": np.nan,
            })
            continue

        abs_z = np.abs(g["z_score"].astype(float))
        rows.append({
            "feature": f,
            "n_tracks_ok": len(g),
            "mean_delta_pct": np.nanmean(g["delta_percent"]),
            "median_delta_pct": np.nanmedian(g["delta_percent"]),
            "median_abs_z": np.nanmedian(abs_z),
            "std_z": np.nanstd(abs_z, ddof=1),
            "frac_outliers": np.nanmean(abs_z > 2),
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Summarized scalar PPC across {len(df)} features.")
    return df


def save_scalar_ppc_reports(per_track_df, summary_df, out_dir):
    """
    Save scalar PPC results (per-track and summary) to disk.
    Outputs both CSV and JSON versions for downstream use.
    """
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_track_csv = out_dir / "ppc_scalar_per_track.csv"
    summary_csv = out_dir / "ppc_scalar_summary.csv"

    # Save as CSV
    per_track_df.to_csv(per_track_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # Also save JSON summaries for programmatic access
    summary_json = out_dir / "ppc_scalar_summary.json"
    summary_df.to_json(summary_json, orient="records", indent=2)

    print(f"[INFO] Saved scalar PPC per-track results → {per_track_csv}")
    print(f"[INFO] Saved scalar PPC summary results → {summary_csv}")
    print(f"[INFO] Saved scalar PPC summary (JSON) → {summary_json}")

# Visualization 


import os
import numpy as np
import matplotlib.pyplot as plt


def plot_scalar_ppc_histograms(per_track_df, sim_dict, obs_dict, out_dir):
    """
    Plot per-track posterior predictive histograms for scalar features.

    Parameters
    ----------
    per_track_df : pd.DataFrame
        Output from compute_scalar_ppc_per_track().
    sim_dict : dict
        Simulated posterior samples per track:
        { track_id: { feature: [floats] } }
    obs_dict : dict
        Observed scalars per track:
        { track_id: { feature: float } }
    out_dir : str | Path
        Output directory to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    if per_track_df.empty:
        print("[WARN] No PPC results to plot.")
        return

    features = ["mean_depth_A", "max_depth_A", "vacancies_per_ion"]
    colors = {"mean_depth_A": "C0", "max_depth_A": "C1", "vacancies_per_ion": "C2"}

    for feature in features:
        df_f = per_track_df[per_track_df["feature"] == feature]
        for _, row in df_f.iterrows():
            tid = row["track_id"]
            status = row["status"]
            if status != "OK":
                continue

            sim_vals = sim_dict.get(tid, {}).get(feature, [])
            obs_val = obs_dict.get(tid, {}).get(feature, np.nan)
            if len(sim_vals) == 0 or not np.isfinite(obs_val):
                continue

            mu_sim = row["mu_sim"]
            z_score = row["z_score"]
            delta_pct = row["delta_percent"]

            plt.figure(figsize=(6, 4))
            plt.hist(
                sim_vals,
                bins=30,
                alpha=0.7,
                color=colors.get(feature, "C0"),
                edgecolor="k",
                density=True,
                label="Simulated Posterior",
            )
            plt.axvline(
                mu_sim,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                label=f"μ_sim = {mu_sim:.2f}",
            )
            plt.axvline(
                obs_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Observed = {obs_val:.2f}",
            )
            plt.title(f"{feature} — Track {tid}")
            plt.xlabel(feature)
            plt.ylabel("Density")

            plt.legend()
            plt.tight_layout()

            # Annotation box
            plt.text(
                0.02,
                0.95,
                f"Δ% = {delta_pct:+.1f}%\nZ = {z_score:+.2f}",
                transform=plt.gca().transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
            )

            save_path = os.path.join(out_dir, f"PPC_hist_{feature}_{tid}.png")
            plt.savefig(save_path, dpi=150)
            plt.close()

    print(f"[INFO] Saved per-track scalar PPC histograms → {out_dir}")


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scalar_ppc_aggregates(per_track_df: pd.DataFrame, out_dir: str):
    """
    Plot overall PPC distributions (z-scores and percent differences) for each feature.
    Saves one pair of plots per feature.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if per_track_df.empty:
        print("[WARN] No PPC data to plot.")
        return

    for feat, g in per_track_df.groupby("feature", sort=False):
        g_ok = g[g["status"] == "OK"]
        if g_ok.empty:
            print(f"[WARN] No valid data for {feat}, skipping.")
            continue

        z = g_ok["z_score"].dropna().to_numpy()
        d = g_ok["delta_percent"].dropna().to_numpy()

        # Z-score distribution
        plt.figure(figsize=(6, 4))
        plt.hist(z, bins=30, color="C0", alpha=0.75, edgecolor="k", density=True)
        plt.axvline(0, color="black", ls="--", lw=1)
        plt.axvline(2, color="red", ls=":", lw=1)
        plt.axvline(-2, color="red", ls=":", lw=1)
        plt.title(f"Z-Score Distribution — {feat}")
        plt.xlabel("Z-score (obs − sim)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"PPC_agg_zscore_{feat}.png", dpi=150)
        plt.close()

        # Percent difference distribution
        plt.figure(figsize=(6, 4))
        plt.hist(d, bins=30, color="C1", alpha=0.75, edgecolor="k", density=True)
        plt.axvline(0, color="black", ls="--", lw=1)
        plt.title(f"Percent Difference (Δ%) — {feat}")
        plt.xlabel("Δ% (obs − sim mean)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"PPC_agg_delta_{feat}.png", dpi=150)
        plt.close()

    print(f"[INFO] Saved aggregate PPC plots to {out_dir}")

# ---------------------------
# 4️⃣ Orchestrator
# ---------------------------


def run_scalar_ppc(df_post: pd.DataFrame, df_obs: pd.DataFrame, out_dir: str) -> dict:
    """
    High-level orchestrator to compute, summarize, and visualize scalar PPC.

    Parameters
    ----------
    df_post : pd.DataFrame
        SRIM posterior results from summarize_all_runs()
    df_obs : pd.DataFrame
        Observed preprocessed track data
    out_dir : str
        Directory to save PPC outputs

    Returns
    -------
    results : dict
        {
            "per_track": <DataFrame>,
            "summary": <DataFrame>
        }
    """
    # prepare inputs
    sim_dict = group_simulated_scalars(df_post)
    obs_dict = load_observed_scalars(df_obs)

    # compute metrics
    per_track_df = compute_scalar_ppc_per_track(obs_dict, sim_dict)
    summary_df = summarize_scalar_ppc(per_track_df)

    # visualize + save
    plot_scalar_ppc_histograms(per_track_df, sim_dict, obs_dict, out_dir)
    plot_scalar_ppc_aggregates(per_track_df, out_dir)
    save_scalar_ppc_reports(per_track_df, summary_df, out_dir)

    return {"per_track": per_track_df, "summary": summary_df}


import numpy as np
import pandas as pd


def group_simulated_vectors(df_post: pd.DataFrame, prefix: str = "rbin_frac_") -> dict:
    """
    Collapse SRIM simulation results (posterior runs) into per-track arrays of shape vectors.

    Parameters
    ----------
    df_post : pd.DataFrame
        Output from summarize_all_runs(), containing multiple SRIM runs per track.
        Must contain columns like rbin_frac_1, rbin_frac_2, ..., rbin_frac_n.
    prefix : str
        Column prefix identifying relative bin fractions (default: "rbin_frac_")

    Returns
    -------
    sim_dict : dict
        Mapping from track_id → list of np.ndarray vectors.
        Example:
            {
                "track_01": [array([...]), array([...]), ...],
                "track_02": [array([...]), ...],
            }
    """
    # Identify all bin fraction columns dynamically
    bin_cols = [c for c in df_post.columns if c.startswith(prefix)]
    if len(bin_cols) == 0:
        raise ValueError(f"No columns found with prefix '{prefix}' in df_post.")

    if "track_id" not in df_post.columns:
        raise ValueError("Missing required column 'track_id' in df_post.")

    sim_dict = {}

    # Group by track and store each SRIM run as a shape vector
    for track_id, group in df_post.groupby("track_id", sort=False):
        vectors = []
        for _, row in group.iterrows():
            vec = row[bin_cols].to_numpy(dtype=float)
            if np.all(np.isfinite(vec)):
                vectors.append(vec)
        if len(vectors) > 0:
            sim_dict[str(track_id)] = vectors

    print(f"[INFO] Built simulated vector dictionary for {len(sim_dict)} tracks.")
    print(f"[DEBUG] Example vector length: {len(bin_cols)} bins.")
    return sim_dict


import numpy as np
import pandas as pd


def load_observed_vectors(df_obs: pd.DataFrame, prefix: str = "rbin_frac_") -> dict:
    """
    Extract observed shape vectors (relative bin fractions) per track.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observed or preprocessed SRIM data with one row per track.
        Must include columns like rbin_frac_1, rbin_frac_2, ..., rbin_frac_n.
    prefix : str, default="rbin_frac_"
        Column prefix for relative bin fraction columns.

    Returns
    -------
    obs_dict : dict
        Mapping from track_id → np.ndarray (single shape vector).
        Example:
            {
                "track_01": array([...]),
                "track_02": array([...]),
            }
    """
    # Identify bin columns dynamically
    bin_cols = [c for c in df_obs.columns if c.startswith(prefix)]
    if len(bin_cols) == 0:
        raise ValueError(f"No columns found with prefix '{prefix}' in df_obs.")

    if "track_id" not in df_obs.columns:
        raise ValueError("Missing required column 'track_id' in df_obs.")

    obs_dict = {}

    for _, row in df_obs.iterrows():
        tid = str(row["track_id"]).strip()
        vec = row[bin_cols].to_numpy(dtype=float)
        # Only include valid numeric vectors
        if np.all(np.isfinite(vec)):
            obs_dict[tid] = vec

    print(f"[INFO] Loaded observed shape vectors for {len(obs_dict)} tracks.")
    print(f"[DEBUG] Vector length = {len(bin_cols)} bins.")
    return obs_dict


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from numpy.linalg import norm


def compute_shape_ppc_per_track(obs_dict: dict, sim_dict: dict) -> pd.DataFrame:
    """
    Compute per-track shape PPC metrics comparing observed and simulated vectors.

    Metrics:
      - Earth Mover's Distance (EMD)
      - Cosine similarity
      - Mean absolute percent difference (per bin)
      - Z-scores of deviation per bin

    Parameters
    ----------
    obs_dict : dict
        { track_id: np.ndarray([...]) } - observed shape vector per track.
    sim_dict : dict
        { track_id: [np.ndarray([...]), ...] } - simulated shape vectors per track.

    Returns
    -------
    pd.DataFrame
        Columns:
        track_id, n_samples, emd_mean, emd_std,
        cosine_mean, cosine_std, mean_abs_pct_diff, mean_abs_z, status
    """
    records = []

    for tid, obs_vec in obs_dict.items():
        obs_vec = np.asarray(obs_vec, dtype=float)
        sim_vectors = sim_dict.get(tid, [])

        if len(sim_vectors) == 0:
            records.append(
                {
                    "track_id": tid,
                    "n_samples": 0,
                    "emd_mean": np.nan,
                    "emd_std": np.nan,
                    "cosine_mean": np.nan,
                    "cosine_std": np.nan,
                    "mean_abs_pct_diff": np.nan,
                    "mean_abs_z": np.nan,
                    "status": "MISSING_SIM",
                }
            )
            continue

        # Ensure valid observed vector
        if not np.all(np.isfinite(obs_vec)):
            records.append(
                {
                    "track_id": tid,
                    "n_samples": len(sim_vectors),
                    "emd_mean": np.nan,
                    "emd_std": np.nan,
                    "cosine_mean": np.nan,
                    "cosine_std": np.nan,
                    "mean_abs_pct_diff": np.nan,
                    "mean_abs_z": np.nan,
                    "status": "BAD_OBS",
                }
            )
            continue

        # Collect metrics across all simulated vectors
        emds, cosines = [], []
        for sim_vec in sim_vectors:
            sim_vec = np.asarray(sim_vec, dtype=float)
            if not np.all(np.isfinite(sim_vec)):
                continue

            # --- EMD (shape distance) ---
            emds.append(wasserstein_distance(obs_vec, sim_vec))

            # --- Cosine similarity (orientation) ---
            denom = norm(obs_vec) * norm(sim_vec) + 1e-12
            cos_sim = float(np.dot(obs_vec, sim_vec) / denom)
            cosines.append(cos_sim)

        if len(emds) == 0:
            status = "INVALID_SIM"
            emd_mean = emd_std = cosine_mean = cosine_std = np.nan
        else:
            emd_mean, emd_std = np.mean(emds), (
                np.std(emds, ddof=1) if len(emds) > 1 else 0.0
            )
            cosine_mean, cosine_std = np.mean(cosines), (
                np.std(cosines, ddof=1) if len(cosines) > 1 else 0.0
            )
            status = "OK"

        # --- Per-bin summary stats ---
        sim_matrix = np.stack(sim_vectors, axis=0)
        mu_sim = np.nanmean(sim_matrix, axis=0)
        sigma_sim = np.nanstd(sim_matrix, axis=0, ddof=1)
        abs_pct_diff = np.abs((obs_vec - mu_sim) / (mu_sim + 1e-12)) * 100
        z_per_bin = np.abs((obs_vec - mu_sim) / (sigma_sim + 1e-12))
        mean_abs_pct_diff = float(np.nanmean(abs_pct_diff))
        mean_abs_z = float(np.nanmean(z_per_bin))

        records.append(
            {
                "track_id": tid,
                "n_samples": len(sim_vectors),
                "emd_mean": emd_mean,
                "emd_std": emd_std,
                "cosine_mean": cosine_mean,
                "cosine_std": cosine_std,
                "mean_abs_pct_diff": mean_abs_pct_diff,
                "mean_abs_z": mean_abs_z,
                "status": status,
            }
        )

    df = pd.DataFrame.from_records(records)
    print(f"[INFO] Computed shape PPC metrics for {len(df)} tracks.")
    return df


import pandas as pd
import numpy as np


def summarize_shape_ppc(per_track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-track shape PPC results into overall summary statistics.

    Parameters
    ----------
    per_track_df : pd.DataFrame
        Output from compute_shape_ppc_per_track().
        Expected columns: ['track_id', 'emd_mean', 'emd_std',
                           'cosine_mean', 'mean_abs_pct_diff', 'mean_abs_z', 'status']

    Returns
    -------
    pd.DataFrame
        Summary across all valid tracks:
        Columns:
          n_tracks_ok, mean_emd, std_emd,
          mean_cosine, std_cosine,
          mean_pct_diff, std_pct_diff,
          mean_z, std_z,
          frac_outliers_z>2
    """
    if per_track_df.empty:
        print("[WARN] No shape PPC data to summarize.")
        return pd.DataFrame()

    g = per_track_df[per_track_df["status"] == "OK"].copy()

    if g.empty:
        print("[WARN] No OK tracks available for shape PPC summary.")
        return pd.DataFrame()

    summary = {
        "n_tracks_ok": len(g),
        "mean_emd": np.nanmean(g["emd_mean"]),
        "std_emd": np.nanstd(g["emd_mean"], ddof=1),
        "mean_cosine": np.nanmean(g["cosine_mean"]),
        "std_cosine": np.nanstd(g["cosine_mean"], ddof=1),
        "mean_pct_diff": np.nanmean(g["mean_abs_pct_diff"]),
        "std_pct_diff": np.nanstd(g["mean_abs_pct_diff"], ddof=1),
        "mean_z": np.nanmean(g["mean_abs_z"]),
        "std_z": np.nanstd(g["mean_abs_z"], ddof=1),
        "frac_outliers_z>2": np.mean(g["mean_abs_z"] > 2),
    }

    df_summary = pd.DataFrame([summary])
    print(f"[INFO] Computed shape PPC summary across {len(g)} tracks.")
    return df_summary


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_shape_ppc_distributions(per_track_df: pd.DataFrame, out_dir: str):
    """
    Plot aggregate histograms for shape PPC metrics (EMD, cosine similarity, etc.).

    Parameters
    ----------
    per_track_df : pd.DataFrame
        Output from compute_shape_ppc_per_track().
    out_dir : str | Path
        Directory to save plots.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    g = per_track_df[per_track_df["status"] == "OK"].copy()
    if g.empty:
        print("[WARN] No OK data available for shape PPC plotting.")
        return

    metrics = {
        "emd_mean": {"title": "Earth Mover's Distance (EMD)", "color": "C0"},
        "cosine_mean": {"title": "Cosine Similarity", "color": "C1"},
        "mean_abs_pct_diff": {"title": "Mean Abs Percent Diff (%)", "color": "C2"},
        "mean_abs_z": {"title": "Mean Abs Z-Score", "color": "C3"},
    }

    for col, meta in metrics.items():
        vals = g[col].dropna().values
        if len(vals) == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(
            vals, bins=30, color=meta["color"], alpha=0.75, edgecolor="k", density=True
        )
        plt.axvline(
            np.mean(vals),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Mean = {np.mean(vals):.3f}",
        )
        plt.title(meta["title"])
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        save_path = Path(out_dir) / f"PPC_shape_{col}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"[INFO] Saved shape PPC distribution plots → {out_dir}")

    import json


from pathlib import Path


def save_shape_ppc_reports(
    per_track_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str
):
    """
    Save shape PPC metrics and summaries to disk (CSV + JSON).

    Parameters
    ----------
    per_track_df : pd.DataFrame
        Output from compute_shape_ppc_per_track()
    summary_df : pd.DataFrame
        Output from summarize_shape_ppc()
    out_dir : str | Path
        Directory to save the files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    per_track_path = Path(out_dir) / "ppc_shape_per_track.csv"
    summary_path = Path(out_dir) / "ppc_shape_summary.json"

    per_track_df.to_csv(per_track_path, index=False)
    summary_data = summary_df.to_dict(orient="records")

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[INFO] Saved shape PPC reports → {out_dir}")


def run_shape_ppc(
    df_post: pd.DataFrame,
    df_obs: pd.DataFrame,
    out_dir: str,
    prefix: str = "rbin_frac_",
) -> dict:
    """
        High-level orchestrator for vector (shape) PPC using EMD, cosine, and per-bin metrics.

        Parameters
        ----------
        df_post : pd.DataFrame
            SRIM posterior results from summarize_all_runs()
        df_obs : pd.DataFrame
            Observed preprocessed track data
        out_dir : str
            Directory to save PPC outputs
        prefix : str
            Prefix for bin fraction columns (default "rbin_frac_")
    x
        Returns
        -------
        results : dict
            {
                "per_track": <DataFrame>,
                "summary": <DataFrame>
            }
    """
    # --- prepare inputs ---
    sim_dict = group_simulated_vectors(df_post, prefix=prefix)
    obs_dict = load_observed_vectors(df_obs, prefix=prefix)

    # --- compute metrics ---
    per_track_df = compute_shape_ppc_per_track(obs_dict, sim_dict)
    summary_df = summarize_shape_ppc(per_track_df)

    # --- visualize + save ---
    plot_shape_ppc_distributions(per_track_df, out_dir)
    save_shape_ppc_reports(per_track_df, summary_df, out_dir)

    return {"per_track": per_track_df, "summary": summary_df}
