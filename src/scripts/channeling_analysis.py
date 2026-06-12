"""Channeling vs non-channeling performance breakdown for the GVP-EGNN v2.1 model.

Usage:
    python -m src.scripts.channeling_analysis \
        --eval-results  results/gvp_egnn_v21_siimpl/eval_results.csv \
        --siimpl-eval   data/siimpl/siimpl_eval.csv \
        --out-dir       results/gvp_egnn_v21_siimpl/channeling_analysis

If --siimpl-eval is omitted it falls back to the default path in the repo.

Channeling definition: the true recoil direction lies within N * psi_c of any
equivalent channeling axis (100/110/111) in the SIIMPL lab frame (Ry(35 deg)
pre-rotation). Default N=2; adjustable via --psi-scale.

The script:
  1. Groups siimpl_eval.csv by ion_number to build per-track ground-truth
     (direction, energy, n_vac).
  2. Filters n_vac < 3 to match preprocess_egnn, then aligns row-by-row
     with eval_results.csv (same sort order).
  3. Classifies each track: channeling / near-axis / bulk.
  4. Plots angular error distributions, energy error distributions,
     calibration curves, and head-tail accuracy — all split by group.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# paths / imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
from channeling_axes import AXES_LAB, psi_c_deg, angle_to_nearest_axis

DEFAULT_SIIMPL_EVAL = REPO_ROOT / "data" / "siimpl" / "siimpl_eval.csv"
FAMILIES = ["<100>", "<110>", "<111>"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _angle_to_any_axis(vx: float, vy: float, vz: float) -> tuple[float, str]:
    """Return (min_angle_deg, family) across all three channeling families."""
    v = np.array([vx, vy, vz], dtype=float)
    v /= np.linalg.norm(v)
    best_angle, best_family = 360.0, ""
    for fam in FAMILIES:
        a = angle_to_nearest_axis(v, fam)
        if a < best_angle:
            best_angle, best_family = a, fam
    return best_angle, best_family


def _psi_c_any(E_keV: float, family: str) -> float:
    """Critical angle for the winning family."""
    return psi_c_deg(E_keV, family)


def load_siimpl_tracks(csv_path: Path) -> pd.DataFrame:
    """Group siimpl_eval.csv by ion_number → one row per track."""
    print(f"Loading {csv_path} …")
    raw = pd.read_csv(
        csv_path,
        usecols=["ion_number", "energy_keV", "target_vx", "target_vy", "target_vz"],
    )
    agg = (
        raw.groupby("ion_number")
        .agg(
            n_vac=("energy_keV", "count"),
            energy_keV=("energy_keV", "first"),
            target_vx=("target_vx", "first"),
            target_vy=("target_vy", "first"),
            target_vz=("target_vz", "first"),
        )
        .reset_index()
        .sort_values("ion_number")
    )
    print(f"  Total tracks: {len(agg):,}  |  with n_vac<3: {(agg.n_vac < 3).sum():,}")
    agg = agg[agg.n_vac >= 3].reset_index(drop=True)
    return agg


def classify_tracks(tracks: pd.DataFrame, psi_scale: float = 2.0) -> pd.DataFrame:
    """Add columns: angle_deg, axis_family, psi_c, is_channeling, category."""
    print("Classifying tracks (computing angle to nearest axis) …")
    rows = [
        _angle_to_any_axis(r.target_vx, r.target_vy, r.target_vz)
        for _, r in tracks.iterrows()
    ]
    angles = np.array([r[0] for r in rows])
    families = [r[1] for r in rows]
    psi_c_vals = np.array(
        [_psi_c_any(e, f) for e, f in zip(tracks.energy_keV.values, families)]
    )

    tracks = tracks.copy()
    tracks["angle_to_axis_deg"] = angles
    tracks["axis_family"] = families
    tracks["psi_c_deg"] = psi_c_vals

    # Three-way split:
    #   channeling:  within 1 * psi_c  (core channelers)
    #   near-axis:   within psi_scale * psi_c but > 1 * psi_c
    #   bulk:        beyond psi_scale * psi_c
    tracks["is_channeling"] = angles <= psi_c_vals
    tracks["is_near_axis"] = (angles > psi_c_vals) & (angles <= psi_scale * psi_c_vals)
    tracks["category"] = "bulk"
    tracks.loc[tracks.is_near_axis, "category"] = "near-axis"
    tracks.loc[tracks.is_channeling, "category"] = "channeling"

    counts = tracks.category.value_counts()
    for cat in ["channeling", "near-axis", "bulk"]:
        print(f"  {cat:12s}: {counts.get(cat, 0):6,}")
    return tracks


def align_eval_results(tracks: pd.DataFrame, eval_csv: Path) -> pd.DataFrame:
    """Load eval_results.csv and merge with track metadata by position."""
    print(f"Loading eval results from {eval_csv} …")
    eval_df = pd.read_csv(eval_csv)
    if len(eval_df) != len(tracks):
        raise ValueError(
            f"Row count mismatch: eval_results.csv has {len(eval_df):,} rows "
            f"but siimpl_eval has {len(tracks):,} tracks after filtering. "
            "Make sure both files come from the same model run / eval CSV."
        )
    merged = pd.concat([tracks.reset_index(drop=True), eval_df.reset_index(drop=True)], axis=1)
    return merged


# ---------------------------------------------------------------------------
# analysis & plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "channeling": "#d62728",   # red
    "near-axis":  "#ff7f0e",   # orange
    "bulk":       "#1f77b4",   # blue
}
LABELS = {
    "channeling": "Channeling  (ψ ≤ ψ_c)",
    "near-axis":  "Near-axis  (ψ_c < ψ ≤ 2ψ_c)",
    "bulk":       "Bulk  (ψ > 2ψ_c)",
}


def _energy_bins(df: pd.DataFrame, n_bins: int = 8) -> list[tuple[float, float]]:
    log_edges = np.linspace(
        np.log10(df.true_energy.min()),
        np.log10(df.true_energy.max()),
        n_bins + 1,
    )
    return [(10**log_edges[i], 10**log_edges[i + 1]) for i in range(n_bins)]


def plot_angular_error_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, cat in zip(axes, ["channeling", "near-axis", "bulk"]):
        sub = df[df.category == cat]["angular_error_deg"]
        ax.hist(sub, bins=60, range=(0, 180), color=PALETTE[cat], alpha=0.75,
                density=True, label=f"n={len(sub):,}")
        med = sub.median()
        ax.axvline(med, color="k", ls="--", lw=1.5, label=f"median {med:.1f}°")
        r68 = float(np.percentile(sub, 68))
        ax.axvline(r68, color="k", ls=":", lw=1.2, label=f"R68 {r68:.1f}°")
        ax.set_xlabel("Angular error (deg)", fontsize=11)
        ax.set_title(LABELS[cat], fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density", fontsize=11)
    fig.suptitle("Angular Error Distribution by Track Category", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "angular_error_distributions.png", dpi=150)
    plt.close(fig)
    print("  -> angular_error_distributions.png")


def plot_angular_error_vs_energy(df: pd.DataFrame, out_dir: Path) -> None:
    bins = _energy_bins(df)
    bin_centers = [np.sqrt(lo * hi) for lo, hi in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in ["channeling", "near-axis", "bulk"]:
        sub = df[df.category == cat]
        medians, r68s = [], []
        for lo, hi in bins:
            mask = (sub.true_energy >= lo) & (sub.true_energy < hi)
            grp = sub.loc[mask, "angular_error_deg"]
            if len(grp) < 5:
                medians.append(np.nan); r68s.append(np.nan)
            else:
                medians.append(grp.median())
                r68s.append(float(np.percentile(grp, 68)))
        medians = np.array(medians)
        r68s = np.array(r68s)
        ax.semilogx(bin_centers, medians, "o-", color=PALETTE[cat],
                    label=LABELS[cat], lw=1.8, ms=5)
        ax.fill_between(bin_centers, medians, r68s,
                        color=PALETTE[cat], alpha=0.15)

    ax.set_xlabel("True energy (keV)", fontsize=11)
    ax.set_ylabel("Angular error (deg)", fontsize=11)
    ax.set_title("Angular Error vs Energy — Channeling vs Bulk", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "angular_error_vs_energy.png", dpi=150)
    plt.close(fig)
    print("  -> angular_error_vs_energy.png")


def plot_head_tail_vs_energy(df: pd.DataFrame, out_dir: Path) -> None:
    bins = _energy_bins(df)
    bin_centers = [np.sqrt(lo * hi) for lo, hi in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in ["channeling", "near-axis", "bulk"]:
        sub = df[df.category == cat]
        ht_acc = []
        for lo, hi in bins:
            mask = (sub.true_energy >= lo) & (sub.true_energy < hi)
            grp = sub.loc[mask, "angular_error_deg"]
            if len(grp) < 5:
                ht_acc.append(np.nan)
            else:
                ht_acc.append(100.0 * (grp <= 90).mean())
        ax.semilogx(bin_centers, ht_acc, "s--", color=PALETTE[cat],
                    label=LABELS[cat], lw=1.8, ms=5)

    ax.axhline(50, color="gray", ls=":", lw=1, label="random (50%)")
    ax.set_xlabel("True energy (keV)", fontsize=11)
    ax.set_ylabel("Head-tail accuracy (%)", fontsize=11)
    ax.set_title("Head-Tail Discrimination vs Energy", fontsize=12)
    ax.set_ylim(45, 102)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "head_tail_vs_energy.png", dpi=150)
    plt.close(fig)
    print("  -> head_tail_vs_energy.png")


def plot_angle_to_axis_vs_error(df: pd.DataFrame, out_dir: Path) -> None:
    """2D scatter: angle-to-nearest-axis (x) vs angular error (y), colored by category."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, cat in zip(axes, ["channeling", "near-axis", "bulk"]):
        sub = df[df.category == cat]
        ax.scatter(
            sub.angle_to_axis_deg, sub.angular_error_deg,
            c=PALETTE[cat], alpha=0.25, s=3, rasterized=True,
        )
        ax.set_xlabel("Angle to nearest axis (deg)", fontsize=10)
        ax.set_title(f"{LABELS[cat]}  (n={len(sub):,})", fontsize=9)
    axes[0].set_ylabel("Angular error (deg)", fontsize=10)
    fig.suptitle("Angular Error vs Proximity to Channeling Axis", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "angle_vs_error_scatter.png", dpi=150)
    plt.close(fig)
    print("  -> angle_vs_error_scatter.png")


def plot_calibration_by_category(df: pd.DataFrame, out_dir: Path) -> None:
    """Coverage calibration for angular cone (cone_68 used as proxy)."""
    if "angular_cone_68" not in df.columns:
        print("  (skipping calibration: angular_cone_68 not in eval_results)")
        return

    levels = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")

    for cat in ["channeling", "near-axis", "bulk"]:
        sub = df[df.category == cat]
        if len(sub) < 50:
            continue
        coverage = []
        for lv in levels:
            frac_covered = (sub.angular_error_deg <= sub.angular_cone_68 * (lv / 0.68)).mean()
            coverage.append(frac_covered)
        ax.plot(levels, coverage, "o-", color=PALETTE[cat],
                label=LABELS[cat], lw=1.8, ms=4)

    ax.set_xlabel("Credible level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Angular Posterior Calibration by Category", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_by_category.png", dpi=150)
    plt.close(fig)
    print("  -> calibration_by_category.png")


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(f"{'Category':<14} {'N':>6} {'Med.AE(°)':>10} {'R68(°)':>8} "
          f"{'HT-acc(%)':>10} {'Med.EE(keV)':>12}")
    print("-" * 72)
    for cat in ["channeling", "near-axis", "bulk", "ALL"]:
        sub = df if cat == "ALL" else df[df.category == cat]
        if len(sub) == 0:
            continue
        ae = sub.angular_error_deg
        ee = sub.energy_error
        ht = 100.0 * (ae <= 90).mean()
        print(f"{cat:<14} {len(sub):>6,} {ae.median():>10.2f} "
              f"{np.percentile(ae, 68):>8.2f} {ht:>10.1f} "
              f"{ee.abs().median():>12.3f}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-results", required=True,
                    help="Path to eval_results.csv from ContinuousEvaluator3D")
    ap.add_argument("--siimpl-eval", default=str(DEFAULT_SIIMPL_EVAL),
                    help="Path to siimpl_eval.csv (raw vacancy data)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for plots and summary CSV")
    ap.add_argument("--psi-scale", type=float, default=2.0,
                    help="Threshold multiplier: channeling if angle <= psi_c, "
                         "near-axis if <= psi_scale*psi_c (default 2.0)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_siimpl_tracks(Path(args.siimpl_eval))
    tracks = classify_tracks(tracks, psi_scale=args.psi_scale)
    df = align_eval_results(tracks, Path(args.eval_results))

    print_summary_table(df)

    print("Generating plots …")
    plot_angular_error_distributions(df, out_dir)
    plot_angular_error_vs_energy(df, out_dir)
    plot_head_tail_vs_energy(df, out_dir)
    plot_angle_to_axis_vs_error(df, out_dir)
    plot_calibration_by_category(df, out_dir)

    summary_path = out_dir / "channeling_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull merged table saved to: {summary_path}")


if __name__ == "__main__":
    main()
