"""
ContinuousEvaluator3D — Full Posterior Evaluation for 3D SBI Pipelines
======================================================================
Extracts full posterior information from SBI, not just point estimates.

Metrics produced:
  Level 1 (Point Estimates):
    - Median energy error (keV)
    - Median angular error (deg)
    - Head-tail flip rate (%)
    - R68 containment angle (deg)

  Level 2 (Posterior Quality — the SBI advantage):
    - Per-track posterior uncertainty (std) for energy and direction
    - SBI Calibration: Expected coverage vs actual coverage
    - Posterior sharpness: how tight are the credible intervals?
    - Energy credible interval widths
    - Angular credible cone widths

  Level 3 (Diagnostics):
    - Calibration plot (expected vs observed coverage)
    - Posterior uncertainty vs error scatter
    - Energy-binned calibration
    - Corner plot for selected tracks
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import EGNN preprocessing (same pipeline as training for consistency)
try:
    from src.utils.data_utils import preprocess_egnn
except ImportError:
    sys.path.append(".")
    from src.utils.data_utils import preprocess_egnn


class ContinuousEvaluator3D:
    def __init__(self, posterior, device=None):
        self.posterior = posterior

        try:
            self.device = next(posterior.posterior_estimator.parameters()).device
        except:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"[Eval] Model detected on device: {self.device}")

    def run_eval(self, eval_csv_path, num_samples=200, batch_size=500):
        """
        Full posterior evaluation.

        Args:
            eval_csv_path: path to evaluation CSV
            num_samples: number of posterior samples per track (200+ recommended
                         for calibration; 50 is too few for coverage analysis)
            batch_size: tracks per batch for GPU memory management

        Returns:
            results_df: DataFrame with point estimates AND posterior statistics
        """
        # 1. Load and preprocess eval data (same pipeline as training)
        print(f"[Eval] Loading & preprocessing from {eval_csv_path}...")

        # Get k from the posterior's embedding network
        try:
            emb_net = self.posterior.posterior_estimator._neural_net.embedding_net
            k_neighbors = emb_net.k
            n_max_model = emb_net.n_max
        except AttributeError:
            # Fallback for older posteriors
            emb_net = self.posterior.net._neural_net.embedding_net
            k_neighbors = emb_net.k
            n_max_model = emb_net.n_max

        x_padded, mask, targets_tensor, n_max, knn_idx = preprocess_egnn(
            eval_csv_path, k_neighbors=k_neighbors
        )

        # Flatten same as training: coords + knn_idx (memory-efficient chunked)
        n_eval = x_padded.shape[0]
        features = torch.empty(n_eval, n_max * (3 + k_neighbors), dtype=torch.float32)
        features[:, :n_max * 3] = x_padded.view(n_eval, -1)
        CHUNK = 10000
        for s in range(0, n_eval, CHUNK):
            e = min(s + CHUNK, n_eval)
            features[s:e, n_max * 3:] = knn_idx[s:e].float().reshape(e - s, -1)
        del x_padded, mask, knn_idx

        targets = targets_tensor.numpy()
        num_tracks = len(features)
        print(f"      -> Inference on {num_tracks} tracks with {num_samples} posterior samples each...")

        # 2. Sampling Loop
        all_samples = []
        sys.setrecursionlimit(3000)

        with torch.no_grad():
            for i in range(0, num_tracks, batch_size):
                if i % 1000 == 0:
                    print(f"         Processing batch starting at index {i}/{num_tracks}...")

                # A. Get Batch
                batch_ctx = features[i : i + batch_size].to(self.device)

                try:
                    # B. Sample bypassing SBI wrapper
                    flow = self.posterior.posterior_estimator

                    batch_samples = flow.sample((num_samples,), condition=batch_ctx)

                    # Ensure shape is (num_samples, batch_size, 4)
                    if batch_samples.shape[0] != num_samples:
                        batch_samples = batch_samples.permute(1, 0, 2)

                    batch_samples = batch_samples.cpu()

                    # C. CLIPPING (Physical Sanity Check)
                    batch_samples[:, :, 0] = torch.clamp(batch_samples[:, :, 0], min=0.001, max=100.0)
                    batch_samples[:, :, 1:] = torch.clamp(batch_samples[:, :, 1:], min=-1.0, max=1.0)

                    all_samples.append(batch_samples)

                except Exception as e:
                    print(f"\n[Error] Batch {i} failed: {e}")
                    continue

        if not all_samples:
            print("[CRITICAL] No samples generated.")
            return pd.DataFrame()

        # Concatenate: (num_samples, total_tracks, 4)
        samples = torch.cat(all_samples, dim=1).numpy()

        # ================================================================
        # 3. LEVEL 1: Point Estimates (same as before, for comparison)
        # ================================================================
        print("      -> Calculating Level 1: Point Estimates...")

        pred_means = np.mean(samples, axis=0)  # (N_tracks, 4)

        pred_E  = pred_means[:, 0]
        pred_vx = pred_means[:, 1]
        pred_vy = pred_means[:, 2]
        pred_vz = pred_means[:, 3]

        true_E  = targets[:, 0]
        true_vx = targets[:, 1]
        true_vy = targets[:, 2]
        true_vz = targets[:, 3]

        # Energy Error
        energy_error = np.abs(pred_E - true_E)

        # Angular Error (from posterior mean direction)
        pred_norms = np.sqrt(pred_vx**2 + pred_vy**2 + pred_vz**2)
        pred_vx_n = pred_vx / (pred_norms + 1e-9)
        pred_vy_n = pred_vy / (pred_norms + 1e-9)
        pred_vz_n = pred_vz / (pred_norms + 1e-9)

        dot_prod = (pred_vx_n * true_vx) + (pred_vy_n * true_vy) + (pred_vz_n * true_vz)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angular_error_deg = np.degrees(np.arccos(dot_prod))

        # ================================================================
        # 4. LEVEL 2: Posterior Uncertainty Quantification
        # ================================================================
        print("      -> Calculating Level 2: Posterior Uncertainty...")

        # 4a. Per-parameter posterior std
        pred_stds = np.std(samples, axis=0)  # (N_tracks, 4)
        energy_std = pred_stds[:, 0]
        vx_std = pred_stds[:, 1]
        vy_std = pred_stds[:, 2]
        vz_std = pred_stds[:, 3]

        # 4b. Energy credible intervals (per track)
        energy_samples = samples[:, :, 0]  # (num_samples, N_tracks)
        energy_ci_90_lo = np.percentile(energy_samples, 5, axis=0)
        energy_ci_90_hi = np.percentile(energy_samples, 95, axis=0)
        energy_ci_90_width = energy_ci_90_hi - energy_ci_90_lo

        energy_ci_50_lo = np.percentile(energy_samples, 25, axis=0)
        energy_ci_50_hi = np.percentile(energy_samples, 75, axis=0)
        energy_ci_50_width = energy_ci_50_hi - energy_ci_50_lo

        # 4c. Angular spread of posterior direction samples (credible cone)
        # For each track, compute the angular spread of sampled directions
        dir_samples = samples[:, :, 1:]  # (num_samples, N_tracks, 3)
        dir_norms = np.sqrt(np.sum(dir_samples**2, axis=-1, keepdims=True))
        dir_samples_normed = dir_samples / (dir_norms + 1e-9)

        # Mean direction per track (from samples)
        mean_dir = np.mean(dir_samples_normed, axis=0)  # (N_tracks, 3)
        mean_dir_norm = np.sqrt(np.sum(mean_dir**2, axis=-1, keepdims=True))
        mean_dir_normed = mean_dir / (mean_dir_norm + 1e-9)

        # Angular deviation of each sample from the mean direction
        # dot product of each sample with the mean direction
        dots_to_mean = np.sum(dir_samples_normed * mean_dir_normed[np.newaxis, :, :], axis=-1)
        dots_to_mean = np.clip(dots_to_mean, -1.0, 1.0)
        angles_to_mean = np.degrees(np.arccos(dots_to_mean))  # (num_samples, N_tracks)

        # R68 credible cone: 68% of samples fall within this angle of the mean
        angular_cone_68 = np.percentile(angles_to_mean, 68, axis=0)  # (N_tracks,)
        angular_cone_90 = np.percentile(angles_to_mean, 90, axis=0)

        # ================================================================
        # 5. LEVEL 3: SBI Calibration (THE key diagnostic)
        # ================================================================
        print("      -> Calculating Level 3: SBI Calibration...")

        # For each nominal credible level (e.g., 50%, 90%), check what fraction
        # of true values actually fall within that interval.
        # A well-calibrated posterior should have: actual coverage ≈ nominal coverage

        credible_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        # Energy calibration
        energy_coverage = []
        for cl in credible_levels:
            alpha = (1 - cl) / 2  # Two-tailed
            lo = np.percentile(energy_samples, alpha * 100, axis=0)
            hi = np.percentile(energy_samples, (1 - alpha) * 100, axis=0)
            # Fraction of true values within this interval
            in_interval = np.mean((true_E >= lo) & (true_E <= hi))
            energy_coverage.append(in_interval)
        energy_coverage = np.array(energy_coverage)

        # Per-parameter calibration for direction components
        param_names = ['E', 'Vx', 'Vy', 'Vz']
        calibration_data = {}
        for p_idx, p_name in enumerate(param_names):
            param_samples = samples[:, :, p_idx]  # (num_samples, N_tracks)
            param_true = targets[:, p_idx]
            coverages = []
            for cl in credible_levels:
                alpha = (1 - cl) / 2
                lo = np.percentile(param_samples, alpha * 100, axis=0)
                hi = np.percentile(param_samples, (1 - alpha) * 100, axis=0)
                in_interval = np.mean((param_true >= lo) & (param_true <= hi))
                coverages.append(in_interval)
            calibration_data[p_name] = np.array(coverages)

        # ================================================================
        # 6. Build comprehensive results DataFrame
        # ================================================================
        results_df = pd.DataFrame({
            # Point estimates
            'true_energy': true_E,
            'pred_energy': pred_E,
            'energy_error': energy_error,
            'true_vx': true_vx,
            'pred_vx': pred_vx_n,
            'true_vy': true_vy,
            'pred_vy': pred_vy_n,
            'true_vz': true_vz,
            'pred_vz': pred_vz_n,
            'angular_error_deg': angular_error_deg,
            # Posterior uncertainty
            'energy_std': energy_std,
            'vx_std': vx_std,
            'vy_std': vy_std,
            'vz_std': vz_std,
            'energy_ci90_width': energy_ci_90_width,
            'energy_ci50_width': energy_ci_50_width,
            'angular_cone_68': angular_cone_68,
            'angular_cone_90': angular_cone_90,
        })

        # Store calibration data and raw samples as attributes for plotting
        self._calibration_data = calibration_data
        self._credible_levels = credible_levels
        self._raw_samples = samples  # Keep for corner plots

        # ================================================================
        # 7. Print Summary
        # ================================================================
        med_E_err = np.median(energy_error)
        med_ang_err = np.median(angular_error_deg)
        flip_rate = np.mean(angular_error_deg > 90) * 100
        r68_ang = np.percentile(angular_error_deg, 68)

        print("\n" + "=" * 65)
        print("  LEVEL 1: Point Estimates")
        print("-" * 65)
        print(f"  Median Energy Error:        {med_E_err:.2f} keV")
        print(f"  Median Angular Error:       {med_ang_err:.2f}°")
        print(f"  68% Containment (R68):      {r68_ang:.2f}°")
        print(f"  Head-Tail Flip Rate:        {flip_rate:.2f}%")
        print("-" * 65)
        print("  LEVEL 2: Posterior Quality (SBI-specific)")
        print("-" * 65)
        print(f"  Median Energy Posterior σ:   {np.median(energy_std):.2f} keV")
        print(f"  Median Energy 90% CI Width: {np.median(energy_ci_90_width):.2f} keV")
        print(f"  Median Angular Cone (68%):  {np.median(angular_cone_68):.2f}°")
        print(f"  Median Angular Cone (90%):  {np.median(angular_cone_90):.2f}°")
        print("-" * 65)
        print("  LEVEL 3: Calibration (actual coverage @ nominal level)")
        print("-" * 65)
        for cl, actual in zip(credible_levels, energy_coverage):
            status = "✓" if abs(actual - cl) < 0.05 else "✗"
            print(f"  Energy {cl*100:5.1f}% CI → actual {actual*100:5.1f}%  {status}")

        # Overall calibration error (ECE = Expected Calibration Error)
        ece_energy = np.mean(np.abs(energy_coverage - credible_levels))
        print(f"\n  Energy ECE (lower=better):  {ece_energy:.4f}")
        print("=" * 65 + "\n")

        return results_df

    def plot_results(self, results_df, save_dir=None):
        """
        Generate comprehensive evaluation plots.

        Creates 3 figure files:
          1. eval_point_estimates.png   — standard energy/angle diagnostics
          2. eval_posterior_quality.png  — uncertainty and credible intervals
          3. eval_calibration.png       — SBI calibration curves
        """
        if results_df.empty:
            print("[Eval] No results to plot.")
            return

        sns.set_theme(style="whitegrid")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # ============================================================
        # FIGURE 1: Point Estimates (classic diagnostics)
        # ============================================================
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))

        # 1a. Energy Scatter
        sns.scatterplot(data=results_df, x='true_energy', y='pred_energy',
                        ax=axes1[0, 0], alpha=0.3, s=10, color='dodgerblue')
        axes1[0, 0].plot([0, 100], [0, 100], 'k--', lw=2)
        axes1[0, 0].set_title("Energy Reconstruction")
        axes1[0, 0].set_xlabel("True Energy (keV)")
        axes1[0, 0].set_ylabel("Predicted Energy (keV)")

        # 1b. Angular Error Histogram
        sns.histplot(results_df['angular_error_deg'], bins=50, ax=axes1[0, 1],
                     color='teal', kde=True)
        axes1[0, 1].set_title("Angular Error Distribution")
        axes1[0, 1].set_xlabel("Error (Degrees)")
        axes1[0, 1].set_xlim(0, 180)

        # 1c. Angular Error vs Energy
        bins = np.linspace(0, 100, 20)
        centers = (bins[:-1] + bins[1:]) / 2
        medians = []
        for i in range(len(bins) - 1):
            mask = (results_df['true_energy'] >= bins[i]) & (results_df['true_energy'] < bins[i + 1])
            if mask.sum() > 0:
                medians.append(results_df.loc[mask, 'angular_error_deg'].median())
            else:
                medians.append(np.nan)
        axes1[1, 0].plot(centers, medians, 'o-', color='crimson', lw=2)
        axes1[1, 0].set_title("Median Angular Error vs. Energy")
        axes1[1, 0].set_xlabel("True Energy (keV)")
        axes1[1, 0].set_ylabel("Median Angular Error (°)")
        axes1[1, 0].set_ylim(0, 90)

        # 1d. Energy Error % vs Energy
        energy_pct_err = (results_df['pred_energy'] - results_df['true_energy']) / (results_df['true_energy'] + 1e-9) * 100
        axes1[1, 1].scatter(results_df['true_energy'], energy_pct_err,
                            alpha=0.1, s=5, color='purple')
        axes1[1, 1].axhline(0, color='k', linestyle='--')
        axes1[1, 1].set_title("Energy Bias (%) vs Energy")
        axes1[1, 1].set_xlabel("True Energy (keV)")
        axes1[1, 1].set_ylabel("Energy Error (%)")
        axes1[1, 1].set_ylim(-50, 50)

        fig1.tight_layout()
        if save_dir:
            fig1.savefig(f"{save_dir}/eval_point_estimates.png", dpi=150, bbox_inches='tight')
            print(f"[Plot] Saved {save_dir}/eval_point_estimates.png")
        plt.close(fig1)

        # ============================================================
        # FIGURE 2: Posterior Quality
        # ============================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

        # 2a. Energy Uncertainty vs Error (are confident predictions accurate?)
        axes2[0, 0].scatter(results_df['energy_std'], results_df['energy_error'],
                            alpha=0.15, s=8, color='darkorange')
        axes2[0, 0].set_xlabel("Posterior σ (Energy, keV)")
        axes2[0, 0].set_ylabel("Actual Energy Error (keV)")
        axes2[0, 0].set_title("Uncertainty vs Error (Energy)")
        # Add diagonal guide: error should scale with uncertainty
        max_val = max(results_df['energy_std'].max(), results_df['energy_error'].max())
        axes2[0, 0].plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='σ = error')
        axes2[0, 0].legend()

        # 2b. Angular Cone vs Angular Error
        axes2[0, 1].scatter(results_df['angular_cone_68'], results_df['angular_error_deg'],
                            alpha=0.15, s=8, color='seagreen')
        axes2[0, 1].set_xlabel("68% Credible Cone (°)")
        axes2[0, 1].set_ylabel("Actual Angular Error (°)")
        axes2[0, 1].set_title("Uncertainty vs Error (Direction)")
        max_val = max(results_df['angular_cone_68'].max(), results_df['angular_error_deg'].max())
        axes2[0, 1].plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5)

        # 2c. Energy CI Width vs True Energy
        e_bins = np.linspace(0, 100, 20)
        e_centers = (e_bins[:-1] + e_bins[1:]) / 2
        ci_medians = []
        for i in range(len(e_bins) - 1):
            mask = (results_df['true_energy'] >= e_bins[i]) & (results_df['true_energy'] < e_bins[i + 1])
            if mask.sum() > 0:
                ci_medians.append(results_df.loc[mask, 'energy_ci90_width'].median())
            else:
                ci_medians.append(np.nan)
        axes2[1, 0].plot(e_centers, ci_medians, 'o-', color='royalblue', lw=2)
        axes2[1, 0].set_title("Median 90% CI Width vs. Energy")
        axes2[1, 0].set_xlabel("True Energy (keV)")
        axes2[1, 0].set_ylabel("90% CI Width (keV)")

        # 2d. Angular Cone vs True Energy
        cone_medians = []
        for i in range(len(e_bins) - 1):
            mask = (results_df['true_energy'] >= e_bins[i]) & (results_df['true_energy'] < e_bins[i + 1])
            if mask.sum() > 0:
                cone_medians.append(results_df.loc[mask, 'angular_cone_68'].median())
            else:
                cone_medians.append(np.nan)
        axes2[1, 1].plot(e_centers, cone_medians, 'o-', color='darkviolet', lw=2)
        axes2[1, 1].set_title("Median 68% Angular Cone vs. Energy")
        axes2[1, 1].set_xlabel("True Energy (keV)")
        axes2[1, 1].set_ylabel("68% Credible Cone (°)")

        fig2.tight_layout()
        if save_dir:
            fig2.savefig(f"{save_dir}/eval_posterior_quality.png", dpi=150, bbox_inches='tight')
            print(f"[Plot] Saved {save_dir}/eval_posterior_quality.png")
        plt.close(fig2)

        # ============================================================
        # FIGURE 3: Calibration Curves
        # ============================================================
        if hasattr(self, '_calibration_data') and self._calibration_data:
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

            cl = self._credible_levels

            # 3a. Per-parameter calibration curves
            colors = {'E': 'dodgerblue', 'Vx': 'crimson', 'Vy': 'seagreen', 'Vz': 'darkorange'}
            for p_name, coverages in self._calibration_data.items():
                axes3[0].plot(cl * 100, coverages * 100, 'o-', color=colors[p_name],
                              label=p_name, lw=2, markersize=5)
            axes3[0].plot([0, 100], [0, 100], 'k--', lw=2, alpha=0.5, label='Perfect')
            axes3[0].set_xlabel("Nominal Credible Level (%)")
            axes3[0].set_ylabel("Actual Coverage (%)")
            axes3[0].set_title("SBI Calibration (all parameters)")
            axes3[0].legend()
            axes3[0].set_xlim(0, 100)
            axes3[0].set_ylim(0, 100)
            axes3[0].set_aspect('equal')
            axes3[0].grid(True, alpha=0.3)

            # 3b. Calibration error per parameter (bar chart)
            param_ece = {}
            for p_name, coverages in self._calibration_data.items():
                param_ece[p_name] = np.mean(np.abs(coverages - cl))

            bars = axes3[1].bar(param_ece.keys(), param_ece.values(),
                                color=[colors[k] for k in param_ece.keys()])
            axes3[1].set_ylabel("Expected Calibration Error (ECE)")
            axes3[1].set_title("Calibration Error by Parameter\n(lower = better calibrated)")
            axes3[1].axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Good threshold')
            axes3[1].legend()

            fig3.tight_layout()
            if save_dir:
                fig3.savefig(f"{save_dir}/eval_calibration.png", dpi=150, bbox_inches='tight')
                print(f"[Plot] Saved {save_dir}/eval_calibration.png")
            plt.close(fig3)

    def plot_corner(self, track_idx, results_df, save_dir=None):
        """
        Plot a corner plot (pair plot) of the posterior for a single track.

        This is the gold standard visualization for SBI: shows the full
        joint posterior distribution, not just a point estimate.

        Args:
            track_idx: index of the track to visualize
            results_df: the results DataFrame (used for true values)
            save_dir: if set, saves to file
        """
        if not hasattr(self, '_raw_samples') or self._raw_samples is None:
            print("[Error] No raw samples stored. Re-run eval first.")
            return

        samples = self._raw_samples[:, track_idx, :]  # (num_samples, 4)
        true_vals = [
            results_df.iloc[track_idx]['true_energy'],
            results_df.iloc[track_idx]['true_vx'],
            results_df.iloc[track_idx]['true_vy'],
            results_df.iloc[track_idx]['true_vz'],
        ]
        param_names = ['Energy (keV)', 'Vx', 'Vy', 'Vz']

        n_params = 4
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: 1D marginal histogram
                    ax.hist(samples[:, i], bins=30, density=True, alpha=0.7,
                            color='steelblue', edgecolor='white')
                    ax.axvline(true_vals[i], color='red', lw=2, label='Truth')
                    ax.axvline(np.mean(samples[:, i]), color='orange', lw=2,
                               linestyle='--', label='Post. Mean')
                    if i == 0:
                        ax.legend(fontsize=8)
                else:
                    # Off-diagonal: 2D scatter
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=3,
                               color='steelblue')
                    ax.axvline(true_vals[j], color='red', lw=1, alpha=0.5)
                    ax.axhline(true_vals[i], color='red', lw=1, alpha=0.5)

                # Labels
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j], fontsize=10)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(param_names[i], fontsize=10)
                else:
                    ax.set_yticklabels([])

        true_E = results_df.iloc[track_idx]['true_energy']
        err = results_df.iloc[track_idx]['angular_error_deg']
        fig.suptitle(f"Track {track_idx} Posterior | True E={true_E:.1f} keV | "
                     f"Angular Err={err:.1f}°", fontsize=13, y=1.01)
        fig.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            path = f"{save_dir}/corner_track_{track_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Saved {path}")
        else:
            plt.show()
        plt.close(fig)


# ==========================================
# MOCK FOR TESTING
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "models/mcpe_3d_posterior.pt"
    EVAL_DATA = "data/mcpe-3d/mcpe_3d_eval.csv"

    class MockEstimator:
        def parameters(self):
            yield torch.tensor([0.0])

    class MockPosterior:
        def __init__(self):
            self.posterior_estimator = MockEstimator()

        def sample_batched(self, shape, x, **kwargs):
            n_samples = shape[0]
            batch_size = x.shape[0]
            return torch.randn(n_samples, batch_size, 4) + torch.tensor([50.0, 1.0, 0.0, 0.0])

    posterior = MockPosterior()

    evaluator = ContinuousEvaluator3D(posterior)
    if not Path(EVAL_DATA).exists():
        print("Note: Eval data not found, skipping run.")
    else:
        df_res = evaluator.run_eval(EVAL_DATA, num_samples=200)
        evaluator.plot_results(df_res, save_dir="results")
