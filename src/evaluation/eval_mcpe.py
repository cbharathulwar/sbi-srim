# src/evaluation/eval_mcpe.py

import os
import sys

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Fix Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_utils import preprocess_mcpe

try:
    from src.utils.data_utils import preprocess_mcpe_3d
except ImportError:
    sys.path.append(".")
    from src.utils.data_utils import preprocess_mcpe_3d


# ---------------------------------------------------------
# HELPER: Circular Math
# ---------------------------------------------------------
def get_shortest_angular_diff(angle_a, angle_b):
    diff = torch.abs(angle_a - angle_b)
    return torch.min(diff, 360 - diff)

def tensor_to_angle(sin_vals, cos_vals):
    rads = torch.atan2(sin_vals, cos_vals)
    deg = torch.rad2deg(rads)
    return (deg + 360) % 360


# ---------------------------------------------------------
# MCPE 2D EVALUATOR (Continuous In-Plane Angle)
# ---------------------------------------------------------
class ContinuousEvaluator:
    def __init__(self, posterior):
        self.posterior = posterior

        # AUTO-DETECT DEVICE from the posterior
        try:
            self.device = next(posterior.posterior_estimator.parameters()).device
        except:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def run_eval(self, eval_csv_path, num_samples=1000, batch_size=1000):
        print(f"[Eval] Preprocessing Eval Data from {eval_csv_path}...")

        # A. Preprocess
        features, targets, track_ids, _, _ = preprocess_mcpe(eval_csv_path)

        num_tracks = len(features)
        print(f"      -> Inference on {num_tracks} tracks (Manual Batching)...")

        # B. Manual Sampling Loop (Bypassing SBI Wrapper)
        flow = self.posterior.posterior_estimator
        flow.eval()

        batch_size = 1000
        all_samples = []

        with torch.no_grad():
                for i in range(0, num_tracks, batch_size):
                    if i % 5000 == 0:
                        print(f"         Processing batch starting at index {i}/{num_tracks}...")

                    # A. Get Batch
                    batch_ctx = features[i : i + batch_size].to(self.device)

                    try:
                        # --- THE FIX: BYPASS SBI WRAPPER ---
                        # We access the raw neural net (.posterior_estimator) directly.
                        # This gives us the raw numbers without 'Rejection Sampling' blocking us.
                        flow = self.posterior.posterior_estimator

                        # We try the two standard keywords for conditional flow sampling
                        try:
                            # Modern SBI/nflows uses 'context'
                            batch_samples = flow.sample(num_samples, context=batch_ctx)
                        except TypeError:
                            # Older versions use 'cond_inputs' or just 'x'
                            try:
                                batch_samples = flow.sample(num_samples, cond_inputs=batch_ctx)
                            except TypeError:
                                # Last resort for some specific flow types
                                batch_samples = flow.sample(num_samples, x=batch_ctx)

                        # Output is (batch_size, num_samples, dim) or (num_samples, batch, dim)
                        # We force it to be (num_samples, batch_size, 4)
                        if batch_samples.shape[0] != num_samples:
                            batch_samples = batch_samples.permute(1, 0, 2)

                        batch_samples = batch_samples.cpu()

                        # C. CLIPPING (Manual Physics Enforcement)
                        # Instead of rejecting, we just snap the values to valid ranges

                        # Energy (Index 0): Snap negatives to 0.001
                        batch_samples[:, :, 0] = torch.clamp(batch_samples[:, :, 0], min=0.001, max=100.0)

                        # Unit Vectors (Index 1-3): Snap to [-1, 1]
                        batch_samples[:, :, 1:] = torch.clamp(batch_samples[:, :, 1:], min=-1.0, max=1.0)

                        all_samples.append(batch_samples)

                    except Exception as e:
                        print(f"\n[Error] Batch {i} failed: {e}")
                        continue

        if not all_samples:
            print("[CRITICAL] No samples generated.")
            return pd.DataFrame()

        # Concatenate along the Batch Dimension (Dim 1)
        # Shape becomes: (Num_Samples, Total_Tracks, Dim)
        samples = torch.cat(all_samples, dim=1)

        # C. Metrics
        print("      -> Calculating Metrics...")
        # Average over Dim 0 (Samples) -> Collapses to (Total_Tracks, Dim)
        pred_means = torch.mean(samples, dim=0)

        pred_E   = pred_means[:, 0]
        pred_sin = pred_means[:, 1]
        pred_cos = pred_means[:, 2]
        pred_angles = tensor_to_angle(pred_sin, pred_cos)

        true_E   = targets[:, 0]
        true_sin = targets[:, 1]
        true_cos = targets[:, 2]
        true_angles = tensor_to_angle(true_sin, true_cos)

        energy_errors = torch.abs(pred_E - true_E)
        angular_errors = get_shortest_angular_diff(pred_angles, true_angles)

        results_df = pd.DataFrame({
            'track_id': track_ids,
            'true_energy': true_E.numpy(),
            'pred_energy': pred_E.numpy(),
            'energy_error': energy_errors.numpy(),
            'true_angle': true_angles.numpy(),
            'pred_angle': pred_angles.numpy(),
            'angular_error': angular_errors.numpy()
        })

        print(f"\n[SUMMARY] Median Energy Error: {results_df['energy_error'].median():.2f} keV")
        print(f"[SUMMARY] Median Angle Error:  {results_df['angular_error'].median():.2f} deg")

        return results_df

    def plot_results(self, results_df):
        if results_df.empty:
            return

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Energy Parity
        sns.scatterplot(data=results_df, x='true_energy', y='pred_energy', ax=axes[0,0], alpha=0.3, s=10, color='dodgerblue')
        axes[0,0].plot([0, 100], [0, 100], 'r--', lw=2)
        axes[0,0].set_title(f"Energy Parity (MAE: {results_df['energy_error'].mean():.2f} keV)")

        # 2. Angle Parity
        sns.scatterplot(data=results_df, x='true_angle', y='pred_angle', ax=axes[0,1], alpha=0.3, s=10, color='seagreen')
        axes[0,1].plot([0, 360], [0, 360], 'r--', lw=2)
        axes[0,1].set_title(f"Angular Parity (Mean Error: {results_df['angular_error'].mean():.1f}°)")
        axes[0,1].set_xlim(0, 360); axes[0,1].set_ylim(0, 360)

        # 3. Error Hist
        sns.histplot(results_df['angular_error'], bins=50, ax=axes[1,0], color='teal')
        axes[1,0].set_title("Angular Error Distribution")

        # 4. Error vs Energy
        sns.scatterplot(data=results_df, x='true_energy', y='angular_error', ax=axes[1,1], alpha=0.3, color='purple', s=10)
        axes[1,1].set_title("Error vs Energy")

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------
# MCPE 3D EVALUATOR (Full 3D Unit Vector)
# ---------------------------------------------------------
class ContinuousEvaluator3D:
    def __init__(self, posterior, device=None):
        self.posterior = posterior

        # AUTO-DETECT DEVICE from the posterior itself
        # This prevents the mismatch error by asking the model "Where are you?"
        try:
            # Check the first parameter of the underlying neural net
            self.device = next(posterior.posterior_estimator.parameters()).device
        except:
            # Fallback if we can't find it (rare)
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"[Eval] Model detected on device: {self.device}")

    def run_eval(self, eval_csv_path, num_samples=50, batch_size=1000):
        print(f"[Eval] Loading Raw Data from {eval_csv_path}...")

        # 1. Load Raw Data
        raw_df = pd.read_csv(eval_csv_path)

        # 2. Preprocess
        print("[Eval] Preprocessing features...")
        df_processed = preprocess_mcpe_3d(raw_df)

        # 3. Split into Inputs (Features) and Targets (Truth)
        target_cols = ["energy_true", "vx_true", "vy_true", "vz_true"]
        drop_cols = target_cols + ["run_id", "ion_number"]
        feature_cols = [c for c in df_processed.columns if c not in drop_cols]

        features = torch.tensor(df_processed[feature_cols].values, dtype=torch.float32)
        targets = df_processed[target_cols].values

        num_tracks = len(features)
        print(f"      -> Inference on {num_tracks} tracks...")

        # 4. Sampling Loop
        all_samples = []
        sys.setrecursionlimit(3000)

        with torch.no_grad():
            for i in range(0, num_tracks, batch_size):
                if i % 5000 == 0:
                    print(f"         Processing batch starting at index {i}/{num_tracks}...")

                # A. Get Batch and FORCE IT TO THE MODEL'S DEVICE
                batch_ctx = features[i : i + batch_size].to(self.device)

                try:
                    # B. Sample Batched
                    batch_samples = self.posterior.sample_batched(
                        (num_samples,),
                        x=batch_ctx,
                        show_progress_bars=False,
                        max_sampling_batch_size=2500 # explicit limit to stop warnings
                    )

                    # Move result back to CPU immediately to save GPU memory
                    batch_samples = batch_samples.cpu()

                    # C. CLIPPING (Physical Sanity Check)
                    batch_samples[:, :, 0] = torch.clamp(batch_samples[:, :, 0], min=0.001, max=100.0)
                    batch_samples[:, :, 1:] = torch.clamp(batch_samples[:, :, 1:], min=-1.0, max=1.0)

                    all_samples.append(batch_samples)

                except Exception as e:
                    print(f"\n[Error] Batch {i} failed: {e}")
                    # Debug info
                    print(f"        Input device: {batch_ctx.device}")
                    print(f"        Model device: {self.device}")
                    continue

        if not all_samples:
            print("[CRITICAL] No samples generated.")
            return pd.DataFrame()

        # Concatenate along the Batch Dimension (Dim 1)
        samples = torch.cat(all_samples, dim=1)

        # 5. Metrics Calculation
        print("      -> Calculating Physics Metrics...")

        pred_means = torch.mean(samples, dim=0).numpy()

        pred_E  = pred_means[:, 0]
        pred_vx = pred_means[:, 1]
        pred_vy = pred_means[:, 2]
        pred_vz = pred_means[:, 3]

        true_E  = targets[:, 0]
        true_vx = targets[:, 1]
        true_vy = targets[:, 2]
        true_vz = targets[:, 3]

        # --- Energy Error ---
        energy_error = np.abs(pred_E - true_E)

        # --- Angular Error ---
        pred_norms = np.sqrt(pred_vx**2 + pred_vy**2 + pred_vz**2)
        pred_vx /= (pred_norms + 1e-9)
        pred_vy /= (pred_norms + 1e-9)
        pred_vz /= (pred_norms + 1e-9)

        dot_prod = (pred_vx * true_vx) + (pred_vy * true_vy) + (pred_vz * true_vz)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angular_error_deg = np.degrees(np.arccos(dot_prod))

        results_df = pd.DataFrame({
            'true_energy': true_E,
            'pred_energy': pred_E,
            'energy_error': energy_error,
            'true_vx': true_vx,
            'pred_vx': pred_vx,
            'angular_error_deg': angular_error_deg,
            'true_vy': true_vy,
            'pred_vy': pred_vy,
            'true_vz': true_vz,
            'pred_vz': pred_vz,
        })

        med_E_err = np.median(energy_error)
        med_ang_err = np.median(angular_error_deg)
        flip_rate = np.mean(angular_error_deg > 90) * 100

        print("\n" + "="*40)
        print(f"[SUMMARY] Median Energy Error: {med_E_err:.2f} keV")
        print(f"[SUMMARY] Median Angle Error:  {med_ang_err:.2f}")
        print(f"[SUMMARY] Head-Tail Flip Rate: {flip_rate:.2f}%")
        print("="*40 + "\n")

        return results_df

    def plot_results(self, results_df, save_dir=None):
        if results_df.empty:
            return

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Energy Scatter
        sns.scatterplot(data=results_df, x='true_energy', y='pred_energy', ax=axes[0,0], alpha=0.3, s=10, color='dodgerblue')
        axes[0,0].plot([0, 100], [0, 100], 'k--', lw=2)
        axes[0,0].set_title(f"Energy Reconstruction")
        axes[0,0].set_xlabel("True Energy (keV)")
        axes[0,0].set_ylabel("Predicted Energy (keV)")

        # 2. Angular Error Histogram
        sns.histplot(results_df['angular_error_deg'], bins=50, ax=axes[0,1], color='teal', kde=True)
        axes[0,1].set_title("Angular Error Distribution")
        axes[0,1].set_xlabel("Error (Degrees)")
        axes[0,1].set_xlim(0, 180)

        # 3. Angular Error vs Energy
        bins = np.linspace(0, 100, 20)
        centers = (bins[:-1] + bins[1:]) / 2
        medians = []
        for i in range(len(bins)-1):
            mask = (results_df['true_energy'] >= bins[i]) & (results_df['true_energy'] < bins[i+1])
            if mask.sum() > 0:
                medians.append(results_df.loc[mask, 'angular_error_deg'].median())
            else:
                medians.append(np.nan)

        axes[1,0].plot(centers, medians, 'o-', color='crimson', lw=2)
        axes[1,0].set_title("Median Angular Error vs. Energy")
        axes[1,0].set_ylim(0, 90)

        # 4. Energy Error % vs Energy
        energy_pct_err = (results_df['pred_energy'] - results_df['true_energy']) / results_df['true_energy'] * 100
        axes[1,1].scatter(results_df['true_energy'], energy_pct_err, alpha=0.1, s=5, color='purple')
        axes[1,1].axhline(0, color='k', linestyle='--')
        axes[1,1].set_title("Energy Bias (%) vs Energy")
        axes[1,1].set_ylim(-50, 50)

        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_dir}/eval_plots.png")
            print(f"Plots saved to {save_dir}/eval_plots.png")
        else:
            plt.show()


# ==========================================
# MOCK FOR TESTING
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "models/mcpe_3d_posterior.pt"
    EVAL_DATA = "data/mcpe-3d/mcpe_3d_eval.csv"

    class MockEstimator:
        def parameters(self):
            # Fake a parameter on CPU
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
        df_res = evaluator.run_eval(EVAL_DATA, num_samples=10)
        evaluator.plot_results(df_res, save_dir="results")
