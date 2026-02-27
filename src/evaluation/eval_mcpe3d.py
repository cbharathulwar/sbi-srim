import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import the new PAD/LOADER instead of the feature preprocessor
try:
    from src.utils.data_utils import load_and_pad_3d_tracks
except ImportError:
    sys.path.append(".") 
    from src.utils.data_utils import load_and_pad_3d_tracks

class ContinuousEvaluator3D:
    def __init__(self, posterior, device=None):
        self.posterior = posterior
        
        try:
            self.device = next(posterior.posterior_estimator.parameters()).device
        except:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
        print(f"[Eval] Model detected on device: {self.device}")

    def run_eval(self, eval_csv_path, num_samples=50, batch_size=500):
        # NOTE: Lowered default batch_size to 500 because 3D Point Clouds take more RAM than flat features
        
        # 1. Load and Pad Raw Data directly into Tensors
        print(f"[Eval] Loading Raw Data from {eval_csv_path}...")
        # We use max_points="auto" to dynamically fit the eval set
        features, targets_tensor = load_and_pad_3d_tracks(eval_csv_path, max_points="auto")
        
        targets = targets_tensor.numpy()
        num_tracks = len(features)
        print(f"      -> Inference on {num_tracks} tracks...")

        # 2. Sampling Loop (Using the "Unsafe" Bypass to avoid Prior Leakage hangs)
        all_samples = []
        sys.setrecursionlimit(3000)
        
        with torch.no_grad():
            for i in range(0, num_tracks, batch_size):
                if i % 1000 == 0:
                    print(f"         Processing batch starting at index {i}/{num_tracks}...")
                
               # A. Get Batch (Shape: Batch_Size, Max_Points, 3)
                batch_ctx = features[i : i + batch_size].to(self.device)
                
                try:
                    # B. Sample bypassing SBI wrapper
                    flow = self.posterior.posterior_estimator
                    
                    # THE FIX: Wrap num_samples in a tuple so PyTorch knows it's a shape!
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

        # Concatenate along the Batch Dimension (Dim 1)
        samples = torch.cat(all_samples, dim=1)
        
        # 3. Metrics Calculation
        print("      -> Calculating Physics Metrics...")
        
        pred_means = torch.mean(samples, dim=0).numpy()
        
        pred_E  = pred_means[:, 0]
        pred_vx = pred_means[:, 1]
        pred_vy = pred_means[:, 2]
        pred_vz = pred_means[:, 3]
        
        # Targets mapping matches load_and_pad_3d_tracks output: [E, Vx, Vy, Vz]
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
        
        # Build Results DataFrame (Now correctly including Vy and Vz)
        results_df = pd.DataFrame({
            'true_energy': true_E,
            'pred_energy': pred_E,
            'energy_error': energy_error,
            'true_vx': true_vx,
            'pred_vx': pred_vx,
            'true_vy': true_vy,
            'pred_vy': pred_vy,
            'true_vz': true_vz,
            'pred_vz': pred_vz,
            'angular_error_deg': angular_error_deg
        })
        
        # Level 1 & Level 2 Metrics
        med_E_err = np.median(energy_error)
        med_ang_err = np.median(angular_error_deg)
        flip_rate = np.mean(angular_error_deg > 90) * 100
        r68_ang = np.percentile(angular_error_deg, 68) # The 68% containment radius
        
        print("\n" + "="*40)
        print(f"[SUMMARY] Median Energy Error: {med_E_err:.2f} keV")
        print(f"[SUMMARY] Median Angle Error:  {med_ang_err:.2f}°")
        print(f"[SUMMARY] 68% Containment (R68): {r68_ang:.2f}°")
        print(f"[SUMMARY] Head-Tail Flip Rate: {flip_rate:.2f}%")
        print("="*40 + "\n")
        
        return results_df