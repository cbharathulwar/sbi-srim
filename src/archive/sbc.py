"""
SBI Calibration Check for Your Trained Posterior
Run this to validate your posterior's uncertainty estimates.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# Paths
EVAL_CSV = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/mcpe-3d/mcpe_3d_eval_10k.csv"
POSTERIOR_PATH = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/results/pipeline_b/pointnet_3d_posterior.pt"
OUTPUT_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/results/pipeline_b/calibration_analysis")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load posterior
print("[1/5] Loading posterior...")
posterior = torch.load(POSTERIOR_PATH)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"     Using device: {device}")

# Load and preprocess eval data
print("[2/5] Loading evaluation data...")

from src.preprocessing.mcpe3d import load_and_pad_3d_tracks

print(f"     Loading from: {EVAL_CSV}")
features, targets_tensor = load_and_pad_3d_tracks(EVAL_CSV, max_points="auto")
targets = targets_tensor.numpy()

print(f"     Features shape: {features.shape}")
print(f"     Targets shape: {targets.shape}")
print(f"     Total tracks available: {len(features)}")

# Subsample to 1000 tracks for faster analysis (or use all 10k if you want)
NUM_TRACKS = min(1000, len(features))  # Change to 10000 to use all tracks
indices = np.random.choice(len(features), NUM_TRACKS, replace=False)

features_subset = features[indices].to(device)
targets_subset = targets[indices]

print(f"\n[3/5] Using {NUM_TRACKS} tracks for calibration analysis")

print(f"[4/5] Sampling from posterior for {NUM_TRACKS} tracks...")
print("     This will take a few minutes...")

NUM_POSTERIOR_SAMPLES = 500

all_samples = []
batch_size = 50  # Smaller batch for point clouds

with torch.no_grad():
    for i in range(0, NUM_TRACKS, batch_size):
        batch_features = features_subset[i:i+batch_size]
        
        # Sample from posterior
        try:
            batch_samples = posterior.sample_batched(
                (NUM_POSTERIOR_SAMPLES,),
                x=batch_features,
                show_progress_bars=False,
                max_sampling_batch_size=2500
            )
        except:
            # Fallback if sample_batched doesn't work
            try:
                batch_samples = posterior.posterior_estimator.sample(
                    (NUM_POSTERIOR_SAMPLES,), 
                    condition=batch_features
                )
                if batch_samples.shape[0] != NUM_POSTERIOR_SAMPLES:
                    batch_samples = batch_samples.permute(1, 0, 2)
            except Exception as e:
                print(f"     Error sampling batch {i}: {e}")
                continue
        
        all_samples.append(batch_samples.cpu())
        
        if (i // batch_size) % 10 == 0:
            print(f"     Processed {i}/{NUM_TRACKS} tracks...")

samples = torch.cat(all_samples, dim=1)  # [num_samples, num_tracks, 4]
print(f"     Samples shape: {samples.shape}")

# Convert to numpy
samples_np = samples.numpy()

print("[5/5] Computing calibration metrics...")

# ============================================
# CALIBRATION ANALYSIS
# ============================================

results = {}

# 1. COVERAGE TEST
print("\n" + "="*60)
print("COVERAGE TEST: Do credible intervals contain true values?")
print("="*60)

param_names = ['Energy', 'vx', 'vy', 'vz']
coverages = {68: {}, 95: {}}

for param_idx, param_name in enumerate(param_names):
    in_68 = []
    in_95 = []
    
    for track_idx in range(NUM_TRACKS):
        true_val = targets_subset[track_idx, param_idx]
        param_samples = samples_np[:, track_idx, param_idx]
        
        # 68% credible interval
        lower_68 = np.percentile(param_samples, 16)
        upper_68 = np.percentile(param_samples, 84)
        in_68.append(lower_68 <= true_val <= upper_68)
        
        # 95% credible interval
        lower_95 = np.percentile(param_samples, 2.5)
        upper_95 = np.percentile(param_samples, 97.5)
        in_95.append(lower_95 <= true_val <= upper_95)
    
    coverage_68 = np.mean(in_68) * 100
    coverage_95 = np.mean(in_95) * 100
    
    coverages[68][param_name] = coverage_68
    coverages[95][param_name] = coverage_95
    
    print(f"\n{param_name}:")
    print(f"  68% CI coverage: {coverage_68:.1f}% (target: 68%)")
    print(f"  95% CI coverage: {coverage_95:.1f}% (target: 95%)")
    
    # Interpretation
    if 60 <= coverage_68 <= 76:
        status_68 = "✓ GOOD"
    elif coverage_68 < 60:
        status_68 = "⚠️  OVERCONFIDENT"
    else:
        status_68 = "⚠️  UNDERCONFIDENT"
    
    if 90 <= coverage_95 <= 98:
        status_95 = "✓ GOOD"
    elif coverage_95 < 90:
        status_95 = "⚠️  OVERCONFIDENT"
    else:
        status_95 = "⚠️  UNDERCONFIDENT"
    
    print(f"  Status: {status_68} (68%), {status_95} (95%)")

results['coverages'] = coverages

# 2. POSTERIOR UNCERTAINTY ANALYSIS
print("\n" + "="*60)
print("UNCERTAINTY METRICS")
print("="*60)

uncertainties = {}

for param_idx, param_name in enumerate(param_names):
    # Standard deviation across posterior samples
    stds = np.std(samples_np[:, :, param_idx], axis=0)
    
    uncertainties[param_name] = stds
    
    print(f"\n{param_name}:")
    print(f"  Mean uncertainty (std): {np.mean(stds):.4f}")
    print(f"  Median uncertainty: {np.median(stds):.4f}")
    print(f"  Min uncertainty: {np.min(stds):.4f}")
    print(f"  Max uncertainty: {np.max(stds):.4f}")

results['uncertainties'] = uncertainties

# 3. ANGULAR UNCERTAINTY (Special handling for direction)
print("\n" + "="*60)
print("ANGULAR UNCERTAINTY")
print("="*60)

angular_uncertainties = []
angular_errors = []

for track_idx in range(NUM_TRACKS):
    # True direction
    true_vx = targets_subset[track_idx, 1]
    true_vy = targets_subset[track_idx, 2]
    true_vz = targets_subset[track_idx, 3]
    
    # Posterior samples for direction
    vx_samples = samples_np[:, track_idx, 1]
    vy_samples = samples_np[:, track_idx, 2]
    vz_samples = samples_np[:, track_idx, 3]
    
    # Normalize
    norms = np.sqrt(vx_samples**2 + vy_samples**2 + vz_samples**2)
    vx_samples = vx_samples / (norms + 1e-9)
    vy_samples = vy_samples / (norms + 1e-9)
    vz_samples = vz_samples / (norms + 1e-9)
    
    true_norm = np.sqrt(true_vx**2 + true_vy**2 + true_vz**2)
    true_vx_n = true_vx / (true_norm + 1e-9)
    true_vy_n = true_vy / (true_norm + 1e-9)
    true_vz_n = true_vz / (true_norm + 1e-9)
    
    # Angular spread (std of dot products converted to angle)
    dot_prods = vx_samples * true_vx_n + vy_samples * true_vy_n + vz_samples * true_vz_n
    dot_prods = np.clip(dot_prods, -1, 1)
    angles = np.degrees(np.arccos(dot_prods))
    
    angular_uncertainty = np.std(angles)
    angular_uncertainties.append(angular_uncertainty)
    
    # Angular error (mean prediction)
    mean_vx = np.mean(vx_samples)
    mean_vy = np.mean(vy_samples)
    mean_vz = np.mean(vz_samples)
    
    dot_prod = mean_vx * true_vx_n + mean_vy * true_vy_n + mean_vz * true_vz_n
    dot_prod = np.clip(dot_prod, -1, 1)
    angular_error = np.degrees(np.arccos(dot_prod))
    angular_errors.append(angular_error)

angular_uncertainties = np.array(angular_uncertainties)
angular_errors = np.array(angular_errors)

print(f"Mean angular uncertainty: {np.mean(angular_uncertainties):.2f}°")
print(f"Median angular uncertainty: {np.median(angular_uncertainties):.2f}°")
print(f"Mean angular error: {np.mean(angular_errors):.2f}°")
print(f"Median angular error: {np.median(angular_errors):.2f}°")

# 4. UNCERTAINTY-ERROR CORRELATION
print("\n" + "="*60)
print("UNCERTAINTY-ERROR CORRELATION")
print("="*60)
print("(High uncertainty should correlate with high error)")

corr, pval = spearmanr(angular_uncertainties, angular_errors)

print(f"\nAngular: Correlation = {corr:.3f}, p-value = {pval:.4f}")

if corr > 0.3 and pval < 0.05:
    print("✓ GOOD: Uncertainty is informative!")
elif corr > 0.1 and pval < 0.05:
    print("⚠️  WEAK: Some correlation, but uncertainty could be better")
else:
    print("❌ BAD: Uncertainty is not informative")

results['correlation'] = {'corr': corr, 'pval': pval}
results['angular_uncertainties'] = angular_uncertainties
results['angular_errors'] = angular_errors

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

sns.set_theme(style="whitegrid")

# Plot 1: Coverage Summary
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x = np.arange(len(param_names))
width = 0.35

coverage_68_vals = [coverages[68][name] for name in param_names]
coverage_95_vals = [coverages[95][name] for name in param_names]

bars1 = ax.bar(x - width/2, coverage_68_vals, width, label='68% CI', alpha=0.8)
bars2 = ax.bar(x + width/2, coverage_95_vals, width, label='95% CI', alpha=0.8)

ax.axhline(68, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target 68%')
ax.axhline(95, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Target 95%')

ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('Posterior Calibration: Coverage Test', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(param_names)
ax.legend()
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coverage_summary.png", dpi=150)
print(f"✓ Saved: {OUTPUT_DIR / 'coverage_summary.png'}")

# Plot 2: Uncertainty vs Error
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.scatter(angular_uncertainties, angular_errors, alpha=0.3, s=20)
ax.set_xlabel('Posterior Uncertainty (degrees)', fontsize=12)
ax.set_ylabel('Angular Error (degrees)', fontsize=12)
ax.set_title(f'Uncertainty vs Error (Correlation: {corr:.3f})', 
             fontsize=14, fontweight='bold')

# Add trend line
z = np.polyfit(angular_uncertainties, angular_errors, 1)
p = np.poly1d(z)
x_trend = np.linspace(angular_uncertainties.min(), angular_uncertainties.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, label='Trend')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "uncertainty_vs_error.png", dpi=150)
print(f"✓ Saved: {OUTPUT_DIR / 'uncertainty_vs_error.png'}")

# Plot 3: Uncertainty Distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, param_name in enumerate(param_names):
    ax = axes[idx]
    ax.hist(uncertainties[param_name], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f'{param_name} Uncertainty (std)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{param_name} Posterior Uncertainty Distribution', fontsize=11)
    ax.axvline(np.median(uncertainties[param_name]), color='red', 
               linestyle='--', linewidth=2, label=f'Median: {np.median(uncertainties[param_name]):.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "uncertainty_distributions.png", dpi=150)
print(f"✓ Saved: {OUTPUT_DIR / 'uncertainty_distributions.png'}")

# Plot 4: Energy-stratified analysis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

energy_bins = np.linspace(0, 100, 11)
bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2

mean_uncertainties = []
mean_errors = []

for i in range(len(energy_bins) - 1):
    mask = (targets_subset[:, 0] >= energy_bins[i]) & (targets_subset[:, 0] < energy_bins[i+1])
    if mask.sum() > 0:
        mean_uncertainties.append(angular_uncertainties[mask].mean())
        mean_errors.append(angular_errors[mask].mean())
    else:
        mean_uncertainties.append(np.nan)
        mean_errors.append(np.nan)

ax.plot(bin_centers, mean_uncertainties, 'o-', label='Mean Uncertainty', linewidth=2, markersize=8)
ax.plot(bin_centers, mean_errors, 's-', label='Mean Error', linewidth=2, markersize=8)
ax.set_xlabel('True Energy (keV)', fontsize=12)
ax.set_ylabel('Degrees', fontsize=12)
ax.set_title('Uncertainty and Error vs Energy', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "energy_stratified.png", dpi=150)
print(f"✓ Saved: {OUTPUT_DIR / 'energy_stratified.png'}")

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save summary
with open(OUTPUT_DIR / "calibration_summary.txt", 'w') as f:
    f.write("="*60 + "\n")
    f.write("SBI CALIBRATION ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Number of tracks analyzed: {NUM_TRACKS}\n")
    f.write(f"Posterior samples per track: {NUM_POSTERIOR_SAMPLES}\n\n")
    
    f.write("COVERAGE TEST:\n")
    f.write("-"*60 + "\n")
    for param_name in param_names:
        f.write(f"{param_name}:\n")
        f.write(f"  68% CI: {coverages[68][param_name]:.1f}% (target: 68%)\n")
        f.write(f"  95% CI: {coverages[95][param_name]:.1f}% (target: 95%)\n\n")
    
    f.write("\nUNCERTAINTY METRICS:\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean angular uncertainty: {np.mean(angular_uncertainties):.2f}°\n")
    f.write(f"Median angular uncertainty: {np.median(angular_uncertainties):.2f}°\n")
    f.write(f"Mean angular error: {np.mean(angular_errors):.2f}°\n")
    f.write(f"Median angular error: {np.median(angular_errors):.2f}°\n\n")
    
    f.write("UNCERTAINTY-ERROR CORRELATION:\n")
    f.write("-"*60 + "\n")
    f.write(f"Correlation: {corr:.3f}\n")
    f.write(f"P-value: {pval:.4f}\n")
    
    if corr > 0.3 and pval < 0.05:
        f.write("Status: ✓ GOOD - Uncertainty is informative\n")
    else:
        f.write("Status: ⚠️  Uncertainty may not be fully informative\n")

print(f"✓ Saved: {OUTPUT_DIR / 'calibration_summary.txt'}")

# Save detailed results
results_df = pd.DataFrame({
    'angular_uncertainty': angular_uncertainties,
    'angular_error': angular_errors,
    'true_energy': targets_subset[:, 0],
})

for idx, param_name in enumerate(param_names):
    results_df[f'{param_name}_uncertainty'] = uncertainties[param_name]

results_df.to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'calibration_results.csv'}")

print("\n" + "="*60)
print("DONE! Check the calibration_analysis/ folder for results.")
print("="*60)