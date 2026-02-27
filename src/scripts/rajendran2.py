import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

MNPE_RESULTS = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/Nov20_mnpe_fixed_eval.csv")
OUTPUT_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results")

# ==========================================================
# COMPUTE TRACK ASYMMETRY (RAJENDRAN METHOD)
# ==========================================================

def compute_asymmetry(track_df):
    """
    Rajendran asymmetry A = N_end / N_begin
    where N_begin = first 1/3 of track length,
          N_end   = last 1/3
    """
    # Assuming columns are 'x', 'y', 'z' or similar
    # Adjust column name if needed
    if 'x' not in track_df.columns:
        return np.nan
        
    s = track_df["x"].values
    if len(s) < 3:
        return np.nan

    s_min, s_max = s.min(), s.max()
    L = s_max - s_min
    if L <= 0:
        return np.nan

    b_end = s_min + L/3
    e_start = s_min + 2*L/3

    N_begin = np.sum((s >= s_min) & (s < b_end))
    N_end   = np.sum((s >= e_start) & (s <= s_max))

    if N_begin == 0:
        return np.nan

    return N_end / N_begin


# ==========================================================
# RAJENDRAN METHOD - EXACT FROM PAPER
# ==========================================================

def rajendran_classify(asymmetry, threshold):
    """
    Classify track as correctly oriented using Rajendran method.
    
    From the paper:
    - If A > threshold T: track can be oriented (predict parity = 1, left-to-right)
    - If A < 1/threshold: track is backward (predict parity = 0, right-to-left)  
    - If 1/T <= A <= T: ambiguous, cannot determine
    
    Returns:
    - predicted_parity: 0 or 1 (or None if ambiguous)
    - confident: True if above threshold, False if ambiguous
    """
    if asymmetry > threshold:
        # High asymmetry, oriented forward
        return 1, True
    elif asymmetry < (1.0 / threshold):
        # Low asymmetry, oriented backward
        return 0, True
    else:
        # Ambiguous
        return None, False


def find_rajendran_threshold(asymmetries, fp_target=0.05):
    """
    Find threshold T using Rajendran's method.
    
    Based on Figure 3 of the paper:
    - Want 5% false positive rate
    - False positive = tracks with A < 1 (backward when they should be forward)
    
    Since all SRIM tracks are generated going +x direction (parity=1),
    A < 1 means the asymmetry suggests backward motion = wrong = false positive
    
    The threshold T is chosen such that:
    1. Tracks with A < 1/T are classified as backward (false positives if true parity is 1)
    2. We want this to be ~5% of all tracks
    3. Efficiency = fraction of tracks with A > T (correctly classified as forward)
    """
    asymmetries = np.array(asymmetries)
    asymmetries = asymmetries[np.isfinite(asymmetries)]
    
    if len(asymmetries) == 0:
        return None
    
    # Method from paper: find T such that 5% of tracks have A < 1/T
    # Rearranging: if 5% have A < 1/T, then 1/T is the 5th percentile
    # Therefore: T = 1 / (5th percentile of A)
    
    percentile_5 = np.percentile(asymmetries, 5)
    
    if percentile_5 <= 0:
        # If 5th percentile is 0 or negative, use a different approach
        # Find threshold that maximizes efficiency while keeping FP <= 5%
        fp_actual = np.mean(asymmetries < 1.0)
        print(f"  Warning: 5th percentile = {percentile_5:.3f}, FP rate (A<1) = {fp_actual:.3f}")
        
        # Use asymmetry = 1 as threshold (anything above 1 is forward)
        return 1.0
    
    threshold = 1.0 / percentile_5
    
    return threshold


# ==========================================================
# MAIN COMPARISON
# ==========================================================

print("Loading MNPE results...")
mnpe_df = pd.read_csv(MNPE_RESULTS)

print(f"Loaded {len(mnpe_df)} tracks")
print(f"Energies: {sorted(mnpe_df['energy_keV'].unique())}")

# Group by energy
results_by_energy = []

for energy in sorted(mnpe_df['energy_keV'].unique()):
    print(f"\n{'='*60}")
    print(f"Processing {energy} keV")
    print(f"{'='*60}")
    
    energy_df = mnpe_df[mnpe_df['energy_keV'] == energy].copy()
    
    # Step 1: Compute asymmetries for all tracks
    asymmetries = []
    valid_indices = []
    
    for idx, row in energy_df.iterrows():
        csv_path = row['csv_path']
        
        # Load raw track data
        try:
            track_df = pd.read_csv(csv_path)
            asym = compute_asymmetry(track_df)
            
            if np.isfinite(asym):
                asymmetries.append(asym)
                valid_indices.append(idx)
        except Exception as e:
            print(f"  Error loading {csv_path}: {e}")
            continue
    
    if len(asymmetries) == 0:
        print(f"  No valid tracks for {energy} keV")
        continue
    
    print(f"  Computed asymmetries for {len(asymmetries)} tracks")
    
    # Step 2: Find Rajendran threshold
    threshold = find_rajendran_threshold(asymmetries)
    
    if threshold is None:
        print(f"  Could not find threshold for {energy} keV")
        continue
    
    print(f"  Rajendran threshold T = {threshold:.3f}")
    
    # Step 3: Apply Rajendran classification
    raj_predictions = []
    raj_confident = []
    
    for asym in asymmetries:
        pred, confident = rajendran_classify(asym, threshold)
        raj_predictions.append(pred)
        raj_confident.append(confident)
    
    # Step 4: Compare with MNPE results
    energy_df_valid = energy_df.loc[valid_indices].copy()
    energy_df_valid['asymmetry'] = asymmetries
    energy_df_valid['raj_pred'] = raj_predictions
    energy_df_valid['raj_confident'] = raj_confident
    
    # Calculate metrics
    # MNPE accuracy
    mnpe_accuracy = energy_df_valid['parity_correct'].mean()
    
    # Rajendran accuracy (only on confident predictions)
    raj_confident_mask = energy_df_valid['raj_confident']
    raj_correct = (energy_df_valid.loc[raj_confident_mask, 'raj_pred'] == 
                   energy_df_valid.loc[raj_confident_mask, 'true_parity'])
    raj_accuracy = raj_correct.mean() if len(raj_correct) > 0 else 0.0
    
    # Rajendran efficiency (fraction that are confident)
    raj_efficiency = raj_confident_mask.mean()
    
    # False positive rate (A < 1 when true_parity = 1)
    true_forward = energy_df_valid['true_parity'] == 1
    fp_rate = (energy_df_valid.loc[true_forward, 'asymmetry'] < 1.0).mean()
    
    print(f"\n  Results:")
    print(f"    ML (MNPE) Accuracy:        {mnpe_accuracy:.3f} ({mnpe_accuracy*100:.1f}%)")
    print(f"    Rajendran Accuracy:        {raj_accuracy:.3f} ({raj_accuracy*100:.1f}%)")
    print(f"    Rajendran Efficiency:      {raj_efficiency:.3f} ({raj_efficiency*100:.1f}%)")
    print(f"    False Positive Rate (A<1): {fp_rate:.3f} ({fp_rate*100:.1f}%)")
    print(f"    Improvement (ML - Raj):    {(mnpe_accuracy - raj_accuracy):.3f} ({(mnpe_accuracy - raj_accuracy)*100:.1f}%)")
    
    results_by_energy.append({
        'energy_keV': energy,
        'n_tracks': len(asymmetries),
        'threshold': threshold,
        'mnpe_accuracy': mnpe_accuracy,
        'raj_accuracy': raj_accuracy,
        'raj_efficiency': raj_efficiency,
        'false_positive_rate': fp_rate,
        'improvement': mnpe_accuracy - raj_accuracy
    })

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(results_by_energy)
output_path = OUTPUT_DIR / "rajendran_vs_ml_comparison.csv"
results_df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(results_df.to_string(index=False))
print(f"\nSaved to: {output_path}")

# ==========================================================
# PLOT COMPARISON
# ==========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
ax1.plot(results_df['energy_keV'], results_df['mnpe_accuracy'] * 100, 
         'o-', label='ML (MNPE)', linewidth=2, markersize=8)
ax1.plot(results_df['energy_keV'], results_df['raj_accuracy'] * 100, 
         's-', label='Rajendran et al.', linewidth=2, markersize=8)
ax1.set_xlabel('Energy (keV)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('ML vs Rajendran Method: Accuracy Comparison', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Improvement
ax2.bar(range(len(results_df)), results_df['improvement'] * 100)
ax2.set_xlabel('Energy (keV)', fontsize=12)
ax2.set_ylabel('Improvement (%)', fontsize=12)
ax2.set_title('ML Improvement over Rajendran Method', fontsize=14)
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels([f"{e:.0f}" for e in results_df['energy_keV']], rotation=45)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = OUTPUT_DIR / "rajendran_vs_ml_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.close()

print("\nDone!")