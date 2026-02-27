import numpy as np
import pandas as pd

# ==========================================
# 1. Core Physics Feature Extraction
# ==========================================
def calculate_raj_features(track_df):
    """
    Analyzes a single track to determine the Rajendran Asymmetry.
    Returns the guessed direction and the asymmetry ratio.
    """
    x_coords = track_df['x'].values
    
    if len(x_coords) == 0:
        return 0, 1.0 # Default fallback
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    span = x_max - x_min
    
    # If the track is just a single point or has no span
    if span == 0:
        return 0, 1.0 
    
    third = span / 3.0
    left_bound = x_min + third
    right_bound = x_max - third
    
    # Count vacancies in the left and right thirds
    n_left = np.sum(x_coords <= left_bound)
    n_right = np.sum(x_coords >= right_bound)
    
    # Avoid division by zero by adding a tiny epsilon
    n_left = max(1, n_left)
    n_right = max(1, n_right)
    
    # The heuristic: Particle travels TOWARD the denser end
    if n_right > n_left:
        pred_parity = 1       # Guesses it traveled +x
        asymmetry = n_right / n_left
    else:
        pred_parity = 0       # Guesses it traveled -x
        asymmetry = n_left / n_right
        
    return pred_parity, asymmetry

def batch_process_raj_features(eval_df):
    """
    Processes a full dataset (e.g., raj_calib.csv or raj_eval.csv)
    and appends the Rajendran predictions and asymmetries.
    """
    print(f"[INFO] Calculating Rajendran features for {len(eval_df['ion_number'].unique())} tracks...")
    
    results = []
    # Group by energy, ion, and the true parity
    grouped = eval_df.groupby(['energy_keV', 'ion_number', 'parity'])
    
    for (energy, ion, true_parity), track_data in grouped:
        pred_parity, asymmetry = calculate_raj_features(track_data)
        
        results.append({
            'energy_keV': energy,
            'ion_number': ion,
            'true_parity': true_parity,
            'pred_parity': pred_parity,
            'asymmetry': asymmetry,
            'is_correct': int(pred_parity == true_parity)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 2. Threshold Calibration (Finding the 5% FPR)
# ==========================================
def calibrate_raj_thresholds(calib_results_df, target_accuracy=0.95):
    """
    Finds the exact asymmetry threshold (T) for each energy bin 
    required to hit a target accuracy (default 95%, i.e., 5% FPR).
    """
    print(f"[INFO] Calibrating thresholds for target accuracy: {target_accuracy*100}%")
    
    energies = sorted(calib_results_df['energy_keV'].unique())
    threshold_map = {}
    
    for energy in energies:
        energy_df = calib_results_df[calib_results_df['energy_keV'] == energy]
        best_t = 1.0
        
        # Sweep threshold T from 1.0 to 10.0
        for t in np.linspace(1.0, 10.0, 500):
            kept_tracks = energy_df[energy_df['asymmetry'] >= t]
            
            if len(kept_tracks) == 0:
                continue
                
            accuracy = kept_tracks['is_correct'].mean()
            
            # As soon as we hit the target accuracy, lock in the threshold
            if accuracy >= target_accuracy:
                best_t = t
                break
                
        threshold_map[energy] = best_t
        print(f"  -> {energy} keV: Threshold T = {best_t:.3f}")
        
    return threshold_map

# ==========================================
# 3. Curve Generation (Efficiency vs Accuracy)
# ==========================================
# Add total_events as the 4th parameter
def generate_raj_curve_data(eval_results_df, energy_min, energy_max, total_events):
    """
    Generates the Efficiency vs. Accuracy plotting points for Rajendran.
    """
    bin_df = eval_results_df[
        (eval_results_df['energy_keV'] >= energy_min) & 
        (eval_results_df['energy_keV'] < energy_max)
    ]
    
    curve_points = []
    
    # Find the maximum asymmetry in this bin, and sweep up to it
    max_t = bin_df['asymmetry'].max()
    if pd.isna(max_t): return pd.DataFrame()
    
    # Sweep from 1.0 to whatever the maximum is
    for t in np.linspace(1.0, max_t, 100):
        kept_tracks = bin_df[bin_df['asymmetry'] >= t]
        
        if len(kept_tracks) == 0: continue
            
        # TRUE efficiency penalty to avoid survivorship bias!
        efficiency = len(kept_tracks) / total_events 
        accuracy = kept_tracks['is_correct'].mean()
        
        curve_points.append({
            "threshold": t,
            "efficiency": efficiency,
            "accuracy": accuracy
        })
        
    return pd.DataFrame(curve_points)