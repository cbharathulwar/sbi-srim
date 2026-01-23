import torch
import pandas as pd
import numpy as np
from scipy.stats import skew
from pathlib import Path

# Import math helpers
from src.utils.data_utils import (
    infer_relative_bin_edges,
    relative_bin_fractions_from_events,
    compute_centered_track_asymmetry,
    compute_centered_nn_asymmetry
)

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/siimpl_800keV_train.csv")
EVAL_CSV  = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nv_800keV2_eval.csv") 
POSTERIOR_FILE = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/posterior5_siimpl_800keV.pt")
OUTPUT_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/results_NV_FINAL2_112.csv")
N_BINS = 6

def custom_preprocess_NO_FILTER(data_path, n_bins=6):
    """
    Standard preprocessing but with NO length filter.
    """
    df = pd.read_csv(data_path)
    
    # Standard Cleanup
    df.columns = [c.strip() for c in df.columns]
    if "energy_keV" not in df.columns and "energy" in df.columns: 
        df["energy_keV"] = df["energy"]/1e3
    if "parity" not in df.columns: 
        df["parity"] = 0
    
    x_obs, theta_list, tids = [], [], []
    r_edges = infer_relative_bin_edges(n_bins)
    
    # Group by track
    # sort=False preserves order
    grouped = df.groupby(["energy_keV", "ion_number", "parity"], sort=False)
    
    print(f"   [CUSTOM] Found {len(grouped)} unique tracks/groups.")
    
    for (E, ion, par), g in grouped:
        raw_z = g["z"].values.astype(float)
        x_coords = g["x"].values.astype(float)
        
        # --- NO FILTER ---
        # Processing even if len < 5
        
        # Centering
        if len(raw_z) > 0:
            z = raw_z - np.mean(raw_z)
        else:
            z = raw_z # Empty

        # Physics Features 
        # Handle empty/sparse cases safely to avoid crashing
        abs_z = np.abs(z)
        mean_depth = np.mean(abs_z) if len(z) > 0 else 0
        max_depth = np.max(abs_z) if len(z) > 0 else 0
        
        if len(z) > 0:
            depth_coord = z - np.min(z)
            norm = np.percentile(depth_coord, 95)
        else:
            depth_coord = []
            norm = 1.0
            
        r_fracs = relative_bin_fractions_from_events(depth_coord, norm, r_edges)
        
        asym_c = compute_centered_track_asymmetry(z)
        asym_nn = compute_centered_nn_asymmetry(z, x_coords)
        
        try: skew_z = float(skew(z))
        except: skew_z = 0.0
            
        var_up = np.var(z[z < 0]) if np.any(z < 0) else 0.0
        var_down = np.var(z[z > 0]) if np.any(z > 0) else 0.0
        var_diff = np.log1p(var_down) - np.log1p(var_up)

        row = [mean_depth, max_depth, len(z), *r_fracs, asym_c, asym_nn, skew_z, var_diff]
        
        x_obs.append(row)
        theta_list.append([float(E), float(par)])
        tids.append(f"E{E:.0f}_ion{int(ion)}")

    x_t = torch.tensor(x_obs, dtype=torch.float32)
    theta_t = torch.tensor(theta_list, dtype=torch.float32)
    
    return x_t, theta_t, tids

def run_forced_eval():
    print("[INIT] Starting FORCED Evaluation (OFFSET ID FIX)...")

    # 1. Merge for Normalization 
    print(f"[STEP 1] Merging Train + Eval...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_eval = pd.read_csv(EVAL_CSV)
    
    # --- THE FIX: OFFSET IDS ---
    # We add 1,000,000 to Eval IDs so they NEVER collide with Train IDs (0-4000)
    print("   Applying ID Offset (+1,000,000) to Eval tracks to prevent merging...")
    df_eval['ion_number'] = df_eval['ion_number'] + 1000000
    
    # Combine
    df_combined = pd.concat([df_train, df_eval], ignore_index=True)
    temp_csv = Path("temp_force_eval.csv")
    df_combined.to_csv(temp_csv, index=False)
    
    # 2. Run CUSTOM Preprocessing
    # Should now find 4000 (Train) + 112 (Eval) = 4112 tracks
    x_combined, theta_combined, track_ids = custom_preprocess_NO_FILTER(temp_csv, n_bins=N_BINS)
    
    # 3. Apply Normalization
    mean = x_combined.mean(dim=0)
    std = x_combined.std(dim=0) + 1e-6
    x_norm = (x_combined - mean) / std
    
    # 4. Extract Eval Tracks by ID Range
    print("[STEP 2] Extracting Eval tracks...")
    valid_indices = []
    
    for i, tid in enumerate(track_ids):
        try:
            # tid format: "E800_ion1000042"
            ion_num = int(tid.split('ion')[-1])
            
            # Simple check: If ID > 900,000, it's one of ours
            if ion_num >= 900000:
                valid_indices.append(i)
        except: continue
            
    x_eval = x_norm[valid_indices]
    theta_eval = theta_combined[valid_indices]
    
    # Remove the offset from the ID for the report
    ids_eval = [track_ids[i].replace("ion100", "ion00").replace("ion10", "ion") for i in valid_indices]
    
    print(f"   ✅ Extracted {len(x_eval)} tracks (Should be exactly 112).")

    # 5. Run Inference
    print("[STEP 3] Predicting...")
    posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
    
    results = []
    for i in range(len(x_eval)):
        x_obs = x_eval[i].unsqueeze(0)
        samples = posterior.sample((100,), x=x_obs, show_progress_bars=False)
        pred_prob = samples[:, -1].mean().item()
        
        results.append({
            'track_id': ids_eval[i],
            'true_parity': int(theta_eval[i, -1].item()),
            'pred_parity': 1 if pred_prob > 0.5 else 0,
            'is_correct': (1 if pred_prob > 0.5 else 0) == int(theta_eval[i, -1].item())
        })

    # 6. Save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    
    acc = pd.DataFrame(results)['is_correct'].mean()
    print(f"\n[DONE] Saved {len(results)} predictions to: {OUTPUT_CSV}")
    print(f"       Accuracy on ALL 112 tracks: {acc:.2%}")
    
    # Cleanup
    if temp_csv.exists(): temp_csv.unlink()

if __name__ == "__main__":
    run_forced_eval()