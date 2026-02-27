import torch
import pandas as pd
import numpy as np
from scipy.stats import skew
from pathlib import Path

# Import your existing SBI helpers
from src.utils.sbi_runner import (
    make_mnpe_prior,
    make_mnpe_inference,
    train_mnpe_posterior,
)
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
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results")
POSTERIOR_FILE = RESULTS_DIR / "posterior_SIIMPL_TRAINED.pt"
OUTPUT_CSV = RESULTS_DIR / "results_SIIMPL_TRAINED.csv"

PRIOR_LOW, PRIOR_HIGH = 100.0, 1000.0
N_BINS = 6

def compute_rajendran_asymmetry(z):
    z = np.asarray(z, float)
    if z.size < 2: return 0.0
    z_min, z_max = np.min(z), np.max(z)
    one_third = z_min + (z_max - z_min) / 3.0
    two_thirds = z_min + 2.0 * (z_max - z_min) / 3.0
    n_first = np.sum(z <= one_third)
    n_last = np.sum(z >= two_thirds)
    return n_last / (n_first + 1e-12)

def custom_preprocess_df(df, n_bins=6):
    """Processes a DataFrame already in memory."""
    df.columns = [str(c).strip() for c in df.columns]
    if "energy_keV" not in df.columns and "energy" in df.columns: df["energy_keV"] = df["energy"]/1e3
    if "parity" not in df.columns: df["parity"] = 0
    
    x_obs, theta_list, tids = [], [], []
    r_edges = infer_relative_bin_edges(n_bins)
    
    # Use ion_number for the ID
    grouped = df.groupby(["energy_keV", "ion_number", "parity"], sort=False)
    
    for (E, ion, par), g in grouped:
        z = g["z"].values.astype(float)
        z = z - np.mean(z) if len(z) > 0 else z # Centering
        
        # Geometry Features
        abs_z = np.abs(z)
        mean_depth = np.mean(abs_z) if len(z) > 0 else 0
        r_fracs = relative_bin_fractions_from_events(z - np.min(z), np.percentile(z-np.min(z), 95), r_edges) if len(z) > 0 else np.zeros(n_bins)
        
        # Asymmetry Metrics
        asym_c = compute_centered_track_asymmetry(z)
        asym_raj = compute_rajendran_asymmetry(z) # Daniel's fix 
        
        try: skew_z = float(skew(z))
        except: skew_z = 0.0
            
        var_up = np.var(z[z < 0]) if np.any(z < 0) else 0.0
        var_down = np.var(z[z > 0]) if np.any(z > 0) else 0.0
        var_diff = np.log1p(var_down) - np.log1p(var_up)

        row = [mean_depth, np.max(abs_z) if len(z)>0 else 0, len(z), *r_fracs, asym_c, asym_raj, skew_z, var_diff]
        x_obs.append(row)
        theta_list.append([float(E), float(par)])
        tids.append(f"E{E:.0f}_ion{int(ion)}")

    return torch.tensor(x_obs, dtype=torch.float32), torch.tensor(theta_list, dtype=torch.float32), tids

def run_training_pipeline():
    print("[INIT] Starting Training with Normalization Fix...")

    # 1. THE NORMALIZATION FIX: Load and Merge
    print("   Merging Train + Eval for Global Scaling...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_eval = pd.read_csv(EVAL_CSV)
    df_eval['ion_number'] += 1000000 # Offset Eval IDs
    
    df_combined = pd.concat([df_train, df_eval], ignore_index=True)
    
    # 2. Extract All Features
    x_all, theta_all, tids_all = custom_preprocess_df(df_combined, n_bins=N_BINS)
    
    # 3. Apply Global Normalization
    mean, std = x_all.mean(dim=0), x_all.std(dim=0) + 1e-6
    x_norm = (x_all - mean) / std
    
    # 4. Split back into Train/Eval using the ID Offset
    train_mask = [i for i, tid in enumerate(tids_all) if int(tid.split('ion')[-1]) < 900000]
    eval_mask = [i for i, tid in enumerate(tids_all) if int(tid.split('ion')[-1]) >= 900000]
    
    x_train, theta_train = x_norm[train_mask], theta_all[train_mask]
    x_eval, theta_eval = x_norm[eval_mask], theta_all[eval_mask]
    
    # 5. ACTUAL TRAINING [cite: 14]
    print(f"[STEP 2] Training SBI model on {len(x_train)} tracks...")
    # Jitter energy so SBI treats it as continuous
    theta_train[:, 0] += torch.randn_like(theta_train[:, 0]) * 0.1
    
    prior = make_mnpe_prior(PRIOR_LOW, PRIOR_HIGH)
    inference = make_mnpe_inference(prior)
    posterior = train_mnpe_posterior(inference, theta_train, x_train)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(posterior, POSTERIOR_FILE)

    # 6. EVALUATION
# STEP 3: EVALUATION
    # Identify the device (matching what you used for training)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[STEP 3] Evaluating on {len(x_eval)} tracks using device: {device}")
    
    # Ensure the posterior's internal model is on the right device
    posterior.to(device) 

    results = []
    for i in range(len(x_eval)):
        # MOVE the specific observation 'x' to the device
        x_obs = x_eval[i].unsqueeze(0).to(device)
        
        # Sampling now happens entirely on the GPU/MPS
        samples = posterior.sample((100,), x=x_obs, show_progress_bars=False)
        pred_prob = samples[:, -1].mean().item()
        
        results.append({
            'track_id': tids_all[eval_mask[i]].replace("ion100", "ion0"),
            'true_parity': int(theta_eval[i, -1].item()),
            'pred_parity': 1 if pred_prob > 0.5 else 0,
            'is_correct': (1 if pred_prob > 0.5 else 0) == int(theta_eval[i, -1].item())
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Accuracy: {pd.DataFrame(results)['is_correct'].mean():.2%}")

if __name__ == "__main__":
    run_training_pipeline()