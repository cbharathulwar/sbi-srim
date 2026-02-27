import torch
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Custom imports
from src.utils.data_utils import preprocess_mdpe, create_embedding_dataset
from src.utils.sbi_runner import guarded_posterior_sample

def evaluate_from_saved_tracks_mdpe(
    posterior,
    eval_csv,
    output_csv,
    max_points=None,
    n_post_samples=500,
):
    """
    Evaluates the posterior on a test set.
    OPTIMIZED: Pre-calculates features to avoid Disk I/O loop.
    """
    MAX_TIMEOUTS_PER_TRACK = 2      
    MAX_GLOBAL_TIMEOUTS   = 50      
    global_timeouts = 0
    skipped_tracks = 0

    # 0. SMART DEVICE DETECTION
    try:
        sample_param = next(posterior.potential_fn.parameters())
        device = sample_param.device
    except Exception as e:
        print(f"[WARN] Could not detect model device ({e}). Defaulting to CPU.")
        device = torch.device("cpu")
    
    print(f"[EVAL] Model is on device: {device}.")

    results = []

    # =======================================================
    # OPTIMIZATION: BRANCHING LOGIC
    # =======================================================
    
    # CASE A: SUMMARY FEATURES (Pipeline A) - THE FAST WAY
    if max_points is None:
        print(f"[EVAL] Mode: Summary Features. Pre-calculating ALL 26D vectors...")
        
        # 1. Run preprocess ONCE for the whole file
        # This returns tensors for all 5000 tracks instantly. No temp files.
        x_all, theta_all, _, _, _ = preprocess_mdpe(eval_csv)
        
        print(f"[EVAL] processing {len(x_all)} tracks from memory...")

        # 2. Loop through the TENSORS directly
        for i in range(len(x_all)):
            # Get data for track 'i'
            x_input = x_all[i].unsqueeze(0).to(device) # Shape (1, 26)
            true_energy = theta_all[i, 0].item()
            true_class  = theta_all[i, 1].item()
            
            # --- SAMPLING BLOCK ---
            samples = None
            attempts = 0
            while samples is None and attempts < MAX_TIMEOUTS_PER_TRACK:
                samples = guarded_posterior_sample(
                    posterior, x_input, n_samples=n_post_samples, hard_timeout_sec=30
                )
                if samples is None:
                    attempts += 1
                    global_timeouts += 1

            if samples is None:
                skipped_tracks += 1
                if global_timeouts >= MAX_GLOBAL_TIMEOUTS: break
                continue
            
            samples = samples.cpu()
            
            # Store Result
            pred_energy = samples[:,0].mean().item()
            pred_class  = samples[:,1].round().clamp(0,7).mode()[0].item()

            results.append({
                "energy_true": true_energy,
                "energy_pred_mean": pred_energy,
                "energy_error_pct": 100 * abs(pred_energy - true_energy) / true_energy,
                "class_true": int(true_class),
                "class_pred": int(pred_class),
                "class_correct": int(pred_class == int(true_class)),
                "n_samples": len(samples),
            })

    # CASE B: EMBEDDING NET (Pipeline B) - STANDARD WAY
    else:
        print(f"[EVAL] Mode: Embedding Network (Padding to {max_points})")
        # Load CSV manually for PointNet handling
        df = pd.read_csv(eval_csv)
        # ... [Standard cleaning code for columns] ...
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {"energy": "energy_keV", "x (ang)": "x", "y (ang)": "y", "z (ang)": "z"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.dropna(subset=["x", "energy_keV"])

        groups = df.groupby(["energy_keV", "ion_number", "angle_class"], sort=False)
        
        for (E, ion, cls), g in groups:
            xyz = g[["x", "y", "z"]].values.astype(np.float32)
            x_input = create_embedding_dataset([xyz], max_points=max_points).to(device)
            
            # --- SAMPLING BLOCK ---
            samples = None
            attempts = 0
            while samples is None and attempts < MAX_TIMEOUTS_PER_TRACK:
                samples = guarded_posterior_sample(
                    posterior, x_input, n_samples=n_post_samples, hard_timeout_sec=30
                )
                if samples is None:
                    attempts += 1
                    global_timeouts += 1
            
            if samples is None: 
                skipped_tracks += 1
                if global_timeouts >= MAX_GLOBAL_TIMEOUTS: break
                continue
                
            samples = samples.cpu()
            
            # Store Result
            pred_energy = samples[:,0].mean().item()
            pred_class  = samples[:,1].round().clamp(0,7).mode()[0].item()

            results.append({
                "energy_true": float(E),
                "energy_pred_mean": pred_energy,
                "energy_error_pct": 100 * abs(pred_energy - float(E)) / float(E),
                "class_true": int(cls),
                "class_pred": int(pred_class),
                "class_correct": int(pred_class == int(cls)),
                "ion_number": int(ion), # Only available in this mode
                "n_samples": len(samples),
            })

    # =======================================================
    # 3. SAVE RESULTS
    # =======================================================
    df_out = pd.DataFrame(results)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"\n[DONE] Saved -> {output_csv} | Acc: {df_out['class_correct'].mean()*100:.1f}%")
    return df_out