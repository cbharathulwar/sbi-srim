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
    Automatically detects if the model is on CPU or GPU.
    """

    MAX_TIMEOUTS_PER_TRACK = 2      
    MAX_GLOBAL_TIMEOUTS   = 50      
    global_timeouts = 0
    skipped_tracks = 0

    # =======================================================
    # 0. SMART DEVICE DETECTION (The Fix)
    # =======================================================
    # Instead of forcing MPS, we ask the model where it lives.
    try:
        # Peek at the first parameter of the neural network
        # posterior.potential_fn is the underlying density estimator in SBI
        sample_param = next(posterior.potential_fn.parameters())
        device = sample_param.device
    except Exception as e:
        # Fallback to CPU if we can't find parameters
        print(f"[WARN] Could not detect model device ({e}). Defaulting to CPU.")
        device = torch.device("cpu")
    
    print(f"[EVAL] Model is on device: {device}. Moving input data to match.")

    # =======================================================
    # 1. LOAD & CLEAN DATA
    # =======================================================
    df = pd.read_csv(eval_csv)
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize column names
    rename_map = {
        "ion #": "ion_number", "ion": "ion_number",
        "energy": "energy_keV", "x (ang)": "x", "y (ang)": "y", "z (ang)": "z",
        "x_ang": "x", "y_ang": "y", "z_ang": "z"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()] 

    # Validate
    if "energy_keV" not in df.columns:
        raise ValueError("No 'energy_keV' column found.")
    
    # Ensure numeric
    for c in ["x", "y", "z", "energy_keV", "ion_number", "angle_class"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=["x", "energy_keV"])

    # =======================================================
    # 2. GROUP BY TRACK
    # =======================================================
    groups = df.groupby(["energy_keV", "ion_number", "angle_class"], sort=False)
    results = []

    print(f"[EVAL] Processing {len(groups)} tracks...")
    if max_points:
        print(f"[EVAL] Mode: Embedding Network (Padding to {max_points})")
    else:
        print(f"[EVAL] Mode: Summary Features (26D+)")

    for (E, ion, cls), g in groups:
        true_energy = float(E)
        true_class  = int(cls)
        
        # ---------------------------------------------------
        # A. PREPARE INPUT
        # ---------------------------------------------------
        if max_points is not None:
            # --- EMBEDDING NET PATH ---
            xyz = g[["x", "y", "z"]].values.astype(np.float32)
            x_input = create_embedding_dataset([xyz], max_points=max_points)
        else:
            # --- SUMMARY FEATURES PATH ---
            tmp = Path("tmp_eval_track.csv")
            g.to_csv(tmp, index=False)
            x_obs, _, _, _, _ = preprocess_mdpe(tmp)
            tmp.unlink(missing_ok=True)

            if len(x_obs) != 1: continue
            x_input = x_obs[0].unsqueeze(0)

        # --- CRITICAL: Move input to the SAME device as the model ---
        x_input = x_input.to(device)

        # ---------------------------------------------------
        # B. SAMPLE POSTERIOR
        # ---------------------------------------------------
        samples = None
        attempts = 0

        while samples is None and attempts < MAX_TIMEOUTS_PER_TRACK:
            samples = guarded_posterior_sample(
                posterior,
                x_input, 
                n_samples=n_post_samples,
                hard_timeout_sec=30,
            )
            if samples is None:
                attempts += 1
                global_timeouts += 1

        if samples is None:
            skipped_tracks += 1
            if global_timeouts >= MAX_GLOBAL_TIMEOUTS:
                print("\n🚨 TOO MANY TIMEOUTS — EXITING EVAL EARLY 🚨\n")
                break
            continue
        
        # Move back to CPU for storage
        samples = samples.cpu()

        # ---------------------------------------------------
        # C. STORE PREDICTIONS
        # ---------------------------------------------------
        pred_energy = samples[:,0].mean().item()
        pred_class  = samples[:,1].round().clamp(0,7).mode()[0].item()

        results.append({
            "energy_true": true_energy,
            "energy_pred_mean": pred_energy,
            "energy_error_pct": 100 * abs(pred_energy - true_energy) / true_energy,
            "class_true": true_class,
            "class_pred": int(pred_class),
            "class_correct": int(pred_class == true_class),
            "ion_number": int(ion),
            "n_samples": len(samples),
        })

    # =======================================================
    # 3. SAVE RESULTS
    # =======================================================
    df_out = pd.DataFrame(results)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print("\n================= MDPE Evaluation Complete =================")
    print(f"Saved → {output_csv}")
    print(f"Tracks evaluated:      {len(df_out)}")
    print(f"Tracks skipped:        {skipped_tracks}")
    if len(df_out) > 0:
        print(f"Direction accuracy:    {df_out['class_correct'].mean()*100:.1f}%")
        print(f"Median energy error:   {df_out['energy_error_pct'].median():.2f}%")
    print("============================================================\n")

    return df_out