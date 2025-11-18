#!/usr/bin/env python3

import os
import io
import time
import torch
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import contextlib
import signal

# ============================================================
# CONFIG — CHANGE THESE EASILY
# ============================================================
N_RANDOM_TESTS = 100       # How many random SRIM tracks to evaluate
N_posterior_SAMPLES = 200   # How many samples per posterior evaluation
HARD_TIMEOUT_SEC = 180       # Hard timeout for sampling (seconds)
N_IONS = 200                # Ions per SRIM simulation
N_BINS = 6                  # Histogram bins


RAW_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/nov3srim/centered_tracks.csv")
SRIM_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013")
RESULTS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
X_NORM_STATS   = RESULTS_DIR / "x_norm_stats.pt"


POSTERIOR_FILE = RESULTS_DIR / "trained_posterior_mnpe.pt"



PRIOR_LOW, PRIOR_HIGH = 1.0, 100.0  # keV range
# ============================================================



# ============================================================
# HARD TIMEOUT HELPER
# ============================================================
class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Runs func(*args, **kwargs) but aborts if the timeout triggers.
    Returns None on timeout.
    """

    def handler(signum, frame):
        raise TimeoutException

    # Set up the signal-based timeout
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)

    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        print(f"[TIMEOUT] Exceeded {timeout_sec} sec → skipping.")
        return None
    finally:
        signal.alarm(0)

    return result



# ============================================================
# ACCEPTANCE-GUARD POSTERIOR SAMPLING
# ============================================================
def guarded_posterior_sample(posterior, x, n_samples, hard_timeout_sec):
    """
    Try posterior.sample((n_samples,), x=x).

    - If sbi prints "Only 0.000% proposal samples" → skip
    - If it takes longer than hard_timeout_sec → skip

    Returns:
        Tensor   → samples
        None     → skip
    """

    def run_sample():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            samples = posterior.sample((n_samples,), x=x)
        text = buf.getvalue()

        if "Only 0.000% proposal samples" in text:
            print("[GUARD] 0% acceptance detected → skipping.")
            return None
        return samples

    # Wrap sampling inside the hard timeout logic
    return run_with_timeout(run_sample, hard_timeout_sec)



# ============================================================
# IMPORT PROJECT UTILS
# ============================================================
from src.utils.data_utils import preprocess_mnpe
from src.utils.sbi_runner import (
    make_mnpe_prior,
    make_mnpe_inference,
    normalize_energy,
    train_mnpe_posterior,
)
from src.utils.data_generator import run_srim_for_energy, parse_collisions



# ============================================================
# STEP 1 — PREPROCESS TRAINING DATA
# ============================================================
def preprocess_data():
    print("\n[STEP 1] Preprocessing SRIM data…")

    # Extract raw features
    x_obs, theta, _, _, _ = preprocess_mnpe(RAW_CSV, n_bins=N_BINS)

    print("[TRAIN] x_obs mean per feature:", x_obs.mean(dim=0))
    print("[TRAIN] x_obs std per feature:", x_obs.std(dim=0))

    # NO NORMALIZATION
    return x_obs, theta


# ============================================================
# STEP 2 — TRAIN / LOAD POSTERIOR
# ============================================================
def train_or_load_posterior(x_obs, theta):
    # -----------------------------------------
    # Load if already trained
    # -----------------------------------------
    if POSTERIOR_FILE.exists():
        d = torch.load(POSTERIOR_FILE, map_location="cpu")
        print(f"[STEP 2] Loaded trained posterior → {POSTERIOR_FILE}")
        return d["posterior"], d["energy_stats"]

    print("[STEP 2] Training new MNPE posterior…")

    # -----------------------------------------
    # NO θ NORMALIZATION – work directly in keV
    # -----------------------------------------
    E = theta[:, 0]
    energy_stats = {
        "E_mean": E.mean().item(),
        "E_std":  E.std().item(),
    }
    print(f"[INFO] Energy stats (raw keV): mean={E.mean():.3f}, std={E.std():.3f}")

    # Prior is already defined in raw keV
    prior = make_mnpe_prior(PRIOR_LOW, PRIOR_HIGH)

    inference = make_mnpe_inference(prior)

    # Train MNPE with:
    #   - RAW θ  = [energy_keV, parity]
    #   - RAW x_obs (for now)
    posterior = train_mnpe_posterior(inference, theta, x_obs)

    # Save posterior + stats
    torch.save(
        {"posterior": posterior, "energy_stats": energy_stats},
        POSTERIOR_FILE
    )
    print(f"[INFO] Posterior saved → {POSTERIOR_FILE}")

    return posterior, energy_stats



# ============================================================
# STEP 3 — Evaluate ONE random SRIM track
# ============================================================
def evaluate_one_random(posterior, energy_stats, srim_dir, out_dir):
    # -----------------------------------------
    # 1. Sample true parameters
    # -----------------------------------------
    E_true = random.uniform(PRIOR_LOW, PRIOR_HIGH)
    P_true = random.choice([0, 1])
    print(f"[INFO] Running SRIM for {E_true:.1f} keV, parity {P_true}")

    # -----------------------------------------
    # 2. Generate SRIM track
    # -----------------------------------------
    run_dir = run_srim_for_energy(E_true, srim_dir, out_dir, number_ions=N_IONS)
    df = parse_collisions(run_dir, E_true)
    if df.empty:
        return None

    # Inject parity orientation
    if P_true == 0:
        df["x"] *= -1
    df["parity"] = P_true

    # -----------------------------------------
    # 2.5 CENTER THE TRACK (MATCH TRAINING)
    # -----------------------------------------
    cx = df["x"].mean()
    cy = df["y"].mean()
    cz = df["z"].mean()
    df["x"] -= cx
    df["y"] -= cy
    df["z"] -= cz

    # -----------------------------------------
    # 3. Save centered track for preprocessing
    # -----------------------------------------
    fn = os.path.join(run_dir, f"{E_true:.1f}keV_p{P_true}.csv")
    df.to_csv(fn, index=False)

    # -----------------------------------------
    # 4. Extract features (same as training)
    # -----------------------------------------
    # 1. Extract raw x features
    x_obs_new, _, track_ids, _, _ = preprocess_mnpe(fn, n_bins=N_BINS)

    x_o = x_obs_new[0].unsqueeze(0)


    print("[EVAL] x_obs_new shape:", x_obs_new.shape)
    print("[EVAL] x_obs_new[0]:", x_o[0])

    # -----------------------------------------
    # 5. Sample posterior
    # -----------------------------------------
    samples = guarded_posterior_sample(
        posterior,
        x_o,
        n_samples=N_posterior_SAMPLES,
        hard_timeout_sec=HARD_TIMEOUT_SEC
    )

    if samples is None:
        return {
            "track_id": track_ids[0],
            "true_energy": E_true,
            "true_parity": P_true,
            "status": "SKIPPED",
        }

    samples = samples.cpu()

    # -----------------------------------------
    # 6. ENERGY IS ALREADY IN keV
    # -----------------------------------------
    E_pred = samples[:, 0].mean().item()
    E_std_pred = samples[:, 0].std().item()

    # -----------------------------------------
    # 7. Parity prediction
    # -----------------------------------------
    P_pred = int(samples[:, 1].round().mode()[0].item())
    E_std_pred = samples[:, 0].std().item()

    P_pred = int(samples[:, 1].round().mode()[0].item())
    parity_correct = int(P_pred == P_true)

    return {
        "track_id": track_ids[0],
        "true_energy": E_true,
        "true_parity": P_true,
        "pred_energy_mean": E_pred,
        "pred_energy_std": E_std_pred,
        "pred_parity": P_pred,
        "parity_correct": parity_correct,
        "percent_error_abs": 100 * abs(E_pred - E_true) / E_true,
        "status": "OK",
    }
# ============================================================
# STEP 4 — Evaluate many random SRIM runs
# ============================================================
def evaluate_random_srims(posterior, energy_stats):
    out_dir = RESULTS_DIR / "random_eval_outputs_mnpe2"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n[STEP 3] Running {N_RANDOM_TESTS} random SRIM tests…")

    results = []
    t0 = time.time()

    for i in range(N_RANDOM_TESTS):
        print(f"\n[TEST {i+1}/{N_RANDOM_TESTS}]")
        t_start = time.time()

        try:
            r = evaluate_one_random(
                posterior=posterior,
                energy_stats=energy_stats,
                srim_dir=SRIM_DIR,
                out_dir=out_dir,
            )
            results.append(r)
        except Exception as e:
            print(f"[WARN] Test {i+1} error: {e}")

        print(f"[INFO] Track time: {time.time() - t_start:.2f} sec")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "random_eval_results_mnpe2.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Results saved → {out_csv}")

    # Summary
    total = len(df)
    skipped = (df["status"] == "SKIPPED").sum()
    pct_skip = 100 * skipped / total if total > 0 else 0.0

    print("\n========== SUMMARY ==========")
    print(f"Total tracks:     {total}")
    print(f"Skipped:          {skipped} ({pct_skip:.1f}%)")
    print("=============================")

    print(f"\n[TOTAL] Random-eval runtime: {(time.time()-t0)/60:.2f} min")

    return df

# ============================================================
# MASTER PIPELINE
# ============================================================
def run_pipeline():
    print(f"[INIT] Starting pipeline @ {datetime.now()}")

    # STEP 1 — preprocess SRIM training data
    x_norm, theta = preprocess_data()

    # STEP 2 — train or load posterior
    # must return BOTH posterior and energy_stats
    posterior, energy_stats = train_or_load_posterior(x_norm, theta)

    # STEP 3 — evaluate random SRIM tracks
    evaluate_random_srims(posterior, energy_stats)

    print(f"[DONE] Finished @ {datetime.now()}")




if __name__ == "__main__":
    run_pipeline()