import os
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sbi.inference import NPE
from sbi.diagnostics import calc_misspecification_logprob

# ✅ Reuse existing utilities from your project
from src.utils.data_generator import run_srim_for_energy, parse_collisions
from src.utils.data_utils import preprocess2
def check_x_misspecification(x_train, x_new):
    """
    Train a density estimator ONLY on x_train, then test x_new
    for model misspecification (OOD).
    """
    npe = NPE()  # estimator over x only
    npe = npe.append_simulations(None, x_train).train()
    density_estimator = npe.build_density_estimator()

    p_val, reject_H0 = calc_misspecification_logprob(
        x_train=x_train,
        x_o=x_new,
        trained_density_estimator=density_estimator,
    )
    return p_val, reject_H0




def evaluate_multiple_random_energy_parity(
    posterior,
    srim_dir,
    output_root,
    n_random: int = 20,
    n_ions: int = 100,
    n_post_samples: int = 5000,
    n_bins: int = 6,
    save_csv: str = "random_energy_parity_eval.csv",
):
    """
    Loop over n_random random (E, P) pairs.
    Each run → SRIM simulation, parity flip injection, OOD test, posterior sampling.
    """

    all_results = []

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    # Load training x once for all runs
    x_train_path = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/x_obs_train.pt")
    x_train = torch.load(x_train_path)

    for i in range(n_random):
        print(f"\n===== Random (E, P) Test {i+1}/{n_random} =====")

        try:
            # --- Run single test ---
            res = evaluate_one_random_energy_parity_fixed(
                posterior=posterior,
                srim_dir=srim_dir,
                output_root=output_root,
                x_train=x_train,             # pass preloaded train x
                n_ions=n_ions,
                n_post_samples=n_post_samples,
                n_bins=n_bins,
            )

            # append results (list of dicts)
            all_results.extend(res)

        except Exception as e:
            print(f"[WARN] Test {i+1}/{n_random} failed: {e}")
            continue

    if not all_results:
        raise RuntimeError("No successful runs — check SRIM or preprocessing pipeline.")

    df = pd.DataFrame(all_results)
    save_path = Path(save_csv)
    df.to_csv(save_path, index=False)

    # ---- Aggregates ----
    df_valid = df[df["status"] == "OK"]

    if len(df_valid) > 0:
        mean_energy_err = df_valid["percent_error_abs"].mean()
        std_energy_err = df_valid["percent_error_abs"].std()
        parity_acc = 100.0 * df_valid["parity_correct"].mean()
    else:
        mean_energy_err = std_energy_err = parity_acc = float("nan")

    print("\n===== Summary over all tests =====")
    print(f"Total tracks evaluated: {len(df)}")
    print(f"Valid (not OOD) tracks: {len(df_valid)}")
    print(f"Avg. energy % error: {mean_energy_err:.2f} ± {std_energy_err:.2f}")
    print(f"Parity accuracy: {parity_acc:.2f}%")
    print(f"[DONE] Saved results → {save_path}")

    return df

def evaluate_one_random(posterior, srim_dir, out_dir):
    # Load x normalization
    norm = torch.load(X_NORM_STATS)
    x_mean, x_std = norm["x_mean"], norm["x_std"]

    # Random GT
    E_true = random.uniform(PRIOR_LOW, PRIOR_HIGH)
    P_true = random.choice([0, 1])
    print(f"[INFO] Running SRIM for {E_true:.1f} keV, parity {P_true}")

    # Run SRIM
    run_dir = run_srim_for_energy(E_true, srim_dir, out_dir, number_ions=N_IONS)
    df = parse_collisions(run_dir, E_true)

    # If SRIM produced nothing → always return dict
    if df.empty:
        return {
            "track_id": None,
            "true_energy": E_true,
            "true_parity": P_true,
            "status": "SKIPPED_EMPTY",
        }

    # Inject parity
    if P_true == 0:
        df["x"] *= -1
    df["parity"] = P_true

    # Save local csv
    csv_path = os.path.join(run_dir, f"{E_true:.1f}keV_parity{P_true}.csv")
    df.to_csv(csv_path, index=False)

    # Preprocess
    x_obs_new, _, track_ids, _, _ = preprocess2(csv_path, n_bins=N_BINS)

    # Normalize x using training stats
    x_norm = (x_obs_new - x_mean) / x_std
    x_o = x_norm[0].unsqueeze(0)

    # Sample posterior
    samples = guarded_posterior_sample(
        posterior,
        x_o,
        n_samples=N_posterior_SAMPLES,
        hard_timeout_sec=HARD_TIMEOUT_SEC,
    )

    # If acceptance fails → return safe row
    if samples is None:
        return {
            "track_id": track_ids[0],
            "true_energy": E_true,
            "true_parity": P_true,
            "status": "SKIPPED",
        }

    samples = samples.cpu()

    # ---- RAW predictions (MNPE outputs raw keV + raw parity) ----
    E_pred = samples[:, 0].mean().item()
    E_std  = samples[:, 0].std().item()
    P_pred = int(samples[:, 1].round().mode()[0].item())
    parity_correct = int(P_pred == P_true)

    return {
        "track_id": track_ids[0],
        "true_energy": E_true,
        "true_parity": P_true,
        "pred_energy_mean": E_pred,
        "pred_energy_std": E_std,
        "pred_parity": P_pred,
        "parity_correct": parity_correct,
        "percent_error_abs": 100 * abs(E_pred - E_true) / E_true,
        "status": "OK",
    }
def evaluate_many_random(posterior, energy_stats, srim_dir, out_dir,
                         n_tests=100, n_ions=100, n_samples=1000, n_bins=6):

    results = []
    for i in range(n_tests):
        print(f"[TEST {i+1}/{n_tests}]")
        r = evaluate_one_random(
                posterior, energy_stats, srim_dir, out_dir,
                n_ions=n_ions,
                n_samples=n_samples,
                n_bins=n_bins,
        )
        results.append(r)

    return pd.DataFrame(results)