#!/usr/bin/env python3
"""
Pipeline C: EGNN-SBI (Equivariant Graph Neural Network)
-------------------------------------------------------
1. Preprocesses 3D tracks (center, PCA-align, normalize, pad).
2. Trains SNPE with EGNN embedding network end-to-end.
3. Evaluates angular resolution and energy accuracy.

Usage:
    # Full training (all 197k tracks):
    python -m src.scripts.egnn_3d

    # Quick sanity check (~5k tracks, ~5-10 min on CPU):
    python -m src.scripts.egnn_3d --quick

    # Monitor training in real time (in another PowerShell):
    tensorboard --logdir "C:\\Users\\walsworthlab\\Inverse ML\\sbi-logs"
    # Then open http://localhost:6006
"""

from pathlib import Path
from time import time
import torch
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Ensure 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_utils import preprocess_egnn
from src.models.egnn import EGNNEmbedding
from src.evaluation.eval_mcpe3d import ContinuousEvaluator3D

# SBI Imports
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR      = BASE_DIR / "data/mcpe3d"
TRAIN_CSV     = DATA_DIR / "mcpe_3d_train.csv"
EVAL_CSV      = DATA_DIR / "mcpe_3d_eval_10k.csv"
RESULTS_DIR   = BASE_DIR / "results/egnn_3d"

# Filenames
POSTERIOR_FILE = RESULTS_DIR / "egnn_3d_posterior.pt"
EVAL_RESULTS   = RESULTS_DIR / "egnn_3d_eval_results.csv"
TRAIN_LOG      = RESULTS_DIR / "training_log.csv"

# Hyperparameters
FORCE_RETRAIN  = False
HIDDEN_DIM     = 64
N_LAYERS       = 4
K_NEIGHBORS    = 8
N_HEADS        = 4
D_PROJ         = 64
D_LATENT       = 256
BATCH_SIZE     = 128
LEARNING_RATE  = 5e-4
STOP_AFTER     = 30          # Early stopping patience (epochs)
MAX_EPOCHS     = 200         # Hard cap — never train more than this
# =================================================


def attach_live_monitor(inference, tb_log_dir, csv_path):
    """
    Monkey-patch SBI's _maybe_show_progress to write TensorBoard + CSV
    after EVERY epoch (SBI normally only writes after training completes).

    Args:
        inference: SNPE inference object (after append_simulations)
        tb_log_dir: directory for TensorBoard logs
        csv_path: path for CSV training log
    """
    writer = SummaryWriter(str(tb_log_dir))
    last_written = [0]  # mutable for closure
    t_start = time()

    # Write CSV header
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,epoch_duration_sec,elapsed_min\n")

    # Save original method
    original_show_progress = inference._maybe_show_progress

    def patched_show_progress(show, epoch):
        original_show_progress(show, epoch)

        # Write any new epoch data to TensorBoard + CSV
        val_losses = inference._summary.get("validation_loss", [])
        train_losses = inference._summary.get("training_loss", [])
        durations = inference._summary.get("epoch_durations_sec", [])
        elapsed = (time() - t_start) / 60

        for i in range(last_written[0], len(val_losses)):
            vl = val_losses[i]
            tl = train_losses[i] if i < len(train_losses) else 0
            dur = durations[i] if i < len(durations) else 0

            writer.add_scalar("validation_loss", vl, i)
            writer.add_scalar("training_loss", tl, i)
            writer.add_scalar("epoch_duration_sec", dur, i)
            writer.flush()

            # Append to CSV
            with open(csv_path, 'a') as f:
                f.write(f"{i},{tl:.6f},{vl:.6f},{dur:.1f},{elapsed:.1f}\n")

        last_written[0] = len(val_losses)

    inference._maybe_show_progress = patched_show_progress
    return writer


def main():
    parser = argparse.ArgumentParser(description="EGNN-SBI Pipeline C")
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity check: train on ~5k tracks, ~5-10 min on CPU')
    parser.add_argument('--n-quick', type=int, default=5000,
                        help='Number of tracks to use in quick mode (default: 5000)')
    args = parser.parse_args()

    t_start = time()
    print(f"\n[START] Pipeline C: EGNN-SBI (3D Directional Regression)")
    print(f"        Train Data: {TRAIN_CSV.name}")
    print(f"        Eval Data:  {EVAL_CSV.name}")
    if args.quick:
        print(f"        QUICK MODE: Using only {args.n_quick} tracks")

    # Create Results Directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[DEVICE] Using: {device}")

    # 1. LOAD / TRAIN
    if POSTERIOR_FILE.exists() and not FORCE_RETRAIN and not args.quick:
        print(f"\n[STEP 1] Loading existing posterior -> {POSTERIOR_FILE.name}")
        posterior = torch.load(POSTERIOR_FILE, map_location="cpu")
        train_duration = 0.0
        n_max = posterior.net._neural_net.embedding_net.n_max
    else:
        print("\n[STEP 1] Training New Posterior (EGNN + SNPE)...")
        t_train_start = time()

        # A. Preprocessing (includes kNN precomputation with scipy cKDTree)
        print("[DATA] Preprocessing training data (EGNN pipeline)...")
        x_padded, mask, theta_train, n_max, knn_idx = preprocess_egnn(
            TRAIN_CSV, k_neighbors=K_NEIGHBORS
        )

        # Quick mode: subsample
        if args.quick:
            n_use = min(args.n_quick, len(x_padded))
            indices = torch.randperm(len(x_padded))[:n_use]
            x_padded = x_padded[indices]
            mask = mask[indices]
            theta_train = theta_train[indices]
            knn_idx = knn_idx[indices]
            print(f"[QUICK] Subsampled to {n_use} tracks")

        # B. Flatten for SBI: coords (N_max*3) + knn_idx (N_max*k) -> single flat tensor
        coords_flat = x_padded.view(x_padded.shape[0], -1)          # (N, N_max*3)
        knn_flat = knn_idx.float().view(knn_idx.shape[0], -1)       # (N, N_max*k)
        x_flat = torch.cat([coords_flat, knn_flat], dim=1)           # (N, N_max*(3+k))

        print(f"[DATA] Flattened X shape:  {x_flat.shape} -> (N_tracks, N_max*(3+k))")
        print(f"[DATA] Theta shape:        {theta_train.shape} -> [E, Vx, Vy, Vz]")
        print(f"[DATA] N_max (padding):    {n_max}")
        print(f"[DATA] kNN precomputed:    k={K_NEIGHBORS} (embedded in x_flat)")

        # C. Define Prior
        prior_min = torch.tensor([0.0, -1.0, -1.0, -1.0])
        prior_max = torch.tensor([105.0, 1.0, 1.0, 1.0])
        prior = BoxUniform(low=prior_min, high=prior_max)

        # D. Build EGNN embedding network
        print(f"[MODEL] Building EGNN embedding net:")
        print(f"        Hidden dim:  {HIDDEN_DIM}")
        print(f"        Layers:      {N_LAYERS}")
        print(f"        k neighbors: {K_NEIGHBORS}")
        print(f"        Attn heads:  {N_HEADS}")
        print(f"        d_latent:    {D_LATENT}")

        embedding_net = EGNNEmbedding(
            n_max=n_max,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LAYERS,
            k=K_NEIGHBORS,
            n_heads=N_HEADS,
            d_proj=D_PROJ,
            d_latent=D_LATENT,
        )

        # Count parameters
        n_params = sum(p.numel() for p in embedding_net.parameters())
        print(f"        Parameters:  {n_params:,}")

        # Compile for CPU optimization (fuses ops, optimizes scatter patterns)
        try:
            embedding_net = torch.compile(embedding_net, mode="reduce-overhead")
            print(f"        torch.compile: ENABLED (reduce-overhead mode)")
        except Exception as e:
            print(f"        torch.compile: SKIPPED ({e})")

        # E. Configure SNPE with NSF + EGNN embedding
        density_estimator_build_fn = posterior_nn(
            model="nsf",
            embedding_net=embedding_net,
            hidden_features=128,
            num_transforms=8,
        )

        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fn,
                        device=device)

        # F. Append data & train
        inference.append_simulations(theta_train, x_flat)

        n_batches = len(x_flat) // BATCH_SIZE
        max_ep = MAX_EPOCHS if not args.quick else 30

        print(f"\n[TRAIN] Starting end-to-end training (EGNN + NSF)...")
        print(f"        Batch size:    {BATCH_SIZE}")
        print(f"        Learning rate: {LEARNING_RATE}")
        print(f"        Patience:      {STOP_AFTER} epochs")
        print(f"        Max epochs:    {max_ep}")
        print(f"        Batches/epoch: {n_batches}")
        print(f"        Started at:    {datetime.now().strftime('%H:%M:%S')}")

        # Attach live monitor (TensorBoard + CSV updated every epoch)
        TB_LOG_DIR = RESULTS_DIR / "tb_logs"
        tb_writer = attach_live_monitor(inference, TB_LOG_DIR, TRAIN_LOG)

        print(f"\n[MONITOR] Live TensorBoard + CSV logging enabled!")
        print(f"[MONITOR] In another PowerShell window run:")
        print(f'          tensorboard --logdir "{TB_LOG_DIR}"')
        print(f"          Then open http://localhost:6006")
        print(f"[MONITOR] Or watch CSV:")
        print(f'          Get-Content "{TRAIN_LOG}" -Wait\n')

        density_estimator = inference.train(
            training_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            validation_fraction=0.1,
            stop_after_epochs=STOP_AFTER,
            max_num_epochs=max_ep,
            show_train_summary=True,
        )

        train_duration = (time() - t_train_start) / 60

        # Close TensorBoard writer
        tb_writer.close()

        # G. Build & Save Posterior
        posterior = inference.build_posterior(density_estimator, sample_with="direct")

        save_path = POSTERIOR_FILE if not args.quick else RESULTS_DIR / "egnn_3d_posterior_quick.pt"
        torch.save(posterior, save_path)
        print(f"[SAVE] Posterior saved -> {save_path}")
        print(f"[SAVE] Training took {train_duration:.1f} minutes")

    # 2. EVALUATE (Full Posterior Analysis)
    print("\n[STEP 2] Evaluating (Full Posterior)...")
    evaluator = ContinuousEvaluator3D(posterior, device=device)

    # Quick mode: fewer samples for faster eval
    n_eval_samples = 50 if args.quick else 200
    df_eval_results = evaluator.run_eval(EVAL_CSV, num_samples=n_eval_samples)

    # Save CSV Results (now includes posterior uncertainty columns)
    df_eval_results.to_csv(EVAL_RESULTS, index=False)

    # Generate All Plots (point estimates + posterior quality + calibration)
    print("[PLOTS] Generating Full Evaluation Plots...")
    evaluator.plot_results(df_eval_results, save_dir=RESULTS_DIR)

    # Corner plots for a few example tracks (best, median, worst)
    if len(df_eval_results) > 0:
        sorted_by_angle = df_eval_results['angular_error_deg'].argsort()
        example_tracks = {
            'best': sorted_by_angle.iloc[0],
            'median': sorted_by_angle.iloc[len(sorted_by_angle) // 2],
            'worst': sorted_by_angle.iloc[-1],
        }
        for label, idx in example_tracks.items():
            print(f"[PLOTS] Corner plot for {label} track (idx={idx})...")
            evaluator.plot_corner(idx, df_eval_results,
                                  save_dir=str(RESULTS_DIR / "corners"))

    # 3. REPORT
    print("\n==================== RESULTS (EGNN-SBI) ====================")
    if len(df_eval_results) > 0:
        med_E_err = df_eval_results["energy_error"].median()
        med_A_err = df_eval_results["angular_error_deg"].median()
        flip_rate = (df_eval_results["angular_error_deg"] > 90).mean() * 100
        med_E_std = df_eval_results["energy_std"].median()
        med_cone68 = df_eval_results["angular_cone_68"].median()

        print(f"Median Energy Error:      {med_E_err:.2f} keV")
        print(f"Median Angular Error:     {med_A_err:.2f} deg")
        print(f"Head-Tail Flip Rate:      {flip_rate:.2f} %")
        print(f"Median Energy sigma:      {med_E_std:.2f} keV")
        print(f"Median 68% Cone:          {med_cone68:.2f} deg")

    total_time = (time() - t_start) / 60
    print(f"Training Time:            {train_duration if 'train_duration' in locals() else 0.0:.2f} min")
    print(f"Total Time:               {total_time:.2f} min")
    print(f"Saved Results:            {EVAL_RESULTS}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
