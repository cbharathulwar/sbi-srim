#!/usr/bin/env python3
"""
GVP-EGNN Unified Training Script
=================================
Trains the GVP-enhanced EGNN backbone with three simultaneous losses:
  1. NSF flow loss (SBI posterior estimation)
  2. vMF direction loss (auxiliary, axis-aware)
  3. Gaussian energy loss (auxiliary, heteroscedastic)

All three losses backprop through the shared backbone in a single backward pass.
The auxiliary losses provide direct supervised signal that scaffolds the flow
training, especially in early epochs.

Usage:
    # Smoke test (~5 epochs, T4 GPU):
    python -m src.scripts.train_gvp_egnn --smoke

    # Full training (A100):
    python -m src.scripts.train_gvp_egnn

    # Resume from checkpoint:
    python -m src.scripts.train_gvp_egnn --resume
"""

import argparse
import gc
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Ensure 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_utils import preprocess_egnn
from src.models.egnn import EGNNEmbedding, CachedEGNNEmbedding, DirectionHead, EnergyHead
from src.models.vmf_loss import axis_aware_vmf_nll, gaussian_nll

# SBI Imports (only for building the flow architecture)
from sbi.neural_nets import posterior_nn


# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR      = BASE_DIR / "data/mcpe3d"
TRAIN_CSV     = DATA_DIR / "mcpe_3d_train.csv"
EVAL_CSV      = DATA_DIR / "mcpe_3d_eval.csv"
RESULTS_DIR   = BASE_DIR / "results/gvp_egnn"

# Output files
CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.pt"
BEST_CKPT_FILE   = RESULTS_DIR / "best_checkpoint.pt"
TRAIN_LOG        = RESULTS_DIR / "training_log.csv"
TB_LOG_DIR       = RESULTS_DIR / "tb_logs"

# Architecture hyperparameters (GVP-EGNN)
HIDDEN_DIM     = 96       # up from 64
N_LAYERS       = 6        # up from 4
K_NEIGHBORS    = 16       # same
N_HEADS        = 8        # same
D_PROJ         = 48       # up from 32
D_LATENT       = 384      # up from 256
V_DIM          = 8        # GVP vector channels
NUM_TRANSFORMS = 8        # NSF coupling layers (was 12, reduced for speed)
NSF_HIDDEN     = 192      # NSF hidden features (up from 128)

# Training hyperparameters
BATCH_SIZE     = 256
LR_MAX         = 3e-4
LR_MIN         = 1e-5
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 5
MAX_EPOCHS     = 200
PATIENCE       = 40       # early stopping
VAL_FRACTION   = 0.1
GRAD_CLIP      = 1.0
MAX_TRAIN_TRACKS = 100_000

# Auxiliary loss weights (decay schedule)
# NOTE: energy loss is O(1000) at init (sigma~1, residuals~50),
# so beta must be small to avoid destabilizing the NSF flow.
ALPHA_START    = 0.5      # vMF direction loss weight (start)
ALPHA_END      = 0.05     # vMF direction loss weight (end)
BETA_START     = 0.01     # Gaussian energy loss weight (start)
BETA_END       = 0.001    # Gaussian energy loss weight (end)

# Per-sample loss clamp: prevents outlier samples from destabilizing training
LOSS_CLAMP     = 1000.0   # clamp per-sample NLL before averaging
# =================================================


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr_with_warmup(epoch, warmup_epochs, max_epochs, lr_max, lr_min):
    """Compute learning rate with linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return lr_min + (lr_max - lr_min) * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def aux_weight_schedule(epoch, max_epochs, start, end):
    """Linear decay of auxiliary loss weights."""
    progress = min(epoch / max(max_epochs - 1, 1), 1.0)
    return start + (end - start) * progress


def build_flow(embedding_net, n_max, k, d_latent, device):
    """
    Build the NSF flow using SBI's posterior_nn factory.

    Returns the instantiated NFlowsFlow object (ready for .loss() calls).
    """
    # Wrap embedding in CachedEGNNEmbedding for aux head access
    cached_embedding = CachedEGNNEmbedding(embedding_net)

    # Create the flow builder function
    # z_score_x="none": EGNN already normalizes coords to [-1,1]
    # z_score_theta="independent": let SBI standardize theta (E is 0-105,
    #   directions are -1 to 1 — the scale mismatch causes NaN without z-scoring)
    build_fn = posterior_nn(
        model="nsf",
        embedding_net=cached_embedding,
        hidden_features=NSF_HIDDEN,
        num_transforms=NUM_TRANSFORMS,
        z_score_x="none",
        z_score_theta="independent",
    )

    # Instantiate the flow by calling build_fn with REPRESENTATIVE sample data
    # SBI uses this to compute z-scoring statistics for theta
    # Must reflect actual data distribution, not random noise
    dummy_theta = torch.zeros(100, 4)
    dummy_theta[:, 0] = torch.rand(100) * 105.0          # Energy: [0, 105]
    dummy_theta[:, 1:4] = F.normalize(torch.randn(100, 3), dim=-1)  # unit dirs
    dummy_x = torch.randn(100, n_max * (3 + k))

    flow = build_fn(dummy_theta, dummy_x)
    flow = flow.to(device)

    return flow, cached_embedding


def prepare_data(args, batch_size=BATCH_SIZE):
    """Load and preprocess training data, return dataloaders."""
    print("[DATA] Preprocessing training data (EGNN pipeline)...")
    x_padded, mask, theta_train, n_max, knn_idx = preprocess_egnn(
        TRAIN_CSV, k_neighbors=K_NEIGHBORS
    )

    # Subsample if needed
    n_total = len(x_padded)
    if args.smoke:
        n_use = min(5000, n_total)
    elif n_total > MAX_TRAIN_TRACKS:
        n_use = MAX_TRAIN_TRACKS
    else:
        n_use = n_total

    if n_use < n_total:
        indices = torch.randperm(n_total)[:n_use]
        x_padded = x_padded[indices]
        mask = mask[indices]
        theta_train = theta_train[indices]
        knn_idx = knn_idx[indices]
        mode = 'smoke test' if args.smoke else 'memory cap'
        print(f"[DATA] Subsampled {n_total:,} -> {n_use:,} tracks ({mode})")

    # Print parameter coverage
    labels = ['Energy', 'Vx', 'Vy', 'Vz']
    print(f"[DATA] Parameter coverage:")
    for j, name in enumerate(labels):
        col = theta_train[:, j]
        print(f"        {name:8s}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}  std={col.std():.3f}")

    # Flatten for SBI: coords (N_max*3) + knn_idx (N_max*k)
    n_tracks = x_padded.shape[0]
    x_flat = torch.empty(n_tracks, n_max * (3 + K_NEIGHBORS), dtype=torch.float32)
    x_flat[:, :n_max * 3] = x_padded.view(n_tracks, -1)
    del x_padded

    CHUNK = 10000
    for start in range(0, n_tracks, CHUNK):
        end = min(start + CHUNK, n_tracks)
        x_flat[start:end, n_max * 3:] = knn_idx[start:end].float().reshape(end - start, -1)
    del knn_idx, mask
    gc.collect()

    print(f"[DATA] x_flat shape: {x_flat.shape}")
    print(f"[DATA] theta shape:  {theta_train.shape}")
    print(f"[DATA] N_max:        {n_max}")

    # Train/val split
    n_val = int(n_tracks * VAL_FRACTION)
    n_train = n_tracks - n_val
    perm = torch.randperm(n_tracks)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_dataset = TensorDataset(theta_train[train_idx], x_flat[train_idx])
    val_dataset = TensorDataset(theta_train[val_idx], x_flat[val_idx])

    # DataLoaders — pin_memory for GPU, but keep data on CPU
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=0,
    )

    print(f"[DATA] Train: {n_train:,} tracks ({len(train_loader)} batches)")
    print(f"[DATA] Val:   {n_val:,} tracks ({len(val_loader)} batches)")

    return train_loader, val_loader, n_max


def train_one_epoch(flow, cached_embedding, dir_head, energy_head,
                    optimizer, train_loader, device, epoch, max_epochs):
    """Train for one epoch with combined loss."""
    flow.train()
    dir_head.train()
    energy_head.train()

    alpha = aux_weight_schedule(epoch, max_epochs, ALPHA_START, ALPHA_END)
    beta = aux_weight_schedule(epoch, max_epochs, BETA_START, BETA_END)

    total_loss_sum = 0.0
    flow_loss_sum = 0.0
    dir_loss_sum = 0.0
    energy_loss_sum = 0.0
    n_batches = 0

    nan_printed = False  # print debug info on first NaN per epoch

    for batch_idx, (theta_batch, x_batch) in enumerate(train_loader):
        theta_batch = theta_batch.to(device)
        x_batch = x_batch.to(device)

        optimizer.zero_grad()

        # 1. Flow loss: -log_prob(theta | x)
        # flow.loss() returns per-sample NLL, shape (batch,)
        flow_nll = flow.loss(theta_batch, condition=x_batch)

        # Clamp per-sample losses to prevent outlier-driven instability
        flow_nll = flow_nll.clamp(max=LOSS_CLAMP)
        flow_loss = flow_nll.mean()

        # 2. Grab cached embedding from the EGNN forward pass
        z = cached_embedding.last_z  # (batch, d_latent)

        # 3. Auxiliary direction loss (axis-aware vMF)
        target_dir = theta_batch[:, 1:4]  # (batch, 3) — Vx, Vy, Vz
        mu_hat, kappa = dir_head(z)
        dir_nll = axis_aware_vmf_nll(mu_hat, kappa, target_dir)
        dir_nll = dir_nll.clamp(max=LOSS_CLAMP)
        dir_loss = dir_nll.mean()

        # 4. Auxiliary energy loss (heteroscedastic Gaussian)
        target_energy = theta_batch[:, 0]  # (batch,) — Energy
        E_pred, log_sigma = energy_head(z)
        e_nll = gaussian_nll(E_pred, log_sigma, target_energy)
        e_nll = e_nll.clamp(max=LOSS_CLAMP)
        energy_loss = e_nll.mean()

        # 5. Combined loss
        total_loss = flow_loss + alpha * dir_loss + beta * energy_loss

        # NaN guard: skip batch if any loss is NaN/Inf
        if not torch.isfinite(total_loss):
            if not nan_printed:
                # Debug: print what's going wrong on the first NaN batch
                nan_printed = True
                print(f"  [NaN DEBUG epoch={epoch} batch={batch_idx}]")
                print(f"    flow_loss={flow_loss.item()}, "
                      f"dir_loss={dir_loss.item()}, energy_loss={energy_loss.item()}")
                print(f"    z: min={z.min().item():.3f}, max={z.max().item():.3f}, "
                      f"mean={z.mean().item():.3f}, NaN={z.isnan().any().item()}, "
                      f"Inf={z.isinf().any().item()}")
                print(f"    kappa: min={kappa.min().item():.3f}, "
                      f"max={kappa.max().item():.3f}")
                print(f"    flow_nll: NaN={flow_nll.isnan().any().item()}, "
                      f"Inf={flow_nll.isinf().any().item()}, "
                      f"max={flow_nll.max().item():.3f}")
                # Check model weights for NaN
                n_nan_params = sum(p.isnan().any().item()
                                   for p in flow.parameters())
                print(f"    flow params with NaN: {n_nan_params}")
            continue

        # 6. Backward + clip + step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(flow.parameters()) + list(dir_head.parameters()) + list(energy_head.parameters()),
            max_norm=GRAD_CLIP
        )
        optimizer.step()

        total_loss_sum += total_loss.item()
        flow_loss_sum += flow_loss.item()
        dir_loss_sum += dir_loss.item()
        energy_loss_sum += energy_loss.item()
        n_batches += 1

    if n_batches == 0:
        print(f"  [WARN] ALL batches produced NaN in epoch {epoch}! "
              f"Returning inf losses.")
        return {
            'total': float('inf'), 'flow': float('inf'),
            'direction': float('inf'), 'energy': float('inf'),
            'alpha': alpha, 'beta': beta,
        }

    return {
        'total': total_loss_sum / n_batches,
        'flow': flow_loss_sum / n_batches,
        'direction': dir_loss_sum / n_batches,
        'energy': energy_loss_sum / n_batches,
        'alpha': alpha,
        'beta': beta,
    }


@torch.no_grad()
def validate(flow, cached_embedding, dir_head, energy_head,
             val_loader, device, epoch, max_epochs):
    """Validate with combined loss."""
    flow.eval()
    dir_head.eval()
    energy_head.eval()

    alpha = aux_weight_schedule(epoch, max_epochs, ALPHA_START, ALPHA_END)
    beta = aux_weight_schedule(epoch, max_epochs, BETA_START, BETA_END)

    total_loss_sum = 0.0
    flow_loss_sum = 0.0
    dir_loss_sum = 0.0
    energy_loss_sum = 0.0
    n_batches = 0

    for theta_batch, x_batch in val_loader:
        theta_batch = theta_batch.to(device)
        x_batch = x_batch.to(device)

        flow_nll = flow.loss(theta_batch, condition=x_batch)
        flow_nll = flow_nll.clamp(max=LOSS_CLAMP)
        flow_loss = flow_nll.mean()

        z = cached_embedding.last_z
        target_dir = theta_batch[:, 1:4]
        mu_hat, kappa = dir_head(z)
        dir_nll = axis_aware_vmf_nll(mu_hat, kappa, target_dir)
        dir_nll = dir_nll.clamp(max=LOSS_CLAMP)
        dir_loss = dir_nll.mean()

        target_energy = theta_batch[:, 0]
        E_pred, log_sigma = energy_head(z)
        e_nll = gaussian_nll(E_pred, log_sigma, target_energy)
        e_nll = e_nll.clamp(max=LOSS_CLAMP)
        energy_loss = e_nll.mean()

        total_loss = flow_loss + alpha * dir_loss + beta * energy_loss

        # NaN guard: skip bad batches in validation too
        if not torch.isfinite(total_loss):
            continue

        total_loss_sum += total_loss.item()
        flow_loss_sum += flow_loss.item()
        dir_loss_sum += dir_loss.item()
        energy_loss_sum += energy_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {
            'total': float('inf'), 'flow': float('inf'),
            'direction': float('inf'), 'energy': float('inf'),
        }

    return {
        'total': total_loss_sum / n_batches,
        'flow': flow_loss_sum / n_batches,
        'direction': dir_loss_sum / n_batches,
        'energy': energy_loss_sum / n_batches,
    }


def save_checkpoint(flow, cached_embedding, dir_head, energy_head,
                    optimizer, epoch, val_metrics, best_val_loss, path):
    """Save full training state for resuming."""
    torch.save({
        'epoch': epoch,
        'flow_state_dict': flow.state_dict(),
        'dir_head_state_dict': dir_head.state_dict(),
        'energy_head_state_dict': energy_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        # Save the full flow for SBI posterior reconstruction
        'flow_module': flow,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="GVP-EGNN Unified Training")
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 5 epochs on 5k tracks')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-epochs', type=int, default=None)
    args = parser.parse_args()

    if args.max_epochs is None:
        args.max_epochs = 5 if args.smoke else MAX_EPOCHS

    max_epochs = args.max_epochs

    t_start = time()
    print(f"\n{'='*60}")
    print(f"  GVP-EGNN Unified Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    if args.smoke:
        print(f"  MODE: Smoke test ({max_epochs} epochs, 5k tracks)")
    else:
        print(f"  MODE: Full training ({max_epochs} epochs)")

    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"  DEVICE: {device}")

    # Auto-detect batch size based on GPU VRAM
    # GVP layers use more memory per edge than vanilla EGNN
    batch_size = args.batch_size
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name()
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")

        # Auto-adjust batch size if user didn't override
        if args.batch_size == BATCH_SIZE:  # user didn't explicitly set it
            if vram_gb < 20:       # T4 (15GB) or similar
                batch_size = 32
            elif vram_gb < 45:     # A100-40GB
                batch_size = 128
            else:                   # A100-80GB
                batch_size = 256
            print(f"  Auto batch size: {batch_size} (based on {vram_gb:.0f}GB VRAM)")

    # 1. Load data
    print(f"\n[STEP 1] Loading and preprocessing data...")
    train_loader, val_loader, n_max = prepare_data(args, batch_size=batch_size)

    # 2. Build model components
    print(f"\n[STEP 2] Building GVP-EGNN + NSF flow...")
    print(f"  EGNN: hidden_dim={HIDDEN_DIM}, layers={N_LAYERS}, GVP v_dim={V_DIM}")
    print(f"  Readout: {N_HEADS} heads x {D_PROJ}d, d_latent={D_LATENT}")
    print(f"  NSF: {NUM_TRANSFORMS} transforms, hidden={NSF_HIDDEN}")

    embedding_net = EGNNEmbedding(
        n_max=n_max,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        k=K_NEIGHBORS,
        n_heads=N_HEADS,
        d_proj=D_PROJ,
        d_latent=D_LATENT,
        use_gvp=True,
        v_dim=V_DIM,
    )

    flow, cached_embedding = build_flow(embedding_net, n_max, K_NEIGHBORS, D_LATENT, device)

    dir_head = DirectionHead(d_latent=D_LATENT).to(device)
    energy_head = EnergyHead(d_latent=D_LATENT).to(device)

    # Count parameters
    n_params_egnn = sum(p.numel() for p in embedding_net.parameters())
    n_params_flow = sum(p.numel() for p in flow.parameters())
    n_params_aux = sum(p.numel() for p in dir_head.parameters()) + \
                   sum(p.numel() for p in energy_head.parameters())
    # Note: flow params include EGNN params (embedding_net is inside flow)
    n_params_total = n_params_flow + n_params_aux
    print(f"  EGNN params:  {n_params_egnn:,}")
    print(f"  Flow params:  {n_params_flow:,} (includes EGNN)")
    print(f"  Aux params:   {n_params_aux:,}")
    print(f"  Total params: {n_params_total:,}")

    # 3. Optimizer (all parameters together)
    all_params = list(flow.parameters()) + list(dir_head.parameters()) + \
                 list(energy_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and CHECKPOINT_FILE.exists():
        print(f"\n[RESUME] Loading checkpoint: {CHECKPOINT_FILE}")
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        flow.load_state_dict(ckpt['flow_state_dict'])
        dir_head.load_state_dict(ckpt['dir_head_state_dict'])
        energy_head.load_state_dict(ckpt['energy_head_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"[RESUME] Starting from epoch {start_epoch}, best_val={best_val_loss:.4f}")
        del ckpt

    # 4. Logging
    writer = SummaryWriter(str(TB_LOG_DIR))
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    if start_epoch == 0:
        with open(TRAIN_LOG, 'w') as f:
            f.write("epoch,train_total,train_flow,train_dir,train_energy,"
                    "val_total,val_flow,val_dir,val_energy,"
                    "lr,alpha,beta,epoch_sec,elapsed_min\n")

    print(f"\n[STEP 3] Training...")
    print(f"  LR: {LR_MAX} -> {LR_MIN} (cosine + {WARMUP_EPOCHS}-epoch warmup)")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Grad clip: {GRAD_CLIP}")
    print(f"  Aux weights: alpha {ALPHA_START}->{ALPHA_END}, beta {BETA_START}->{BETA_END}")
    print(f"  Patience: {PATIENCE} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"\n  TensorBoard: tensorboard --logdir \"{TB_LOG_DIR}\"")
    print(f"  CSV log: {TRAIN_LOG}\n")

    epochs_no_improve = 0

    for epoch in range(start_epoch, max_epochs):
        t_epoch = time()

        # Update learning rate
        lr = cosine_lr_with_warmup(epoch, WARMUP_EPOCHS, max_epochs, LR_MAX, LR_MIN)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Train
        train_metrics = train_one_epoch(
            flow, cached_embedding, dir_head, energy_head,
            optimizer, train_loader, device, epoch, max_epochs
        )

        # Validate
        val_metrics = validate(
            flow, cached_embedding, dir_head, energy_head,
            val_loader, device, epoch, max_epochs
        )

        epoch_sec = time() - t_epoch
        elapsed_min = (time() - t_start) / 60

        # Log
        writer.add_scalar("train/total_loss", train_metrics['total'], epoch)
        writer.add_scalar("train/flow_loss", train_metrics['flow'], epoch)
        writer.add_scalar("train/dir_loss", train_metrics['direction'], epoch)
        writer.add_scalar("train/energy_loss", train_metrics['energy'], epoch)
        writer.add_scalar("val/total_loss", val_metrics['total'], epoch)
        writer.add_scalar("val/flow_loss", val_metrics['flow'], epoch)
        writer.add_scalar("val/dir_loss", val_metrics['direction'], epoch)
        writer.add_scalar("val/energy_loss", val_metrics['energy'], epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("alpha", train_metrics['alpha'], epoch)
        writer.add_scalar("beta", train_metrics['beta'], epoch)
        writer.flush()

        with open(TRAIN_LOG, 'a') as f:
            f.write(f"{epoch},{train_metrics['total']:.6f},{train_metrics['flow']:.6f},"
                    f"{train_metrics['direction']:.6f},{train_metrics['energy']:.6f},"
                    f"{val_metrics['total']:.6f},{val_metrics['flow']:.6f},"
                    f"{val_metrics['direction']:.6f},{val_metrics['energy']:.6f},"
                    f"{lr:.6f},{train_metrics['alpha']:.4f},{train_metrics['beta']:.4f},"
                    f"{epoch_sec:.1f},{elapsed_min:.1f}\n")

        # Print progress
        print(f"Epoch {epoch:3d}/{max_epochs} | "
              f"train={train_metrics['total']:.3f} (flow={train_metrics['flow']:.3f} "
              f"dir={train_metrics['direction']:.3f} E={train_metrics['energy']:.3f}) | "
              f"val={val_metrics['total']:.3f} | "
              f"lr={lr:.1e} | {epoch_sec:.0f}s", flush=True)

        # Checkpointing + early stopping
        val_loss = val_metrics['total']
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(flow, cached_embedding, dir_head, energy_head,
                          optimizer, epoch, val_metrics, best_val_loss, BEST_CKPT_FILE)
            print(f"  -> New best val_loss={best_val_loss:.4f} (saved)")
        else:
            epochs_no_improve += 1

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(flow, cached_embedding, dir_head, energy_head,
                          optimizer, epoch, val_metrics, best_val_loss, CHECKPOINT_FILE)

            # Auto-save to Google Drive if running on Colab (crash protection)
            drive_dir = Path('/content/drive/MyDrive/sbi-srim-results/gvp_egnn')
            if drive_dir.parent.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(CHECKPOINT_FILE, drive_dir / 'checkpoint.pt')
                if BEST_CKPT_FILE.exists():
                    shutil.copy2(BEST_CKPT_FILE, drive_dir / 'best_checkpoint.pt')
                shutil.copy2(TRAIN_LOG, drive_dir / 'training_log.csv')
                print(f"  [DRIVE] Checkpoints + log saved to Google Drive")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n[STOP] No improvement for {PATIENCE} epochs. Stopping.")
            break

    writer.close()
    total_min = (time() - t_start) / 60

    # Save final checkpoint
    save_checkpoint(flow, cached_embedding, dir_head, energy_head,
                  optimizer, epoch, val_metrics, best_val_loss, CHECKPOINT_FILE)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Total time:    {total_min:.1f} min")
    print(f"  Checkpoints:   {BEST_CKPT_FILE}")
    print(f"  Log:           {TRAIN_LOG}")
    print(f"{'='*60}")

    # 5. Build SBI posterior from best checkpoint (for eval compatibility)
    print("\n[STEP 4] Building SBI posterior from best checkpoint...")
    posterior = None
    try:
        best_ckpt = torch.load(BEST_CKPT_FILE, map_location='cpu',
                               weights_only=False)
        best_flow = best_ckpt['flow_module']

        from sbi.inference import SNPE
        from sbi.utils import BoxUniform

        # Use BoxUniform for posterior (SBI needs it for sampling)
        prior = BoxUniform(
            low=torch.tensor([0.0, -1.0, -1.0, -1.0]),
            high=torch.tensor([105.0, 1.0, 1.0, 1.0])
        )
        inference = SNPE(prior=prior, device='cpu')
        posterior = inference.build_posterior(best_flow, sample_with="direct")

        posterior_path = RESULTS_DIR / "gvp_egnn_posterior.pt"
        torch.save(posterior, posterior_path)
        print(f"[SAVE] Posterior saved -> {posterior_path}")
    except Exception as e:
        print(f"[WARN] Could not build posterior: {e}")
        import traceback
        traceback.print_exc()
        print(f"       Best checkpoint is still available at {BEST_CKPT_FILE}")

    # 6. Run evaluation if eval data exists and posterior was built
    if posterior is None:
        print("\n[SKIP] Evaluation skipped — posterior could not be built")
    elif EVAL_CSV.exists():
        print("\n[STEP 5] Running Full Posterior Evaluation...")
        try:
            from src.evaluation.eval_mcpe3d import ContinuousEvaluator3D

            eval_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            evaluator = ContinuousEvaluator3D(posterior, device=eval_device)

            # More samples = better calibration analysis
            n_eval_samples = 50 if args.smoke else 200
            df_eval = evaluator.run_eval(EVAL_CSV, num_samples=n_eval_samples)

            # Save CSV results (17 columns: point estimates + posterior stats)
            eval_path = RESULTS_DIR / "eval_results.csv"
            df_eval.to_csv(eval_path, index=False)

            # Generate all diagnostic plots (3 figures: point estimates,
            # posterior quality, calibration)
            print("[PLOTS] Generating evaluation plots...")
            evaluator.plot_results(df_eval, save_dir=RESULTS_DIR)

            # Corner plots for best, median, and worst tracks
            if len(df_eval) > 0:
                sorted_by_angle = df_eval['angular_error_deg'].argsort()
                example_tracks = {
                    'best': sorted_by_angle.iloc[0],
                    'median': sorted_by_angle.iloc[len(sorted_by_angle) // 2],
                    'worst': sorted_by_angle.iloc[-1],
                }
                corners_dir = str(RESULTS_DIR / "corners")
                for label, idx in example_tracks.items():
                    print(f"[PLOTS] Corner plot for {label} track (idx={idx})...")
                    evaluator.plot_corner(idx, df_eval, save_dir=corners_dir)

            # =============================================================
            # SBI Gold-Standard Diagnostics: SBC + TARP
            # (using SBI's built-in tools — the rigorous way)
            # =============================================================
            print("\n[STEP 6] Running SBI Posterior Diagnostics (SBC + TARP)...")
            try:
                from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
                from sbi.analysis import sbc_rank_plot, plot_tarp
                import matplotlib.pyplot as plt

                # Prepare eval data for SBI diagnostics
                # Need theta (true params) and x (observations) as tensors
                emb_net = posterior.posterior_estimator.embedding_net
                k_neighbors = emb_net.k
                n_max_model = emb_net.n_max

                x_padded, mask, theta_eval, n_max_eval, knn_idx = preprocess_egnn(
                    EVAL_CSV, k_neighbors=k_neighbors
                )

                # Pad to match model n_max if needed
                if n_max_eval < n_max_model:
                    pad_pts = n_max_model - n_max_eval
                    n_ev = x_padded.shape[0]
                    x_padded = torch.cat([x_padded,
                        torch.zeros(n_ev, pad_pts, 3)], dim=1)
                    knn_idx = torch.cat([knn_idx,
                        torch.full((n_ev, pad_pts, k_neighbors), -1,
                                   dtype=knn_idx.dtype)], dim=1)
                    n_max_eval = n_max_model

                # Flatten same as training
                n_ev = x_padded.shape[0]
                x_eval_flat = torch.empty(n_ev, n_max_eval * (3 + k_neighbors),
                                          dtype=torch.float32)
                x_eval_flat[:, :n_max_eval * 3] = x_padded.view(n_ev, -1)
                CHUNK = 10000
                for s in range(0, n_ev, CHUNK):
                    e = min(s + CHUNK, n_ev)
                    x_eval_flat[s:e, n_max_eval * 3:] = \
                        knn_idx[s:e].float().reshape(e - s, -1)
                del x_padded, mask, knn_idx

                # Use subset for diagnostics (SBC/TARP are expensive)
                n_diag = min(500, n_ev) if not args.smoke else min(100, n_ev)
                diag_idx = torch.randperm(n_ev)[:n_diag]
                theta_diag = theta_eval[diag_idx]
                x_diag = x_eval_flat[diag_idx]
                n_sbc_samples = 100 if args.smoke else 500

                # --- SBC (Simulation-Based Calibration) ---
                print(f"  Running SBC on {n_diag} tracks, {n_sbc_samples} posterior samples each...")
                try:
                    ranks, dap_samples = run_sbc(
                        theta_diag, x_diag, posterior,
                        num_posterior_samples=n_sbc_samples,
                        use_batched_sampling=True,
                        show_progress_bar=True,
                    )

                    # SBC rank plot
                    fig_sbc, ax_sbc = sbc_rank_plot(
                        ranks, n_sbc_samples,
                        parameter_labels=['Energy', 'Vx', 'Vy', 'Vz'],
                        plot_type='cdf',
                    )
                    fig_sbc.suptitle("SBC Rank Plot (CDF)", fontsize=14)
                    fig_sbc.tight_layout()
                    sbc_path = RESULTS_DIR / "eval_sbc_ranks.png"
                    fig_sbc.savefig(sbc_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_sbc)
                    print(f"  [PLOT] Saved {sbc_path}")

                    # SBC check (C2ST statistics)
                    sbc_stats = check_sbc(
                        ranks, theta_diag, dap_samples,
                        num_posterior_samples=n_sbc_samples,
                    )
                    print(f"  SBC C2ST scores (0.5=perfect, >0.6=poor):")
                    param_names = ['Energy', 'Vx', 'Vy', 'Vz']
                    for i, name in enumerate(param_names):
                        c2st = sbc_stats['c2st_ranks'][i].item()
                        status = "OK" if c2st < 0.6 else "WARN"
                        print(f"    {name:8s}: C2ST={c2st:.3f}  [{status}]")
                    sbc_mean = sbc_stats['c2st_ranks'].mean().item()
                    print(f"    Mean C2ST: {sbc_mean:.3f}")

                except Exception as e:
                    print(f"  [WARN] SBC failed: {e}")

                # --- TARP (Tests of Accuracy with Random Points) ---
                print(f"\n  Running TARP on {n_diag} tracks...")
                try:
                    ecp, alpha = run_tarp(
                        theta_diag, x_diag, posterior,
                        num_posterior_samples=n_sbc_samples,
                        use_batched_sampling=True,
                        show_progress_bar=True,
                    )

                    # TARP plot
                    fig_tarp, ax_tarp = plot_tarp(ecp, alpha,
                                                  title="TARP Diagnostic")
                    tarp_path = RESULTS_DIR / "eval_tarp.png"
                    fig_tarp.savefig(tarp_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_tarp)
                    print(f"  [PLOT] Saved {tarp_path}")

                    # TARP check (KS test)
                    tarp_ks, tarp_pval = check_tarp(ecp, alpha)
                    cal_status = "WELL CALIBRATED" if tarp_pval > 0.05 else "MISCALIBRATED"
                    print(f"  TARP: KS stat={tarp_ks:.4f}, p-value={tarp_pval:.4f}")
                    print(f"  Posterior is {cal_status} (p>0.05 = well calibrated)")

                except Exception as e:
                    print(f"  [WARN] TARP failed: {e}")

            except ImportError as e:
                print(f"  [SKIP] SBI diagnostics not available: {e}")

            # Full results report
            print(f"\n{'='*60}")
            print(f"  RESULTS (GVP-EGNN-SBI)")
            print(f"{'='*60}")
            if len(df_eval) > 0:
                med_E_err = df_eval["energy_error"].median()
                med_ang = df_eval["angular_error_deg"].median()
                flip_rate = (df_eval["angular_error_deg"] > 90).mean() * 100
                head_tail = 100 - flip_rate

                print(f"  Median Energy Error:      {med_E_err:.2f} keV")
                print(f"  Median Angular Error:     {med_ang:.2f} deg")
                print(f"  Head-Tail Accuracy:       {head_tail:.1f}%")

                if 'energy_std' in df_eval.columns:
                    med_E_std = df_eval["energy_std"].median()
                    print(f"  Median Energy sigma:      {med_E_std:.2f} keV")
                if 'angular_cone_68' in df_eval.columns:
                    med_cone68 = df_eval["angular_cone_68"].median()
                    print(f"  Median 68% Cone:          {med_cone68:.2f} deg")
                if 'angular_cone_90' in df_eval.columns:
                    med_cone90 = df_eval["angular_cone_90"].median()
                    print(f"  Median 90% Cone:          {med_cone90:.2f} deg")
                if 'energy_ci90_width' in df_eval.columns:
                    med_ci90 = df_eval["energy_ci90_width"].median()
                    print(f"  Median Energy 90% CI:     {med_ci90:.2f} keV")

                # Angular error percentiles
                p25 = df_eval["angular_error_deg"].quantile(0.25)
                p75 = df_eval["angular_error_deg"].quantile(0.75)
                p90 = df_eval["angular_error_deg"].quantile(0.90)
                print(f"\n  Angular Error Percentiles:")
                print(f"    25th: {p25:.2f} deg")
                print(f"    50th: {med_ang:.2f} deg")
                print(f"    75th: {p75:.2f} deg")
                print(f"    90th: {p90:.2f} deg")

                # Comparison to previous model
                print(f"\n  Previous model (EGNN v1):")
                print(f"    Median Angular Error: 15.83 deg")
                print(f"    Head-Tail Accuracy:   93.4%")
                improvement = (15.83 - med_ang) / 15.83 * 100
                print(f"\n  Improvement: {improvement:+.1f}% angular error")

            print(f"\n  Training time:  {total_min:.1f} min")
            print(f"  Best val_loss:  {best_val_loss:.4f}")
            print(f"  Results dir:    {RESULTS_DIR}")
            print(f"  Eval CSV:       {eval_path}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
