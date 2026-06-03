#!/usr/bin/env python3
"""
GVP-EGNN Unified Training Script — v2.1 SIIMPL VARIANT
=======================================================
Trains the GVP-EGNN + NSF flow on SIIMPL crystalline-target data (replaces
v2's SRIM amorphous-target data). Five differences from train_gvp_egnn.py:

  1. Data paths point to data/siimpl/siimpl_{train,eval}.csv
  2. SO(3) augmentation replaced with the 48-element Oh group (diamond's
     Fd-3m point group, with inversion). SO(3) would map channeling axes
     to non-lattice directions and destroy the channeling signal.
  3. Train/val split is BY CONFIG, not by individual track: the SIIMPL
     data has multiple ion realizations per (E, theta, phi) config, and
     random per-track splitting would leak reps of the same config into
     val.
  4. Stage 2 (flow-only fine-tune) trains on the Pool A subset only,
     keeping the posterior's implicit prior at uniform-on-S^2 instead of
     the channeling-enriched Pool A+B mixture.
  5. Results land in results/gvp_egnn_v21_siimpl/.

Otherwise identical to v2: same architecture, hyperparameters, loss
combination, eval suite (SBC/TARP/ECE).

Usage:
    python -m src.scripts.train_gvp_egnn_v21 --smoke      # 5-epoch smoke
    python -m src.scripts.train_gvp_egnn_v21              # full training
    python -m src.scripts.train_gvp_egnn_v21 --resume     # from checkpoint
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
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Ensure 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_utils import preprocess_egnn
from src.models.egnn import (EGNNEmbedding, CachedEGNNEmbedding, DirectionHead, EnergyHead,
                              ScalarAugmentedEmbedding)
from src.models.vmf_loss import axis_aware_vmf_nll, gaussian_nll

# SBI Imports (only for building the flow architecture)
from sbi.neural_nets import posterior_nn


# ================= CONFIGURATION =================
BASE_DIR      = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR      = BASE_DIR / "data/siimpl"
TRAIN_CSV     = DATA_DIR / "siimpl_train.csv"
EVAL_CSV      = DATA_DIR / "siimpl_eval.csv"
RESULTS_DIR   = BASE_DIR / "results/gvp_egnn_v21_siimpl"

# Pool A spans ion_number < POOL_B_ION_START; Pool B is the channeling-fill
# subset. Stage 2 (flow-only) is trained on Pool A only so the posterior's
# implicit prior is uniform-on-S^2 (the prior we report against).
POOL_B_ION_START = 245_000

# Output files
CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.pt"
BEST_CKPT_FILE   = RESULTS_DIR / "best_checkpoint.pt"
BEST_S2_CKPT_FILE = RESULTS_DIR / "best_checkpoint_stage2.pt"
TRAIN_LOG        = RESULTS_DIR / "training_log.csv"
TB_LOG_DIR       = RESULTS_DIR / "tb_logs"

# Architecture hyperparameters (GVP-EGNN)
HIDDEN_DIM     = 96       # up from 64
N_LAYERS       = 6        # up from 4
K_NEIGHBORS    = 16       # same
N_HEADS        = 8        # same
D_PROJ         = 48       # up from 32
D_LATENT       = 384      # up from 256
N_SCALAR_FEATS = 2        # scalar features appended to z in Stage 2 (n_vac, lateral_spread)
D_AUG          = D_LATENT + N_SCALAR_FEATS  # augmented conditioning dim for Stage 2 flow
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
PATIENCE       = 30       # early stopping (stage 1)
VAL_FRACTION   = 0.05     # 10k val tracks is plenty with 197k total
GRAD_CLIP      = 1.0
MAX_TRAIN_TRACKS = 280_000  # SIIMPL v2.1 train: 267k tracks (200k Pool A + 22k Pool B + headroom)

# Auxiliary loss weights (decay schedule)
# v2: raised beta 10x (safe now that log_sigma is clamped in EnergyHead)
ALPHA_START    = 0.5      # vMF direction loss weight (start)
ALPHA_END      = 0.1      # vMF direction loss weight (end)
BETA_START     = 0.1      # Gaussian energy loss weight (start)  — was 0.01
BETA_END       = 0.05     # Gaussian energy loss weight (end)    — was 0.001

# Stage 2: flow-only fine-tuning
STAGE2_EPOCHS    = 60
STAGE2_LR_MAX    = 5e-5
STAGE2_LR_MIN    = 1e-6
STAGE2_WARMUP    = 3
STAGE2_PATIENCE  = 20
STAGE2_GRAD_CLIP = 0.5

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
    """Cosine decay of auxiliary loss weights (holds near start longer than linear)."""
    progress = min(epoch / max(max_epochs - 1, 1), 1.0)
    cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
    return start + (end - start) * cosine_progress


def _build_oh_group():
    """Construct the 48 elements of the full octahedral group Oh (point group
    m-3m, the symmetry group of diamond's lattice). Built from all signed
    permutation matrices of 3 axes: 3! permutations x 2^3 sign combinations
    = 48 matrices. 24 have det=+1 (proper rotations of the cube), 24 have
    det=-1 (improper, equal to inversion composed with a proper rotation).
    """
    from itertools import permutations
    mats = []
    for perm in permutations([0, 1, 2]):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    M = np.zeros((3, 3), dtype=np.float32)
                    M[0, perm[0]] = sx
                    M[1, perm[1]] = sy
                    M[2, perm[2]] = sz
                    mats.append(M)
    return np.stack(mats)


OH_GROUP_NP = _build_oh_group()                       # (48, 3, 3) numpy
OH_GROUP_T  = torch.from_numpy(OH_GROUP_NP)           # (48, 3, 3) cpu tensor
assert OH_GROUP_NP.shape == (48, 3, 3), "Oh group must have 48 elements"


def apply_oh_augmentation(x_batch, theta_batch, n_max):
    """Apply random Oh-group element to coordinates and direction targets.

    Oh = the 48-element octahedral point group (diamond's point group, Fd-3m).
    Contains 24 proper rotations of the cube + 24 improper (inversion-composed).

    Why Oh instead of SO(3): the lattice is invariant only under Oh, not under
    arbitrary SO(3) rotations. A generic SO(3) rotation would map a channeling
    direction like <110> to a non-lattice direction, destroying the channeling
    signal we paid SIIMPL generation time to capture.

    Why inversion matters: training data is sampled from the UPPER hemisphere
    only (SIIMPL's implantation geometry). The 24 improper Oh elements (each =
    inversion x proper rotation) map upper-hemisphere samples to the lower
    hemisphere, recovering full-S^2 coverage at training time. Diamond's
    inversion symmetry (the "-3" in Fd-3m) guarantees the cascade physics is
    parity-invariant, so this augmentation is exact.

    kNN indices are distance-based, so unchanged under rotation/reflection.
    """
    B = x_batch.shape[0]
    device = x_batch.device

    # Sample one Oh element per batch sample
    idx = torch.randint(0, 48, (B,), device=device)
    R = OH_GROUP_T.to(device)[idx]  # (B, 3, 3)

    # Rotate coordinates (first n_max*3 elements of x_flat)
    coords = x_batch[:, :n_max * 3].view(B, n_max, 3)
    coords_rot = torch.bmm(coords, R.transpose(-1, -2))
    x_batch[:, :n_max * 3] = coords_rot.reshape(B, -1)

    # Rotate direction targets (columns 1:4 of theta = Vx, Vy, Vz)
    dirs = theta_batch[:, 1:4].unsqueeze(1)  # (B, 1, 3)
    dirs_rot = torch.bmm(dirs, R.transpose(-1, -2)).squeeze(1)  # (B, 3)
    theta_batch[:, 1:4] = dirs_rot

    return x_batch, theta_batch


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


def build_flow_from_embedding(aug_embedding, n_max, k, d_aug, device):
    """Build a fresh NSF flow conditioned on an already-wrapped embedding.
    Used for Stage 2 with ScalarAugmentedEmbedding (d_aug = D_LATENT + N_SCALAR_FEATS).
    The embedding is NOT wrapped again — pass it directly as embedding_net."""
    build_fn = posterior_nn(
        model="nsf",
        embedding_net=aug_embedding,
        hidden_features=NSF_HIDDEN,
        num_transforms=NUM_TRANSFORMS,
        z_score_x="none",
        z_score_theta="independent",
    )
    dummy_theta = torch.zeros(100, 4)
    dummy_theta[:, 0] = torch.rand(100) * 105.0
    dummy_theta[:, 1:4] = F.normalize(torch.randn(100, 3), dim=-1)
    dummy_x = torch.randn(100, n_max * (3 + k))
    flow = build_fn(dummy_theta, dummy_x)
    return flow.to(device)


def _load_ion_numbers_aligned(csv_path):
    """Replay preprocess_egnn's filtering (sort by ion_number, drop tracks
    with <3 vacancies) on the ion_number column only, so we get the array
    of ion_numbers in the same order as theta_train. One CSV read of a
    single column — fast even for the 1.9 GB file."""
    nums = pd.read_csv(csv_path, usecols=["ion_number"])["ion_number"]
    vac_counts = nums.value_counts().sort_index()
    valid = vac_counts[vac_counts >= 3].index.values  # sorted, filtered
    return torch.from_numpy(valid.astype(np.int64))


def prepare_data(args, batch_size=BATCH_SIZE):
    """Load and preprocess training data, return dataloaders.

    Returns:
        train_loader:        Pool A + Pool B (Stage 1 training)
        val_loader:          Pool A + Pool B (validation)
        train_loader_poolA:  Pool A subset of train (Stage 2 training)
        n_max:               padding length
        scalar_mean/std:     scalar feature normalization stats
    """
    print("[DATA] Preprocessing training data (EGNN pipeline)...")
    x_padded, mask, theta_train, n_max, knn_idx = preprocess_egnn(
        TRAIN_CSV, k_neighbors=K_NEIGHBORS
    )

    # Recover per-track ion_number (preprocess_egnn sorted-by-ion_number
    # and dropped n<3, so the alignment is recoverable from the CSV alone).
    track_ions = _load_ion_numbers_aligned(TRAIN_CSV)
    assert len(track_ions) == len(theta_train), (
        f"Ion-number alignment mismatch: {len(track_ions)} vs "
        f"{len(theta_train)}. preprocess_egnn filter may have changed.")
    n_pool_A = int((track_ions < POOL_B_ION_START).sum())
    print(f"[DATA] Pool A: {n_pool_A:,} tracks  "
          f"(ion_number < {POOL_B_ION_START:,})")
    print(f"[DATA] Pool B: {len(track_ions) - n_pool_A:,} tracks "
          f"(ion_number >= {POOL_B_ION_START:,})")

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
        track_ions = track_ions[indices]
        mode = 'smoke test' if args.smoke else 'memory cap'
        print(f"[DATA] Subsampled {n_total:,} -> {n_use:,} tracks ({mode})")

    # Print parameter coverage
    labels = ['Energy', 'Vx', 'Vy', 'Vz']
    print(f"[DATA] Parameter coverage:")
    for j, name in enumerate(labels):
        col = theta_train[:, j]
        print(f"        {name:8s}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}  std={col.std():.3f}")

    # Compute scalar feature normalisation stats (before deleting x_padded/mask)
    _n_vac   = mask.sum(dim=1).float()                              # (N,)
    _yz      = x_padded[:, :, 1:]                                   # (N, N_max, 2)
    _mf      = mask.float().unsqueeze(-1)
    _yz_mean = (_yz * _mf).sum(dim=1) / _n_vac.unsqueeze(-1).clamp(min=1)
    _yz_dev  = (_yz - _yz_mean.unsqueeze(1)) * _mf
    _lat_var = (_yz_dev.pow(2) * _mf).sum(dim=(1, 2)) / _n_vac.clamp(min=1)
    _log_nv  = torch.log(_n_vac + 1.0)
    _log_lv  = torch.log(_lat_var + 1e-4)
    scalar_mean = torch.stack([_log_nv.mean(), _log_lv.mean()])
    scalar_std  = torch.stack([_log_nv.std().clamp(min=1e-8), _log_lv.std().clamp(min=1e-8)])
    print(f"[DATA] Scalar stats: log_nvac={scalar_mean[0]:.3f}±{scalar_std[0]:.3f}  "
          f"log_latvar={scalar_mean[1]:.3f}±{scalar_std[1]:.3f}")
    del _n_vac, _yz, _mf, _yz_mean, _yz_dev, _lat_var, _log_nv, _log_lv

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

    # --- Config-aware train/val split -------------------------------
    # SIIMPL data has multiple ion realizations per (E, theta, phi) config
    # (50 per Pool A config, 25 per Pool B config). Random per-track splitting
    # would put reps of the SAME config on both sides => val leakage.
    # Group by exact label tuple; each unique label = one config.
    theta_np = theta_train.numpy()
    _, config_id_np = np.unique(theta_np, axis=0, return_inverse=True)
    config_id = torch.from_numpy(config_id_np.astype(np.int64))
    n_configs = int(config_id.max().item()) + 1
    print(f"[DATA] {n_configs:,} unique configs "
          f"({n_tracks/n_configs:.1f} reps/config on average)")

    n_val_configs = max(1, int(n_configs * VAL_FRACTION))
    config_perm = torch.randperm(n_configs)
    val_config_ids = config_perm[:n_val_configs]
    val_mask = torch.isin(config_id, val_config_ids)

    val_idx = torch.where(val_mask)[0]
    train_idx = torch.where(~val_mask)[0]
    n_train = len(train_idx)
    n_val = len(val_idx)

    # Pool A subset of train (for Stage 2 flow training — keeps prior at S^2)
    train_is_poolA = (track_ions[train_idx] < POOL_B_ION_START)
    train_poolA_idx = train_idx[train_is_poolA]
    n_train_poolA = len(train_poolA_idx)

    train_dataset = TensorDataset(theta_train[train_idx], x_flat[train_idx])
    val_dataset = TensorDataset(theta_train[val_idx], x_flat[val_idx])
    train_dataset_poolA = TensorDataset(theta_train[train_poolA_idx],
                                        x_flat[train_poolA_idx])

    # DataLoaders — pin_memory for GPU, data stays on CPU
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=0,
    )
    train_loader_poolA = DataLoader(
        train_dataset_poolA, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True,
    )

    print(f"[DATA] Train:        {n_train:,} tracks "
          f"({len(train_loader)} batches) — Stage 1")
    print(f"[DATA] Train Pool A: {n_train_poolA:,} tracks "
          f"({len(train_loader_poolA)} batches) — Stage 2")
    print(f"[DATA] Val:          {n_val:,} tracks ({len(val_loader)} batches)")

    return train_loader, val_loader, train_loader_poolA, n_max, scalar_mean, scalar_std


def train_one_epoch(flow, cached_embedding, dir_head, energy_head,
                    optimizer, train_loader, device, epoch, max_epochs,
                    n_max=None, augment=True):
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

        # Oh-group data augmentation (training only) — replaces v2's SO(3)
        # to preserve diamond's lattice/channeling structure.
        if augment and n_max is not None:
            x_batch, theta_batch = apply_oh_augmentation(
                x_batch, theta_batch, n_max
            )

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
    parser.add_argument('--stage2-only', action='store_true',
                        help='Skip stage 1, load best checkpoint, run stage 2 + eval only')
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
    if args.stage2_only:
        print(f"  MODE: Stage 2 only (flow fine-tuning + eval)")
    elif args.smoke:
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
    (train_loader, val_loader, train_loader_poolA,
     n_max, scalar_mean, scalar_std) = prepare_data(args, batch_size=batch_size)

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

    # Load best checkpoint for stage2-only mode
    start_epoch = 0
    best_val_loss = float('inf')
    if args.stage2_only:
        if not BEST_CKPT_FILE.exists():
            print(f"[ERROR] --stage2-only requires {BEST_CKPT_FILE} to exist!")
            sys.exit(1)
        print(f"\n[STAGE2-ONLY] Loading best checkpoint: {BEST_CKPT_FILE}")
        ckpt = torch.load(BEST_CKPT_FILE, map_location=device, weights_only=False)
        flow.load_state_dict(ckpt['flow_state_dict'])   # loads backbone weights too
        best_val_loss = ckpt['best_val_loss']
        print(f"[STAGE2-ONLY] Best val_loss from stage 1: {best_val_loss:.4f}")
        print(f"[STAGE2-ONLY] dir_head / energy_head will be reinitialized with D_AUG={D_AUG}")
        del ckpt

    # Resume from checkpoint if requested
    elif args.resume and CHECKPOINT_FILE.exists():
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

    # 4. Stage 1 Training (skip if --stage2-only)
    if args.stage2_only:
        stage1_min = 0.0
        print(f"\n[SKIP] Stage 1 skipped (--stage2-only)")
    else:
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
                optimizer, train_loader, device, epoch, max_epochs,
                n_max=n_max, augment=True
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
            val_loss = val_metrics['flow']  # posterior quality is the goal
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

        stage1_min = (time() - t_start) / 60
        print(f"\n{'='*60}")
        print(f"  Stage 1 Complete")
        print(f"  Best val flow loss: {best_val_loss:.4f}")
        print(f"  Stage 1 time:       {stage1_min:.1f} min")
        print(f"{'='*60}")

    # ================================================================
    # STAGE 2: Scalar-Augmented Flow Fine-Tuning  (POOL A ONLY)
    # ================================================================
    # The flow is retrained on the Pool A subset alone (uniform-on-S^2
    # configs) so the posterior's implicit prior matches the prior we
    # report in the paper. Pool B's channeling-enriched configs would
    # bias the flow's implicit prior away from uniform-on-S^2.
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Scalar-Augmented Flow Fine-Tuning ({STAGE2_EPOCHS} epochs)")
    print(f"  Backbone frozen. Flow + aux heads rebuilt with D_AUG={D_AUG}.")
    print(f"  Training set: Pool A only ({len(train_loader_poolA.dataset):,} tracks)")
    print(f"{'='*60}")

    # Load best Stage 1 backbone weights (if not already loaded via --stage2-only)
    if not args.stage2_only:
        best_ckpt = torch.load(BEST_CKPT_FILE, map_location=device, weights_only=False)
        flow.load_state_dict(best_ckpt['flow_state_dict'])   # restores backbone
        print(f"  Loaded best stage 1 checkpoint (val_loss={best_val_loss:.4f})")
        del best_ckpt

    # Freeze backbone
    for param in cached_embedding.embedding.parameters():
        param.requires_grad = False
    print(f"  Backbone frozen ({sum(p.numel() for p in cached_embedding.embedding.parameters()):,} params)")

    # Wrap frozen backbone with scalar feature augmentation
    aug_embedding = ScalarAugmentedEmbedding(
        cached_embedding, n_max=n_max,
        scalar_mean=scalar_mean.to(device),
        scalar_std=scalar_std.to(device),
    )

    # Build new flow conditioned on z_aug (D_AUG = D_LATENT + N_SCALAR_FEATS)
    # Coupling layers start from random; backbone is frozen inside aug_embedding.
    print(f"  Building new NSF flow with conditioning dim {D_AUG}...")
    flow_s2 = build_flow_from_embedding(aug_embedding, n_max, K_NEIGHBORS, D_AUG, device)

    # Fresh aux heads sized for D_AUG
    dir_head_s2    = DirectionHead(d_latent=D_AUG).to(device)
    energy_head_s2 = EnergyHead(d_latent=D_AUG).to(device)

    # Trainable params: new flow coupling layers + new aux heads (backbone excluded)
    s2_params = ([p for p in flow_s2.parameters() if p.requires_grad]
                 + list(dir_head_s2.parameters())
                 + list(energy_head_s2.parameters()))
    print(f"  Trainable params (Stage 2): {sum(p.numel() for p in s2_params):,}")

    optimizer_s2 = torch.optim.AdamW(s2_params, lr=STAGE2_LR_MAX,
                                      weight_decay=WEIGHT_DECAY)

    best_val_loss_s2 = float('inf')   # flow starts fresh — don't compare to Stage 1 NLL
    epochs_no_improve_s2 = 0
    t_s2_start = time()

    for s2_epoch in range(STAGE2_EPOCHS):
        ep_start = time()

        lr = cosine_lr_with_warmup(s2_epoch, STAGE2_WARMUP, STAGE2_EPOCHS,
                                    STAGE2_LR_MAX, STAGE2_LR_MIN)
        for pg in optimizer_s2.param_groups:
            pg['lr'] = lr

        # ── Train (Pool A subset only — keeps flow prior uniform-on-S^2) ──
        flow_s2.train()
        dir_head_s2.train()
        energy_head_s2.train()
        flow_loss_sum = energy_loss_sum = dir_loss_sum = 0.0
        n_batches = 0

        for theta_batch, x_batch in train_loader_poolA:
            theta_batch = theta_batch.to(device)
            x_batch     = x_batch.to(device)

            optimizer_s2.zero_grad()

            flow_nll  = flow_s2.loss(theta_batch, condition=x_batch).clamp(max=LOSS_CLAMP)
            flow_loss = flow_nll.mean()

            z_aug = aug_embedding.last_z   # (B, D_AUG) — set by flow_s2.loss()

            E_pred, log_sigma = energy_head_s2(z_aug)
            e_nll  = gaussian_nll(E_pred, log_sigma, theta_batch[:, 0]).clamp(max=LOSS_CLAMP)
            e_loss = e_nll.mean()

            mu_hat, kappa = dir_head_s2(z_aug)
            d_nll  = axis_aware_vmf_nll(mu_hat, kappa, theta_batch[:, 1:4]).clamp(max=LOSS_CLAMP)
            d_loss = d_nll.mean()

            total = flow_loss + BETA_END * e_loss + ALPHA_END * d_loss
            if not torch.isfinite(total):
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(s2_params, max_norm=STAGE2_GRAD_CLIP)
            optimizer_s2.step()

            flow_loss_sum   += flow_loss.item()
            energy_loss_sum += e_loss.item()
            dir_loss_sum    += d_loss.item()
            n_batches += 1

        train_flow = flow_loss_sum   / max(n_batches, 1)
        train_e    = energy_loss_sum / max(n_batches, 1)
        train_d    = dir_loss_sum    / max(n_batches, 1)

        # ── Validate ──
        flow_s2.eval()
        dir_head_s2.eval()
        energy_head_s2.eval()
        vf_sum = ve_sum = vd_sum = 0
        vn = 0
        with torch.no_grad():
            for theta_batch, x_batch in val_loader:
                theta_batch = theta_batch.to(device)
                x_batch     = x_batch.to(device)
                vfl = flow_s2.loss(theta_batch, condition=x_batch).clamp(max=LOSS_CLAMP).mean()
                z_aug = aug_embedding.last_z
                ep, ls = energy_head_s2(z_aug)
                vel = gaussian_nll(ep, ls, theta_batch[:, 0]).clamp(max=LOSS_CLAMP).mean()
                mh, kp = dir_head_s2(z_aug)
                vdl = axis_aware_vmf_nll(mh, kp, theta_batch[:, 1:4]).clamp(max=LOSS_CLAMP).mean()
                if torch.isfinite(vfl + vel + vdl):
                    vf_sum += vfl.item(); ve_sum += vel.item(); vd_sum += vdl.item(); vn += 1

        val_flow = vf_sum / max(vn, 1)
        val_e    = ve_sum / max(vn, 1)

        ep_sec = time() - ep_start
        print(f"  S2 Epoch {s2_epoch}/{STAGE2_EPOCHS} | "
              f"flow={train_flow:.4f}/{val_flow:.4f}  E={train_e:.3f}/{val_e:.3f}  "
              f"dir={train_d:.3f} | lr={lr:.1e} | {ep_sec:.0f}s", flush=True)

        if math.isfinite(val_flow) and val_flow < best_val_loss_s2:
            best_val_loss_s2 = val_flow
            epochs_no_improve_s2 = 0
            # Save: use flow_s2 + new aux heads; store scalar stats for inference
            torch.save({
                'epoch': s2_epoch,
                'flow_state_dict':        flow_s2.state_dict(),
                'dir_head_state_dict':    dir_head_s2.state_dict(),
                'energy_head_state_dict': energy_head_s2.state_dict(),
                'optimizer_state_dict':   optimizer_s2.state_dict(),
                'val_metrics': {'flow': val_flow, 'energy': val_e},
                'best_val_loss': best_val_loss_s2,
                'flow_module':   flow_s2,          # full object for posterior building
                'scalar_mean':   scalar_mean,
                'scalar_std':    scalar_std,
                'n_scalar_feats': N_SCALAR_FEATS,
            }, BEST_S2_CKPT_FILE)
            print(f"  [SAVE] New best S2 → {BEST_S2_CKPT_FILE.name} (val_flow={val_flow:.4f})")
        else:
            epochs_no_improve_s2 += 1

        if epochs_no_improve_s2 >= STAGE2_PATIENCE:
            print(f"\n  [STOP] Stage 2: No improvement for {STAGE2_PATIENCE} epochs.")
            break

    # Alias so the posterior-building code below can use flow_s2 / new aux heads
    flow        = flow_s2
    dir_head    = dir_head_s2
    energy_head = energy_head_s2

    s2_min = (time() - t_s2_start) / 60
    total_min = (time() - t_start) / 60

    print(f"\n{'='*60}")
    print(f"  Training Complete (Both Stages)")
    print(f"  Stage 1 best val_flow: {best_val_loss:.4f}")
    print(f"  Stage 2 best val_flow: {best_val_loss_s2:.4f}")
    print(f"  Stage 1 time: {stage1_min:.1f} min | Stage 2 time: {s2_min:.1f} min")
    print(f"  Total time:   {total_min:.1f} min")
    print(f"  Stage 1 ckpt: {BEST_CKPT_FILE}")
    print(f"  Stage 2 ckpt: {BEST_S2_CKPT_FILE}")
    print(f"  Log:          {TRAIN_LOG}")
    print(f"{'='*60}")

    # (backbone stays frozen; flow_s2 / aux heads are already unfrozen)

    # 5. Build SBI posterior from best checkpoint (for eval compatibility)
    # Prefer Stage 2 checkpoint if it exists, otherwise fall back to Stage 1.
    posterior_ckpt_file = BEST_S2_CKPT_FILE if BEST_S2_CKPT_FILE.exists() else BEST_CKPT_FILE
    print(f"\n[STEP 4] Building SBI posterior from {posterior_ckpt_file.name}...")
    posterior = None
    try:
        best_ckpt = torch.load(posterior_ckpt_file, map_location='cpu',
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
        print(f"       Best checkpoint is still available at {posterior_ckpt_file}")

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
            t_eval_start = time()
            df_eval = evaluator.run_eval(EVAL_CSV, num_samples=n_eval_samples)
            t_eval_end = time()

            # Inference timing report
            n_eval_tracks = len(df_eval)
            eval_total_sec = t_eval_end - t_eval_start
            if n_eval_tracks > 0:
                ms_per_track = (eval_total_sec / n_eval_tracks) * 1000
                tracks_per_sec = n_eval_tracks / eval_total_sec
                print(f"\n  [INFERENCE TIMING]")
                print(f"    Total eval time:    {eval_total_sec:.1f}s for {n_eval_tracks} tracks")
                print(f"    Per-track (w/ {n_eval_samples} posterior samples): {ms_per_track:.2f} ms")
                print(f"    Throughput:         {tracks_per_sec:.1f} tracks/sec")
                print(f"    Note: includes preprocessing + {n_eval_samples} posterior samples + stats")

                # Single-track inference benchmark (pure forward pass + sampling)
                print(f"\n  [SINGLE-TRACK BENCHMARK]")
                try:
                    flow_net = posterior.posterior_estimator
                    # Grab one precomputed observation
                    emb_net = flow_net.embedding_net
                    k_nb = emb_net.k
                    n_max_m = emb_net.n_max
                    x_test, _, _, n_max_t, knn_test = preprocess_egnn(
                        EVAL_CSV, k_neighbors=k_nb
                    )
                    if n_max_t < n_max_m:
                        pad = n_max_m - n_max_t
                        x_test = torch.cat([x_test, torch.zeros(x_test.shape[0], pad, 3)], dim=1)
                        knn_test = torch.cat([knn_test,
                            torch.full((knn_test.shape[0], pad, k_nb), -1,
                                       dtype=knn_test.dtype)], dim=1)
                        n_max_t = n_max_m
                    n_t = x_test.shape[0]
                    x_flat = torch.empty(n_t, n_max_t * (3 + k_nb), dtype=torch.float32)
                    x_flat[:, :n_max_t * 3] = x_test.view(n_t, -1)
                    for s in range(0, n_t, 10000):
                        e = min(s + 10000, n_t)
                        x_flat[s:e, n_max_t * 3:] = knn_test[s:e].float().reshape(e - s, -1)

                    single_obs = x_flat[:1].to(eval_device)
                    # Warmup
                    with torch.no_grad():
                        for _ in range(3):
                            flow_net.sample((n_eval_samples,), condition=single_obs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Benchmark: 50 runs
                    n_bench = 50
                    t_bench_start = time()
                    with torch.no_grad():
                        for _ in range(n_bench):
                            flow_net.sample((n_eval_samples,), condition=single_obs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_bench_end = time()

                    bench_ms = (t_bench_end - t_bench_start) / n_bench * 1000
                    print(f"    Single track ({n_eval_samples} samples): {bench_ms:.2f} ms")
                    print(f"    Single track (1 sample):   {bench_ms / n_eval_samples:.2f} ms")
                    del x_test, knn_test, x_flat, single_obs
                except Exception as e:
                    print(f"    Benchmark failed: {e}")

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
                # SBC: N/B rule of thumb requires N/B ≈ 20, where B = n_sbc_samples + 1
                # With n_diag=500 observations, n_sbc_samples=25 gives N/B = 500/26 ≈ 19
                n_sbc_samples = 50 if args.smoke else 25
                n_tarp_samples = 100 if args.smoke else 500  # TARP doesn't have the N/B constraint

                # --- SBC (Simulation-Based Calibration) ---
                print(f"  Running SBC on {n_diag} tracks, {n_sbc_samples} posterior samples each (N/B≈{n_diag/(n_sbc_samples+1):.0f})...")
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
                print(f"\n  Running TARP on {n_diag} tracks, {n_tarp_samples} posterior samples each...")
                try:
                    ecp, alpha = run_tarp(
                        theta_diag, x_diag, posterior,
                        num_posterior_samples=n_tarp_samples,
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
