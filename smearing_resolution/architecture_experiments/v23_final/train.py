"""v23_final: two-stage training for the v23 model.

Structure is v22's, preserved deliberately (train_gvp_egnn_v22.py):
  Stage 1  joint training of backbone + DirectionalPosterior + two AUXILIARY
           heads (DirectionHead -> vMF aux loss, EnergyHead -> Gaussian aux
           loss), with the sign-aware curriculum on the AUXILIARY direction loss
           only.  EMA weights are validated and checkpointed.
  Stage 2  the ENTIRE backbone is frozen and a FRESH, newly-initialized
           DirectionalPosterior is trained from scratch on Pool-A-only data
           (isotropic, no channeling enrichment) so the posterior's implicit
           prior stays uniform-on-S^2.  The deployed/evaluated model is Stage 2's
           checkpoint.

What v23 CHANGES (reports/v23_architecture_plan.tex):
  1. S{training}: dynamic continuous blur sampling is wired into BOTH stages'
     data pipelines.  v22's Stage 1 was called with the default smear_sigma=0.0
     and never overridden -- the backbone never saw a single blurred training
     example -- and Stage 2 trained on the same clean data.  Both now sample
     sigma from the P_ZERO / MIN_SIGMA_A / MAX_SIGMA_A continuous convention.
     Note these are ADDITIVE requirements, not substitutes: sigma-conditioning
     (model.py) tells the head how much to trust its input; it cannot manufacture
     structure the backbone never learned to see.
  2. S{calibration} item 1: the kappa regularizer, lambda * sum_k w_k kappa_k,
     added directly to the training loss.  Default LAMBDA_KAPPA=0.0 (inactive)
     per the staged build plan; verify_kappa_reg.py confirms it does not change
     forward-pass behaviour before it is turned on.
  3. S{calibration} item 2: calibration-aware checkpoint SELECTION.
  4. The corpus is the 1,009,384-track v2 pool, not v22's ~320k cap.

What v23 deliberately does NOT change:
  * AUX_SIGN_WARMUP's timing LOGIC (`epoch < 30` -> axis-aware, else signed) is
    byte-for-byte v22's.  S{loss} asks only that the resulting low-sigma exposure
    be VERIFIED EMPIRICALLY rather than assumed to transfer from the
    clean-data-only regime, so this script instruments and reports it.
  * Stage 2's auxiliary loss stays axis_aware throughout, as v22 has it
    (lines 1222, 1261).  S{loss} flags this for confirmation rather than change:
    it is a deliberate choice to keep Stage 2's auxiliary signal conservative
    since Stage 1 already transferred head/tail-capable features, and the Stage-2
    auxiliary head is a throwaway regularizer on a FROZEN backbone whose
    gradients cannot reach the backbone at all -- so its sign-awareness cannot
    affect the deployed posterior's head/tail capability either way.  Carried
    forward unchanged.

Usage
    python -m smearing_resolution.architecture_experiments.v23_final.train
    SMOKE=1 python -m ...v23_final.train           # tiny end-to-end shape check
    LAMBDA_KAPPA=0.002 SEED=1 python -m ...v23_final.train
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from itertools import permutations

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from src.models.egnn import DirectionHead, EnergyHead
from src.models.vmf_loss import axis_aware_vmf_nll, vmf_nll, gaussian_nll
from smearing_resolution.architecture_experiments.v23_final import config as C
from smearing_resolution.architecture_experiments.v23_final.data_pipeline import (
    N_PHYS_COND, TrackDataset, TrackPool, build_batch_serial, flat_dim,
    load_raw_tracks_v23, make_loader, pool_a_mask, sample_sigma_continuous,
)
from smearing_resolution.architecture_experiments.v23_final.metrics import (
    angular_coverage_ece,
)
from smearing_resolution.architecture_experiments.v23_final.model import (
    DirectionalPosteriorV23, build_v23,
)

if C.SMOKE:
    C.MAX_EPOCHS, C.STAGE2_EPOCHS = 2, 2
    C.SUB_EPOCH_STEPS = 8
    C.MAX_TRAIN_TRACKS = 800
    C.TRAIN_NROWS = 120_000
    C.VAL_ECE_TRACKS, C.VAL_ECE_SAMPLES = 64, 32
    C.PATIENCE, C.STAGE2_PATIENCE = 99, 99

DEVICE = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(C.RESULTS_DIR, exist_ok=True)
BEST_S1 = os.path.join(C.RESULTS_DIR, "best_checkpoint_stage1.pt")
BEST_S2_NLL = os.path.join(C.RESULTS_DIR, "best_checkpoint_stage2_nll.pt")
BEST_S2_SEL = os.path.join(C.RESULTS_DIR, "best_checkpoint_stage2.pt")
TRAIN_LOG = os.path.join(C.RESULTS_DIR, "training_log.csv")
METRICS_JSON = os.path.join(C.RESULTS_DIR, "metrics.json")
# [added post-build, ported from v22's --resume, which this build initially
# omitted] periodic "latest" checkpoints, separate from the best/selected
# ones above, saved every CHECKPOINT_EVERY_SEC regardless of whether that
# epoch improved anything -- the actual resume point.
LATEST_S1 = os.path.join(C.RESULTS_DIR, "checkpoint_latest_stage1.pt")
LATEST_S2 = os.path.join(C.RESULTS_DIR, "checkpoint_latest_stage2.pt")
RESUME = os.environ.get("RESUME", "1") == "1"


# ----------------------------------------------------------- Oh augmentation
def _build_oh_group():
    """[v22, VERBATIM] The 48 elements of Oh (diamond's point group Fd-3m):
    all signed permutation matrices, 3! x 2^3.  24 proper + 24 improper.

    SO(3) is NOT used: a generic rotation would map a channeling direction like
    <110> to a non-lattice direction and destroy the channeling signal.  The
    improper elements map the upper-hemisphere training samples onto the lower
    hemisphere, recovering full-S^2 coverage; diamond's inversion symmetry makes
    this exact.

    Every column of the phys block is Oh-invariant and must NOT be rotated:
    log n_vac, log R_g, the covariance-eigenvalue shares, the higher-moment
    cosines/ratios and sigma are invariant under any orthogonal map, and
    max|coordinate| is invariant specifically under SIGNED PERMUTATIONS (they
    permute the multiset of |coordinates| exactly) -- which is Oh, and is a
    further reason Oh rather than SO(3) is the right group here.  kNN indices are
    distance-based and likewise unchanged.
    """
    mats = []
    for perm in permutations([0, 1, 2]):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    M = torch.zeros(3, 3)
                    M[0, perm[0]] = sx
                    M[1, perm[1]] = sy
                    M[2, perm[2]] = sz
                    mats.append(M)
    return torch.stack(mats)


OH_GROUP_T = _build_oh_group()
assert OH_GROUP_T.shape == (48, 3, 3)


def apply_oh_augmentation(x_flat, theta_batch, n_max):
    B = x_flat.shape[0]
    idx = torch.randint(0, 48, (B,), device=x_flat.device)
    R = OH_GROUP_T.to(x_flat.device)[idx]
    coords = x_flat[:, :n_max * 3].view(B, n_max, 3)
    x_flat = x_flat.clone()
    x_flat[:, :n_max * 3] = torch.bmm(coords, R.transpose(-1, -2)).reshape(B, -1)
    theta_batch = theta_batch.clone()
    theta_batch[:, 1:4] = torch.bmm(theta_batch[:, 1:4].unsqueeze(1),
                                    R.transpose(-1, -2)).squeeze(1)
    return x_flat, theta_batch


# ---------------------------------------------------------------------- EMA
class EMA:
    """[v22, VERBATIM] Polyak averaging over requires_grad params only, so a
    frozen backbone (Stage 2) is left untouched.  We VALIDATE and CHECKPOINT on
    the averaged weights, so the deployed posterior uses them."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model):
        backup = {}
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                backup[n] = p.detach().clone()
                p.data.copy_(s)
        return backup

    @torch.no_grad()
    def swap_out(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self):
        return self.shadow


# ------------------------------------------------------------------ helpers
def cosine_lr_with_warmup(epoch, warmup_epochs, max_epochs, lr_max, lr_min):
    """[v22, VERBATIM]"""
    if epoch < warmup_epochs:
        return lr_min + (lr_max - lr_min) * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def aux_weight_schedule(epoch, max_epochs, start, end):
    """[v22, VERBATIM]"""
    progress = min(epoch / max(max_epochs - 1, 1), 1.0)
    cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
    return start + (end - start) * cosine_progress


def config_split(theta, val_fraction, seed):
    """[v22] Split BY CONFIG, not by track: the SIIMPL data has many ion
    realizations per (E, theta, phi) config and random per-track splitting would
    leak reps of the same config into val."""
    _, cid = np.unique(theta.numpy(), axis=0, return_inverse=True)
    cid = torch.from_numpy(cid.astype(np.int64))
    n_cfg = int(cid.max().item()) + 1
    g = torch.Generator().manual_seed(seed)
    val_ids = torch.randperm(n_cfg, generator=g)[:max(1, int(n_cfg * val_fraction))]
    vm = torch.isin(cid, val_ids)
    return (torch.where(~vm)[0].numpy(), torch.where(vm)[0].numpy(), n_cfg)


def build_fixed_val_set(pool, theta, val_idx, n_max, seed=12345):
    """Pre-build ONE fixed blurred validation set, with one sigma draw per val
    track from the same continuous distribution training uses.

    Fixed, not re-drawn per epoch, on purpose: validation NLL and validation ECE
    are the checkpoint-SELECTION criteria, and a fresh blur draw each epoch would
    add per-epoch noise to exactly the quantity being compared across epochs.
    The draw covers the whole sigma range, so it is not a zero-blur-only metric.
    """
    rng = np.random.default_rng(seed)
    n = len(val_idx)
    sigmas = sample_sigma_continuous(n, rng)
    batches = []
    for s in range(0, n, C.BATCH_SIZE):
        sel = val_idx[s:s + C.BATCH_SIZE]
        th, x, _ = build_batch_serial(pool, theta, sel, n_max, 'cpu', rng,
                                      C.H0, C.K_MIN, C.K_MAX,
                                      sigmas=sigmas[s:s + C.BATCH_SIZE])
        batches.append((th, x))
    frac0 = float((sigmas == 0).mean())
    print(f"[VAL] fixed val set: {n:,} tracks in {len(batches)} batches "
          f"(sigma=0 fraction {frac0:.2f}, median nonzero sigma "
          f"{np.median(sigmas[sigmas > 0]) / 10 if (sigmas > 0).any() else 0:.1f}nm)")
    return batches


def compute_phys_stats(pool, theta, train_idx, n_max, n_batches=40, seed=777):
    """Physics-descriptor standardization statistics.

    v22 computed these once from the static, UNBLURRED preprocess_egnn output.
    v23's phys block is blur-DEPENDENT (the deconvolved normalizer, and the two
    higher-moment cosines/ratios all move with sigma), so the statistics must be
    taken under the training blur distribution or the standardization would be
    badly off-centre for most of the training data.  Sampled from the same
    dynamic pipeline the training loader uses.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_batches):
        sel = rng.choice(train_idx, size=min(C.BATCH_SIZE, len(train_idx)),
                         replace=False)
        _th, x, _s = build_batch_serial(pool, theta, sel, n_max, 'cpu', rng,
                                        C.H0, C.K_MIN, C.K_MAX)
        rows.append(x[:, -C.N_PHYS:][:, :N_PHYS_COND].clone())
    P = torch.cat(rows, 0)
    mean, std = P.mean(0), P.std(0).clamp(min=1e-8)
    names = ['log_nvac', 'log_extent', 'log_Rg', 'elong1', 'elong2',
             'cos_corr', 'cos_uncorr', 'ratio_corr', 'ratio_uncorr']
    print(f"[PHYS] standardization stats from {len(P):,} dynamically-blurred "
          f"samples:")
    for j, nm in enumerate(names):
        print(f"        {nm:13s}: {mean[j]:+.4f} +- {std[j]:.4f}")
    return mean, std


@torch.no_grad()
def calibrate_lambda_kappa(model, val_batches, frac=None, max_batches=8):
    """S{calibration} item 1's own tuning rule, implemented literally: "start
    small (e.g. tuned so L_calib is ~1% of the direction NLL at initialization)".

    Measures the mean DIRECTION NLL (not the total, which also carries the energy
    GMM term) and the mean kappa penalty on real batches at initialization, then
    solves lambda = frac * mean_direction_NLL / mean_kappa_penalty.
    """
    frac = C.LAMBDA_KAPPA_TARGET_FRAC if frac is None else frac
    model.eval()
    dn, kp, n = 0.0, 0.0, 0
    for th_cpu, x_cpu in val_batches[:max_batches]:
        th = th_cpu.to(DEVICE)
        x = x_cpu.to(DEVICE)
        d = F.normalize(th[:, 1:4], dim=-1, eps=1e-8)
        p = model.params(x)
        lp = model.direction.log_prob_from_params(p['mu'], p['kappa'],
                                                 p['logits'], d)
        dn += float((-lp).mean())
        kp += float(model.direction.kappa_penalty(p['kappa'],
                                                  p['logits']).mean())
        n += 1
    model.train()
    dn, kp = dn / max(n, 1), kp / max(n, 1)
    lam = frac * dn / max(kp, 1e-8)
    print(f"[LAMBDA] auto-tuned kappa regularizer: mean direction NLL at init "
          f"= {dn:.4f}, mean kappa penalty = {kp:.4f}")
    print(f"[LAMBDA] lambda = {frac:.3f} * {dn:.4f} / {kp:.4f} = {lam:.6f}   "
          f"(L_calib = {100*frac:.1f}% of the direction NLL at init)")
    return lam


@torch.no_grad()
def validate(model, dir_head, energy_head, val_batches, epoch, max_epochs,
             aux_axis_aware, ece_batches, ece_samples):
    """Validation NLL (the selection metric v22 used) plus angular-coverage ECE
    (the S{calibration} item-2 addition).  Returns a dict."""
    model.eval()
    dir_head.eval()
    energy_head.eval()
    alpha = aux_weight_schedule(epoch, max_epochs, C.ALPHA_START, C.ALPHA_END)
    beta = aux_weight_schedule(epoch, max_epochs, C.BETA_START, C.BETA_END)
    sums = dict(total=0.0, nll=0.0, direction=0.0, energy=0.0, kappa=0.0)
    n = 0
    for th_cpu, x_cpu in val_batches:
        th = th_cpu.to(DEVICE)
        x = x_cpu.to(DEVICE)
        th = th.clone()
        if C.LOG_ENERGY:
            th[:, 0] = torch.log(th[:, 0].clamp(min=1e-3))
        nll, kpen = model.loss_terms(th, x)
        nll = nll.clamp(max=C.LOSS_CLAMP)
        z = model.embedding_net.last_z
        mu_hat, kappa = dir_head(z)
        dn = (axis_aware_vmf_nll if aux_axis_aware else vmf_nll)(
            mu_hat, kappa, th[:, 1:4]).clamp(max=C.LOSS_CLAMP)
        ep, ls = energy_head(z)
        en = gaussian_nll(ep, ls, th[:, 0]).clamp(max=C.LOSS_CLAMP)
        tot = nll.mean() + alpha * dn.mean() + beta * en.mean()
        if not torch.isfinite(tot):
            continue
        sums['total'] += float(tot)
        sums['nll'] += float(nll.mean())
        sums['direction'] += float(dn.mean())
        sums['energy'] += float(en.mean())
        sums['kappa'] += float(kpen.mean())
        n += 1
    out = {k: (v / max(n, 1)) for k, v in sums.items()}

    # ---- angular-coverage ECE on a fixed subset of the same val batches
    samples, targets = [], []
    seen = 0
    for th_cpu, x_cpu in ece_batches:
        x = x_cpu.to(DEVICE)
        pr = model.predict(x, n_samples=ece_samples)
        samples.append(pr['samples'].cpu())
        targets.append(F.normalize(th_cpu[:, 1:4], dim=-1))
        seen += x.shape[0]
    if samples:
        ece, cov = angular_coverage_ece(torch.cat(samples, dim=1),
                                        torch.cat(targets, dim=0))
        out['ece'] = ece
        out['coverage'] = cov.tolist()
        out['ece_n'] = seen
    else:
        out['ece'] = float('nan')
        out['coverage'] = []
        out['ece_n'] = 0
    model.train()
    dir_head.train()
    energy_head.train()
    return out


class CalibrationAwareSelector:
    """S{calibration} item 2: select the checkpoint minimizing validation ECE
    among those within ECE_NLL_TOL of the best validation NLL.

    This is NOT a recalibration map: no additional trainable parameters are fit
    on validation or eval data and no transformation is applied to the model's
    outputs.  It only changes WHICH already-trained epoch's weights are kept,
    using a metric measured on data the model did not train on -- standard model
    selection, explicitly distinct from the post-hoc temperature scaling the
    spec's Goal 2 rules out.

    Rule, stated precisely (the spec gives the criterion, not the bookkeeping):
      * track the running best validation NLL;
      * an epoch QUALIFIES if val_nll <= best_nll + ECE_NLL_TOL;
      * among qualifying epochs keep the lowest val_ece;
      * if a later epoch improves best_nll by MORE than ECE_NLL_TOL, every
        previously qualifying epoch has now dropped out of the tolerance band,
        so the ECE competition is RESET rather than left holding a stale winner.
    Every epoch's (nll, ece) pair is also written to training_log.csv, so the
    selection is auditable after the fact and can be recomputed offline.
    """

    def __init__(self, tol, enabled=True):
        self.tol = tol
        self.enabled = enabled
        self.best_nll = float('inf')
        self.best_ece = float('inf')
        self.selected_epoch = None

    def offer(self, epoch, nll, ece):
        """Returns (improved_nll, select_now)."""
        improved_nll = math.isfinite(nll) and nll < self.best_nll
        if improved_nll and nll < self.best_nll - self.tol:
            self.best_ece = float('inf')          # tolerance band moved: reset
        if improved_nll:
            self.best_nll = nll
        if not self.enabled:
            return improved_nll, improved_nll
        qualifies = math.isfinite(nll) and nll <= self.best_nll + self.tol
        select = qualifies and math.isfinite(ece) and ece < self.best_ece
        if select:
            self.best_ece = ece
            self.selected_epoch = epoch
        return improved_nll, select


def save_ckpt(path, model, ema, dir_head, energy_head, optimizer, epoch,
              val_metrics, phys_mean, phys_std, n_max, stage, extra=None):
    """Save EMA weights (the model we deploy) plus everything needed to rebuild
    the model from scratch.

    Deliberately stores STATE DICTS and a plain config dict only -- never the
    pickled module object.  v22 stored `'flow_module': flow`, which makes the
    checkpoint unloadable whenever a class it references moves or changes, a
    failure this project has hit before.  eval.py reconstructs via build_v23().
    """
    backup = ema.swap_in(model) if ema is not None else None
    try:
        torch.save({
            'stage': stage,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'dir_head_state_dict': dir_head.state_dict(),
            'energy_head_state_dict': energy_head.state_dict(),
            'optimizer_state_dict': (optimizer.state_dict() if optimizer
                                     else None),
            'val_metrics': val_metrics,
            'phys_mean': phys_mean.cpu(),
            'phys_std': phys_std.cpu(),
            'n_max': n_max,
            'cfg': dict(n_max=n_max, hidden_dim=C.HIDDEN_DIM,
                        n_layers=C.N_LAYERS, k=C.K_MAX,
                        d_cond=C.D_AUG, n_dir_comp=C.N_DIR_COMP,
                        n_e_comp=C.N_E_COMP, hidden=C.HEAD_HIDDEN),
            'env': C.summary(),
            **(extra or {}),
        }, path)
    finally:
        if backup is not None:
            ema.swap_out(model, backup)


# ============================================================ training stage
def run_stage(stage, model, dir_head, energy_head, pool, theta, train_idx,
              val_batches, ece_batches, n_max, phys_mean, phys_std,
              max_epochs, lr_max, lr_min, warmup, patience, grad_clip,
              aux_always_axis_aware, best_path_nll, best_path_sel, log_rows,
              t_global, latest_path=None, resume=True):
    """One training stage.  Both stages draw from the SAME dynamic-blur pipeline
    (S{training}); they differ in the index pool (Stage 2 = Pool A only), what is
    trainable (Stage 2 = frozen backbone, fresh head), and the aux curriculum.
    """
    steps_per_epoch = (C.SUB_EPOCH_STEPS if C.SUB_EPOCH_STEPS > 0
                       else max(1, len(train_idx) // C.BATCH_SIZE))
    full_pass = max(1, len(train_idx) // C.BATCH_SIZE)
    total_steps = steps_per_epoch * max_epochs
    print(f"\n[STAGE {stage}] {len(train_idx):,} train tracks | "
          f"{steps_per_epoch} steps/scheduler-epoch | {full_pass} steps per full "
          f"pass | {max_epochs} epochs = {total_steps:,} steps "
          f"({total_steps / full_pass:.2f} full passes)")

    params = ([p for p in model.parameters() if p.requires_grad]
              + list(dir_head.parameters()) + list(energy_head.parameters()))
    print(f"[STAGE {stage}] trainable params: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=lr_max,
                                  weight_decay=C.WEIGHT_DECAY)

    selector = CalibrationAwareSelector(C.ECE_NLL_TOL, C.SELECT_ON_ECE)
    epochs_no_improve = 0
    # S{loss} empirical verification instrumentation (does NOT change behaviour)
    n_seen = n_seen_zero = n_seen_low = 0
    budget = C.TIME_BUDGET_HOURS * 3600 if C.TIME_BUDGET_HOURS > 0 else None
    stop_reason = "max_epochs"
    start_epoch = 0

    # ---- resume from the periodic "latest" checkpoint, if one exists ----
    # [added post-build] model_state_dict in every checkpoint this project saves
    # is the EMA-swapped-in snapshot (see save_ckpt's docstring), not raw
    # training weights -- so resuming loads that snapshot as the new starting
    # point for BOTH the live model and a freshly-seeded EMA shadow. This is a
    # deliberate simplification (continue from the smoothed point, not a
    # bit-exact mid-epoch resume) rather than changing the checkpoint format,
    # which would also require updating eval.py's loader.
    if resume and latest_path and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        if ck.get('stage_done'):
            print(f"[STAGE {stage}] [RESUME] {latest_path} is already marked "
                  f"complete (stop_reason={ck.get('stop_reason')}) -- skipping "
                  f"straight to the stage result, no training this run.")
            return dict(best_nll=ck['best_nll'], best_ece=ck['best_ece'],
                       selected_epoch=ck.get('selected_epoch'),
                       stop_reason=ck.get('stop_reason', 'resumed_done'),
                       n_seen=ck.get('n_seen', 0),
                       n_seen_sigma0=ck.get('n_seen_sigma0', 0),
                       n_seen_sigma_lt_3nm=ck.get('n_seen_sigma_lt_3nm', 0))
        model.load_state_dict(ck['model_state_dict'])
        dir_head.load_state_dict(ck['dir_head_state_dict'])
        energy_head.load_state_dict(ck['energy_head_state_dict'])
        if ck.get('optimizer_state_dict'):
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        selector.best_nll = ck.get('selector_best_nll', selector.best_nll)
        selector.best_ece = ck.get('selector_best_ece', selector.best_ece)
        selector.selected_epoch = ck.get('selector_selected_epoch')
        epochs_no_improve = ck.get('epochs_no_improve', 0)
        n_seen = ck.get('n_seen', 0)
        n_seen_zero = ck.get('n_seen_sigma0', 0)
        n_seen_low = ck.get('n_seen_sigma_lt_3nm', 0)
        print(f"[STAGE {stage}] [RESUME] {latest_path}: continuing from epoch "
              f"{start_epoch}/{max_epochs} (best val NLL so far "
              f"{selector.best_nll:.4f})")

    ema = EMA(model, C.EMA_DECAY) if C.USE_EMA else None

    ds = TrackDataset(pool=pool, theta=theta, idx_arr=np.asarray(train_idx),
                     steps_per_epoch=steps_per_epoch, batch_size=C.BATCH_SIZE,
                     n_max=n_max, start_step=start_epoch * steps_per_epoch,
                     total_steps=total_steps,
                     seed=C.SEED + stage, h0=C.H0, k_min=C.K_MIN, k_max=C.K_MAX,
                     p_zero=C.P_ZERO, min_sigma_A=C.MIN_SIGMA_A,
                     max_sigma_A=C.MAX_SIGMA_A)
    loader = make_loader(ds, C.BATCH_SIZE, C.N_WORKERS, C.PREFETCH, n_max,
                         C.K_MAX, pin_memory=(DEVICE == 'cuda'))
    it = iter(loader)
    last_ckpt_t = time.time()

    for epoch in range(start_epoch, max_epochs):
        t_ep = time.time()
        lr = cosine_lr_with_warmup(epoch, warmup, max_epochs, lr_max, lr_min)
        for g in optimizer.param_groups:
            g['lr'] = lr
        alpha = aux_weight_schedule(epoch, max_epochs, C.ALPHA_START, C.ALPHA_END)
        beta = aux_weight_schedule(epoch, max_epochs, C.BETA_START, C.BETA_END)
        # [v22, VERBATIM LOGIC] sign-aware curriculum on the AUXILIARY loss only
        axis_aware = aux_always_axis_aware or (epoch < C.AUX_SIGN_WARMUP)

        model.train()
        dir_head.train()
        energy_head.train()
        sums = dict(total=0.0, nll=0.0, direction=0.0, energy=0.0, kappa=0.0)
        nb = 0
        nan_strikes = 0
        wait_s = 0.0
        for _ in range(steps_per_epoch):
            t_w = time.time()
            try:
                th_cpu, x_cpu, _step = next(it)
            except StopIteration:
                stop_reason = "loader_exhausted"
                break
            wait_s += time.time() - t_w
            th = th_cpu.to(DEVICE, non_blocking=True)
            x = x_cpu.to(DEVICE, non_blocking=True)
            sig_A = x[:, -C.N_PHYS:][:, -2]
            n_seen += x.shape[0]
            n_seen_zero += int((sig_A == 0).sum())
            n_seen_low += int((sig_A < 30.0).sum())     # < 3 nm
            x, th = apply_oh_augmentation(x, th, n_max)
            if C.LOG_ENERGY:
                th[:, 0] = torch.log(th[:, 0].clamp(min=1e-3))

            optimizer.zero_grad(set_to_none=True)
            nll, kpen = model.loss_terms(th, x)
            nll = nll.clamp(max=C.LOSS_CLAMP)
            nll_loss = nll.mean()
            z = model.embedding_net.last_z
            mu_hat, kappa = dir_head(z)
            dn = (axis_aware_vmf_nll if axis_aware else vmf_nll)(
                mu_hat, kappa, th[:, 1:4]).clamp(max=C.LOSS_CLAMP)
            ep_pred, ls = energy_head(z)
            en = gaussian_nll(ep_pred, ls, th[:, 0]).clamp(max=C.LOSS_CLAMP)
            calib = C.LAMBDA_KAPPA * kpen.mean()
            total = nll_loss + alpha * dn.mean() + beta * en.mean() + calib

            if not torch.isfinite(total):
                nan_strikes += 1
                if nan_strikes <= 3:
                    print(f"  [WARN] non-finite loss (epoch {epoch}, strike "
                          f"{nan_strikes}): nll={float(nll_loss)} "
                          f"dir={float(dn.mean())} E={float(en.mean())}")
                if nan_strikes > 50:
                    raise RuntimeError("too many non-finite losses; aborting")
                continue
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if not torch.isfinite(gnorm):
                nan_strikes += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            if ema is not None:
                ema.update(model)
            sums['total'] += float(total)
            sums['nll'] += float(nll_loss)
            sums['direction'] += float(dn.mean())
            sums['energy'] += float(en.mean())
            sums['kappa'] += float(kpen.mean())
            nb += 1

        tr = {k: v / max(nb, 1) for k, v in sums.items()}

        # ---- validate on EMA weights (the model we deploy)
        backup = ema.swap_in(model) if ema is not None else None
        try:
            va = validate(model, dir_head, energy_head, val_batches, epoch,
                          max_epochs, axis_aware, ece_batches, C.VAL_ECE_SAMPLES)
        finally:
            if backup is not None:
                ema.swap_out(model, backup)

        ep_sec = time.time() - t_ep
        print(f"  S{stage} ep {epoch:3d}/{max_epochs} | "
              f"train nll={tr['nll']:.4f} dir={tr['direction']:.3f} "
              f"E={tr['energy']:.3f} kap={tr['kappa']:.2f} | "
              f"val nll={va['nll']:.4f} ECE={100*va['ece']:.2f}% | "
              f"lr={lr:.2e} | {ep_sec:.0f}s "
              f"(loader-wait {100*wait_s/max(ep_sec,1e-9):.0f}%)"
              + (f" | {nan_strikes} nan-skips" if nan_strikes else ""), flush=True)

        improved_nll, select = selector.offer(epoch, va['nll'], va['ece'])
        if improved_nll:
            epochs_no_improve = 0
            save_ckpt(best_path_nll, model, ema, dir_head, energy_head,
                      optimizer, epoch, va, phys_mean, phys_std, n_max, stage)
            print(f"    -> best val NLL {va['nll']:.4f} (saved {os.path.basename(best_path_nll)})")
        else:
            epochs_no_improve += 1
        if select and best_path_sel:
            save_ckpt(best_path_sel, model, ema, dir_head, energy_head,
                      optimizer, epoch, va, phys_mean, phys_std, n_max, stage,
                      extra=dict(selection='min-ECE within NLL tolerance',
                                 selection_tol=C.ECE_NLL_TOL,
                                 selection_best_nll=selector.best_nll))
            print(f"    -> SELECTED epoch {epoch}: ECE {100*va['ece']:.2f}% "
                  f"at val NLL {va['nll']:.4f} (within {C.ECE_NLL_TOL} of best "
                  f"{selector.best_nll:.4f})")

        log_rows.append(dict(
            stage=stage, epoch=epoch, lr=lr, alpha=alpha, beta=beta,
            aux_axis_aware=int(axis_aware), lambda_kappa=C.LAMBDA_KAPPA,
            train_nll=tr['nll'], train_dir=tr['direction'],
            train_energy=tr['energy'], train_kappa=tr['kappa'],
            val_nll=va['nll'], val_dir=va['direction'],
            val_energy=va['energy'], val_kappa=va['kappa'], val_ece=va['ece'],
            n_seen=n_seen, n_seen_sigma0=n_seen_zero, n_seen_sigma_lt_3nm=n_seen_low,
            epoch_sec=ep_sec, elapsed_min=(time.time() - t_global) / 60))
        with open(TRAIN_LOG, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

        # ---- periodic "latest" checkpoint, THE actual resume point -- saved
        # every CHECKPOINT_EVERY_SEC regardless of whether this epoch improved
        # anything, so a disconnect loses at most one interval's progress.
        if latest_path and (time.time() - last_ckpt_t >= C.CHECKPOINT_EVERY_SEC):
            save_ckpt(latest_path, model, ema, dir_head, energy_head,
                     optimizer, epoch, va, phys_mean, phys_std, n_max, stage,
                     extra=dict(stage_done=False,
                                selector_best_nll=selector.best_nll,
                                selector_best_ece=selector.best_ece,
                                selector_selected_epoch=selector.selected_epoch,
                                epochs_no_improve=epochs_no_improve,
                                n_seen=n_seen, n_seen_sigma0=n_seen_zero,
                                n_seen_sigma_lt_3nm=n_seen_low))
            last_ckpt_t = time.time()
            print(f"    -> [RESUME POINT] saved {os.path.basename(latest_path)}")

        # ---- S{loss} empirical verification, reported at the curriculum switch
        if stage == 1 and epoch == C.AUX_SIGN_WARMUP - 1:
            # v22's equivalent exposure: every clean epoch was 100% sigma=0, so
            # AUX_SIGN_WARMUP epochs x steps/epoch x batch clean examples.
            v22_equiv = C.AUX_SIGN_WARMUP * steps_per_epoch * C.BATCH_SIZE
            print(f"\n  [S{{loss}} CHECK] curriculum switches to the SIGNED aux "
                  f"loss after this epoch.")
            print(f"    examples seen so far        : {n_seen:,}")
            print(f"    at sigma = 0 exactly        : {n_seen_zero:,} "
                  f"({100*n_seen_zero/max(n_seen,1):.1f}%)")
            print(f"    at sigma < 3 nm             : {n_seen_low:,} "
                  f"({100*n_seen_low/max(n_seen,1):.1f}%)")
            print(f"    v22's clean-only equivalent : {v22_equiv:,}")
            print(f"    low-sigma exposure vs v22   : "
                  f"{n_seen_low/max(v22_equiv,1):.2f}x")
            if n_seen_zero < len(train_idx):
                print(f"    [WARN] fewer sigma=0 examples ({n_seen_zero:,}) than "
                      f"training tracks ({len(train_idx):,}): the head/tail-"
                      f"informative low-blur regime may be under-exposed by the "
                      f"switch. AUX_SIGN_WARMUP is left UNCHANGED per S{{loss}}; "
                      f"this is the empirical datum the spec asks for.\n")
            else:
                print(f"    OK: every training track has been seen at sigma=0 "
                      f"on average {n_seen_zero/len(train_idx):.1f} times.\n")

        if epochs_no_improve >= patience:
            stop_reason = f"early_stop(patience={patience})"
            print(f"  [STOP] no val-NLL improvement for {patience} epochs.")
            break
        if budget is not None and time.time() - t_global >= budget:
            stop_reason = "time_budget"
            print(f"  [STOP] wall-clock budget ({C.TIME_BUDGET_HOURS}h) reached.")
            break
        if stop_reason == "loader_exhausted":
            print("  [STOP] loader exhausted.")
            break

    del loader
    print(f"[STAGE {stage}] done: {stop_reason}; best val NLL "
          f"{selector.best_nll:.4f}"
          + (f", selected epoch {selector.selected_epoch} "
             f"(ECE {100*selector.best_ece:.2f}%)"
             if selector.selected_epoch is not None else ""))
    result = dict(best_nll=selector.best_nll, best_ece=selector.best_ece,
                 selected_epoch=selector.selected_epoch,
                 stop_reason=stop_reason, n_seen=n_seen,
                 n_seen_sigma0=n_seen_zero, n_seen_sigma_lt_3nm=n_seen_low)
    # mark this stage complete on the resume point too, so a later resume
    # attempt (e.g. Stage 2 disconnects, Stage 1 is untouched) skips Stage 1
    # entirely instead of re-training it (see the stage_done check above).
    if latest_path:
        save_ckpt(latest_path, model, ema, dir_head, energy_head,
                 optimizer, epoch, dict(nll=selector.best_nll, ece=selector.best_ece,
                                        direction=float('nan'), energy=float('nan'),
                                        kappa=float('nan')),
                 phys_mean, phys_std, n_max, stage,
                 extra=dict(stage_done=True, **result))
    return result


# ===================================================================== main
def main():
    import random
    t0 = time.time()
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    random.seed(C.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.SEED)
    print(f"[CONFIG] {json.dumps(C.summary(), indent=None)}")
    print(f"[DEVICE] {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))
    print(f"[RESULTS] {C.RESULTS_DIR}")

    # ------------------------------------------------------------ data
    raw, theta, ions, n_max_obs, _mp = load_raw_tracks_v23(
        C.TRAIN_CSV, C.MAX_POINTS, nrows=C.TRAIN_NROWS)
    n_max = C.MAX_POINTS
    print(f"[NMAX] forced n_max={n_max} (observed {n_max_obs}) -> train and eval "
          f"share one padding width")
    is_pool_a = pool_a_mask(ions, expect_full_corpus=(C.TRAIN_NROWS == 0))
    train_idx, val_idx, n_cfg = config_split(theta, C.VAL_FRACTION, C.SEED)
    if C.MAX_TRAIN_TRACKS:
        train_idx = train_idx[:C.MAX_TRAIN_TRACKS]
        val_idx = val_idx[:max(C.BATCH_SIZE, C.MAX_TRAIN_TRACKS // 10)]
    train_idx_a = train_idx[is_pool_a[train_idx]]
    print(f"[SPLIT] {n_cfg:,} configs -> {len(train_idx):,} train / "
          f"{len(val_idx):,} val tracks; Stage 2 Pool-A train subset: "
          f"{len(train_idx_a):,}")
    pool = TrackPool(raw)
    del raw

    phys_mean, phys_std = compute_phys_stats(pool, theta, train_idx, n_max)
    val_batches = build_fixed_val_set(pool, theta, val_idx, n_max)
    n_ece_b = max(1, C.VAL_ECE_TRACKS // C.BATCH_SIZE)
    ece_batches = val_batches[:n_ece_b]

    # --------------------------------------------------------- Stage 1
    model = build_v23(n_max=n_max, phys_mean=phys_mean, phys_std=phys_std,
                      device=DEVICE)
    dir_head = DirectionHead(d_latent=C.D_AUG).to(DEVICE)
    energy_head = EnergyHead(d_latent=C.D_AUG, log_energy=C.LOG_ENERGY).to(DEVICE)
    print(f"[MODEL] total params {sum(p.numel() for p in model.parameters()):,} "
          f"(backbone "
          f"{sum(p.numel() for p in model.embedding_net.base.parameters()):,})")

    if C.LAMBDA_KAPPA_AUTO:
        C.LAMBDA_KAPPA = calibrate_lambda_kappa(model, val_batches)

    log_rows = []
    s1 = run_stage(1, model, dir_head, energy_head, pool, theta, train_idx,
                   val_batches, ece_batches, n_max, phys_mean, phys_std,
                   C.MAX_EPOCHS, C.LR_MAX, C.LR_MIN, C.WARMUP_EPOCHS,
                   C.PATIENCE, C.GRAD_CLIP, aux_always_axis_aware=False,
                   best_path_nll=BEST_S1, best_path_sel=None,
                   log_rows=log_rows, t_global=t0,
                   latest_path=LATEST_S1, resume=RESUME)

    # --------------------------------------------------------- Stage 2
    print(f"\n{'='*70}\n  STAGE 2: frozen backbone, FRESH posterior, Pool A only"
          f"\n{'='*70}")
    ck = torch.load(BEST_S1, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f"  loaded Stage-1 best (epoch {ck['epoch']}, "
          f"val NLL {ck['val_metrics']['nll']:.4f})")
    del ck

    for p in model.embedding_net.base.parameters():
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.embedding_net.base.parameters())
    print(f"  backbone frozen ({n_frozen:,} params)")

    # a FRESH, newly-initialized posterior on the SAME (now frozen) embedding --
    # v22's Stage 2 does exactly this, and the zero-init discipline applies again
    # to the fresh head, so Stage 2 also starts mathematically identical to PCA.
    model_s2 = DirectionalPosteriorV23(model.embedding_net).to(DEVICE)
    dir_head_s2 = DirectionHead(d_latent=C.D_AUG).to(DEVICE)
    energy_head_s2 = EnergyHead(d_latent=C.D_AUG,
                                log_energy=C.LOG_ENERGY).to(DEVICE)

    s2 = run_stage(2, model_s2, dir_head_s2, energy_head_s2, pool, theta,
                   train_idx_a, val_batches, ece_batches, n_max, phys_mean,
                   phys_std, C.STAGE2_EPOCHS, C.STAGE2_LR_MAX, C.STAGE2_LR_MIN,
                   C.STAGE2_WARMUP, C.STAGE2_PATIENCE, C.STAGE2_GRAD_CLIP,
                   aux_always_axis_aware=True,
                   best_path_nll=BEST_S2_NLL, best_path_sel=BEST_S2_SEL,
                   log_rows=log_rows, t_global=t0,
                   latest_path=LATEST_S2, resume=RESUME)

    if not os.path.exists(BEST_S2_SEL):
        print("[WARN] no calibration-selected Stage-2 checkpoint was written; "
              "falling back to the best-NLL checkpoint for deployment.")
        import shutil
        shutil.copyfile(BEST_S2_NLL, BEST_S2_SEL)

    with open(METRICS_JSON, 'w') as f:
        json.dump(dict(env=C.summary(), stage1=s1, stage2=s2,
                       total_min=(time.time() - t0) / 60,
                       deployed_checkpoint=BEST_S2_SEL), f, indent=2)
    print(f"\n{'='*70}")
    print(f"  Training complete in {(time.time()-t0)/60:.1f} min")
    print(f"  Stage 1 best val NLL : {s1['best_nll']:.4f}")
    print(f"  Stage 2 best val NLL : {s2['best_nll']:.4f}")
    print(f"  Stage 2 SELECTED     : epoch {s2['selected_epoch']} "
          f"(val ECE {100*s2['best_ece']:.2f}%)")
    print(f"  DEPLOY               : {BEST_S2_SEL}")
    print(f"  Next: python -m ...v23_final.eval  with CKPT={BEST_S2_SEL}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
