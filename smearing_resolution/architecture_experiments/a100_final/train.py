"""a100_final: training script for DiagModelV2.

Loss structure is the winner's: sign-blind (axis) direction NLL + energy NLL +
sign BCE against the geometric v_ref anchor.  What changed is documented in
README.md.  Everything is env-configurable via config.py.

  CONFIG=small python train.py
  CONFIG=large TARGET_STEPS=30000 python train.py
  SMOKE=1 python train.py          # ~12-step shape/NaN correctness check
"""
import sys, os, time, json, csv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from src.models.vmf_loss import gaussian_nll
from smearing_resolution.architecture_experiments.a100_final import config as C
from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    load_raw_tracks, flat_dim, build_eval_batch, TrackPool,
)
from smearing_resolution.architecture_experiments.a100_final.loader import (
    TrackDataset, make_loader as _make_loader,
)
from smearing_resolution.architecture_experiments.a100_final.model import DiagModelV2

if C.SMOKE:
    C.TARGET_STEPS, C.TIME_BUDGET_HOURS = 12, 0.2
    C.EVAL_N_PER_BIN, C.MAX_TRAIN_TRACKS = 20, 400
    C.CHECKPOINT_EVERY_SEC = 1e9

os.makedirs(C.RESULTS_DIR, exist_ok=True)
torch.manual_seed(C.SEED)
device = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"[CONFIG] {json.dumps(C.summary())}")
print(f"[DEVICE] {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))


# ------------------------------------------------------------ Oh augmentation
def _build_oh_group():
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1.0, -1.0], repeat=3):
            M = torch.zeros(3, 3)
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            mats.append(M)
    return torch.stack(mats)


OH_GROUP_T = _build_oh_group()


def apply_oh_augmentation(x_flat, theta_batch, n_max):
    """Rotate ONLY the coordinate block; the kNN index block and the phys block
    are rotation-invariant by construction and must be left alone."""
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


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

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


def true_dir(theta_batch):
    return F.normalize(theta_batch[:, 1:4], dim=-1)


def config_split(theta, val_fraction=0.02, seed=42):
    theta_np = theta.numpy()
    _, cid = np.unique(theta_np, axis=0, return_inverse=True)
    cid = torch.from_numpy(cid.astype(np.int64))
    n_cfg = int(cid.max().item()) + 1
    g = torch.Generator().manual_seed(seed)
    val_ids = torch.randperm(n_cfg, generator=g)[:max(1, int(n_cfg * val_fraction))]
    vm = torch.isin(cid, val_ids)
    return torch.where(~vm)[0].tolist(), torch.where(vm)[0].tolist()


# ---------------------------------------------------------------- data load
print("[LOAD] train tracks")
raw_train, theta_train, _n1, _ = load_raw_tracks(C.TRAIN_CSV, max_points=C.MAX_POINTS)
print("[LOAD] eval tracks")
raw_eval, theta_eval, _n2, _ = load_raw_tracks(C.EVAL_CSV, max_points=C.MAX_POINTS)
# force a single padding width so train and eval share one tensor layout
N_MAX = C.MAX_POINTS
assert _n1 <= N_MAX and _n2 <= N_MAX, (_n1, _n2, N_MAX)
print(f"[NMAX] forced n_max={N_MAX} (observed train={_n1}, eval={_n2}) -> "
      f"no truncation")

train_idx, val_idx = config_split(theta_train)
if C.MAX_TRAIN_TRACKS:
    train_idx = train_idx[:C.MAX_TRAIN_TRACKS]
print(f"[SPLIT] {len(train_idx)} train / {len(val_idx)} val")
# collapse the 293k-element python list into 2 numpy objects so fork-based
# DataLoader workers share it copy-on-write instead of copying refcounts
track_pool = TrackPool(raw_train)
del raw_train
energy_np_eval = theta_eval[:, 0].numpy()
FLAT_DIM = flat_dim(N_MAX, C.K_MAX)

# ------------------------------------------------------------------- model
model = DiagModelV2(n_max=N_MAX, k_max=C.K_MAX, hidden_dim=C.HIDDEN_DIM,
                    n_layers=C.N_LAYERS, coef_hidden=C.COEF_HIDDEN,
                    n_dir_comp=C.N_DIR_COMP, posterior_type=C.POSTERIOR_TYPE,
                    grid_subdiv=C.GRID_SUBDIV,
                    amp=bool(C.AMP) and device == 'cuda').to(device)
print(f"[MODEL] {sum(p.numel() for p in model.parameters()):,} trainable params "
      f"({C.POSTERIOR_TYPE})")
optimizer = torch.optim.Adam(model.parameters(), lr=C.LR_MAX)
ema = EMA(model, C.EMA_DECAY)

CKPT_PATH = os.path.join(C.RESULTS_DIR, "checkpoint_latest.pt")
start_step = 0
if os.environ.get("RESUME", "1") == "1" and os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    start_step = ck.get('step', 0)
    if ck.get('ema'):
        ema.shadow = {k: v.to(device) for k, v in ck['ema'].items()}
    print(f"[RESUME] {CKPT_PATH} @ step {start_step}")


def save_ckpt(path, step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'ema': ema.shadow, 'n_max': N_MAX,
                'cfg': model.cfg, 'config_name': C.CONFIG,
                'env': C.summary()}, path)


# ------------------------------------------------------- prefetching loader
train_idx_arr = np.array(train_idx)

# EPOCH-BASED SHUFFLED ITERATION (replaces the winner's per-step
# rng.choice(..., replace=False), which samples WITH replacement ACROSS steps
# and therefore never touches ~exp(-steps*batch/pool) of the pool -- ~4.8% at
# the winner's 8000 x 112 over 295k tracks).  Here each epoch is one full
# random permutation of the pool, consumed in batch-sized chunks; the last
# partial batch of an epoch is DROPPED so every step has a uniform batch size
# (at batch 512 over ~287k tracks that discards <0.2% of one pass, and a
# different remainder is dropped each epoch because the permutation changes).
# Sigma is still drawn fresh at batch-build time, so this is purely an
# indexing change -- the augmentation distribution is untouched.
STEPS_PER_EPOCH = max(1, len(train_idx_arr) // C.BATCH_SIZE)
print(f"[EPOCH] {STEPS_PER_EPOCH} steps/epoch "
      f"({len(train_idx_arr)} tracks / batch {C.BATCH_SIZE}, remainder dropped)"
      f" -> {C.TARGET_STEPS / STEPS_PER_EPOCH:.2f} full passes at "
      f"{C.TARGET_STEPS} steps")

def epoch_of(step):
    return (step - 1) // STEPS_PER_EPOCH


def make_loader(start_step):
    ds = TrackDataset(
        pool=track_pool, theta=theta_train, idx_arr=train_idx_arr,
        steps_per_epoch=STEPS_PER_EPOCH, batch_size=C.BATCH_SIZE, n_max=N_MAX,
        start_step=start_step, target_steps=C.TARGET_STEPS, seed=C.SEED,
        h0=C.H0, k_min=C.K_MIN, k_max=C.K_MAX, p_zero=C.P_ZERO,
        min_sigma_A=C.MIN_SIGMA_A, max_sigma_A=C.MAX_SIGMA_A)
    return _make_loader(ds, C.BATCH_SIZE, C.N_WORKERS, C.PREFETCH, N_MAX,
                        C.K_MAX, pin_memory=(device == 'cuda'))


# ------------------------------------------------------------------- losses
def compute_loss(o, theta_b):
    target = true_dir(theta_b)
    dir_loss = model.direction_nll_axis(o, target).mean()
    e_target = torch.log(theta_b[:, 0].clamp(min=1e-3)) if C.LOG_ENERGY else theta_b[:, 0]
    e_loss = gaussian_nll(o['E_pred'], o['log_sigma'], e_target).mean()
    sign_label = ((target * o['v_ref']).sum(-1) > 0).float()
    sign_bce = F.binary_cross_entropy_with_logits(o['sign_logit'], sign_label)
    total = dir_loss + C.ENERGY_LOSS_WEIGHT * e_loss + C.SIGN_LOSS_WEIGHT * sign_bce
    return total, dir_loss, e_loss, sign_bce


# --------------------------------------------------------------- quick eval
@torch.no_grad()
def quick_eval(seed=999):
    model.eval()
    backup = ema.swap_in(model)
    rows = []
    for name, lo, hi in C.ENERGY_BINS:
        pool = np.where((energy_np_eval >= lo) & (energy_np_eval < hi))[0]
        for s_nm in C.EVAL_SIGMAS_NM_QUICK:
            rng = np.random.default_rng(seed + s_nm)
            idx = pool if len(pool) <= C.EVAL_N_PER_BIN else rng.choice(
                pool, size=C.EVAL_N_PER_BIN, replace=False)
            x = build_eval_batch(raw_eval, idx, s_nm * 10.0, N_MAX, rng,
                                 C.H0, C.K_MIN, C.K_MAX)
            preds = [model.predict_signed(x[i:i + C.EVAL_CHUNK].to(device))
                     for i in range(0, len(idx), C.EVAL_CHUNK)]
            pred = torch.cat(preds, 0)
            tgt = true_dir(theta_eval[idx]).to(device)
            cos = (pred * tgt).sum(-1).clamp(-1, 1)
            rows.append(dict(bin=name, sigma_nm=s_nm, n=len(idx),
                             axis_err_deg=float(np.median(torch.rad2deg(
                                 torch.arccos(cos.abs())).cpu().numpy())),
                             headtail_pct=float((cos > 0).float().mean()) * 100))
    ema.swap_out(model, backup)
    model.train()
    return rows


def fmt(rows):
    return "  |  ".join(f"{r['bin'][:4]}@{r['sigma_nm']}nm: "
                        f"ax={r['axis_err_deg']:.1f} ht={r['headtail_pct']:.0f}%"
                        for r in rows)


# ------------------------------------------------------------- training loop
def main():
    print(f"\n[TRAIN] batch={C.BATCH_SIZE} target={C.TARGET_STEPS} "
          f"budget={C.TIME_BUDGET_HOURS}h")
    loader = make_loader(start_step)
    t0 = time.time()
    last_ckpt = t0
    loss_ema, step, epoch = None, start_step, epoch_of(max(start_step, 1))
    train_log, budget = [], C.TIME_BUDGET_HOURS * 3600
    nan_strikes = 0
    wait_s = 0.0          # cumulative time the GPU sat idle waiting on batches
    it = iter(loader)
    try:
        while True:
            _t = time.time()
            try:
                theta_b, x_b, s = next(it)
            except StopIteration:
                break
            wait_s += time.time() - _t
            if step >= C.TARGET_STEPS or time.time() - t0 >= budget:
                break
            # batches arrive strictly in order, so `s` (the data-order step the
            # worker used) tracks the monotonic optimizer step exactly.
            step += 1
            epoch = epoch_of(int(s))
            if step < C.WARMUP_STEPS:
                lr = C.LR_MIN + (C.LR_MAX - C.LR_MIN) * step / max(C.WARMUP_STEPS, 1)
            else:
                prog = min((step - C.WARMUP_STEPS) /
                           max(C.TARGET_STEPS - C.WARMUP_STEPS, 1), 1.0)
                lr = C.LR_MIN + 0.5 * (C.LR_MAX - C.LR_MIN) * (1 + np.cos(np.pi * prog))
            for g in optimizer.param_groups:
                g['lr'] = lr

            theta_b = theta_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            x_b, theta_b = apply_oh_augmentation(x_b, theta_b, N_MAX)

            o = model(x_b)
            loss, dl, el, sb = compute_loss(o, theta_b)

            if not torch.isfinite(loss):
                nan_strikes += 1
                print(f"  [WARN] non-finite loss at step {step} "
                      f"(strike {nan_strikes}) -- batch skipped")
                optimizer.zero_grad(set_to_none=True)
                if nan_strikes > 20:
                    raise RuntimeError("too many non-finite losses; aborting")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            if not torch.isfinite(gnorm):
                nan_strikes += 1
                print(f"  [WARN] non-finite grad norm at step {step} -- skipped")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            ema.update(model)

            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % 50 == 0 or step == start_step + 1:
                el_t = time.time() - t0
                print(f"  step {step:6d} ep{epoch:3d}  loss_ema={loss_ema:.3f}  "
                      f"dir={dl.item():.3f} E={el.item():.3f} sign={sb.item():.3f}  "
                      f"|g|={float(gnorm):.2f} lr={lr:.2e}  ({el_t/60:.1f}min, "
                      f"{(step-start_step)/max(el_t,1e-9):.2f} it/s, "
                      f"loader-wait {100*wait_s/max(el_t,1e-9):.1f}%)")

            now = time.time()
            if now - last_ckpt > C.CHECKPOINT_EVERY_SEC:
                last_ckpt = now
                save_ckpt(CKPT_PATH, step)
                rows = quick_eval()
                print(f"  [CKPT {step} ep{epoch} "
                      f"({step/STEPS_PER_EPOCH:.2f} passes)] {fmt(rows)}")
                train_log.append(dict(step=step, epoch=epoch,
                                      passes=step / STEPS_PER_EPOCH,
                                      elapsed_min=(now - t0) / 60,
                                      loss_ema=loss_ema, eval=rows))
                with open(os.path.join(C.RESULTS_DIR, "train_log.json"), 'w') as f:
                    json.dump(train_log, f, indent=2)
    finally:
        # release the persistent worker processes deterministically
        del loader

    elapsed = time.time() - t0
    ips = (step - start_step) / max(elapsed, 1e-9)
    print(f"\n[STOP] step {step} in {elapsed/3600:.2f}h "
          f"({step/STEPS_PER_EPOCH:.2f} full passes over the training pool)")
    # machine-readable; launch_a100.sh greps this to size TARGET_STEPS
    print(f"[THROUGHPUT] {ips:.4f} it/s")
    print(f"[LOADER-WAIT] {100*wait_s/max(elapsed,1e-9):.1f}% of wall clock "
          f"({wait_s/60:.1f} min) spent blocked on batch preparation "
          f"-- raise N_WORKERS (now {C.N_WORKERS}) if this is large")
    rows = quick_eval()
    print("[FINAL] " + fmt(rows))
    with open(os.path.join(C.RESULTS_DIR, "energy_split_final.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    save_ckpt(os.path.join(C.RESULTS_DIR, "checkpoint_final.pt"), step)
    save_ckpt(CKPT_PATH, step)
    print("[DONE]")


if __name__ == '__main__':
    main()
