"""v23_final: fast correctness check -- a SHORT REAL RUN on REAL data.

Scope, stated plainly so no one over-reads it: this confirms the path EXECUTES
and the optimizer makes progress.  It is 300-ish steps on a few thousand tracks.
NO conclusion about whether the numbers improve is drawn or implied, and none
should be -- that is what the full run plus eval.py's pre-registered gates are
for.  (Same discipline a100_final's README applied to its own smoke test.)

What it checks
  1. the real dynamic-blur pipeline produces finite, correctly-shaped batches
     from the real v2 corpus, with the expected sigma distribution;
  2. a real training loop runs with no NaN/Inf loss and no crash;
  3. the training loss DECREASES (first-fifth mean vs last-fifth mean);
  4. the kappa regularizer can be switched on without destabilizing anything;
  5. a checkpoint SAVE -> RELOAD round-trip reproduces bit-identical outputs.
     This project has been burned by silently-broken checkpoints and
     version-mismatched pickled objects, so the round-trip is checked here
     rather than discovered later: v23 stores state_dicts plus a plain config
     dict and rebuilds via build_v23(), never a pickled module object.
  6. the eval path runs end to end on two real grid cells against the real PCA
     baseline.

Run:  python -m smearing_resolution.architecture_experiments.v23_final.check_fast
      STEPS=600 CHECK_NROWS=600000 python -m ...v23_final.check_fast
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from src.models.egnn import DirectionHead, EnergyHead
from src.models.vmf_loss import axis_aware_vmf_nll, gaussian_nll
from smearing_resolution.architecture_experiments.v23_final import config as C
from smearing_resolution.architecture_experiments.v23_final.data_pipeline import (
    N_PHYS_COND, TrackDataset, TrackPool, build_eval_batch, flat_dim,
    load_raw_tracks_v23, make_loader, pool_a_mask,
)
from smearing_resolution.architecture_experiments.v23_final.metrics import (
    angular_coverage_ece, axis_error_and_headtail, pca_axis_and_signed,
)
from smearing_resolution.architecture_experiments.v23_final.model import build_v23
from smearing_resolution.architecture_experiments.v23_final.train import (
    apply_oh_augmentation, compute_phys_stats, save_ckpt,
)

DEVICE = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')
STEPS = int(os.environ.get("STEPS", 300))
NROWS = int(os.environ.get("CHECK_NROWS", 400_000))
BATCH = int(os.environ.get("CHECK_BATCH", 64))
N_MAX = C.MAX_POINTS
OUT = os.path.join(C.RESULTS_DIR, "check_fast")


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return bool(ok)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    all_ok = True
    print(f"v23 fast correctness check | device={DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else "")
          + f" | {STEPS} steps @ batch {BATCH}")
    if DEVICE != 'cuda':
        print("  !! NO CUDA DEVICE AVAILABLE -- running on CPU. This still checks "
              "correctness,\n  !! but the throughput number below is NOT "
              "representative of the paper run.")

    # ------------------------------------------------------ 1. real data
    print(f"\n=== 1. real data pipeline ({NROWS:,} CSV rows of the v2 corpus) ===")
    raw, theta, ions, obs_max, _ = load_raw_tracks_v23(C.TRAIN_CSV, N_MAX,
                                                       nrows=NROWS)
    all_ok &= report("real tracks loaded", len(raw) > 100, f"{len(raw):,} tracks")
    is_a = pool_a_mask(ions, expect_full_corpus=(NROWS == 0))
    pool = TrackPool(raw)
    del raw
    idx = np.arange(len(pool))
    phys_mean, phys_std = compute_phys_stats(pool, theta, idx, N_MAX,
                                             n_batches=8)
    all_ok &= report("phys standardization stats finite",
                     bool(torch.isfinite(phys_mean).all()
                          and torch.isfinite(phys_std).all()
                          and (phys_std > 0).all()))

    ds = TrackDataset(pool=pool, theta=theta, idx_arr=idx,
                      steps_per_epoch=max(1, len(idx) // BATCH),
                      batch_size=BATCH, n_max=N_MAX, start_step=0,
                      total_steps=STEPS + 5, seed=C.SEED, h0=C.H0,
                      k_min=C.K_MIN, k_max=C.K_MAX, p_zero=C.P_ZERO,
                      min_sigma_A=C.MIN_SIGMA_A, max_sigma_A=C.MAX_SIGMA_A)
    loader = make_loader(ds, BATCH, C.N_WORKERS, C.PREFETCH, N_MAX, C.K_MAX,
                         pin_memory=(DEVICE == 'cuda'))
    it = iter(loader)
    th0, x0, _ = next(it)
    all_ok &= report("batch shape correct",
                     tuple(x0.shape) == (BATCH, flat_dim(N_MAX, C.K_MAX)),
                     f"{tuple(x0.shape)} expected "
                     f"{(BATCH, flat_dim(N_MAX, C.K_MAX))}")
    all_ok &= report("batch finite", bool(torch.isfinite(x0).all()
                                          and torch.isfinite(th0).all()))

    # ------------------------------------------------------ 2-4. real run
    print(f"\n=== 2-4. short REAL training run ===")
    model = build_v23(n_max=N_MAX, phys_mean=phys_mean, phys_std=phys_std,
                      device=DEVICE)
    dir_head = DirectionHead(d_latent=C.D_AUG).to(DEVICE)
    energy_head = EnergyHead(d_latent=C.D_AUG, log_energy=C.LOG_ENERGY).to(DEVICE)
    lam = float(os.environ.get("LAMBDA_KAPPA_CHECK", 0.002))
    print(f"  lambda_kappa = {lam} (deliberately NON-zero here: check 4 confirms "
          f"the regularizer\n  can be switched on without destabilizing training; "
          f"the staged build plan runs\n  the swap-in verification at lambda=0, "
          f"which verify_zero_init.py check 7 covers)")
    params = (list(model.parameters()) + list(dir_head.parameters())
              + list(energy_head.parameters()))
    opt = torch.optim.AdamW(params, lr=C.LR_MAX, weight_decay=C.WEIGHT_DECAY)

    losses, nlls, kaps = [], [], []
    n_nan = 0
    sig_all = []
    t_run = time.time()
    for step in range(STEPS):
        try:
            th_c, x_c, _ = next(it)
        except StopIteration:
            print(f"  loader exhausted at step {step}")
            break
        th = th_c.to(DEVICE)
        x = x_c.to(DEVICE)
        sig_all.append(x[:, -C.N_PHYS:][:, -2].detach().cpu().numpy())
        x, th = apply_oh_augmentation(x, th, N_MAX)
        th[:, 0] = torch.log(th[:, 0].clamp(min=1e-3))
        opt.zero_grad(set_to_none=True)
        nll, kpen = model.loss_terms(th, x)
        nll = nll.clamp(max=C.LOSS_CLAMP)
        z = model.embedding_net.last_z
        mu_hat, kap = dir_head(z)
        dn = axis_aware_vmf_nll(mu_hat, kap, th[:, 1:4]).clamp(max=C.LOSS_CLAMP)
        ep, ls = energy_head(z)
        en = gaussian_nll(ep, ls, th[:, 0]).clamp(max=C.LOSS_CLAMP)
        total = (nll.mean() + C.ALPHA_START * dn.mean() + C.BETA_START * en.mean()
                 + lam * kpen.mean())
        if not torch.isfinite(total):
            n_nan += 1
            continue
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
        if not torch.isfinite(gn):
            n_nan += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        losses.append(float(total))
        nlls.append(float(nll.mean()))
        kaps.append(float(kpen.mean()))
        if step % 50 == 0 or step == STEPS - 1:
            print(f"    step {step:4d}  total={float(total):8.3f}  "
                  f"nll={float(nll.mean()):8.3f}  "
                  f"mean_kappa={float(kpen.mean()):6.2f}  |g|={float(gn):7.2f}")
    run_s = time.time() - t_run
    del loader

    sig_all = np.concatenate(sig_all)
    frac0 = float((sig_all == 0).mean())
    print(f"  sigma distribution actually seen: {100*frac0:.1f}% at exactly 0 "
          f"(P_ZERO={C.P_ZERO}), nonzero median "
          f"{np.median(sig_all[sig_all>0])/10:.1f}nm, max "
          f"{sig_all.max()/10:.0f}nm")
    all_ok &= report("Stage-1 data pipeline actually delivers BLUR (the v22 bug "
                     "this fixes)", (sig_all > 0).any() and frac0 < 0.9,
                     f"{100*(1-frac0):.1f}% of examples blurred")
    all_ok &= report("no NaN/Inf losses", n_nan == 0, f"{n_nan} non-finite steps")
    all_ok &= report("ran the requested number of steps", len(losses) >= STEPS - 5,
                     f"{len(losses)} optimizer steps in {run_s:.0f}s "
                     f"({len(losses)/max(run_s,1e-9):.2f} it/s, batch {BATCH})")
    q = max(1, len(losses) // 5)
    first, last = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
    all_ok &= report("training loss DECREASES", last < first,
                     f"first-fifth mean {first:.3f} -> last-fifth mean "
                     f"{last:.3f}  ({100*(first-last)/abs(first):+.1f}%)")
    fn, ln = float(np.mean(nlls[:q])), float(np.mean(nlls[-q:]))
    all_ok &= report("posterior NLL component decreases too", ln < fn,
                     f"{fn:.3f} -> {ln:.3f}")
    all_ok &= report("the kappa penalty stays finite and bounded",
                     np.isfinite(kaps).all() and max(kaps) < 1e4,
                     f"mean kappa {kaps[0]:.2f} -> {kaps[-1]:.2f}")

    # ------------------------------------------------- 5. ckpt round-trip
    print(f"\n=== 5. checkpoint save -> reload round-trip ===")
    ck_path = os.path.join(OUT, "check_fast_ckpt.pt")
    save_ckpt(ck_path, model, None, dir_head, energy_head, opt, 0,
              dict(nll=ln, ece=float('nan')), phys_mean, phys_std, N_MAX, 1)
    model.eval()
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    cfg = dict(ck['cfg'])
    cfg.pop('n_max')
    m2 = build_v23(n_max=ck['n_max'], phys_mean=ck['phys_mean'],
                   phys_std=ck['phys_std'], device=DEVICE,
                   hidden_dim=cfg.pop('hidden_dim'), n_layers=cfg.pop('n_layers'),
                   k=cfg.pop('k'), **cfg)
    m2.load_state_dict(ck['model_state_dict'])
    m2.eval()
    # every saved tensor must come back bit-identical
    sd = model.state_dict()
    mism = [k for k, v in m2.state_dict().items() if not torch.equal(sd[k], v)]
    all_ok &= report("every parameter/buffer round-trips bit-identically",
                     not mism, f"{len(mism)} mismatched tensor(s)"
                     + (f": {mism[:5]}" if mism else ""))
    # Forward-output comparison is done on CPU.  On CUDA the backbone's
    # scatter_add_ uses atomicAdd, whose summation ORDER is not deterministic, so
    # even one model forwarded twice on GPU differs at ~1e-7 relative -- a GPU
    # bitwise claim would be measuring atomics, not the checkpoint.  The GPU
    # same-model-twice control is reported alongside for context.
    xr = x0.to(DEVICE)
    with torch.no_grad():
        g1 = model.params(xr)
        g2 = model.params(xr)
        gr = m2.params(xr)
    ctrl = max((g1[k] - g2[k]).abs().max().item() for k in ('mu', 'kappa', 'logits'))
    gpu_d = max((g1[k] - gr[k]).abs().max().item() for k in ('mu', 'kappa', 'logits'))
    model.to('cpu')
    m2.to('cpu')
    with torch.no_grad():
        ref = model.params(x0)
        got = m2.params(x0)
    for k in ('mu', 'kappa', 'logits', 'u'):
        d = (ref[k] - got[k]).abs().max().item()
        all_ok &= report(f"[cpu] reloaded '{k}' bit-identical", d == 0.0,
                         f"max|delta|={d:.3e}")
    print(f"    (gpu, for context: round-trip delta {gpu_d:.3e} vs the "
          f"same-model-forwarded-twice control {ctrl:.3e} -- CUDA atomicAdd "
          f"non-determinism, not a checkpoint defect)")
    model.to(DEVICE)
    m2.to(DEVICE)
    all_ok &= report("checkpoint stores no pickled module object",
                     not any(hasattr(v, 'state_dict') for v in ck.values()),
                     f"keys={sorted(ck.keys())}")

    # ------------------------------------------------------- 6. eval path
    print(f"\n=== 6. eval path on two real grid cells ===")
    raw_e, theta_e, _i, _om, _ = load_raw_tracks_v23(C.EVAL_CSV, N_MAX,
                                                     nrows=120_000)
    energy = theta_e[:, 0].numpy()
    for bin_name, lo, hi in (C.ENERGY_BINS[2], C.ENERGY_BINS[1]):
        p = np.where((energy >= lo) & (energy < hi))[0]
        if len(p) == 0:
            print(f"    [SKIP] {bin_name}: no tracks in this nrows slice")
            continue
        for s_nm in (0, 10):
            rng = np.random.default_rng(2000 + s_nm)
            sel = p if len(p) <= 200 else rng.choice(p, 200, replace=False)
            xe = build_eval_batch(raw_e, sel, s_nm * 10.0, N_MAX, rng,
                                  C.H0, C.K_MIN, C.K_MAX)
            tgt = F.normalize(theta_e[sel][:, 1:4], dim=-1)
            with torch.no_grad():
                pr = m2.predict(xe.to(DEVICE), n_samples=64)
            ax, ht = axis_error_and_headtail(pr['mean_dir'].cpu(), tgt)
            axm, htm = axis_error_and_headtail(pr['mode_dir'].cpu(), tgt)
            ece, _cov = angular_coverage_ece(pr['samples'].cpu(), tgt)
            rg = np.random.default_rng(3000 + s_nm)
            ge = [np.degrees(np.arccos(abs(float(np.clip(
                np.dot(pca_axis_and_signed(raw_e[i].copy(), s_nm * 10.0, rg),
                       tgt[j].numpy()), -1, 1)))))
                for j, i in enumerate(sel)]
            print(f"    {bin_name:15s} s={s_nm:3d}nm  n={len(sel):3d}  "
                  f"model ax={ax:6.2f} ht={ht:5.1f}%  (mode ax={axm:6.2f} "
                  f"ht={htm:5.1f}%)  pca ax={np.median(ge):6.2f}  "
                  f"ECE={100*ece:5.2f}%")
            all_ok &= report(f"eval cell {bin_name}@{s_nm}nm produced finite "
                             f"metrics",
                             np.isfinite([ax, ht, ece, np.median(ge)]).all())
    print("\n  REMINDER: these are ~300 steps on a few thousand tracks. They are "
          "evidence the\n  path executes, NOT evidence about final performance. "
          "Do not quote them.")

    print(f"\n{'='*66}")
    print(f"  fast correctness check: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}"
          f"   ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*66}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
