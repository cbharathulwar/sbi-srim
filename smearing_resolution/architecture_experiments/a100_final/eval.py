"""a100_final: evaluate a checkpoint on the 9-sigma x 3-energy reference grid.

Reports, per cell: model axis error (median deg), model head-tail %, the
classical PCA/geometric baseline (recomputed here with the SAME convention as
eval_r7_full_sweep.py so numbers are directly comparable to
results_diag_r7_trunc/full_sweep_9tier_6nm_EMA.csv), and angular-coverage ECE
(same definition as eval_diag_r7_trunc_calibration.py).

  CKPT=.../checkpoint_final.pt CONFIG=large python eval.py
  N_PER_BIN=1000 EMA=1 python eval.py
"""
import sys, os, csv, json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from smearing_resolution.architecture_experiments.a100_final import config as C
from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    load_raw_tracks, build_eval_batch)
from smearing_resolution.architecture_experiments.a100_final.model import DiagModelV2

CKPT = os.environ.get("CKPT", os.path.join(C.RESULTS_DIR, "checkpoint_final.pt"))
OUT_CSV = os.environ.get("OUT_CSV", os.path.join(C.RESULTS_DIR, "full_sweep_9tier.csv"))
N_PER_BIN = int(os.environ.get("N_PER_BIN", "1000"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "200"))
USE_EMA = os.environ.get("EMA", "1") == "1"
ANG_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95])
device = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')


def pca_axis_and_signed(pts, sigma_A, rng):
    """Classical baseline -- identical convention to eval_r7_full_sweep.py:211."""
    if sigma_A > 0:
        pts = pts + rng.normal(0.0, sigma_A, size=pts.shape)
    c = pts - pts.mean(axis=0)
    cov = (c.T @ c) / len(c)
    axis = np.linalg.eigh(cov)[1][:, -1]
    if ((c @ axis) ** 3).sum() > 0:
        axis = -axis
    return axis


def angular_coverage_ece(samples, true_dir):
    """Unchanged from eval_diag_r7_trunc_calibration.py:281."""
    mean_dir = F.normalize(samples.mean(dim=0), dim=-1, eps=1e-8)
    dots = (samples * mean_dir.unsqueeze(0)).sum(-1).clamp(-1, 1)
    angles = torch.rad2deg(torch.arccos(dots)).cpu().numpy()
    cos_true = (mean_dir * true_dir).sum(-1).clamp(-1, 1)
    ang_true = torch.rad2deg(torch.arccos(cos_true)).cpu().numpy()
    coverage = np.array([float(np.mean(ang_true <= np.percentile(angles, q * 100, axis=0)))
                         for q in ANG_LEVELS])
    return float(np.mean(np.abs(coverage - ANG_LEVELS))), coverage


def main():
    print(f"[LOAD] {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck.get('cfg', {})
    n_max = cfg.get('n_max', ck.get('n_max', C.MAX_POINTS))
    k_max = cfg.get('k_max', C.K_MAX)
    model = DiagModelV2(**{**cfg, 'n_max': n_max}).to(device)
    sd = ck['model']
    if USE_EMA and ck.get('ema'):
        sd = dict(sd)
        sd.update({k: v.to(device) for k, v in ck['ema'].items()})
        print("[EMA] using shadow weights")
    model.load_state_dict(sd)
    model.eval()
    print(f"[MODEL] step={ck.get('step')} cfg={json.dumps(cfg)}")

    raw_eval, theta_eval, _n, _ = load_raw_tracks(C.EVAL_CSV, max_points=n_max)
    assert _n <= n_max, (_n, n_max)
    energy = theta_eval[:, 0].numpy()

    rows = []
    for name, lo, hi in C.ENERGY_BINS:
        pool = np.where((energy >= lo) & (energy < hi))[0]
        for s_nm in C.SIGMAS_NM:
            s_A = s_nm * 10.0
            rng = np.random.default_rng(2000 + s_nm)
            idx = pool if len(pool) <= N_PER_BIN else rng.choice(
                pool, size=N_PER_BIN, replace=False)
            x = build_eval_batch(raw_eval, idx, s_A, n_max, rng,
                                 C.H0, C.K_MIN, k_max)
            tgt = F.normalize(theta_eval[idx][:, 1:4], dim=-1).to(device)

            preds, ece_parts = [], []
            with torch.no_grad():
                for i in range(0, len(idx), C.EVAL_CHUNK):
                    xb = x[i:i + C.EVAL_CHUNK].to(device)
                    o = model(xb)
                    sgn = torch.where(torch.sigmoid(o['sign_logit']) > 0.5,
                                      1.0, -1.0).unsqueeze(-1)
                    preds.append(o['axis_ref_aligned'] * sgn)
                    ece_parts.append(model.sample_direction(o, N_SAMPLES).cpu())
            pred = torch.cat(preds, 0)
            samples = torch.cat(ece_parts, 1)
            cos = (pred * tgt).sum(-1).clamp(-1, 1)
            axis_err = float(np.median(torch.rad2deg(
                torch.arccos(cos.abs())).cpu().numpy()))
            ht = float((cos > 0).float().mean()) * 100
            ece, _cov = angular_coverage_ece(samples, tgt.cpu())

            rng_g = np.random.default_rng(3000 + s_nm)
            gt = tgt.cpu().numpy()
            ge, gh = [], []
            for j, i in enumerate(idx):
                pg = pca_axis_and_signed(raw_eval[i].copy(), s_A, rng_g)
                cg = float(np.clip(np.dot(pg, gt[j]), -1, 1))
                ge.append(np.degrees(np.arccos(abs(cg))))
                gh.append(cg > 0)

            r = dict(energy_bin=name, sigma_nm=s_nm,
                     model_axis_err_deg=axis_err, model_headtail_pct=ht,
                     geometric_axis_err_deg=float(np.median(ge)),
                     geometric_headtail_pct=float(np.mean(gh) * 100),
                     ece_pct=100 * ece, n=len(idx))
            rows.append(r)
            print(f"  {name:15s} s={s_nm:3d}nm  model ax={axis_err:6.2f} "
                  f"ht={ht:5.1f}%  |  pca ax={r['geometric_axis_err_deg']:6.2f} "
                  f"ht={r['geometric_headtail_pct']:5.1f}%  |  ECE={100*ece:5.2f}%")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[SAVE] {OUT_CSV}")
    for name, _lo, _hi in C.ENERGY_BINS:
        sub = [r for r in rows if r['energy_bin'] == name]
        d = np.mean([r['geometric_axis_err_deg'] - r['model_axis_err_deg'] for r in sub])
        print(f"[TIER] {name:15s} mean axis-err edge over PCA = {d:+.2f} deg   "
              f"mean ECE = {np.mean([r['ece_pct'] for r in sub]):.2f}%")


if __name__ == '__main__':
    main()
