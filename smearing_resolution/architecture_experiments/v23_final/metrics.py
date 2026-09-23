"""v23_final: the evaluation metrics, ported UNCHANGED so v23's numbers are
directly comparable to every earlier sweep in this project.

  * angular_coverage_ece  -- identical to a100_final/eval.py:49 and to
    src/evaluation/eval_mcpe3d.py:319-336 (which is itself
    eval_diag_r7_trunc_calibration.py:281).  Same 11 nominal levels.
  * pca_axis_and_signed   -- identical to a100_final/eval.py:37 and
    eval_r7_full_sweep.py:211, so the classical baseline is computed with the
    same convention the reference sweeps used.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from smearing_resolution.architecture_experiments.v23_final import config as C

ANG_LEVELS = np.array(C.ANG_LEVELS)


def pca_axis_and_signed(pts, sigma_A, rng):
    """Classical PCA/geometric baseline -- identical convention to
    a100_final/eval.py:37 / eval_r7_full_sweep.py:211.  Physical coordinates, no
    normalization, no deconvolution, no truncation: smear -> center ->
    covariance/n -> eigh, largest eigenvalue -> flip on POSITIVE projection
    skewness."""
    if sigma_A > 0:
        pts = pts + rng.normal(0.0, sigma_A, size=pts.shape)
    c = pts - pts.mean(axis=0)
    cov = (c.T @ c) / len(c)
    axis = np.linalg.eigh(cov)[1][:, -1]
    if ((c @ axis) ** 3).sum() > 0:
        axis = -axis
    return axis


def angular_coverage_ece(samples, true_dir, levels=None):
    """Angular-radius credible-interval coverage and its ECE.

    samples : (n_samples, B, 3) unit direction samples
    true_dir: (B, 3) unit true directions
    Returns (ece, coverage) with ece = mean_q |coverage(q) - q|.

    Self-referenced to the posterior's OWN mean direction, per-track percentile
    -- unchanged from the reference implementation.
    """
    levels = ANG_LEVELS if levels is None else np.asarray(levels)
    mean_dir = F.normalize(samples.mean(dim=0), dim=-1, eps=1e-8)
    dots = (samples * mean_dir.unsqueeze(0)).sum(-1).clamp(-1, 1)
    angles = torch.rad2deg(torch.arccos(dots)).cpu().numpy()
    cos_true = (mean_dir * true_dir).sum(-1).clamp(-1, 1)
    ang_true = torch.rad2deg(torch.arccos(cos_true)).cpu().numpy()
    coverage = np.array([
        float(np.mean(ang_true <= np.percentile(angles, q * 100, axis=0)))
        for q in levels])
    return float(np.mean(np.abs(coverage - levels))), coverage


def axis_error_and_headtail(pred_dir, true_dir):
    """Median axis error in degrees (arccos|cos|) and head-tail accuracy in %."""
    cos = (pred_dir * true_dir).sum(-1).clamp(-1, 1)
    err = torch.rad2deg(torch.arccos(cos.abs())).cpu().numpy()
    return float(np.median(err)), float((cos > 0).float().mean()) * 100.0
