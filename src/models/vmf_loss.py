"""
von Mises-Fisher and Gaussian NLL losses for auxiliary supervision.
==================================================================
Numerically stable implementation with dual-regime kappa handling.

Reference: Mardia & Jupp (2000), "Directional Statistics"
"""

import torch
import torch.nn.functional as F
import math


def _log_vmf_normalization(kappa):
    """
    Compute log C_3(kappa) = log(kappa / (4*pi*sinh(kappa)))
    for the 3D von Mises-Fisher distribution.

    Uses dual-regime for numerical stability:
      - kappa < 2.65: Taylor expansion of sinh to avoid cancellation
      - kappa >= 2.65: asymptotic form to avoid sinh overflow (sinh overflows at ~87 in float32)
    """
    # Regime 1: small kappa — Taylor: sinh(k) ≈ k + k³/6 + k⁵/120
    # log(k / (4π·sinh(k))) ≈ log(1/(4π)) - log(1 + k²/6 + k⁴/120)
    small = kappa < 2.65
    k2 = kappa ** 2
    taylor = -math.log(4 * math.pi) - torch.log1p(k2 / 6.0 + k2 * k2 / 120.0)

    # Regime 2: large kappa — sinh(k) ≈ exp(k)/2
    # log(k / (4π·exp(k)/2)) = log(k) - log(2π) - k
    asymp = torch.log(kappa.clamp(min=1e-8)) - math.log(2 * math.pi) - kappa

    return torch.where(small, taylor, asymp)


def vmf_nll(mu_hat, kappa, target):
    """
    Negative log-likelihood of the von Mises-Fisher distribution on S².

    Args:
        mu_hat: (B, 3) predicted unit direction vectors
        kappa:  (B,) or (B, 1) concentration parameter (> 0)
        target: (B, 3) true unit direction vectors

    Returns:
        nll: (B,) per-sample negative log-likelihood
    """
    kappa = kappa.squeeze(-1)  # ensure (B,)
    cos_sim = (mu_hat * target).sum(dim=-1)  # (B,) dot product
    log_norm = _log_vmf_normalization(kappa)  # (B,)
    return -(log_norm + kappa * cos_sim)


def axis_aware_vmf_nll(mu_hat, kappa, target):
    """
    Axis-aware vMF NLL: handles head/tail ambiguity by taking
    the minimum NLL over forward and backward directions.

    Args:
        mu_hat: (B, 3) predicted unit direction vectors
        kappa:  (B,) or (B, 1) concentration parameter
        target: (B, 3) true unit direction vectors

    Returns:
        nll: (B,) per-sample negative log-likelihood (min of fwd/bwd)
    """
    loss_fwd = vmf_nll(mu_hat, kappa, target)
    loss_bwd = vmf_nll(mu_hat, kappa, -target)
    return torch.minimum(loss_fwd, loss_bwd)


def gaussian_nll(pred, log_sigma, target):
    """
    Heteroscedastic Gaussian NLL for energy prediction.

    Args:
        pred:      (B,) predicted energy
        log_sigma: (B,) log standard deviation (learned uncertainty)
        target:    (B,) true energy

    Returns:
        nll: (B,) per-sample negative log-likelihood
    """
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    return log_sigma + 0.5 * ((pred - target) / sigma) ** 2
