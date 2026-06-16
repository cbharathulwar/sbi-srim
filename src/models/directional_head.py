"""Directional posterior head — native S^2 modeling of the recoil direction.

Drop-in replacement for the NSF flow used as the SBI posterior estimator. The
NSF models the direction as three FREE Euclidean coordinates (vx,vy,vz) and
normalizes post-hoc, which (a) leaks the spherical constraint into the
per-component calibration (the C2ST failure we measured) and (b) handles the
head/tail (antipodal) decision only implicitly. Here we instead model:

    p(theta | x) = p_E(logE | z) * p_dir(direction-on-S^2 | z)

with z = embedding_net(x):
  * p_E   : a 1-D Gaussian mixture over log-energy (captures the skew).
  * p_dir : a mixture of von Mises-Fisher distributions on S^2 — multimodal,
            so two components can sit at the head and the tail, making head/tail
            an explicit modeled structure rather than an emergent one.

API mirrors the sbi flow used in training/eval:
    .loss(theta, condition)   -> (B,) per-sample NLL   (theta = [logE, vx,vy,vz])
    .sample(shape, condition) -> (shape[0], B, 4)       (samples = [logE, vx,vy,vz])
and it exposes `.embedding_net` so the eval/diagnostics chain-walk still finds
.k / .n_max / .n_phys / .log_energy, and aux heads still read embedding.last_z.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_4PI = math.log(4.0 * math.pi)


def _log_vmf_norm_3d(kappa):
    """log C_3(kappa) = log(kappa / (4*pi*sinh(kappa))) for the 3-D vMF, with a
    dual regime for numerical stability (sinh overflows ~87 in float32)."""
    small = kappa < 2.65
    k2 = kappa * kappa
    taylor = -_LOG_4PI - torch.log1p(k2 / 6.0 + k2 * k2 / 120.0)
    asymp = torch.log(kappa.clamp(min=1e-8)) - _LOG_2PI - kappa
    return torch.where(small, taylor, asymp)


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _sample_vmf_3d(mu, kappa):
    """Sample from vMF on S^2. mu: (..., 3) unit, kappa: (...,) -> (..., 3).
    Wood/Ulrich method specialized to p=3 (closed-form inverse-CDF for the
    polar cosine w), built directly in an orthonormal frame aligned with mu."""
    shape = kappa.shape
    u = torch.rand(shape, device=mu.device, dtype=mu.dtype)
    # w = 1 + (1/k) * log( u + (1-u) e^{-2k} ); large-k safe, small-k clamps
    ek = torch.exp(-2.0 * kappa)
    w = 1.0 + torch.log(u + (1.0 - u) * ek + 1e-20) / kappa.clamp(min=1e-6)
    w = w.clamp(-1.0, 1.0)
    phi = torch.rand(shape, device=mu.device, dtype=mu.dtype) * (2.0 * math.pi)
    s = torch.sqrt((1.0 - w * w).clamp(min=0.0))
    # orthonormal basis (b1, b2) perpendicular to mu
    ref = torch.zeros_like(mu); ref[..., 0] = 1.0
    near = mu[..., 0].abs() > 0.9
    ref2 = torch.zeros_like(mu); ref2[..., 1] = 1.0
    ref = torch.where(near.unsqueeze(-1), ref2, ref)
    b1 = F.normalize(torch.cross(ref, mu, dim=-1), dim=-1, eps=1e-8)
    b2 = torch.cross(mu, b1, dim=-1)
    out = (s.unsqueeze(-1) * (torch.cos(phi).unsqueeze(-1) * b1
                              + torch.sin(phi).unsqueeze(-1) * b2)
           + w.unsqueeze(-1) * mu)
    return F.normalize(out, dim=-1, eps=1e-8)


class VMFMixture(nn.Module):
    """Mixture of K von Mises-Fisher components on S^2, conditioned on z."""
    def __init__(self, d_in, n_comp=4, hidden=128):
        super().__init__()
        self.K = n_comp
        self.net = _MLP(d_in, n_comp * 5, hidden)  # per comp: mu(3), kappa(1), logit(1)

    def _params(self, z):
        B = z.shape[0]
        o = self.net(z).view(B, self.K, 5)
        mu = F.normalize(o[..., :3], dim=-1, eps=1e-8)        # (B,K,3)
        kappa = F.softplus(o[..., 3]) + 1e-2                  # (B,K) > 0
        logits = o[..., 4]                                   # (B,K)
        return mu, kappa, logits

    def log_prob(self, d, z):
        """d: (B,3) unit direction. Returns (B,) log density."""
        mu, kappa, logits = self._params(z)
        cos = (mu * d.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)  # (B,K)
        log_comp = _log_vmf_norm_3d(kappa) + kappa * cos       # (B,K)
        log_w = F.log_softmax(logits, dim=-1)
        return torch.logsumexp(log_w + log_comp, dim=-1)       # (B,)

    @torch.no_grad()
    def sample(self, n, z):
        mu, kappa, logits = self._params(z)
        B, K = kappa.shape
        w = F.softmax(logits, dim=-1)
        idx = torch.multinomial(w, n, replacement=True).t()    # (n,B)
        br = torch.arange(B, device=z.device).unsqueeze(0).expand(n, B)
        mu_s = mu[br, idx]                                     # (n,B,3)
        kap_s = kappa[br, idx]                                 # (n,B)
        return _sample_vmf_3d(mu_s, kap_s)                     # (n,B,3)


class GMM1D(nn.Module):
    """1-D Gaussian mixture (for log-energy), conditioned on z."""
    def __init__(self, d_in, n_comp=3, hidden=128):
        super().__init__()
        self.K = n_comp
        self.net = _MLP(d_in, n_comp * 3, hidden)  # per comp: mean, log_std, logit

    def _params(self, z):
        o = self.net(z).view(z.shape[0], self.K, 3)
        return o[..., 0], o[..., 1].clamp(-7.0, 3.0), o[..., 2]  # mean, log_std, logit

    def log_prob(self, y, z):
        mean, log_std, logits = self._params(z)
        inv = torch.exp(-log_std)
        log_comp = -0.5 * ((y.unsqueeze(1) - mean) * inv) ** 2 - log_std - 0.5 * _LOG_2PI
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + log_comp, dim=-1)

    @torch.no_grad()
    def sample(self, n, z):
        mean, log_std, logits = self._params(z)
        B = z.shape[0]
        idx = torch.multinomial(F.softmax(logits, dim=-1), n, replacement=True).t()
        br = torch.arange(B, device=z.device).unsqueeze(0).expand(n, B)
        m = mean[br, idx]; s = torch.exp(log_std)[br, idx]
        return m + s * torch.randn(n, B, device=z.device, dtype=z.dtype)


class DirectionalPosterior(nn.Module):
    """Drop-in for the NSF flow: factorized p_E(logE|z) * p_dir(dir-on-S^2|z).

    Args:
        embedding_net: the (Physics)AugmentedEmbedding; called on the flat x and
            expected to set `.last_z` (for the aux heads) and expose .k/.n_max/
            .n_phys/.log_energy (for the eval chain-walk).
        d_cond: conditioning dimension (= D_AUG).
    """
    def __init__(self, embedding_net, d_cond, n_dir_comp=4, n_e_comp=3, hidden=128):
        super().__init__()
        self.embedding_net = embedding_net
        self.energy = GMM1D(d_cond, n_e_comp, hidden)
        self.direction = VMFMixture(d_cond, n_dir_comp, hidden)

    def loss(self, theta, condition):
        """theta = [logE, vx, vy, vz]; returns (B,) per-sample NLL."""
        z = self.embedding_net(condition)
        logE = theta[:, 0]
        d = F.normalize(theta[:, 1:4], dim=-1, eps=1e-8)
        return -(self.energy.log_prob(logE, z) + self.direction.log_prob(d, z))

    @torch.no_grad()
    def sample(self, shape, condition):
        """Returns (shape[0], B, 4) samples = [logE, vx, vy, vz] (dir on S^2)."""
        z = self.embedding_net(condition)
        n = shape[0]
        E = self.energy.sample(n, z)            # (n, B)
        d = self.direction.sample(n, z)         # (n, B, 3)
        return torch.cat([E.unsqueeze(-1), d], dim=-1)
