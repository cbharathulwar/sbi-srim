"""v23_final: the v23 model -- v22's backbone + v22's unconstrained
DirectionalPosterior, with sigma-conditioning, the residual-on-PCA structural
fallback, and the higher-moment physics features.

Per reports/v23_architecture_plan.tex.  The V-bottleneck / candidate-vector-pool
restriction of the R7 / a100_final lineage is deliberately NOT reintroduced.

Conditioning vector layout (D_AUG = 409), fixed and relied upon by the zero-init
slices below:

    [ 0   : 384 ]  z from the GVP-EGNN backbone (D_LATENT)
    [ 384 : 389 ]  v22's 5 physics features, standardized        (unchanged)
    [ 389 : 393 ]  the 4 NEW higher-moment scalars, standardized (zero-init'd)
    [ 393 : 409 ]  the 16 sinusoidal sigma-embedding dims        (zero-init'd)

Three independent zero-init guarantees, each verified empirically by
verify_zero_init.py (bitwise, not "approximately"):

  (A) S{head}: the first-Linear weight COLUMNS feeding the 16 embedded sigma
      dims are zeroed in VMFMixture.net and GMM1D.net, so at initialization the
      model is behaviourally identical to v22 before any training signal
      distinguishes sigma values.
  (B) staged build plan item 1: likewise for the 4 new higher-moment columns.
  (C) S{fallback}: the mu-output ROWS of the head's shared final Linear are
      zeroed, so delta_k == 0 exactly and therefore
          mu_k = normalize(u + g*delta_k) = u   for ANY value of the gate g.
      No special initialization of g is required for the guarantee to hold.
      There is no separate "delta_k layer" in the code to zero as a unit -- the
      2-hidden-layer _MLP emits all n_comp*5 outputs from one shared final
      nn.Linear -- so this is implemented as row-level zeroing on that existing
      layer, per the spec's precise implementation note.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.egnn import EGNNEmbedding
from src.models.directional_head import (_log_vmf_norm_3d, _sample_vmf_3d, _MLP)
from smearing_resolution.architecture_experiments.v23_final import config as C
from smearing_resolution.architecture_experiments.v23_final.data_pipeline import (
    N_PHYS, N_PHYS_COND, PHYS_SIGMA_A, PHYS_SIGMA_N,
)

_LOG_2PI = math.log(2.0 * math.pi)

# ------------------------------------------------- conditioning-vector slices
Z_END = C.D_LATENT                              # 384
PHYS_END = Z_END + N_PHYS_COND                  # 393
MOMENT_COLS = slice(Z_END + 5, PHYS_END)        # 389:393  (the 4 new scalars)
SIGMA_COLS = slice(PHYS_END, PHYS_END + C.D_SIGMA_EMB)   # 393:409
assert PHYS_END + C.D_SIGMA_EMB == C.D_AUG, (PHYS_END, C.D_SIGMA_EMB, C.D_AUG)


# ==================================================== sigma embedding S{head}
def sigma_sinusoidal_embed(sigma_A, sigma_n, n_freq=None, sigma0=None,
                           sn_floor=None):
    """S{head}, exact formula, revised per external review to use a fixed
    sinusoidal (Fourier) embedding rather than raw-scalar concatenation:

        e(v) = [sin(v f_1), cos(v f_1), ..., sin(v f_4), cos(v f_4)],
        f_j  = pi * 2^(j-1),  j = 1..4

    applied SEPARATELY to log(sigma_A + sigma_0) and log(sigma_n + eps_n)
    (8 dims each, 16 total), replacing the two raw scalars.  sigma_0 = 1.0 A
    matches a100_final's SIGMA0_A; eps_n matches its SIGMA_N_FLOOR.

    Both the absolute and the normalized form are carried, matching
    a100_final's PHYS_LOG_SIGMA_A / PHYS_LOG_SIGMA_N convention and for the same
    reason: the absolute value reflects how much the normalizer itself was
    corrected, the normalized value how much blur has washed out relative to
    track size.

    FiLM / adaptive-norm modulation was explicitly REJECTED for v23 as
    disproportionate -- VMFMixture and GMM1D are shallow 2-hidden-layer heads
    where the signal-dilution problem FiLM solves barely applies.
    """
    n_freq = C.N_SIGMA_FREQ if n_freq is None else n_freq
    sigma0 = C.SIGMA0_A if sigma0 is None else sigma0
    sn_floor = C.SIGMA_N_FLOOR if sn_floor is None else sn_floor
    B = sigma_A.shape[0]
    freqs = math.pi * torch.pow(
        2.0, torch.arange(n_freq, device=sigma_A.device, dtype=sigma_A.dtype))
    parts = []
    for v in (torch.log(sigma_A.clamp(min=0.0) + sigma0),
              torch.log(sigma_n.clamp(min=0.0) + sn_floor)):
        a = v.unsqueeze(-1) * freqs                        # (B, F)
        # stack-then-reshape gives [sin f1, cos f1, sin f2, cos f2, ...]
        parts.append(torch.stack([torch.sin(a), torch.cos(a)],
                                 dim=-1).reshape(B, 2 * n_freq))
    return torch.cat(parts, dim=-1)                        # (B, 16)


# ================================================= raw-PCA axis S{fallback}
def pca_axis_torch(coords, mask):
    """The raw-PCA axis u of the (normalized, centered) real points, with the
    CLASSICAL BASELINE's sign convention -- ZERO learned parameters.

    Convention is bit-for-bit the one the reported PCA/geometric baseline uses
    (a100_final/eval.py::pca_axis_and_signed, itself eval_r7_full_sweep.py:211):
    top eigenvector of the n-normalized coordinate covariance, flipped when the
    third moment of the projections is POSITIVE.  This is what makes v23 at
    initialization mathematically identical to plain PCA *including its head/tail
    decision*, which is the guarantee S{fallback} is buying.

    Computed in-graph from the coordinates the backbone sees, i.e. AFTER the Oh
    augmentation, rather than carried as extra phys columns: phys columns are not
    rotated by apply_oh_augmentation (they are Oh-invariant by construction), so
    a carried u would silently desynchronize from the rotated cloud.  Both the
    axis and the sign-fix are equivariant under O(3) -- for R in O(3),
    sum((Rp).(Ru))^3 = sum(p.u)^3 -- so this is exact, not an approximation.

    Detached: structural, not learned.
    """
    dt = coords.dtype
    m = mask.unsqueeze(-1).to(dt)
    cnt = mask.sum(dim=1).clamp(min=1).to(dt)                     # (B,)
    centroid = (coords * m).sum(dim=1) / cnt.unsqueeze(-1)        # (B,3)
    cc = (coords - centroid.unsqueeze(1)) * m                     # (B,N,3)
    cov = torch.einsum('bni,bnj->bij', cc, cc) / cnt.view(-1, 1, 1)
    _lam, vec = torch.linalg.eigh(cov.double())                   # ascending
    u = vec[..., -1].to(dt)                                       # (B,3)
    p = (cc * u.unsqueeze(1)).sum(-1) * mask.to(dt)
    skew = p.pow(3).sum(dim=1)
    s = torch.where(skew > 0, -torch.ones_like(skew), torch.ones_like(skew))
    return F.normalize(u * s.unsqueeze(-1), dim=-1, eps=1e-8).detach()


def _zero_input_cols(linear, zero_sigma=True, zero_moment=True):
    """Zero the first-Linear weight columns for the new input blocks (A) and (B).
    Biases are untouched -- a bias is not an input column, and zeroing it would
    change the v22-equivalent behaviour we are trying to reproduce exactly."""
    with torch.no_grad():
        if zero_sigma:
            linear.weight[:, SIGMA_COLS].zero_()
        if zero_moment:
            linear.weight[:, MOMENT_COLS].zero_()


# ================================================================ direction
class VMFMixtureV23(nn.Module):
    """v22's VMFMixture (K=4, 2-hidden-layer _MLP, hidden=256) with:

      * the 16 embedded sigma dims zero-init'd in the first Linear   [S{head}]
      * the 4 new higher-moment dims zero-init'd in the first Linear [build p1]
      * mu reparameterized as a zero-initialized residual on the raw-PCA axis:

            mu_k = normalize( u + g(sigma) * delta_k(z') )          [S{fallback}]

        delta_k is the MLP's raw (previously final) mu output for component k;
        g in [0,1] is a learned, sigma-aware gate -- one sigmoid-activated
        scalar per component, taking the embedded sigma features as input -- that
        the network can use to shrink the residual toward zero under high
        uncertainty.  g -> 1 with an expressive delta recovers fully
        unconstrained v22 behaviour wherever the data supports it (low blur,
        where v22's real strength lives); g -> 0 under high blur recovers
        graceful PCA-like degradation instead of arbitrary collapse.

    kappa and the mixture logits are UNCHANGED in form from v22
    (softplus(.) + 1e-2, raw logits) and their rows of the shared final Linear
    keep their normal initialization.
    """

    def __init__(self, d_in, n_comp=4, hidden=128, d_sigma=None,
                 zero_init_mu_rows=True, zero_init_sigma_cols=True,
                 zero_init_moment_cols=True):
        super().__init__()
        self.K = n_comp
        self.d_sigma = C.D_SIGMA_EMB if d_sigma is None else d_sigma
        self.net = _MLP(d_in, n_comp * 5, hidden)   # per comp: mu(3), kappa, logit
        self.gate = nn.Linear(self.d_sigma, n_comp)
        _zero_input_cols(self.net.net[0], zero_init_sigma_cols,
                         zero_init_moment_cols)
        if zero_init_mu_rows:
            self.zero_init_mu_rows()

    def zero_init_mu_rows(self):
        """Zero ONLY the weight/bias rows of the shared final nn.Linear that
        correspond to the 3 mu-output dimensions per component, leaving the
        kappa/logit rows at their normal initialization.  (The spec notes zeroing
        the entire final layer is equally safe -- then kappa = softplus(0)+eps
        ~ 0.70 and logits are uniform, both reasonable -- but row-level keeps
        kappa/logits at their tuned v22 initialization, so that is what v23 does.)
        Outputs are laid out INTERLEAVED per component: [mu(3), kappa, logit]*K.
        """
        last = self.net.net[-1]
        rows = [k * 5 + j for k in range(self.K) for j in range(3)]
        with torch.no_grad():
            last.weight[rows, :] = 0.0
            last.bias[rows] = 0.0

    def _params(self, z, u, sigma_emb):
        """z: (B,D_AUG); u: (B,3) unit raw-PCA axis; sigma_emb: (B,16)."""
        B = z.shape[0]
        o = self.net(z).view(B, self.K, 5)
        delta = o[..., :3]                                     # (B,K,3)
        g = torch.sigmoid(self.gate(sigma_emb)).unsqueeze(-1)  # (B,K,1) in [0,1]
        mu = F.normalize(u.unsqueeze(1) + g * delta, dim=-1, eps=1e-8)
        kappa = F.softplus(o[..., 3]) + 1e-2                   # (B,K) > 0
        logits = o[..., 4]                                     # (B,K)
        return mu, kappa, logits

    def log_prob_from_params(self, mu, kappa, logits, d):
        cos = (mu * d.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
        log_comp = _log_vmf_norm_3d(kappa) + kappa * cos
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + log_comp, dim=-1)

    def log_prob(self, d, z, u, sigma_emb):
        mu, kappa, logits = self._params(z, u, sigma_emb)
        return self.log_prob_from_params(mu, kappa, logits, d)

    @staticmethod
    def kappa_penalty(kappa, logits):
        """S{calibration} item 1: L_calib = lambda * sum_k w_k kappa_k.

        A cheap, directly-tunable stand-in for the negative differential entropy
        of the mixture: the entropy of a 3-D vMF is monotonically DECREASING in
        kappa, so penalizing kappa directly is a valid proxy and avoids needing
        the vMF entropy's closed form.  Applied uniformly across the training
        set -- including easy, low-sigma examples -- directly counteracting the
        runaway-confidence-when-accurate mechanism, and touching no held-out
        data at all.  Returns (B,); the caller scales by lambda.
        """
        return (F.softmax(logits, dim=-1) * kappa).sum(-1)

    @torch.no_grad()
    def sample_from_params(self, mu, kappa, logits, n):
        B, K = kappa.shape
        w = F.softmax(logits, dim=-1)
        idx = torch.multinomial(w, n, replacement=True).t()     # (n,B)
        br = torch.arange(B, device=mu.device).unsqueeze(0).expand(n, B)
        return _sample_vmf_3d(mu[br, idx], kappa[br, idx])      # (n,B,3)

    @torch.no_grad()
    def sample(self, n, z, u, sigma_emb):
        return self.sample_from_params(*self._params(z, u, sigma_emb), n)

    @staticmethod
    def mode_axis(mu, logits):
        """Analytic mode readout: the highest-weight component's mean (v22's
        eval measured this ~0.3 deg sharper than the sample mean)."""
        top = logits.argmax(dim=-1)
        br = torch.arange(mu.shape[0], device=mu.device)
        return F.normalize(mu[br, top], dim=-1, eps=1e-8)


# =================================================================== energy
class GMM1DV23(nn.Module):
    """v22's GMM1D (K=3) over log-energy, with the same two input-column
    zero-inits as the direction head.  No residual reparameterization: there is
    no classical structural fallback for energy analogous to the PCA axis, and
    the spec's S{fallback} is explicitly about the direction mean only."""

    def __init__(self, d_in, n_comp=3, hidden=128, zero_init_sigma_cols=True,
                 zero_init_moment_cols=True):
        super().__init__()
        self.K = n_comp
        self.net = _MLP(d_in, n_comp * 3, hidden)   # per comp: mean, log_std, logit
        _zero_input_cols(self.net.net[0], zero_init_sigma_cols,
                         zero_init_moment_cols)

    def _params(self, z):
        o = self.net(z).view(z.shape[0], self.K, 3)
        return o[..., 0], o[..., 1].clamp(-7.0, 3.0), o[..., 2]

    def log_prob(self, y, z):
        mean, log_std, logits = self._params(z)
        inv = torch.exp(-log_std)
        log_comp = (-0.5 * ((y.unsqueeze(1) - mean) * inv) ** 2 - log_std
                    - 0.5 * _LOG_2PI)
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + log_comp, dim=-1)

    @torch.no_grad()
    def sample(self, n, z):
        mean, log_std, logits = self._params(z)
        B = z.shape[0]
        idx = torch.multinomial(F.softmax(logits, dim=-1), n,
                                replacement=True).t()
        br = torch.arange(B, device=z.device).unsqueeze(0).expand(n, B)
        m = mean[br, idx]
        s = torch.exp(log_std)[br, idx]
        return m + s * torch.randn(n, B, device=z.device, dtype=z.dtype)


# ============================================================ the embedding
class V23Embedding(nn.Module):
    """Assembles the 409-dim conditioning vector and caches what the head and the
    auxiliary heads need.

    Replaces src/models/egnn.py::PhysicsAugmentedEmbedding (which is left
    untouched):  z_aug = [ z | phys_std(9) | e(sigma)(16) ].

    The 9 standardized scalar features are v22's 5 (unchanged) plus the 4 new
    higher-moment scalars.  The two RAW sigma columns of the phys block are NOT
    standardized -- they feed only the fixed sinusoidal embedding -- and are
    DETACHED (they are a known property of the observation, not a signal to
    backpropagate through), matching a100_final's treatment.

    Exposes .k / .n_max / .n_phys / .log_energy for the eval chain-walk and
    .last_z for the auxiliary heads, exactly like the v22 wrapper it replaces.
    """

    def __init__(self, base_embedding, n_max, k, phys_mean, phys_std,
                 log_energy=True):
        super().__init__()
        self.base = base_embedding
        self.n_max = n_max
        self.k = k
        self.n_phys = N_PHYS
        self.log_energy = log_energy
        self.register_buffer('phys_mean', phys_mean.float())
        self.register_buffer('phys_std', phys_std.float())
        self.last_z = None
        self.last_u = None
        self.last_sigma_emb = None

    def set_phys_stats(self, phys_mean, phys_std):
        with torch.no_grad():
            self.phys_mean.copy_(phys_mean.to(self.phys_mean.device).float())
            self.phys_std.copy_(phys_std.to(self.phys_std.device).float())

    def forward(self, x_flat):
        B = x_flat.shape[0]
        core_dim = self.n_max * (3 + self.k)
        x_core = x_flat[:, :core_dim]
        phys = x_flat[:, core_dim:core_dim + self.n_phys]

        coords = x_flat[:, :self.n_max * 3].view(B, self.n_max, 3)
        # same exact-zero padding test the backbone itself uses
        mask = coords.abs().sum(dim=-1) > 0.0
        self.last_u = pca_axis_torch(coords, mask)

        sigma_A = phys[:, PHYS_SIGMA_A].detach()
        sigma_n = phys[:, PHYS_SIGMA_N].detach()
        self.last_sigma_emb = sigma_sinusoidal_embed(sigma_A, sigma_n)

        z = self.base(x_core)                                    # (B, 384)
        phys_cond = (phys[:, :N_PHYS_COND] - self.phys_mean) / \
            self.phys_std.clamp(min=1e-8)
        z_aug = torch.cat([z, phys_cond, self.last_sigma_emb], dim=-1)
        self.last_z = z_aug
        return z_aug


# ============================================================ the posterior
class DirectionalPosteriorV23(nn.Module):
    """Drop-in for v22's DirectionalPosterior: p_E(logE|z) * p_dir(dir on S^2|z).

    Same API as v22 (.loss(theta, condition) -> (B,) NLL, .sample(shape,
    condition) -> (n,B,4)) so the training loop and the eval chain-walk are
    unchanged, plus:
        .loss_terms(theta, condition) -> (nll, kappa_penalty) in ONE forward
        .params(condition)            -> everything eval needs
    """

    def __init__(self, embedding_net, d_cond=None, n_dir_comp=None,
                 n_e_comp=None, hidden=None, zero_init_mu_rows=None,
                 zero_init_sigma_cols=None, zero_init_moment_cols=None):
        super().__init__()
        d_cond = C.D_AUG if d_cond is None else d_cond
        n_dir_comp = C.N_DIR_COMP if n_dir_comp is None else n_dir_comp
        n_e_comp = C.N_E_COMP if n_e_comp is None else n_e_comp
        hidden = C.HEAD_HIDDEN if hidden is None else hidden
        zm = C.ZERO_INIT_MU_ROWS if zero_init_mu_rows is None else zero_init_mu_rows
        zs = (C.ZERO_INIT_SIGMA_COLS if zero_init_sigma_cols is None
              else zero_init_sigma_cols)
        zc = (C.ZERO_INIT_MOMENT_COLS if zero_init_moment_cols is None
              else zero_init_moment_cols)
        self.embedding_net = embedding_net
        self.energy = GMM1DV23(d_cond, n_e_comp, hidden,
                               zero_init_sigma_cols=zs,
                               zero_init_moment_cols=zc)
        self.direction = VMFMixtureV23(d_cond, n_dir_comp, hidden,
                                       zero_init_mu_rows=zm,
                                       zero_init_sigma_cols=zs,
                                       zero_init_moment_cols=zc)

    # ------------------------------------------------------------- internals
    def _encode(self, condition):
        z = self.embedding_net(condition)
        return z, self.embedding_net.last_u, self.embedding_net.last_sigma_emb

    def params(self, condition):
        z, u, se = self._encode(condition)
        mu, kappa, logits = self.direction._params(z, u, se)
        return dict(z=z, u=u, sigma_emb=se, mu=mu, kappa=kappa, logits=logits)

    # ------------------------------------------------------------------- API
    def loss(self, theta, condition):
        """theta = [logE, vx, vy, vz]; returns (B,) per-sample NLL. Signed
        vMF-mixture NLL from the first step, exactly as v22 -- there is no
        sign-blind option inside the deployed posterior, and the sign-aware
        curriculum applies only to the separate auxiliary DirectionHead."""
        return self.loss_terms(theta, condition)[0]

    def loss_terms(self, theta, condition):
        z, u, se = self._encode(condition)
        logE = theta[:, 0]
        d = F.normalize(theta[:, 1:4], dim=-1, eps=1e-8)
        mu, kappa, logits = self.direction._params(z, u, se)
        nll = -(self.energy.log_prob(logE, z)
                + self.direction.log_prob_from_params(mu, kappa, logits, d))
        return nll, VMFMixtureV23.kappa_penalty(kappa, logits)

    @torch.no_grad()
    def sample(self, shape, condition):
        """(shape[0], B, 4) samples = [logE, vx, vy, vz]."""
        z, u, se = self._encode(condition)
        n = shape[0]
        E = self.energy.sample(n, z)                       # (n,B)
        d = self.direction.sample(n, z, u, se)             # (n,B,3)
        return torch.cat([E.unsqueeze(-1), d], dim=-1)

    @torch.no_grad()
    def predict(self, condition, n_samples=None):
        """Point estimates + posterior samples for eval, in one forward.

        Returns dict with
            mean_dir : (B,3) normalize(mean of direction samples) -- the same
                       readout ContinuousEvaluator3D's angular_error_deg uses,
                       i.e. the readout v22's 6.8 deg / 97.0% came from
            mode_dir : (B,3) analytic highest-weight component mean
            samples  : (n,B,3) direction samples (for coverage/ECE)
        """
        n_samples = C.EVAL_N_SAMPLES if n_samples is None else n_samples
        p = self.params(condition)
        s = self.direction.sample_from_params(p['mu'], p['kappa'], p['logits'],
                                              n_samples)
        return dict(mean_dir=F.normalize(s.mean(dim=0), dim=-1, eps=1e-8),
                    mode_dir=VMFMixtureV23.mode_axis(p['mu'], p['logits']),
                    samples=s, kappa=p['kappa'], logits=p['logits'],
                    u=p['u'])


# ================================================================== factory
def build_v23(n_max, phys_mean=None, phys_std=None, device='cpu',
              hidden_dim=None, n_layers=None, k=None, **head_kw):
    """Build the full v23 estimator: GVP-EGNN backbone (v22 hyperparameters,
    unchanged) -> V23Embedding -> DirectionalPosteriorV23."""
    hidden_dim = C.HIDDEN_DIM if hidden_dim is None else hidden_dim
    n_layers = C.N_LAYERS if n_layers is None else n_layers
    k = C.K_MAX if k is None else k
    if phys_mean is None:
        phys_mean = torch.zeros(N_PHYS_COND)
    if phys_std is None:
        phys_std = torch.ones(N_PHYS_COND)
    base = EGNNEmbedding(
        n_max=n_max, hidden_dim=hidden_dim, n_layers=n_layers, k=k,
        n_heads=C.N_HEADS, d_proj=C.D_PROJ, d_latent=C.D_LATENT,
        use_gvp=C.USE_GVP, v_dim=C.V_DIM, use_axis_feats=C.USE_AXIS_FEATS,
        n_axes=C.N_AXES,
    )
    emb = V23Embedding(base, n_max=n_max, k=k, phys_mean=phys_mean,
                       phys_std=phys_std, log_energy=C.LOG_ENERGY)
    head_kw.setdefault('d_cond', C.D_AUG)
    return DirectionalPosteriorV23(emb, **head_kw).to(device)
