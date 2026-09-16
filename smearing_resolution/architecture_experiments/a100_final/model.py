"""a100_final: consolidated architecture (DiagModelV2).

Differences vs. the winner (train_diag_r7_trunc_only.py::DiagModel):

  (1) SIGMA CONDITIONING.  Two DETACHED scalars -- log(sigma_A + 1 A) and
      log(sigma_n + 1e-3) where sigma_n = sigma_A/scale is the blur in the
      network's own normalized units -- are concatenated into the direction
      head's trunk input, so they reach BOTH coef_out (the candidate-vector
      WEIGHTING pathway) and kl_out (kappa/logits).  The winner had sigma
      nowhere in the weighting pathway.  The corresponding weight slice of
      the trunk's first Linear is ZERO-INITIALIZED, so at step 0 the model is
      bit-identical to one without the feature.

  (3) 4th-MOMENT CANDIDATE VECTOR.  A new, 13th pool vector derived from the
      projection-weighted second-moment tensor
          M = mean_i (xc_i . u)^2 * outer(xc_i, xc_i),      u = raw-PCA axis
      ANALYTICALLY deconvolved for isotropic Gaussian blur (full derivation in
      README S3; the correction is NOT zero for this construction, unlike the
      plain 3rd central moment which we show is exactly blur-unbiased):
          A_hat = M_obs - s2 ( <q^2> I + 4 <q^2> u u^T + C_hat )
                        - s2^2 ( I + 2 u u^T )
      with s2 = sigma_n^2 (n-1)/n, <q^2> = <p^2>_obs - s2, C_hat = C_obs - s2 I.
      Dominant eigenvector by power iteration, sign-fixed by the (blur-unbiased)
      third moment of the projections.  Its Gram-matrix columns into the trunk
      and its coefficient rows out of coef_out are ZERO-INITIALIZED, so at step
      0 it contributes exactly nothing.

  (4) POSTERIOR.  vMF mixture widened K=4 -> K=10 (default), plus an optional
      Implicit-PDF-style grid posterior over a subdivided icosahedron
      (POSTERIOR_TYPE=grid).

  (6) HIDDEN_DIM / N_LAYERS / posterior type / k_max all constructor args.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.egnn import EGNNBackbone, ScalarReadout, build_edges_precomputed
from src.models.directional_head import _log_vmf_norm_3d, _sample_vmf_3d  # noqa: F401
from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    N_PHYS, PHYS_LOG_SIGMA_A, PHYS_LOG_SIGMA_N, PHYS_SIGMA_N,
)

N_HEADS_ATTN = 8
N_AXES = 2          # principal axes of the BACKBONE-updated coordinates
N_AXES_INPUT = 2    # principal axes of the RAW INPUT coordinates
# TWO new 4th-moment axes: the blur-CORRECTED one and the UNCORRECTED one.
# validate_math.py TEST B1 confirms the closed-form correction removes the bias
# (up to 196x bias reduction), but TEST B2 shows it also amplifies VARIANCE, so
# on a single realization at extreme sigma/R_g the uncorrected tensor -- which
# is biased-but-shrunk toward the PCA axis -- actually has the smaller angle
# error.  Rather than hard-code a shrinkage, we hand BOTH to the pool and let
# the (now sigma-conditioned, see change 1) coefficient head pick per blur
# level.  This is exactly the bias/variance trade the sigma feature exists to
# arbitrate, and it costs one extra eigen-solve on an already-computed tensor.
N_KURT = 2
N_VECTORS = N_HEADS_ATTN + N_AXES + N_AXES_INPUT + N_KURT     # 14
V_REF_IDX = N_HEADS_ATTN + N_AXES                            # 10
KURT_CORR_IDX = N_VECTORS - 2                                # 12
KURT_RAW_IDX = N_VECTORS - 1                                 # 13
NEW_VEC_IDX = (KURT_CORR_IDX, KURT_RAW_IDX)
N_SIGMA_FEATS = 2


# ---------------------------------------------------------------- utilities
def _scatter_mean_vec(src, batch_vec, B, counts):
    out = torch.zeros(B, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(src), src)
    return out / counts.unsqueeze(-1)


def _scatter_sum_scalar(src, batch_vec, B):
    out = torch.zeros(B, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, batch_vec, src)
    return out


def _power_iter(A, n_iter=30, v0=None):
    """Dominant eigenvector of a batch of symmetric PSD (B,3,3)."""
    B = A.shape[0]
    v = torch.ones(B, 3, device=A.device, dtype=A.dtype) if v0 is None else v0
    v = F.normalize(v, dim=-1, eps=1e-8)
    for _ in range(n_iter):
        v = F.normalize(torch.bmm(A, v.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-8)
    return v


def _skew_sign_fix(v, xc, batch_vec, B):
    proj = (xc * v[batch_vec]).sum(-1)
    skew = _scatter_sum_scalar(proj.pow(3), batch_vec, B)
    s = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
    return v * s.unsqueeze(-1)


def icosphere(n_subdiv=3):
    """Unit-sphere grid by recursive icosahedron subdivision.
    n_subdiv=2 -> 162 pts, 3 -> 642, 4 -> 2562."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [list(v) for v in verts]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            p = [(verts[a][i] + verts[b][i]) / 2.0 for i in range(3)]
            verts.append(p)
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(n_subdiv):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    V = torch.tensor(verts, dtype=torch.float32)
    return F.normalize(V, dim=-1)


# ------------------------------------------------------------- vector pool
class EquivariantVectorPoolV2(nn.Module):
    """8 attention-weighted centroids + 2 backbone PCA axes + 2 raw-input PCA
    axes + 1 blur-corrected 4th-moment axis = 13 candidate direction vectors."""

    def __init__(self, hidden_dim, n_heads=N_HEADS_ATTN, n_axes=N_AXES,
                 axis_iters=30):
        super().__init__()
        self.n_heads = n_heads
        self.n_axes = n_axes
        self.axis_iters = axis_iters
        self.attn_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
            for _ in range(n_heads)])

    @staticmethod
    def _scatter_max(src, index, num_groups):
        out = torch.full((num_groups,), float('-inf'), device=src.device, dtype=src.dtype)
        out.scatter_reduce_(0, index, src, reduce='amax')
        return out[index]

    def _attention_centroids(self, h, x, batch_vec, B):
        vecs = []
        for k in range(self.n_heads):
            alpha = self.attn_mlps[k](h).squeeze(-1)
            alpha_exp = torch.exp(alpha - self._scatter_max(alpha, batch_vec, B))
            alpha_sum = _scatter_sum_scalar(alpha_exp, batch_vec, B)
            w = alpha_exp / alpha_sum[batch_vec].clamp(min=1e-8)
            wx = w.unsqueeze(-1) * x
            v = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
            v.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(wx), wx)
            vecs.append(v)
        return vecs

    @staticmethod
    def _centered(x, batch_vec, counts, B):
        centroid = _scatter_mean_vec(x, batch_vec, B, counts)
        return x - centroid[batch_vec]

    def _principal_axes(self, xc, batch_vec, counts, B, n_axes):
        outer = xc.unsqueeze(-1) * xc.unsqueeze(-2)
        Csum = torch.zeros(B, 3, 3, device=xc.device, dtype=xc.dtype)
        Csum.scatter_add_(0, batch_vec.view(-1, 1, 1).expand_as(outer), outer)
        C = Csum / counts.view(B, 1, 1)
        axes, C_work = [], C
        for _a in range(n_axes):
            v = _power_iter(C_work, self.axis_iters)
            lam = torch.einsum('bi,bij,bj->b', v, C_work, v)
            v = _skew_sign_fix(v, xc, batch_vec, B)
            axes.append(v)
            C_work = C_work - lam.view(B, 1, 1) * torch.bmm(v.unsqueeze(-1), v.unsqueeze(1))
        return axes, C

    def _kurtosis_axis(self, xc, u, C_obs, batch_vec, counts, B, sigma_n):
        """Blur-deconvolved projection-weighted second-moment axis. See README S3.

        s2 = sigma_n^2 (n-1)/n   (exact centered-noise variance per coordinate)
        A_hat = M_obs - s2(<q^2> I + 4<q^2> u u^T + C_hat) - s2^2 (I + 2 u u^T)
        """
        dtype, device = xc.dtype, xc.device
        n = counts
        s2 = (sigma_n ** 2) * ((n - 1.0) / n.clamp(min=1.0))          # (B,)
        p = (xc * u[batch_vec]).sum(-1)                                # (total_real,)
        p2 = p * p
        p2_mean = _scatter_sum_scalar(p2, batch_vec, B) / n            # (B,)
        outer = xc.unsqueeze(-1) * xc.unsqueeze(-2)
        Msum = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        Msum.scatter_add_(0, batch_vec.view(-1, 1, 1).expand_as(outer),
                          p2.view(-1, 1, 1) * outer)
        M_obs = Msum / n.view(B, 1, 1)

        I3 = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3)
        uu = torch.bmm(u.unsqueeze(-1), u.unsqueeze(1))                # (B,3,3)
        q2 = (p2_mean - s2).clamp(min=0.0)                             # (B,)
        C_hat = C_obs - s2.view(B, 1, 1) * I3
        s2b = s2.view(B, 1, 1)
        q2b = q2.view(B, 1, 1)
        A = M_obs - s2b * (q2b * I3 + 4.0 * q2b * uu + C_hat) - (s2b ** 2) * (I3 + 2.0 * uu)

        out = []
        for T in (A, M_obs):
            T = 0.5 * (T + T.transpose(1, 2))
            # the correction can make A indefinite; a multiple of I shifts every
            # eigenvalue equally and so leaves eigenVECTORS untouched, while
            # making power iteration converge to the algebraically-largest one.
            shift = torch.sqrt((T * T).sum(dim=(1, 2))) + 1e-12
            v = _power_iter(T + shift.view(B, 1, 1) * I3, self.axis_iters)
            out.append(_skew_sign_fix(v, xc, batch_vec, B))
        return out   # [corrected, uncorrected]

    def forward(self, h, x_final, batch_vec, real_counts, x_input, sigma_n):
        B = real_counts.shape[0]
        counts = real_counts.clamp(min=1).to(x_input.dtype)
        vecs = self._attention_centroids(h, x_final, batch_vec, B)
        xc_f = self._centered(x_final, batch_vec, counts, B)
        ax_f, _ = self._principal_axes(xc_f, batch_vec, counts, B, self.n_axes)
        vecs += ax_f
        xc_i = self._centered(x_input, batch_vec, counts, B)
        ax_i, C_obs = self._principal_axes(xc_i, batch_vec, counts, B, N_AXES_INPUT)
        vecs += ax_i
        vecs += self._kurtosis_axis(xc_i, ax_i[0], C_obs, batch_vec, counts, B, sigma_n)
        return torch.stack(vecs, dim=1), xc_i


# --------------------------------------------------------------- posteriors
class VMFMixtureHead(nn.Module):
    """K-component vMF mixture whose means are reweighted combinations of the
    candidate vectors (identical in form to the winner, K widened)."""

    def __init__(self, n_vectors, d_trunk, n_comp=10):
        super().__init__()
        self.n_vectors, self.n_comp = n_vectors, n_comp
        self.coef_out = nn.Linear(d_trunk, n_comp * n_vectors)
        self.kl_out = nn.Linear(d_trunk, n_comp * 2)

    def zero_init_vector(self, idxs):
        with torch.no_grad():
            w = self.coef_out.weight.view(self.n_comp, self.n_vectors, -1)
            b = self.coef_out.bias.view(self.n_comp, self.n_vectors)
            for idx in (idxs if isinstance(idxs, (tuple, list)) else [idxs]):
                w[:, idx, :].zero_()
                b[:, idx].zero_()

    def forward(self, t, V):
        B = V.shape[0]
        coeffs = self.coef_out(t).view(B, self.n_comp, self.n_vectors)
        mu = F.normalize(torch.bmm(coeffs, V), dim=-1, eps=1e-8)
        kl = self.kl_out(t).view(B, self.n_comp, 2)
        return mu, F.softplus(kl[..., 0]) + 1e-3, kl[..., 1]


class GridPosteriorHead(nn.Module):
    """Implicit-PDF-style (Murphy et al. 2021) grid posterior on S^2.

    The scoring function is f([d . V_1 ... d . V_n], context) -- a function of
    ROTATION-INVARIANT quantities only, so score(Rd | RV) = score(d | V) and the
    posterior stays equivariant even though the grid itself is lab-fixed.
    """

    def __init__(self, n_vectors, d_trunk, n_subdiv=3, d_ctx=64, hidden=128):
        super().__init__()
        self.register_buffer('grid', icosphere(n_subdiv))       # (G,3)
        self.n_grid = self.grid.shape[0]
        self.ctx = nn.Linear(d_trunk, d_ctx)
        self.score = nn.Sequential(
            nn.Linear(n_vectors + d_ctx, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.log_cell = math.log(4.0 * math.pi / self.n_grid)
        self.n_vectors = n_vectors

    def zero_init_vector(self, idxs):
        with torch.no_grad():
            for idx in (idxs if isinstance(idxs, (tuple, list)) else [idxs]):
                self.score[0].weight[:, idx].zero_()

    def _score(self, d, V, c):
        """d:(B,G,3) or (B,3); V:(B,N,3); c:(B,d_ctx) -> (B,G) / (B,)"""
        single = (d.dim() == 2)
        if single:
            d = d.unsqueeze(1)
        dots = torch.einsum('bgd,bnd->bgn', d, V)
        cc = c.unsqueeze(1).expand(-1, d.shape[1], -1)
        s = self.score(torch.cat([dots, cc], dim=-1)).squeeze(-1)
        return s.squeeze(1) if single else s

    def forward(self, t, V):
        c = self.ctx(t)
        g = self.grid.unsqueeze(0).expand(V.shape[0], -1, -1)
        return self._score(g, V, c), c

    def log_prob(self, d, t_or_c, V, grid_scores, c):
        """Normalized log density at arbitrary unit d, Riemann-normalized."""
        s_d = self._score(d, V, c)
        return s_d - torch.logsumexp(grid_scores, dim=-1) - self.log_cell

    @torch.no_grad()
    def sample(self, n, grid_scores, jitter=True):
        """(n,B,3) unit samples; jitter spreads within a cell."""
        w = F.softmax(grid_scores, dim=-1)
        idx = torch.multinomial(w, n, replacement=True).t()         # (n,B)
        d = self.grid[idx]                                          # (n,B,3)
        if jitter:
            # cell angular radius ~ sqrt(4pi/G/pi) rad
            r = math.sqrt(4.0 / self.n_grid)
            d = F.normalize(d + r * 0.5 * torch.randn_like(d), dim=-1, eps=1e-8)
        return d

    @torch.no_grad()
    def mean_axis(self, grid_scores):
        """Sign-blind mean axis: top eigenvector of sum_g w_g g g^T."""
        w = F.softmax(grid_scores, dim=-1)                          # (B,G)
        g = self.grid
        T = torch.einsum('bg,gi,gj->bij', w, g, g)
        return _power_iter(T, 30)


# ------------------------------------------------------------------- trunk
class DirectionTrunk(nn.Module):
    """Gram-matrix invariants + pooled scalars + sigma features -> hidden."""

    def __init__(self, n_vectors, d_scalar, n_sigma=N_SIGMA_FEATS, hidden=256,
                 new_vec_idx=None):
        super().__init__()
        self.n_vectors = n_vectors
        idx = torch.triu_indices(n_vectors, n_vectors)
        self.register_buffer('triu_i', idx[0])
        self.register_buffer('triu_j', idx[1])
        self.n_gram = idx.shape[1]
        self.d_scalar = d_scalar
        self.n_sigma = n_sigma
        in_dim = self.n_gram + d_scalar + n_sigma
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.out_dim = hidden
        # zero-init discipline: the sigma slice and every Gram entry touching
        # the new candidate vector contribute exactly 0 at step 0.
        with torch.no_grad():
            self.net[0].weight[:, self.n_gram + d_scalar:].zero_()
            if new_vec_idx is not None:
                idxs = (new_vec_idx if isinstance(new_vec_idx, (tuple, list))
                        else [new_vec_idx])
                m = torch.zeros(self.n_gram, dtype=torch.bool)
                for i in idxs:
                    m |= (self.triu_i == i) | (self.triu_j == i)
                gram_w = self.net[0].weight[:, :self.n_gram]
                gram_w[:, m] = 0.0

    def features(self, V, scalar_feats, sigma_feats):
        gram = torch.bmm(V, V.transpose(1, 2))
        gram_flat = gram[:, self.triu_i, self.triu_j]
        return torch.cat([gram_flat, scalar_feats, sigma_feats], dim=-1)

    def forward(self, V, scalar_feats, sigma_feats):
        feats = self.features(V, scalar_feats, sigma_feats)
        return self.net(feats), feats


# ------------------------------------------------------------------- model
class DiagModelV2(nn.Module):
    def __init__(self, n_max, k_max=24, hidden_dim=112, n_layers=6,
                 coef_hidden=256, n_dir_comp=10, posterior_type='vmf_mixture',
                 grid_subdiv=3, n_phys=N_PHYS, zero_init_new_vector=True,
                 amp=False):
        super().__init__()
        self.n_max, self.k, self.n_phys = n_max, k_max, n_phys
        # bf16 autocast is applied to the EGNN backbone ONLY.  Everything
        # downstream (power iteration, Gram matrix, vMF normalizer) runs in
        # fp32: bf16's 8-bit mantissa is not enough for 30-step power
        # iteration or for _log_vmf_norm_3d's dual-regime branch.
        self.amp = amp
        self.cfg = dict(n_max=n_max, k_max=k_max, hidden_dim=hidden_dim,
                        n_layers=n_layers, coef_hidden=coef_hidden,
                        n_dir_comp=n_dir_comp, posterior_type=posterior_type,
                        grid_subdiv=grid_subdiv, n_phys=n_phys)
        self.core_dim = n_max * (3 + k_max)
        self.posterior_type = posterior_type
        self.backbone = EGNNBackbone(hidden_dim=hidden_dim, n_layers=n_layers,
                                     use_gvp=False)
        self.scalar_readout = ScalarReadout(hidden_dim)
        self.vector_pool = EquivariantVectorPoolV2(hidden_dim)
        new_idx = NEW_VEC_IDX if zero_init_new_vector else None
        self.trunk = DirectionTrunk(N_VECTORS, self.scalar_readout.output_dim,
                                    hidden=coef_hidden, new_vec_idx=new_idx)
        if posterior_type == 'vmf_mixture':
            self.posterior = VMFMixtureHead(N_VECTORS, self.trunk.out_dim, n_dir_comp)
        elif posterior_type == 'grid':
            self.posterior = GridPosteriorHead(N_VECTORS, self.trunk.out_dim,
                                               n_subdiv=grid_subdiv)
        else:
            raise ValueError(f"unknown POSTERIOR_TYPE={posterior_type}")
        if zero_init_new_vector:
            self.posterior.zero_init_vector(NEW_VEC_IDX)
        inv_dim = self.trunk.in_dim
        self.sign_head = nn.Sequential(nn.Linear(inv_dim + 1, 64), nn.SiLU(),
                                       nn.Linear(64, 1))
        self.energy_head = nn.Sequential(nn.Linear(inv_dim, 64), nn.SiLU(),
                                         nn.Linear(64, 2))
        # The aux heads read the RAW invariant feature vector, so they need the
        # same zero-init treatment as the trunk: both the sigma slice and every
        # Gram column touching a new candidate vector.  (inv_feats layout is
        # [gram | pooled scalars | sigma]; sign_head appends range_mid AFTER,
        # so the gram block is still the leading n_gram columns.)
        with torch.no_grad():
            ng = self.trunk.n_gram
            gm = torch.zeros(ng, dtype=torch.bool)
            if zero_init_new_vector:
                for i in NEW_VEC_IDX:
                    gm |= (self.trunk.triu_i == i) | (self.trunk.triu_j == i)
            for head in (self.sign_head, self.energy_head):
                head[0].weight[:, inv_dim - N_SIGMA_FEATS:inv_dim].zero_()
                blk = head[0].weight[:, :ng]
                blk[:, gm] = 0.0

    # ------------------------------------------------------------- forward
    def forward(self, x_flat, ablate_new_vector=False, ablate_sigma=False):
        B = x_flat.shape[0]
        c_end = self.n_max * 3
        coords = x_flat[:, :c_end].view(B, self.n_max, 3)
        neighbor_idx = x_flat[:, c_end:self.core_dim].view(B, self.n_max, self.k).long()
        phys = x_flat[:, self.core_dim:]
        mask = coords.abs().sum(dim=-1) > 0.0

        sigma_feats = phys[:, [PHYS_LOG_SIGMA_A, PHYS_LOG_SIGMA_N]].detach()
        if ablate_sigma:
            sigma_feats = torch.zeros_like(sigma_feats)
        sigma_n = phys[:, PHYS_SIGMA_N].detach()

        edge_index, batch_vec, flat_coords, real_counts, _ = build_edges_precomputed(
            coords, mask, neighbor_idx)
        if self.amp and flat_coords.is_cuda:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h_final, x_final = self.backbone(flat_coords, edge_index,
                                                 real_counts, batch_vec)
            h_final, x_final = h_final.float(), x_final.float()
        else:
            h_final, x_final = self.backbone(flat_coords, edge_index,
                                             real_counts, batch_vec)
        e_scalar = self.scalar_readout(h_final, batch_vec, real_counts)
        V, xc_i = self.vector_pool(h_final, x_final, batch_vec, real_counts,
                                   flat_coords, sigma_n)
        if ablate_new_vector:
            V = V.clone()
            V[:, list(NEW_VEC_IDX), :] = 0.0

        t, inv_feats = self.trunk(V, e_scalar, sigma_feats)

        v_ref = V[:, V_REF_IDX, :].detach()

        # range-midpoint statistic relative to v_ref (unchanged from the winner)
        proj = (xc_i * v_ref[batch_vec]).sum(-1)
        pmax = torch.full((B,), float('-inf'), device=proj.device, dtype=proj.dtype)
        pmin = torch.full((B,), float('inf'), device=proj.device, dtype=proj.dtype)
        pmax.scatter_reduce_(0, batch_vec, proj, reduce='amax')
        pmin.scatter_reduce_(0, batch_vec, proj, reduce='amin')
        extent = (pmax - pmin).clamp(min=1e-6)
        range_mid = ((pmax + pmin) * 0.5 / extent).detach().unsqueeze(-1)

        sign_logit = self.sign_head(torch.cat([inv_feats, range_mid], dim=-1)).squeeze(-1)
        el = self.energy_head(inv_feats)

        out = dict(sign_logit=sign_logit, E_pred=el[:, 0], log_sigma=el[:, 1],
                   v_ref=v_ref, V=V, t=t, inv_feats=inv_feats)

        if self.posterior_type == 'vmf_mixture':
            mu, kappa, logits = self.posterior(t, V)
            out.update(mu=mu, kappa=kappa, logits=logits)
            axis = mixture_axis_ref(mu, kappa, logits)
        else:
            gs, ctx = self.posterior(t, V)
            out.update(grid_scores=gs, grid_ctx=ctx)
            axis = self.posterior.mean_axis(gs)

        flip = torch.sign((axis * v_ref).sum(-1, keepdim=True))
        flip = torch.where(flip == 0, torch.ones_like(flip), flip)
        out['axis_ref_aligned'] = axis * flip
        return out

    @torch.no_grad()
    def predict_signed(self, x_flat):
        o = self.forward(x_flat)
        sgn = torch.where(torch.sigmoid(o['sign_logit']) > 0.5, 1.0, -1.0).unsqueeze(-1)
        return o['axis_ref_aligned'] * sgn

    # -------------------------------------------------------------- losses
    def direction_nll_axis(self, out, d):
        """Sign-blind (axis) NLL of the true direction under the posterior."""
        if self.posterior_type == 'vmf_mixture':
            return mixture_nll_axis(out['mu'], out['kappa'], out['logits'], d)
        gs, V, c = out['grid_scores'], out['V'], out['grid_ctx']
        lp = self.posterior.log_prob(d, None, V, gs, c)
        lm = self.posterior.log_prob(-d, None, V, gs, c)
        return -torch.maximum(lp, lm)

    @torch.no_grad()
    def sample_direction(self, out, n):
        if self.posterior_type == 'vmf_mixture':
            return sample_mixture(out['mu'], out['kappa'], out['logits'], n)
        return self.posterior.sample(n, out['grid_scores'])


# ----------------------------------------------------- mixture loss helpers
def _signed_nll(mu, kappa, logits, d):
    cos = (mu * d.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
    log_comp = _log_vmf_norm_3d(kappa) + kappa * cos
    return -torch.logsumexp(F.log_softmax(logits, dim=-1) + log_comp, dim=-1)


def mixture_nll_axis(mu, kappa, logits, d):
    return torch.minimum(_signed_nll(mu, kappa, logits, d),
                         _signed_nll(mu, kappa, logits, -d))


def mixture_axis_ref(mu, kappa, logits):
    w = F.softmax(logits, dim=-1)
    top = w.argmax(1)
    br = torch.arange(w.shape[0], device=w.device)
    mu_top = F.normalize(mu[br, top], dim=-1)
    s = torch.sign((mu * mu_top.unsqueeze(1)).sum(-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return F.normalize((w.unsqueeze(-1) * (mu * s.unsqueeze(-1))).sum(1), dim=-1)


@torch.no_grad()
def sample_mixture(mu, kappa, logits, n_samples, kappa_scale=1.0):
    B = mu.shape[0]
    w = F.softmax(logits, dim=-1)
    comp = torch.multinomial(w, n_samples, replacement=True).T
    br = torch.arange(B, device=mu.device).unsqueeze(0).expand(n_samples, -1)
    return _sample_vmf_3d(mu[br, comp], (kappa[br, comp] * kappa_scale).clamp(min=1e-3))
