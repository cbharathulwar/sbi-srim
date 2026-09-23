"""Verify v23's zero-init discipline EMPIRICALLY (not by assumption).

Same discipline and the same bitwise standard used throughout a100_final
(verify_zero_init.py there), applied to v23's three new learned inputs.

Claims under test, at step 0 before any optimizer step:

  1. S{head}: the 16 sinusoidal sigma-embedding columns are zero-init'd, so
     forcing them off (or changing sigma outright) is a NO-OP on every output.
  2. Staged build plan item 1: the 4 new higher-moment feature columns are
     zero-init'd, so perturbing those features is a NO-OP on every output.
  3. S{fallback}: with the mu-output ROWS of the shared final Linear zeroed,
     mu_k == u EXACTLY for every component and for ANY value of the gate g --
     so v23 begins training mathematically identical to plain PCA, including
     its head/tail decision.
  4. The kappa / logit / energy pathway is BIT-IDENTICAL to a v22-equivalent
     head that never had the new inputs at all (built by copying the surviving
     weight columns into a fresh v22-shaped head).  This is the spec's
     "bit-identical outputs to plain v22 wherever the new inputs are zeroed".
  5. The direction readout at init equals the CLASSICAL PCA/GEOMETRIC BASELINE
     computed by a fully independent numpy implementation (metrics.py, the same
     convention the reference sweeps use).
  6. The zero-init'd slices are NOT DEAD -- each must receive non-zero gradient
     on the very first backward pass, or it could never learn.
  7. S{calibration} item 1: the kappa regularizer at lambda=0 does not change
     the forward pass, the loss value, or any gradient.

Run:  python -m smearing_resolution.architecture_experiments.v23_final.verify_zero_init
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from src.models.directional_head import GMM1D, VMFMixture
from smearing_resolution.architecture_experiments.v23_final import config as C
from smearing_resolution.architecture_experiments.v23_final.data_pipeline import (
    _smear_and_prepare_one_v23, flat_dim, pack_one,
)
from smearing_resolution.architecture_experiments.v23_final.metrics import (
    pca_axis_and_signed,
)
from smearing_resolution.architecture_experiments.v23_final.model import (
    MOMENT_COLS, SIGMA_COLS, build_v23, pca_axis_torch, sigma_sinusoidal_embed,
)

N_MAX, K_MAX, B = 120, C.K_MAX, 16
DEVICE = os.environ.get("DEVICE", "cpu")   # CPU by default: exact bitwise compare


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return bool(ok)


def synth_batch(seed=0, sigmas=None):
    """Half the batch at sigma=0 exactly, half log-uniform over [1,1000] A --
    the same construction a100_final's harness used."""
    rng = np.random.default_rng(seed)
    x = torch.zeros(B, flat_dim(N_MAX, K_MAX), dtype=torch.float64)
    if sigmas is None:
        sigmas = np.concatenate([
            np.zeros(B // 2),
            np.exp(rng.uniform(np.log(1.0), np.log(1000.0), B - B // 2))])
    raw = []
    for j in range(B):
        n = int(rng.integers(20, N_MAX))
        # an elongated, skewed, bent cloud: a plausible cascade shape, so the
        # PCA axis and the higher-moment axes are genuinely distinct
        p = rng.normal(0, 1, (n, 3)) * np.array([120., 30., 20.])
        p[:, 0] += 0.5 * np.abs(p[:, 0])                  # skew along the axis
        p[:, 1] += 0.15 * (p[:, 0] / 120.0) ** 2 * 30.0   # bend
        raw.append(p)
        co, kn, ph = _smear_and_prepare_one_v23(p, sigmas[j], N_MAX, rng)
        pack_one(x[j], co, kn, ph, N_MAX, K_MAX)
    return x.float(), sigmas, raw


def _out(m, x):
    """All step-0 outputs of interest, as a flat dict of tensors."""
    with torch.no_grad():
        p = m.params(x)
        e_mean, e_logstd, e_logit = m.energy._params(p['z'])
    return dict(mu=p['mu'], kappa=p['kappa'], logits=p['logits'],
                u=p['u'], e_mean=e_mean, e_log_std=e_logstd, e_logit=e_logit)


def check():
    torch.manual_seed(0)
    all_ok = True
    x, sigmas, raw = synth_batch(0)
    m = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    base = _out(m, x)
    n_phys = C.N_PHYS

    # ---------------------------------------------------------------- (1)
    print("\n=== 1. sigma-conditioning columns are zero-init'd (S{head}) ===")
    x_sz = x.clone()
    x_sz[:, -n_phys:][:, -2:] = 0.0                       # force sigma_A=sigma_n=0
    o = _out(m, x_sz)
    for k in ('kappa', 'logits', 'e_mean', 'e_log_std', 'e_logit'):
        d = (base[k] - o[k]).abs().max().item()
        all_ok &= report(f"step-0 '{k}' unchanged when sigma is zeroed",
                         d == 0.0, f"max|delta|={d:.3e}")
    # ... and unchanged for an ARBITRARY sigma, not only zero
    x_sr = x.clone()
    x_sr[:, -n_phys:][:, -2] = 777.0
    x_sr[:, -n_phys:][:, -1] = 3.5
    o = _out(m, x_sr)
    for k in ('kappa', 'logits', 'e_mean', 'e_log_std', 'e_logit'):
        d = (base[k] - o[k]).abs().max().item()
        all_ok &= report(f"step-0 '{k}' unchanged for an arbitrary sigma",
                         d == 0.0, f"max|delta|={d:.3e}")
    # the embedding itself must be LIVE (non-constant), else the test is vacuous
    se = sigma_sinusoidal_embed(torch.tensor([0.0, 10.0, 1000.0]),
                               torch.tensor([0.0, 0.1, 5.0]))
    all_ok &= report("the sigma embedding is non-degenerate (16 dims, varies)",
                     se.shape == (3, 16) and float(se.std(0).max()) > 1e-3,
                     f"shape={tuple(se.shape)} max col std={float(se.std(0).max()):.3f}")

    # ---------------------------------------------------------------- (2)
    print("\n=== 2. higher-moment feature columns are zero-init'd ===")
    x_mz = x.clone()
    x_mz[:, -n_phys:][:, 5:9] = torch.randn(B, 4)         # arbitrary values
    o = _out(m, x_mz)
    for k in ('kappa', 'logits', 'e_mean', 'e_log_std', 'e_logit'):
        d = (base[k] - o[k]).abs().max().item()
        all_ok &= report(f"step-0 '{k}' unchanged by the 4 new moment features",
                         d == 0.0, f"max|delta|={d:.3e}")
    # the features must be LIVE: corrected and uncorrected must differ under blur
    ph = x[:, -n_phys:]
    nz = torch.from_numpy(sigmas > 0)
    sep = float((ph[nz, 5] - ph[nz, 6]).abs().max())
    all_ok &= report("corrected vs uncorrected moment cosines differ at sigma>0",
                     sep > 1e-6, f"max|delta|={sep:.3e}")
    z = torch.from_numpy(sigmas == 0)
    dev0 = float((ph[z, 5] - ph[z, 6]).abs().max())
    all_ok &= report("...and coincide at sigma=0 (s2=0 => A_hat = M_obs)",
                     dev0 < 1e-6, f"max|delta|={dev0:.3e}")
    # and they must carry information distinct from the plain PCA axis
    ang = np.degrees(np.arccos(np.clip(ph[:, 6].numpy(), -1, 1)))
    all_ok &= report("moment axis carries info distinct from the raw-PCA axis",
                     float(np.median(ang)) > 0.5,
                     f"median angle(u, eig1(M_obs)) = {np.median(ang):.2f} deg")

    # ---------------------------------------------------------------- (3)
    print("\n=== 3. mu_k == u exactly, for ANY gate value (S{fallback}) ===")
    u = base['u']
    d = (base['mu'] - u.unsqueeze(1)).abs().max().item()
    all_ok &= report("mu_k == u for every component at init", d == 0.0,
                     f"max|delta|={d:.3e}")
    with torch.no_grad():       # randomize the gate hard: the guarantee must hold
        m.direction.gate.weight.normal_(0, 5.0)
        m.direction.gate.bias.normal_(0, 5.0)
    o = _out(m, x)
    g = torch.sigmoid(m.direction.gate(m.embedding_net.last_sigma_emb))
    d = (o['mu'] - u.unsqueeze(1)).abs().max().item()
    all_ok &= report("mu_k == u still, after randomizing the gate", d == 0.0,
                     f"max|delta|={d:.3e}, gate range "
                     f"[{float(g.min()):.3f},{float(g.max()):.3f}]")
    all_ok &= report("the randomized gate is genuinely non-trivial (not all ~0)",
                     float(g.max()) > 0.6, f"max g={float(g.max()):.3f}")
    # restore a fresh model for the remaining checks
    torch.manual_seed(0)
    m = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    base = _out(m, x)

    # ---------------------------------------------------------------- (4)
    print("\n=== 4. identical to a v22-equivalent head (kappa/logit/energy) ===")
    # Build genuine v22 heads (src/models/directional_head.py, untouched) sized
    # for a conditioning vector WITHOUT the new columns, and copy across the
    # weights that v23's zero-init leaves in play.
    #
    # NOTE on the standard of proof here.  Checks 1-3 above are BITWISE
    # (max|delta| == 0.0 exactly) because they compare the SAME matmul shape with
    # the new inputs zeroed -- which is the spec's actual requirement ("wherever
    # the new inputs are zeroed").  This check is different in kind: it compares
    # a (.,409) matmul against a (.,389) one.  A dot product with 20 extra
    # exactly-zero terms is mathematically identical but NOT bitwise identical in
    # floating point, because the accumulation blocking/order differs.  So the
    # honest standard is: float32 agreement at the rounding floor, PLUS float64
    # agreement ~12 orders tighter, which proves the residual is rounding and not
    # a live contribution from the new columns.
    keep = torch.ones(C.D_AUG, dtype=torch.bool)
    keep[SIGMA_COLS] = False
    keep[MOMENT_COLS] = False
    d_v22 = int(keep.sum())
    all_ok &= report("v22-equivalent conditioning dim is v22's D_AUG (384+5)",
                     d_v22 == C.D_LATENT + 5, f"{d_v22}")
    for dtype, tol, label in ((torch.float32, 1e-5, 'float32'),
                              (torch.float64, 1e-11, 'float64')):
        v22_dir = VMFMixture(d_v22, C.N_DIR_COMP, C.HEAD_HIDDEN).eval().to(dtype)
        v22_e = GMM1D(d_v22, C.N_E_COMP, C.HEAD_HIDDEN).eval().to(dtype)
        d23 = m.direction.net.to(dtype)
        e23 = m.energy.net.to(dtype)
        with torch.no_grad():
            for src, dst in ((d23, v22_dir.net), (e23, v22_e.net)):
                dst.net[0].weight.copy_(src.net[0].weight[:, keep])
                dst.net[0].bias.copy_(src.net[0].bias)
                for i in (2, 4):
                    dst.net[i].weight.copy_(src.net[i].weight)
                    dst.net[i].bias.copy_(src.net[i].bias)
            z_full = m.embedding_net(x).to(dtype)
            o23 = d23(z_full).view(B, C.N_DIR_COMP, 5)
            kap23 = F.softplus(o23[..., 3]) + 1e-2
            log23 = o23[..., 4]
            oe23 = e23(z_full).view(B, C.N_E_COMP, 3)
            _mu22, kap22, log22 = v22_dir._params(z_full[:, keep])
            em22, es22, el22 = v22_e._params(z_full[:, keep])
        pairs = (('kappa', kap23, kap22), ('logits', log23, log22),
                 ('energy mean', oe23[..., 0], em22),
                 ('energy log_std', oe23[..., 1].clamp(-7.0, 3.0), es22),
                 ('energy logit', oe23[..., 2], el22))
        for nm, a, b in pairs:
            dd = (a - b).abs().max().item()
            all_ok &= report(f"[{label}] v23 '{nm}' == v22-equivalent head",
                             dd < tol, f"max|delta|={dd:.3e} (tol {tol:.0e})")
        m.direction.net.to(torch.float32)
        m.energy.net.to(torch.float32)

    # ---------------------------------------------------------------- (5)
    print("\n=== 5. at init the direction readout IS the classical PCA baseline ===")
    # independent numpy implementation, the exact convention the reference
    # sweeps use (metrics.pca_axis_and_signed), applied to the same blurred
    # physical cloud the pipeline saw.
    worst = 0.0
    for j in range(B):
        rng = np.random.default_rng(1)   # unused at sigma>0? no: re-smear needed
        pts = raw[j]
        # reproduce the SAME blurred cloud: re-derive it from the stored coords
        # instead of re-smearing (a fresh draw would differ).  The pipeline's
        # normalized coords are an affine map of the blurred cloud, and the PCA
        # axis is invariant to uniform scaling and translation, so the stored
        # normalized coordinates give the identical axis.
        n_real = int((x[j, :N_MAX * 3].view(N_MAX, 3).abs().sum(-1) > 0).sum())
        cn = x[j, :N_MAX * 3].view(N_MAX, 3)[:n_real].numpy().astype(np.float64)
        ax_np = pca_axis_and_signed(cn, 0.0, rng)
        cos = abs(float(np.dot(ax_np, base['u'][j].numpy())))
        signed = float(np.dot(ax_np, base['u'][j].numpy()))
        worst = max(worst, abs(1.0 - signed))
    all_ok &= report("in-graph u == independent numpy PCA axis (incl. SIGN)",
                     worst < 1e-4, f"max|1 - u.u_numpy| = {worst:.2e}")
    # and the model's own mode readout is that axis
    md = m.direction.mode_axis(base['mu'], base['logits'])
    dd = (md - base['u']).abs().max().item()
    all_ok &= report("mode readout == u at init", dd == 0.0,
                     f"max|delta|={dd:.3e}")
    # equivariance of the sign-fixed axis under Oh (needed for the augmentation
    # to be consistent with the in-graph u)
    R = torch.tensor([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])   # an Oh element
    coords = x[:, :N_MAX * 3].view(B, N_MAX, 3)
    mask = coords.abs().sum(-1) > 0
    u_rot = pca_axis_torch(torch.einsum('bni,ij->bnj', coords, R.T), mask)
    dd = (u_rot - torch.einsum('bi,ij->bj', base['u'], R.T)).abs().max().item()
    all_ok &= report("u is Oh-equivariant incl. the sign fix (R u == u(R x))",
                     dd < 1e-5, f"max|delta|={dd:.3e}")

    # ---------------------------------------------------------------- (6)
    print("\n=== 6. the zero-init'd slices are NOT dead ===")
    m.train()
    th = torch.randn(B, 4)
    th[:, 1:4] = F.normalize(torch.randn(B, 3), dim=-1)
    nll, kpen = m.loss_terms(th, x)
    loss = nll.mean()
    loss.backward()
    for nm, lin, sl in (('direction sigma cols', m.direction.net.net[0], SIGMA_COLS),
                        ('direction moment cols', m.direction.net.net[0], MOMENT_COLS),
                        ('energy sigma cols', m.energy.net.net[0], SIGMA_COLS),
                        ('energy moment cols', m.energy.net.net[0], MOMENT_COLS)):
        gmax = lin.weight.grad[:, sl].abs().max().item()
        all_ok &= report(f"{nm} receive gradient (not dead)", gmax > 0,
                         f"max|grad|={gmax:.3e}")
    rows = [k * 5 + j for k in range(C.N_DIR_COMP) for j in range(3)]
    gmax = m.direction.net.net[-1].weight.grad[rows, :].abs().max().item()
    all_ok &= report("the zeroed mu ROWS receive gradient (not dead)", gmax > 0,
                     f"max|grad|={gmax:.3e}")
    # The GATE is a documented exception, and it is exact, not a bug: with
    # delta_k == 0 identically, mu = normalize(u + g*0) = u, so d mu / d g = 0 and
    # the gate correctly receives NO gradient at step 0.  It is unlocked by the
    # first optimizer step, because the mu rows above DO receive gradient, making
    # delta_k non-zero from step 1 onward.  Verify BOTH halves of that statement.
    gmax = m.direction.gate.weight.grad.abs().max().item()
    all_ok &= report("the sigma gate receives NO gradient at init "
                     "(exact: d mu/d g = 0 when delta == 0)", gmax == 0.0,
                     f"max|grad|={gmax:.3e}")
    with torch.no_grad():   # simulate one optimizer step on the mu rows.
        # NB assignment, not `weight[rows, :].normal_()`: advanced indexing
        # returns a COPY, so an in-place op on it would silently write to a
        # temporary and leave the weight untouched.
        w = m.direction.net.net[-1].weight
        w[rows, :] = 0.05 * torch.randn(len(rows), w.shape[1])
    m.zero_grad(set_to_none=True)
    m.loss_terms(th, x)[0].mean().backward()
    gmax = m.direction.gate.weight.grad.abs().max().item()
    all_ok &= report("...and IS unlocked once delta != 0 (i.e. from step 1)",
                     gmax > 0, f"max|grad|={gmax:.3e}")
    all_ok &= report("loss and all grads finite",
                     bool(torch.isfinite(loss))
                     and all(bool(torch.isfinite(p.grad).all())
                             for p in m.parameters() if p.grad is not None))
    # and once trained, sigma MUST be able to matter -- perturb the zeroed slice
    with torch.no_grad():
        m.direction.net.net[0].weight[:, SIGMA_COLS].normal_(0, 0.1)
    o = _out(m.eval(), x)
    dk = (o['kappa'] - base['kappa']).abs().max().item()
    all_ok &= report("sigma reaches kappa once its weights are non-zero",
                     dk > 0, f"max|dkappa|={dk:.3e}")

    # ---------------------------------------------------------------- (7)
    print("\n=== 7. the kappa regularizer at lambda=0 is a no-op ===")
    # .eval() is REQUIRED here, not cosmetic: v22's FusionMLP (carried forward
    # unchanged) contains nn.Dropout(0.1), so a train()-mode forward is
    # stochastic and two forwards of the same weights differ by ~1e-2.  Comparing
    # in train() mode would measure dropout noise, not the lambda=0 term.
    torch.manual_seed(0)
    m1 = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    torch.manual_seed(0)
    m2 = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    dpar = max((a - b).abs().max().item()
               for a, b in zip(m1.parameters(), m2.parameters()))
    all_ok &= report("the two comparison models are identically initialized",
                     dpar == 0.0, f"max|delta param|={dpar:.3e}")
    n1, k1 = m1.loss_terms(th, x)
    n2, k2 = m2.loss_terms(th, x)
    l1 = n1.mean()                             # lambda = 0  (term omitted)
    l2 = n2.mean() + 0.0 * k2.mean()           # lambda = 0  (term present)
    l1.backward()
    l2.backward()
    dl = abs(float(l1) - float(l2))

    def _gdiff(a, b):
        return max((p1.grad - p2.grad).abs().max().item()
                   for p1, p2 in zip(a.parameters(), b.parameters())
                   if p1.grad is not None and p2.grad is not None)

    gmax = _gdiff(m1, m2)
    # CONTROL: the same loss expression, twice, on two identical models.  Any
    # residual here is the backward pass's own run-to-run noise floor (the
    # backbone's scatter/reduction order), NOT an effect of the lambda=0 term --
    # so the term is a no-op iff its delta does not exceed this control.
    torch.manual_seed(0)
    c1 = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    torch.manual_seed(0)
    c2 = build_v23(n_max=N_MAX, k=K_MAX, device=DEVICE).eval()
    c1.loss_terms(th, x)[0].mean().backward()
    c2.loss_terms(th, x)[0].mean().backward()
    ctrl = _gdiff(c1, c2)
    all_ok &= report("loss identical with the lambda=0 term present", dl == 0.0,
                     f"|delta|={dl:.3e}")
    all_ok &= report("gradients unchanged by the lambda=0 term, at or below the "
                     "backward's own noise floor", gmax <= max(ctrl, 1e-12),
                     f"max|delta grad|={gmax:.3e}  control(identical loss, "
                     f"twice)={ctrl:.3e}")
    all_ok &= report("the penalty itself is live and positive (so lambda>0 bites)",
                     float(k1.mean()) > 0, f"mean penalty={float(k1.mean()):.4f}")
    return all_ok


if __name__ == '__main__':
    print(f"v23 zero-init verification  (D_AUG={C.D_AUG}, "
          f"sigma cols {SIGMA_COLS.start}:{SIGMA_COLS.stop}, "
          f"moment cols {MOMENT_COLS.start}:{MOMENT_COLS.stop})")
    ok = check()
    print("\n" + ("ALL ZERO-INIT CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)
