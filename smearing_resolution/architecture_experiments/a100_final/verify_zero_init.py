"""Verify the zero-init discipline EMPIRICALLY (not by assumption).

Claim under test: at step 0, before any optimizer step, DiagModelV2's outputs
are EXACTLY what a model without (a) the sigma-conditioning features and
(b) the two new 4th-moment candidate vectors would produce.  If both new
inputs are zero-init'd correctly, forcing them off must be a no-op.

Also checks that the zero-init'd slices are not DEAD -- they must receive
non-zero gradient on the very first backward pass, or they would never learn.

Run:  python verify_zero_init.py
"""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    _smear_and_prepare_one_v2, flat_dim, pack_one)
from smearing_resolution.architecture_experiments.a100_final.model import (
    DiagModelV2, NEW_VEC_IDX, KURT_CORR_IDX, KURT_RAW_IDX, N_VECTORS,
    N_SIGMA_FEATS)

N_MAX, K_MAX, B = 120, 24, 16
torch.manual_seed(0)


def synth_batch(seed=0):
    rng = np.random.default_rng(seed)
    x = torch.zeros(B, flat_dim(N_MAX, K_MAX), dtype=torch.float32)
    sig = np.concatenate([np.zeros(B // 2),
                          np.exp(rng.uniform(np.log(1.), np.log(1000.), B - B // 2))])
    for j in range(B):
        n = int(rng.integers(20, N_MAX))
        p = rng.normal(0, 1, (n, 3)) * np.array([120., 30., 20.])
        co, kn, ph = _smear_and_prepare_one_v2(p, sig[j], N_MAX, rng)
        pack_one(x[j], co, kn, ph, N_MAX, K_MAX)
    return x, sig


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def check(posterior_type):
    print(f"\n=== POSTERIOR_TYPE={posterior_type} ===")
    m = DiagModelV2(n_max=N_MAX, k_max=K_MAX, hidden_dim=112, n_layers=6,
                    posterior_type=posterior_type, n_dir_comp=10).eval()
    x, sig = synth_batch()
    all_ok = True

    with torch.no_grad():
        full = m(x)
        ablt = m(x, ablate_new_vector=True, ablate_sigma=True)

    keys = (['mu', 'kappa', 'logits'] if posterior_type == 'vmf_mixture'
            else ['grid_scores'])
    keys += ['sign_logit', 'E_pred', 'log_sigma', 'axis_ref_aligned']
    for k in keys:
        d = (full[k] - ablt[k]).abs().max().item()
        all_ok &= report(f"step-0 output '{k}' unchanged by new inputs",
                         d == 0.0, f"max|delta|={d:.3e}")

    # the new candidate VECTORS themselves must be live, finite and unit-norm
    V = full['V']
    all_ok &= report("pool has N_VECTORS entries", V.shape[1] == N_VECTORS,
                     f"{V.shape[1]}")
    for idx, nm in [(KURT_CORR_IDX, 'kurt_corrected'), (KURT_RAW_IDX, 'kurt_raw')]:
        v = V[:, idx]
        nrm = v.norm(dim=-1)
        all_ok &= report(f"'{nm}' is finite & unit-norm",
                         bool(torch.isfinite(v).all()) and
                         float((nrm - 1).abs().max()) < 1e-4,
                         f"|n-1|max={float((nrm-1).abs().max()):.2e}")
    # and they must actually differ from the plain PCA axis (else no new info)
    for idx, nm in [(KURT_CORR_IDX, 'kurt_corrected'), (KURT_RAW_IDX, 'kurt_raw')]:
        cos = (V[:, idx] * V[:, 10]).sum(-1).abs().clamp(max=1.0)
        a = torch.rad2deg(torch.arccos(cos))
        all_ok &= report(f"'{nm}' carries info distinct from the raw-PCA axis",
                         float(a.median()) > 0.5,
                         f"median angle to v_ref = {float(a.median()):.2f} deg")

    # the corrected and uncorrected 4th-moment axes must actually DIVERGE once
    # there is blur to correct (they are identical by construction at sigma=0)
    nz = torch.from_numpy(sig > 0)
    cos = (V[nz, KURT_CORR_IDX] * V[nz, KURT_RAW_IDX]).sum(-1).abs().clamp(max=1.0)
    sep = float(torch.rad2deg(torch.arccos(cos)).max())
    all_ok &= report("corrected vs uncorrected 4th-moment axes differ at sigma>0",
                     sep > 0.1, f"max separation = {sep:.2f} deg")
    # At sigma=0, s2=0 and A == M_obs bitwise, so the two axes must coincide.
    # Compare on COSINE, not angle: arccos near 1 amplifies float32 rounding
    # (1-cos ~ 1e-7 already shows up as ~0.03 deg), so an angle tolerance below
    # ~0.03 deg is measuring the metric's noise floor rather than the vectors.
    z = torch.from_numpy(sig == 0)
    cos0 = (V[z, KURT_CORR_IDX] * V[z, KURT_RAW_IDX]).sum(-1).abs().clamp(max=1.0)
    dev0 = float((1.0 - cos0).max())
    all_ok &= report("...and coincide at sigma=0 (s2=0 => A=M_obs)",
                     dev0 < 1e-6, f"max(1-cos) = {dev0:.2e}")

    # gradients must reach the zero-init'd slices on the FIRST backward
    m.train()
    o = m(x)
    d = F.normalize(torch.randn(B, 3), dim=-1)
    loss = m.direction_nll_axis(o, d).mean() + o['E_pred'].pow(2).mean() \
        + o['sign_logit'].pow(2).mean()
    loss.backward()

    g = m.trunk.net[0].weight.grad
    ng = m.trunk.n_gram
    sig_g = g[:, ng + m.trunk.d_scalar:].abs().max().item()
    all_ok &= report("sigma slice receives gradient (not dead)", sig_g > 0,
                     f"max|grad|={sig_g:.3e}")
    mask = torch.zeros(ng, dtype=torch.bool)
    for i in NEW_VEC_IDX:
        mask |= (m.trunk.triu_i == i) | (m.trunk.triu_j == i)
    gram_g = g[:, :ng][:, mask].abs().max().item()
    all_ok &= report("new-vector Gram columns receive gradient", gram_g > 0,
                     f"max|grad|={gram_g:.3e}")
    if posterior_type == 'vmf_mixture':
        cg = m.posterior.coef_out.weight.grad.view(m.posterior.n_comp,
                                                   N_VECTORS, -1)
        v = cg[:, list(NEW_VEC_IDX), :].abs().max().item()
        all_ok &= report("new-vector coefficient rows receive gradient", v > 0,
                         f"max|grad|={v:.3e}")
    all_ok &= report("loss and all grads finite",
                     bool(torch.isfinite(loss)) and
                     all(bool(torch.isfinite(p.grad).all())
                         for p in m.parameters() if p.grad is not None))

    # sigma must MATTER once trained -- perturb the zero slice and confirm the
    # coefficient pathway responds (proves it is wired to coef_out, not only kl_out)
    if posterior_type == 'vmf_mixture':
        with torch.no_grad():
            m.trunk.net[0].weight[:, ng + m.trunk.d_scalar:].normal_(0, 0.1)
            pert = m(x)
        dmu = (pert['mu'] - full['mu']).abs().max().item()
        all_ok &= report("sigma reaches the COEFFICIENT pathway (mu responds)",
                         dmu > 0, f"max|dmu|={dmu:.3e}")
    return all_ok


if __name__ == '__main__':
    ok = check('vmf_mixture')
    ok &= check('grid')
    print("\n" + ("ALL ZERO-INIT CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)
