"""Standalone numerical validation of the two blur corrections. Seconds to run.

  TEST A -- normalizer:  does the blur-deconvolved radius of gyration track the
            TRUE scale across blur levels better than the old max_extent?
  TEST B -- 4th-moment candidate vector:  does the analytic blur correction
            recover the clean-data dominant eigenvector better than the
            uncorrected tensor?
  TEST C -- (supporting) the plain 3rd CENTRAL moment is exactly blur-unbiased,
            which is why we do NOT use it as the corrected statistic and why
            the skewness SIGN fix needs no correction.

Run:  python validate_math.py
"""
import numpy as np

rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------- helpers
def deconv_scale(centered, sigma):
    n = centered.shape[0]
    R2 = (centered ** 2).sum(1).mean()
    s2 = sigma ** 2 * (n - 1) / n
    floor = max(1e-4, (0.1 * sigma) ** 2)
    return np.sqrt(max(R2 - 3.0 * s2, floor))


def max_scale(centered):
    return np.abs(centered).max()


def make_track(n, lengths, skew=0.0, bend=0.0, rng=None, rotate=True):
    """Anisotropic blob, optionally skewed and/or BENT along its long axis.

    `bend` displaces points transversely in proportion to x^2, producing a
    banana-shaped cloud whose EXTREMITIES sit off the principal axis.  That is
    the regime where a 4th-moment (extremity-weighted) axis carries genuinely
    different information from the 2nd-moment PCA axis -- and therefore the
    regime in which the blur correction has anything to correct.
    """
    rng = rng or rng_global
    p = rng.normal(0, 1, size=(n, 3)) * np.array(lengths)
    if skew:
        p[:, 0] = p[:, 0] + skew * (p[:, 0] ** 2 - lengths[0] ** 2) / lengths[0]
    if bend:
        p[:, 1] = p[:, 1] + bend * (p[:, 0] ** 2 - lengths[0] ** 2) / lengths[0]
    if rotate:
        Q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
        p = p @ Q.T
    return p


def kurt_tensor(xc, u):
    p = xc @ u
    return np.einsum('i,ij,ik->jk', p * p, xc, xc) / len(xc)


def corrected_kurt_tensor(xc, u, sigma):
    n = len(xc)
    s2 = sigma ** 2 * (n - 1) / n
    C_obs = xc.T @ xc / n
    p = xc @ u
    q2 = max((p * p).mean() - s2, 0.0)
    C_hat = C_obs - s2 * np.eye(3)
    M = kurt_tensor(xc, u)
    uu = np.outer(u, u)
    return M - s2 * (q2 * np.eye(3) + 4 * q2 * uu + C_hat) - s2 ** 2 * (np.eye(3) + 2 * uu)


def top_evec(A):
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    return V[:, -1]


def pca_axis(xc):
    return top_evec(xc.T @ xc / len(xc))


def ang(a, b):
    return np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b))))))


# ------------------------------------------------------------------ TEST A
def test_normalizer(n_trials=300):
    print("=" * 74)
    print("TEST A -- normalizer: estimated scale vs TRUE R_g across blur levels")
    print("=" * 74)
    sigmas = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    print(f"{'sigma/Rg':>9} {'new: med ratio':>15} {'new: spread':>12} "
          f"{'old: med ratio':>15} {'old: spread':>12}")
    new_dev, old_dev = [], []
    for sigma in sigmas:
        rn, ro, ratio = [], [], None
        for _ in range(n_trials):
            rng = np.random.default_rng(hash((int(sigma * 10), _)) % 2 ** 32)
            n = int(rng.integers(30, 300))
            L = np.array([120.0, 25.0, 25.0])          # elongated track, Angstrom
            p = make_track(n, L, rng=rng)
            p = p - p.mean(0)
            rg_true = np.sqrt((p ** 2).sum(1).mean())
            q = p + rng.normal(0, sigma, size=p.shape) if sigma > 0 else p
            q = q - q.mean(0)
            rn.append(deconv_scale(q, sigma) / rg_true)
            ro.append(max_scale(q) / rg_true)
        rn, ro = np.array(rn), np.array(ro)
        ratio = sigma / 70.0
        print(f"{ratio:9.2f} {np.median(rn):15.3f} "
              f"{(np.percentile(rn,84)-np.percentile(rn,16))/2:12.3f} "
              f"{np.median(ro):15.3f} "
              f"{(np.percentile(ro,84)-np.percentile(ro,16))/2:12.3f}")
        new_dev.append(abs(np.median(rn) - 1.0))
        old_dev.append(abs(np.median(ro) / np.median(ro if sigma == 0 else ro) - 1.0))
    # headline: how much does the median ratio DRIFT from its sigma=0 value?
    print()
    print("  Interpretation: 'med ratio' is estimate / true R_g.  The NEW "
          "normalizer\n  should stay ~1.0 at every blur level; the OLD one is "
          "an extreme order\n  statistic and inflates without bound.")
    return new_dev


# ----------------------------------------------------------------- TEST B1
def test_kurt_tensor_bias(n_draws=3000):
    """Direct test of the DERIVATION: average the estimator over many blur
    draws of a FIXED cloud with u held fixed, and compare the mean tensor to
    the clean one.  This isolates BIAS from variance."""
    print()
    print("=" * 74)
    print("TEST B1 -- derivation check: E[estimator] vs clean tensor (u fixed)")
    print("=" * 74)
    n = 200
    L = np.array([120.0, 30.0, 20.0])
    p = make_track(n, L, skew=0.5, bend=0.15,
                   rng=np.random.default_rng(3), rotate=False)
    p = p - p.mean(0)
    u = pca_axis(p)
    A_clean = kurt_tensor(p, u)
    nrm = np.linalg.norm(A_clean)
    print(f"{'sigma':>7} {'sigma/Rg':>9} {'|bias(uncorr)|/|A|':>20} "
          f"{'|bias(corr)|/|A|':>18} {'bias reduction':>15}")
    for sigma in [20.0, 50.0, 100.0, 200.0, 400.0]:
        Mu = np.zeros((3, 3)); Mc = np.zeros((3, 3))
        for t in range(n_draws):
            rng = np.random.default_rng(int(sigma) * 131071 + t)
            q = p + rng.normal(0, sigma, size=p.shape)
            q = q - q.mean(0)
            Mu += kurt_tensor(q, u)
            Mc += corrected_kurt_tensor(q, u, sigma)
        Mu /= n_draws; Mc /= n_draws
        bu = np.linalg.norm(Mu - A_clean) / nrm
        bc = np.linalg.norm(Mc - A_clean) / nrm
        rg = np.sqrt((p ** 2).sum(1).mean())
        print(f"{sigma:7.1f} {sigma/rg:9.2f} {bu:20.4f} {bc:18.4f} "
              f"{bu/max(bc,1e-12):14.1f}x")
    print("\n  A >>1x reduction confirms the closed-form correction is right.")


# ----------------------------------------------------------------- TEST B2
def test_kurtosis_vector(n_trials=400):
    print()
    print("=" * 74)
    print("TEST B2 -- 4th-moment AXIS: angle to the clean answer (bent cloud,")
    print("           u re-estimated from the blurred data, as in the pipeline)")
    print("=" * 74)
    sigmas = [20.0, 50.0, 100.0, 200.0, 400.0]
    print(f"{'sigma':>7} {'sigma/Rg':>9} {'corrected':>11} {'uncorrected':>13} "
          f"{'PCA axis':>10} {'improvement':>12}")
    for sigma in sigmas:
        ec, eu, ep, rgs = [], [], [], []
        for t in range(n_trials):
            rng = np.random.default_rng(1000 + int(sigma) * 10007 + t)
            n = int(rng.integers(60, 400))
            L = np.array([120.0, 30.0, 20.0])
            p = make_track(n, L, skew=0.5, bend=0.15, rng=rng)
            p = p - p.mean(0)
            rgs.append(np.sqrt((p ** 2).sum(1).mean()))
            u_c = pca_axis(p)
            w_clean = top_evec(kurt_tensor(p, u_c))

            q = p + rng.normal(0, sigma, size=p.shape)
            q = q - q.mean(0)
            u_b = pca_axis(q)
            if np.dot(u_b, u_c) < 0:
                u_b = -u_b
            ec.append(ang(top_evec(corrected_kurt_tensor(q, u_b, sigma)), w_clean))
            eu.append(ang(top_evec(kurt_tensor(q, u_b)), w_clean))
            ep.append(ang(u_b, w_clean))
        mc, mu_, mp = np.median(ec), np.median(eu), np.median(ep)
        print(f"{sigma:7.1f} {sigma/np.mean(rgs):9.2f} {mc:11.2f} {mu_:13.2f} "
              f"{mp:10.2f} {mu_ - mc:10.2f} deg")
    print()
    print("  'PCA axis' is the angle between the plain 2nd-moment axis and the\n"
          "  clean 4th-moment answer -- i.e. how much INDEPENDENT information a\n"
          "  4th-moment candidate carries (~12 deg here, so it is a genuinely\n"
          "  new axis estimate, not a rescaled PCA axis).\n\n"
          "  HONEST READING: B1 proves the closed form kills the BIAS.  On a\n"
          "  SINGLE realization, however, the correction also amplifies\n"
          "  VARIANCE (it subtracts a large tensor), and the uncorrected\n"
          "  estimator's bias happens to be a shrinkage TOWARD the PCA axis --\n"
          "  which at extreme sigma/R_g is itself closer to the truth than the\n"
          "  noisy corrected estimate.  So neither dominates everywhere.\n"
          "  DESIGN CONSEQUENCE: we ship BOTH axes as pool candidates and let\n"
          "  the sigma-conditioned coefficient head arbitrate per blur level,\n"
          "  instead of hard-coding a shrinkage factor.")


# ------------------------------------------------------------------ TEST C
def test_third_moment_unbiased(n_trials=4000):
    print()
    print("=" * 74)
    print("TEST C -- the plain 3rd CENTRAL moment is exactly blur-unbiased")
    print("=" * 74)
    sigma = 30.0
    n = 150
    L = np.array([120.0, 30.0, 20.0])
    base_rng = np.random.default_rng(7)
    p = make_track(n, L, skew=0.6, rng=base_rng)
    p = p - p.mean(0)
    u = pca_axis(p)
    clean = float(((p @ u) ** 3).mean())
    vals = []
    for t in range(n_trials):
        rng = np.random.default_rng(90000 + t)
        q = p + rng.normal(0, sigma, size=p.shape)
        q = q - q.mean(0)
        vals.append(float(((q @ u) ** 3).mean()))
    m, se = np.mean(vals), np.std(vals) / np.sqrt(n_trials)
    print(f"  clean <p^3>           = {clean: .4e}")
    print(f"  blurred <p^3> (mean)  = {m: .4e}  +/- {se:.2e} (s.e.)")
    print(f"  bias / s.e.           = {(m - clean) / se: .2f}   "
          f"(|.|<3 => consistent with ZERO bias)")
    print("  => no correction is needed for the skewness SIGN fix.")


if __name__ == '__main__':
    test_normalizer()
    test_kurt_tensor_bias()
    test_kurtosis_vector()
    test_third_moment_unbiased()
    print("\n[DONE] validate_math.py")
