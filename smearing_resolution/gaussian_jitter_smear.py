import numpy as np


def gaussian_jitter_smear(pts, sigma, rng=None):
    """Simple resolution-blur model: add independent Gaussian noise to every point.

    Each point's reported position is its true position plus random Gaussian
    noise of width sigma, drawn independently per point. No merging -- the
    output always has the same number of points as the input, just with
    noisier positions. This does not model two points becoming indistinguishable
    from each other; it only models the reported position of each vacancy
    being less precise. Simple, rough estimate of resolution effects, not a
    resolvability/blob-count model.

    Args:
        pts: (N,3) array of true positions.
        sigma: noise standard deviation per axis (same units as pts, e.g.
            Angstrom).
        rng: optional numpy Generator for reproducibility. Defaults to a
            fresh, unseeded Generator.

    Returns:
        (N,3) array of jittered positions, same shape as pts.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        return pts.copy()
    noise = rng.normal(scale=sigma, size=pts.shape)
    return pts + noise


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3)) * 5.0
    for sigma in [0, 1, 5, 20]:
        jittered = gaussian_jitter_smear(pts, sigma, rng=rng)
        rms_shift = np.linalg.norm(jittered - pts, axis=1).mean()
        print(f'sigma={sigma:>4}: {len(pts)} points -> {len(jittered)} points (unchanged count), '
              f'mean position shift = {rms_shift:.3f}')
