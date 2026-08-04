
import numpy as np


def gaussian_mean_shift_merge(pts, sigma, max_iter=500, tol_frac=1e-5, merge_frac=0.25,
                               return_brightness=False):
    """Merge points into density-mode blobs via Gaussian mean-shift.

    Args:
        pts: (N,3) array of true positions.
        sigma: bandwidth (same units as pts, e.g. Angstrom) -- how blurry the
            imaging is. Bigger sigma = points have to be farther apart to
            stay resolved.
        max_iter: number of iterations to move trackers
        tol_frac: once a point's marker moves less than this fraction of
            sigma in one step, treat it as converged
        merge_frac: after everything converges, merge markers within this fraction
            of sigma of each other
        return_brightness: if True, also return how many true points merged
            into each blob.

    Returns:
        (M,3) array of blob centroids, M <= N.
        If return_brightness: also an (M,) array of per-blob point counts.
    """
    n = len(pts)
    if sigma <= 0 or n < 2:
        # nothing to blur, or too few points to possibly merge -- pass through unchanged
        if return_brightness:
            return pts.copy(), np.ones(n, dtype=int)
        return pts.copy()

    y = pts.copy().astype(np.float64)  # each point's own "marker," starts at its true position
    tol = tol_frac * sigma
    two_sig2 = 2.0 * sigma * sigma     # precompute the denominator used in the Gaussian bump formula

    for _ in range(max_iter):
        # for every marker's CURRENT position, get its distance to every ORIGINAL point
        # (always measured against the original, fixed points -- that's what makes this
        # "climb the hill built from the real data," not something that drifts around)
        d2 = ((y[:, None, :] - pts[None, :, :]) ** 2).sum(-1)

        # closer original points get more weight (the "bump" fading with distance)
        w = np.exp(-d2 / two_sig2)
        w_sum = np.clip(w.sum(1, keepdims=True), 1e-300, None)  # avoid divide-by-zero

        # move each marker to the weighted-average position of all the true points --
        # this is the "walk uphill" step
        y_new = (w @ pts) / w_sum

        shift = np.linalg.norm(y_new - y, axis=1).max()  # how far did the slowest-converging marker move?
        y = y_new
        if shift < tol:
            break  # everyone has basically stopped moving -- hilltops found

    # group markers that ended up on (almost) the same hilltop. This has to be a proper
    # connected-components grouping, not "whoever i loops to first claims nearby markers" --
    # a greedy first-come-first-served pass is order-dependent: if A-B and B-C are both
    # within merge_dist but A-C isn't, the greedy version can arbitrarily split them into
    # {A,B},{C} or {A},{B,C} depending on point order, so reordering the same input could
    # change the blob count. Union-find groups anything chained together, regardless of order.
    merge_dist = merge_frac * sigma
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)  # pairwise distances between converged markers
    ii, jj = np.where(np.triu(d < merge_dist, k=1))
    for i, j in zip(ii, jj):
        union(int(i), int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # report, per group:
    #   - position: the average of the TRUE original points in that group (not the
    #     converged marker position itself -- we report where the real emitters were)
    #   - brightness: how many true points are in that group
    merged_pts = []
    merged_brightness = []
    for idx in groups.values():
        idx = np.array(idx)
        merged_pts.append(pts[idx].mean(axis=0))
        merged_brightness.append(len(idx))            # how many true points fell into this blob

    if return_brightness:
        return np.array(merged_pts), np.array(merged_brightness)
    return np.array(merged_pts)


if __name__ == '__main__':
    # quick smoke test
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3)) * 5.0
    for sigma in [0, 1, 5, 20]:
        merged, brightness = gaussian_mean_shift_merge(pts, sigma, return_brightness=True)
        print(f'sigma={sigma:>4}: {len(pts)} points -> {len(merged)} blobs, '
              f'brightness per blob = {brightness.tolist()}')
