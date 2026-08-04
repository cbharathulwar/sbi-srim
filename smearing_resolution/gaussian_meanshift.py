"""Correct resolution-blur model: Gaussian mean-shift mode-finding.

PLAIN-ENGLISH VERSION OF THE IDEA:
  1. Put a soft, fading "bump" of brightness at every true point (a Gaussian
     blob of width sigma). Add all the bumps together -- that sum is the
     blurred picture a real resolution-limited detector would produce.
  2. What you actually SEE as a distinct spot is a local peak ("hilltop") in
     that blurred picture. Two points close together have their bumps
     overlap so much they merge into ONE hilltop (unresolvable). Two points
     far apart keep separate hilltops (resolvable).
  3. Mean-shift finds these hilltops without scanning the whole picture:
     drop a marker on each point and let it "walk uphill" -- each step, move
     it to the weighted-average position of all the true points, weighting
     closer points more heavily. This always moves toward higher brightness
     and is guaranteed to converge onto a hilltop (a real, proven math
     result, not a heuristic). Points whose markers walk to the same hilltop
     get merged into one blob.

Two simpler alternatives were tried first and rejected:
  - independent jitter per point: adds noise but never actually MERGES two
    points into one, so it can't model unresolvability at all.
  - single-linkage clustering: chains points together through any path of
    small hops, even across an entire elongated track -- doesn't match how
    real blur works (only things that are ACTUALLY close together merge).

BRIGHTNESS: a resolved blob made of many merged points is genuinely brighter
in a real fluorescence image (more emitters = more photons) than a blob made
of just one point. That information used to be thrown away here -- this
version keeps it. `brightness` for each returned blob = how many true points
merged into it, the simplest available proxy (assumes every point/NV
contributes equally; a real system could have unequal per-emitter
brightness, e.g. from orientation or local strain, but this is a reasonable
default absent better information).
"""
import numpy as np


def gaussian_mean_shift_merge(pts, sigma, max_iter=500, tol_frac=1e-5, merge_frac=0.25,
                               return_brightness=False):
    """Merge points into density-mode blobs via Gaussian mean-shift.

    Args:
        pts: (N,3) array of true positions.
        sigma: bandwidth (same units as pts, e.g. Angstrom) -- how blurry the
            imaging is. Bigger sigma = points have to be farther apart to
            stay resolved.
        max_iter: give up "walking uphill" after this many steps per point.
            Near a marginal (barely-resolvable) separation the walk converges
            slowly, so this is set generously high -- too low a cap can stop
            the walk before same-peak markers have actually come together,
            making the code report two blobs where there's really only one.
        tol_frac: once a point's marker moves less than this fraction of
            sigma in one step, treat it as converged (found its hilltop). Set
            small (not just "small enough") so the leftover gap between
            same-peak markers ends up much smaller than merge_frac*sigma --
            otherwise the grouping step below can't tell "same peak, tiny
            numerical residue" apart from "two really distinct peaks that
            happen to be close."
        merge_frac: after everything converges, markers within this fraction
            of sigma of each other are considered "the same hilltop" and get
            merged (they won't land on EXACTLY the same spot due to finite
            iterations, so this is a small tolerance, not zero). Only
            meaningful once tol_frac is tight enough that same-peak markers
            converge to much less than merge_frac*sigma apart -- see above.
        return_brightness: if True, also return how many true points merged
            into each blob (see BRIGHTNESS note above).

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
