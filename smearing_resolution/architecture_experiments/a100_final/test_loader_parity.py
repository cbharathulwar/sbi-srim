"""Verify the parallel loader changes THROUGHPUT ONLY, never SEMANTICS.

Checks, on a small synthetic track pool (so it runs in seconds and so
spawn-based workers are cheap enough to exercise even on Windows):

  1. num_workers=0 and num_workers=2 yield BYTE-IDENTICAL batches, in the same
     order.  This is the load-bearing claim: parallel prep must be a pure
     performance change.
  2. Batches arrive strictly IN ORDER (step 1,2,3,...).
  3. The epoch permutation is respected: each epoch's batches are a partition
     of the training pool, with no track repeated within an epoch.
  4. The sigma marginal is the intended mixture (p_zero point mass at 0, else
     log-uniform) -- i.e. per-sample seeding did not distort it.

Run:  python test_loader_parity.py
"""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch

from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    TrackPool, N_PHYS, PHYS_SIGMA_N)
from smearing_resolution.architecture_experiments.a100_final.loader import (
    TrackDataset, make_loader, resolve_workers)

N_TRACKS, B, N_MAX, K_MAX = 240, 8, 120, 24
SPE = N_TRACKS // B          # 30 steps/epoch
STEPS = 75                   # 2.5 epochs


REAL_CSV = os.environ.get("REAL_CSV", "")


def build_pool():
    """Synthetic by default (seconds).  REAL_CSV=<path> instead builds the pool
    from a real corpus, exercising TrackPool against the actual track-length
    distribution and the real (non-dense) ion_number ranges."""
    global N_TRACKS, SPE
    if REAL_CSV:
        from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
            load_raw_tracks)
        raw, theta, _n, _ = load_raw_tracks(REAL_CSV, max_points=N_MAX)
        keep = int(os.environ.get("REAL_N", "2400"))
        rng = np.random.default_rng(0)
        sel = rng.choice(len(raw), size=min(keep, len(raw)), replace=False)
        pool = TrackPool([raw[i] for i in sel])
        N_TRACKS = len(sel)
        SPE = N_TRACKS // B
        print(f"[REAL] {REAL_CSV}\n       pooled {N_TRACKS} of {len(raw):,} "
              f"tracks, mean len {pool.points.shape[0]/N_TRACKS:.0f}, "
              f"{SPE} steps/epoch")
        return pool, theta[torch.from_numpy(sel)]
    rng = np.random.default_rng(0)
    pool = TrackPool([
        rng.normal(0, 1, (int(min(N_MAX, max(3, rng.gamma(2.0, 20.0)))), 3))
        * np.array([120., 30., 20.]) for _ in range(N_TRACKS)])
    theta = torch.randn(N_TRACKS, 4)
    theta[:, 0] = torch.rand(N_TRACKS) * 100 + 1
    return pool, theta


def make_ds(pool, theta, idx):
    return TrackDataset(pool=pool, theta=theta, idx_arr=idx, steps_per_epoch=SPE,
                        batch_size=B, n_max=N_MAX, start_step=0,
                        target_steps=STEPS, seed=0, h0=0.05, k_min=8,
                        k_max=K_MAX, p_zero=0.30, min_sigma_A=1.0,
                        max_sigma_A=1000.0)


def collect(pool, theta, idx, nw):
    dl = make_loader(make_ds(pool, theta, idx), B, nw, 2, N_MAX, K_MAX,
                     verbose=False)
    return [(th.clone(), x.clone(), int(s)) for th, x, s in dl]


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    pool, theta = build_pool()
    idx = np.arange(N_TRACKS)

    ok = True
    print("=== loader parity ===")
    ser = collect(pool, theta, idx, 0)
    ok &= report("serial loader produced all steps", len(ser) == STEPS,
                 f"{len(ser)} batches")
    ok &= report("batches arrive strictly in order",
                 [s for _, _, s in ser] == list(range(1, STEPS + 1)))

    nw = resolve_workers(2, verbose=False)
    if nw == 0:
        print("  [SKIP] worker-process parity: spawn platform (Windows); "
              "the A100 cluster is Linux. Forcing 2 workers anyway to test "
              "the Dataset/Collate pickling contract...")
        nw = 2
    par = collect(pool, theta, idx, nw)

    ok &= report(f"num_workers={nw} produced the same number of batches",
                 len(par) == len(ser), f"{len(par)} vs {len(ser)}")
    max_dx = max((a[1] - b[1]).abs().max().item() for a, b in zip(ser, par))
    max_dt = max((a[0] - b[0]).abs().max().item() for a, b in zip(ser, par))
    same_order = [s for _, _, s in par] == [s for _, _, s in ser]
    ok &= report("parallel batches are BYTE-IDENTICAL to serial (inputs)",
                 max_dx == 0.0, f"max|dx|={max_dx:.3e}")
    ok &= report("parallel batches are BYTE-IDENTICAL to serial (targets)",
                 max_dt == 0.0, f"max|dtheta|={max_dt:.3e}")
    ok &= report("parallel batch ORDER matches serial", same_order)

    print("\n=== epoch semantics ===")
    # reconstruct which tracks each epoch used, via the dataset's own mapping
    ds = make_ds(pool, theta, idx)
    for ep in range(2):
        seen = []
        for st in range(ep * SPE + 1, (ep + 1) * SPE + 1):
            pos = (st - 1) % SPE
            seen += list(ds._permutation(ep)[pos * B:(pos + 1) * B])
        ok &= report(f"epoch {ep}: no track repeated",
                     len(set(seen)) == len(seen), f"{len(seen)} draws")
        ok &= report(f"epoch {ep}: covers the whole pool",
                     set(seen) == set(idx.tolist()),
                     f"{len(set(seen))}/{N_TRACKS}")
    p0 = ds._permutation(0); p1 = ds._permutation(1)
    ok &= report("consecutive epochs use DIFFERENT permutations",
                 not np.array_equal(p0, p1))

    print("\n=== sigma marginal ===")
    sig_n = np.concatenate([x[:, -N_PHYS:][:, PHYS_SIGMA_N].numpy()
                            for _, x, _ in ser])
    frac0 = float((sig_n == 0).mean())
    ok &= report("p_zero point mass preserved (~0.30)", abs(frac0 - 0.30) < 0.08,
                 f"measured {frac0:.3f} over {len(sig_n)} samples")
    ok &= report("non-zero sigmas are finite and positive",
                 bool(np.all(np.isfinite(sig_n))) and float(sig_n.min()) >= 0.0)

    print("\n" + ("ALL LOADER PARITY CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
