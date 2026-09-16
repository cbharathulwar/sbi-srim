"""a100_final: parallel batch preparation.

Batch prep (smear -> deconvolved-Rg normalize -> blur-adaptive graph, per
track) is CPU-bound and measured at ~0.51 ms/track single-core.  On the 2080 Ti
that hides behind a 126 ms GPU step, but an A100 step is several times faster,
so it becomes the bottleneck.  Threads do NOT fix it -- measured speedup is
only 1.23x at 16 threads, because numpy's RNG and the per-track packing hold
the GIL -- so this module runs real worker PROCESSES via
torch.utils.data.DataLoader.

Smearing stays ON-THE-FLY and continuous by design.  Sigma is a fresh
continuous draw every time a track is used, which gives unbounded augmentation
diversity; precomputing blurred clouds would force a finite discrete sigma grid
and risk reintroducing a milder form of the sigma-blindness this rebuild
exists to fix.

Everything the workers need is held as an attribute of the Dataset, so under
fork (Linux) it is inherited copy-on-write at zero cost.
"""
from __future__ import annotations
import os
import numpy as np
import torch

from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    _smear_and_prepare_one_v2, sample_sigma_continuous, flat_dim)


class TrackDataset(torch.utils.data.Dataset):
    """Map-style dataset over GLOBAL SAMPLE INDEX g = (step-1)*B + j.

    Paired with a sequential sampler this reproduces the epoch-permutation
    batch composition EXACTLY and delivers batches strictly IN ORDER.

    Training semantics are unchanged from the serial path: same epoch
    permutation, same sigma distribution (p_zero point mass + log-uniform),
    same on-the-fly continuous smearing.  Only the RNG STREAM differs -- sigma
    comes from a per-SAMPLE generator seeded by (seed, g) rather than a
    per-batch vector draw, so any worker can produce any sample independently.
    The draws remain iid with the identical marginal, which is all the
    augmentation depends on.
    """

    def __init__(self, pool, theta, idx_arr, steps_per_epoch, batch_size,
                 n_max, start_step, target_steps, seed,
                 h0, k_min, k_max, p_zero, min_sigma_A, max_sigma_A):
        self.pool, self.theta, self.idx_arr = pool, theta, idx_arr
        self.spe, self.B, self.n_max = steps_per_epoch, batch_size, n_max
        self.seed = seed
        self.h0, self.k_min, self.k_max = h0, k_min, k_max
        self.p_zero, self.min_s, self.max_s = p_zero, min_sigma_A, max_sigma_A
        self.start = start_step * batch_size
        self.n = max(0, target_steps - start_step) * batch_size
        self._perm_cache = {}

    def __len__(self):
        return self.n

    def epoch_of(self, step):
        return (step - 1) // self.spe

    def _permutation(self, epoch):
        """Per-PROCESS cache.  A sequential sampler hands each worker whole
        contiguous batches, so consecutive samples share one epoch and a
        2-entry cache hits essentially always."""
        p = self._perm_cache.get(epoch)
        if p is None:
            p = np.random.default_rng(
                (self.seed + 1) * 7_919 + epoch).permutation(self.idx_arr)
            self._perm_cache[epoch] = p
            for k in [k for k in self._perm_cache if k < epoch - 1]:
                del self._perm_cache[k]
        return p

    def __getitem__(self, i):
        g = self.start + i
        step = g // self.B + 1
        j = g % self.B
        pos = (step - 1) % self.spe
        t_idx = int(self._permutation(self.epoch_of(step))[pos * self.B + j])
        rng = np.random.default_rng((self.seed + 1) * 2_000_003 + g)
        sig = float(sample_sigma_continuous(1, rng, self.p_zero,
                                            self.min_s, self.max_s)[0])
        co, kn, ph = _smear_and_prepare_one_v2(
            self.pool[t_idx], sig, self.n_max, rng,
            self.h0, self.k_min, self.k_max, pad=False)
        return co, kn, ph, self.theta[t_idx], step


class Collate:
    """Pad the compact per-track arrays into the flat model input.

    A class, not a closure, so it stays picklable for spawn-based workers.
    """

    def __init__(self, n_max, k_max):
        self.n_max, self.k_max = n_max, k_max
        self.flat = flat_dim(n_max, k_max)

    def __call__(self, items):
        B = len(items)
        n_max, k_max = self.n_max, self.k_max
        x = torch.zeros(B, self.flat, dtype=torch.float32)
        th = torch.empty(B, 4, dtype=torch.float32)
        c_end, k_end = n_max * 3, n_max * (3 + k_max)
        for b, (co, kn, ph, theta, _s) in enumerate(items):
            n = co.shape[0]
            x[b, :c_end].view(n_max, 3)[:n] = torch.from_numpy(co)
            xk = x[b, c_end:k_end].view(n_max, k_max)
            xk.fill_(-1.0)
            xk[:n] = torch.from_numpy(kn.astype(np.float32))
            x[b, k_end:] = torch.from_numpy(ph)
            th[b] = theta
        return th, x, items[0][4]


def resolve_workers(requested, verbose=True):
    """Windows uses spawn, not fork: every worker would re-import the training
    module and reload the CSVs.  Force synchronous loading there."""
    if requested > 0 and os.name == 'nt':
        if verbose:
            print(f"[LOADER] WARNING: N_WORKERS={requested} ignored on Windows "
                  f"(spawn would reload the dataset per worker); using 0. "
                  f"The A100 cluster is Linux, where workers are enabled.")
        return 0
    return requested


def make_loader(dataset, batch_size, n_workers, prefetch, n_max, k_max,
                pin_memory=False, verbose=True):
    n_workers = resolve_workers(n_workers, verbose)
    kw = dict(persistent_workers=True, prefetch_factor=prefetch) if n_workers > 0 else {}
    if verbose:
        print(f"[LOADER] {n_workers} worker process(es), "
              f"prefetch_factor={kw.get('prefetch_factor', 'n/a')}, "
              f"{len(dataset) // max(batch_size, 1)} steps queued")
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
        collate_fn=Collate(n_max, k_max), drop_last=True,
        pin_memory=pin_memory, **kw)
