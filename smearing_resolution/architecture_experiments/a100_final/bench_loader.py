"""Quantify the batch-preparation bottleneck: pure CPU prep cost per batch vs
GPU step time, and how well it parallelizes across threads.

Run:  python bench_loader.py            (uses a synthetic track pool, fast)
      REAL=1 python bench_loader.py     (uses the real CSV pool)
"""
import sys, os, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from smearing_resolution.architecture_experiments.a100_final.data_pipeline import (
    _smear_and_prepare_one_v2, flat_dim, pack_one, sample_sigma_continuous,
    load_raw_tracks)
from smearing_resolution.architecture_experiments.a100_final.model import DiagModelV2

N_MAX, K_MAX = 600, 24
B = int(os.environ.get("B", "128"))
FLAT = flat_dim(N_MAX, K_MAX)

if os.environ.get("REAL", "0") == "1":
    pool, _th, _n, _ = load_raw_tracks("data/siimpl_rot/siimpl_train.csv",
                                       max_points=N_MAX)
else:
    rng = np.random.default_rng(0)
    # match the measured real distribution: mean ~91 points, tail to ~579
    pool = [rng.normal(0, 1, (int(min(579, max(3, rng.gamma(2.0, 45.0)))), 3))
            * np.array([120., 30., 20.]) for _ in range(4000)]
print(f"[POOL] {len(pool)} tracks, mean len "
      f"{np.mean([len(p) for p in pool]):.0f}, batch={B}")


def prep_one(args):
    pts, sig, seed = args
    return _smear_and_prepare_one_v2(pts, sig, N_MAX, np.random.default_rng(seed))


def make_args(step):
    r = np.random.default_rng(step)
    idx = r.choice(len(pool), size=B, replace=False)
    sig = sample_sigma_continuous(B, r)
    return [(pool[i], sig[j], step * 100003 + j) for j, i in enumerate(idx)]


def timeit(fn, n=5):
    fn()                      # warm
    t = time.perf_counter()
    for i in range(n):
        fn(i + 1)
    return (time.perf_counter() - t) / n


def run_serial(step=0):
    out = torch.empty(B, FLAT, dtype=torch.float32)
    for j, a in enumerate(make_args(step)):
        co, kn, ph = prep_one(a)
        pack_one(out[j], co, kn, ph, N_MAX, K_MAX)
    return out


def run_threads(nw):
    ex = ThreadPoolExecutor(nw)

    def f(step=0):
        out = torch.empty(B, FLAT, dtype=torch.float32)
        for j, (co, kn, ph) in enumerate(ex.map(prep_one, make_args(step))):
            pack_one(out[j], co, kn, ph, N_MAX, K_MAX)
        return out
    return f


print("\n=== CPU batch preparation ===")
t_ser = timeit(run_serial)
print(f"  serial (1 thread)     {t_ser*1000:8.1f} ms/batch   "
      f"{B/t_ser:7.1f} tracks/s")
for nw in (4, 8, 16):
    t = timeit(run_threads(nw))
    print(f"  ThreadPool({nw:2d})        {t*1000:8.1f} ms/batch   "
          f"{B/t:7.1f} tracks/s   speedup {t_ser/t:4.2f}x")

print("\n=== GPU step (fwd+bwd) ===")
if torch.cuda.is_available():
    dev = 'cuda'
    for name, hid, lay in (("small", 112, 6), ("large", 224, 10)):
        m = DiagModelV2(n_max=N_MAX, k_max=K_MAX, hidden_dim=hid,
                        n_layers=lay).to(dev)
        opt = torch.optim.Adam(m.parameters())
        x = run_serial().to(dev)
        d = torch.nn.functional.normalize(torch.randn(B, 3, device=dev), dim=-1)

        def step():
            o = m(x)
            loss = m.direction_nll_axis(o, d).mean() + o['E_pred'].pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        step(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t_gpu = (time.perf_counter() - t0) / 5
        print(f"  {name:5s} (h={hid},L={lay})  {t_gpu*1000:8.1f} ms/batch"
              f"   -> CPU/GPU ratio (serial) = {t_ser/t_gpu:5.2f}")
        del m, opt
        torch.cuda.empty_cache()
else:
    print("  (no CUDA available)")
print("\nRatio > 1 means CPU prep is the bottleneck and parallel loading wins.")
