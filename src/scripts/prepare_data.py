"""Merge the generated pools into the files the trainer expects, then build the
kNN / physics caches.

Steps:
  1. siimpl_train.csv        = Pool A  +  Pool B (ion_number offset by +OFFSET so
                               the two pools keep disjoint track ids)
  2. siimpl_eval_merged.csv  = copy of siimpl_eval.csv
  3. preprocess both (builds the kNN/phys cache consumed on the first training run)

    python src/scripts/prepare_data.py

CPU-bound; the cache build is the slow part (a few minutes for the default sizes).
"""
import os, sys, time, shutil
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
import pandas as pd
from src.utils.data_utils import preprocess_egnn

D = os.path.join(_ROOT, "data", "siimpl_rot")
POOLA = f"{D}/siimpl_train_poolA.csv"
POOLB = f"{D}/siimpl_train_poolB.csv"
EVSRC = f"{D}/siimpl_eval.csv"
TRAIN = f"{D}/siimpl_train.csv"
EVAL  = f"{D}/siimpl_eval_merged.csv"
OFFSET = 300000                       # keeps Pool B ion_numbers disjoint from Pool A

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

if __name__ == "__main__":
    log("merging Pool A (as-is) + Pool B (ion +%d) -> siimpl_train.csv ..." % OFFSET)
    shutil.copy(POOLA, TRAIN)                                   # includes header
    n = 0
    with open(TRAIN, "a", newline="") as f:
        for ch in pd.read_csv(POOLB, chunksize=400000):
            ch["ion_number"] = ch["ion_number"] + OFFSET
            ch.to_csv(f, header=False, index=False); n += len(ch)
    log(f"appended {n:,} Pool B rows")
    shutil.copy(EVSRC, EVAL); log("copied eval -> siimpl_eval_merged.csv")

    log("preprocessing TRAIN (builds the kNN/phys cache) ...")
    xtr = preprocess_egnn(TRAIN, k_neighbors=16, return_phys=True)
    log(f"train cache built: n_max={xtr[3]}, tracks={xtr[0].shape[0]:,}")
    log("preprocessing EVAL ...")
    xev = preprocess_egnn(EVAL, k_neighbors=16, return_phys=True)
    log(f"eval cache built: n_max={xev[3]}, tracks={xev[0].shape[0]:,}")
    log("DONE merge + preprocess")
