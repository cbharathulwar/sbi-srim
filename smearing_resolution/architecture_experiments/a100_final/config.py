"""a100_final: single source of truth for every env-overridable knob.

Named scaling-ladder configs are selected with CONFIG=small|large; any
individual knob can still be overridden by its own env var, which always wins.
"""
import os

# Final v2 corpus: 1,009,384 train / 101,731 eval tracks, tier-weighted
# ~25/30/45 low/mid/high.  These are the DEFAULTS for the paper run.  Override
# with TRAIN_CSV / EVAL_CSV to fall back to the original smaller files
# (siimpl_train.csv / siimpl_eval_merged.csv).
TRAIN_CSV = os.environ.get("TRAIN_CSV", "data/siimpl_rot/siimpl_train_v2.csv")
EVAL_CSV = os.environ.get("EVAL_CSV", "data/siimpl_rot/siimpl_eval_v2.csv")

# ---- scaling ladder -------------------------------------------------------
# small : the winner's capacity (hidden 112 / 6 layers, ~0.8M params), so the
#         "is it undersized?" question is answered against a like-for-like
#         control that differs ONLY by the 5 architectural changes.
# large : 2x width, ~1.7x depth.  Width dominates cost here (edge messages are
#         O(E * hidden)), so 112->224 is ~4x the MLP flops and 6->10 layers
#         another ~1.7x: roughly 7x the winner's compute per sample, ~3.2M
#         params.  That is a real capacity test while still leaving an 80GB
#         A100 comfortable at batch 384 with full 600-point clouds.
_LADDER = {
    "small": dict(HIDDEN_DIM=112, N_LAYERS=6, BATCH_SIZE=512),
    "large": dict(HIDDEN_DIM=224, N_LAYERS=10, BATCH_SIZE=384),
}
CONFIG = os.environ.get("CONFIG", "small")
if CONFIG not in _LADDER:
    raise SystemExit(f"CONFIG must be one of {sorted(_LADDER)}, got {CONFIG!r}")
_D = _LADDER[CONFIG]


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


HIDDEN_DIM = _i("HIDDEN_DIM", _D["HIDDEN_DIM"])
N_LAYERS = _i("N_LAYERS", _D["N_LAYERS"])
BATCH_SIZE = _i("BATCH_SIZE", _D["BATCH_SIZE"])
COEF_HIDDEN = _i("COEF_HIDDEN", 256)

# ---- posterior ------------------------------------------------------------
POSTERIOR_TYPE = os.environ.get("POSTERIOR_TYPE", "vmf_mixture")
N_DIR_COMP = _i("N_DIR_COMP", 10)          # widened from the winner's 4
GRID_SUBDIV = _i("GRID_SUBDIV", 3)         # 642 grid points

# ---- data pipeline --------------------------------------------------------
# Longest track is 579 (train) / 545 (eval) points, so 600 is a hard ceiling
# that truncates NOTHING -- the winner's 450 cut the top ~0.1% of tracks.
# n_max is FORCED to MAX_POINTS so train and eval share one padding width.
MAX_POINTS = _i("MAX_POINTS", 600)
K_MAX = _i("K_MAX", 24)
K_MIN = _i("K_MIN", 8)
H0 = _f("H0", 0.05)
P_ZERO, MIN_SIGMA_A, MAX_SIGMA_A = _f("P_ZERO", 0.30), _f("MIN_SIGMA_A", 1.0), _f("MAX_SIGMA_A", 1000.0)

# ---- optimization ---------------------------------------------------------
LR_MAX = _f("LR_MAX", 3e-4)
LR_MIN = _f("LR_MIN", 1e-5)
WARMUP_STEPS = _i("WARMUP_STEPS", 300)
GRAD_CLIP = _f("GRAD_CLIP", 1.0)
ENERGY_LOSS_WEIGHT = _f("ENERGY_LOSS_WEIGHT", 0.1)
SIGN_LOSS_WEIGHT = _f("SIGN_LOSS_WEIGHT", 1.0)
EMA_DECAY = _f("EMA_DECAY", 0.999)
LOG_ENERGY = _i("LOG_ENERGY", 0)
# bf16 autocast on the backbone only.  DEFAULT OFF: src/models/egnn.py's
# EGNNLayer allocates its message-aggregation buffer with
# `torch.zeros(...)` (always fp32) and then scatter_add_s the autocast bf16
# messages into it, which raises
#   RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype
# Fixing that needs a one-line dtype change inside src/models/egnn.py, and this
# rebuild is not permitted to modify files outside a100_final/.  Left as a flag
# so it can be switched on cheaply if that upstream fix ever lands; expect
# roughly 1.5-2x throughput on A100 if it does.
AMP = _i("AMP", 0)
SEED = _i("SEED", 0)

TARGET_STEPS = _i("TARGET_STEPS", 60000)
TIME_BUDGET_HOURS = _f("TIME_BUDGET_HOURS", 14.5)
CHECKPOINT_EVERY_SEC = _f("CHECKPOINT_EVERY_SEC", 20 * 60)
# ---- data loading ---------------------------------------------------------
# Batch prep is CPU-bound and does NOT parallelize across THREADS (measured:
# only 1.23x at 16 threads -- numpy's RNG and the pack loop hold the GIL), so
# these are real worker PROCESSES via torch.utils.data.DataLoader.
#
# Default assumption: a typical A100 node allocates ~8-16 CPU cores per GPU, so
# take the box's core count capped at 16.  Measured: ~0.51 ms/track/core
# (~1950 tracks/s).  At batch 512 a single core needs ~262 ms/batch against an
# estimated ~30-60 ms A100 step, so 8-16 workers is the right order of
# magnitude.  Raise N_WORKERS if [LOADER-WAIT] in the logs stays above a few %.
#
# WINDOWS: multiprocessing uses spawn, not fork, so each worker would re-import
# this module and reload the CSVs.  Default to 0 there (synchronous,
# in-process).  The A100 cluster is Linux, where fork + TrackPool shares the
# track pool copy-on-write for free.
_default_workers = 0 if os.name == 'nt' else max(4, min(16, os.cpu_count() or 8))
N_WORKERS = _i("N_WORKERS", _default_workers)
PREFETCH = _i("PREFETCH", 4)                # batches per worker kept in flight

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    f"smearing_resolution/architecture_experiments/a100_final/results_{CONFIG}")

# ---- eval grid (identical to the reference sweep) -------------------------
ENERGY_BINS = [("low_1_5keV", 1.0, 5.0), ("mid_5_20keV", 5.0, 20.0),
               ("high_20_105keV", 20.0, 105.0)]
SIGMAS_NM = [0, 1, 3, 6, 10, 20, 30, 50, 100]
EVAL_SIGMAS_NM_QUICK = [0, 10, 50]
EVAL_N_PER_BIN = _i("EVAL_N_PER_BIN", 600)
EVAL_CHUNK = _i("EVAL_CHUNK", 256)

SMOKE = os.environ.get("SMOKE", "0") == "1"
MAX_TRAIN_TRACKS = _i("MAX_TRAIN_TRACKS", 0)


def summary():
    return dict(CONFIG=CONFIG, HIDDEN_DIM=HIDDEN_DIM, N_LAYERS=N_LAYERS,
                BATCH_SIZE=BATCH_SIZE, MAX_POINTS=MAX_POINTS, K_MAX=K_MAX,
                K_MIN=K_MIN, H0=H0, POSTERIOR_TYPE=POSTERIOR_TYPE,
                N_DIR_COMP=N_DIR_COMP, TARGET_STEPS=TARGET_STEPS,
                TIME_BUDGET_HOURS=TIME_BUDGET_HOURS, AMP=AMP, SEED=SEED)
