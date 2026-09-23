"""v23_final: single source of truth for every env-overridable knob.

v23 = the v22 GVP-EGNN baseline (train_gvp_egnn_v22.py) + the a100_final
blur-robustness diagnosis, per reports/v23_architecture_plan.tex.

Everything here is either
  (a) carried forward VERBATIM from v22            -> marked [v22]
  (b) ported VERBATIM from a100_final              -> marked [a100]
  (c) new in v23 and pinned by the spec            -> marked [v23-spec]
  (d) a judgment call the spec left open           -> marked [v23-judgment]
      (every (d) is also documented in README.md)
"""
import os

# ---------------------------------------------------------------- data paths
# [v23-spec] S{training}: the expanded v2 corpus, NOT v22's ~320k pool.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRAIN_CSV = os.environ.get("TRAIN_CSV", "data/siimpl_rot/siimpl_train_v2.csv")
EVAL_CSV = os.environ.get("EVAL_CSV", "data/siimpl_rot/siimpl_eval_v2.csv")


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


def _b(name, default):
    return os.environ.get(name, str(int(default))) == "1"


# ---------------------------------------------------- Pool A / Pool B split
# [v23-judgment] Stage 2 trains on Pool A only (isotropic, no channeling
# enrichment) so the posterior's implicit prior stays uniform-on-S^2 -- v22's
# rule was simply `ion_number < POOL_B_ION_START(300_000)`.  That rule is WRONG
# for the v2 corpus and would silently mislabel ~714k isotropic tracks as
# Pool B.  merge_tiered_data.py renumbers each source file with a running
# offset = previous local_max + 1, and the FIRST source (siimpl_train.csv,
# offset 0) keeps its original ids.  Verified by `tail -1`:
#     max(ion_number) in siimpl_train.csv          = 347_834
#     max(ion_number) in siimpl_train_v2.csv       = 1_062_059
# and generate_data_tiered.py samples directions with sample_iso() -- the three
# `_extra_{low,mid,high}` tier files are ISOTROPIC, i.e. Pool-A-like.  So:
#     [0, 300_000)          original Pool A   (isotropic)
#     [300_000, 347_834]    original Pool B   (channeling-enriched)
#     (347_834, ...]        the v2 tier extras (isotropic)  -> Pool A
POOL_B_LO = _i("POOL_B_LO", 300_000)
POOL_B_HI = _i("POOL_B_HI", 347_834)
# gen_meta.json records train_poolB = 47,835 tracks; assert against it at load
# time so a corpus regeneration cannot silently break the split.
POOL_B_EXPECTED = _i("POOL_B_EXPECTED", 47_835)
POOL_B_TOLERANCE = _f("POOL_B_TOLERANCE", 0.02)   # 2% slack for n_vac>=3 filter

# ------------------------------------------------------- backbone [v22]
# S{backbone}: carried forward from v22 lines 110-118, UNCHANGED.  Capacity is
# deliberately not re-opened (information-limited, not capacity-limited).
HIDDEN_DIM = _i("HIDDEN_DIM", 96)
N_LAYERS = _i("N_LAYERS", 6)
N_HEADS = _i("N_HEADS", 8)
D_PROJ = _i("D_PROJ", 48)
D_LATENT = _i("D_LATENT", 384)
V_DIM = _i("V_DIM", 8)
USE_GVP = _b("USE_GVP", True)
USE_AXIS_FEATS = _b("USE_AXIS_FEATS", True)
N_AXES = _i("N_AXES", 2)

# --------------------------------------------- data pipeline [a100] + [v23]
# [a100] MAX_POINTS=600 truncates NOTHING (measured train_v2 max = 587 points,
# eval_v2 max = 545).  Margin is only 13 points -- data_pipeline asserts it.
MAX_POINTS = _i("MAX_POINTS", 600)
# [a100] blur-adaptive hybrid graph, S{graph}.  Do NOT re-derive: H0 was fit to
# the measured median unblurred nearest-neighbour spacing in R_g-normalized
# units (low 0.100 / mid 0.039 / high 0.012, a100_final/_data_stats.json).
K_MAX = _i("K_MAX", 24)
K_MIN = _i("K_MIN", 8)
H0 = _f("H0", 0.05)
# [a100] blur-deconvolved R_g normalizer, S{normalizer}
FLOOR_ABS = _f("FLOOR_ABS", 1e-4)      # Angstrom^2
FLOOR_FRAC = _f("FLOOR_FRAC", 0.1)     # deconvolution capped at 10% of sigma
# [a100] sigma-feature log floors
SIGMA0_A = _f("SIGMA0_A", 1.0)         # Angstrom
SIGMA_N_FLOOR = _f("SIGMA_N_FLOOR", 1e-3)
# [a100] continuous blur sampling (P_ZERO point mass at exactly 0, else
# log-uniform over [MIN_SIGMA_A, MAX_SIGMA_A] = [0.1nm, 100nm]).
# [v23-spec] S{training} item 2: this same convention now drives BOTH stages.
P_ZERO = _f("P_ZERO", 0.30)
MIN_SIGMA_A = _f("MIN_SIGMA_A", 1.0)
MAX_SIGMA_A = _f("MAX_SIGMA_A", 1000.0)

# -------------------------------------------- conditioning vector [v23-spec]
# S{physfeat}: D_AUG = 384 + 5 (v22 phys) + 4 (new higher-moment scalars)
#                    + 16 (sigma-conditioning, sinusoidal-embedded) = 409
N_PHYS_V22 = 5        # [log n_vac, log extent, log R_g, elong1, elong2]
N_MOMENT = 4          # [c_corr, c_uncorr, ratio_corr, ratio_uncorr]
N_SIGMA_RAW = 2       # [sigma_A, sigma_n]  -- raw, carried for the embedding
N_PHYS_COND = N_PHYS_V22 + N_MOMENT          # 9 standardized scalar features
N_PHYS = N_PHYS_COND + N_SIGMA_RAW           # 11 columns in x_flat's phys block
N_SIGMA_FREQ = _i("N_SIGMA_FREQ", 4)         # f_j = pi * 2^(j-1), j=1..4
D_SIGMA_EMB = 4 * N_SIGMA_FREQ               # sin/cos x 2 scalars x 4 freqs = 16
D_AUG = D_LATENT + N_PHYS_COND + D_SIGMA_EMB  # 409

# ---------------------------------------------------------- head [v22]+[v23]
# S{head}: keep v22's unconstrained DirectionalPosterior exactly (K=4 vMF,
# 2-hidden-layer MLP hidden=256, GMM1D K=3), add sigma-conditioning and the
# residual-on-PCA fallback.  Do NOT reintroduce the V-bottleneck.
N_DIR_COMP = _i("N_DIR_COMP", 4)
N_E_COMP = _i("N_E_COMP", 3)
HEAD_HIDDEN = _i("HEAD_HIDDEN", 256)
# [v23-spec] S{fallback}: zero-init the mu output ROWS of the head's shared
# final Linear (not a separate delta submodule -- none exists).
ZERO_INIT_MU_ROWS = _b("ZERO_INIT_MU_ROWS", True)
# [v23-spec] S{head}: zero-init the first-layer weight COLUMNS feeding the 16
# embedded sigma dims, and (staged build plan item 1) the 4 new higher-moment
# feature columns.
ZERO_INIT_SIGMA_COLS = _b("ZERO_INIT_SIGMA_COLS", True)
ZERO_INIT_MOMENT_COLS = _b("ZERO_INIT_MOMENT_COLS", True)

# ------------------------------------------------ energy space [v22]
LOG_ENERGY = _b("LOG_ENERGY", True)
E_MIN_KEV = _f("E_MIN_KEV", 1.0)
E_MAX_KEV = _f("E_MAX_KEV", 105.0)

# ------------------------------------------------------- optimization [v22]
BATCH_SIZE = _i("BATCH_SIZE", 256)
LR_MAX = _f("LR_MAX", 3e-4)
LR_MIN = _f("LR_MIN", 1e-5)
WEIGHT_DECAY = _f("WEIGHT_DECAY", 1e-4)
WARMUP_EPOCHS = _i("WARMUP_EPOCHS", 5)
MAX_EPOCHS = _i("MAX_EPOCHS", 300)
PATIENCE = _i("PATIENCE", 40)
GRAD_CLIP = _f("GRAD_CLIP", 1.0)
LOSS_CLAMP = _f("LOSS_CLAMP", 1000.0)
ALPHA_START = _f("ALPHA_START", 0.5)
ALPHA_END = _f("ALPHA_END", 0.1)
BETA_START = _f("BETA_START", 0.1)
BETA_END = _f("BETA_END", 0.05)
# [v22] S{loss}: sign-aware curriculum on the AUXILIARY DirectionHead only.
# Spec S{loss}: "Keep AUX_SIGN_WARMUP=30 as-is."  The timing LOGIC is unchanged;
# train.py only ADDS instrumentation reporting how many low-sigma examples have
# been seen by epoch 30 (the empirical verification the spec asks for).
AUX_SIGN_WARMUP = _i("AUX_SIGN_WARMUP", 30)

# [v23-judgment] v22 defined an "epoch" as one pass over its ~304k-track train
# split => 1,187 steps at batch 256, and tuned MAX_EPOCHS=300 / PATIENCE=40 /
# AUX_SIGN_WARMUP=30 against THAT step count.  The v2 corpus is 3.3x larger, so
# a full pass is 3,940 steps; keeping "epoch == full pass" would either blow the
# budget (300 x 3,940 = 1.18M steps) or, if MAX_EPOCHS were cut to ~90 to
# compensate, would silently stretch the aux-sign warmup from 10% to 33% of
# training.  Both break v22's tuned curriculum.  We therefore decouple the two:
# an "epoch" is a fixed STEPS_PER_EPOCH of the shuffled epoch-permutation
# stream, defaulted to ~v22's own steps/epoch, so BOTH the absolute step count
# per epoch AND the 30/300 curriculum proportion are preserved.  Sampling is
# still a full random permutation of the whole 1M pool consumed in order, so
# every track is still visited -- it just takes ~3.15 "epochs" to complete one
# full pass.  Set SUB_EPOCH_STEPS=0 to fall back to epoch == full pass.
SUB_EPOCH_STEPS = _i("SUB_EPOCH_STEPS", 1250)

# ------------------------------------------------------------- Stage 2 [v22]
STAGE2_EPOCHS = _i("STAGE2_EPOCHS", 80)
STAGE2_LR_MAX = _f("STAGE2_LR_MAX", 3e-4)
STAGE2_LR_MIN = _f("STAGE2_LR_MIN", 1e-6)
STAGE2_WARMUP = _i("STAGE2_WARMUP", 3)
STAGE2_PATIENCE = _i("STAGE2_PATIENCE", 20)
STAGE2_GRAD_CLIP = _f("STAGE2_GRAD_CLIP", 0.5)

# ------------------------------------------------------------------ EMA [v22]
USE_EMA = _b("USE_EMA", True)
EMA_DECAY = _f("EMA_DECAY", 0.999)

# ------------------------------------------------- calibration [v23-spec]
# S{calibration} item 1: L_calib = lambda * sum_k w_k kappa_k.
# Staged build plan item 1 requires lambda=0 (INACTIVE) for the swap-in check,
# then lambda>0 for the full run.  Default 0.0 so the build/verification pass
# cannot accidentally change forward-pass behaviour; set LAMBDA_KAPPA at launch.
_LK = os.environ.get("LAMBDA_KAPPA", "0.0").strip()
# LAMBDA_KAPPA=auto implements the spec's own tuning rule literally: "start small
# (e.g. tuned so L_calib is ~1% of the direction NLL at initialization)".  train.py
# measures both quantities on real batches at init and solves for lambda.
LAMBDA_KAPPA_AUTO = _LK.lower() == "auto"
LAMBDA_KAPPA = 0.0 if LAMBDA_KAPPA_AUTO else float(_LK)
LAMBDA_KAPPA_TARGET_FRAC = _f("LAMBDA_KAPPA_TARGET_FRAC", 0.01)
# S{calibration} item 2: calibration-aware checkpoint SELECTION (not training).
# Select the checkpoint minimizing validation ECE among those within
# ECE_NLL_TOL (absolute nats) of the best validation NLL.  No parameters are fit
# on validation data and no transformation is applied to the outputs -- this is
# model selection, explicitly NOT post-hoc temperature scaling.
SELECT_ON_ECE = _b("SELECT_ON_ECE", True)
ECE_NLL_TOL = _f("ECE_NLL_TOL", 0.02)
VAL_ECE_SAMPLES = _i("VAL_ECE_SAMPLES", 200)
VAL_ECE_TRACKS = _i("VAL_ECE_TRACKS", 2000)

# -------------------------------------------------------------- run control
SEED = _i("SEED", 42)
VAL_FRACTION = _f("VAL_FRACTION", 0.05)
TIME_BUDGET_HOURS = _f("TIME_BUDGET_HOURS", 0.0)   # 0 = no wall-clock cap
CHECKPOINT_EVERY_SEC = _f("CHECKPOINT_EVERY_SEC", 20 * 60)
MAX_TRAIN_TRACKS = _i("MAX_TRAIN_TRACKS", 0)       # 0 = use the whole v2 corpus
TRAIN_NROWS = _i("TRAIN_NROWS", 0)                 # 0 = read the whole CSV
EVAL_NROWS = _i("EVAL_NROWS", 0)
SMOKE = _b("SMOKE", False)

# [a100] real worker PROCESSES: batch prep is CPU-bound and does not
# parallelize across threads.  Windows uses spawn -> forced to 0 (see loader).
_default_workers = 0 if os.name == 'nt' else max(4, min(16, os.cpu_count() or 8))
N_WORKERS = _i("N_WORKERS", _default_workers)
PREFETCH = _i("PREFETCH", 4)

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "smearing_resolution/architecture_experiments/v23_final/results")

# ----------------------------------------------- eval grid [a100] + [v23-spec]
# S{eval}: the identical 9-sigma x 3-energy-tier convention already validated in
# a100_final/eval.py and the reference full_sweep_9tier_6nm_EMA.csv.
ENERGY_BINS = [("low_1_5keV", 1.0, 5.0), ("mid_5_20keV", 5.0, 20.0),
               ("high_20_105keV", 20.0, 105.0)]
SIGMAS_NM = [0, 1, 3, 6, 10, 20, 30, 50, 100]
EVAL_SIGMAS_NM_QUICK = [0, 10, 50]
EVAL_N_PER_BIN = _i("EVAL_N_PER_BIN", 1000)   # S{eval}: 1000 held-out tracks/cell min
EVAL_CHUNK = _i("EVAL_CHUNK", 256)
EVAL_N_SAMPLES = _i("EVAL_N_SAMPLES", 200)
# [a100] unchanged from eval_diag_r7_trunc_calibration.py:281
ANG_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95]

# ------------------------------------------- pre-registered gates [v23-spec]
# S{eval}.  These are DATA, printed verbatim by eval.py alongside the numbers so
# the gate is stated in the script's own output, not only computed.
SUCCESS_CELLS = [("mid_5_20keV", 6), ("mid_5_20keV", 10), ("mid_5_20keV", 20),
                 ("high_20_105keV", 6), ("high_20_105keV", 10),
                 ("high_20_105keV", 20), ("high_20_105keV", 30),
                 ("high_20_105keV", 50)]
GUARD_CELLS = [(b, s) for b, _lo, _hi in ENERGY_BINS for s in (0, 1)]
# v22 ROT_FINAL pooled zero-blur reference (real eval_results.csv, 49,200 rows):
# median angular error 6.8 deg (mean-readout) / 6.7 (mode), head-tail 97.0/96.6%.
V22_POOLED_AXIS_ERR_DEG = _f("V22_POOLED_AXIS_ERR_DEG", 6.8)
V22_POOLED_HEADTAIL_PCT = _f("V22_POOLED_HEADTAIL_PCT", 97.0)
# S{eval}: "must not regress ... by more than a small, explicitly-agreed margin".
# The spec does not fix the margin -> these are the explicit values v23 uses.
GUARD_AXIS_MARGIN_DEG = _f("GUARD_AXIS_MARGIN_DEG", 1.0)
GUARD_HEADTAIL_MARGIN_PCT = _f("GUARD_HEADTAIL_MARGIN_PCT", 1.0)
# S{eval}: reference curves the success gate is evaluated against.  Both must be
# measured BEFORE v23 training (from_scratch_dr already is; v22 ROT_FINAL's full
# sweep is the cheap eval-only pass the spec requires).  Supplied as CSVs.
REF_SWEEP_CSV = os.environ.get("REF_SWEEP_CSV", "")


def summary():
    return dict(HIDDEN_DIM=HIDDEN_DIM, N_LAYERS=N_LAYERS, D_LATENT=D_LATENT,
                D_AUG=D_AUG, N_PHYS=N_PHYS, D_SIGMA_EMB=D_SIGMA_EMB,
                N_DIR_COMP=N_DIR_COMP, N_E_COMP=N_E_COMP,
                HEAD_HIDDEN=HEAD_HIDDEN, MAX_POINTS=MAX_POINTS,
                K_MAX=K_MAX, K_MIN=K_MIN, H0=H0,
                P_ZERO=P_ZERO, MIN_SIGMA_A=MIN_SIGMA_A, MAX_SIGMA_A=MAX_SIGMA_A,
                BATCH_SIZE=BATCH_SIZE, MAX_EPOCHS=MAX_EPOCHS,
                SUB_EPOCH_STEPS=SUB_EPOCH_STEPS,
                AUX_SIGN_WARMUP=AUX_SIGN_WARMUP,
                STAGE2_EPOCHS=STAGE2_EPOCHS, LAMBDA_KAPPA=LAMBDA_KAPPA,
                LAMBDA_KAPPA_AUTO=LAMBDA_KAPPA_AUTO,
                SELECT_ON_ECE=SELECT_ON_ECE, ECE_NLL_TOL=ECE_NLL_TOL,
                USE_EMA=USE_EMA, EMA_DECAY=EMA_DECAY, SEED=SEED,
                TRAIN_CSV=TRAIN_CSV, EVAL_CSV=EVAL_CSV)
