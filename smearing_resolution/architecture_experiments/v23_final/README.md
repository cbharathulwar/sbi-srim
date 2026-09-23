# v23_final

Implementation of **`reports/v23_architecture_plan.tex`** — v22's GVP-EGNN baseline
(`src/scripts/train_gvp_egnn_v22.py`) grafted onto the `a100_final` blur-robustness
diagnosis, trained on the 1,009,384-track v2 corpus.

Nothing outside this folder is modified. Every reused component from `a100_final/` or
`src/` is copied/adapted in here.

```
config.py            every knob, with [v22] / [a100] / [v23-spec] / [v23-judgment] provenance
data_pipeline.py     blur-deconvolved normalizer, blur-adaptive graph, higher-moment
                     features, dynamic blur sampling, worker-process loader
model.py             sigma-conditioned + residual-on-PCA DirectionalPosterior
metrics.py           ECE / coverage / PCA-baseline, ported unchanged
train.py             two-stage training, blur in BOTH stages, kappa regularizer,
                     calibration-aware checkpoint selection
eval.py              9-sigma x 3-tier grid vs PCA, with the pre-registered gates printed
verify_zero_init.py  the spec's required zero-init verification  (44 checks)
validate_math.py     re-validation of the ported math against v23's own code (5 tests)
check_fast.py        short REAL run: no-NaN / loss-decreases / ckpt round-trip / eval path
```

---

## 1. Spec → code map

| Spec section | Where it lives |
|---|---|
| §backbone (hidden=96, 6 layers, n_heads=8, d_proj=48, d_latent=384, v_dim=8, n_axes=2) | `config.py`, `model.build_v23` — v22 values, unchanged |
| §head — K=4 vMF, hidden=256 MLP, GMM1D K=3, **no V-bottleneck** | `model.VMFMixtureV23`, `model.GMM1DV23` |
| §head — sinusoidal σ embedding, `f_j = π·2^(j-1)`, j=1..4, applied to `log(σ_A+1.0)` and `log(σ_n+1e-3)`, 16 dims | `model.sigma_sinusoidal_embed` |
| §head — zero-init the weight columns feeding the 16 σ dims in `VMFMixture.net` and `GMM1D.net`'s first linear | `model._zero_input_cols`, verified check 1 |
| §fallback — `μ_k = normalize(u + g(σ,z')·δ_k(z'))`, row-level zero-init of the mu rows of the shared final Linear | `model.VMFMixtureV23.zero_init_mu_rows`, verified check 3 |
| §physfeat — v22's 5 features unchanged | `data_pipeline._smear_and_prepare_one_v23` |
| §physfeat — blur-corrected higher-moment tensor `Â`, **signed cosine not raw angle**, both corrected and uncorrected | `data_pipeline.moment_features` |
| §physfeat — D_AUG = 384 + 5 + 4 + 16 = **409** | `config.D_AUG`, asserted in `model.py` |
| §normalizer — `scale = sqrt(max(R²_obs − 3σ_A²(n−1)/n, floor))` | `data_pipeline.deconvolved_scale` (verbatim from `a100_final`) |
| §graph — `r = sqrt(H0² + σ_n²)` or K_MIN, capped at K_MAX; H0=0.05, K_MIN=8, K_MAX=24 | `data_pipeline.hybrid_graph` (verbatim) |
| §training — dynamic blur in **BOTH** stages; two-stage structure preserved | `train.run_stage` (both stages use one `TrackDataset`) |
| §training — v2 corpus, not v22's 320k cap | `config.TRAIN_CSV`, `MAX_TRAIN_TRACKS=0` |
| §loss — `AUX_SIGN_WARMUP=30`, timing logic unchanged; low-σ exposure verified empirically | `train.run_stage`, the `[S{loss} CHECK]` block |
| §calibration 1 — `L_calib = λ Σ_k w_k κ_k`, λ=0 initially | `model.VMFMixtureV23.kappa_penalty`, verified check 7 |
| §calibration 1 — λ "tuned so L_calib ≈1% of the direction NLL at initialization" | `train.calibrate_lambda_kappa` (`LAMBDA_KAPPA=auto`) |
| §calibration 2 — select on validation ECE among near-best-NLL checkpoints | `train.CalibrationAwareSelector` |
| §calibration — verify the premise: κ growth vs mixture-weight collapse | `eval.py` MECHANISM table (`mean_kappa`, `mean_top_weight` per cell) |
| §calibration — 2–3 seeds, ensemble the posteriors | `eval.py` `CKPT=a.pt,b.pt,c.pt` |
| §eval — 9σ × 3 tiers, ≥1000 tracks/cell, PCA baseline same convention, ECE per cell | `eval.py`, `metrics.py` |
| §eval — guard cells / success cells / the "ECE worst where most accurate" named diagnostic, **stated in the script's own output** | `eval.py`, printed verbatim |

---

## 2. Verification actually run (real results)

Hardware: **NVIDIA GeForce RTX 2080 Ti**, torch 2.13.0+cu126, Windows.
This is the only GPU on this machine; the A100 paper run is a separate, later job.
Windows forces `N_WORKERS=0` (spawn would reload the CSVs per worker), so all
throughput numbers below are worst-case synchronous loading.

### 2.1 `verify_zero_init.py` — **44/44 PASS**

```
=== 1. sigma-conditioning columns are zero-init'd (S{head}) ===
  [PASS] step-0 'kappa'/'logits'/'e_mean'/'e_log_std'/'e_logit'
         unchanged when sigma is zeroed          max|delta| = 0.000e+00  (all 5)
  [PASS] ... and unchanged for an ARBITRARY sigma max|delta| = 0.000e+00  (all 5)
  [PASS] the sigma embedding is non-degenerate   shape=(3,16) max col std=1.014

=== 2. higher-moment feature columns are zero-init'd ===
  [PASS] all 5 outputs unchanged by the 4 new moment features  max|delta| = 0.000e+00
  [PASS] corrected vs uncorrected cosines differ at sigma>0    max|delta| = 6.965e-01
  [PASS] ...and coincide at sigma=0 (s2=0 => A_hat=M_obs)      max|delta| = 0.000e+00
  [PASS] moment axis distinct from the raw-PCA axis   median angle = 3.13 deg

=== 3. mu_k == u exactly, for ANY gate value (S{fallback}) ===
  [PASS] mu_k == u for every component at init          max|delta| = 0.000e+00
  [PASS] mu_k == u after randomizing the gate N(0,5)    max|delta| = 0.000e+00
         (gate driven to the full range [0.000, 1.000] and mu still does not move)

=== 4. identical to a v22-equivalent head (kappa/logit/energy) ===
  [PASS] v22-equivalent conditioning dim is v22's D_AUG (384+5) = 389
  [PASS] [float32] kappa/logits/energy(3)   max|delta| <= 6.0e-08  (tol 1e-05)
  [PASS] [float64] kappa/logits/energy(3)   max|delta| <= 1.1e-16  (tol 1e-11)

=== 5. at init the direction readout IS the classical PCA baseline ===
  [PASS] in-graph u == independent numpy PCA axis (incl. SIGN)  max|1-u.u_np| = 4.6e-08
  [PASS] mode readout == u at init                              max|delta| = 0.000e+00
  [PASS] u is Oh-equivariant incl. the sign fix                 max|delta| = 6.0e-08

=== 6. the zero-init'd slices are NOT dead ===
  [PASS] direction sigma cols  max|grad| = 1.5e-03
  [PASS] direction moment cols max|grad| = 1.5e-03
  [PASS] energy sigma cols     max|grad| = 3.4e-03
  [PASS] energy moment cols    max|grad| = 3.9e-03
  [PASS] the zeroed mu ROWS    max|grad| = 1.9e-03
  [PASS] the sigma gate receives NO gradient at init (exact)    max|grad| = 0.000e+00
  [PASS] ...and IS unlocked once delta != 0 (i.e. from step 1)  max|grad| = 3.8e-04
  [PASS] loss and all grads finite
  [PASS] sigma reaches kappa once its weights are non-zero      max|dkappa| = 1.9e-02

=== 7. the kappa regularizer at lambda=0 is a no-op ===
  [PASS] the two comparison models are identically initialized  max|delta param| = 0
  [PASS] loss identical with the lambda=0 term present          |delta| = 0.000e+00
  [PASS] gradients unchanged, at or below the backward's noise floor
         max|delta grad| = 2.328e-10   control(identical loss, twice) = 3.492e-10
  [PASS] the penalty itself is live and positive   mean penalty = 0.7100

ALL ZERO-INIT CHECKS PASSED
```

**Three things this harness caught that are worth knowing** (all are properties of
the design, not defects, but two of them made my first draft of the *test* wrong):

1. **The σ-gate receives exactly zero gradient at step 0, and that is correct.**
   With `δ_k ≡ 0`, `μ = normalize(u + g·0) = u`, so `∂μ/∂g = 0` identically. The gate
   is not permanently dead: the mu rows *do* receive gradient, so `δ_k ≠ 0` from
   step 1 and the gate is unlocked then. The harness asserts both halves. There is a
   one-step activation delay on the gate; it is harmless, and the alternative
   (initializing `δ` non-zero) would give up the §fallback guarantee.
2. **"Bit-identical to plain v22" is attainable only for same-shape comparisons.**
   Checks 1–3 are exactly `0.000e+00` because they compare the *same* 409-column
   matmul with the new inputs zeroed — which is what the spec asks for
   ("wherever the new inputs are zeroed"). Check 4 compares a 409-column matmul to a
   389-column one; a dot product with 20 extra exactly-zero terms is mathematically
   identical but *not* bitwise identical in floating point, because the accumulation
   blocking differs. Reported honestly as float32-at-the-rounding-floor plus a
   float64 run 12 orders tighter, which proves the residual is rounding.
3. **v22's `FusionMLP` contains `nn.Dropout(0.1)`** (carried forward unchanged), so
   any train()-mode forward is stochastic by ~1e-2. Every bitwise comparison must be
   made in `eval()` mode; check 7 says so in a comment because my first version of
   that test measured dropout noise and "failed".

### 2.2 `validate_math.py` — **5/5 PASS**

Re-validates the ported math against **v23's own** implementations, so a porting
error cannot hide behind "it was validated in `a100_final`".

```
TEST A  normalizer      new median ratio 1.000 flat from sigma/Rg = 0 to 0.80
                        (0.993 at 0.80, 1.036 at 2.40)
                        legacy max_extent normalizer: 2.74x the true Rg already at
                        sigma=0, rising to 7.79x  -> 2.8x drift
TEST B1 correction bias  bias reduction 14.6x / 61.8x / 72.3x / 114.3x / 59.1x
                        at sigma/Rg = 0.15 / 0.38 / 0.75 / 1.51 / 3.01
TEST B2 single-realization  corrected 1.05 / 2.84 / 7.67 / 29.81 / 57.88 deg
                        uncorrected  1.05 / 2.68 / 5.67 / 15.35 / 40.40 deg
                        plain PCA    2.09 / 2.79 / 4.98 / 15.45 / 39.59 deg
                        -> the 4th-moment axis carries >2 deg of information
                           independent of PCA at every level, AND the UNCORRECTED
                           estimator wins from sigma/Rg ~ 0.8 upward.  This
                           reproduces a100_final's crossover finding and is exactly
                           why the spec says "Do not pick one" -- v23 feeds both.
TEST C  3rd moment      bias/s.e. = -0.56  (|.| < 3 => consistent with zero), so
                        the skewness sign fix needs no blur correction
TEST D  exported feats   finite, cosines in [0,1], ratios in [-1,1], corrected ==
                        uncorrected EXACTLY at sigma=0, mean gap 0.441 at 100nm
```

(`a100_final`'s own `validate_math.py` had two bugs, found while porting and fixed
here: its TEST A printed `sigma/R_g` against a hard-coded `70.0` while the cloud's
true `R_g` was 125.1 Å, overstating that axis by ~1.79x — so the *labels* in its
README table are shifted, though every ratio is correct; and its `old_dev`
accumulator was dead code with a tautological `ro if sigma == 0 else ro` expression
that was always 0.0 and never read. Neither affects any published number.)

### 2.3 `check_fast.py` — **all checks PASS**, 300 real steps in 39 s

```
[LOAD] 3,626 valid tracks from 400,000 rows of data/siimpl_rot/siimpl_train_v2.csv
  [PASS] real tracks loaded
  [PASS] phys standardization stats finite
  [PASS] batch shape correct  (64, 16211)
  [PASS] batch finite

  step   0  total=8.187  nll=6.744  mean_kappa=0.71  |g|= 7.64
  step  50  total=4.829  nll=3.708  mean_kappa=6.89  |g|=25.22
  step 100  total=4.014  nll=2.912  mean_kappa=4.69  |g|=15.89
  step 150  total=4.081  nll=2.963  mean_kappa=5.15  |g|=12.89
  step 200  total=3.574  nll=2.584  mean_kappa=6.60  |g|=10.91
  step 250  total=3.704  nll=2.587  mean_kappa=7.27  |g|=14.97
  step 299  total=3.036  nll=2.030  mean_kappa=5.54  |g|=15.51

  sigma actually seen: 29.9% at exactly 0 (P_ZERO=0.30), nonzero median 3.1nm, max 100nm
  [PASS] Stage-1 data pipeline actually delivers BLUR (the v22 bug this fixes)  70.1%
  [PASS] no NaN/Inf losses                     0 non-finite steps
  [PASS] ran the requested number of steps     300 steps in 39s (7.70 it/s, batch 64)
  [PASS] training loss DECREASES               5.126 -> 3.579  (-30.2%)
  [PASS] posterior NLL component decreases     3.962 -> 2.508
  [PASS] the kappa penalty stays finite        0.71 -> 5.54   (lambda = 0.002 active)

  [PASS] every parameter/buffer round-trips bit-identically   0 mismatched tensors
  [PASS] [cpu] reloaded mu / kappa / logits / u bit-identical  max|delta| = 0.000e+00
         (gpu, for context: round-trip 2.861e-06 vs same-model-twice control
          1.907e-06 -- CUDA atomicAdd non-determinism, not a checkpoint defect)
  [PASS] checkpoint stores no pickled module object

  eval path, 2 tiers x 2 sigmas, real PCA baseline: all cells finite
```

The `lambda=0.002` here is deliberately non-zero, to show the regularizer can be
switched on without destabilizing anything; the λ=0 swap-in requirement is covered
by `verify_zero_init.py` check 7.

**Do not quote the loss or error numbers above.** They are 300 steps on 3,626
tracks. They are evidence the path executes, not evidence about performance — the
same discipline `a100_final`'s README applied to its own smoke test.

### 2.4 End-to-end two-stage smoke of `train.py` and `eval.py`

`SMOKE=1` ran both stages to completion (2 epochs × 8 steps each, 800 tracks):
Stage 1 → freeze backbone (1,005,198 params) → fresh Stage-2 posterior (489,063
trainable) → calibration-aware selection correctly **kept epoch 0** (ECE 14.09%)
over epoch 1 (ECE 14.27%) because epoch 1's NLL improvement (9.8555 → 9.8497) was
inside `ECE_NLL_TOL=0.02`. `eval.py` then produced the full 27-cell grid, printed
every pre-registered gate with a verdict, and correctly returned
`guard gate: VIOLATED` for a 16-step model. Model: **1,354,223 params** total
(1,005,198 backbone). Smoke artifacts were deleted so nobody mistakes them for
results.

### 2.5 `LAMBDA_KAPPA=auto`

```
[LAMBDA] mean direction NLL at init = 2.2078, mean kappa penalty = 0.7033
[LAMBDA] lambda = 0.010 * 2.2078 / 0.7033 = 0.031393  (1.0% of the direction NLL at init)
```

---

## 3. Deviations and judgment calls

**No spec formula was changed.** Everything the spec pins down with an exact formula
is implemented exactly as written. Below is every place the spec left a choice open,
plus one place where following v22 literally would have been a correctness bug.

### 3.1 A correctness fix: the Pool A / Pool B rule for the v2 corpus

**This is the one place I deliberately did not copy v22, and it matters.**

v22 defines Pool A as `ion_number < POOL_B_ION_START = 300_000` and trains Stage 2
on it so the posterior's implicit prior stays uniform-on-S². Applied to
`siimpl_train_v2.csv` that rule is **wrong**. `src/scripts/merge_tiered_data.py`
renumbers each source file with a running `offset = previous local_max + 1`, and the
first source (`siimpl_train.csv`, offset 0) keeps its original ids. Verified:

* `max(ion_number)` in `siimpl_train.csv` = **347,834**
* `max(ion_number)` in `siimpl_train_v2.csv` = **1,062,059**
* `src/scripts/generate_data_tiered.py` samples directions with `sample_iso()`, so
  the three `_extra_{low,mid,high}` tier files are **isotropic**, i.e. Pool-A-like.

So v22's rule would label all ~714k isotropic v2 tier-extra tracks as
channeling-enriched Pool B, leaving Stage 2 with roughly a quarter of its intended
data and — worse — a Stage-2 training set that is 0% channeling-enriched only by
accident. v23 uses:

```
Pool B  =  300_000 <= ion_number <= 347_834      (config.POOL_B_LO / POOL_B_HI)
Pool A  =  everything else
```

`data_pipeline.pool_a_mask` asserts the resulting Pool B count against
`gen_meta.json`'s recorded **47,835** (2% tolerance) and prints a loud warning if it
drifts, so a corpus regeneration cannot break this silently. The check is skipped
when the CSV is read with an `nrows` cap.

### 3.2 Epoch definition: `SUB_EPOCH_STEPS = 1250`

The spec pins the *curriculum* (`AUX_SIGN_WARMUP=30`, `MAX_EPOCHS=300`,
`PATIENCE=40`, cosine LR over `MAX_EPOCHS`) and separately pins the *corpus*
(1,009,384 tracks, up from v22's 320k cap). Those two are not jointly satisfiable
under v22's "epoch = one full pass" definition:

* v22: ~304k train split / batch 256 = **1,187 steps/epoch**; 300 epochs = 356k steps;
  warmup = 30/300 = **10%** of training.
* v2 corpus: 3,940 steps/epoch. 300 epochs = 1.18M steps (≈3.3x the budget). Cutting
  `MAX_EPOCHS` to ~90 to compensate would stretch the warmup to **33%** of training.

So an "epoch" for scheduler purposes is a fixed `SUB_EPOCH_STEPS = 1250` steps
(≈ v22's own 1,187), which preserves both the absolute steps per epoch and the
30/300 curriculum proportion. The **sampling** is unaffected: the loader still
consumes full random permutations of the entire pool in order, so every track is
visited; it just takes ~3.15 scheduler-epochs to complete one full pass, and
`train.py` prints exactly that. Set `SUB_EPOCH_STEPS=0` to revert to
epoch = full pass.

### 3.3 `AUX_SIGN_WARMUP` — logic unchanged, exposure instrumented

§loss says keep `AUX_SIGN_WARMUP=30` as-is, and that what *should* change is to
"verify empirically" that enough low-σ, head/tail-informative examples have been
seen by epoch 30 rather than assume it transfers from the clean-data-only regime.
So the timing logic is byte-for-byte v22's (`epoch < 30` → `axis_aware_vmf_nll`,
else `vmf_nll`), and `train.py` prints at the switch:

```
[S{loss} CHECK] examples seen / at sigma = 0 exactly / at sigma < 3 nm /
                v22's clean-only equivalent / low-sigma exposure vs v22
```
plus a warning if fewer σ=0 examples have been seen than there are training tracks.
**Nothing is auto-adjusted.** At the default settings the arithmetic is
30 × 1250 × 256 = 9.6M examples, of which ~30% (2.88M) are at exactly σ=0 — about
3x the 1,009,384-track pool, so each track is seen unblurred ~3 times before the
switch, against v22's ~30. That is the number to look at in the real log.

### 3.4 The eigenvalue "ratio" normalization

§physfeat asks for "the corresponding top eigenvalue ratios as a rough confidence
proxy" without fixing the normalization. `Â` can be **indefinite** (the correction
subtracts a large tensor), so `λ₁/Σλ` is ill-defined there. v23 uses

```
ratio = lambda_max / (sum_i |lambda_i| + eps)
```

which is bounded in `[1/3, 1]` for the PSD `M_obs` and in `[-1, 1]` for `Â`, and is
well-defined in both cases. Validated in `validate_math.py` TEST D.

### 3.5 The gate's input: the embedded σ features only

§fallback writes the gate as `g(σ, z')` in the formula but then specifies the
implementation as "a single sigmoid-activated scalar per component, taking **the
embedded σ features** as input". Those two readings differ (the first would also
take the full conditioning vector). I implemented the narrower, explicitly-specified
one: `nn.Linear(16, K)` on the σ embedding, i.e. `g = g(σ)` only. Flagging it
because it is a real ambiguity in the spec. If the wider reading is wanted it is a
one-line change (`nn.Linear(D_AUG, K)` on `z'`); the zero-init guarantee holds
either way, since it does not depend on `g` at all.

### 3.6 The sign convention for `u`

§fallback requires that v23 at initialization is "mathematically identical to plain
PCA". An axis has no sign, so this only holds *including head/tail* if `u` uses the
same sign convention as the **reported PCA baseline** —
`a100_final/eval.py::pca_axis_and_signed` / `eval_r7_full_sweep.py:211`, which flips
when the projection skewness is **positive**. Note that `a100_final/model.py`'s
internal `_skew_sign_fix` flips on the **opposite** sign. v23 uses the *baseline's*
convention, so the §fallback guarantee covers the head/tail decision too;
`verify_zero_init.py` check 5 asserts agreement with an independent numpy
implementation including the sign (`max|1 − u·u_numpy| = 4.6e-08`).

### 3.7 `u` is computed in-graph, not carried in the phys block

The cheaper option would be to compute `u` once in the data pipeline and ship it as
3 extra phys columns. That would be **wrong**: `apply_oh_augmentation` rotates only
the coordinate block (the phys block is Oh-invariant by construction and must not be
rotated), so a carried `u` would desynchronize from the rotated cloud. `u` is
therefore recomputed in torch from the coordinates the backbone sees, via
`torch.linalg.eigh` on the 3×3 masked covariance, and detached (zero learned
parameters, per the spec). Both the axis and the sign fix are exactly O(3)-equivariant
— for `R ∈ O(3)`, `Σ((Rp)·(Ru))³ = Σ(p·u)³` — so this is exact, not an
approximation; `verify_zero_init.py` check 5 asserts it numerically.

### 3.8 The higher-moment tensor: numpy in the pipeline, `eigh` not power iteration

§physfeat classes the higher-moment quantities as *scalar features* appended to the
conditioning vector (unlike `a100_final`, where they were *candidate vectors* inside
the V-bottleneck and therefore had to be computed in-network). v23 computes them
per-track in numpy in `data_pipeline.moment_features`, which is where the other
physics features live. Two consequences:

* **`numpy.linalg.eigh` replaces the shifted power iteration.** `a100_final` had to
  add a `‖T‖_F·I` shift because power iteration on an indefinite `Â` would converge
  to the algebraically *most negative* eigenvector. `eigh` returns all eigenvalues
  sorted, so index `-1` **is** the algebraically largest — the same target, computed
  exactly. The shift is unnecessary, not omitted.
* **Coordinates and units.** The tensor is computed on the **normalized** cloud with
  `s² = σ_n²(n−1)/n`, matching `a100_final` exactly. (Feeding physical coordinates
  with `σ_n`, or normalized coordinates with `σ_A`, would be wrong by `scale²`.)
  It is computed *pre-truncation*, like the other phys features; with
  `MAX_POINTS=600` nothing is truncated anyway (longest train track 587, eval 545),
  so the distinction is moot in practice but preserved for correctness.

### 3.9 Row-level (not whole-layer) zeroing of the mu outputs

§fallback explicitly permits either. v23 zeros only the 3 mu rows per component of
the shared final `nn.Linear`, leaving κ/logit rows at v22's tuned initialization
(whole-layer zeroing would also force `κ = softplus(0)+ε ≈ 0.70` and uniform
logits — reasonable, per the spec, but a gratuitous change to v22's head).

### 3.10 `MAX_POINTS = 600`, `K_MAX = 24`

Inherited from `a100_final` along with the graph and pipeline the spec tells me to
port (§graph pins K_MIN/K_MAX/H0; n_max is implied by the same port). v22 used
`max_points="auto"` (95th percentile ≈ 381), which truncated real tracks. 600
truncates nothing. **Standing hazard, carried over from `a100_final` README §8b:**
the longest `train_v2` track is 587 points, a margin of only 13.
`load_raw_tracks_v23` therefore guards on the *observed* longest track and raises
`SystemExit` rather than silently truncating (`a100_final`'s assert could not catch
this, because `load_raw_tracks` caps its returned `n_max` at `max_points`), and warns
when the longest track is within 5% of the cap.

### 3.11 Physics-feature standardization statistics

v22 computed `phys_mean`/`phys_std` once from the static, **unblurred**
`preprocess_egnn` output. v23's phys block is blur-**dependent** (the deconvolved
normalizer and both higher-moment cosines/ratios all move with σ), so statistics
taken at σ=0 would be badly off-centre for ~70% of the training data.
`train.compute_phys_stats` samples 40 batches from the *same* dynamic-blur pipeline
training uses. The stats are stored in the checkpoint and reused verbatim at eval.

### 3.12 A fixed validation blur draw

Validation NLL **and** validation ECE are the checkpoint-selection criteria
(§calibration 2). Re-drawing blur each epoch would inject per-epoch noise into
exactly the quantity being compared across epochs, so `build_fixed_val_set` draws
one σ per validation track once (seed 12345) from the same continuous distribution
and reuses it. It spans the full σ range, so it is not a zero-blur-only metric.

### 3.13 Checkpoints store state dicts, never a pickled module

v22 stored `'flow_module': flow` — the live module object — which makes a checkpoint
unloadable as soon as a referenced class moves or changes. That failure mode is in
this project's history, so v23 stores state dicts plus a plain config dict and
rebuilds through `build_v23()`. `check_fast.py` asserts the round-trip is bitwise
and that no value in the checkpoint has a `state_dict` attribute.

### 3.14 The explicit guard margins

§eval says zero-blur performance "must not regress ... by more than a small,
explicitly-agreed margin" without fixing it. v23's explicit values, printed by
`eval.py`: **+1.0°** on axis error and **−1.0 pp** on head-tail, against v22's pooled
6.8° / 97.0%. Change via `GUARD_AXIS_MARGIN_DEG` / `GUARD_HEADTAIL_MARGIN_PCT`.

### 3.15 The calibration-aware selection bookkeeping

§calibration 2 gives the criterion ("minimizing validation ECE among those within
some small tolerance of the best validation NLL") but not the bookkeeping for a
running best. `CalibrationAwareSelector`: an epoch qualifies if
`val_nll <= best_nll + ECE_NLL_TOL`; among qualifying epochs the lowest `val_ece`
wins; and if a later epoch improves `best_nll` by *more* than the tolerance, every
previously-qualifying epoch has left the band, so the ECE competition is **reset**
rather than left holding a stale winner. Every epoch's `(nll, ece)` pair is written
to `training_log.csv`, so the selection is auditable and recomputable offline.
Both a best-NLL checkpoint and the selected checkpoint are saved separately.

### 3.16 Stage 2's auxiliary loss stays `axis_aware` — confirmed, not changed

§loss flags v22's Stage-2 `axis_aware_vmf_nll` (lines 1222, 1261) for confirmation.
Carried forward unchanged, and here is the reason it is safe: the Stage-2 auxiliary
head sits on a **frozen** backbone, so its gradients cannot reach the backbone at
all. It is a throwaway regularizer on the auxiliary head's own parameters, and its
sign-awareness cannot affect the deployed posterior's head/tail capability either
way. (The deployed posterior's own loss is the fully **signed** vMF-mixture NLL from
the first step, in both stages, exactly as v22.)

---

## 4. Not done, and why

* **The reference sweep for the success gate is not produced here.** §eval requires
  running v22's `ROT_FINAL` checkpoint through this same 9σ × 3-tier sweep *before*
  v23 training, so the success gate has two real measured baselines instead of one
  curve and one data point. **That checkpoint is not on this machine** — `results/`
  holds only `ABL_R0`, `ABL_R1`, `DIRHEAD_SMOKE`, `ROT_SMOKE`, `ROT_VAL`; ROT_FINAL
  was a Colab/Drive run. `eval.py` handles its absence loudly (prints
  `NO REFERENCE SWEEP SUPPLIED`, states the gate cannot be evaluated, and reports
  v23-vs-PCA edges only) and accepts it via `REF_SWEEP_CSV=` once available.
  **This is a required pre-training step, not an optional one.**
* **The CI-excludes-zero half of the success gate** needs the reference's *per-track*
  errors for a paired bootstrap, not its per-cell medians. With medians only,
  `eval.py` can certify the *sign* of each improvement and says so explicitly in its
  own output. If the reference sweep is regenerated, emit per-track errors.
* **The κ-premise check against `a100_final`'s real "small" checkpoint** (§calibration:
  "pull the a100_final 'small' checkpoint and inspect mean κ by energy/σ cell") could
  not be run: only 150-step smoke checkpoints exist locally
  (`a100_final/results_smoke{,2}`, `results_v2check`), whose κ tells you nothing. The
  *tooling* is in place instead: `eval.py` reports `mean_kappa` and
  `mean_top_weight` per cell and prints a MECHANISM table that discriminates the two
  hypotheses the spec names — runaway κ (which λ targets) vs mixture-weight collapse
  onto a single overconfident component (which λ cannot fix).
* Out of scope by instruction, and absent: the low-energy floor, the
  matched-filter/channeling-template idea, any scaling ladder (v23 is one fixed
  hidden=96 model), and a Colab notebook.

## 5. Known caveats

* **AMP/bf16 is not available**, same blocker as `a100_final`: `src/models/egnn.py`'s
  `EGNNLayer` allocates its aggregation buffer in fp32 and `scatter_add_`s autocast
  bf16 messages into it, raising `RuntimeError: scatter(): Expected self.dtype to be
  equal to src.dtype`. Fixing it means editing a file outside this folder, which is
  not permitted. Expect ~1.5–2x throughput on A100 if that one-line upstream fix
  ever lands.
* **GPU forward passes are not bitwise reproducible.** The backbone's `scatter_add_`
  uses `atomicAdd`, so the same model forwarded twice on CUDA differs at ~1e-6
  absolute on κ for a trained model. All bitwise claims in this folder are made on
  CPU; `check_fast.py` prints the GPU same-model-twice control alongside the GPU
  round-trip delta so the two are never confused.
* **Windows forces `N_WORKERS=0`** (spawn would reload the CSVs per worker). The
  A100 cluster is Linux, where fork + `TrackPool` shares the pool copy-on-write.
  Watch the `loader-wait` percentage that `train.py` prints each epoch; raise
  `N_WORKERS` if it does not fall below a few percent.
* **`TARGET`/budget sizing is not guessed.** §staged-build item 3 says to scope the
  budget once Stage 1's real throughput is measured. `train.py` prints `it/s` and
  `loader-wait` per epoch and honours `TIME_BUDGET_HOURS`; size `MAX_EPOCHS` from the
  first few real epochs, not from this README.
* **The v2 corpus's ion ids are not dense** (1,009,384 unique over 0..1,062,059).
  Harmless — `load_raw_tracks_v23` segments on `np.diff(ion_ids) != 0` — but it is
  the property that made the original merge-collision bug possible, and it is why
  §3.1's Pool split is expressed as an id *range* rather than a count.

---

## 6. Running it

```bash
# 0. verification (do this first; both must pass)
python -m smearing_resolution.architecture_experiments.v23_final.verify_zero_init
python -m smearing_resolution.architecture_experiments.v23_final.validate_math
python -m smearing_resolution.architecture_experiments.v23_final.check_fast

# 1. REQUIRED before the real run: the v22 ROT_FINAL reference sweep (eval-only)
#    -> produces the CSV that the success gate is measured against

# 2. full two-stage run, kappa regularizer active, on the v2 corpus
LAMBDA_KAPPA=auto SEED=42 \
  RESULTS_DIR=.../v23_final/results_s42 \
  python -m smearing_resolution.architecture_experiments.v23_final.train

# 2b. the near-free calibration addition: 2-3 seeds
LAMBDA_KAPPA=auto SEED=43 RESULTS_DIR=.../results_s43 python -m ...v23_final.train

# 2c. the spec asks for pre- and post-lambda coverage curves; the ablation is
#     the same command with LAMBDA_KAPPA=0.0
LAMBDA_KAPPA=0.0 SEED=42 RESULTS_DIR=.../results_s42_nolambda python -m ...v23_final.train

# 3. evaluation against the pre-registered grid and gates
CKPT=.../results_s42/best_checkpoint_stage2.pt \
  REF_SWEEP_CSV=.../v22_rot_final_sweep.csv \
  python -m smearing_resolution.architecture_experiments.v23_final.eval

# 3b. seed ensemble
CKPT=.../results_s42/best_checkpoint_stage2.pt,.../results_s43/best_checkpoint_stage2.pt \
  python -m ...v23_final.eval
```

`SMOKE=1` on `train.py` runs a tiny end-to-end two-stage shape/NaN check in ~20 s.
