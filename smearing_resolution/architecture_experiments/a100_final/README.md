# a100_final — consolidated architecture for the A100 paper run

Standalone rebuild of the directional-reconstruction model. Imports only
low-level shared utilities (`load_raw_tracks`, `sample_sigma_continuous`,
`EGNNBackbone`, `ScalarReadout`, `build_edges_precomputed`, `_log_vmf_norm_3d`,
`_sample_vmf_3d`, `gaussian_nll`) from the existing repo; **no file outside this
folder is modified**. Model, data pipeline, training and eval are reimplemented.

Baseline ("the winner") = `integrated/gate_diagnostics/ablation/train_diag_r7_trunc_only.py`,
788k params, 8000 steps, 2080 Ti, batch 112, 450-point truncation, k=16 graph,
K=4 vMF mixture, `max_extent` normalizer, no sigma in the weighting pathway.

```
model.py              DiagModelV2 + vector pool + both posterior heads
data_pipeline.py      smear / deconvolved-Rg normalize / blur-adaptive graph
loader.py             parallel (multi-process) batch preparation
config.py             every knob, env-overridable; CONFIG=small|large ladder
train.py              training loop (epoch-shuffled, EMA, Oh-aug)
eval.py               9-sigma x 3-energy sweep + PCA baseline + coverage ECE
validate_math.py      numerical validation of the blur corrections      [test]
verify_zero_init.py   verification of the zero-init discipline          [test]
test_loader_parity.py parallel loader == serial loader, byte for byte   [test]
bench_loader.py       CPU-prep vs GPU-step throughput benchmark
launch_a100.sh        scaling-ladder launcher (seq / par / slurm)
_data_stats.py        one-off: track-length and neighbour-spacing measurement
```

---

## 1. Sigma fed into the direction-weighting pathway

**What.** `DirectionTrunk` input is now
`[gram_flat | pooled_scalars | sigma_feats]`, where `sigma_feats` are two
**detached** scalars:

* `log(sigma_A + 1 A)` — the physical blur (`sigma0 = 1 A` keeps the log finite
  at `sigma = 0`), and
* `log(sigma_n + 1e-3)` where `sigma_n = sigma_A / scale` — the blur in the
  network's own normalized units.

Both reach the trunk **before** it splits, so they feed **both** `coef_out` (the
candidate-vector weighting) and `kl_out` (kappa/logits). They are also fed to
`sign_head` and `energy_head`.

**Why two.** The spec asked for one; `sigma_n` is added because it is the
physically meaningful quantity — the same 10 nm blur is catastrophic for a
50 A low-energy track and irrelevant for a 600 A high-energy one, and only the
*ratio* expresses that. Zero-init makes the extra feature free at step 0.

**Why this is allowed.** The prior project constraint blocked sigma from the
coefficient pathway on the grounds that "sigma is exogenous and
label-independent". That argument establishes sigma is **safe** to condition on
(it is drawn independently of the label, so conditioning cannot leak the
target); it never established that it must be **withheld**. Correcting that.
The diagnosis is direct: the plain PCA axis is already in the candidate pool
and the model still loses to PCA at high blur — a weighting failure, and the
weighting head could not see the one variable that determines the right
weighting.

**Zero-init.** `trunk.net[0].weight[:, -2:]` is zeroed at construction, as are
the corresponding slices in `sign_head[0]` and `energy_head[0]`.

---

## 2. Normalizer fix — blur-deconvolved radius of gyration

**Old.** `scale = max_extent = |centered|.max()` — an L-infinity **extreme order
statistic** over ~3N coordinates. Under blur it grows like
`sigma * sqrt(2 ln 3N)`: a bias that depends on *both* the blur level and the
point count, and that the network cannot undo because it never sees either
cleanly.

**New.**

```
R2_obs = mean_i |y_i - ybar|^2
s2     = sigma_A^2 * (n-1)/n
scale  = sqrt( max( R2_obs - 3*s2, floor ) )
floor  = max( 1e-4 A^2 , (0.1 * sigma_A)^2 )
```

**Derivation.** With `y_i = x_i + n_i`, `n_i ~ N(0, sigma^2 I_3)` iid, and
`ybar` the sample mean, write `m_i = n_i - nbar`. Then
`Cov(m_i) = sigma^2 (1 - 1/n) I`, so

```
E[R2_obs] = (1/n) sum_i |x_i - xbar|^2 + 3 * sigma^2 * (n-1)/n
          = R2_true + 3*s2
```

**exactly** — no approximation. Hence `R2_obs - 3*s2` is an unbiased estimator
of `R2_true`.

Note the `(n-1)/n` factor: the task statement gave `- 3*sigma^2`, which is the
`n -> inf` limit. The exact factor is used instead because tracks here have a
mean length of 91 points and a low-energy tail down to 3, where the 1/n
correction is not negligible (at n=10 it is a 10% error on the subtracted term).

**Floor.** `max(1e-4, (0.1*sigma_A)^2)`: when blur swamps the track, the
deconvolution is capped at 10% of sigma rather than allowed to collapse to
zero (which would blow up the normalized coordinates). The absolute `1e-4 A^2`
handles a degenerate/collinear cloud at `sigma = 0`.

**Validation** (`validate_math.py` TEST A, elongated 120x25x25 A clouds,
n in [30,300], 300 trials/level; column = estimate / true R_g):

| sigma/R_g | new: med ratio | new: spread | old (max): med ratio | old: spread |
|---|---|---|---|---|
| 0.00 | **1.000** | 0.000 | 2.268 | 0.396 |
| 0.01 | **1.000** | 0.001 | 2.288 | 0.401 |
| 0.04 | **1.000** | 0.002 | 2.308 | 0.419 |
| 0.14 | **1.000** | 0.007 | 2.258 | 0.424 |
| 0.43 | **0.999** | 0.020 | 2.415 | 0.355 |
| 1.43 | **0.989** | 0.091 | 3.286 | 0.434 |
| 4.29 | **0.935** | 0.582 | 7.768 | 0.952 |

The new normalizer is flat at 1.000 across four decades of blur; the old one
inflates by 3.4x over the same range. **PASS.**

`_smear_and_prepare_one_v2` is a *new* function; the original
`_smear_and_prepare_one` is untouched and still importable for A/B.

---

## 3. Blur-corrected higher-moment candidate vector(s)

### 3a. First: the 3rd central moment needs no correction (and why that matters)

The obvious candidate — the third central moment tensor
`S_abc = mean_i(xc_a xc_b xc_c)` or its vector contraction — turns out to be
**exactly blur-unbiased**. With `yc_i = xc_i + m_i`:

```
E[T_abc] = S_abc + s2*( d_bc * mean_i(xc_ia) + d_ac * mean_i(xc_ib) + d_ab * mean_i(xc_ic) )
                 + E[m_a m_b m_c]
```

The last term vanishes (zero-mean Gaussian, odd moment). Every remaining term
carries a factor `mean_i(xc_i) = 0` **identically**, because the cloud is
centered. So `E[T] = S`: no correction exists to make.

Confirmed numerically (`validate_math.py` TEST C, 4000 blur draws at
sigma/R_g ~ 0.4): bias/s.e. = **1.17** (|.| < 3 ⇒ consistent with zero).

Two consequences: (i) a 3rd-moment tensor is *not* the right place to
demonstrate a blur correction, and (ii) the **skewness sign-fix** used
throughout the vector pool is already blur-unbiased and needs no change.

### 3b. What we actually use: the projection-weighted 2nd moment (a 4th moment)

```
M = mean_i ( p_i^2 * outer(xc_i, xc_i) ),     p_i = xc_i . u,  u = raw-PCA axis
```

This is a genuine 4th-moment (kurtosis-type) tensor: it weights each point by
the square of its position along the track, so it is dominated by the
**extremities** — exactly the information a 2nd-moment PCA axis throws away.
Because it is *even* in the noise, it does pick up blur bias, through the
cross terms the task statement warned about.

**Derivation.** With `yc = xc + m`, `p = q + t`, `q = xc.u`, `t = m.u`,
`E[m_a m_b] = s2 d_ab`, expand `E[p^2 yc_a yc_b]` and drop odd-in-`m` terms:

| # of `m` factors | term | expectation |
|---|---|---|
| 0 | `q^2 xc_a xc_b` | `q^2 xc_a xc_b` |
| 2 | `q^2 m_a m_b` | `s2 q^2 d_ab` |
| 2 | `2qt (xc_a m_b + m_a xc_b)` | `2 s2 q (xc_a u_b + u_a xc_b)` |
| 2 | `t^2 xc_a xc_b` | `s2 xc_a xc_b` |
| 4 | `t^2 m_a m_b` | `s2^2 (d_ab + 2 u_a u_b)`  (Isserlis) |

Averaging over `i`, with `C = mean(xc xc^T)`, `<q^2> = mean q_i^2`, and
`b = mean(q_i xc_i) = C u`:

```
E[M_obs] = A + s2 [ <q^2> I + 2(b u^T + u b^T) + C ] + s2^2 [ I + 2 u u^T ]
```

`u` is the top eigenvector of `C`, and blur adds `s2 I` to the covariance —
which shifts every eigenvalue equally and so leaves the **eigenvectors
unchanged**. Hence `b = <q^2> u` and `b u^T + u b^T = 2 <q^2> u u^T`, giving the
estimator actually implemented:

```
A_hat = M_obs - s2 ( <q^2>_hat I + 4 <q^2>_hat u u^T + C_hat )
              - s2^2 ( I + 2 u u^T )

s2          = sigma_n^2 (n-1)/n
<q^2>_hat   = <p^2>_obs - s2
C_hat       = C_obs - s2 I
```

**Physical reading of the bias:** every correction term is proportional to `I`,
`u u^T`, or `C`. The `I` part shifts all eigenvalues equally and is harmless.
The `u u^T` and `C` parts are **not** — they drag the dominant eigenvector
toward the plain PCA axis `u`, i.e. blur silently collapses the new candidate
back onto the old one, destroying exactly the independent information it was
added to provide.

*Assumption:* `u` is treated as fixed in the derivation. It is in fact
estimated from the same blurred cloud, so there is an O(1/n) correlation
between `u` and `m` that is not modelled. Numerically this is far below the
terms retained (TEST B1 below).

**Implementation.** `A_hat` is symmetrized, then shifted by
`||A_hat||_F * I` before power iteration — the correction can leave `A_hat`
indefinite, and power iteration would otherwise converge to the
algebraically-*most-negative* eigenvector. A multiple of `I` shifts all
eigenvalues equally and leaves eigenvectors untouched. Sign-fixed by the
(blur-unbiased, §3a) third moment of the projections.

**Validation — TEST B1 (the derivation itself).** Fixed cloud, `u` held fixed,
estimator averaged over 3000 blur draws, compared to the clean tensor. This
isolates *bias* from variance:

| sigma | sigma/R_g | \|bias(uncorrected)\|/\|A\| | \|bias(corrected)\|/\|A\| | bias reduction |
|---|---|---|---|---|
| 20 | 0.10 | 0.0022 | 0.0008 | 2.6x |
| 50 | 0.26 | 0.0157 | 0.0012 | **13x** |
| 100 | 0.52 | 0.0727 | 0.0021 | **35x** |
| 200 | 1.04 | 0.4314 | 0.0133 | **32x** |
| 400 | 2.08 | 3.6862 | 0.0188 | **196x** |

The uncorrected estimator's bias reaches 369% of the tensor norm; the corrected
one stays under 2%. The closed form is **confirmed. PASS.**

**Validation — TEST B2 (single-realization axis, honest result).** Bent
("banana") clouds whose extremities sit off the principal axis, `u`
re-estimated from blurred data as in the real pipeline, median angle to the
clean 4th-moment answer:

| sigma | sigma/R_g | corrected | uncorrected | PCA axis |
|---|---|---|---|---|
| 20 | 0.12 | 0.91 | 0.89 | 12.28 |
| 50 | 0.30 | 2.11 | 2.20 | 12.73 |
| 100 | 0.59 | 4.44 | 4.36 | 12.89 |
| 200 | 1.19 | 11.14 | 11.17 | 17.38 |
| 400 | 2.37 | 40.65 | **29.55** | 31.60 |

The "PCA axis" column shows the new candidate carries ~12 deg of genuinely
independent information — it is a new axis estimate, not a rescaled PCA axis,
which is the whole point (a new scalar feature could never do this: the head's
output is `mu = normalize(sum_a coef[a] V[a])`, so only a new **vector** adds a
new direction).

But **the correction does not dominate on a single realization.** It removes
bias while amplifying variance (it subtracts a large tensor), and the
uncorrected estimator's bias happens to be a *shrinkage toward the PCA axis* —
which at extreme `sigma/R_g` is itself closer to the truth than the noisy
corrected estimate. At `sigma/R_g = 2.37` the uncorrected axis wins by 11 deg.

**Design consequence.** Rather than hard-code a shrinkage factor, **both** axes
are shipped as pool candidates (`KURT_CORR_IDX=12`, `KURT_RAW_IDX=13`,
`N_VECTORS = 14`) and the **sigma-conditioned** coefficient head from change 1
arbitrates between them per blur level. This is precisely the bias/variance
trade that the sigma feature exists to resolve, and it costs one extra
eigen-solve on an already-computed tensor. At `sigma = 0` the two coincide
exactly (`s2 = 0` ⇒ `A_hat = M_obs`), verified.

**Zero-init.** Both new vectors' Gram-matrix columns into the trunk (and into
`sign_head`/`energy_head`) and their coefficient rows out of `coef_out` are
zeroed at construction. The vectors themselves are *not* suppressed — they are
computed normally and participate in the pool; it is their *weighting* that
starts at exactly zero, so step-0 behaviour is identical to a 12-vector model
while gradients still flow into both paths from the first backward pass
(verified below).

---

## 4. Richer posterior family

**Shipped and default: K=4 -> K=10 vMF mixture** (`N_DIR_COMP=10`). Cheap,
low-risk, and directly targets the leading hypothesis for the ~20% coverage
ECE — that K=4 is too rigid to represent band-like/great-circle uncertainty
when a sparse cloud constrains only one shape axis.

**Also implemented: grid posterior** (`POSTERIOR_TYPE=grid`),
Implicit-PDF-style (Murphy et al. 2021). A subdivided icosahedron
(`GRID_SUBDIV=3` -> **642** points; 2 -> 162, 4 -> 2562) is scored by a learned
scalar and softmax-normalized, with Riemann normalization
`log p(d) = s(d) - logsumexp_g s(g) - log(4*pi/G)`.

Equivariance is preserved despite the grid being lab-fixed: the score is
`f([d.V_1 ... d.V_14], context)` — a function of rotation-**invariant** dot
products only, so `score(Rd | RV) = score(d | V)`. The grid is only the
quadrature, and 642 near-uniform points make the normalizer's discretization
error negligible.

**STATUS — read before using.** The grid path is **implemented, shape-checked,
zero-init-verified, and confirmed to produce finite loss and finite gradients**
(`verify_zero_init.py` runs the full check suite against it and it passes).
It has **not** been trained, tuned, or evaluated end-to-end. The A100 launch
uses `POSTERIOR_TYPE=vmf_mixture`. Treat the grid option as untested research
code, not as a second production config.

---

## 5. Blur-adaptive graph

Old: fixed k=16 kNN, blur-unaware. New hybrid, in **normalized** coordinates:

```
radius r = sqrt(H0^2 + sigma_n^2)
each point connects to  max( #neighbours within r , K_MIN )  neighbours,
capped at K_MAX
```

Because the kNN query returns neighbours **distance-sorted**, the radius set is
exactly a prefix of the kNN list — so this needs no second tree query and costs
essentially nothing over the old path.

**H0 = 0.05** (normalized units). Chosen from the data, not guessed:
`_data_stats.py` measures the median nearest-neighbour spacing in
R_g-normalized coordinates of unblurred tracks, by energy tier —
**low 0.100, mid 0.039, high 0.012**. `H0 = 0.05` therefore sits between the
mid and low tiers, which gives the intended behaviour: for dense high-energy
clouds the radius captures ~8 neighbours and the graph is radius-driven; for
sparse mid/low-energy clouds the radius captures ~1 and `K_MIN` takes over.
As `sigma_n` grows the radius grows with it and everything becomes
radius-driven, degrading gracefully to near-kNN once the ball covers most of
the cloud.

**K_MIN = 8** — the anti-starvation floor, so a sparse low-energy track can
never end up with zero or one neighbour.
**K_MAX = 24** — a hard cap. This is a real approximation and is documented as
such: the flat tensor layout is fixed-width, so a point whose radius ball
contains more than 24 neighbours keeps only its 24 nearest. 24 (vs the winner's
16) buys headroom for the radius regime at 1.5x the edge count.

Scope note: the project's own evidence says graph instability is a
**secondary** mechanism (averaging over repeated blur draws showed no benefit,
so the dominant mid/high-energy failure is bias, not variance). This change is
included because it is cheap and well-motivated, not because it is expected to
carry the result.

---

## 6. Scaling ladder and capacity

| | HIDDEN_DIM | N_LAYERS | BATCH_SIZE | params |
|---|---|---|---|---|
| winner | 112 | 6 (hardcoded) | 112 | 788k |
| `CONFIG=small` | 112 | 6 | 512 | **826k** |
| `CONFIG=large` | 224 | 10 | 384 | ~3.2M |

`small` is a like-for-like capacity control, so the ladder isolates the effect
of the five architectural changes from the effect of size. `large` is ~2x width
and ~1.7x depth. Width dominates cost (edge messages are `O(E * hidden)`), so
this is roughly 4x x 1.7x ~ 7x the winner's compute per sample.

`HIDDEN_DIM` **and** `N_LAYERS` are both env-overridable (the winner hardcoded
`N_LAYERS = 6`), as is everything else — see `config.py`.

**Truncation removed.** The winner used `max_points=450`. Measured actual
maxima: **579** (train) / **545** (eval), mean 91, 99.9th pct 432. `MAX_POINTS`
defaults to **600**, which truncates nothing, and `n_max` is *forced* to
`MAX_POINTS` so train and eval share one padding width (the winner asserted
train/eval `n_max` equality, which only held because both saturated the 450
cap). Padding costs only flat-tensor memory — graph and backbone cost scale
with `total_real` (~91/track), not with `n_max`.

**Batch size.** 512 (small) / 384 (large), up from 112. Rationale: at ~91 real
points/track, batch 512 is ~47k nodes and ~750k edges, which at hidden 224 x 10
layers is a comfortable fraction of an 80GB A100. **This may need tuning** —
on a 40GB A100 start `large` at `BATCH_SIZE=256`. See §6b for how batches are
kept flowing at that size.

**Epoch-based iteration.** The winner drew each batch with
`rng.choice(pool, size=B, replace=False)` *per step* — i.e. sampling with
replacement *across* steps, which never touches ~`exp(-steps*B/pool)` of the
training set (~4.8% at 8000 x 112 over 295k). Replaced with genuine epoch
iteration: one full random permutation per epoch, consumed in batch-sized
chunks, reshuffled on exhaustion. The last partial batch of each epoch is
**dropped** so every step has a uniform batch size (<0.2% of a pass at batch
512, and a different remainder is dropped each epoch since the permutation
changes). Sigma is still drawn fresh at batch-build time, so this is a pure
indexing change with no effect on the augmentation distribution. Epoch and
cumulative "full passes" are logged alongside step count.

---

## 6b. Parallel batch preparation (throughput only)

Batch prep — per track: smear, center, deconvolved-R_g normalize, blur-adaptive
cKDTree graph — ran synchronously and single-threaded in the winner's training
loop. On an A100 that becomes the bottleneck. **Measured** (`bench_loader.py`,
batch 128, mean track length 91, matching the real distribution):

| | ms/batch | tracks/s | speedup |
|---|---|---|---|
| serial (1 thread) | 65.6 | 1951 | 1.00x |
| ThreadPool(4) | 54.0 | 2369 | 1.21x |
| ThreadPool(8) | 54.0 | 2372 | 1.22x |
| ThreadPool(16) | 53.4 | 2399 | **1.23x** |

| GPU step (2080 Ti) | ms/batch | CPU/GPU ratio (serial) |
|---|---|---|
| `small` (h=112, L=6) | 125.6 | 0.52 |
| `large` (h=224, L=10) | 259.3 | 0.25 |

Two conclusions. First, **threads do not solve this** — 1.23x at 16 threads,
because numpy's RNG and the per-track packing hold the GIL. (The first version
of this code used a thread prefetcher on the assumption that cKDTree and numpy
release the GIL; the benchmark refuted that, which is why it was replaced.)
Second, on the 2080 Ti prep still hides behind the GPU (ratio < 1), but an A100
step is several times faster — `small` would be ~28 ms GPU against 66 ms CPU,
a **ratio of ~2.3, i.e. CPU-bound with the GPU idle over half the time.**

So `loader.py` uses real worker **processes** via `torch.utils.data.DataLoader`:

* **`TrackDataset`** is map-style over a global sample index
  `g = (step-1)*B + j`. With a sequential sampler this reproduces the epoch
  permutation exactly *and* delivers batches strictly **in order** (an
  improvement on the thread version, which could reorder within the prefetch
  depth). `N_WORKERS=0` gives the synchronous in-process path.
* **Semantics are unchanged.** Same epoch permutation, same sigma distribution,
  same on-the-fly continuous smearing. The one difference is the RNG *stream*:
  sigma now comes from a per-**sample** generator seeded by `(SEED, g)` rather
  than a per-batch vector draw, so any worker can produce any sample
  independently. Draws stay iid with the identical marginal.
* **Smearing stays on-the-fly** — deliberately not precomputed. A fresh
  continuous sigma every time a track is used gives unbounded augmentation
  diversity; precomputing would force a finite discrete sigma grid and risk
  reintroducing a milder form of the sigma-blindness this rebuild exists to fix.
* **`TrackPool` (fork safety).** `load_raw_tracks` returns a Python list of
  ~293k small arrays. Under fork-based workers that is a memory trap: CoW
  shares pages until CPython writes each object's **refcount**, at which point
  the headers are copied into every worker and RSS grows by ~the pool size per
  worker. `TrackPool` stores one concatenated `(M,3)` array plus offsets — two
  Python objects total — so a worker slicing it takes a view and touches no
  per-track refcount. Memory stays genuinely shared.
* **IPC.** Workers return *compact* `(n,3)`/`(n,k)` arrays
  (`_smear_and_prepare_one_v2(pad=False)`) and `Collate` pads to `n_max`. Mean
  length is 91 against `n_max=600`, so this cuts loader IPC ~6.5x.
* **`N_WORKERS`** defaults to the box's core count capped at 16 — assuming a
  typical A100 node allocates ~8-16 CPU cores per GPU. At ~1950 tracks/s/core,
  batch 512 needs ~262 ms/core against an estimated 30-60 ms A100 step, so
  8-16 workers is the right order. `PREFETCH=4` batches per worker.
* **Windows** uses spawn, not fork, so every worker would re-import the
  training module and reload the CSVs. `resolve_workers()` hard-forces 0 there
  with a warning. The cluster is Linux; this only affects local testing.

**Monitoring.** `train.py` reports `loader-wait N%` on every log line and a
final `[LOADER-WAIT]` summary — the fraction of wall clock the GPU spent
blocked on batches. **If that stays above a few percent on the A100, raise
`N_WORKERS`.** This is the number to watch in the first minutes of the run.

**Verification** (`python test_loader_parity.py`) — the claim "throughput only,
never semantics" is tested, not asserted:

```
=== loader parity ===
  [PASS] serial loader produced all steps  75 batches
  [PASS] batches arrive strictly in order
  [PASS] num_workers=2 produced the same number of batches  75 vs 75
  [PASS] parallel batches are BYTE-IDENTICAL to serial (inputs)   max|dx|=0.000e+00
  [PASS] parallel batches are BYTE-IDENTICAL to serial (targets)  max|dtheta|=0.000e+00
  [PASS] parallel batch ORDER matches serial
=== epoch semantics ===
  [PASS] epoch 0: no track repeated  240 draws
  [PASS] epoch 0: covers the whole pool  240/240
  [PASS] epoch 1: no track repeated  240 draws
  [PASS] epoch 1: covers the whole pool  240/240
  [PASS] consecutive epochs use DIFFERENT permutations
=== sigma marginal ===
  [PASS] p_zero point mass preserved (~0.30)  measured 0.297 over 600 samples
  [PASS] non-zero sigmas are finite and positive

ALL LOADER PARITY CHECKS PASSED
```

Worker processes were genuinely exercised (2 spawn workers, byte-identical
output), so the Dataset/Collate pickling contract is verified cross-platform
even though production workers only run on Linux.

---

**Mixed precision is OFF** (`AMP=0`). bf16 autocast on the backbone raises
`RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype` because
`src/models/egnn.py`'s `EGNNLayer` allocates its aggregation buffer as fp32 and
scatter-adds bf16 messages into it. Fixing that requires editing a file outside
`a100_final/`, which is out of scope. The flag is retained; expect ~1.5-2x
throughput on A100 if that upstream one-line fix ever lands.

---

## 7. Zero-init verification (measured, not assumed)

`python verify_zero_init.py` — asserts that forcing the new inputs off is a
**no-op** at step 0, that the zero-init'd slices are **not dead**, and that the
new vectors are well-formed. Run against **both** posterior types.

```
=== POSTERIOR_TYPE=vmf_mixture ===
  [PASS] step-0 output 'mu' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'kappa' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'logits' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'sign_logit' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'E_pred' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'log_sigma' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] step-0 output 'axis_ref_aligned' unchanged by new inputs  max|delta|=0.000e+00
  [PASS] pool has N_VECTORS entries  14
  [PASS] 'kurt_corrected' is finite & unit-norm  |n-1|max=5.96e-08
  [PASS] 'kurt_raw' is finite & unit-norm  |n-1|max=5.96e-08
  [PASS] 'kurt_corrected' carries info distinct from the raw-PCA axis  median angle to v_ref = 1.75 deg
  [PASS] 'kurt_raw' carries info distinct from the raw-PCA axis  median angle to v_ref = 1.75 deg
  [PASS] corrected vs uncorrected 4th-moment axes differ at sigma>0  max separation = 64.29 deg
  [PASS] ...and coincide at sigma=0 (s2=0 => A=M_obs)  max(1-cos) = 5.96e-08
  [PASS] sigma slice receives gradient (not dead)  max|grad|=1.377e-01
  [PASS] new-vector Gram columns receive gradient  max|grad|=3.080e-02
  [PASS] new-vector coefficient rows receive gradient  max|grad|=1.452e-01
  [PASS] loss and all grads finite
  [PASS] sigma reaches the COEFFICIENT pathway (mu responds)  max|dmu|=1.657e+00
=== POSTERIOR_TYPE=grid ===   (all 16 applicable checks PASS, same values)

ALL ZERO-INIT CHECKS PASSED
```

The direction pathway is **exactly** zero at step 0 (`0.000e+00`, not
"approximately"). Two notes on how that was achieved and one on tolerance:

* The aux heads (`sign_head`, `energy_head`) read the raw invariant feature
  vector directly, so they needed the same zero-init treatment as the trunk —
  the first run of this harness caught non-zero deltas of ~2e-2 in
  `sign_logit`/`E_pred`/`log_sigma` and the heads were fixed. This is exactly
  the kind of thing the "verify, don't assume" instruction exists to catch.
* The sigma=0 coincidence check compares **cosines**, not angles: `arccos` near
  1 amplifies float32 rounding (`1-cos ~ 1e-7` already reads as ~0.03 deg), so
  an angle tolerance below ~0.03 deg measures the metric's noise floor rather
  than the vectors.

### Deviations from the winner that are NOT zero-init-able (by design)

These are deliberate and cannot be matched at step 0:

1. **K=4 -> K=10 mixture.** A different number of mixture components is a
   different output shape; there is no initialization that makes a 10-component
   mixture identical to a 4-component one.
2. **Normalizer.** Dividing by deconvolved `R_g` instead of `max_extent`
   changes the input coordinate distribution by a factor of ~2.3 at sigma=0
   (TEST A) and much more under blur. That change *is the point* of item 2.
3. **Graph.** Different edges are a different graph; no initialization hides it.
4. **Truncation 450 -> 600 and k=16 -> k_max=24.** Different input tensor shapes.

Because of 2-4 this is a **from-scratch** train, not a warm start — which the
task anticipated. The zero-init discipline still does real work: it guarantees
the *new learned components* (sigma conditioning, both 4th-moment candidates)
start with exactly zero influence, so they can only help as they are learned,
and none of them perturbs the optimization at initialization.

---

## 8. Required fast correctness check

Hardware check first — the 2080 Ti was idle (963 MiB / 11264 MiB, 3% util, no
compute processes), so no GPU contention.

```bash
cd "C:/Inverse ML"
nvidia-smi
cd smearing_resolution/architecture_experiments/a100_final
python validate_math.py
python verify_zero_init.py
python test_loader_parity.py
B=128 python bench_loader.py
cd "C:/Inverse ML"
RESUME=0 CONFIG=small TARGET_STEPS=150 BATCH_SIZE=32 MAX_TRAIN_TRACKS=4000 \
  EVAL_N_PER_BIN=50 CHECKPOINT_EVERY_SEC=100000 TIME_BUDGET_HOURS=0.5 \
  RESULTS_DIR=smearing_resolution/architecture_experiments/a100_final/results_smoke2 \
  python smearing_resolution/architecture_experiments/a100_final/train.py
```

**First attempt FAILED and the failure was real** — worth recording, since it is
the bug the check existed to find:

```
File "src/models/egnn.py", line 295, in forward
  agg_m.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)
RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype
```

bf16 autocast on the backbone is incompatible with the shared `EGNNLayer`
(fp32 buffer, bf16 messages). Since `src/models/egnn.py` may not be modified,
`AMP` now defaults to 0 (see §6). Rerun:

```
[CONFIG] {"CONFIG": "small", "HIDDEN_DIM": 112, "N_LAYERS": 6, "BATCH_SIZE": 32,
          "MAX_POINTS": 600, "K_MAX": 24, "K_MIN": 8, "H0": 0.05,
          "POSTERIOR_TYPE": "vmf_mixture", "N_DIR_COMP": 10, "AMP": 0, "SEED": 0}
[DEVICE] cuda (NVIDIA GeForce RTX 2080 Ti)
[DYNAMIC] 292,755 valid tracks, N_max=579, max_points=600
[DYNAMIC]  49,200 valid tracks, N_max=545, max_points=600
[NMAX] forced n_max=600 (observed train=579, eval=545) -> no truncation
[MODEL] 826,161 trainable params (vmf_mixture)
[EPOCH] 125 steps/epoch (4000 tracks / batch 32, remainder dropped) -> 1.20 full passes at 150 steps

[LOADER] 0 worker process(es), prefetch_factor=n/a, 150 steps queued
  step      1 ep  0  loss_ema=61.615  dir=2.451 E=584.574 sign=0.707  |g|=704.62 lr=1.10e-05  (1.91 it/s, loader-wait  3.7%)
  step     50 ep  0  loss_ema=33.383  dir=2.131 E= 13.482 sign=0.671  |g|= 17.48 lr=5.83e-05  (6.69 it/s, loader-wait 10.2%)
  step    100 ep  0  loss_ema=14.163  dir=1.755 E=  3.159 sign=0.676  |g|=  1.34 lr=1.07e-04  (6.92 it/s, loader-wait 10.4%)
  step    150 ep  1  loss_ema= 6.749  dir=1.517 E=  3.403 sign=0.652  |g|=  2.39 lr=1.55e-04  (6.98 it/s, loader-wait 10.5%)

[STOP] step 150 in 0.01h (1.20 full passes over the training pool)
[THROUGHPUT] 6.9845 it/s
[LOADER-WAIT] 10.5% of wall clock spent blocked on batch preparation
[DONE]
```

(Run on Windows, so `N_WORKERS` is forced to 0 — see §6b. The 10.5% loader-wait
at zero workers is the instrumentation doing its job; on the A100 with workers
enabled this should fall to near zero, and is the number to watch.)

This was re-run **after** the parallel-loader change (§6b) to confirm it altered
throughput only. Against the pre-loader run, the trajectory and the final
per-cell numbers are materially unchanged — e.g. final axis error
low@0nm 30.1 vs 30.3, mid@0nm 26.8 vs 27.5, high@0nm 22.4 vs 22.4. The absolute
loss values differ (61.6 vs 37.9 at step 1) purely because the sigma RNG
*stream* changed to per-sample seeding, so step 1 draws a different batch of
blur levels; the energy term dominates early and is sensitive to that draw. The
distribution is identical, as `test_loader_parity.py` verifies directly.

Confirmed:

* **No crashes**, exit code 0, all tensor shapes consistent end to end.
* **No NaNs** — zero `[WARN] non-finite` lines (the loop guards both loss and
  grad-norm and would have printed and skipped); `loss_ema` falls monotonically
  37.9 -> 5.6 and every component loss stays finite.
* **Gradients healthy** — `|g|` settles from 443 to 2.4. The step-1 spike is the
  energy head seeing raw keV targets (the winner behaves identically; this is
  what `GRAD_CLIP=1.0` is for), not an instability.
* **Epoch iteration works** — rollover at step 126 (`ep 0` -> `ep 1`) as
  predicted by `125 steps/epoch`, and the "full passes" counter reports 1.20.
* **Truncation removal works** — `n_max` forced to 600 against observed maxima
  579/545, so nothing is cut.
* **Step-0 behaviour matches the winner** modulo the four unavoidable deviations
  in §7 — established exactly (`0.000e+00`) by `verify_zero_init.py`, which is
  the stronger and more direct test than reading it off a training curve.

`eval.py` was then run end-to-end against that checkpoint to confirm the
evaluation path (model + PCA baseline + coverage ECE + CSV + tier summary):

```bash
CONFIG=small N_PER_BIN=40 N_SAMPLES=100 EVAL_CHUNK=40 \
  CKPT=.../results_smoke/checkpoint_final.pt \
  OUT_CSV=.../results_smoke/sweep_smoke.csv \
  python smearing_resolution/architecture_experiments/a100_final/eval.py
```

```
  high_20_105keV  s=100nm  model ax= 43.53 ht= 52.5%  |  pca ax= 60.50 ht= 55.0%  |  ECE= 8.23%
[SAVE] .../results_smoke/sweep_smoke.csv
[TIER] low_1_5keV      mean axis-err edge over PCA = +1.63 deg   mean ECE = 7.60%
[TIER] mid_5_20keV     mean axis-err edge over PCA = -0.03 deg   mean ECE = 8.90%
[TIER] high_20_105keV  mean axis-err edge over PCA = -1.46 deg   mean ECE = 16.69%
```

All 27 cells populate and the CSV matches the reference schema. **The numbers
above are meaningless** — 150 steps at batch 32 on 4000 tracks, with 40
tracks/cell — and are reported solely as evidence that the path executes.

This is a **correctness check only** — no conclusion about whether the numbers
improve is drawn or implied from 150 steps, and none should be.

---

## 8b. Re-verification against the final v2 corpus

Everything in §8 was first run against the original ~295k/49k corpus. Because
the v2 files are a materially different scale (3.4x the tracks, an 8.4 GB CSV,
and a different `ion_number` range), the whole check was re-run pointed at them
before launch.

**Corpus integrity (independent check, not taken on trust).** The v2 merge had
a latent ID-collision bug that was caught and fixed upstream; duplicate
`ion_number`s would silently fuse two physically distinct tracks into one, so
this was re-verified here from the files themselves:

| | rows | unique ions | ID range | collisions | tiers low/mid/high | max len |
|---|---|---|---|---|---|---|
| `train_v2` | 90,799,740 | 1,009,384 | 0..1,062,059 (**not dense**) | **0** | 24.9 / 30.0 / 45.2 % | **587** |
| `eval_v2` | 9,610,085 | 101,731 | 0..101,730 (dense) | **0** | 24.7 / 30.1 / 45.2 % | 545 |

("Collisions" = ions carrying more than one distinct `energy_keV`, the
signature of two tracks fused under one ID.) Zero in both files, unique-ion
counts match the stated 1,009,384 / 101,731 exactly, and tier fractions match
the 25/30/45 weighting. 1,001,967 train tracks survive the `n_vac>=3` filter,
which is exactly the 981,806 + 20,161 the training run reported.

Two findings worth recording, both caught only because this was measured rather
than assumed:

1. **`train_v2` IDs are NOT dense** — 1,009,384 unique ions spread over the
   range 0..1,062,059. This is expected (it is the artifact of the upstream
   merge offsetting) and is harmless, because `load_raw_tracks` segments on
   `np.diff(ion_ids) != 0` and so keys off *changes*, not contiguity. It is
   worth stating explicitly because it is precisely the property that made the
   original collision bug possible.
2. **Longest `train_v2` track is 587 points, not 579** — the v2 corpus contains
   slightly longer tracks than the original. `MAX_POINTS=600` still truncates
   nothing, but the margin is only **13 points**. Note that the `n_max` assert
   in `train.py` does *not* guard this: `load_raw_tracks` caps its returned
   `n_max` at `max_points`, so a corpus with a 650-point track would yield
   `n_max=600`, pass the assert, and silently truncate. **If the corpus is ever
   regenerated, re-measure the max track length** (`_data_stats.py`) and raise
   `MAX_POINTS` if it approaches 600. For the current, final corpus this is
   verified clear.

**Training run** (200 steps, batch 32, full 981,806-track pool):

```
[DYNAMIC] Loading raw tracks from: data\siimpl_rot\siimpl_train_v2.csv
[DYNAMIC] 1,001,967 valid tracks, N_max=587, max_points=600
[DYNAMIC] Loading raw tracks from: data\siimpl_rot\siimpl_eval_v2.csv
[DYNAMIC]   101,149 valid tracks, N_max=545, max_points=600
[NMAX] forced n_max=600 (observed train=587, eval=545) -> no truncation
[SPLIT] 981806 train / 20161 val
[MODEL] 826,161 trainable params (vmf_mixture)
[EPOCH] 30681 steps/epoch (981806 tracks / batch 32) -> 0.01 full passes at 200 steps
[LOADER] 0 worker process(es), 200 steps queued
  step      1 ep 0  loss_ema=116.375  dir=2.432 E=1132.523 sign=0.691  |g|=1473.96  (1.71 it/s, loader-wait  7.1%)
  step     50 ep 0  loss_ema= 65.960  dir=2.123 E=  24.396 sign=0.633  |g|=  34.56  (6.26 it/s, loader-wait 11.7%)
  step    100 ep 0  loss_ema= 26.089  dir=1.689 E=   4.019 sign=0.637  |g|=   3.05  (6.43 it/s, loader-wait 12.0%)
  step    150 ep 0  loss_ema= 11.051  dir=1.469 E=   3.765 sign=0.740  |g|=   4.23  (6.52 it/s, loader-wait 12.1%)
  step    200 ep 0  loss_ema=  5.507  dir=1.265 E=   3.758 sign=0.490  |g|=   2.56  (6.56 it/s, loader-wait 12.1%)
[STOP] step 200 in 0.01h
[THROUGHPUT] 6.5569 it/s
[LOADER-WAIT] 12.1% of wall clock spent blocked on batch preparation
[DONE]   (exit code 0)
```

Confirmed on v2: **no crashes** (exit 0), **no NaNs** (zero `[WARN] non-finite`
lines; loss falls monotonically 116.4 -> 5.5, every component finite),
gradients healthy (1474 -> 2.56; the step-1 spike is the energy head on raw
keV, handled by `GRAD_CLIP`), and `n_max` forced to 600 against observed 587.
Peak host RAM during the 8.4 GB load stayed well inside the 67 GB box
(~44 GB free throughout) — relevant because `TrackPool`'s concatenated array
is what fork workers will share on the cluster.

**`[LOADER-WAIT]` = 12.1% on the 2080 Ti at `N_WORKERS=0`** (Windows forces
synchronous loading, §6b). That is the requested sanity baseline and it is the
*worst case*: zero parallelism, every batch built serially in the training
loop. It is consistent with the §6b benchmark — serial prep ~66 ms against a
~126 ms `small` GPU step gives a prep:step ratio that a single prefetch depth
cannot fully hide.

**Expectation on the A100, stated explicitly so it can be checked:** the ratio
moves the wrong way (a faster GPU makes prep relatively more expensive), so
with workers *disabled* loader-wait would rise well above 12% — but with
`N_WORKERS=8-16` and `PREFETCH=4` the prep is spread across cores and
overlapped, and loader-wait should fall to near zero. **If it does not drop
below ~5% in the first minutes of the run, raise `N_WORKERS`** — that single
number is the thing to watch at launch.

**Loader parity on real v2 data** (`REAL_CSV=data/siimpl_rot/siimpl_eval_v2.csv
REAL_N=2400 python test_loader_parity.py`) — the full suite re-run against real
v2 tracks rather than the synthetic pool, with 2 genuine worker processes:

```
[REAL] data/siimpl_rot/siimpl_eval_v2.csv
       pooled 2400 of 101,149 tracks, mean len 95, 300 steps/epoch
  [PASS] batches arrive strictly in order
  [PASS] parallel batches are BYTE-IDENTICAL to serial (inputs)   max|dx|=0.000e+00
  [PASS] parallel batches are BYTE-IDENTICAL to serial (targets)  max|dtheta|=0.000e+00
  [PASS] parallel batch ORDER matches serial
  [PASS] epoch 0/1: no track repeated, covers the whole pool (2400/2400)
  [PASS] consecutive epochs use DIFFERENT permutations
  [PASS] p_zero point mass preserved (~0.30)  measured 0.297
ALL LOADER PARITY CHECKS PASSED
```

So `TrackPool` and the multi-process loader are now verified against the real
v2 track-length distribution and real ID ranges, not just synthetic data.

---

## 9. Launching on the A100

**Data: the v2 corpus is now the DEFAULT.** No env var needs setting.
`config.py` and `launch_a100.sh` both point at:

| | tracks | tiers (low/mid/high) | size |
|---|---|---|---|
| `data/siimpl_rot/siimpl_train_v2.csv` | 1,009,384 | 24.9 / 30.0 / 45.2 % | 8.4 GB |
| `data/siimpl_rot/siimpl_eval_v2.csv` | 101,731 | 24.7 / 30.1 / 45.2 % | 0.9 GB |

That is 3.4x the training tracks of the original corpus and a deliberate
re-weighting toward the mid/high tiers where the model has a real edge (the
original was 35/28/37). To fall back to the old files, override
`TRAIN_CSV` / `EVAL_CSV`.

Note on `ion_number`: the v2 files have non-dense ID ranges. `load_raw_tracks`
segments tracks with `np.diff(ion_ids) != 0` on the sorted IDs, so gaps are
harmless — it detects *changes*, not contiguity. What would NOT be harmless is
duplicate IDs across merged shards, which would silently fuse two physically
distinct tracks into one; the corpus was verified collision-free before use.
Track ordering within a shard is likewise irrelevant here because
`MAX_POINTS=600` exceeds the longest track, so nothing is truncated and no
point is order-dependent.

**Scheduler assumption.** The cluster's scheduler was not specified, so
`launch_a100.sh` defaults to plain `python` invocations, which work on any box
you can ssh into, in a container, or inside an already-allocated SLURM
allocation. `MODE=slurm` emits and submits real `sbatch` scripts — **edit the
`#SBATCH --partition` / `--account` lines for your cluster before using it.**

```bash
# ONE A100: the two ladder configs run back to back, ~14.5h each
bash smearing_resolution/architecture_experiments/a100_final/launch_a100.sh

# TWO A100s on one node: both configs concurrently, full 30h each
MODE=par TOTAL_HOURS=30 bash .../launch_a100.sh

# SLURM: one job per config
MODE=slurm TOTAL_HOURS=30 bash .../launch_a100.sh
```

**Step count is measured, not guessed.** `TARGET_STEPS` shapes the cosine LR
schedule, so setting it above what the job can finish means the LR never
anneals. The launcher therefore runs a short `CALIB_STEPS=200` calibration per
config, reads the `[THROUGHPUT] <it/s>` line `train.py` emits, and sets
`TARGET_STEPS = it/s * budget_seconds * 0.92`. Override with
`TARGET_STEPS_SMALL=... TARGET_STEPS_LARGE=...` to skip calibration.

Checkpointing: `checkpoint_latest.pt` every `CHECKPOINT_EVERY_SEC` (default
20 min) with model + optimizer + EMA + step + config; `RESUME=1` (default)
picks it up automatically, so a preempted job restarts in place. Each run also
writes `checkpoint_final.pt`, `train_log.json` (with epoch/passes),
`energy_split_final.csv`, and the launcher then runs `eval.py` to produce
`full_sweep_9tier.csv`.

**Evaluation.**

```bash
CONFIG=large CKPT=.../results_large/checkpoint_final.pt N_PER_BIN=1000 \
  python smearing_resolution/architecture_experiments/a100_final/eval.py
```

Produces the same 9-sigma x 3-energy grid as
`results_diag_r7_trunc/full_sweep_9tier_6nm_EMA.csv`, with identical column
names, plus an `ece_pct` column. The PCA baseline is **recomputed here** with
the same convention as `eval_r7_full_sweep.py:211` (eigh, skewness sign-fix) so
model and baseline are measured on the identical smeared realizations; ECE uses
the definition from `eval_diag_r7_trunc_calibration.py:281` unchanged. EMA
weights are used by default (`EMA=1`), matching the reference sweep.

---

## 10. Scope notes

Out of scope by instruction and deliberately not attempted: the low-energy
information floor (a proven physical limit, not an architecture problem), and
the physics-informed matched-filter/template idea (separately reviewed and
rejected — the bias-not-variance evidence undermines its justification, and it
carries misspecification and physics-systematic risk).
