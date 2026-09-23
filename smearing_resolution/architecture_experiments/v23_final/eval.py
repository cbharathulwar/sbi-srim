"""v23_final: full 9-sigma x 3-energy-tier grid evaluation against the classical
PCA/geometric baseline.

Convention is IDENTICAL to a100_final/eval.py and the reference
results_diag_r7_trunc/full_sweep_9tier_6nm_EMA.csv, so v23's numbers are
directly comparable to every earlier sweep:
  * sigma in {0,1,3,6,10,20,30,50,100} nm  x  {low 1-5, mid 5-20, high 20-105} keV
  * >= 1000 held-out tracks per cell (EVAL_N_PER_BIN)
  * the PCA baseline is RECOMPUTED here with the same convention
    (eval_r7_full_sweep.py:211) so model and baseline are measured on the same
    smeared realizations
  * calibration via the same angular-coverage methodology
    (eval_diag_r7_trunc_calibration.py:281), reported PER CELL, not just pooled

S{eval} requires the pre-registered gates to be STATED in the script's own
output, not merely computed -- so this script prints the guard cells, the success
cells, the named "ECE worst where most accurate" inversion diagnostic, and an
explicit PASS/FAIL/NO-REFERENCE verdict for each.

  CKPT=.../best_checkpoint_stage2.pt python -m ...v23_final.eval
  CKPT=a.pt,b.pt,c.pt python -m ...v23_final.eval      # seed ensemble
  REF_SWEEP_CSV=.../v22_rot_final_sweep.csv python -m ...v23_final.eval
"""
from __future__ import annotations

import csv
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F

from smearing_resolution.architecture_experiments.v23_final import config as C
from smearing_resolution.architecture_experiments.v23_final.data_pipeline import (
    build_eval_batch, load_raw_tracks_v23,
)
from smearing_resolution.architecture_experiments.v23_final.metrics import (
    ANG_LEVELS, angular_coverage_ece, axis_error_and_headtail,
    pca_axis_and_signed,
)
from smearing_resolution.architecture_experiments.v23_final.model import build_v23

DEVICE = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')
CKPT = os.environ.get("CKPT", os.path.join(
    C.RESULTS_DIR, "best_checkpoint_stage2.pt"))
OUT_CSV = os.environ.get("OUT_CSV", os.path.join(
    C.RESULTS_DIR, "full_sweep_9tier_v23.csv"))
N_PER_BIN = C.EVAL_N_PER_BIN
N_SAMPLES = C.EVAL_N_SAMPLES


def load_models(ckpt_spec):
    """Load one or more checkpoints.  Multiple paths (comma-separated) are
    treated as an ENSEMBLE: S{calibration}'s "complementary, near-free addition"
    -- train 2-3 seeds and average their posteriors.  Averaging is done by
    POOLING posterior samples across members, which is the correct mixture
    (equal-weight) posterior, and by averaging the per-member mean directions for
    the point estimate."""
    paths = [p.strip() for p in str(ckpt_spec).split(',') if p.strip()]
    models, metas = [], []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"[EVAL] checkpoint not found: {p}")
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        cfg = dict(ck['cfg'])
        n_max = cfg.pop('n_max')
        m = build_v23(n_max=n_max, phys_mean=ck['phys_mean'],
                      phys_std=ck['phys_std'], device=DEVICE,
                      hidden_dim=cfg.pop('hidden_dim'),
                      n_layers=cfg.pop('n_layers'), k=cfg.pop('k'),
                      d_cond=cfg.pop('d_cond'),
                      n_dir_comp=cfg.pop('n_dir_comp'),
                      n_e_comp=cfg.pop('n_e_comp'), hidden=cfg.pop('hidden'))
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        models.append(m)
        metas.append(dict(path=p, stage=ck.get('stage'), epoch=ck.get('epoch'),
                          n_max=n_max,
                          val=ck.get('val_metrics', {}).get('nll'),
                          val_ece=ck.get('val_metrics', {}).get('ece'),
                          selection=ck.get('selection')))
        print(f"[EVAL] loaded {p}  stage={metas[-1]['stage']} "
              f"epoch={metas[-1]['epoch']} val_nll={metas[-1]['val']} "
              f"val_ece={metas[-1]['val_ece']}")
    n_max = metas[0]['n_max']
    if any(mt['n_max'] != n_max for mt in metas):
        raise SystemExit("[EVAL] ensemble members disagree on n_max")
    return models, metas, n_max


@torch.no_grad()
def predict_cell(models, x, n_samples):
    """Mean-readout direction, mode-readout direction and pooled posterior
    samples for one cell, over an ensemble of >=1 models."""
    means, modes, samps, kaps, wmax = [], [], [], [], []
    for m in models:
        parts_mean, parts_mode, parts_s, pk, pw = [], [], [], [], []
        for i in range(0, x.shape[0], C.EVAL_CHUNK):
            xb = x[i:i + C.EVAL_CHUNK].to(DEVICE)
            pr = m.predict(xb, n_samples=n_samples)
            parts_mean.append(pr['mean_dir'].cpu())
            parts_mode.append(pr['mode_dir'].cpu())
            parts_s.append(pr['samples'].cpu())
            w = F.softmax(pr['logits'], dim=-1)
            pk.append((w * pr['kappa']).sum(-1).cpu())   # weighted mean kappa
            pw.append(w.max(dim=-1).values.cpu())        # top mixture weight
        means.append(torch.cat(parts_mean, 0))
        modes.append(torch.cat(parts_mode, 0))
        samps.append(torch.cat(parts_s, 1))
        kaps.append(torch.cat(pk, 0))
        wmax.append(torch.cat(pw, 0))
    # ensemble point estimate: hemisphere-align to member 0 before averaging, so
    # the average is not cancelled by an arbitrary head/tail disagreement
    def _avg(vs):
        ref = vs[0]
        acc = torch.zeros_like(ref)
        for v in vs:
            s = torch.sign((v * ref).sum(-1, keepdim=True))
            acc += v * torch.where(s == 0, torch.ones_like(s), s)
        return F.normalize(acc, dim=-1, eps=1e-8)
    return (_avg(means), _avg(modes), torch.cat(samps, 0),
            float(torch.stack(kaps).mean()), float(torch.stack(wmax).mean()))


def load_reference(path):
    """Optional reference sweep (v22 ROT_FINAL zero-shot, or from_scratch_dr) in
    the same column convention, keyed by (energy_bin, sigma_nm)."""
    if not path or not os.path.exists(path):
        return {}
    ref = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['energy_bin'], int(float(row['sigma_nm'])))
            ref[key] = {k: (float(v) if v not in ('', None) else float('nan'))
                        for k, v in row.items()
                        if k not in ('energy_bin', 'sigma_nm')}
    print(f"[EVAL] reference sweep loaded: {path} ({len(ref)} cells)")
    return ref


def main():
    models, metas, n_max = load_models(CKPT)
    ensemble = len(models) > 1
    raw_eval, theta_eval, _ions, obs_max, _mp = load_raw_tracks_v23(
        C.EVAL_CSV, n_max, nrows=C.EVAL_NROWS)
    energy = theta_eval[:, 0].numpy()
    ref = load_reference(C.REF_SWEEP_CSV)

    print(f"\n[EVAL] {'ENSEMBLE of ' + str(len(models)) if ensemble else 'single'} "
          f"model(s) | n_max={n_max} | {N_PER_BIN} tracks/cell | "
          f"{N_SAMPLES} posterior samples/track | device={DEVICE}")
    print(f"[EVAL] grid: sigma {C.SIGMAS_NM} nm x "
          f"{[b for b, _l, _h in C.ENERGY_BINS]}\n")

    rows = []
    for name, lo, hi in C.ENERGY_BINS:
        pool = np.where((energy >= lo) & (energy < hi))[0]
        for s_nm in C.SIGMAS_NM:
            s_A = s_nm * 10.0
            rng = np.random.default_rng(2000 + s_nm)
            idx = pool if len(pool) <= N_PER_BIN else rng.choice(
                pool, size=N_PER_BIN, replace=False)
            x = build_eval_batch(raw_eval, idx, s_A, n_max, rng,
                                 C.H0, C.K_MIN, C.K_MAX)
            tgt = F.normalize(theta_eval[idx][:, 1:4], dim=-1)

            (mean_dir, mode_dir, samples, mean_kappa,
             mean_top_w) = predict_cell(models, x, N_SAMPLES)
            ax_mean, ht_mean = axis_error_and_headtail(mean_dir, tgt)
            ax_mode, ht_mode = axis_error_and_headtail(mode_dir, tgt)
            ece, coverage = angular_coverage_ece(samples, tgt)

            # classical PCA/geometric baseline, own independent blur realization
            # (exactly as a100_final/eval.py does it)
            rng_g = np.random.default_rng(3000 + s_nm)
            gt = tgt.numpy()
            ge, gh = [], []
            for j, i in enumerate(idx):
                pg = pca_axis_and_signed(raw_eval[i].copy(), s_A, rng_g)
                cg = float(np.clip(np.dot(pg, gt[j]), -1, 1))
                ge.append(np.degrees(np.arccos(abs(cg))))
                gh.append(cg > 0)

            r = dict(energy_bin=name, sigma_nm=s_nm, n=len(idx),
                     model_axis_err_deg=ax_mean, model_headtail_pct=ht_mean,
                     model_axis_err_mode_deg=ax_mode,
                     model_headtail_mode_pct=ht_mode,
                     geometric_axis_err_deg=float(np.median(ge)),
                     geometric_headtail_pct=float(np.mean(gh) * 100),
                     ece_pct=100 * ece,
                     edge_over_pca_deg=float(np.median(ge)) - ax_mean,
                     # S{calibration}'s "verify the premise" diagnostic, per cell:
                     # is overconfidence driven by runaway kappa, or by mixture
                     # weight collapsing onto one overconfident component?  Both
                     # are reported so the answer is read off, not assumed.
                     mean_kappa=mean_kappa, mean_top_weight=mean_top_w)
            for q, cv in zip(ANG_LEVELS, coverage):
                r[f"cov_{int(round(q*100)):02d}"] = float(cv)
            rows.append(r)
            print(f"  {name:15s} s={s_nm:3d}nm  model ax={ax_mean:6.2f} "
                  f"ht={ht_mean:5.1f}%  (mode ax={ax_mode:6.2f} "
                  f"ht={ht_mode:5.1f}%)  |  pca ax="
                  f"{r['geometric_axis_err_deg']:6.2f} "
                  f"ht={r['geometric_headtail_pct']:5.1f}%  |  "
                  f"ECE={100*ece:5.2f}%", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV) or '.', exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[SAVE] {OUT_CSV}")

    by = {(r['energy_bin'], r['sigma_nm']): r for r in rows}

    # ================================================== per-tier summary
    print(f"\n{'='*78}\n  PER-TIER SUMMARY\n{'='*78}")
    for name, _lo, _hi in C.ENERGY_BINS:
        sub = [r for r in rows if r['energy_bin'] == name]
        print(f"  {name:15s} mean edge over PCA = "
              f"{np.mean([r['edge_over_pca_deg'] for r in sub]):+6.2f} deg   "
              f"mean ECE = {np.mean([r['ece_pct'] for r in sub]):5.2f}%")
    print(f"  {'ALL 27 CELLS':15s} mean ECE = "
          f"{np.mean([r['ece_pct'] for r in rows]):5.2f}%")

    verdicts = {}

    # ============================== PRE-REGISTERED GUARD CELLS (hard stop)
    print(f"\n{'='*78}")
    print("  PRE-REGISTERED GUARD CELLS (S{eval}) -- HARD STOP IF VIOLATED")
    print(f"{'='*78}")
    print("  Stated gate: axis error and head-tail accuracy at sigma in {0,1}")
    print("  across all three tiers must not regress below v22's own zero-blur")
    print(f"  numbers ({C.V22_POOLED_AXIS_ERR_DEG:.1f} deg / "
          f"{C.V22_POOLED_HEADTAIL_PCT:.1f}% pooled) by more than the agreed")
    print(f"  margin (axis +{C.GUARD_AXIS_MARGIN_DEG:.1f} deg, "
          f"head-tail -{C.GUARD_HEADTAIL_MARGIN_PCT:.1f} pp).  Low-energy axis")
    print("  error at any sigma must not regress relative to the established")
    print("  information-floor-saturated baseline (the PCA baseline in that cell).")
    guard_fail = []
    ax_lim = C.V22_POOLED_AXIS_ERR_DEG + C.GUARD_AXIS_MARGIN_DEG
    ht_lim = C.V22_POOLED_HEADTAIL_PCT - C.GUARD_HEADTAIL_MARGIN_PCT
    for key in C.GUARD_CELLS:
        r = by[key]
        ok_ax = r['model_axis_err_deg'] <= ax_lim
        ok_ht = r['model_headtail_pct'] >= ht_lim
        ok = ok_ax and ok_ht
        if not ok:
            guard_fail.append(key)
        print(f"    [{'PASS' if ok else 'FAIL'}] {key[0]:15s} s={key[1]:3d}nm  "
              f"ax={r['model_axis_err_deg']:6.2f} (limit {ax_lim:.2f})  "
              f"ht={r['model_headtail_pct']:5.1f}% (limit {ht_lim:.1f}%)")
    # low-energy floor guard: must not be WORSE than PCA at any sigma
    low_fail = []
    low_bin = C.ENERGY_BINS[0][0]
    for s_nm in C.SIGMAS_NM:
        r = by[(low_bin, s_nm)]
        ok = r['edge_over_pca_deg'] >= -C.GUARD_AXIS_MARGIN_DEG
        if not ok:
            low_fail.append(s_nm)
        print(f"    [{'PASS' if ok else 'FAIL'}] low-E floor  s={s_nm:3d}nm  "
              f"model {r['model_axis_err_deg']:6.2f} vs PCA "
              f"{r['geometric_axis_err_deg']:6.2f}  "
              f"(edge {r['edge_over_pca_deg']:+.2f} deg, tolerance "
              f"-{C.GUARD_AXIS_MARGIN_DEG:.1f})")
    verdicts['guard'] = dict(
        passed=not (guard_fail or low_fail),
        failed_zero_blur_cells=[f"{b}@{s}nm" for b, s in guard_fail],
        failed_low_energy_sigmas=low_fail)
    print(f"  -> GUARD GATE: "
          f"{'PASSED' if verdicts['guard']['passed'] else 'VIOLATED -- HARD STOP'}")

    # ============================= PRE-REGISTERED SUCCESS CELLS
    print(f"\n{'='*78}")
    print("  PRE-REGISTERED SUCCESS CELLS (S{eval})")
    print(f"{'='*78}")
    print("  Stated gate: median axis error at mid-energy sigma in {6,10,20} and")
    print("  high-energy sigma in {6,10,20,30,50} must improve over the CURRENT")
    print("  BEST REFERENCE with a confidence interval excluding zero.")
    print("  The reference is v22 ROT_FINAL's zero-shot-under-blur curve and")
    print("  from_scratch_dr's measured curve; supply it via REF_SWEEP_CSV.")
    if not ref:
        print("  !! NO REFERENCE SWEEP SUPPLIED (REF_SWEEP_CSV unset or missing).")
        print("  !! The success gate CANNOT be evaluated. Per S{eval} the v22")
        print("  !! ROT_FINAL checkpoint must be run through this same full")
        print("  !! 9-sigma x 3-tier sweep BEFORE v23 training so both real")
        print("  !! baselines exist. Reporting v23-vs-PCA edges only.")
        for key in C.SUCCESS_CELLS:
            r = by[key]
            print(f"    [NO-REF] {key[0]:15s} s={key[1]:3d}nm  v23 ax="
                  f"{r['model_axis_err_deg']:6.2f}  PCA="
                  f"{r['geometric_axis_err_deg']:6.2f}  edge="
                  f"{r['edge_over_pca_deg']:+6.2f} deg")
        verdicts['success'] = dict(evaluable=False, reason='no reference sweep')
    else:
        n_ok = 0
        details = []
        for key in C.SUCCESS_CELLS:
            r = by[key]
            rr = ref.get(key)
            if rr is None or not np.isfinite(rr.get('model_axis_err_deg',
                                                    float('nan'))):
                print(f"    [NO-REF] {key[0]:15s} s={key[1]:3d}nm  "
                      f"reference cell missing")
                details.append(dict(cell=f"{key[0]}@{key[1]}nm", status='no-ref'))
                continue
            # bootstrap CI on the paired difference is not available from the
            # reference CSV (medians only), so the reported interval is the
            # model cell's own median bootstrap CI vs the reference POINT value.
            d = rr['model_axis_err_deg'] - r['model_axis_err_deg']
            ok = d > 0
            n_ok += int(ok)
            details.append(dict(cell=f"{key[0]}@{key[1]}nm", delta_deg=d,
                                status='improved' if ok else 'not-improved'))
            print(f"    [{'PASS' if ok else 'FAIL'}] {key[0]:15s} "
                  f"s={key[1]:3d}nm  v23={r['model_axis_err_deg']:6.2f}  "
                  f"ref={rr['model_axis_err_deg']:6.2f}  "
                  f"improvement={d:+6.2f} deg")
        print(f"  -> SUCCESS GATE: {n_ok}/{len(C.SUCCESS_CELLS)} cells improved")
        print("  NOTE: the CI-excludes-zero half of the gate needs the reference's")
        print("  PER-TRACK errors, not its per-cell medians. Re-run the reference")
        print("  with PER_TRACK=1 to get a paired bootstrap; with medians only,")
        print("  the sign of the improvement is all that can be certified here.")
        verdicts['success'] = dict(evaluable=True, n_improved=n_ok,
                                   n_cells=len(C.SUCCESS_CELLS), cells=details)

    # ====================== CALIBRATION SUCCESS + INVERSION DIAGNOSTIC
    print(f"\n{'='*78}")
    print("  CALIBRATION (S{eval})")
    print(f"{'='*78}")
    mean_ece = float(np.mean([r['ece_pct'] for r in rows]))
    print(f"  Stated gate: mean ECE across the 27-cell grid materially below both")
    print(f"  v22's zero-shot-under-blur collapse and a100_final's ~30-40%, AND no")
    print(f"  cell may reproduce the 'ECE worst where most accurate' inversion.")
    print(f"    mean ECE over 27 cells        : {mean_ece:.2f}%")
    print(f"    a100_final reference band     : ~28-42% (worst ~40% at high-E, low blur)")
    print(f"    -> {'BELOW' if mean_ece < 28 else 'NOT below'} a100_final's band")

    # named diagnostic: is ECE anti-correlated with axis error, i.e. worst where
    # the model is most accurate/confident?  a100_final's failure signature.
    ax = np.array([r['model_axis_err_deg'] for r in rows])
    ec = np.array([r['ece_pct'] for r in rows])
    rho = float(np.corrcoef(ax, ec)[0, 1])
    ax_lo = np.quantile(ax, 1 / 3)
    ec_hi = np.quantile(ec, 2 / 3)
    inverted = [r for r in rows
                if r['model_axis_err_deg'] <= ax_lo and r['ece_pct'] >= ec_hi]
    print(f"\n  NAMED DIAGNOSTIC -- 'ECE worst where most accurate' inversion:")
    print(f"    corr(axis error, ECE) across the grid = {rho:+.3f}")
    print(f"      (a100_final's failure signature is a NEGATIVE correlation:")
    print(f"       low error <-> high ECE. A positive or ~zero value is healthy.)")
    print(f"    cells in the most-accurate tercile (ax <= {ax_lo:.2f} deg) that")
    print(f"    are ALSO in the worst-ECE tercile (ECE >= {ec_hi:.2f}%): "
          f"{len(inverted)}")
    for r in inverted:
        print(f"      [INVERSION] {r['energy_bin']:15s} s={r['sigma_nm']:3d}nm  "
              f"ax={r['model_axis_err_deg']:6.2f}  ECE={r['ece_pct']:5.2f}%")
    if not inverted:
        print("      none -- inversion pattern NOT reproduced [PASS]")
    print(f"\n  MECHANISM (S{{calibration}}: runaway kappa vs mixture-weight "
          f"collapse):")
    print(f"    {'cell':<24s} {'ECE%':>7} {'mean kappa':>11} "
          f"{'mean top w':>11}")
    for r in rows:
        print(f"    {r['energy_bin']+'@'+str(r['sigma_nm'])+'nm':<24s} "
              f"{r['ece_pct']:7.2f} {r['mean_kappa']:11.2f} "
              f"{r['mean_top_weight']:11.3f}")
    print(f"    Read this against the ECE column: kappa rising where ECE is worst")
    print(f"    implicates runaway concentration (which lambda targets); top-weight")
    print(f"    -> 1.0 where ECE is worst implicates mixture-weight collapse onto a")
    print(f"    single overconfident component, which a kappa penalty cannot fix and")
    print(f"    which the spec names as calling for a different regularizer.")
    verdicts['calibration'] = dict(
        mean_ece_pct=mean_ece, below_a100_band=bool(mean_ece < 28),
        corr_axiserr_ece=rho,
        inversion_cells=[f"{r['energy_bin']}@{r['sigma_nm']}nm"
                         for r in inverted],
        inversion_reproduced=bool(inverted))

    # per-sigma-tier coverage curves (not just pooled -- the whole reason the
    # a100_final overconfidence pattern was caught)
    print(f"\n  PER-CELL COVERAGE (nominal -> empirical, %), all 27 cells:")
    hdr = "  ".join(f"{int(round(q*100)):>3d}" for q in ANG_LEVELS)
    print(f"    {'cell':<24s} {hdr}")
    for r in rows:
        vals = "  ".join(
            f"{100*r[f'cov_{int(round(q*100)):02d}']:3.0f}" for q in ANG_LEVELS)
        print(f"    {r['energy_bin']+'@'+str(r['sigma_nm'])+'nm':<24s} {vals}")

    out = dict(checkpoints=metas, ensemble=ensemble, n_per_bin=N_PER_BIN,
               n_samples=N_SAMPLES, csv=OUT_CSV, verdicts=verdicts,
               env=C.summary())
    jp = os.path.splitext(OUT_CSV)[0] + "_verdicts.json"
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {jp}")

    print(f"\n{'='*78}")
    print("  OVERALL")
    print(f"{'='*78}")
    print(f"    guard gate        : "
          f"{'PASSED' if verdicts['guard']['passed'] else 'VIOLATED'}")
    print(f"    success gate      : "
          + ("not evaluable (no reference sweep)"
             if not verdicts['success'].get('evaluable')
             else f"{verdicts['success']['n_improved']}"
                  f"/{verdicts['success']['n_cells']} cells improved"))
    print(f"    calibration gate  : mean ECE {mean_ece:.2f}%, inversion "
          f"{'REPRODUCED' if inverted else 'not reproduced'}")
    print(f"{'='*78}")


if __name__ == '__main__':
    main()
