"""One-off: measure max track length + typical NN spacing in normalized coords.
Used to pick MAX_POINTS and the blur-adaptive graph's base bandwidth h0.
Writes _data_stats.json next to this file. Not part of training."""
import sys, os, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = os.path.join(_ROOT, 'smearing_resolution/architecture_experiments/a100_final/_data_stats.json')
res = {}
for name, path in [('train', 'data/siimpl_rot/siimpl_train.csv'),
                   ('eval', 'data/siimpl_rot/siimpl_eval_merged.csv')]:
    df = pd.read_csv(path, usecols=['x', 'y', 'z', 'ion_number', 'energy_keV'])
    g = df.groupby('ion_number').size()
    g = g[g >= 3]
    res[name] = dict(n_tracks=int(len(g)), max_len=int(g.max()),
                     q95=int(g.quantile(0.95)), q99=int(g.quantile(0.99)),
                     q999=int(g.quantile(0.999)), mean_len=float(g.mean()))
    if name == 'train':
        # NN spacing in R_g-normalized coords (unblurred), by energy tier
        df = df.sort_values('ion_number')
        ions = df['ion_number'].values
        xyz = df[['x', 'y', 'z']].values.astype(np.float64)
        en = df['energy_keV'].values
        bnd = np.where(np.diff(ions) != 0)[0] + 1
        starts = np.concatenate([[0], bnd]); ends = np.concatenate([bnd, [len(ions)]])
        rng = np.random.default_rng(0)
        sel = rng.choice(len(starts), size=min(3000, len(starts)), replace=False)
        tiers = {'low': [], 'mid': [], 'high': []}
        for t in sel:
            s, e = starts[t], ends[t]
            if e - s < 4:
                continue
            p = xyz[s:e]
            c = p - p.mean(0)
            rg = np.sqrt((c ** 2).sum(1).mean())
            if rg <= 0:
                continue
            cn = c / rg
            d, _ = cKDTree(cn).query(cn, k=2)
            nn = float(np.median(d[:, 1]))
            E = en[s]
            k = 'low' if E < 5 else ('mid' if E < 20 else 'high')
            tiers[k].append(nn)
        res['nn_spacing_rg_normalized'] = {
            k: dict(n=len(v), median=float(np.median(v)) if v else None,
                    p10=float(np.percentile(v, 10)) if v else None,
                    p90=float(np.percentile(v, 90)) if v else None)
            for k, v in tiers.items()}
    del df
print(json.dumps(res, indent=2))
with open(OUT, 'w') as f:
    json.dump(res, f, indent=2)
