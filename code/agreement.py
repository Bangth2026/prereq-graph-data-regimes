"""Krippendorff's alpha (interval and ordinal) and r_wg for the crowd prerequisite ratings.
Rater identities are not released, so each pair is a unit with m exchangeable ratings (alpha handles unequal m).
Output: out_agreement.json"""
import numpy as np, pandas as pd, json, krippendorff
from common import crowd
res = {}
for nm, f in [('main_1131', 'data/relationship_annotation_training.csv'), ('lowpanel_823', 'data/relationship_annotation_testing.csv')]:
    d = pd.read_csv(f); R = [np.array([int(v) for v in s.split('_')]) for s in d.Prerequisite_raw]
    m = max(len(r) for r in R); M = np.full((m, len(R)), np.nan)
    for j, r in enumerate(R): M[:len(r), j] = r          # rows = rating slots (not rater identities), cols = pairs
    out = {'pairs': len(R), 'ratings': int(sum(len(r) for r in R))}
    for lvl in ['interval', 'ordinal']:
        out[f'alpha_{lvl}'] = float(krippendorff.alpha(reliability_data=M, level_of_measurement=lvl))
    # bootstrap over pairs for alpha (interval)
    rng = np.random.default_rng(7); bs = []
    for _ in range(500):
        s = rng.integers(0, len(R), len(R)); bs.append(krippendorff.alpha(reliability_data=M[:, s], level_of_measurement='interval'))
    out['alpha_interval_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # r_wg (James, Demaree & Wolf, 1984) with a uniform null on a 10-point scale: sigma2_EU = (A^2 - 1)/12
    s2eu = (10 ** 2 - 1) / 12
    rwg = np.array([max(0.0, 1 - np.var(r, ddof=1) / s2eu) for r in R])
    out.update(rwg_median=float(np.median(rwg)), rwg_iqr=[float(np.percentile(rwg, 25)), float(np.percentile(rwg, 75))],
               rwg_share_ge_070=float((rwg >= 0.70).mean()))
    res[nm] = out
cr = crowd(); R = [np.array([int(v) for v in s.split('_')]) for s in cr.raw]
s2eu = 99 / 12; rwg = np.array([max(0.0, 1 - np.var(r, ddof=1) / s2eu) for r in R])
res['spearman_Q_rwg_1131'] = float(pd.Series(cr.Q.values).corr(pd.Series(rwg), method='spearman'))
res['rwg_ge_070_vs_Q_ge_080_agreement'] = float(((rwg >= 0.70) == (cr.Q.values >= 0.80 - 1e-9)).mean())
res['admitted_by_rwg_070'] = int((rwg >= 0.70).sum())
json.dump(res, open('out_agreement.json', 'w'), indent=1); print(json.dumps(res, indent=1))
