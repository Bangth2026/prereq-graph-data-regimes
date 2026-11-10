"""Summary of the extended data-regime study (Junyi; post hoc) and the regime figure.

python data_data_regime_summary_basic.py JUNYI_REGIME_DIR XES_REGIME_DIR
Reads f{F}_s{SEED}/out2.json (validation learners kept whole) and out2_vs.json (validation learners subsampled in the same
proportion as the training learners) for Junyi, and out.json for XES3G5M.

Pooled intervals: for each size the three seeds are pooled as a mixture d_s + (b_s,r - mean_r b_s) over seeds s and shared
learner-bootstrap replicates r (900 values); 2.5/97.5 percentiles give the 95% interval of one platform-size realisation,
5/95 percentiles the 90% interval used for the equivalence test against the +/-0.005 margin (TOST).
Trend: weighted least squares of the pooled increment on log10(training learners), weights 1/variance; N* = training learners
at which the fitted line reaches +0.005; intervals from 2,000 joint resamples (one seed per size, one shared learner replicate
for all sizes, because the held-out learners are the same at every size).
Outputs: out_regime2.json, regime2_table.csv, fig/Fig7_regime.png
"""
import sys, json, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
JD, XD = sys.argv[1], sys.argv[2]
MARGIN = 0.005
import os; MINS = int(os.environ.get('MINSEED', '3'))
def load(var):
    P = {}
    for f in glob.glob(f'{JD}/f*_s*/out2{var}.json'):
        F, S = re.search(r'f([\d.]+)_s(\d+)', f).groups(); P[(float(F), int(S))] = json.load(open(f))
    return P
def mixture(points, key_point, key_boot):
    """points: list of out dicts (seeds). Returns pooled mean, mixture draws (n_seed x 300)."""
    d = np.array([key_point(o) for o in points]); B = np.array([key_boot(o) for o in points])
    return d, d[:, None] + (B - B.mean(1, keepdims=True))
OUT = {}; ROWS = []
for var, vname in [('', 'validation_whole'), ('_vs', 'validation_subsampled')]:
    P = load(var); sizes = sorted({F for F, _ in P})
    res = {}
    for F in sizes:
        pts = [P[(F, s)] for s in sorted(s for f_, s in P if f_ == F)]
        r = {'fraction': F, 'seeds': len(pts), 'learners': int(np.mean([o['n_training_learners'] for o in pts])),
             'validation_cases': int(np.mean([o['validation_cases_used'] for o in pts])), 'validation_learners': int(np.mean([o['validation_learners_used'] for o in pts]))}
        r['DKT_auc'] = float(np.mean([o['accuracy']['DKT']['auc'] for o in pts]))
        r['GPE-DKT_auc'] = float(np.mean([o['accuracy']['GDKT']['auc'] for o in pts]))
        r['DKT_brier'] = float(np.mean([o['accuracy']['DKT']['brier'] for o in pts])); r['GPE-DKT_brier'] = float(np.mean([o['accuracy']['GDKT']['brier'] for o in pts]))
        for pn in ['snapshot', 'fully_sequential']:
            for base in ['B1', 'B1RK']:
                tag = f'{pn}|{base}'; X = lambda o: o['extended']
                r[f'{tag}|auc'] = float(np.mean([o['roles'][tag]['auc'] for o in pts]))
                d, M = mixture(pts, lambda o: o['roles'][tag]['ESD-DKT_vs_base'][0], lambda o: X(o)[f'{tag}|boot_weighting'])
                r[f'{tag}|weighting'] = float(d.mean()); r[f'{tag}|weighting_ci'] = np.percentile(M, [2.5, 97.5]).tolist()
                r[f'{tag}|weighting_ci90'] = np.percentile(M, [5, 95]).tolist(); r[f'{tag}|weighting_seed_range'] = [float(d.min()), float(d.max())]
                r[f'{tag}|_mix'] = M
                d, M = mixture(pts, lambda o: X(o)[f'{tag}|ESD-ZERO_vs_base'][0], lambda o: X(o)[f'{tag}|boot_ESD-ZERO'])
                r[f'{tag}|ESD-ZERO'] = float(d.mean()); r[f'{tag}|ESD-ZERO_ci'] = np.percentile(M, [2.5, 97.5]).tolist()
                r[f'{tag}|ESD-POP'] = float(np.mean([X(o)[f'{tag}|ESD-POP_vs_base'][0] for o in pts]))
                r[f'{tag}|estimation_DKT'] = float(np.mean([o['roles'][tag]['estimation_role_DKT'][0] for o in pts]))
                r[f'{tag}|estimation_DKT_range'] = [float(min(o['roles'][tag]['estimation_role_DKT'][0] for o in pts)), float(max(o['roles'][tag]['estimation_role_DKT'][0] for o in pts))]
                fl = [X(o)[f'{tag}|flags20_ESD-DKT_minus_base'] for o in pts]
                r[f'{tag}|flags20_diff'] = float(np.mean([f[0] for f in fl])); r[f'{tag}|flags20_diff_seed_ci'] = [float(min(f[1] for f in fl)), float(max(f[2] for f in fl))]
                r[f'{tag}|flags20_base'] = float(np.mean([X(o)[f'{tag}|flags20_base'] for o in pts]))
                rr = [x for o in pts for x in X(o).get(f'{tag}|real_minus_random', [])]
                rd = [x for o in pts for x in X(o).get(f'{tag}|random_ESD-DKT_vs_base', [])]
                if rr:
                    r[f'{tag}|real_minus_random'] = float(np.mean([x[0] for x in rr])); r[f'{tag}|real_minus_random_range'] = [float(min(x[0] for x in rr)), float(max(x[0] for x in rr))]
                    r[f'{tag}|real_minus_random_min_lo'] = float(min(x[1] for x in rr)); r[f'{tag}|random_vs_base'] = float(np.mean([x[0] for x in rd]))
            bl = [o['extended']['best_learner_model'][pn] for o in pts]
            d = np.array([b['weighting'][0] for b in bl]); B = np.array([b['boot_weighting'] for b in bl]); M = d[:, None] + (B - B.mean(1, keepdims=True))
            r[f'{pn}|best|chosen'] = sorted({b['chosen'] for b in bl}); r[f'{pn}|best|auc'] = float(np.mean([b['auc'] for b in bl]))
            r[f'{pn}|best|weighting'] = float(d.mean()); r[f'{pn}|best|weighting_ci'] = np.percentile(M, [2.5, 97.5]).tolist(); r[f'{pn}|best|_mix'] = M
            r[f'{pn}|best|weighting_ci90'] = np.percentile(M, [5, 95]).tolist()
            r[f'{pn}|best|ESD_ZERO'] = float(np.mean([b['ESD_ZERO'][0] for b in bl]))
            r[f'{pn}|best|auc_by_model'] = {k: float(np.mean([b['auc_by_model'][k] for b in bl])) for k in bl[0]['auc_by_model']}
        res[F] = r
    # trend and N*, equivalence crossing
    TR = {}
    for pn in ['snapshot', 'fully_sequential']:
        for lm in ['B1RK', 'best']:
            key = f'{pn}|{lm}|_mix' if lm == 'best' else f'{pn}|B1RK|_mix'
            Fs = [F for F in sizes if res[F]['seeds'] >= MINS]
            if len(Fs) < 3: continue
            xs = np.log10([res[F]['learners'] for F in Fs]); Ms = [res[F][key] for F in Fs]
            ys = np.array([M.mean() for M in Ms]); w = 1 / np.array([M.var() for M in Ms])
            def wls(y):
                W = np.diag(w); A = np.c_[np.ones_like(xs), xs]; beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ y); return beta
            a, b = wls(ys); nstar = 10 ** ((MARGIN - a) / b)
            rng = np.random.default_rng(2028); bs_b = []; bs_n = []
            for _ in range(2000):
                rr = rng.integers(300); y = np.array([M[rng.integers(M.shape[0]), rr] for M in Ms]); aa, bb = wls(y)
                bs_b.append(bb); bs_n.append(10 ** ((MARGIN - aa) / bb) if bb < 0 else np.nan)
            bs_n = np.array(bs_n)
            below = [res[F]['learners'] for F in Fs if np.percentile(res[F][key], 95) < MARGIN]
            above = [res[F]['learners'] for F in Fs if np.percentile(res[F][key], 5) > MARGIN]
            TR[f'{pn}|{lm}'] = dict(intercept=float(a), slope_per_decade=float(b), slope_ci=np.percentile(bs_b, [2.5, 97.5]).tolist(),
                                    n_star=float(nstar), n_star_ci=np.nanpercentile(bs_n, [2.5, 97.5]).tolist(), n_star_undefined_share=float(np.isnan(bs_n).mean()),
                                    smallest_size_equivalent_below_margin=min(below) if below else None,
                                    largest_size_above_margin=max(above) if above else None, sizes_used=[res[F]['learners'] for F in Fs])
    OUT[vname] = dict(points={str(F): {k: v for k, v in r.items() if not k.endswith('_mix')} for F, r in res.items()}, trend=TR)
    for F, r in res.items():
        ROWS.append({'variant': vname, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items() if not k.endswith('_mix')}})
    OUT[vname]['_res'] = res
pd.DataFrame(ROWS).to_csv('regime2_table.csv', index=False)
# ---------------- XES3G5M (unchanged runs)
xr = []
for f in glob.glob(f'{XD}/f*_s*/out.json'):
    F, S = re.search(r'f([\d.]+)_s(\d+)', f).groups(); o = json.load(open(f))
    xr.append(dict(fraction=float(F), seed=int(S), learners=o['n_training_learners'], **{f'{pn}|{nm}': o['roles'][f'{pn}|B1RK'][k][0]
              for pn in ['snapshot', 'fully_sequential'] for nm, k in [('weighting', 'ESD-DKT_vs_base'), ('estimation_DKT', 'estimation_role_DKT')]}))
XS = pd.DataFrame(xr); XG = XS.groupby('fraction').agg(['mean', 'min', 'max']) if len(XS) else None
# ---------------- figure
INK = '#222222'; MUTED = '#6b6b6b'; GRID = '#e3e3e3'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex='col')
for i, pn in enumerate(['snapshot', 'fully_sequential']):
    ax = axs[i, 0]
    for vname, col, mk, lab, key in [('validation_whole', C1, 'o', 'Weighting role, logistic layer fitted on all validation learners', 'B1RK'),
                                     ('validation_subsampled', C3, '^', 'Weighting role, every layer fitted on the sampled learners', 'B1RK')]:
        res = OUT[vname]['_res']; Fs = sorted(F for F in res if res[F]['seeds'] >= MINS)
        if not Fs: continue
        x = np.array([res[F]['learners'] for F in Fs]); y = np.array([res[F][f'{pn}|B1RK|weighting'] for F in Fs]) * 1000
        ci = np.array([res[F][f'{pn}|B1RK|weighting_ci'] for F in Fs]) * 1000
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0); ax.plot(x, y, color=col, marker=mk, ms=4.5, lw=1.6, mec='white', mew=0.8, label=lab)
    res = OUT['validation_whole']['_res']; Fs = sorted(F for F in res if res[F]['seeds'] >= MINS and f'{pn}|B1RK|real_minus_random' in res[F])
    if Fs:
        x = np.array([res[F]['learners'] for F in Fs]); y = np.array([res[F][f'{pn}|B1RK|real_minus_random'] for F in Fs]) * 1000
        ax.plot(x, y, color=C4, marker='D', ms=4, lw=1.4, ls='--', mec='white', mew=0.8, label='Real minus topic- and exposure-matched random relations')
    Fs = sorted(F for F in res if res[F]['seeds'] >= MINS)
    x = np.array([res[F]['learners'] for F in Fs]); y = np.array([res[F][f'{pn}|B1RK|estimation_DKT'] for F in Fs]) * 1000
    ax.plot(x, y, color=C2, marker='s', ms=4, lw=1.4, mec='white', mew=0.8, label='Estimation role: GPE-DKT − DKT inside ESD')
    ax.set_title(f"Junyi, {pn.replace('_', ' ')}", fontsize=8.5, color=INK, loc='left')
    ax2 = axs[i, 1]
    if XG is not None:
        xg = XG.sort_values(('learners', 'mean')); xx = xg[('learners', 'mean')].values
        for nm, col, mk in [('weighting', C1, 'o'), ('estimation_DKT', C2, 's')]:
            y = xg[(f'{pn}|{nm}', 'mean')].values * 1000; lo = y - xg[(f'{pn}|{nm}', 'min')].values * 1000; hi = xg[(f'{pn}|{nm}', 'max')].values * 1000 - y
            ax2.errorbar(xx, y, yerr=[lo, hi], color=col, marker=mk, ms=4.5, lw=1.6, capsize=2, mec='white', mew=0.8)
    ax2.set_title(f"XES3G5M, {pn.replace('_', ' ')}", fontsize=8.5, color=INK, loc='left')
    for a_ in (ax, ax2):
        a_.axhline(0, color=MUTED, lw=0.8); a_.axhline(5, color=MUTED, lw=0.8, ls=':'); a_.set_xscale('log')
        a_.grid(axis='y', color=GRID, lw=0.6); a_.spines[['top', 'right']].set_visible(False)
        if i == 1: a_.set_xlabel('Training learners (log scale)')
    ax.set_ylabel('ΔAUC × 1000 over B1RK')
    if i == 0: ax.text(ax.get_xlim()[0] * 1.05, 5.3, '+0.005 margin', color=MUTED, fontsize=7)
    ax2.set_ylim(min(ax2.get_ylim()[0], -1.5), max(6, ax2.get_ylim()[1]))
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig('fig/Fig7_regime.png', dpi=300, bbox_inches='tight')
for v in OUT.values(): v.pop('_res', None)
json.dump(OUT, open('out_regime2.json', 'w'), indent=1, default=float)
for vname in OUT:
    print(vname, json.dumps(OUT[vname]['trend'], indent=None, default=float)[:1500])
