"""Relation-specific part of the graph's gain and the N* table by configuration (Junyi; post hoc).

python data_regime_relation_summary.py JUNYI_REGIME_DIR XES_REGIME_DIR
Reads out3_vs.json (every layer fitted on the sampled learners) at every size and out3.json (logistic layers on all validation
learners) at 0.5, 2 and 5%; pooled intervals as in data_data_regime_summary_basic.py (seed mixture of re-centred learner-bootstrap
replicates; this is the uncertainty of one sample of learners, conservative for the mean of three).
Also collects N* for the eight configurations from out_regime2.json.
Outputs: out_regime3.json, nstar_table.csv, fig/Fig7_regime.png (three panels, snapshot protocol).
"""
import sys, json, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
JD, XD = sys.argv[1], sys.argv[2]
def load(name):
    P = {}
    for f in glob.glob(f'{JD}/f*_s*/{name}.json'):
        F, S = re.search(r'f([\d.]+)_s(\d+)', f).groups(); P.setdefault(float(F), {})[int(S)] = json.load(open(f))
    return P
def pooled(pts, kp, kb):
    d = np.array([kp(o) for o in pts]); B = np.array([kb(o) for o in pts]); M = d[:, None] + (B - B.mean(1, keepdims=True))
    return float(d.mean()), np.percentile(M, [2.5, 97.5]).tolist(), M
OUT = {'relation_specific': {}}
for name, vn in [('out3_vs', 'validation_subsampled'), ('out3', 'validation_whole')]:
    P = load(name); R = {}
    for F in sorted(P):
        if len(P[F]) < 3: continue
        pts = [P[F][s] for s in sorted(P[F])]; r = {'learners': int(pts[0]['n_training_learners'])}
        for pn in ['snapshot', 'fully_sequential']:
            for lm in ['B1RK', 'BEST']:
                X = lambda o: o['extended']
                m, ci, M = pooled(pts, lambda o: X(o)[f'{pn}|{lm}|real_minus_random_point'], lambda o: X(o)[f'{pn}|{lm}|boot_real_minus_random'])
                r[f'{pn}|{lm}|real_minus_random'] = m; r[f'{pn}|{lm}|real_minus_random_ci'] = ci
                r[f'{pn}|{lm}|random_vs_base'] = float(np.mean([np.mean([x[0] for x in X(o)[f'{pn}|{lm}|random_ESD-DKT_vs_base']]) for o in pts]))
                r[f'{pn}|{lm}|real_vs_base'] = float(np.mean([X(o)['best_learner_model'][pn]['weighting'][0] if lm == 'BEST' else o['roles'][f'{pn}|B1RK']['ESD-DKT_vs_base'][0] for o in pts]))
        R[F] = r
    OUT['relation_specific'][vn] = {str(k): v for k, v in R.items()}
# N* by configuration
R2 = json.load(open('out_regime2.json'))
rows = []
for vn, vlab in [('validation_whole', 'Logistic layers on all validation learners'), ('validation_subsampled', 'Every layer fitted on the sampled learners')]:
    for lm, llab in [('B1RK', 'over B1RK'), ('best', 'over the strongest learner model')]:
        row = {'configuration': f'{vlab}, {llab}'}
        for pn in ['snapshot', 'fully_sequential']:
            t = R2[vn]['trend'][f'{pn}|{lm}']; pts = R2[vn]['points']
            key = f'{pn}|B1RK|weighting_ci90' if lm == 'B1RK' else f'{pn}|best|weighting_ci90'
            above = [pts[F]['learners'] for F in pts if pts[F][key][0] > 0.005]
            below = [pts[F]['learners'] for F in pts if pts[F][key][1] < 0.005]
            smallest = min(pts[F]['learners'] for F in pts)
            row[f'{pn}|n_star'] = t['n_star']; row[f'{pn}|n_star_ci'] = t['n_star_ci']; row[f'{pn}|slope'] = t['slope_per_decade']; row[f'{pn}|slope_ci'] = t['slope_ci']
            row[f'{pn}|in_range'] = bool(t['n_star'] >= smallest)
            row[f'{pn}|largest_above'] = max(above) if above else None; row[f'{pn}|smallest_below'] = min(below) if below else None
        rows.append(row)
OUT['n_star_table'] = rows
pd.DataFrame(rows).to_csv('nstar_table.csv', index=False)
json.dump(OUT, open('out_regime3.json', 'w'), indent=1)
# ---------------- figure: (a) weighting role, two pipelines; (b) relation-specific part and estimation role; (c) XES3G5M
INK = '#222222'; MUTED = '#6b6b6b'; GRID = '#e3e3e3'; C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
pn = 'snapshot'
ax = axs[0]
for vn, col, mk, lab in [('validation_subsampled', C3, '^', 'Every layer fitted on the sampled learners (primary)'), ('validation_whole', C1, 'o', 'Logistic layers on all validation learners')]:
    pts = R2[vn]['points']; Fs = sorted(pts, key=float)
    x = np.array([pts[F]['learners'] for F in Fs]); y = np.array([pts[F][f'{pn}|B1RK|weighting'] for F in Fs]) * 1000
    ci = np.array([pts[F][f'{pn}|B1RK|weighting_ci'] for F in Fs]) * 1000
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0); ax.plot(x, y, color=col, marker=mk, ms=4.5, lw=1.6, mec='white', mew=0.8, label=lab)
pts = R2['validation_subsampled']['points']; Fs = sorted(pts, key=float)
x = np.array([pts[F]['learners'] for F in Fs]); y = np.array([pts[F][f'{pn}|best|weighting'] for F in Fs]) * 1000
ax.plot(x, y, color=C3, ls='--', marker='^', mfc='white', ms=4.5, lw=1.2, label='Primary pipeline, over the strongest learner model')
ax.set_title('(a) Junyi: graph gain over a KT learner model', fontsize=8.5, color=INK, loc='left')
ax = axs[1]
RS = OUT['relation_specific']['validation_subsampled']; Fs = sorted(RS, key=float)
x = np.array([RS[F]['learners'] for F in Fs])
for lm, col, mk, ls, lab in [('B1RK', C4, 'D', '-', 'Real minus matched random relations, over B1RK'), ('BEST', C4, 'D', '--', 'Real minus matched random, over the strongest model')]:
    y = np.array([RS[F][f'{pn}|{lm}|real_minus_random'] for F in Fs]) * 1000; ci = np.array([RS[F][f'{pn}|{lm}|real_minus_random_ci'] for F in Fs]) * 1000
    if lm == 'B1RK': ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0)
    ax.plot(x, y, color=col, marker=mk, ms=4, lw=1.5 if ls == '-' else 1.2, ls=ls, mec='white' if ls == '-' else col, mfc=col if ls == '-' else 'white', mew=0.8, label=lab)
pts = R2['validation_subsampled']['points']; Fs2 = sorted(pts, key=float)
x2 = np.array([pts[F]['learners'] for F in Fs2]); y2 = np.array([pts[F][f'{pn}|B1RK|estimation_DKT'] for F in Fs2]) * 1000
ax.plot(x2, y2, color=C2, marker='s', ms=4, lw=1.4, mec='white', mew=0.8, label='Estimation role: GPE-DKT − DKT inside ESD')
ax.set_title('(b) Junyi: relation-specific part; estimation role', fontsize=8.5, color=INK, loc='left')
ax = axs[2]
xr = []
for f in glob.glob(f'{XD}/f*_s*/out.json'):
    F, S = re.search(r'f([\d.]+)_s(\d+)', f).groups(); o = json.load(open(f))
    xr.append(dict(F=float(F), learners=o['n_training_learners'], w=o['roles'][f'{pn}|B1RK']['ESD-DKT_vs_base'][0], e=o['roles'][f'{pn}|B1RK']['estimation_role_DKT'][0]))
XS = pd.DataFrame(xr).groupby('F').agg(['mean', 'min', 'max']).sort_values(('learners', 'mean'))
xx = XS[('learners', 'mean')].values
for nm, col, mk, lab in [('w', C1, 'o', 'Weighting role'), ('e', C2, 's', 'Estimation role')]:
    y = XS[(nm, 'mean')].values * 1000
    ax.errorbar(xx, y, yerr=[y - XS[(nm, 'min')].values * 1000, XS[(nm, 'max')].values * 1000 - y], color=col, marker=mk, ms=4.5, lw=1.6, capsize=2, mec='white', mew=0.8, label=lab)
ax.set_title('(c) XES3G5M (counter-example)', fontsize=8.5, color=INK, loc='left')
ax.legend(frameon=False, fontsize=7, loc='upper right')
for a_ in axs:
    a_.axhline(0, color=MUTED, lw=0.8); a_.axhline(5, color=MUTED, lw=0.8, ls=':'); a_.set_xscale('log')
    a_.grid(axis='y', color=GRID, lw=0.6); a_.spines[['top', 'right']].set_visible(False); a_.set_xlabel('Training learners (log scale)')
    a_.set_ylim(-1.5, 10)
axs[0].set_ylabel('ΔAUC × 1000 (snapshot protocol)'); axs[0].text(700, 5.3, '+0.005 margin', color=MUTED, fontsize=7)
h, l = [], []
for a_ in axs[:2]:
    hh, ll = a_.get_legend_handles_labels(); h += hh; l += ll
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.42, -0.1))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig('fig/Fig7_regime.png', dpi=300, bbox_inches='tight')
for vn, R in OUT['relation_specific'].items():
    for F, r in sorted(R.items(), key=lambda kv: float(kv[0])):
        print(vn, r['learners'], *[f"{k.split('|')[0][:4]}|{k.split('|')[1]} {r[k]:+.4f} [{r[k+'_ci'][0]:+.4f},{r[k+'_ci'][1]:+.4f}] rnd {r[k.replace('real_minus_random','random_vs_base')]:+.4f} real {r[k.replace('real_minus_random','real_vs_base')]:+.4f}" for k in r if k.endswith('real_minus_random')], sep=' | ')
for row in rows: print(row)
