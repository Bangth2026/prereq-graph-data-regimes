"""Summary of the data-regime study (Junyi and XES3G5M): table (out_regime.json, regime_summary.csv) and Fig. 7.

python data_regime_summary_basic.py JUNYI_REGIME_DIR XES_REGIME_DIR
Each point folder (f{F}_s{SEED}) holds out.json from graph_estimator.py / graph_estimator_replication.py. Values are means over seeds with the range
over seeds; the 95% intervals of single runs are learner-bootstrap intervals with the fitted models fixed.
"""
import sys, json, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows = []
for ds, d in [('Junyi', sys.argv[1]), ('XES3G5M', sys.argv[2])]:
    for f in glob.glob(f'{d}/f*_s*/out.json'):
        F, S = re.search(r'f([\d.]+)_s(\d+)', f).groups(); o = json.load(open(f))
        a = o['accuracy']; r = {'dataset': ds, 'fraction': float(F), 'seed': int(S), 'learners': o['n_training_learners']}
        for k in ['POP', 'LKT', 'DKT']:
            r[f'{k}_auc'] = a[k]['auc']; r[f'{k}_brier'] = a[k]['brier']; r[f'GPE-{k}_auc'] = a['G' + k]['auc']; r[f'GPE-{k}_brier'] = a['G' + k]['brier']
        for pn in ['snapshot', 'fully_sequential']:
            for base in ['B1', 'B1RK']:
                x = o['roles'][f'{pn}|{base}']; tag = f'{pn}|{base}'
                r[f'{tag}|auc'] = x['auc']
                for nm, key in [('weighting', 'ESD-DKT_vs_base'), ('estimation_DKT', 'estimation_role_DKT'), ('estimation_POP', 'estimation_role_POP'), ('estimation_LKT', 'estimation_role_LKT')]:
                    r[f'{tag}|{nm}'], r[f'{tag}|{nm}_lo'], r[f'{tag}|{nm}_hi'] = x[key]
        rows.append(r)
D = pd.DataFrame(rows).sort_values(['dataset', 'fraction', 'seed'])
D.to_csv('regime_runs.csv', index=False)
num = [c for c in D.columns if c not in ('dataset', 'seed', 'fraction')]
G = D.groupby(['dataset', 'fraction'])
S = G[num].mean().add_suffix('').reset_index(drop=False)
mn = G[num].min().add_suffix('_min').reset_index(drop=True); mx = G[num].max().add_suffix('_max').reset_index(drop=True)
S = pd.concat([S, mn, mx], axis=1); S['seeds'] = G.size().values
S.to_csv('regime_summary.csv', index=False)
json.dump({'runs': rows}, open('out_regime.json', 'w'), indent=1, default=float)
# ---------------- Fig. 7
INK = '#222222'; MUTED = '#6b6b6b'; GRID = '#e3e3e3'; CW = '#1e6fb8'; CE = '#c46b2b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex='col')
for j, ds in enumerate(['Junyi', 'XES3G5M']):
    for i, pn in enumerate(['snapshot', 'fully_sequential']):
        ax = axs[i, j]; Sd = S[S.dataset == ds].sort_values('learners'); x = Sd.learners.values
        if len(Sd) == 0: continue
        for nm, col, lab, mk in [('weighting', CW, 'Weighting role: ESD-DKT over B1RK', 'o'), ('estimation_DKT', CE, 'Estimation role: GPE-DKT − DKT (in ESD, over B1RK)', 's')]:
            k = f'{pn}|B1RK|{nm}'; y = Sd[k].values * 1000
            lo = (Sd[k] - Sd[k + '_min']).values * 1000; hi = (Sd[k + '_max'] - Sd[k]).values * 1000
            ax.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=4.5, lw=1.6, capsize=2, label=lab, mec='white', mew=0.8)
        ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(5, color=MUTED, lw=0.8, ls=':')
        ax.set_xscale('log'); ax.grid(axis='y', color=GRID, lw=0.6); ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f"{ds}, {pn.replace('_', ' ')}", fontsize=8.5, color=INK, loc='left')
        if j == 0: ax.set_ylabel('ΔAUC × 1000')
        if i == 1: ax.set_xlabel('Training learners (log scale)')
        if i == 0 and j == 0: ax.text(x.min(), 5.25, '+0.005 margin', color=MUTED, fontsize=7)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig('fig/Fig7_regime.png', dpi=300, bbox_inches='tight')
pd.set_option('display.width', 250)
print(S[['dataset', 'fraction', 'learners', 'seeds', 'DKT_auc', 'GPE-DKT_auc', 'POP_brier', 'GPE-POP_brier', 'snapshot|B1RK|auc',
         'snapshot|B1RK|weighting', 'snapshot|B1RK|estimation_DKT', 'fully_sequential|B1RK|weighting', 'fully_sequential|B1RK|estimation_DKT']].round(4).to_string())
