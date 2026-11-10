"""Figures 3-6 and 8 of the paper (and the earlier pipeline drawing). Reads out_benchmark.json, out_card.json, worked_example.json (300 dpi)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.labelcolor': '#222222', 'xtick.color': '#444444', 'ytick.color': '#444444'})
INK = '#222222'; MUTED = '#6b6b6b'; GRID = '#e3e3e3'
C1 = '#2a6f97'; C2 = '#c46b2b'; C3 = '#7a7a7a'          # blue, orange, gray (CVD-separable, print-safe)
d = json.load(open('out_benchmark.json'))

def box(ax, x, y, w, h, text, fc, ec=INK, ls='-', fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02', fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.35)
def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.0))

# ---------------- F1 pipeline
fig, ax = plt.subplots(figsize=(10.5, 3.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
st = [('Stage 1\nIngestion and\npositional split', '#eef1f5', 'Junyi log:\n25.9 M attempts\n1,131 crowd pairs\n978 expert edges\n60/20/20 split per learner'),
      ('Stage 2\nConsensus and\nadmission\n(Alg. 1)', '#d9e6ef', 'μ* = mean of r/9\nQ: median-deviation\nagreement (Eq. 4)\nQ ≥ 0.80: 713 admitted,\n418 logged as rejected'),
      ('Stage 3\nEstimators for\nunattempted\nprerequisites', '#d9e6ef', 'ZERO · POP · BKT prior\ncold-start logistic KT\nDKT · SAKT · AKT\n(training segments only)'),
      ('Stage 4\nEvidence states,\nESD scoring\n(Alg. 2, 3)', '#d9e6ef', 'unobserved / observed\nD_obs, U_exp, counts\nsnapshot, prerequisite-\nor fully sequential'),
      ('Stage 5\nEvaluation and\nreview card', '#eef1f5', 'logistic layer, ΔAUC\nshared learner bootstrap\nDeLong; alert budgets\nprototype card')]
w = 0.158; gap = (1 - 5 * w) / 6
for k, (t, fc, sub) in enumerate(st):
    x = gap + k * (w + gap)
    box(ax, x, 0.55, w, 0.33, t, fc, fs=7.6, bold=True)
    box(ax, x, 0.08, w, 0.36, sub, 'white', ec=MUTED, ls='--', fs=6.9)
    ax.plot([x + w / 2, x + w / 2], [0.55, 0.44], color=MUTED, lw=0.8)
    if k < 4: arrow(ax, x + w + 0.003, 0.715, x + w + gap - 0.003, 0.715)
ax.text(0.5, 0.965, 'Shaded boxes: stages formalised as algorithms or estimators. Dashed boxes: data and outputs (Junyi Academy counts).',
        ha='center', fontsize=7.5, color=MUTED)
ax.text(0.5, 0.01, 'Every stage writes to the evidence store (Table 1); a score is never stored on an edge.', ha='center', fontsize=7.5, color=MUTED, style='italic')
fig.savefig('fig/Fig2_pipeline_v1.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- F2 attrition
fig, ax = plt.subplots(figsize=(9.5, 5.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
L = '#eef1f5'; R = '#fbf3df'; G = '#e4efe4'
box(ax, 0.02, 0.82, 0.42, 0.14, 'Full log\n25,925,992 attempts · 247,606 learners · 722 exercises', L)
box(ax, 0.02, 0.57, 0.42, 0.14, 'Eligible learners (≥ 10 attempts)\n25,419,475 attempts · 124,882 learners', L)
box(ax, 0.02, 0.30, 0.42, 0.16, 'Validation window (60–80 %)\n378,145 cases, all targets\n182,160 new-target cases · 65,482 learners', L)
box(ax, 0.02, 0.03, 0.42, 0.16, 'Held-out window (80–100 %)\n370,826 cases, all targets\n163,049 new-target cases · 61,347 learners\nfailure rate 0.394', G)
arrow(ax, 0.23, 0.82, 0.23, 0.71); arrow(ax, 0.23, 0.57, 0.23, 0.46); arrow(ax, 0.23, 0.30, 0.23, 0.19)
ax.text(0.245, 0.765, 'drop learners with < 10 attempts (−122,724)', fontsize=7, color=MUTED, va='center')
ax.text(0.245, 0.515, 'positional split per learner; ≥ 5 history attempts;\ntarget never attempted before; has a crowd relation', fontsize=7, color=MUTED, va='center')
box(ax, 0.55, 0.82, 0.43, 0.14, 'Expert map: 988 assertions\n→ 981 distinct pairs → 978 edges (acyclic)', R)
box(ax, 0.55, 0.57, 0.43, 0.14, 'Crowd ratings: 1,131 pairs, 5–15 raters\n713 admitted (Q ≥ 0.80) · 418 rejected\n+ 823 pairs, 3–4 raters (replication only)', R)
box(ax, 0.55, 0.30, 0.43, 0.16, 'Case–prerequisite pairs, held-out window\n503,188 admitted / 791,326 all\nunobserved: 76.1 % at window start,\n65.1 % with sequential update', R)
box(ax, 0.55, 0.03, 0.43, 0.16, 'Learner-level evaluation\n163,049 cases, unit = case,\nclustered by learner (bootstrap over learners)', G)
arrow(ax, 0.765, 0.57, 0.765, 0.46); arrow(ax, 0.765, 0.30, 0.765, 0.19); arrow(ax, 0.44, 0.11, 0.55, 0.11)
fig.savefig('fig/Fig3_attrition.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- F3 composition + estimator quality
fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
comp = {'Snapshot': (76.1, 4.3, 19.6), 'Sequential\nupdate': (65.1, 6.3, 28.7)}
labs = ['Unobserved', 'Observed, shortfall', 'Observed, no shortfall']; cols = ['#3d3d3d', C2, '#cfcfcf']
for k, (nm, v) in enumerate(comp.items()):
    b = 0
    for j, x in enumerate(v):
        ax.bar(k, x, bottom=b, width=0.55, color=cols[j], edgecolor='white', linewidth=1.5, label=labs[j] if k == 0 else None)
        if x > 8: ax.text(k, b + x / 2, f'{x:.1f}%', ha='center', va='center', color='white' if j == 0 else INK, fontsize=8)
        b += x
ax.set_xticks([0, 1]); ax.set_xticklabels(comp.keys()); ax.set_ylabel('Share of case–prerequisite pairs (%)'); ax.set_ylim(0, 100)
ax.set_title('(a) Evidence state, 713 admitted relations', fontsize=9, loc='left')
ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
ax = axs[1]
cal = d['estimator_calibration_unobserved']; ks = ['POP', 'BKT', 'LKT', 'AKT', 'DKT', 'SAKT']
auc = [cal[k]['auc'] for k in ks]; bri = [cal[k]['brier'] for k in ks]
y = np.arange(len(ks))
ax.hlines(y, 0.5, auc, color=GRID, lw=2); ax.plot(auc, y, 'o', color=C1, ms=7)
for yy, a, b in zip(y, auc, bri): ax.text(a + 0.006, yy, f'AUC {a:.3f} · Brier {b:.3f}', va='center', fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(['Population prior', 'BKT prior P(L0)', 'Cold-start logistic KT', 'AKT', 'DKT', 'SAKT'])
ax.set_xlim(0.5, 0.92); ax.set_xlabel(f'AUC for the first answer on an unattempted prerequisite\n(ZERO: AUC 0.5, Brier {cal["ZERO"]["brier"]:.3f})', fontsize=8)
ax.set_title(f'(b) Estimators on {cal["DKT"]["n"]:,} later-attempted pairs', fontsize=9, loc='left')

ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig/Fig4_evidence_estimators.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- F5 estimator x use (snapshot, fully sequential): B5 and B9 increments
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
E = d['E1_E3_estimators_protocols']; ests = ['ZERO', 'POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']
for ax, pn, tt in [(axs[0], 'snapshot', '(a) Snapshot'), (axs[1], 'fully_sequential', '(b) Fully sequential')]:
    for j, (g, col, mk, nm) in enumerate([('B5', C1, 'o', 'Graded weights (B5)'), ('B9', C3, 's', 'Unweighted list (B9)')]):
        for i, k in enumerate(ests):
            for u, (kind, off) in enumerate([('coupled', -0.22), ('esd', 0.22)]):
                v = E[pn][f'{k}|{kind}|{g}']; x = i + off + (j - 0.5) * 0.12
                ax.errorbar(x, v[0] * 1000, yerr=[[1000 * (v[0] - v[1])], [1000 * (v[2] - v[0])]], fmt=mk, ms=5,
                            color=col, mfc=col if kind == 'esd' else 'white', mec=col, lw=1, capsize=0,
                            label=(nm + (', ESD' if kind == 'esd' else ', pooled Eq. 1')) if (i == 0 and pn == 'snapshot') else None)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(5, color=MUTED, lw=0.6, ls=':')
    ax.set_xticks(range(len(ests))); ax.set_xticklabels(ests, fontsize=8); ax.set_title(tt, fontsize=9, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('ΔAUC × 1000 over own baseline')
axs[1].text(-0.45, 5.3, '+0.005', fontsize=7, color=MUTED, ha='left')
fig.legend(loc='lower center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig('fig/Fig5_estimator_benchmark.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- F6 learner-model ladder
fig, ax = plt.subplots(figsize=(7.5, 3.0))
E2 = d['E2_stronger_learner_models']; E2b = d['E2b_attention_kt']
labels = ['Confounders (B1)', '+ recency (B1R)', '+ recency + DKT', '+ recency + SAKT', '+ recency + AKT', 'best KT + prerequisite graph (ESD)']
for j, (pn, col, nm) in enumerate([('snapshot', C3, 'Snapshot'), ('fully_sequential', C1, 'Fully sequential')]):
    vals = [E2[pn]['B1_auc'], E2[pn]['B1R_auc'], E2[pn]['B1RK_auc'], E2b[pn]['B1RK_SAKT_auc'], E2b[pn]['B1RK_AKT_auc']]
    kbest = int(max(range(2, 5), key=lambda i: vals[i])); kt = ['DKT', 'SAKT', 'AKT'][kbest - 2]
    add = E2[pn]['ZERO|esd|B5|total_vs_B1RK'][0] if kt == 'DKT' else E2b[pn][f'{kt}|ESD-ZERO|B5_total_vs_B1RK'][0]
    vals.append(vals[kbest] + add)
    yy = np.arange(len(vals)) + (j - 0.5) * 0.3
    ax.barh(yy, np.array(vals) - 0.74, left=0.74, height=0.28, color=col, label=nm)
    for a_, b_ in zip(yy, vals): ax.text(b_ + 0.0008, a_, f'{b_:.4f}', va='center', fontsize=7, color=INK)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.invert_yaxis(); ax.set_xlim(0.74, 0.825); ax.set_xlabel('Held-out AUC (first-attempt failure on a new exercise), Junyi')
ax.legend(frameon=False, fontsize=7.5, loc='upper right'); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig/Fig6_learner_models.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- F4 teacher card
cd = json.load(open('out_card.json')); rd = {r['prereq']: r for r in json.load(open('worked_example.json'))['rows']}
dirs = {'division_0.9': 'disputed (0.04)', 'division_1': 'disputed (0.02)', 'multiplication_0.5': 'supported (0.97)', 'multiplication_1': 'supported (0.96)',
        'multiplication_1.5': 'disputed (0.12)', 'measuring_segments': 'disputed (0.25)', 'perimeter_1': 'disputed (0.32)'}
order = ['division_0.9', 'division_1', 'multiplication_0.5', 'multiplication_1', 'multiplication_1.5', 'measuring_segments', 'perimeter_1']
cr_ = {r['prereq']: r for r in cd['rows']}
fig, ax = plt.subplots(figsize=(11.5, 3.1)); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
hdr = ['Prerequisite', 'μ*', 'Q', 'Status', 'Direction (log share)', 'Att.', 'θ obs.', 'POP / BKT / LKT / DKT', 'Evidence state → flag']
xs = [0.005, 0.165, 0.21, 0.255, 0.33, 0.49, 0.525, 0.6, 0.785]
ax.text(0.005, 0.95, f"Target: division_0.5   ·   Learner 167   ·   held-out window start   ·   DKT P(correct on target) = {cd['dkt_target']:.2f}",
        fontsize=8.5, fontweight='bold', color=INK)
for x, h in zip(xs, hdr): ax.text(x, 0.84, h, fontsize=7.8, fontweight='bold', color=INK)
ax.plot([0.005, 0.995], [0.815, 0.815], color=INK, lw=0.8)
for i, p in enumerate(order):
    r = rd[p]; e = cr_[p]; yv = 0.75 - i * 0.085
    unobs = r['attempts'] == 0; rej = r['status'] == 'rejected'
    col = MUTED if rej else INK; style = 'italic' if rej else 'normal'; wt = 'bold' if (unobs and not rej) else 'normal'
    th = '—' if unobs else f"{r['pc']:.2f}"
    est = f"{e['pop']:.2f} / {e['bkt_L0']:.2f} / {e['lkt']:.2f} / {e['dkt']:.2f}" if unobs else '(observed)'
    flag = 'hold (rejected, Q < 0.80)' if rej else ('unobserved → diagnose' if unobs else 'observed, no shortfall → none')
    cells = [p, f"{r['mu']:.2f}", f"{r['Q']:.2f}", 'rejected' if rej else 'admitted', dirs[p], str(r['attempts']), th, est, flag]
    for x, cval in zip(xs, cells): ax.text(x, yv, cval, fontsize=7.8, color=col, style=style, fontweight=wt)
ax.text(0.005, 0.07, 'Bold: admitted but never attempted before the window; under θ = 0 these two give the largest contributions (0.89, 0.85). Italic grey: rejected relation.',
        fontsize=7, color=MUTED)
ax.text(0.005, 0.015, 'Estimator values are computed outputs (package: out_card.json, worked_example.json). Flags follow the proposed policy (Table 9) and are not evaluated.',
        fontsize=7, color=MUTED)
fig.savefig('fig/Fig8_teacher_card.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('figures written')
