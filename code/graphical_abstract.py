"""Graphical abstract (EAAI): problem, two roles of the graph, and the data-regime curve (primary configuration)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK = '#222222'; MUTED = '#6b6b6b'; C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
R2 = json.load(open('out_regime2.json'))['validation_subsampled']['points']
R3 = json.load(open('out_regime3.json'))['relation_specific']['validation_subsampled']
fig = plt.figure(figsize=(13, 4.8))
ax = fig.add_axes([0, 0, 0.56, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def box(x, y, w, h, t, fc, fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.02', fc=fc, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.35)
def arr(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1))
ax.text(0.03, 0.93, 'Prerequisite graph + learner model: what is an unattempted prerequisite worth?', fontsize=11, fontweight='bold', color=INK)
box(0.03, 0.56, 0.28, 0.28, 'Junyi Academy, 25.9 M attempts\n65–76% of candidate\nprerequisites never attempted\n\nZero imputation →\nscore counts non-exposure', '#eef1f5')
box(0.38, 0.62, 0.27, 0.2, 'Graph as estimator\ngraph-propagated estimator\n→ no gain beyond recalibration', '#f6ead9')
box(0.38, 0.33, 0.27, 0.24, 'Graph as weight\nevidence-state decomposition\n→ small gain, half relation-\nspecific, shrinks with data', '#d9e6ef')
box(0.71, 0.33, 0.27, 0.49, 'Decision rule\n\nKeep the graph in the score\nonly for an informative graph\nand few training learners\n(crossing near 950\nlearners on Junyi)\n\nOtherwise predict with\nknowledge tracing; use the\ngraph to explain and audit', '#e4efe0')
arr(0.31, 0.72, 0.38, 0.72); arr(0.31, 0.62, 0.38, 0.47); arr(0.65, 0.72, 0.71, 0.66); arr(0.65, 0.45, 0.71, 0.5)
ax.text(0.03, 0.17, 'Seven estimators (zero, population prior, BKT, logistic KT, DKT, SAKT, AKT) · four public logs ·\nlearners subsampled from 0.5% to 100% · matched random-relation controls · open code and outputs',
        fontsize=8.5, color=MUTED, linespacing=1.4)
a2 = fig.add_axes([0.62, 0.17, 0.35, 0.68])
Fs = sorted(R2, key=float); x = np.array([R2[F]['learners'] for F in Fs])
y = np.array([R2[F]['snapshot|B1RK|weighting'] for F in Fs]) * 1000; ci = np.array([R2[F]['snapshot|B1RK|weighting_ci'] for F in Fs]) * 1000
a2.fill_between(x, ci[:, 0], ci[:, 1], color=C3, alpha=0.18, lw=0); a2.plot(x, y, color=C3, marker='^', lw=2, ms=6, mec='white', label='Graph gain over knowledge tracing')
Fr = sorted(R3, key=float); xr = np.array([R3[F]['learners'] for F in Fr]); yr = np.array([R3[F]['snapshot|B1RK|real_minus_random'] for F in Fr]) * 1000
a2.plot(xr, yr, color=C4, marker='D', lw=1.6, ms=5, mec='white', ls='--', label='of which relation-specific')
a2.axhline(5, color=MUTED, ls=':', lw=1); a2.axhline(0, color=MUTED, lw=0.8); a2.set_xscale('log'); a2.set_ylim(-1, 10)
a2.text(700, 5.3, '+0.005 margin', color=MUTED, fontsize=8)
a2.set_xlabel('Training learners (log scale)'); a2.set_ylabel('ΔAUC × 1000'); a2.spines[['top', 'right']].set_visible(False)
a2.legend(frameon=False, fontsize=8.5, loc='upper right'); a2.set_title('Junyi, every layer trained on the subsample', fontsize=9, loc='left', color=INK)
fig.savefig('fig/Graphical_abstract.png', dpi=300)
