"""Study overview (Fig. 1) and method pipeline (Fig. 2) for the manuscript. Writes fig/Fig1_overview.png, fig/Fig2_pipeline.png (300 dpi)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
INK = '#222222'; MUTED = '#6b6b6b'
DATA = '#eef1f5'; METH = '#d9e6ef'; EVAL = '#f6ead9'; OUT = '#e4efe0'

def box(ax, x, y, w, h, text, fc, ec=INK, ls='-', fs=7.4, bold=False, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=fc, ec=ec, lw=0.9, ls=ls))
    tx = x + w / 2 if ha == 'center' else x + 0.008
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.32)
def arrow(ax, x0, y0, x1, y1, style='-|>', ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=INK, lw=0.9, ls=ls, shrinkA=0, shrinkB=0))

# ---------------- Fig. 1 study overview
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
cols = [(0.01, 'Data and relations'), (0.255, 'Method: two roles of the graph'), (0.53, 'Evaluation design'), (0.785, 'Questions and outputs')]
CW = 0.205
for x, t in cols:
    ax.text(x + CW / 2, 0.975, t, ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
# column 1: data
box(ax, 0.01, 0.70, CW, 0.23, 'Junyi Academy (primary)\n25.9 M attempts, 837 exercises\n1,131 crowd-rated pairs (0–9)\n978 expert-map edges', DATA)
box(ax, 0.01, 0.50, CW, 0.16, 'ASSISTments 2009, EdNet\nrelations mined from\ntraining-log order', DATA)
box(ax, 0.01, 0.32, CW, 0.14, 'XES3G5M\ncurriculum-tree relations', DATA)
box(ax, 0.01, 0.07, CW, 0.21, 'Positional split per learner\n60% training / 20% validation\n/ 20% held-out\ncase = first attempt\nat a new exercise', DATA)
# column 2: method
box(ax, 0.255, 0.80, CW, 0.13, 'Consensus admission\nQ ≥ 0.80 (Alg. 1, Eq. 4)', METH, bold=False)
box(ax, 0.255, 0.585, CW, 0.17, 'Estimators for an unattempted\nprerequisite (Section 3.2)\nZERO · POP · BKT · LKT\nDKT · SAKT · AKT', METH)
box(ax, 0.255, 0.32, CW, 0.215, 'Weighting role\nevidence-state decomposition\nD_obs, U_exp, counts\n(Eqs. 2–3; Alg. 2, 3)', METH)
box(ax, 0.255, 0.07, CW, 0.20, 'Estimation role\ngraph-propagated estimator:\nbase estimator + one-hop\nup/down evidence (Eq. 5)', METH)
arrow(ax, 0.3575, 0.80, 0.3575, 0.755)
arrow(ax, 0.3575, 0.585, 0.3575, 0.535)
ax.annotate('', xy=(0.255 + CW, 0.17), xytext=(0.255 + CW, 0.67), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9, connectionstyle='arc3,rad=-0.25'))
# column 3: evaluation
box(ax, 0.53, 0.80, CW, 0.13, 'Learner models\nB1 (accuracy, length, recency)\nB1RK (+ KT prediction)', EVAL)
box(ax, 0.53, 0.615, CW, 0.15, 'Update protocols\nsnapshot · prerequisite-\nsequential · fully sequential', EVAL)
box(ax, 0.53, 0.42, CW, 0.16, 'Controls\nrandom graphs\ntopic- and exposure-matched\nrandom relations', EVAL)
box(ax, 0.53, 0.235, CW, 0.15, 'Data regimes\n0.5–100% of training learners\n3 samples, fixed held-out cases', EVAL)
box(ax, 0.53, 0.07, CW, 0.13, 'Shared learner bootstrap (300)\nDeLong, Brier, alert budgets\nstatus labels P / H / R', EVAL)
# column 4: questions and outputs
box(ax, 0.785, 0.745, CW, 0.185, 'Q1  Which value for an\nunattempted prerequisite?\nzero vs estimated\n(Sections 5.1–5.2)', OUT)
box(ax, 0.785, 0.525, CW, 0.185, 'Q2  In which role does the\ngraph help: weighting or\nestimation?\n(Sections 5.3, 5.7)', OUT)
box(ax, 0.785, 0.305, CW, 0.185, 'Q3  When does it help?\ngain over KT vs training\nlearners; replication\n(Sections 5.6–5.7)', OUT)
box(ax, 0.785, 0.07, CW, 0.20, 'Decision rule and\nteacher-facing policy (6.4)\nevidence store (Table 1)\nreproducibility package', OUT)
arrow(ax, 0.8875, 0.305, 0.8875, 0.27)
for x0, x1 in ((0.01 + CW, 0.255), (0.255 + CW, 0.53), (0.53 + CW, 0.785)):
    ax.annotate('', xy=(x1 - 0.002, 0.865), xytext=(x0 + 0.004, 0.865), arrowprops=dict(arrowstyle='simple,head_width=0.9,head_length=0.7,tail_width=0.35', fc='#9aa5b1', ec='none'))
ax.text(0.5, 0.012, 'Pre-specified primary analysis on Junyi (P); post hoc analyses on the same held-out window (H); replication with the specification frozen on Junyi (R).',
        ha='center', fontsize=7.2, color=MUTED, style='italic')
fig.savefig('fig/Fig1_overview.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Fig. 2 method pipeline (updated with the graph-propagated estimator)
def box2(ax, x, y, w, h, text, fc, ec=INK, ls='-', fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02', fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.35)
fig, ax = plt.subplots(figsize=(10.5, 3.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
st = [('Stage 1\nIngestion and\npositional split', '#eef1f5', 'Junyi log:\n25.9 M attempts\n1,131 crowd pairs\n978 expert edges\n60/20/20 split per learner'),
      ('Stage 2\nConsensus and\nadmission\n(Alg. 1)', '#d9e6ef', 'μ* = mean of r/9\nQ: median-deviation\nagreement (Eq. 4)\nQ ≥ 0.80: 713 admitted,\n418 logged as rejected'),
      ('Stage 3\nEstimators for\nunattempted\nprerequisites', '#d9e6ef', 'ZERO · POP · BKT prior\ncold-start logistic KT\nDKT · SAKT · AKT\n+ GPE versions (Eq. 5)\n(training segments only)'),
      ('Stage 4\nEvidence states,\nESD scoring\n(Alg. 2, 3)', '#d9e6ef', 'unobserved / observed\nD_obs, U_exp, counts\nsnapshot, prerequisite-\nor fully sequential'),
      ('Stage 5\nEvaluation and\nreview card', '#eef1f5', 'logistic layer, ΔAUC\nshared learner bootstrap\nDeLong; alert budgets\nprototype card')]
w = 0.158; gap = (1 - 5 * w) / 6
for k, (t, fc, sub) in enumerate(st):
    x = gap + k * (w + gap)
    box2(ax, x, 0.55, w, 0.33, t, fc, fs=7.6, bold=True)
    box2(ax, x, 0.08, w, 0.36, sub, 'white', ec=MUTED, ls='--', fs=6.9)
    ax.plot([x + w / 2, x + w / 2], [0.55, 0.44], color=MUTED, lw=0.8)
    if k < 4: arrow(ax, x + w + 0.003, 0.715, x + w + gap - 0.003, 0.715)
ax.text(0.5, 0.965, 'Shaded boxes: stages formalised as algorithms or estimators. Dashed boxes: data and outputs (Junyi Academy counts).',
        ha='center', fontsize=7.5, color=MUTED)
ax.text(0.5, 0.01, 'Every stage writes to the evidence store (Table 1); a score is never stored on an edge.', ha='center', fontsize=7.5, color=MUTED, style='italic')
fig.savefig('fig/Fig2_pipeline.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('ok')
