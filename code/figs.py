import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
# Fig 2 fidelity-coverage
G=pd.read_csv('out_gate_heldout.csv')
fig,ax=plt.subplots(figsize=(6.2,3.6))
ax.plot(G.q,G.R2,'o-',color='#1f3b73',label='Retained (admitted) relations, agreement gate')
ax.fill_between(G.q,G.rand_lo,G.rand_hi,color='#999',alpha=.3,label='Size-matched random subsets (95% range)')
ax.plot(G.q,G.freqmatched_mean,'s--',color='#b35900',ms=4,label='Frequency-matched subsets (mean)')
for q,r,n in zip(G.q,G.R2,G.retained): ax.annotate(f'{n} retained',(q,r),textcoords='offset points',xytext=(-14,8),fontsize=8)
ax.set_xlabel('Admission threshold $Q_{min}$'); ax.set_ylabel('Held-out $r^2$ with co-occurrence, retained relations only')
ax.legend(fontsize=8,loc='upper left',frameon=False); ax.set_ylim(0.15,0.72)
fig.tight_layout(); fig.savefig('fig/fig2.png',dpi=300)
# Fig 3 learner-level dAUC with CIs
rows=[('Candidate set, unweighted',0.0041,0.0035,0.0046),('Binary expert graph',0.0024,0.0021,0.0027),
('Interaction-derived (transition)',0.0010,0.0007,0.0012),('Permuted weights (control)',0.0002,0.0001,0.0003),
('Ungated graded graph',-0.0001,-0.0001,0.0000),('FPKG (gated, evidence score)',-0.0001,-0.0001,0.0000)]
fig,ax=plt.subplots(figsize=(6.2,2.9))
y=np.arange(len(rows))[::-1]
for yi,(n,d,l,h) in zip(y,rows):
    ax.errorbar(d,yi,xerr=[[d-l],[h-d]],fmt='o',color='#1f3b73' if 'FPKG' in n or 'graded' in n else '#555',capsize=3)
ax.axvline(0,color='k',lw=.8); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('Held-out ΔAUC vs. confounder-only model (95% CI)')
fig.tight_layout(); fig.savefig('fig/fig3.png',dpi=300)
# Fig 1 temporal protocol
fig,ax=plt.subplots(figsize=(7.0,2.4)); ax.axis('off')
def box(x,w,y,c,t,fs=7.5): ax.add_patch(plt.Rectangle((x,y),w,0.5,color=c,alpha=.9)); ax.text(x+w/2,y+0.25,t,ha='center',va='center',fontsize=fs,color='white')
box(0,6,1.1,'#1f3b73','History: first 60% of each learner\'s events\n(competency, transition baseline, frequency, difficulty)')
box(6,2,1.1,'#b35900','Validation\n60–80%'); box(8,2,1.1,'#7a7a7a','Held-out\n80–100%')
box(0,8,0.2,'#3b5f9e','History for the test decision point: first 80%'); box(8,2,0.2,'#555','Future outcomes')
ax.text(0,1.68,'Stage 1: parameter and threshold selection (validation outcomes only)',fontsize=8)
ax.text(0,0.78,'Stage 2: final evaluation, run once with frozen settings',fontsize=8)
ax.set_xlim(-0.05,10.05); ax.set_ylim(0.1,1.85)
fig.tight_layout(); fig.savefig('fig/fig1.png',dpi=300)
