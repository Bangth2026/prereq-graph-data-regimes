import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, json
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})
# Fig 1 pipeline
fig,ax=plt.subplots(figsize=(7.2,2.3)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(1.3,4.4)
steps=[('Source ratings','Expert: 988 assertions\n→ 981 distinct\n→ 978 edges\nCrowd: 1,131 pairs,\n5–15 raters'),
('Admission &\nprovenance','Q ≥ 0.80\n713 admitted\n418 rejected\nall logged with\nratings and repairs'),
('Relation\nweight μ*','Mean of rescaled\nratings, in (0, 1]'),
('Learner\ncompetency θ','Mastered / deficit /\nno-observation;\nestimator recorded'),
('Gap score','max(0, μ* − θ)\nfor observed\nprerequisites'),
('Teacher\nreview','Candidate priorities,\nprovenance and\nstability flags')]
xs=np.linspace(0.1,10.2,6); w=1.75
for i,(t,d) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((xs[i],3.1),w,1.2,boxstyle='round,pad=0.05',fc='#1f3b73' if i!=5 else '#b35900',ec='none'))
    ax.text(xs[i]+w/2,3.7,t,ha='center',va='center',color='white',fontsize=8.5,weight='bold')
    ax.text(xs[i]+w/2,2.9,d,ha='center',va='top',fontsize=6.8)
    if i<5: ax.annotate('',xy=(xs[i+1]-0.02,3.7),xytext=(xs[i]+w+0.02,3.7),arrowprops=dict(arrowstyle='->',lw=1.2))
fig.savefig('fig/f1.png',dpi=300,bbox_inches='tight')
# Fig 5 (manuscript): evidence states and learner-level increments under three specifications
f=json.load(open('analysis_f.json')); m=json.load(open('out_main_new.json'))['contrasts']
b1=json.load(open('analysis_b.json'))['missingness_aware']; e3=json.load(open('analysis_e.json'))['three_state']
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0),gridspec_kw=dict(width_ratios=[1,1.35]))
ax=axs[0]; names=[('binary_expert','Binary\nexpert (B3)'),('graded_ungated','Graded\nungated (B5)'),('fpkg','FPKG\n(B6)')]
x=np.arange(3); bottoms=np.zeros(3)
for key,lab,col in [('share_unobserved','No observation','#b35900'),('share_observed_deficit','Observed deficit','#1f3b73'),('share_observed_mastered','Observed mastered','#9aa5b8')]:
    v=np.array([f[k][key] for k,_ in names]); ax.bar(x,v,bottom=bottoms,color=col,label=lab,width=0.6); bottoms+=v
for i,(k,_) in enumerate(names): ax.text(i,1.02,f"{f[k]['gps_mass_share_unobserved']*100:.0f}%",ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([n for _,n in names],fontsize=7.5); ax.set_ylim(0,1.1); ax.set_ylabel('Share of candidate relations',fontsize=8)
ax.set_title('(a) Evidence state at the decision point',fontsize=8.5); ax.legend(fontsize=6.5,loc='lower center',bbox_to_anchor=(0.5,-0.48),ncol=3,frameon=False)
ax=axs[1]; specs=[('Pre-specified primary (θ = 0)',{k:m[k+'-BASE'] for k in ['BIN_EXP','ALL1','UNG','FPKG']}),
 ('Post hoc: count-adjusted',{k:b1[k] for k in ['BIN_EXP','ALL1','UNG','FPKG']}),
 ('Post hoc: three-state',{k:e3[k] for k in ['BIN_EXP','ALL1','UNG','FPKG']})]
mods=[('BIN_EXP','B3 binary','#555'),('ALL1','B9 unweighted','#999'),('UNG','B5 graded','#1f3b73'),('FPKG','B6 FPKG','#6b8cc7')]
for si,(sn,vals) in enumerate(specs):
    for mi,(k,lab,col) in enumerate(mods):
        d,lo,hi=vals[k]; y=si*5+mi
        ax.errorbar(d,-y,xerr=[[d-lo],[hi-d]],fmt='o',ms=3.5,capsize=2,color=col,label=lab if si==0 else None)
    ax.text(-0.0012,-si*5+1.0,sn,fontsize=7,weight='bold',va='center')
ax.axvline(0,color='k',lw=.6); ax.axvline(0.005,color='k',lw=.6,ls=':'); ax.text(0.0052,1.3,'±0.005 margin',fontsize=6.5)
ax.set_yticks([]); ax.set_xlim(-0.0013,0.0072); ax.set_ylim(-14,2.2); ax.set_xlabel('Held-out ΔAUC (95% CI)',fontsize=8)
ax.set_title('(b) Learner-level increments by specification',fontsize=8.5,pad=4); ax.legend(fontsize=6.5,loc='lower center',bbox_to_anchor=(0.5,-0.42),ncol=4,frameon=False)
fig.tight_layout(); fig.savefig('fig/f4.png',dpi=300,bbox_inches='tight')
# Fig 5 teacher card mock-up (real case, learner 167, target division_0.5)
d=json.load(open('worked_example.json'))['rows']
fig,ax=plt.subplots(figsize=(8.6,3.9)); ax.axis('off'); ax.set_xlim(0,12.2); ax.set_ylim(0,10)
ax.add_patch(FancyBboxPatch((0.1,0.1),11.9,9.7,boxstyle='round,pad=0.05',fc='#f6f7fa',ec='#1f3b73',lw=1.2))
ax.text(0.4,9.25,'Target: division_0.5   ·   Learner #167   ·   Estimator: proportion correct (BKT shown for comparison)',fontsize=8,weight='bold')
hdr=['Prerequisite','μ*','Q','Ratings (0–9)','Status','Attempts','θ (PC / BKT)','Evidence','Card note']
cx=[0.3,2.45,3.05,3.65,5.55,6.45,7.55,8.95,10.45]
for x,h in zip(cx,hdr): ax.text(x,8.5,h,fontsize=6.8,weight='bold')
yy=7.8
for r in sorted(d,key=lambda r:-float(r['mu'])):
    att=int(r['attempts']); st=r['status']
    state='no observation' if att==0 else ('mastered' if (r['pc'] not in (None,'None') and float(r['pc'])>=float(r['mu'])) else 'deficit')
    note={'rejected':'raters disagree:\nreview relation','no observation':'check before\nnew material'}.get(st if st=='rejected' else state,'no action')
    if st=='rejected' and att==0: note='raters disagree;\nnot scored'
    pc='—' if att==0 else f"{float(r['pc']):.2f}"
    col='#b35900' if st=='rejected' else ('#1f3b73' if state=='no observation' else '#333')
    vals=[r['prereq'],f"{float(r['mu']):.2f}",f"{float(r['Q']):.2f}",r['ratings'].replace('_',' '),st,str(att),f"{pc} / {float(r['bkt']):.2f}",state,note]
    for x,v in zip(cx,vals): ax.text(x,yy,v,fontsize=6.6,color=col,va='top')
    yy-=0.95
ax.text(0.4,0.35,'Stability flag: division_0.9 and division_1 become gaps only if an unobserved prerequisite is scored θ = 0; under BKT they are not gaps.\nThe card therefore shows them as "no observation, estimator-sensitive" and suggests a short check instead of remediation.',fontsize=6.8,style='italic')
fig.savefig('fig/f5.png',dpi=300,bbox_inches='tight')
