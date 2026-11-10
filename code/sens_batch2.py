"""Supplementary sensitivity analyses (exploratory; same held-out window). Writes sens_batch2.json.
A. One-change-at-a-time ablation of the learner-level specification.
B. Binarisation cut-point sensitivity (graph r^2 and primary learner dAUC).
C. Agreement gate recomputed and re-selected under monotone re-encodings.
D. Threshold-selection stability under repeated validation splits.
E. Refitting learner bootstrap (models refitted on resampled validation learners).
F. Practical-margin sensitivity of classifications.
G. Low-panel replication on the 823 pairs with 3-4 raters.
H. Direction as an evidence attribute: states, counts, and results on the supported subset.
I. Attrition counts for the flow diagram.
"""
from evalcore import *
import json
out={}
Av,Cv,val=load('val'); At,Ct,te=load('test')
valA=val; teA=te
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
yv=val.y.values; yt=te.y.values
W['FPKG']=gated(0.80)
Lu,inv=np.unique(te.L.values,return_inverse=True)
def lboot(p1,p2,B=300,seed=606):
    r=np.random.default_rng(seed); o=[]
    for _ in range(B):
        sw=np.bincount(r.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; k=sw>0
        o.append(roc_auc_score(yt[k],p1[k],sample_weight=sw[k])-roc_auc_score(yt[k],p2[k],sample_weight=sw[k]))
    return [float(roc_auc_score(yt,p1)-roc_auc_score(yt,p2)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
def expand(df,A_,C_):
    _,(rep,eidx,th)=load_scores(df,A_,C_,np.ones(NE)); L=df.L.values[rep]; I=SRC[eidx]; na=np.asarray(A_[L,I]).ravel()
    return rep,eidx,th,na
EV=expand(val,Av,Cv); ET=expand(te,At,Ct)
def counts(df,E):
    rep,eidx,th,na=E; return np.c_[np.bincount(rep,(na<1).astype(float),len(df)),np.bincount(rep,(na>=1).astype(float),len(df))]
def sc_theta0(df,E,w):
    rep,eidx,th,na=E; ww=w[eidx]; return np.bincount(rep,np.maximum(0,ww-th)*(ww>0),len(df))[:,None]
def sc_obs(df,E,w,tau=None):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; thr=ww if tau is None else np.full_like(ww,tau); g=np.maximum(0,thr-th)*(1.0 if tau is None else ww)*on
    return np.bincount(rep,np.where(na>=1,g,0),len(df))[:,None]
def sc_three(df,E,w,tau=None):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; obs=na>0
    if tau is None: d=np.where(obs&(th<ww),ww-th,0)*on
    else: d=np.where(obs,ww*np.maximum(0,tau-th),0)*on
    return np.c_[np.bincount(rep,d,len(df)),np.bincount(rep,np.where(obs&(th>=ww),ww,0)*on,len(df)),np.bincount(rep,np.where(~obs,ww,0)*on,len(df))]
Xb0v=base_X(val); Xb0t=base_X(te); XbCv=np.c_[Xb0v,counts(val,EV)]; XbCt=np.c_[Xb0t,counts(te,ET)]
p0=fit_auc(Xb0v,yv,Xb0t,yt); pC=fit_auc(XbCv,yv,XbCt,yt)
# ---------- A. ablation ladder ----------
A={'step0_confounders_auc':float(roc_auc_score(yt,p0)),'stepB_counts_only_vs_confounders':lboot(pC,p0)}
for m in ['UNG','FPKG','ALL1','BIN_EXP']:
    w=W[m]; r={}
    r['s1_theta0_primary']=lboot(fit_auc(np.c_[Xb0v,sc_theta0(val,EV,w)],yv,np.c_[Xb0t,sc_theta0(te,ET,w)],yt),p0)
    r['s2a_exclude_unobserved_no_counts']=lboot(fit_auc(np.c_[Xb0v,sc_obs(val,EV,w)],yv,np.c_[Xb0t,sc_obs(te,ET,w)],yt),p0)
    r['s2b_exclude_unobserved_plus_counts_vs_counts']=lboot(fit_auc(np.c_[XbCv,sc_obs(val,EV,w)],yv,np.c_[XbCt,sc_obs(te,ET,w)],yt),pC)
    r['s3_three_components_plus_counts_vs_counts']=lboot(fit_auc(np.c_[XbCv,sc_three(val,EV,w)],yv,np.c_[XbCt,sc_three(te,ET,w)],yt),pC)
    r['s4_three_components_decoupled_tau1_vs_counts']=lboot(fit_auc(np.c_[XbCv,sc_three(val,EV,w,1.0)],yv,np.c_[XbCt,sc_three(te,ET,w,1.0)],yt),pC)
    A[m]=r; print('A',m,{k:np.round(v,4).tolist() for k,v in r.items()})
out['A_ablation']=A
# ---------- B. binarisation cut-points ----------
TP=(frac>=0.8)&(n>=10); lt=lid[TP]; et=e[TP]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; mm=a!=b
M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[mm],b[mm]),np.maximum(a[mm],b[mm])),1); ycc=np.log1p((M+M.T)[SRC,DST])
def r2(x,yy): return float(np.corrcoef(x,yy)[0,1]**2) if np.std(x)>0 else 0.0
raw=[np.array([int(v) for v in rr.split('_')],float) for rr in cr.raw]
Bres={}
rules={}
for k in [3,4,5,6,7]:
    rules[f'majority of ratings >= {k}']=np.array([np.mean(x>=k)>0.5 for x in raw],float)
    rules[f'mean rating >= {k}']=np.array([x.mean()>=k for x in raw],float)
rules['median rating >= 5']=np.array([np.median(x)>=5 for x in raw],float)
for nm,wb in rules.items():
    p=fit_auc(np.c_[Xb0v,sc_theta0(val,EV,wb)],yv,np.c_[Xb0t,sc_theta0(te,ET,wb)],yt)
    Bres[nm]=dict(n_nonzero=int(wb.sum()),r2=r2(wb,ycc),r2_graded_minus_binary=r2(W['UNG'],ycc)-r2(wb,ycc),dAUC_primary=lboot(p,p0))
    print('B',nm,Bres[nm]['n_nonzero'],round(Bres[nm]['r2'],4))
out['B_binarisation']=Bres
# ---------- C. gate re-selected under encodings ----------
rng=np.random.default_rng(66)
allr=np.concatenate(raw); ranks={v:(allr<v).mean()+0.5*(allr==v).mean() for v in range(10)}
encs={'linear r/9':lambda x:x/9,'square':lambda x:(x/9)**2,'sqrt':lambda x:np.sqrt(x/9),'mid-rank quantile':lambda x:np.vectorize(ranks.get)(x),'three-level':lambda x:np.where(x<=3,0.0,np.where(x<=6,0.5,1.0))}
h=((val.L.values.astype(np.int64)*2654435761)%7)<4
grid=np.round(np.arange(0.70,0.901,0.025),3)
def Qof(xs): med=np.median(xs); m_=len(xs); return 1-np.abs(xs-med).sum()/(m_-1)
Cres={}
for nm,f in encs.items():
    mu_e=np.array([f(x).mean() for x in raw]); Q_e=np.array([Qof(f(x)) for x in raw])
    # quantile-matched grid: thresholds giving the same retention fractions as the linear grid
    ret_lin=[(Qv>=q-1e-9).mean() for q in grid]
    thr_e=[np.quantile(Q_e,1-r_) for r_ in ret_lin]
    sel=[]
    for q,rl in zip(thr_e,ret_lin):
        w=mu_e*(Q_e>=q-1e-9); sv=sc_theta0(val,EV,w); pv=fit_auc(np.c_[Xb0v,sv][h],yv[h],np.c_[Xb0v,sv][~h],yv[~h])
        sel.append((float(q),float((w>0).mean()),float(roc_auc_score(yv[~h],pv))))
    ok=[s_ for s_ in sel if s_[1]>=0.5]; qstar=max(ok,key=lambda s_:(s_[2],-s_[0]))
    g=Q_e>=qstar[0]-1e-9; rr=[r2(mu_e[s_],ycc[s_]) for s_ in [rng.choice(NE,int(g.sum()),replace=False) for _ in range(200)]]
    Cres[nm]=dict(selected_threshold=qstar[0],retained=int(g.sum()),retained_frac=qstar[1],r2_admitted=r2(mu_e[g],ycc[g]),r2_all=r2(mu_e,ycc),random_mean=float(np.mean(rr)),random_hi=float(np.percentile(rr,97.5)),overlap_with_linear_admitted=float((g&(Qv>=0.8-1e-9)).sum()/713))
    print('C',nm,Cres[nm])
out['C_gate_under_encodings']=Cres
# ---------- D. threshold-selection stability ----------
Lv=val.L.values; Luv=np.unique(Lv)
sels=[]
for b in range(50):
    r=np.random.default_rng(1000+b); samp=r.choice(Luv,len(Luv),replace=True)
    cnt=pd.Series(samp).value_counts(); wgt=cnt.reindex(Lv).fillna(0).values
    hb=((Lv.astype(np.int64)*2654435761+b*7919)%7)<4
    best=None
    for q in grid:
        w=gated(q); ret=(w>0).mean()
        if ret<0.5: continue
        sv=sc_theta0(val,EV,w); X=np.c_[Xb0v,sv]
        tr=hb&(wgt>0); tv=(~hb)&(wgt>0)
        mu_=X[tr].mean(0); sd=X[tr].std(0)+1e-9
        from sklearn.linear_model import LogisticRegression
        mdl=LogisticRegression(max_iter=500).fit((X[tr]-mu_)/sd,yv[tr],sample_weight=wgt[tr]); pv=mdl.predict_proba((X[tv]-mu_)/sd)[:,1]
        auc=roc_auc_score(yv[tv],pv,sample_weight=wgt[tv])
        if best is None or auc>best[1]+1e-12: best=(float(q),float(auc))
    sels.append(best[0])
Dres={'selected_threshold_frequency':{str(k):int(v) for k,v in pd.Series(sels).value_counts().sort_index().items()},'replicates':50}
out['D_threshold_stability']=Dres; print('D',Dres)
# ---------- E. refitting bootstrap ----------
Lvu,invv=np.unique(val.L.values,return_inverse=True)
def refit_boot(fv,ft,fv_base,ft_base,B=100,seed=77):
    """fv/ft: feature matrices for model; fv_base/ft_base for comparator. Resample validation learners (refit) and test learners (evaluate)."""
    r=np.random.default_rng(seed); o=[]
    from sklearn.linear_model import LogisticRegression
    for _ in range(B):
        wv=np.bincount(r.integers(0,len(Lvu),len(Lvu)),minlength=len(Lvu))[invv]; wt=np.bincount(r.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]
        kv=wv>0; kt=wt>0
        def fit(Xv,Xt):
            mu_=Xv[kv].mean(0); sd=Xv[kv].std(0)+1e-9
            m_=LogisticRegression(max_iter=500).fit((Xv[kv]-mu_)/sd,yv[kv],sample_weight=wv[kv]); return m_.predict_proba((Xt-mu_)/sd)[:,1]
        p1=fit(fv,ft); p2=fit(fv_base,ft_base)
        o.append(roc_auc_score(yt[kt],p1[kt],sample_weight=wt[kt])-roc_auc_score(yt[kt],p2[kt],sample_weight=wt[kt]))
    return [float(np.mean(o)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
Eres={}
for m in ['UNG','FPKG','ALL1','BIN_EXP']:
    w=W[m]
    Eres[m+'_primary']=refit_boot(np.c_[Xb0v,sc_theta0(val,EV,w)],np.c_[Xb0t,sc_theta0(te,ET,w)],Xb0v,Xb0t)
    Eres[m+'_three_state']=refit_boot(np.c_[XbCv,sc_three(val,EV,w)],np.c_[XbCt,sc_three(te,ET,w)],XbCv,XbCt)
    print('E',m,Eres[m+'_primary'],Eres[m+'_three_state'])
Eres['UNG_minus_ALL1_three_state']=refit_boot(np.c_[XbCv,sc_three(val,EV,W['UNG'])],np.c_[XbCt,sc_three(te,ET,W['UNG'])],np.c_[XbCv,sc_three(val,EV,W['ALL1'])],np.c_[XbCt,sc_three(te,ET,W['ALL1'])])
out['E_refit_bootstrap']=Eres
# ---------- F. margin sensitivity ----------
out['F_margins']=[0.0025,0.005,0.01,0.02]
# ---------- G. 823-pair low-panel replication ----------
cr2=crowd('data/relationship_annotation_testing.csv')
S2=cr2.src.values; D2=cr2.dst.values; y2=np.log1p((M+M.T)[S2,D2]); mu2=cr2.mu.values; m2=cr2.m.values
bin2=np.array([(s,d) in Ex or (d,s) in Ex for s,d in zip(S2,D2)],float)
Gres={'pairs':int(len(cr2)),'raters_min':int(m2.min()),'raters_max':int(m2.max()),'r2_graded':r2(mu2,y2),'r2_expert_either':r2(bin2,y2),'r2_majority':r2(cr2.maj.values.astype(float),y2),'overlap_pairs_with_training_set':int(len(set(zip(S2,D2))&set(zip(SRC,DST))))}
us2,iS2=np.unique(S2,return_inverse=True); ut2,iT2=np.unique(D2,return_inverse=True); rr=[]
for _ in range(1000):
    w=np.bincount(rng.integers(0,len(us2),len(us2)),minlength=len(us2))[iS2]*np.bincount(rng.integers(0,len(ut2),len(ut2)),minlength=len(ut2))[iT2]; ix=np.repeat(np.arange(len(cr2)),w)
    if len(ix)<50: continue
    rr.append(r2(mu2[ix],y2[ix])-r2(bin2[ix],y2[ix]))
Gres['two_way_ci_graded_minus_expert']=[float(np.percentile(rr,2.5)),float(np.percentile(rr,97.5))]
out['G_823_pairs']=Gres; print('G',Gres)
# ---------- H. direction states ----------
share=np.load('data/dir_share.npy'); expdir=W['BIN_EXP']==1; exprev=np.array([(d,s) in Ex for s,d in zip(SRC,DST)])
ok=~np.isnan(share)
state=np.full(NE,'unresolved',object)
state[ok&(share>0.6)&~exprev]='supported'            # log order agrees (and expert not reverse)
state[ok&(share>0.6)&exprev]='disputed'
state[ok&(share<0.4)]='disputed'
state[ok&(share>=0.4)&(share<=0.6)]='ambiguous'
state[~ok&expdir]='supported (expert only)'
state[~ok&exprev]='disputed'
Hres={'counts':{k:int(v) for k,v in pd.Series(state).value_counts().items()}}
for nm,mask in [('supported',np.isin(state,['supported','supported (expert only)'])),('supported (log order)',state=='supported'),('all',np.ones(NE,bool))]:
    r={'n':int(mask.sum()),'r2_graded':r2(W['UNG'][mask],ycc[mask])}
    w=W['UNG']*mask; r['dAUC_primary_graded']=lboot(fit_auc(np.c_[Xb0v,sc_theta0(val,EV,w)],yv,np.c_[Xb0t,sc_theta0(te,ET,w)],yt),p0)
    r['dAUC_three_state_graded']=lboot(fit_auc(np.c_[XbCv,sc_three(val,EV,w)],yv,np.c_[XbCt,sc_three(te,ET,w)],yt),pC)
    w1=W['ALL1']*mask; r['dAUC_three_state_unweighted']=lboot(fit_auc(np.c_[XbCv,sc_three(val,EV,w1)],yv,np.c_[XbCt,sc_three(te,ET,w1)],yt),pC)
    Hres[nm]=r; print('H',nm,r)
out['H_direction_states']=Hres
# ---------- I. attrition ----------
z=np.load('data/sorted.npz'); nn=z['n']; lidd=z['lid']
Ires={'learners_all':int(lidd.max()+1),'learners_ge10':int(len(np.unique(lidd[nn>=10]))),'val_cases_all_targets':int(len(valA)),'val_cases_new':int(len(val)),'test_cases_all_targets':int(len(teA)),'test_cases_new':int(len(te)),'test_learners_new':int(te.L.nunique())}
out['I_attrition']=Ires; print('I',Ires)
json.dump(out,open('sens_batch2.json','w'),indent=1,default=float); print('DONE')
