from evalcore import *
import json
from scipy.stats import spearmanr
rng=np.random.default_rng(31)
out={}
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80)
share=np.load('data/dir_share.npy')
out['targets_per_learner']=dict(mean=float(te.groupby('L').size().mean()),median=float(te.groupby('L').size().median()),p90=float(te.groupby('L').size().quantile(.9)))
def expand(df,A,C):
    _,(rep,eidx,th)=load_scores(df,A,C,np.ones(NE))
    L=df.L.values[rep]; I=SRC[eidx]; na=np.asarray(A[L,I]).ravel()
    return rep,eidx,th,na
EV=expand(val,Av,Cv); ET=expand(te,At,Ct)
def feats(df,E,w,minatt=1,src=None):
    rep,eidx,th,na=E
    ww=w[eidx]; on=ww>0; obs=na>=minatt
    g=np.where(obs,np.maximum(0,ww-th),0)*on
    return np.bincount(rep,g,len(df))
def missfeat(df,E,minatt=1):
    rep,eidx,th,na=E
    return np.c_[np.bincount(rep,(na<minatt).astype(float),len(df)),np.bincount(rep,(na>=minatt).astype(float),len(df))]
yv=val.y.values; yt=te.y.values
# ---- missingness-aware nested models (base + counts of unobserved/observed candidates)
XbV=np.c_[base_X(val),missfeat(val,EV)]; XbT=np.c_[base_X(te),missfeat(te,ET)]
pb=fit_auc(XbV,yv,XbT,yt); aB=roc_auc_score(yt,pb)
P={'BASE+missingness':pb}
for m in ['BIN_EXP','ALL1','UNG','FPKG','TRANS']:
    P[m]=fit_auc(np.c_[XbV,feats(val,EV,W[m])],yv,np.c_[XbT,feats(te,ET,W[m])],yt)
pm=rng.permutation(W['UNG']); P['PERM']=fit_auc(np.c_[XbV,feats(val,EV,pm)],yv,np.c_[XbT,feats(te,ET,pm)],yt)
Lu,inv=np.unique(te.L.values,return_inverse=True)
def boot(p1,p2,ysub=None,mask=None,B=300):
    y_=yt if mask is None else yt[mask]; i_=inv if mask is None else inv[mask]; a=p1 if mask is None else p1[mask]; b=p2 if mask is None else p2[mask]
    o=[]
    for _ in range(B):
        sw=np.bincount(rng.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[i_]; k=sw>0
        o.append(roc_auc_score(y_[k],a[k],sample_weight=sw[k])-roc_auc_score(y_[k],b[k],sample_weight=sw[k]))
    return [float(roc_auc_score(y_,a)-roc_auc_score(y_,b)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
miss={'base_auc_old':float(roc_auc_score(yt,fit_auc(base_X(val),yv,base_X(te),yt))),'base_auc_miss':float(aB)}
for m in ['BIN_EXP','ALL1','UNG','FPKG','TRANS','PERM']: miss[m]=boot(P[m],pb)
miss['UNG-BIN']=boot(P['UNG'],P['BIN_EXP']); miss['UNG-PERM']=boot(P['UNG'],P['PERM'])
out['missingness_aware']=miss; print('miss',miss)
# ---- coverage subgroups (by observed candidates at >=1 and >=3 attempts)
rep,eidx,th,na=ET
obs1=np.bincount(rep,(na>=1),len(te)); obs3=np.bincount(rep,(na>=3),len(te))
grp={'no candidate attempted':obs1==0,'>=1 candidate attempted':obs1>=1,'>=1 candidate with >=3 attempts':obs3>=1}
sub={}
for g,mk in grp.items():
    r={'n':int(mk.sum()),'fail':float(yt[mk].mean())}
    for m in ['BIN_EXP','ALL1','UNG','FPKG']: r[m]=boot(P[m],pb,mask=mk,B=200)
    sub[g]=r
out['coverage_groups']=sub; print(json.dumps(sub,indent=0)[:1500])
# ---- calibration, Brier, decision curve, equivalence (original primary specification)
Xv0=base_X(val); Xt0=base_X(te)
sv,_=load_scores(val,Av,Cv,W['FPKG']); st,_=load_scores(te,At,Ct,W['FPKG'])
sbv,_=load_scores(val,Av,Cv,W['BIN_EXP']); sbt,_=load_scores(te,At,Ct,W['BIN_EXP'])
p0=fit_auc(Xv0,yv,Xt0,yt); pf=fit_auc(np.c_[Xv0,sv],yv,np.c_[Xt0,st],yt); pbn=fit_auc(np.c_[Xv0,sbv],yv,np.c_[Xt0,sbt],yt)
def ece(p,yy,b=10):
    q=np.quantile(p,np.linspace(0,1,b+1)); idx=np.clip(np.digitize(p,q[1:-1]),0,b-1)
    return float(sum(abs(p[idx==k].mean()-yy[idx==k].mean())*np.mean(idx==k) for k in range(b)))
def nb(p,yy,t): tp=np.mean((p>=t)&(yy==1)); fp=np.mean((p>=t)&(yy==0)); return tp-fp*t/(1-t)
cal={}
for nm,p in [('confounders',p0),('+FPKG',pf),('+binary expert',pbn)]:
    cal[nm]=dict(brier=float(np.mean((p-yt)**2)),ece=ece(p,yt),**{f'NB@{t}':float(nb(p,yt,t)) for t in [0.3,0.4,0.5,0.6]})
cal['treat_all']={f'NB@{t}':float(nb(np.ones_like(yt,float),yt,t)) for t in [0.3,0.4,0.5,0.6]}
out['calibration']=cal; print(pd.DataFrame(cal).T.round(4))
# ---- direction sensitivity (learner level), graph variants rebuilt on alternative edge sets
import evalcore as ec
def run_variant(keep,flip):
    s0=ec.SRC.copy(); d0=ec.DST.copy()
    ns=np.where(flip,d0,s0); nd=np.where(flip,s0,d0)
    ec.SRC[:]=ns; ec.DST[:]=nd
    res={}
    try:
        for m,w in [('BIN',np.array([(a,b) in Ex for a,b in zip(ns,nd)],float)),('UNG',W['UNG']),('FPKG',W['FPKG'])]:
            ww=w*keep
            svv,_=load_scores(val,Av,Cv,ww); stt,_=load_scores(te,At,Ct,ww)
            # recompute indeg-based base with the variant graph
            ec.indeg[:]=np.bincount(nd[keep>0],minlength=NC)
            Xv=base_X(val); Xt=base_X(te)
            b_=roc_auc_score(yt,fit_auc(Xv,yv,Xt,yt))
            res[m]=float(roc_auc_score(yt,fit_auc(np.c_[Xv,svv],yv,np.c_[Xt,stt],yt))-b_)
    finally:
        ec.SRC[:]=s0; ec.DST[:]=d0; ec.indeg[:]=np.bincount(d0,minlength=NC)
    return res
exprev=np.array([(d,s) in Ex for s,d in zip(SRC,DST)]); linked=exprev|(W['BIN_EXP']==1)
ok=~np.isnan(share)
dirs={'as inferred (B→A)':run_variant(np.ones(NE),np.zeros(NE,bool)),
 'direction from training-log order':run_variant(ok.astype(float),ok&(share<0.5)),
 'expert-reversed pairs excluded':run_variant((~exprev).astype(float),np.zeros(NE,bool)),
 'order-ambiguous pairs (0.4–0.6) excluded':run_variant((ok&((share<=0.4)|(share>=0.6))).astype(float),np.zeros(NE,bool)),
 'all directions reversed':run_variant(np.ones(NE),np.ones(NE,bool))}
out['direction_learner']=dirs; print(pd.DataFrame(dirs).T.round(4))
# graph-level R2 is symmetric in direction: note only; crowd-only pairs graph-level
json.dump(out,open('analysis_b.json','w'),default=float,indent=1)
