"""Supplementary analyses: threshold sensitivity at learner level, three-state
specification, ranking candidate distribution, estimator stability split by observation."""
from evalcore import *
import json, pickle
rng=np.random.default_rng(53)
out={}
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
yv=val.y.values; yt=te.y.values
def expand(df,A,C):
    _,(rep,eidx,th)=load_scores(df,A,C,np.ones(NE))
    L=df.L.values[rep]; I=SRC[eidx]; na=np.asarray(A[L,I]).ravel()
    return rep,eidx,th,na
EV=expand(val,Av,Cv); ET=expand(te,At,Ct)
def counts(df,E):
    rep,eidx,th,na=E
    return np.c_[np.bincount(rep,(na==0).astype(float),len(df)),np.bincount(rep,(na>0).astype(float),len(df))]
def primary(df,E,w):   # theta=0 for unobserved (pre-specified)
    rep,eidx,th,na=E; ww=w[eidx]
    return np.bincount(rep,np.maximum(0,ww-th)*(ww>0),len(df))[:,None]
def three_state(df,E,w):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; obs=na>0
    deficit=np.where(obs&(th<ww),ww-th,0)*on            # evidence of deficit, weighted by shortfall
    mastered=np.where(obs&(th>=ww),ww,0)*on              # strength of mastered relations
    unobs=np.where(~obs,ww,0)*on                         # strength of relations without evidence
    return np.c_[np.bincount(rep,deficit,len(df)),np.bincount(rep,mastered,len(df)),np.bincount(rep,unobs,len(df))]
Lu,inv=np.unique(te.L.values,return_inverse=True)
def boot(p1,p2,B=300):
    o=[]
    for _ in range(B):
        sw=np.bincount(rng.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; k=sw>0
        o.append(roc_auc_score(yt[k],p1[k],sample_weight=sw[k])-roc_auc_score(yt[k],p2[k],sample_weight=sw[k]))
    return [float(roc_auc_score(yt,p1)-roc_auc_score(yt,p2)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
# ---- A. threshold sensitivity at learner level (primary and three-state)
Xb0v=base_X(val); Xb0t=base_X(te); p0=fit_auc(Xb0v,yv,Xb0t,yt)
XbCv=np.c_[base_X(val),counts(val,EV)]; XbCt=np.c_[base_X(te),counts(te,ET)]; pC=fit_auc(XbCv,yv,XbCt,yt)
thr={}
for q in np.round(np.arange(0.70,0.901,0.025),3):
    w=gated(q)
    a=roc_auc_score(yt,fit_auc(np.c_[Xb0v,primary(val,EV,w)],yv,np.c_[Xb0t,primary(te,ET,w)],yt))-roc_auc_score(yt,p0)
    b=roc_auc_score(yt,fit_auc(np.c_[XbCv,three_state(val,EV,w)],yv,np.c_[XbCt,three_state(te,ET,w)],yt))-roc_auc_score(yt,pC)
    thr[str(q)]=dict(retained=int((w>0).sum()),dAUC_primary=float(a),dAUC_three_state=float(b))
out['threshold_learner']=thr; print(pd.DataFrame(thr).T.round(4))
# ---- B. three-state specification (fitted on validation, evaluated once on test)
W['FPKG']=gated(0.80); Wx=dict(W); Wx['PERM']=rng.permutation(W['UNG'])
P={}; ts={'base_auc':float(roc_auc_score(yt,pC))}
for m in ['BIN_EXP','ALL1','UNG','FPKG','TRANS','PERM']:
    P[m]=fit_auc(np.c_[XbCv,three_state(val,EV,Wx[m])],yv,np.c_[XbCt,three_state(te,ET,Wx[m])],yt); ts[m]=boot(P[m],pC)
for a,b in [('UNG','BIN_EXP'),('UNG','ALL1'),('UNG','PERM'),('FPKG','UNG'),('TRANS','UNG')]: ts[f'{a}-{b}']=boot(P[a],P[b])
out['three_state']=ts; print(json.dumps(ts,indent=0)[:900])
# ---- C. ranking candidate distribution (primary case set definition)
wm=(n>=10)&(frac>=0.8); wi=np.flatnonzero(wm)
key=lid[wi].astype(np.int64)*NC+e[wi]; uk,first=np.unique(key,return_index=True)
rep,eidx,th,na=ET
L=te.L.values[rep]; I=SRC[eidx]; k2=L.astype(np.int64)*NC+I; pos=np.minimum(np.searchsorted(uk,k2),len(uk)-1); obs=uk[pos]==k2
fcor=c[wi[first]]; rel=np.where(obs,(~fcor[pos]).astype(float),np.nan)
D=pd.DataFrame(dict(case=rep,obs=obs,rel=rel)); D=D[yt[D.case]==1]
g=D.groupby('case').agg(total=('obs','size'),observed=('obs','sum'),pos=('rel',lambda x: np.nansum(x)))
prim=g[(g.observed>=2)&(g.pos>0)]
out['ranking_distribution']=dict(cases=int(len(prim)),total_mean=float(prim.total.mean()),total_median=float(prim.total.median()),total_p90=float(prim.total.quantile(.9)),
    observed_mean=float(prim.observed.mean()),observed_median=float(prim.observed.median()),observed_share=float((prim.observed/prim.total).mean()),
    observed_dist={str(k):int(v) for k,v in prim.observed.clip(upper=6).value_counts().sort_index().items()})
print(out['ranking_distribution'])
# ---- D. estimator stability split by observation status
bp=pickle.load(open('data/bkt_params.pkl','rb'))
Ls=np.random.default_rng(5).choice(te.L.unique(),3000,replace=False)   # same sample rule as run_stab.py
sub=te[te.L.isin(Ls)].reset_index(drop=True)
_,(rep2,eidx2,th2)=load_scores(sub,At,Ct,W['FPKG'])
d=pd.DataFrame(dict(L=sub.L.values[rep2],I=SRC[eidx2],w=W['FPKG'][eidx2],pc=th2)); d=d[d.w>0]
d=d.sort_values('w',ascending=False).drop_duplicates(['L','I'])
H=(n>=10)&(frac<0.8)&np.isin(lid,Ls)
hist=pd.DataFrame(dict(L=lid[H],I=e[H],c=c[H])).groupby(['L','I']).c.apply(np.array)
def bkt(L_,I_):
    if I_ not in bp: return np.nan
    L0,T,G,S=bp[I_]; pL=L0; seq=hist.get((L_,I_))
    if seq is None: return L0
    for o in seq:
        pc=pL*(1-S)+(1-pL)*G; post=pL*(1-S)/pc if o else pL*S/(1-pc); pL=post+(1-post)*T
    return pL
d['bkt']=[bkt(a,b) for a,b in zip(d.L,d.I)]; d['bkt']=d.bkt.fillna(d.pc)
d['att']=np.asarray(At[d.L.values,d.I.values]).ravel()>0
d['gap_pc']=d.pc<d.w; d['gap_bkt']=d.bkt<d.w
def jac(df,K=5):
    r=[]
    for L_,x in df.groupby('L'):
        a=set(x[x.gap_pc].assign(s=lambda z:z.w-z.pc).nlargest(K,'s').I); b=set(x[x.gap_bkt].assign(s=lambda z:z.w-z.bkt).nlargest(K,'s').I)
        if a|b: r.append(len(a&b)/len(a|b))
    return float(np.mean(r)),len(r)
stab=dict(flip_observed=float((d[d.att].gap_pc!=d[d.att].gap_bkt).mean()),flip_unobserved=float((d[~d.att].gap_pc!=d[~d.att].gap_bkt).mean()),
    share_unobserved_among_flips=float((~d.att[d.gap_pc!=d.gap_bkt]).mean()),n_pairs=int(len(d)),n_observed=int(d.att.sum()))
j_all=jac(d); j_obs=jac(d[d.att]); stab['jaccard_all']=j_all[0]; stab['jaccard_observed_only']=j_obs[0]; stab['learners_observed_only']=j_obs[1]
d2=d.copy(); d2.loc[~d2.att,'bkt']=0.0; d2['gap_bkt']=d2.bkt<d2.w
stab['jaccard_bkt_with_unobserved_zero']=jac(d2)[0]
out['stability_split']=stab; print(stab)
json.dump(out,open('analysis_e.json','w'),default=float,indent=1)
