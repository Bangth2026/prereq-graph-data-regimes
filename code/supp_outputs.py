"""Supplementary outputs (Supplementary Material S1).
Run after run_all.sh (needs data/ and data/dir_share.npy). Writes supp_outputs.json.
A. B5-B4 r^2 difference on held-out co-occurrence with pair/source/target/two-way bootstraps (S1.7).
B. 1,000-replicate learner bootstraps for the count-adjusted and three-state dAUC rows, and
   seed-to-seed stability of the 300-replicate bounds (S1.7).
C. Propensity diagnostics for the IPW ranking analysis; zero-IDCG and relabelled counts (S1.2).
D. Calendar ranges of the split windows and share of history events after test decision points (S1.5).
"""
from evalcore import *
import json, datetime
from sklearn.linear_model import LogisticRegression
out={}
# ---------------- A. graph level: B5 (UNG) vs B4 (BIN_MAJ) ----------------
rng=np.random.default_rng(2028)
TP=(frac>=0.8)&(n>=10)
lt=lid[TP]; et=e[TP]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); y=np.log1p((M+M.T)[SRC,DST])
def r2(x,yy): return np.corrcoef(x,yy)[0,1]**2
mu=W['UNG']; maj=W['BIN_MAJ']
us,iS=np.unique(SRC,return_inverse=True); ut,iT=np.unique(DST,return_inverse=True)
def boot(x1,x2,scheme,B=1000):
    res=[]
    for _ in range(B):
        if scheme=='pairs': w=np.bincount(rng.integers(0,NE,NE),minlength=NE)
        elif scheme=='source': w=np.bincount(rng.integers(0,len(us),len(us)),minlength=len(us))[iS]
        elif scheme=='target': w=np.bincount(rng.integers(0,len(ut),len(ut)),minlength=len(ut))[iT]
        else: w=np.bincount(rng.integers(0,len(us),len(us)),minlength=len(us))[iS]*np.bincount(rng.integers(0,len(ut),len(ut)),minlength=len(ut))[iT]
        ix=np.repeat(np.arange(NE),w)
        if len(ix)<50: continue
        res.append(r2(x1[ix],y[ix])-r2(x2[ix],y[ix]))
    return [float(np.percentile(res,2.5)),float(np.percentile(res,97.5))]
A={'r2_UNG':r2(mu,y),'r2_BIN_MAJ':r2(maj,y),'diff':r2(mu,y)-r2(maj,y),'n_maj':int(maj.sum())}
for s_ in ['pairs','source','target','two-way']: A['boot_'+s_]=boot(mu,maj,s_)
out['A_B5_minus_B4']=A; print('A',A)
# ---------------- B. learner level, 1,000 replicates and seed stability ----------------
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
yv=val.y.values; yt=te.y.values
W['FPKG']=gated(0.80)
def expand(df,A_,C_):
    _,(rep,eidx,th)=load_scores(df,A_,C_,np.ones(NE))
    L=df.L.values[rep]; I=SRC[eidx]; na=np.asarray(A_[L,I]).ravel()
    return rep,eidx,th,na
EV=expand(val,Av,Cv); ET=expand(te,At,Ct)
def missfeat(df,E):
    rep,eidx,th,na=E
    return np.c_[np.bincount(rep,(na<1).astype(float),len(df)),np.bincount(rep,(na>=1).astype(float),len(df))]
def feats(df,E,w):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; obs=na>=1
    return np.bincount(rep,np.where(obs,np.maximum(0,ww-th),0)*on,len(df))
def three_state(df,E,w):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; obs=na>0
    return np.c_[np.bincount(rep,np.where(obs&(th<ww),ww-th,0)*on,len(df)),np.bincount(rep,np.where(obs&(th>=ww),ww,0)*on,len(df)),np.bincount(rep,np.where(~obs,ww,0)*on,len(df))]
XbV=np.c_[base_X(val),missfeat(val,EV)]; XbT=np.c_[base_X(te),missfeat(te,ET)]
pb=fit_auc(XbV,yv,XbT,yt)
P_count={m:fit_auc(np.c_[XbV,feats(val,EV,W[m])],yv,np.c_[XbT,feats(te,ET,W[m])],yt) for m in ['ALL1','UNG','FPKG']}
P_three={m:fit_auc(np.c_[XbV,three_state(val,EV,W[m])],yv,np.c_[XbT,three_state(te,ET,W[m])],yt) for m in ['ALL1','UNG','FPKG']}
Lu,inv=np.unique(te.L.values,return_inverse=True)
def lboot(p1,p2,B,seed):
    r=np.random.default_rng(seed); o=[]
    for _ in range(B):
        sw=np.bincount(r.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; k=sw>0
        o.append(roc_auc_score(yt[k],p1[k],sample_weight=sw[k])-roc_auc_score(yt[k],p2[k],sample_weight=sw[k]))
    return [float(roc_auc_score(yt,p1)-roc_auc_score(yt,p2)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
rows={'B9_count_adjusted':(P_count['ALL1'],pb),'B5_count_adjusted':(P_count['UNG'],pb),'B6_count_adjusted':(P_count['FPKG'],pb),
      'B9_three_state':(P_three['ALL1'],pb),'B5_three_state':(P_three['UNG'],pb),'B6_three_state':(P_three['FPKG'],pb),
      'B5_minus_B9_three_state':(P_three['UNG'],P_three['ALL1']),'B6_minus_B5_three_state':(P_three['FPKG'],P_three['UNG'])}
Bres={}
seeds=[31,53,101,102,103,104,105,106,107,108]
for k,(p1,p2) in rows.items():
    b1000=lboot(p1,p2,1000,7001)
    runs=[lboot(p1,p2,300,s) for s in seeds]
    lo=np.array([r_[1] for r_ in runs]); hi=np.array([r_[2] for r_ in runs])
    Bres[k]=dict(point=b1000[0],ci_1000=[b1000[1],b1000[2]],ci_300_by_seed=[[r_[1],r_[2]] for r_ in runs],sd_lower_300=float(lo.std(ddof=1)),sd_upper_300=float(hi.std(ddof=1)),
                 classification_1000=('negligible' if -0.005<b1000[1] and b1000[2]<0.005 else 'relevant' if (b1000[1]>0.005 or b1000[2]<-0.005) else 'inconclusive'))
    print(k,Bres[k]['point'],Bres[k]['ci_1000'],Bres[k]['classification_1000'],'sd lo/hi',Bres[k]['sd_lower_300'],Bres[k]['sd_upper_300'])
out['B_learner_bootstraps']=Bres
# ---------------- C. IPW diagnostics (reproduces the construction of analysis_c.py) ----------------
rng=np.random.default_rng(41)
wm=(n>=10)&(frac>=0.8); wi=np.flatnonzero(wm)
key=lid[wi].astype(np.int64)*NC+e[wi]; uk,first=np.unique(key,return_index=True); fcor=c[wi[first]]
rep,eidx,th,na=ET
L=te.L.values[rep]; I=SRC[eidx]; k2=L.astype(np.int64)*NC+I; pos=np.minimum(np.searchsorted(uk,k2),len(uk)-1); obs=uk[pos]==k2
D=pd.DataFrame(dict(case=rep,L=L,I=I,e=eidx,th=th,obs=obs,att=na>0,rel=np.where(obs,(~fcor[pos]).astype(float),np.nan),mu=W['UNG'][eidx]))
D=D[te.y.values[D.case.values]==1]
cnt=D.groupby('case').size(); D=D[D.case.isin(cnt[cnt>=2].index)]
X=np.c_[D.th,D.att,D.mu,np.log1p(freq[D.I]),diff[D.I]]
pm=LogisticRegression(max_iter=500).fit((X-X.mean(0))/X.std(0),D.obs); ps=pm.predict_proba((X-X.mean(0))/X.std(0))[:,1]
w=np.where(D.obs,1/np.clip(ps,0.02,1),0)
C={'cases':int(D.case.nunique()),'candidates':int(len(D)),'labelled':int(D.obs.sum()),'label_rate':float(D.obs.mean()),
   'propensity_deciles_all':np.percentile(ps,np.arange(0,101,10)).tolist(),
   'propensity_deciles_labelled':np.percentile(ps[D.obs.values],np.arange(0,101,10)).tolist(),
   'propensity_deciles_unlabelled':np.percentile(ps[~D.obs.values],np.arange(0,101,10)).tolist(),
   'min_propensity':float(ps.min()),'share_labelled_below_clip_0.02':float(np.mean(ps[D.obs.values]<0.02)),
   'weight_max':float(w.max()),'weight_mean_labelled':float(w[D.obs.values].mean()),
   'ESS_over_labelled':float(w.sum()**2/(w**2).sum()),'ESS_ratio':float(w.sum()**2/(w**2).sum()/D.obs.sum()),
   'overlap_share_labelled_with_p_above_0.5':float(np.mean(ps[D.obs.values]>0.5)),'overlap_share_unlabelled_with_p_below_0.1':float(np.mean(ps[~D.obs.values]<0.1)),
   'relabelled_as_negative':int((~D.obs).sum())}
# zero-IDCG cases per treatment (observed-only and ipw: cases with no positive labelled candidate; unobserved-as-negative: same)
pos_any=D.assign(r=np.nan_to_num(D.rel.values,nan=0.0)).groupby('case').r.max()>0
C['cases_with_zero_IDCG_excluded']=int((~pos_any).sum()); C['cases_retained_all_treatments']=int(pos_any.sum())
# primary set checks
Dp=D[D.obs]; g=Dp.groupby('case'); okp=(g.rel.transform('size')>=2)&(g.rel.transform('max')>0)
C['primary_cases_recomputed']=int(Dp[okp].case.nunique()); C['primary_cases_dropped_for_zero_IDCG']=int(Dp[(g.rel.transform('size')>=2)&~(g.rel.transform('max')>0)].case.nunique())
C['primary_cases_with_2_candidates']=int((Dp[okp].groupby('case').size()==2).sum())
out['C_ipw_and_ranking_rules']=C; print('C',C)
# ---------------- D. calendar windows ----------------
z=np.load('data/sorted.npz'); t=z['t']; fr=z['frac']; nn=z['n']; lidd=z['lid']
# Junyi time_done is in microseconds since the Unix epoch
def dt(x): return datetime.datetime.fromtimestamp(x/1e6, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
el=nn>=10
H=el&(fr<0.6); V=el&(fr>=0.6)&(fr<0.8); T=el&(fr>=0.8)
# decision points: time of first held-out attempt per eligible learner
ti=np.flatnonzero(T); kk=lidd[ti]; _,fi=np.unique(kk,return_index=True); dec=t[ti[fi]]
Dd={'log_range':[dt(t.min()),dt(t.max())],'history_range':[dt(t[H].min()),dt(t[H].max())],'validation_range':[dt(t[V].min()),dt(t[V].max())],'heldout_range':[dt(t[T].min()),dt(t[T].max())],
    'earliest_test_decision_point':dt(dec.min()),'median_test_decision_point':dt(np.median(dec)),
    'share_history_after_earliest_decision':float(np.mean(t[H]>dec.min())),'share_history_after_median_decision':float(np.mean(t[H]>np.median(dec))),
    'share_history_after_p10_decision':float(np.mean(t[H]>np.percentile(dec,10))),'eligible_learners':int(len(dec))}
out['D_calendar_windows']=Dd; print('D',Dd)
json.dump(out,open('supp_outputs.json','w'),indent=1,default=float)
print('DONE')
