"""Supplementary sensitivity analyses (all exploratory; the held-out window is reused).
A. Independent mastery threshold: w * max(0, tau - theta) with tau chosen on validation only,
   against GPS = max(0, mu* - theta); same cases, covariates, fit; two missing-evidence rules.
B. Confidence intervals for every learner-level increment of Table 12 (B4, B7, permutations).
C. Monotone re-encodings of the ordinal rating scale: graph-level r^2 and primary learner-level dAUC of B5.
D. Calendar-time walk-forward check of the primary learner-level comparison.
Writes sens_batch1.json.
"""
from evalcore import *
import json, datetime, scipy.sparse as sp
out={}
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
yv=val.y.values; yt=te.y.values
W['FPKG']=gated(0.80)
h=((val.L.values.astype(np.int64)*2654435761)%7)<4   # same split-half rule as gate selection
Lu,inv=np.unique(te.L.values,return_inverse=True)
def lboot(p1,p2,B=300,seed=505):
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
    rep,eidx,th,na=E
    return np.c_[np.bincount(rep,(na<1).astype(float),len(df)),np.bincount(rep,(na>=1).astype(float),len(df))]
def score(df,E,w,tau,rule):
    """rule 'theta0': unobserved theta=0 (primary); 'observed': unobserved contribute 0 (count-adjusted)."""
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0
    thr=ww if tau is None else np.full_like(ww,tau)
    g=np.maximum(0,thr-th)*(1.0 if tau is None else ww)*on
    if rule=='observed': g=np.where(na>=1,g,0)
    return np.bincount(rep,g,len(df))
# ---------------- A. independent threshold ----------------
taus=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
A={}
for rule in ['theta0','observed']:
    Xbv=base_X(val) if rule=='theta0' else np.c_[base_X(val),counts(val,EV)]
    Xbt=base_X(te) if rule=='theta0' else np.c_[base_X(te),counts(te,ET)]
    pb=fit_auc(Xbv,yv,Xbt,yt); pbv=fit_auc(Xbv[h],yv[h],Xbv[~h],yv[~h]); vb=roc_auc_score(yv[~h],pbv)
    for mth,w in [('B5 UNG',W['UNG']),('B6 FPKG',W['FPKG']),('B9 ALL1',W['ALL1']),('B3 BIN_EXP',W['BIN_EXP'])]:
        res={}
        # GPS (tau = w)
        sv=score(val,EV,w,None,rule); st=score(te,ET,w,None,rule)
        p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt); res['gps_tau_eq_w']=lboot(p,pb)
        # tau grid: select on validation half-split
        vals={}
        for tau in taus:
            sv=score(val,EV,w,tau,rule); pv=fit_auc(np.c_[Xbv,sv][h],yv[h],np.c_[Xbv,sv][~h],yv[~h]); vals[tau]=float(roc_auc_score(yv[~h],pv)-vb)
        tau_star=max(vals,key=lambda t:(vals[t],-t))
        sv=score(val,EV,w,tau_star,rule); st=score(te,ET,w,tau_star,rule)
        p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt)
        res['tau_grid_validation']=vals; res['tau_star']=tau_star; res['w_times_shortfall_tau_star']=lboot(p,pb)
        # also tau=1 (= w*(1-theta)) and tau=0.5 for reference on test
        for tau in [0.5,1.0]:
            sv=score(val,EV,w,tau,rule); st=score(te,ET,w,tau,rule); p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt); res[f'w_times_shortfall_tau_{tau}']=lboot(p,pb)
        A[f'{rule}|{mth}']=res; print(rule,mth,'gps',np.round(res['gps_tau_eq_w'],5),'tau*',tau_star,np.round(res['w_times_shortfall_tau_star'],5))
    # paired contrasts B5 vs B9 under tau*
out['A_independent_threshold']=A
# ---------------- B. CIs for Table 12 ----------------
Xbv=base_X(val); Xbt=base_X(te); pb=fit_auc(Xbv,yv,Xbt,yt)
_,(repv,eidv,thv)=load_scores(val,Av,Cv,W['FPKG']); _,(rept,eidt,tht)=load_scores(te,At,Ct,W['FPKG'])
tot=np.asarray(Av.sum(0)).ravel(); cor=np.asarray(Cv.sum(0)).ravel(); AI=np.where(tot>0,cor/np.maximum(tot,1),0)
Wout=np.bincount(SRC,weights=W['FPKG'],minlength=NC); RI=Wout/Wout.max(); CI=0.5*AI+0.5*RI
def ci_scores(df,rep,eidx,th):
    g=np.maximum(0,W['FPKG'][eidx]-th)*(W['FPKG'][eidx]>0)*CI[SRC[eidx]]; return np.bincount(rep,weights=g,minlength=len(df))
Bres={}
for nm,sv,st in [('B4 BIN_MAJ',load_scores(val,Av,Cv,W['BIN_MAJ'])[0],load_scores(te,At,Ct,W['BIN_MAJ'])[0]),
                 ('B7 FPKG_CI',ci_scores(val,repv,eidv,thv),ci_scores(te,rept,eidt,tht))]:
    p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt); Bres[nm]=lboot(p,pb); print(nm,np.round(Bres[nm],5))
rng=np.random.default_rng(2026)
perm=[]
for k in range(5):
    wp=rng.permutation(W['FPKG']); sv,_=load_scores(val,Av,Cv,wp); st,_=load_scores(te,At,Ct,wp)
    p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt); perm.append(lboot(p,pb,seed=600+k))
Bres['N1 permuted (5 seeds)']=perm; print('perm',np.round(perm,5))
# direct paired contrasts for the equivalence language
P={}
for m in ['BIN_EXP','ALL1','UNG','FPKG']:
    sv,_=load_scores(val,Av,Cv,W[m]); st,_=load_scores(te,At,Ct,W[m]); P[m]=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt)
Bres['B5-B9 primary']=lboot(P['UNG'],P['ALL1']); Bres['B6-B9 primary']=lboot(P['FPKG'],P['ALL1']); Bres['B5-B3 primary']=lboot(P['UNG'],P['BIN_EXP'])
out['B_table12_cis']=Bres
# ---------------- C. ordinal re-encodings ----------------
TP=(frac>=0.8)&(n>=10)
lt=lid[TP]; et=e[TP]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); ycc=np.log1p((M+M.T)[SRC,DST])
raw=[np.array([int(v) for v in r.split('_')],float) for r in cr.raw]
allr=np.concatenate(raw); ranks={v:(allr<v).mean()+0.5*(allr==v).mean() for v in range(10)}  # mid-rank quantile of each level
encs={'linear r/9':lambda x:x/9,'square (r/9)^2':lambda x:(x/9)**2,'sqrt (r/9)^0.5':lambda x:np.sqrt(x/9),
      'pooled mid-rank quantile':lambda x:np.vectorize(ranks.get)(x),'three-level (0-3,4-6,7-9 -> 0,0.5,1)':lambda x:np.where(x<=3,0.0,np.where(x<=6,0.5,1.0))}
Cres={}
for nm,f in encs.items():
    mu_e=np.array([f(x).mean() for x in raw])
    r2=np.corrcoef(mu_e,ycc)[0,1]**2
    sv,_=load_scores(val,Av,Cv,mu_e); st,_=load_scores(te,At,Ct,mu_e); p=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt)
    Cres[nm]=dict(r2_cooccurrence=float(r2),dAUC_primary_B5=lboot(p,pb),spearman_with_linear=float(pd.Series(mu_e).corr(pd.Series(W['UNG']),method='spearman')))
    print(nm,round(r2,4),np.round(Cres[nm]['dAUC_primary_B5'],5))
out['C_ordinal_encodings']=Cres
# ---------------- D. calendar-time walk-forward ----------------
z=np.load('data/sorted.npz'); t=z['t']; el=n>=10
d60,d80=np.percentile(t[el],[60,80])
def dt(x): return datetime.datetime.fromtimestamp(x/1e6,tz=datetime.timezone.utc).strftime('%Y-%m-%d')
NL=lid.max()+1; D=set(DST)
def build_cal(lo,hi):
    hmask=el&(t<lo); wmask=el&(t>=lo)&(t<hi)
    A_=sp.coo_matrix((np.ones(hmask.sum(),np.float32),(lid[hmask],e[hmask])),shape=(NL,NC)).tocsr()
    C_=sp.coo_matrix((c[hmask].astype(np.float32),(lid[hmask],e[hmask])),shape=(NL,NC)).tocsr()
    wi=np.flatnonzero(wmask); key=lid[wi].astype(np.int64)*NC+e[wi]; _,first=np.unique(key,return_index=True); fi=wi[first]
    L=lid[fi]; T=e[fi]; y=(~c[fi]).astype(np.int8); keep=np.isin(T,list(D)); prior=np.asarray(A_[L,T]).ravel()
    df=pd.DataFrame(dict(L=L[keep],T=T[keep],y=y[keep],prior_att=prior[keep]))
    hn=np.asarray(A_.sum(1)).ravel(); hc=np.asarray(C_.sum(1)).ravel(); df['h_n']=hn[df.L]; df['h_acc']=np.where(hn[df.L]>0,hc[df.L]/np.maximum(hn[df.L],1),np.nan)
    df=df[(df.h_n>=5)&(df.prior_att==0)].reset_index(drop=True)
    return A_,C_,df
Avc,Cvc,valc=build_cal(d60,d80); Atc,Ctc,tec=build_cal(d80,t.max()+1)
TRc=el&(t<d60); freqc=np.bincount(e[TRc],minlength=NC).astype(float); errc=np.bincount(e[TRc],weights=(~c[TRc]).astype(float),minlength=NC)
diffc=np.where(freqc>0,errc/np.maximum(freqc,1),np.nanmean(errc/np.maximum(freqc,1)))
def bx(df): return np.c_[df.h_acc.values,np.log1p(df.h_n.values),diffc[df['T']],np.log1p(freqc[df['T']]),indeg[df['T']]]
yvc=valc.y.values; ytc=tec.y.values; pbc=fit_auc(bx(valc),yvc,bx(tec),ytc)
Luc,invc=np.unique(tec.L.values,return_inverse=True)
def lbootc(p1,p2,B=300,seed=707):
    r=np.random.default_rng(seed); o=[]
    for _ in range(B):
        sw=np.bincount(r.integers(0,len(Luc),len(Luc)),minlength=len(Luc))[invc]; k=sw>0
        o.append(roc_auc_score(ytc[k],p1[k],sample_weight=sw[k])-roc_auc_score(ytc[k],p2[k],sample_weight=sw[k]))
    return [float(roc_auc_score(ytc,p1)-roc_auc_score(ytc,p2)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
Dres={'cutoff_validation':dt(d60),'cutoff_test':dt(d80),'log_end':dt(t.max()),'n_val_cases':int(len(valc)),'n_test_cases':int(len(tec)),'n_test_learners':int(tec.L.nunique()),'fail_rate_test':float(ytc.mean()),'base_auc':float(roc_auc_score(ytc,pbc))}
for m in ['BIN_EXP','BIN_MAJ','ALL1','UNG','FPKG','TRANS']:
    sv,_=load_scores(valc,Avc,Cvc,W[m]); st,_=load_scores(tec,Atc,Ctc,W[m]); p=fit_auc(np.c_[bx(valc),sv],yvc,np.c_[bx(tec),st],ytc)
    Dres[m]=lbootc(p,pbc); print('calendar',m,np.round(Dres[m],5))
# share of unobserved candidates and of gap score from unobserved, calendar design
_,(rep,eidx,th)=load_scores(tec,Atc,Ctc,W['FPKG']); na=np.asarray(Atc[tec.L.values[rep],SRC[eidx]]).ravel(); on=W['FPKG'][eidx]>0
g=np.maximum(0,W['FPKG'][eidx]-th)*on
Dres['share_candidates_unobserved']=float(np.mean(na[on]==0)); Dres['share_gap_score_from_unobserved']=float(g[na==0].sum()/max(g.sum(),1e-9))
out['D_calendar_walk_forward']=Dres; print(Dres)
json.dump(out,open('sens_batch1.json','w'),indent=1,default=float); print('DONE')
