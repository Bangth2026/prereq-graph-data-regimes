"""Step-4 contrasts (three components, decoupled threshold): B5 vs B9, B6 vs B9, with conditional and refitting bootstraps."""
from evalcore import *
import json
from sklearn.linear_model import LogisticRegression
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True); yv=val.y.values; yt=te.y.values; W['FPKG']=gated(0.80)
Lu,inv=np.unique(te.L.values,return_inverse=True); Lvu,invv=np.unique(val.L.values,return_inverse=True)
def expand(df,A_,C_):
    _,(rep,eidx,th)=load_scores(df,A_,C_,np.ones(NE)); L=df.L.values[rep]; I=SRC[eidx]; na=np.asarray(A_[L,I]).ravel(); return rep,eidx,th,na
EV=expand(val,Av,Cv); ET=expand(te,At,Ct)
def counts(df,E):
    rep,eidx,th,na=E; return np.c_[np.bincount(rep,(na<1).astype(float),len(df)),np.bincount(rep,(na>=1).astype(float),len(df))]
def sc_three(df,E,w,tau=None):
    rep,eidx,th,na=E; ww=w[eidx]; on=ww>0; obs=na>0
    d=np.where(obs&(th<ww),ww-th,0)*on if tau is None else np.where(obs,ww*np.maximum(0,tau-th),0)*on
    return np.c_[np.bincount(rep,d,len(df)),np.bincount(rep,np.where(obs&(th>=ww),ww,0)*on,len(df)),np.bincount(rep,np.where(~obs,ww,0)*on,len(df))]
XbCv=np.c_[base_X(val),counts(val,EV)]; XbCt=np.c_[base_X(te),counts(te,ET)]
def lboot(p1,p2,B=300,seed=808):
    r=np.random.default_rng(seed); o=[]
    for _ in range(B):
        sw=np.bincount(r.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; k=sw>0
        o.append(roc_auc_score(yt[k],p1[k],sample_weight=sw[k])-roc_auc_score(yt[k],p2[k],sample_weight=sw[k]))
    return [float(roc_auc_score(yt,p1)-roc_auc_score(yt,p2)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
def refit_boot(fv,ft,gv,gt,B=100,seed=809):
    r=np.random.default_rng(seed); o=[]
    for _ in range(B):
        wv=np.bincount(r.integers(0,len(Lvu),len(Lvu)),minlength=len(Lvu))[invv]; wt=np.bincount(r.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; kv=wv>0; kt=wt>0
        def fit(Xv,Xt):
            mu_=Xv[kv].mean(0); sd=Xv[kv].std(0)+1e-9; m_=LogisticRegression(max_iter=500).fit((Xv[kv]-mu_)/sd,yv[kv],sample_weight=wv[kv]); return m_.predict_proba((Xt-mu_)/sd)[:,1]
        p1=fit(fv,ft); p2=fit(gv,gt); o.append(roc_auc_score(yt[kt],p1[kt],sample_weight=wt[kt])-roc_auc_score(yt[kt],p2[kt],sample_weight=wt[kt]))
    return [float(np.mean(o)),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
F={m:(np.c_[XbCv,sc_three(val,EV,W[m],1.0)],np.c_[XbCt,sc_three(te,ET,W[m],1.0)]) for m in ['UNG','FPKG','ALL1']}
P={m:fit_auc(F[m][0],yv,F[m][1],yt) for m in F}
out={'B5_minus_B9_step4_conditional':lboot(P['UNG'],P['ALL1']),'B6_minus_B9_step4_conditional':lboot(P['FPKG'],P['ALL1']),
     'B5_minus_B9_step4_refit':refit_boot(*F['UNG'],*F['ALL1']),'B5_step4_refit_vs_counts':refit_boot(*F['UNG'],XbCv,XbCt),'B9_step4_refit_vs_counts':refit_boot(*F['ALL1'],XbCv,XbCt)}
print(out); json.dump(out,open('sens_batch3.json','w'),indent=1)
