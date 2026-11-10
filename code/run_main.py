from evalcore import *
import json, sys
rng=np.random.default_rng(2026)
Av,Cv,val=load('val'); At,Ct,te=load('test')
NEWONLY = sys.argv[1]=='new' if len(sys.argv)>1 else True
if NEWONLY: val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
allc=not NEWONLY
# ---- gate selection on validation (pre-specified rule): split val learners in halves,
# fit base+load on half 1, AUC on half 2; maximise AUC s.t. retained edges >= 50%
grid=np.round(np.arange(0.70,0.901,0.025),3)
h=((val.L.values.astype(np.int64)*2654435761)%7)<4
Xb=base_X(val,allc); sel=[]
for q in grid:
    w=gated(q); ret=(w>0).mean()
    sc,_=load_scores(val,Av,Cv,w)
    X=np.c_[Xb,sc]; p=fit_auc(X[h],val.y.values[h],X[~h],val.y.values[~h])
    sel.append((q,ret,roc_auc_score(val.y.values[~h],p)))
sel=pd.DataFrame(sel,columns=['q','retained','valAUC']); print(sel)
ok=sel[sel.retained>=0.5]; qstar=ok.q.values[ok.valAUC.values.argmax()]; print('q*',qstar)
W['FPKG']=gated(qstar)
# CI policy variant
AI=np.zeros(NC); 
tot=np.asarray(Av.sum(0)).ravel(); cor=np.asarray(Cv.sum(0)).ravel(); AI=np.where(tot>0,cor/np.maximum(tot,1),0)
Wout=np.bincount(SRC,weights=W['FPKG'],minlength=NC); RI=Wout/Wout.max(); CI=0.5*AI+0.5*RI
res={'qstar':float(qstar),'sel':sel.to_dict('list')}
# ---- fit on val, evaluate on test
methods=['BIN_EXP','BIN_MAJ','ALL1','UNG','FPKG','TRANS']
S_val={}; S_te={}
for mth in methods:
    S_val[mth],_=load_scores(val,Av,Cv,W[mth]); S_te[mth],_=load_scores(te,At,Ct,W[mth])
# CI policy: multiply per-edge gps by CI(src)
def ci_scores(df,A,C):
    _,(rep,eidx,th)=load_scores(df,A,C,W['FPKG'])
    g=np.maximum(0,W['FPKG'][eidx]-th)*(W['FPKG'][eidx]>0)*CI[SRC[eidx]]
    return np.bincount(rep,weights=g,minlength=len(df))
S_val['FPKG_CI']=ci_scores(val,Av,Cv); S_te['FPKG_CI']=ci_scores(te,At,Ct); methods.append('FPKG_CI')
# negative controls: permuted weights, degree-preserving rewiring (avg over 5 seeds -> report one representative + mean)
negs={}
for k in range(5):
    wp=rng.permutation(W['FPKG']); S_val[f'PERM{k}'],_=load_scores(val,Av,Cv,wp); S_te[f'PERM{k}'],_=load_scores(te,At,Ct,wp)
yv=val.y.values; yt=te.y.values
Xbv=base_X(val,allc); Xbt=base_X(te,allc)
P={}
P['BASE']=fit_auc(Xbv,yv,Xbt,yt)
for mth in [m for m in S_val]:
    P[mth]=fit_auc(np.c_[Xbv,S_val[mth]],yv,np.c_[Xbt,S_te[mth]],yt)
# score-only AUC and nested AUC
rows=[]
for mth in ['BASE']+list(S_val):
    a_only=roc_auc_score(yt,S_te[mth]) if mth!='BASE' else np.nan
    rows.append(dict(method=mth,auc_score_only=a_only,auc_nested=roc_auc_score(yt,P[mth]),ap_nested=average_precision_score(yt,P[mth]),
                     coverage=float((S_te[mth]>0).mean()) if mth!='BASE' else np.nan))
R=pd.DataFrame(rows); print(R.round(4).to_string())
# other single baselines
print('AUC learner deficit only (1-h_acc):',roc_auc_score(yt,1-te.h_acc.values),'target difficulty:',roc_auc_score(yt,diff[te['T']]),'popularity(-freq):',roc_auc_score(yt,-freq[te['T']]))
# paired learner bootstrap for key contrasts
Lu,inv=np.unique(te.L.values,return_inverse=True)
def boot(p1,p2,B=300):
    out=[]
    for b in range(B):
        wts=np.bincount(inv, minlength=0)  # placeholder
        s=rng.integers(0,len(Lu),len(Lu)); cntL=np.bincount(s,minlength=len(Lu)); sw=cntL[inv]
        m=sw>0
        out.append(roc_auc_score(yt[m],p1[m],sample_weight=sw[m])-roc_auc_score(yt[m],p2[m],sample_weight=sw[m]))
    out=np.array(out); return float(np.mean(out)),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
contr={}
for a,b in [('FPKG','BASE'),('UNG','BASE'),('BIN_EXP','BASE'),('ALL1','BASE'),('TRANS','BASE'),('FPKG','BIN_EXP'),('UNG','BIN_EXP'),('FPKG','UNG'),('FPKG','ALL1'),('FPKG','TRANS'),('FPKG_CI','FPKG'),('PERM0','BASE')]:
    d=roc_auc_score(yt,P[a])-roc_auc_score(yt,P[b]); ci=boot(P[a],P[b]); contr[f'{a}-{b}']=(d,ci[1],ci[2]); print(a,'-',b,round(d,4),np.round(ci[1:],4))
res['table']=R.to_dict('list'); res['contrasts']=contr; res['n_test']=len(te); res['n_learners']=int(te.L.nunique()); res['fail_rate']=float(yt.mean())
res['n_val']=len(val)
json.dump(res,open(f'out_main_{"new" if NEWONLY else "all"}.json','w'),indent=1,default=float)
