from evalcore import *
import json
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(41)
out={}
Av,Cv,valA=load('val'); At,Ct,teA=load('test')
val=valA[valA.prior_att==0].reset_index(drop=True); te=teA[teA.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80)
topic=ex.topic.fillna(ex.name).values
# ---- topic-level new targets: no exercise of the target's topic in history
tcode=pd.factorize(topic)[0]; NT=tcode.max()+1
import scipy.sparse as sp
def topic_seen(A,df):
    M=A.tocoo(); T=sp.coo_matrix((M.data,(M.row,tcode[M.col])),shape=(A.shape[0],NT)).tocsr()
    return np.asarray(T[df.L.values,tcode[df['T'].values]]).ravel()>0
tv=topic_seen(Av,val); tt=topic_seen(At,te)
out['topic_new_share_test']=float(np.mean(~tt))
res={}
for nm,mv,mt in [('exercise-level new (primary)',np.ones(len(val),bool),np.ones(len(te),bool)),('topic-level new',~tv,~tt),('exercise-new, topic seen',tv,tt)]:
    v=val[mv].reset_index(drop=True); t=te[mt].reset_index(drop=True)
    Xv=base_X(v); Xt=base_X(t); yv=v.y.values; yt=t.y.values
    b=roc_auc_score(yt,fit_auc(Xv,yv,Xt,yt)); r={'n':len(t),'fail':float(yt.mean()),'base':float(b)}
    for m in ['BIN_EXP','ALL1','UNG','FPKG']:
        s1,_=load_scores(v,Av,Cv,W[m]); s2,_=load_scores(t,At,Ct,W[m])
        r[m]=float(roc_auc_score(yt,fit_auc(np.c_[Xv,s1],yv,np.c_[Xt,s2],yt))-b)
    res[nm]=r
out['newness']=res; print(pd.DataFrame(res).T.round(4))
# ---- edge-level dependency criterion (less direct than co-occurrence)
_,(rep,eidx,th)=load_scores(te,At,Ct,np.ones(NE))
L=te.L.values[rep]; I=SRC[eidx]; na=np.asarray(At[L,I]).ravel()
d=pd.DataFrame(dict(e=eidx,th=th,y=te.y.values[rep]))[na>=1]
d['low']=d.th<0.5
g=d.groupby(['e','low']).y.agg(['mean','size']).unstack()
ok=(g[('size',True)]>=20)&(g[('size',False)]>=20)
dep=(g[('mean',True)]-g[('mean',False)])[ok]
ei=dep.index.values
crit={'n_edges':int(len(ei)),'mean_dependency':float(dep.mean())}
bin_und=np.array([(s,d_) in Ex or (d_,s) in Ex for s,d_ in zip(SRC,DST)],float)
for nm,w in [('binary expert',bin_und),('graded (μ*)',W['UNG']),('interaction-derived (B8)',W['TRANS']),('Q',Qv)]:
    rho=spearmanr(w[ei],dep.values).correlation
    bs=[spearmanr(w[ei][s],dep.values[s]).correlation for s in [rng.integers(0,len(ei),len(ei)) for _ in range(500)]]
    crit[nm]=[float(rho),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
acc=Qv[ei]>=0.8-1e-9
crit['dependency_admitted']=float(dep.values[acc].mean()); crit['dependency_rejected']=float(dep.values[~acc].mean())
out['edge_dependency']=crit; print(crit)
# ---- ranking: observation rates and sensitivity to unobserved treatment
wm=(n>=10)&(frac>=0.8); wi=np.flatnonzero(wm)
key=lid[wi].astype(np.int64)*NC+e[wi]; uk,first=np.unique(key,return_index=True); fcor=c[wi[first]]
_,(rep,eidx,th)=load_scores(te,At,Ct,np.ones(NE))
L=te.L.values[rep]; I=SRC[eidx]; k2=L.astype(np.int64)*NC+I; pos=np.minimum(np.searchsorted(uk,k2),len(uk)-1); obs=uk[pos]==k2
na=np.asarray(At[L,I]).ravel()
D=pd.DataFrame(dict(case=rep,L=L,I=I,e=eidx,th=th,obs=obs,att=na>0,rel=np.where(obs,(~fcor[pos]).astype(float),np.nan),mu=W['UNG'][eidx],dom=area[DST[eidx]]))
D['mu_t']=pd.qcut(D.mu,3,labels=['low','mid','high'])
obsr={'by μ* tercile':D.groupby('mu_t').obs.mean().to_dict(),'by prior attempt':D.groupby('att').obs.mean().to_dict(),
      'by domain':D[D.dom.isin(['arithmetic','algebra','geometry'])].groupby('dom').obs.mean().to_dict(),'overall':float(D.obs.mean())}
out['observation_rates']=obsr; print(obsr)
D=D[te.y.values[D.case.values]==1]
cnt=D.groupby('case').size(); D=D[D.case.isin(cnt[cnt>=2].index)]
# propensity of observation
X=np.c_[D.th,D.att,D.mu,np.log1p(freq[D.I]),diff[D.I]]
pm=LogisticRegression(max_iter=500).fit((X-X.mean(0))/X.std(0),D.obs); ps=pm.predict_proba((X-X.mean(0))/X.std(0))[:,1]
D['ipw']=np.where(D.obs,1/np.clip(ps,0.02,1),0)
sc={'Random':rng.random(len(D)),'Learner deficit only':1-D.th.values,'Binary expert':np.maximum(0,W['BIN_EXP'][D.e]-D.th.values)*(W['BIN_EXP'][D.e]>0),
    'Graded ungated':np.maximum(0,W['UNG'][D.e]-D.th.values),'FPKG':np.maximum(0,W['FPKG'][D.e]-D.th.values)*(W['FPKG'][D.e]>0),
    'Transition × deficit':W['TRANS'][D.e]*(1-D.th.values)}
jit=rng.random(len(D))*1e-6
def ndcg(s,rel,wt,K=3):
    d=pd.DataFrame(dict(c=D.case.values,s=s+jit,r=rel,w=wt)).sort_values(['c','s'],ascending=[True,False])
    d['rk']=d.groupby('c').cumcount(); d['g']=d.r*d.w/np.log2(d.rk+2)*(d.rk<K)
    ideal=d.sort_values(['c','r'],ascending=[True,False]).copy(); ideal['rk']=ideal.groupby('c').cumcount(); ideal['g']=ideal.r*ideal.w/np.log2(ideal.rk+2)*(ideal.rk<K)
    dc=d.groupby('c').g.sum(); ic=ideal.groupby('c').g.sum(); m=ic>0
    return float((dc[m]/ic[m]).mean()), int(m.sum())
R={}
relneg=np.nan_to_num(D.rel.values,nan=0.0)
# observed-only (as before but with cases having >=2 candidates overall), unobserved-as-negative, IPW-weighted observed
Dobs=D.obs.values
for nm,s in sc.items():
    a,_=ndcg(np.where(Dobs,s,-1e9),np.where(Dobs,relneg,0),np.ones(len(D)))
    b,nb_=ndcg(s,relneg,np.ones(len(D)))
    c_,_=ndcg(np.where(Dobs,s,-1e9),np.where(Dobs,relneg,0),D.ipw.values)
    R[nm]=dict(observed_only=a,unobserved_as_negative=b,ipw_observed=c_)
out['ranking_sensitivity']=R; out['ranking_cases_all']=int(D.case.nunique()); print(pd.DataFrame(R).T.round(4))
json.dump(out,open('analysis_c.json','w'),default=float,indent=1)
