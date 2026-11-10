from evalcore import *
import json, sys
rng=np.random.default_rng(11)
lo,hi,tag=(0.8,1.01,'test')
At,Ct,te=load(tag)
te=te[te.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80)
# first-attempt correctness in window for all concepts
wm=(n>=10)&(frac>=lo)&(frac<hi); wi=np.flatnonzero(wm)
key=lid[wi].astype(np.int64)*NC+e[wi]; uk,first=np.unique(key,return_index=True); fcor=c[wi[first]]
_,(rep,eidx,th)=load_scores(te,At,Ct,np.ones(NE))
L=te.L.values[rep]; I=SRC[eidx]
k2=L.astype(np.int64)*NC+I; pos=np.searchsorted(uk,k2); pos=np.minimum(pos,len(uk)-1); obs=uk[pos]==k2
rel=np.where(obs,(~fcor[pos]).astype(float),np.nan)
df=pd.DataFrame(dict(case=rep,L=L,I=I,e=eidx,th=th,rel=rel,obs=obs))
print('candidate rows',len(df),'observed frac',obs.mean())
df=df[df.obs]
g=df.groupby('case'); ok=g.rel.transform('size')>=2
df=df[ok & (g.rel.transform('max')>0)]
df=df[te.y.values[df.case]==1]   # only cases where the learner then failed the new target
print('ranking cases',df.case.nunique(),'learners',df.L.nunique(),'mean cands',df.groupby('case').size().mean())
jit=rng.random(len(df))*1e-6
sc={'RANDOM':rng.random(len(df)),'POPULARITY':freq[df.I.values],'DEFICIT':1-df.th.values}
for m in ['BIN_EXP','UNG','FPKG','TRANS']:
    w=W[m][df.e.values]; sc[m]=np.maximum(0,w-df.th.values)*(w>0)
sc['UNG_wdef']=W['UNG'][df.e.values]*(1-df.th.values)
sc['TRANS_wdef']=W['TRANS'][df.e.values]*(1-df.th.values)
cases=df.case.values; relv=df.rel.values
order_c=np.argsort(cases,kind='stable')
def ndcg_p1(s,K=3):
    d=pd.DataFrame(dict(c=cases,s=s+jit,r=relv)).sort_values(['c','s'],ascending=[True,False])
    d['rk']=d.groupby('c').cumcount()
    d['gain']=d.r/np.log2(d.rk+2)*(d.rk<K)
    dcg=d.groupby('c').gain.sum()
    ideal=d.sort_values(['c','r'],ascending=[True,False]); ideal['rk']=ideal.groupby('c').cumcount()
    idcg=(ideal.r/np.log2(ideal.rk+2)*(ideal.rk<K)).groupby(ideal.c).sum()
    p1=d[d.rk==0].set_index('c').r
    return (dcg/idcg), p1
res={}
base=None
per={}
for m,s in sc.items():
    nd,p1=ndcg_p1(s); per[m]=(nd,p1); res[m]=dict(NDCG3=nd.mean(),P1=p1.mean())
R=pd.DataFrame(res).T; print(R.round(4))
# paired learner bootstrap
cl=df.groupby('case').L.first(); Lu,inv=np.unique(cl.values,return_inverse=True)
def boot(a,b,B=500):
    d=(per[a][0]-per[b][0]).values; out=[]
    for _ in range(B):
        w=np.bincount(rng.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; out.append((d*w).sum()/w.sum())
    return d.mean(),np.percentile(out,2.5),np.percentile(out,97.5)
C={}
for a,b in [('FPKG','DEFICIT'),('UNG','DEFICIT'),('UNG_wdef','DEFICIT'),('FPKG','BIN_EXP'),('UNG','BIN_EXP'),('TRANS_wdef','DEFICIT'),('DEFICIT','POPULARITY'),('DEFICIT','RANDOM')]:
    C[f'{a}-{b}']=boot(a,b); print(a,b,np.round(C[f'{a}-{b}'],4))
json.dump(dict(table=R.to_dict(),contrasts=C,n_cases=int(df.case.nunique()),n_learners=int(df.L.nunique())),open('out_rank.json','w'),default=float,indent=1)
