from evalcore import *
import json, networkx as nx
rng=np.random.default_rng(2027)
out={}
TP=(frac>=0.8)&(n>=10)
def cooc(mask):
    lt=lid[mask]; et=e[mask]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
    M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); return np.log1p((M+M.T)[SRC,DST])
y=cooc(TP)
def r2(x,yy): return np.corrcoef(x,yy)[0,1]**2
mu=cr.mu.values; m_r=cr.m.values; acc=Qv>=0.8-1e-9
bin_und=np.array([(s,d) in Ex or (d,s) in Ex for s,d in zip(SRC,DST)])
# ---- structure of analysis set
G=nx.Graph(); G.add_edges_from(zip(SRC,DST))
out['n_src']=len(set(SRC)); out['n_tgt']=len(set(DST)); out['n_concepts']=len(set(SRC)|set(DST)); out['n_components']=nx.number_connected_components(G)
out['largest_component']=max(len(c) for c in nx.connected_components(G))
# ---- admitted vs rejected profile
deg=np.bincount(np.r_[SRC,DST],minlength=NC); 
prof={}
for nm,mk in [('admitted',acc),('rejected',~acc)]:
    prof[nm]=dict(n=int(mk.sum()),mean_mu=mu[mk].mean(),raters_mean=m_r[mk].mean(),
      raters_5_7=float(np.mean(m_r[mk]<=7)),raters_8_11=float(np.mean((m_r[mk]>=8)&(m_r[mk]<=11))),raters_12_15=float(np.mean(m_r[mk]>=12)),
      expert_linked=float(bin_und[mk].mean()),log_src_freq=float(np.log1p(freq[SRC[mk]]).mean()),log_tgt_freq=float(np.log1p(freq[DST[mk]]).mean()),
      degree=float((deg[SRC[mk]]+deg[DST[mk]]).mean()/2),
      arithmetic=float(np.mean(area[DST[mk]]=='arithmetic')),algebra=float(np.mean(area[DST[mk]]=='algebra')),geometry=float(np.mean(area[DST[mk]]=='geometry')),other=float(np.mean(~np.isin(area[DST[mk]],['arithmetic','algebra','geometry']))))
out['profile']=prof; print(pd.DataFrame(prof).round(3))
# ---- alternative selections of the same size (713), held-out R2 of graded weights on selected set
K=int(acc.sum()); sel={}
def top(score): return np.argsort(-score,kind='stable')[:K]
fq=np.log1p(freq[SRC]+freq[DST])
cands={'Q gate (consensus)':np.flatnonzero(acc),'Top mean rating (no Q)':top(mu),'Top rater count':top(m_r+rng.random(NE)*1e-3),
 'Top pair frequency':top(fq),'Top degree':top(deg[SRC]+deg[DST]+rng.random(NE)*1e-3),'Lowest SD (dispersion, no median)':top(-np.array([np.std([int(v) for v in r.split('_')]) for r in cr.raw])),
 'Lowest MAD':top(-cr.mad.values+rng.random(NE)*1e-6)}
for k,ix in cands.items():
    sel[k]=dict(R2=r2(mu[ix],y[ix]),mean_mu=mu[ix].mean(),expert=bin_und[ix].mean())
# Q gate within strata (stratified random matched on domain x rater group x expert link x freq tercile)
strata=pd.Series(list(zip(area[DST],np.digitize(m_r,[8,12]),bin_und,pd.qcut(fq,3,labels=False)))).astype(str).values
rr=[]
for _ in range(500):
    ix=[]
    for s_ in np.unique(strata):
        pool=np.flatnonzero(strata==s_); k=int((acc&(strata==s_)).sum())
        if k: ix+=list(rng.choice(pool,k,replace=False))
    ix=np.array(ix); rr.append(r2(mu[ix],y[ix]))
sel['Stratified random (domain × raters × expert link × frequency)']=dict(R2=float(np.mean(rr)),lo=float(np.percentile(rr,2.5)),hi=float(np.percentile(rr,97.5)))
# partial: within top-mu selection, does Q add? regress y on mu within admitted vs top-mu
out['selection']=sel; print(pd.DataFrame(sel).T.round(3))
# ---- Q by rater group; gate within each group
rg={}
for g,(lo,hi) in {'5-7':(5,7),'8-11':(8,11),'12-15':(12,15)}.items():
    mk=(m_r>=lo)&(m_r<=hi)
    a=mk&acc
    rr=[r2(mu[s],y[s]) for s in [rng.choice(np.flatnonzero(mk),a.sum(),replace=False) for _ in range(300)]] if a.sum()>5 else [np.nan]
    rg[g]=dict(n=int(mk.sum()),Q_mean=Qv[mk].mean(),Q_sd=Qv[mk].std(),retained=float(a.sum()/mk.sum()),R2_all=r2(mu[mk],y[mk]),R2_admitted=r2(mu[a],y[a]) if a.sum()>5 else np.nan,R2_random=float(np.mean(rr)),R2_random_hi=float(np.percentile(rr,97.5)))
out['rater_groups']=rg; print(pd.DataFrame(rg).T.round(3))
print('corr Q vs m',np.corrcoef(Qv,m_r)[0,1])
# ---- inter-rater reliability: ICC(1) one-way random, unequal group sizes
X=[np.array([int(v) for v in r.split('_')],float) for r in cr.raw]
N=sum(len(x) for x in X); k=len(X); gm=np.mean(np.concatenate(X))
ssb=sum(len(x)*(x.mean()-gm)**2 for x in X); ssw=sum(((x-x.mean())**2).sum() for x in X)
msb=ssb/(k-1); msw=ssw/(N-k); n0=(N-sum(len(x)**2 for x in X)/N)/(k-1)
icc1=(msb-msw)/(msb+(n0-1)*msw); icck=(msb-msw)/msb
out['icc1']=icc1; out['icc_k']=icck; out['n0']=n0; out['mean_within_sd']=float(np.mean([x.std(ddof=1) for x in X]))
out['frac_pairs_with_rating_le2_and_ge7']=float(np.mean([(x.min()<=2)&(x.max()>=7) for x in X]))
print('ICC1',icc1,'ICCk',icck,'n0',n0)
# ---- direction evidence independent of expert map: temporal order in training history
TR=(frac<0.6)&(n>=10)
lt=lid[TR]; et=e[TR]
key=lt.astype(np.int64)*NC+et; uk,fi=np.unique(key,return_index=True)  # first position of each learner-concept (sorted by time within learner)
firstpos=pd.Series(fi,index=uk)
def first_before(a_,b_):
    ka=lt.astype(np.int64)  # unused
    return None
# compute for each pair: among learners attempting both, share who attempted B (source) before A (target)
Lr=uk//NC; Cr=uk%NC
df=pd.DataFrame(dict(L=Lr,C=Cr,p=fi))
piv={}
need=set(SRC)|set(DST)
df=df[df.C.isin(need)]
byc={c:g.set_index('L').p for c,g in df.groupby('C')}
share=[];nboth=[]
for s_,d_ in zip(SRC,DST):
    a_=byc.get(s_); b_=byc.get(d_)
    if a_ is None or b_ is None: share.append(np.nan); nboth.append(0); continue
    j=a_.index.intersection(b_.index); nboth.append(len(j))
    share.append(np.mean(a_.loc[j].values<b_.loc[j].values) if len(j)>=30 else np.nan)
share=np.array(share); nboth=np.array(nboth)
ok=~np.isnan(share)
expdir=W['BIN_EXP']==1; exprev=np.array([(d,s) in Ex for s,d in zip(SRC,DST)])
out['direction']=dict(pairs_with_order_evidence=int(ok.sum()),share_source_first_all=float(np.nanmean(share)),
  frac_source_first_majority=float(np.mean(share[ok]>0.5)),
  expert_same=int(expdir.sum()),expert_reverse=int(exprev.sum()),expert_none=int((~expdir&~exprev).sum()),
  unlinked_pairs_with_evidence=int((ok&~expdir&~exprev).sum()),unlinked_source_first=float(np.mean(share[ok&~expdir&~exprev]>0.5)),
  expert_same_source_first=float(np.mean(share[ok&expdir]>0.5)),expert_reverse_source_first=float(np.mean(share[ok&exprev]>0.5)),
  ambiguous_40_60=int(np.sum(ok&(share>0.4)&(share<0.6))))
print(out['direction'])
np.save('data/dir_share.npy',share)
# ---- bootstrap schemes for graded-binary R2 difference
def boot(scheme,B=1000):
    res=[]
    us,iS=np.unique(SRC,return_inverse=True); ut,iT=np.unique(DST,return_inverse=True)
    comp={}; 
    for ci,cc in enumerate(nx.connected_components(G)):
        for v in cc: comp[v]=ci
    cid=np.array([comp[s_] for s_ in SRC]); uc,iC=np.unique(cid,return_inverse=True)
    for _ in range(B):
        if scheme=='pairs': w=np.bincount(rng.integers(0,NE,NE),minlength=NE)
        elif scheme=='source': w=np.bincount(rng.integers(0,len(us),len(us)),minlength=len(us))[iS]
        elif scheme=='target': w=np.bincount(rng.integers(0,len(ut),len(ut)),minlength=len(ut))[iT]
        elif scheme=='two-way': w=np.bincount(rng.integers(0,len(us),len(us)),minlength=len(us))[iS]*np.bincount(rng.integers(0,len(ut),len(ut)),minlength=len(ut))[iT]
        elif scheme=='component': w=np.bincount(rng.integers(0,len(uc),len(uc)),minlength=len(uc))[iC]
        ix=np.repeat(np.arange(NE),w)
        if len(ix)<50: continue
        res.append(r2(mu[ix],y[ix])-r2(bin_und[ix].astype(float),y[ix]))
    return float(np.percentile(res,2.5)),float(np.percentile(res,97.5))
bs={s_:boot(s_) for s_ in ['pairs','source','target','two-way','component']}
out['boot']=bs; out['diff']=r2(mu,y)-r2(bin_und.astype(float),y); print(bs)
json.dump(out,open('analysis_a.json','w'),default=float,indent=1)
