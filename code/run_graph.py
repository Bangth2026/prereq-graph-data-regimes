from evalcore import *
import json
rng=np.random.default_rng(7)
TE=(frac>=0.8)&(n>=10)
lt=lid[TE]; et=e[TE]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); M=M+M.T
y=np.log1p(M[SRC,DST])
def r2(x,y): return np.corrcoef(x,y)[0,1]**2 if x.std()>0 else 0
Wg={'BIN_EXP_undirected':np.array([(s,d) in Ex or (d,s) in Ex for s,d in zip(SRC,DST)],float),
    'BIN_MAJ':W['BIN_MAJ'],'UNG':W['UNG'],'TRANS(train)':W['TRANS']}
out={}
firsts=SRC  # cluster by first endpoint
uc,inv=np.unique(SRC,return_inverse=True)
def cboot(x1,x2,mask=None,B=1000):
    r=[]
    for _ in range(B):
        s=rng.integers(0,len(uc),len(uc)); w=np.bincount(s,minlength=len(uc))[inv]
        idx=np.repeat(np.arange(NE),w)
        r.append(r2(x1[idx],y[idx])-r2(x2[idx],y[idx]))
    return np.percentile(r,[2.5,97.5])
for k,v in Wg.items(): out[k]=r2(v,y); print(k,round(out[k],3))
print('UNG-BIN',out['UNG']-out['BIN_EXP_undirected'],cboot(Wg['UNG'],Wg['BIN_EXP_undirected']))
print('TRANS-UNG',out['TRANS(train)']-out['UNG'],cboot(Wg['TRANS(train)'],Wg['UNG']))
both=np.c_[W['UNG'],W['TRANS']]
from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(both,y); print('UNG+TRANS R2',lr.score(both,y))
# confounders: frequency and difficulty from training period
conf=np.c_[np.log1p(freq[SRC]),np.log1p(freq[DST]),diff[SRC],diff[DST]]
for nm,X in [('conf',conf),('conf+BIN',np.c_[conf,Wg['BIN_EXP_undirected']]),('conf+UNG',np.c_[conf,W['UNG']]),('conf+UNG+BIN',np.c_[conf,W['UNG'],Wg['BIN_EXP_undirected']]),('conf+TRANS',np.c_[conf,W['TRANS']]),('conf+TRANS+UNG',np.c_[conf,W['TRANS'],W['UNG']])]:
    print(nm, round(LinearRegression().fit(X,y).score(X,y),3))
# gate trade-off on held-out proxy, with size-matched random subsets
rows=[]
for q in np.round(np.arange(0.70,0.901,0.05),2):
    g=Qv>=q-1e-9; r=r2(W['UNG'][g],y[g])
    rr=[]
    for _ in range(200):
        s=rng.choice(NE,g.sum(),replace=False); rr.append(r2(W['UNG'][s],y[s]))
    # frequency-matched: sample within deciles of pair frequency
    fq=np.log1p(freq[SRC]+freq[DST]); bins=np.digitize(fq,np.quantile(fq,np.linspace(0,1,11)[1:-1]))
    rm=[]
    for _ in range(200):
        s=np.concatenate([rng.choice(np.flatnonzero(bins==bb),min((g&(bins==bb)).sum(),(bins==bb).sum()),replace=False) for bb in range(10)])
        rm.append(r2(W['UNG'][s],y[s]))
    rows.append(dict(q=q,retained=int(g.sum()),R2=r,rand_mean=np.mean(rr),rand_lo=np.percentile(rr,2.5),rand_hi=np.percentile(rr,97.5),freqmatched_mean=np.mean(rm),freqmatched_lo=np.percentile(rm,2.5),freqmatched_hi=np.percentile(rm,97.5)))
G=pd.DataFrame(rows); print(G.round(3).to_string()); G.to_csv('out_gate_heldout.csv',index=False)
# negative controls graph-level
print('perm R2 mean',np.mean([r2(rng.permutation(W['UNG']),y) for _ in range(200)]))
json.dump(out,open('out_graph.json','w'))
