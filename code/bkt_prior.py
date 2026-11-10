from evalcore import *
import time, pickle, itertools
rng=np.random.default_rng(3)
srcs=np.unique(SRC); print('source concepts',len(srcs))
H=(n>=10)&(frac<0.6)&np.isin(e,srcs)
Ls=np.unique(lid[H]); samp=rng.choice(Ls,20000,replace=False)
m=H&np.isin(lid,samp)
ee=e[m]; ll=lid[m]; cc=c[m].astype(np.int8)
grid=np.array(list(itertools.product([0.1,0.3,0.5,0.7,0.9],[0.02,0.05,0.1,0.2,0.3],[0.05,0.15,0.25],[0.05,0.1,0.2])))
L0,T,G,S=[grid[:,k][:,None] for k in range(4)]
params={}; t0=time.time(); nev=0
for k in srcs:
    mk=ee==k
    if mk.sum()<50: continue
    l=ll[mk]; x=cc[mk]
    st=np.r_[0,np.flatnonzero(l[1:]!=l[:-1])+1]; ln=np.diff(np.r_[st,len(l)]); ln=np.minimum(ln,50)
    ns=len(st); Tm=ln.max()
    X=np.full((ns,Tm),-1,np.int8)
    for j in range(Tm):
        v=ln>j; X[v,j]=x[st[v]+j]
    pL=np.repeat(L0,ns,1).astype(float); ll_=np.zeros(len(grid))
    for j in range(Tm):
        o=X[:,j]; v=o>=0
        pc=pL*(1-S)+(1-pL)*G
        ll_+=np.where(v, np.where(o==1,np.log(pc),np.log(1-pc)),0).sum(1)
        post=np.where(o==1, pL*(1-S)/pc, pL*S/(1-pc))
        pL=np.where(v,post+(1-post)*T,pL)
    params[k]=grid[ll_.argmax()]; nev+=mk.sum()
print('fitted',len(params),'events',nev,'time',time.time()-t0)
pickle.dump(params,open('data/bkt_params_train.pkl','wb'))
