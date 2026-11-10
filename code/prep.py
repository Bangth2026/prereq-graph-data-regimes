import numpy as np, time
t0=time.time()
z=np.load('data/log.npz'); u=z['u']; e=z['e']; t=z['t']; c=z['c']; s=z['s']; r=z['r']
o=np.lexsort((t,u)); u=u[o]; e=e[o]; t=t[o]; c=c[o]; s=s[o]; r=r[o]
# per-learner position
chg=np.r_[True,u[1:]!=u[:-1]]; start=np.flatnonzero(chg); cnt=np.diff(np.r_[start,len(u)])
lid=np.cumsum(chg)-1
pos=np.arange(len(u))-start[lid]; n=cnt[lid]
frac=(pos)/n
np.savez('data/sorted.npz',u=u,e=e,t=t,c=c,s=s,r=r,lid=lid.astype(np.int32),frac=frac.astype(np.float32),n=n.astype(np.int32))
print(time.time()-t0, len(start), np.percentile(cnt,[10,25,50,75,90,99]))
