from common import *
z=np.load('data/sorted.npz'); e=z['e'].astype(np.int32); lid=z['lid']
same=lid[1:]==lid[:-1]; a=e[:-1][same]; b=e[1:][same]; m=a!=b; a=a[m]; b=b[m]
print('cross adjacent',len(a))
lo=np.minimum(a,b); hi=np.maximum(a,b)
M=np.zeros((NC,NC)); np.add.at(M,(lo,hi),1); M=M+M.T
np.save('data/cooc_all.npy',M)
cr=crowd(); Ex,rem=expert_edges(); print('expert',len(Ex),'removed',rem)
y=np.log1p(M[cr.src,cr.dst])
def r2(x,y):
    return np.corrcoef(x,y)[0,1]**2
link=np.array([(s,d) in Ex or (d,s) in Ex for s,d in zip(cr.src,cr.dst)]).astype(float)
print('binary R2',r2(link,y),'graded',r2(cr.mu,y),'maj',r2(cr.maj.astype(float),y))
g=cr.Q>=0.80; print('gated n',g.sum(),'R2',r2(cr.mu[g],y[g]))
