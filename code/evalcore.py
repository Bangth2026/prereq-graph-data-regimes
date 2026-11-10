from common import *
import pickle, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
cr=crowd(); Ex,_=expert_edges()
SRC=cr.src.values.copy(); DST=cr.dst.values.copy(); NE=len(cr)
z=np.load('data/sorted.npz'); e=z['e'].astype(np.int32); lid=z['lid']; c=z['c']; frac=z['frac']; n=z['n']
TR=(frac<0.6)&(n>=10)
freq=np.bincount(e[TR],minlength=NC).astype(float); err=np.bincount(e[TR],weights=(~c[TR]).astype(float),minlength=NC)
diff=np.where(freq>0,err/np.maximum(freq,1),np.nanmean(err/np.maximum(freq,1)))
# transition baseline from training period only
ltr=lid[TR]; etr=e[TR]; same=ltr[1:]==ltr[:-1]; a=etr[:-1][same]; b=etr[1:][same]; m=a!=b
Tm=np.zeros((NC,NC)); np.add.at(Tm,(a[m],b[m]),1)
outc=Tm.sum(1)
def trans_weights(src=SRC,dst=DST):
    p=Tm[src,dst]/np.maximum(outc[src],1)
    return (pd.Series(p).rank(pct=True).values)  # percentile-rescaled to (0,1]
W={}
W['BIN_EXP']=np.array([(s,d) in Ex for s,d in zip(SRC,DST)],float)
W['BIN_MAJ']=cr.maj.values.astype(float)
W['ALL1']=np.ones(NE)
W['UNG']=cr.mu.values
W['TRANS']=trans_weights()
Qv=cr.Q.values
def gated(q): return cr.mu.values*(Qv>=q-1e-9)
indeg=np.bincount(DST,minlength=NC)
def load_scores(cases,A,C,w,theta=None):
    """sum over candidate in-edges of target: max(0, w - theta_s(i))"""
    # expand
    order=np.argsort(DST,kind='stable'); ds=DST[order]
    st=np.searchsorted(ds,cases['T'].values,'left'); en=np.searchsorted(ds,cases['T'].values,'right')
    k=en-st; rep=np.repeat(np.arange(len(cases)),k)
    eidx=order[np.concatenate([np.arange(a,b) for a,b in zip(st,en)])] if len(cases)<0 else order[(np.repeat(st,k)+ (np.arange(k.sum())-np.repeat(np.cumsum(k)-k,k)))]
    L=cases['L'].values[rep]; I=SRC[eidx]
    if theta is None:
        att=np.asarray(A[L,I]).ravel(); cor=np.asarray(C[L,I]).ravel()
        th=np.where(att>0,cor/np.maximum(att,1),0.0)
    else: th=theta(L,I)
    g=np.maximum(0,w[eidx]-th)*(w[eidx]>0)
    return np.bincount(rep,weights=g,minlength=len(cases)), (rep,eidx,th)
def base_X(df,allcases=False):
    X=np.c_[df.h_acc.values, np.log1p(df.h_n.values), diff[df['T']], np.log1p(freq[df['T']]), indeg[df['T']]]
    if allcases: X=np.c_[X, np.log1p(df.prior_att.values)]
    return X
def fit_auc(Xtr,ytr,Xte,yte):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    m=LogisticRegression(max_iter=500,C=1.0).fit((Xtr-mu)/sd,ytr)
    p=m.predict_proba((Xte-mu)/sd)[:,1]; return p
def load(tag):
    d=pickle.load(open(f'data/cases_{tag}.pkl','rb')); return d['A'],d['C'],d['cases'].reset_index(drop=True)
