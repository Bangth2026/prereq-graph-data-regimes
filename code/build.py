# Build learner-level cases for temporal offline evaluation
from common import *
import scipy.sparse as sp, sys, pickle
z=np.load('data/sorted.npz'); e=z['e'].astype(np.int32); lid=z['lid']; c=z['c']; frac=z['frac']; n=z['n']; s=z['s']
MINEV=int(sys.argv[4]) if len(sys.argv)>4 else 10
elig=n>=MINEV
NL=lid.max()+1
cr=crowd(); D=set(cr.dst)
def hist_mats(mask):
    A=sp.coo_matrix((np.ones(mask.sum(),np.float32),(lid[mask],e[mask])),shape=(NL,NC)).tocsr()
    C=sp.coo_matrix((c[mask].astype(np.float32),(lid[mask],e[mask])),shape=(NL,NC)).tocsr()
    return A,C
def build(lo,hi,tag):
    hmask=elig&(frac<lo); wmask=elig&(frac>=lo)&(frac<hi)
    A,C=hist_mats(hmask)
    # first attempt per learner-concept in window
    wi=np.flatnonzero(wmask)
    key=lid[wi].astype(np.int64)*NC+e[wi]
    _,first=np.unique(key,return_index=True)   # data sorted by time within learner => first occurrence
    fi=wi[first]
    L=lid[fi]; T=e[fi]; y=(~c[fi]).astype(np.int8)
    # mean error over first up to 5 attempts in window (secondary outcome)
    order=np.argsort(key,kind='stable'); ks=key[order]; st=np.r_[0,np.flatnonzero(ks[1:]!=ks[:-1])+1]
    rank=np.arange(len(ks))-np.repeat(st,np.diff(np.r_[st,len(ks)]))
    sel=order[rank<5]; kk=key[sel]; err=(~c[wi[sel]]).astype(float)
    uk,inv=np.unique(kk,return_inverse=True); me=np.bincount(inv,err)/np.bincount(inv)
    y5=me[np.searchsorted(uk,L.astype(np.int64)*NC+T)]
    keep=np.isin(T,list(D))
    prior=np.asarray(A[L,T]).ravel()
    df=pd.DataFrame(dict(L=L[keep],T=T[keep],y=y[keep],y5=y5[keep],prior_att=prior[keep]))
    hn=np.asarray(A.sum(1)).ravel(); hc=np.asarray(C.sum(1)).ravel()
    df['h_n']=hn[df.L]; df['h_acc']=np.where(hn[df.L]>0,hc[df.L]/np.maximum(hn[df.L],1),np.nan)
    df=df[df.h_n>=5]
    pickle.dump(dict(A=A,C=C,cases=df),open(f'data/cases_{tag}.pkl','wb'))
    print(tag,len(df),df.L.nunique(),'new targets',(df.prior_att==0).sum(),'fail rate new',df.y[df.prior_att==0].mean())
lo,hi,tag=float(sys.argv[1]),float(sys.argv[2]),sys.argv[3]
build(lo,hi,tag)
