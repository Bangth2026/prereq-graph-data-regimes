from evalcore import *
import pickle, json
rng=np.random.default_rng(5)
bp=pickle.load(open('data/bkt_params.pkl','rb'))
At,Ct,te=load('test'); te=te[te.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80); w=W['FPKG']
Ls=rng.choice(te.L.unique(),3000,replace=False)
sub=te[te.L.isin(Ls)].reset_index(drop=True)
_,(rep,eidx,th)=load_scores(sub,At,Ct,w)
d=pd.DataFrame(dict(L=sub.L.values[rep],I=SRC[eidx],w=w[eidx],pc=th)); d=d[d.w>0]
d=d.sort_values('w',ascending=False).drop_duplicates(['L','I'])  # one entry per learner-prereq, strongest relation
# BKT theta from history (frac<0.8)
H=(n>=10)&(frac<0.8)&np.isin(lid,Ls)
hl=lid[H]; he=e[H]; hc=c[H]; ht=np.arange(H.sum())
hist=pd.DataFrame(dict(L=hl,I=he,c=hc)).groupby(['L','I']).c.apply(np.array)
def bkt(L,I):
    if I not in bp: return np.nan
    L0,T,G,S=bp[I]; pL=L0
    seq=hist.get((L,I))
    if seq is None: return L0
    for o in seq:
        pc=pL*(1-S)+(1-pL)*G
        post=pL*(1-S)/pc if o else pL*S/(1-pc)
        pL=post+(1-post)*T
    return pL
d['bkt']=[bkt(a,b) for a,b in zip(d.L,d.I)]
d['bkt']=d.bkt.fillna(d.pc)
def resamp(L,I):
    seq=hist.get((L,I))
    if seq is None: return 0.0
    return rng.choice(seq,len(seq)).mean()
d['pc_rs']=[resamp(a,b) for a,b in zip(d.L,d.I)]
att=np.asarray(At[d.L.values,d.I.values]).ravel(); d['att']=att
print('pearson pc-bkt (attempted pairs)',np.corrcoef(d.pc[att>0],d.bkt[att>0])[0,1],'means',d.pc[att>0].mean(),d.bkt[att>0].mean())
K=5
for col in ['pc','bkt','pc_rs']:
    d['g_'+col]=np.maximum(0,d.w-d[col])
def topk(col):
    x=d[d['g_'+col]>0].sort_values(['L','g_'+col],ascending=[True,False]); x=x.groupby('L').head(K)
    return x.groupby('L').I.apply(set)
tp,tb,tr=topk('pc'),topk('bkt'),topk('pc_rs')
def jac(a,b):
    ks=set(a.index)|set(b.index); j=[]
    for k in ks:
        A=a.get(k,set()); B=b.get(k,set())
        if A|B: j.append(len(A&B)/len(A|B))
    return np.array(j)
jb=jac(tp,tb); jr=jac(tp,tr)
print('Jaccard PC-BKT',jb.mean(),'PC-resampled',jr.mean(),'learners',len(jb))
# gap-status flip rate
flip=((d.g_pc>0)!=(d.g_bkt>0)).mean(); print('gap status flip',flip)
# stable vs sensitive and future association
wm=(n>=10)&(frac>=0.8); wi=np.flatnonzero(wm&np.isin(lid,Ls))
key=lid[wi].astype(np.int64)*NC+e[wi]; uk,first=np.unique(key,return_index=True); fcor=c[wi[first]]
k2=d.L.values.astype(np.int64)*NC+d.I.values; pos=np.minimum(np.searchsorted(uk,k2),len(uk)-1); obs=uk[pos]==k2
d['fut_err']=np.where(obs,(~fcor[pos]).astype(float),np.nan)
inp=np.array([i in tp.get(l,set()) for l,i in zip(d.L,d.I)]); inb=np.array([i in tb.get(l,set()) for l,i in zip(d.L,d.I)])
grp=np.where(inp&inb,'stable',np.where(inp,'PC-only',np.where(inb,'BKT-only','neither')))
d['grp']=grp; d['unatt']=att==0
S=d.groupby('grp').agg(n=('I','size'),observed=('fut_err',lambda x: x.notna().mean()),future_error=('fut_err','mean'),unattempted=('unatt','mean'))
print(S)
json.dump(dict(pearson=float(np.corrcoef(d.pc[att>0],d.bkt[att>0])[0,1]),jac_bkt=float(jb.mean()),jac_rs=float(jr.mean()),flip=float(flip),groups=S.reset_index().to_dict('list'),n_learners=int(len(jb)),ident=float(np.mean(jb==1))),open('out_stab.json','w'),default=float,indent=1)
print('identical sets',np.mean(jb==1))
