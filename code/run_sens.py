from evalcore import *
import json
rng=np.random.default_rng(99)
W['FPKG']=gated(0.80)
out={}
Av,Cv,val0=load('val')
def run(val,te,At,Ct,ycol='y',allc=False,methods=('BIN_EXP','ALL1','UNG','FPKG','TRANS'),extraW=None):
    Xbv=base_X(val,allc); Xbt=base_X(te,allc); yv=val[ycol].values; yt=te[ycol].values
    if ycol=='y5': yv=(yv>=0.5).astype(int); yt=(yt>=0.5).astype(int)
    pb=fit_auc(Xbv,yv,Xbt,yt); ab=roc_auc_score(yt,pb); r={'BASE':ab}
    Ws=dict(W); 
    if extraW: Ws.update(extraW)
    for m in methods:
        sv,_=load_scores(val,Av,Cv,Ws[m]); st,_=load_scores(te,At,Ct,Ws[m])
        r[m]=roc_auc_score(yt,fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt))-ab
    r['n']=len(te); return r
valN=val0[val0.prior_att==0].reset_index(drop=True)
# temporal windows
for tag in ['test','t70','t90']:
    At,Ct,te=load(tag); teN=te[te.prior_att==0].reset_index(drop=True)
    out[f'window_{tag}']=run(valN,teN,At,Ct); print(tag,{k:round(v,4) for k,v in out[f'window_{tag}'].items()})
At,Ct,te=load('test'); teN=te[te.prior_att==0].reset_index(drop=True)
out['outcome_y5']=run(valN,teN,At,Ct,'y5'); print('y5',{k:round(v,4) for k,v in out['outcome_y5'].items()})
out['all_targets']=run(val0,te,At,Ct,allc=True); print('all',{k:round(v,4) for k,v in out['all_targets'].items()})
# negative controls: degree-preserving rewiring (swap sources) and label permutation
rw={}
for k in range(3):
    p=rng.permutation(NE); 
    # rewired graph: edges (SRC[p], DST) keep in/out degree sequences; weights follow edges
    rw[f'REWIRE{k}']=None
rew=[]
for k in range(3):
    p=rng.permutation(NE); src2=SRC[p]
    import evalcore as ec
    old=ec.SRC.copy(); ec.SRC[:]=src2
    sv,_=load_scores(valN,Av,Cv,W['FPKG']); st,_=load_scores(teN,At,Ct,W['FPKG'])
    ec.SRC[:]=old
    Xbv=base_X(valN); Xbt=base_X(teN); pb=fit_auc(Xbv,valN.y.values,Xbt,teN.y.values)
    rew.append(roc_auc_score(teN.y.values,fit_auc(np.c_[Xbv,sv],valN.y.values,np.c_[Xbt,st],teN.y.values))-roc_auc_score(teN.y.values,pb))
out['rewire_dAUC']=rew; print('rewire',np.round(rew,4))
yp=rng.permutation(teN.y.values); sv,_=load_scores(valN,Av,Cv,W['FPKG']); st,_=load_scores(teN,At,Ct,W['FPKG'])
out['labelperm_AUC']=roc_auc_score(yp,fit_auc(np.c_[base_X(valN),sv],valN.y.values,np.c_[base_X(teN),st],yp)); print('label perm AUC',out['labelperm_AUC'])
# leave-one-domain-out (graph-level selection on val-period proxy, learner-level test in held-out domain)
VP=(frac>=0.6)&(frac<0.8)&(n>=10); TP=(frac>=0.8)&(n>=10)
def cooc(mask):
    lt=lid[mask]; et=e[mask]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
    M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); return np.log1p((M+M.T)[SRC,DST])
yv=cooc(VP); yt=cooc(TP)
def r2(x,y): return np.corrcoef(x,y)[0,1]**2
darea=area[DST]; grid=np.round(np.arange(0.70,0.901,0.025),3)
lodo=[]
for A in ['arithmetic','algebra','geometry']:
    tr=np.isin(darea,[a for a in ['arithmetic','algebra','geometry'] if a!=A]); ho=darea==A
    best=None
    for q in grid:
        g=tr&(Qv>=q-1e-9)
        if g.sum()<0.5*tr.sum(): continue
        s=r2(cr.mu.values[g],yv[g])
        if best is None or s>best[1]: best=(q,s)
    q=best[0]; g=ho&(Qv>=q-1e-9)
    rr=[r2(cr.mu.values[s],yt[s]) for s in [rng.choice(np.flatnonzero(ho),g.sum(),replace=False) for _ in range(200)]]
    wq=cr.mu.values*(Qv>=q-1e-9)
    tA=teN[area[teN['T']]==A].reset_index(drop=True); vA=valN[area[valN['T']]!=A].reset_index(drop=True)
    lr=run(vA,tA,At,Ct,methods=('BIN_EXP','UNG','FPKGq'),extraW={'FPKGq':wq})
    lodo.append(dict(heldout=A,q=q,pairs=int(ho.sum()),retained=int(g.sum()),coverage=g.sum()/ho.sum(),R2_bin=r2(np.array([(s,d) in Ex or (d,s) in Ex for s,d in zip(SRC[ho],DST[ho])],float),yt[ho]),
        R2_graded=r2(cr.mu.values[ho],yt[ho]),R2_gated=r2(cr.mu.values[g],yt[g]),R2_rand=np.mean(rr),dAUC_bin=lr['BIN_EXP'],dAUC_ung=lr['UNG'],dAUC_fpkg=lr['FPKGq'],n_cases=lr['n']))
L=pd.DataFrame(lodo); print(L.round(4).to_string()); out['lodo']=L.to_dict('list')
json.dump(out,open('out_sens.json','w'),default=float,indent=1)
