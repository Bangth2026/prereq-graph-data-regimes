from evalcore import *
import json
rng=np.random.default_rng(1)
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80)
def feats(df,A,C,w):
    _,(rep,eidx,th)=load_scores(df,A,C,np.ones(NE))
    L=df.L.values[rep]; I=SRC[eidx]; att=np.asarray(A[L,I]).ravel()>0
    ww=w[eidx]
    g=np.where(att,np.maximum(0,ww-th),0)*(ww>0)
    un=np.where(~att,ww,0)   # weighted unattempted (w=1 -> count)
    return np.c_[np.bincount(rep,g,len(df)),np.bincount(rep,un,len(df))]
Xbv=base_X(val); Xbt=base_X(te); yv=val.y.values; yt=te.y.values
pb=fit_auc(Xbv,yv,Xbt,yt); ab=roc_auc_score(yt,pb)
h=((val.L.values.astype(np.int64)*2654435761)%7)<4
pbv=fit_auc(Xbv[h],yv[h],Xbv[~h],yv[~h]); abv=roc_auc_score(yv[~h],pbv)
res={}; P={}
Wx=dict(W); Wx['PERM']=rng.permutation(W['UNG'])
for m in ['BIN_EXP','ALL1','UNG','FPKG','TRANS','PERM']:
    fv=feats(val,Av,Cv,Wx[m]); ft=feats(te,At,Ct,Wx[m])
    pv=fit_auc(np.c_[Xbv,fv][h],yv[h],np.c_[Xbv,fv][~h],yv[~h])
    P[m]=fit_auc(np.c_[Xbv,fv],yv,np.c_[Xbt,ft],yt)
    res[m]=dict(val_dAUC=roc_auc_score(yv[~h],pv)-abv,test_dAUC=roc_auc_score(yt,P[m])-ab)
    print(m,{k:round(v,4) for k,v in res[m].items()})
Lu,inv=np.unique(te.L.values,return_inverse=True)
def boot(p1,p2,B=300):
    o=[]
    for _ in range(B):
        sw=np.bincount(rng.integers(0,len(Lu),len(Lu)),minlength=len(Lu))[inv]; m=sw>0
        o.append(roc_auc_score(yt[m],p1[m],sample_weight=sw[m])-roc_auc_score(yt[m],p2[m],sample_weight=sw[m]))
    return np.percentile(o,[2.5,97.5])
for a,b in [('UNG','BIN_EXP'),('UNG','ALL1'),('FPKG','UNG'),('UNG','PERM'),('TRANS','UNG')]:
    d=roc_auc_score(yt,P[a])-roc_auc_score(yt,P[b]); ci=boot(P[a],P[b]); res[f'{a}-{b}']=[d,*ci]; print(a,b,round(d,4),np.round(ci,4))
json.dump(res,open('out_explor.json','w'),default=float,indent=1)
