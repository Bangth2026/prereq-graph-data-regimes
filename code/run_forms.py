from evalcore import *
import json
Av,Cv,val=load('val'); At,Ct,te=load('test')
val=val[val.prior_att==0].reset_index(drop=True); te=te[te.prior_att==0].reset_index(drop=True)
W['FPKG']=gated(0.80)
def score(df,A,C,w,form):
    _,(rep,eidx,th)=load_scores(df,A,C,w)
    ww=w[eidx]; on=ww>0
    if form=='gps': g=np.maximum(0,ww-th)*on
    elif form=='wdef': g=ww*(1-th)*on
    elif form=='guide': g=ww*np.maximum(0,ww-th)*on
    s=np.bincount(rep,weights=g,minlength=len(df))
    return s
h=((val.L.values.astype(np.int64)*2654435761)%7)<4
Xbv=base_X(val); Xbt=base_X(te); yv=val.y.values; yt=te.y.values
pb_v=fit_auc(Xbv[h],yv[h],Xbv[~h],yv[~h]); pb_t=fit_auc(Xbv,yv,Xbt,yt)
rows=[]
for form in ['gps','wdef','guide']:
    for mth in ['BIN_EXP','ALL1','UNG','FPKG','TRANS']:
        sv=score(val,Av,Cv,W[mth],form); st=score(te,At,Ct,W[mth],form)
        pv=fit_auc(np.c_[Xbv,sv][h],yv[h],np.c_[Xbv,sv][~h],yv[~h]); pt=fit_auc(np.c_[Xbv,sv],yv,np.c_[Xbt,st],yt)
        rows.append(dict(form=form,method=mth,val_dAUC=roc_auc_score(yv[~h],pv)-roc_auc_score(yv[~h],pb_v),test_dAUC=roc_auc_score(yt,pt)-roc_auc_score(yt,pb_t),test_score_only=roc_auc_score(yt,st)))
R=pd.DataFrame(rows); print(R.round(4).to_string())
R.to_csv('out_forms.csv',index=False)
