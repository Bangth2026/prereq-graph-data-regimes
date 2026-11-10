from evalcore import *
import json
At,Ct,te=load('test'); te=te[te.prior_att==0].reset_index(drop=True)
_,(rep,eidx,th)=load_scores(te,At,Ct,np.ones(NE))
L=te.L.values[rep]; I=SRC[eidx]; att=np.asarray(At[L,I]).ravel()
d=pd.DataFrame(dict(mu=W['UNG'][eidx],th=th,att=att>0,y=te.y.values[rep],exp=W['BIN_EXP'][eidx]))
d['mu_t']=pd.qcut(d.mu,3,labels=['low','mid','high'])
g=d.groupby('mu_t').agg(n=('y','size'),attempted=('att','mean'),theta_attempted=('th',lambda x: x[d.loc[x.index,'att']].mean()),target_fail=('y','mean'))
print(g)
# within attempted prerequisites: does deficit predict target failure more when mu high?
a=d[d.att]
for t in ['low','mid','high']:
    x=a[a.mu_t==t]; print(t, 'AUC deficit->fail', round(roc_auc_score(x.y,1-x.th),4), len(x))
print('expert-linked attempted',d[d.exp==1].att.mean(),'not linked',d[d.exp==0].att.mean())
json.dump(dict(tab=g.reset_index().astype(str).to_dict('list')),open('out_diag.json','w'))
