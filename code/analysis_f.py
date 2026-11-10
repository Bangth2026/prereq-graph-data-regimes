"""Supplementary analysis: composition of candidate prerequisites by evidence state and the share
of FPKG gap-score mass contributed by unobserved prerequisites under the theta = 0 convention."""
from evalcore import *
import json
At,Ct,te=load('test'); te=te[te.prior_att==0].reset_index(drop=True)
out={}
for name,w in [('graded_ungated',W['UNG']),('fpkg',gated(0.80)),('binary_expert',W['BIN_EXP'])]:
    _,(rep,eidx,th)=load_scores(te,At,Ct,np.ones(NE))
    L=te.L.values[rep]; I=SRC[eidx]; na=np.asarray(At[L,I]).ravel(); ww=w[eidx]; on=ww>0
    obs=na>0; mastered=obs&(th>=ww)&on; deficit=obs&(th<ww)&on; unobs=(~obs)&on
    gps=np.maximum(0,ww-th)*on
    out[name]=dict(n_relations_in_cases=int(on.sum()),share_unobserved=float(unobs.sum()/on.sum()),
        share_observed_deficit=float(deficit.sum()/on.sum()),share_observed_mastered=float(mastered.sum()/on.sum()),
        gps_mass_share_unobserved=float(gps[unobs].sum()/gps.sum()))
print(json.dumps(out,indent=1))
json.dump(out,open('analysis_f.json','w'),indent=1)
