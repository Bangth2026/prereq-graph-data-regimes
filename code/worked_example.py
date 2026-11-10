from evalcore import *
import pickle, json
At,Ct,te=load('test'); te=te[te.prior_att==0].reset_index(drop=True)
bp=pickle.load(open('data/bkt_params.pkl','rb'))
tgt=idx['division_0.5']; ins=np.flatnonzero(DST==tgt)
print('candidates',[(names[SRC[k]],round(cr.mu[k],3),round(Qv[k],3),cr.raw[k]) for k in ins])
cs=te[te['T']==tgt]
H=(n>=10)&(frac<0.8)
rows=[]
for L in cs.L.values[:4000]:
    att=np.asarray(At[L,SRC[ins]].todense()).ravel(); 
    if (att>=3).sum()>=2: rows.append(L)
print('eligible learners',len(rows))
# rule: first eligible learner (lowest id) with at least 2 attempted candidates
L=sorted(rows)[0]
mk=H&(lid==L)
out=[]
for k in ins:
    i=SRC[k]; a=int(At[L,i]); cc=int(Ct[L,i]); pc=cc/a if a else None
    seq=c[mk&(e==i)]
    if i in bp:
        L0,T,G,S=bp[i]; pL=L0
        for o in seq:
            p=pL*(1-S)+(1-pL)*G; post=pL*(1-S)/p if o else pL*S/(1-p); pL=post+(1-post)*T
    else: pL=None
    st='accepted' if Qv[k]>=0.8-1e-9 else 'rejected'
    out.append(dict(prereq=names[i],mu=round(float(cr.mu[k]),3),Q=round(float(Qv[k]),3),ratings=cr.raw[k],status=st,expert=bool((i,tgt) in Ex),attempts=a,correct=cc,pc=None if pc is None else round(pc,3),bkt=None if pL is None else round(float(pL),3),
      gps_pc=round(max(0,cr.mu[k]-(pc or 0)),3) if st=='accepted' else None))
print('learner',L,'target failed?',int(cs[cs.L==L].y.values[0]))
print(pd.DataFrame(out).to_string())
json.dump(dict(learner=int(L),rows=out,failed=int(cs[cs.L==L].y.values[0])),open('worked_example.json','w'),default=str,indent=1)
