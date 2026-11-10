from evalcore import *
TP=(frac>=0.8)&(n>=10)
lt=lid[TP]; et=e[TP]; same=lt[1:]==lt[:-1]; a=et[:-1][same]; b=et[1:][same]; m=a!=b
M=np.zeros((NC,NC)); np.add.at(M,(np.minimum(a[m],b[m]),np.maximum(a[m],b[m])),1); M=M+M.T
cr['co']=np.log1p(M[SRC,DST]); cr['s']=[names[i] for i in SRC]; cr['d']=[names[i] for i in DST]; cr['exp']=W['BIN_EXP']
cr['co_pct']=cr.co.rank(pct=True); cr['mu_pct']=cr.mu.rank(pct=True)
rej=cr[cr.Q<0.8]
print('rejected edges',len(rej))
print('Rejected, highest held-out co-occurrence (rule: Q<0.80, top co):'); print(rej.sort_values('co',ascending=False).head(3)[['s','d','mu','Q','raw','co_pct','exp']].to_string())
acc=cr[cr.Q>=0.8]
print('Accepted with mu>=0.8 and lowest co:'); print(acc[acc.mu>=0.8].sort_values('co').head(3)[['s','d','mu','Q','raw','co_pct','exp']].to_string())
print('eligible learners',len(np.unique(lid[n>=10])),'events',int((n>=10).sum()))
print('mu median',cr.mu.median(),'Q stats',cr.Q.describe().round(3).to_dict())
print('pool targets',len(set(DST)),'pool sources',len(set(SRC)),'directed expert matches',int(W['BIN_EXP'].sum()))
