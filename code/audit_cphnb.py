"""Reconstruction of the CP-HNB counts from the Junyi release (S1.6). Run from code/ after run_all.sh.
Requires audit_counts.json and audit_pslc.json (streamed counts of the two logs) in ../ ."""
import pandas as pd, numpy as np, json
ex=pd.read_csv('data/junyi_Exercise_table.csv',encoding='utf-8-sig'); z=np.load('data/log.npz'); logged=set(ex.name.iloc[np.unique(z['e'])])
A=set()
for _,r in ex.iterrows():
    if isinstance(r.prerequisites,str):
        for p in r.prerequisites.split(','):
            p=p.strip()
            if p: A.add((p,r['name']))
recon={(p,c) for p,c in A if p in logged and c in logged}        # 899: distinct, both endpoints logged, no repair
fp=pd.read_csv('../edge_list/fpkg_edges_expert.csv'); fpk={(s,t) for s,t in zip(fp.source,fp.target)}   # 978
only_fpkg=sorted(fpk-recon); only_recon=sorted(recon-fpk); common=fpk&recon
rows=[dict(source=s,target=t,in_fpkg_978=True,in_cphnb_899=False,reason='endpoint not among the 722 logged exercises' if (s not in logged or t not in logged) else 'other') for s,t in only_fpkg]
rows+=[dict(source=s,target=t,in_fpkg_978=False,in_cphnb_899=True,reason='self-loop removed in FPKG' if s==t else 'cycle edge removed in FPKG (expert_repairs.txt)') for s,t in only_recon]
pd.DataFrame(rows).to_csv('../edge_difference_fpkg_vs_cphnb.csv',index=False)
out=dict(recon_899=len(recon),fpkg_978=len(fpk),common=len(common),only_fpkg=len(only_fpkg),only_cphnb=len(only_recon),only_cphnb_edges=only_recon,
         reasons_only_fpkg={k:sum(1 for r in rows if r['in_fpkg_978'] and r['reason']==k) for k in set(r['reason'] for r in rows if r['in_fpkg_978'])})
json.dump(out,open('../audit_cphnb.json','w'),indent=1,default=str); print(json.dumps(out,indent=1,default=str))
