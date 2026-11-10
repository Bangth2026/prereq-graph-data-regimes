import sys, pandas as pd, json
from collections import Counter
srt=Counter(); outc=Counter(); kcs=set(); students=set(); students_by_type={}; rows=0; pairs=Counter()
for ch in pd.read_csv(sys.stdin, sep='\t', usecols=['Anon Student Id','Student Response Type','Outcome','KC (Exercise)'], chunksize=2_000_000, dtype=str, quoting=3):
    rows+=len(ch); srt.update(ch['Student Response Type'].value_counts().to_dict()); outc.update(ch['Outcome'].fillna('NA').value_counts().to_dict())
    pairs.update((ch['Student Response Type']+'|'+ch['Outcome'].fillna('NA')).value_counts().to_dict())
    students.update(ch['Anon Student Id'].unique()); kcs.update(ch['KC (Exercise)'].dropna().unique())
    for t,g in ch.groupby('Student Response Type'): students_by_type.setdefault(t,set()).update(g['Anon Student Id'].unique())
    print(rows,file=sys.stderr)
res=dict(rows=rows,by_response_type=dict(srt),by_outcome=dict(outc),by_type_outcome=dict(pairs),students=len(students),students_by_type={k:len(v) for k,v in students_by_type.items()},kc_exercises=len(kcs))
json.dump(res,open('audit_pslc.json','w'),indent=1); print(json.dumps(res,indent=1))
