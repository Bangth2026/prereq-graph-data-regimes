import sys, pandas as pd, numpy as np
tot=dict(rows=0,sum_attempts=0,sum_hints=0,rows_hint=0,rows_review=0,rows_topic=0,rows_suggested=0,rows_pn_na=0,rows_multi_attempt=0)
users=set(); users_nonreview=set(); users_pn1=set(); users_nonhint=set(); users_correct_any=set(); users_prof=set(); ex=set(); ex_nonreview=set()
pt_na_rows=0
for ch in pd.read_csv(sys.stdin, usecols=['user_id','exercise','problem_type','problem_number','topic_mode','suggested','review_mode','correct','count_attempts','hint_used','count_hints','earned_proficiency'], chunksize=2_000_000, dtype={'user_id':'int32','exercise':'str'}, low_memory=False):
    tot['rows']+=len(ch)
    ca=pd.to_numeric(ch.count_attempts,errors='coerce').fillna(0); chh=pd.to_numeric(ch.count_hints,errors='coerce').fillna(0)
    tot['sum_attempts']+=int(ca.sum()); tot['sum_hints']+=int(chh.sum()); tot['rows_multi_attempt']+=int((ca>1).sum())
    hu=ch.hint_used.astype(str).str.lower()=='true'; rv=ch.review_mode.astype(str).str.lower()=='true'; tm=ch.topic_mode.astype(str).str.lower()=='true'; sg=ch.suggested.astype(str).str.lower()=='true'
    tot['rows_hint']+=int(hu.sum()); tot['rows_review']+=int(rv.sum()); tot['rows_topic']+=int(tm.sum()); tot['rows_suggested']+=int(sg.sum())
    pn=pd.to_numeric(ch.problem_number,errors='coerce'); tot['rows_pn_na']+=int(pn.isna().sum()); pt_na_rows+=int(ch.problem_type.isna().sum())
    users.update(ch.user_id.unique()); users_nonreview.update(ch.user_id[~rv].unique()); users_pn1.update(ch.user_id[pn==1].unique()); users_nonhint.update(ch.user_id[~hu].unique())
    users_correct_any.update(ch.user_id[ch.correct.astype(str).str.lower()=='true'].unique()); users_prof.update(ch.user_id[ch.earned_proficiency.astype(str).str.lower()=='true'].unique())
    ex.update(ch.exercise.unique()); ex_nonreview.update(ch.exercise[~rv].unique())
    print(tot['rows'],file=sys.stderr)
tot.update(users=len(users),users_nonreview=len(users_nonreview),users_with_problem_number_1=len(users_pn1),users_nonhint=len(users_nonhint),users_any_correct=len(users_correct_any),users_any_proficiency=len(users_prof),exercises=len(ex),exercises_nonreview=len(ex_nonreview),rows_problem_type_na=pt_na_rows)
tot['rows_plus_hints']=tot['rows']+tot['sum_hints']; tot['rows_plus_hint_rows']=tot['rows']+tot['rows_hint']
import json; json.dump(tot,open('audit_counts.json','w'),indent=1); print(json.dumps(tot,indent=1))
