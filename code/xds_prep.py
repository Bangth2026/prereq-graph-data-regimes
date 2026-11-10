"""Prepare a replication dataset in the format used by the Junyi pipeline.

python xds_prep.py assist09 RAW_CSV OUTDIR      ASSISTments 2009-2010 skill builder (corrected, collapsed)
python xds_prep.py ednet    KT1_PARQUET QUESTIONS_PARQUET OUTDIR    EdNet-KT1 (one shard)
python xds_prep.py xes      XES3G5M_DIR OUTDIR      XES3G5M (question-level files + metadata)

Unit of analysis ("exercise"): ASSISTments skill_id (collapsed multi-skill ids kept as distinct units);
EdNet first tag of the question. Area: none (ASSISTments) / TOEIC part (EdNet).
Writes OUTDIR/sorted.npz (u, e, t, c, lid, frac, n), OUTDIR/units.csv (unit, name, area) and OUTDIR/prep.json.
XES3G5M: unit = first (leaf) knowledge concept of the question; area = second level of the KC route.
Full learner sequences are rebuilt by concatenating the 200-step pieces of train_valid_sequences_quelevel.csv
in file order (padding removed) and adding the untruncated sequences of test_quelevel.csv.
For XES3G5M, OUTDIR/relations.csv holds curriculum relations taken from the KC routes (no learner data used):
two KCs are related when their routes share at least the first three levels; siblings (same parent) get
weight 1, cousins (same grandparent) weight 0.5. Relations are undirected, so each related KC is a candidate
for the other. The mined relations below are written to OUTDIR/relations_mined.csv for comparison.
Relations (OUTDIR/relations.csv; relations_mined.csv for XES3G5M) are mined from training segments only (first 60 % of each learner's attempts):
for target j, candidates are the five units i with the highest pre-exposure rate
  pi_ij = #learners who attempted i before their first attempt at j / #learners who attempted j,
among units with precedence share >= 0.6 (i before j among learners who attempted both) and >= 30 such learners.
The graded weight is pi_ij; the unweighted variant gives every candidate weight 1.
"""
import sys, json, numpy as np, pandas as pd
kind = sys.argv[1]
if kind == 'assist09':
    raw, out = sys.argv[2], sys.argv[3]
    d = pd.read_csv(raw, encoding='latin1', low_memory=False, usecols=['order_id', 'user_id', 'skill_id', 'correct'])
    d = d.dropna(subset=['skill_id']); d['skill_id'] = d.skill_id.astype(str)
    d = d.sort_values(['user_id', 'order_id'], kind='stable')
    units = pd.DataFrame({'name': sorted(d.skill_id.unique())}); units['area'] = 'none'
    uidx = {s: i for i, s in enumerate(units.name)}
    u = d.user_id.values; e = d.skill_id.map(uidx).values; t = d.order_id.values; c = d.correct.values.astype(bool)
    meta = dict(source='ASSISTments 2009-2010 skill builder, corrected collapsed', rows_raw=int(len(d)))
elif kind == 'xes':
    raw, out = sys.argv[2], sys.argv[3]
    def seqs(f):
        d = pd.read_csv(f, usecols=['uid', 'concepts', 'responses', 'timestamps'])
        for uid, cs, rs, ts in zip(d.uid, d.concepts, d.responses, d.timestamps):
            cs = cs.split(','); rs = rs.split(','); ts = ts.split(',')
            for a, b, t_ in zip(cs, rs, ts):
                if a != '-1': yield uid, int(a.split('_')[0]), int(b), int(t_)
    rows_ = list(seqs(f'{raw}/question_level/train_valid_sequences_quelevel.csv')) + list(seqs(f'{raw}/question_level/test_quelevel.csv'))
    d = pd.DataFrame(rows_, columns=['uid', 'kc', 'c', 't']); del rows_
    km = json.load(open(f'{raw}/metadata/kc_routes_map.json')); n2i = {v: int(k) for k, v in km.items()}
    qs = json.load(open(f'{raw}/metadata/questions.json')); routes = {}
    for v in qs.values():
        for r in v['kc_routes']:
            p = tuple(r.split('----'))
            if p[-1] in n2i: routes.setdefault(n2i[p[-1]], set()).add(p)
    kcs = sorted(d.kc.unique()); uidx = {k: i for i, k in enumerate(kcs)}
    units = pd.DataFrame({'name': [km[str(k)] for k in kcs], 'area': [sorted(routes[k])[0][1] for k in kcs]})
    u = d.uid.values; e = d.kc.map(uidx).values; t = d.t.values; c = d.c.values.astype(bool)
    mono = d.groupby('uid', sort=False).t.apply(lambda x: bool((np.diff(x.values) >= 0).all()))
    meta = dict(source='XES3G5M (Liu et al., 2023), question-level files, original release', rows_raw=int(len(d)),
                sequences_time_ordered=float(mono.mean()))
    def cp(a, b):
        k = 0
        for x, y in zip(a, b):
            if x != y: break
            k += 1
        return k
    tree = []
    for j in kcs:
        for i in kcs:
            if i == j: continue
            up = min([len(rj) - cp(ri, rj) for rj in routes[j] for ri in routes[i] if cp(ri, rj) >= 3], default=99)
            if up <= 2: tree.append(dict(src=uidx[i], dst=uidx[j], w=1.0 if up == 1 else 0.5, level='sibling' if up == 1 else 'cousin'))
    TREE = pd.DataFrame(tree)
else:
    raw, qf, out = sys.argv[2], sys.argv[3], sys.argv[4]
    q = pd.read_parquet(qf); q['tag1'] = q.tags.astype(str).str.split(';').str[0]
    d = pd.read_parquet(raw, columns=['timestamp', 'question_id', 'subject_id', 'is_correct'])
    d = d.merge(q[['question_id', 'tag1', 'part']], on='question_id', how='inner')
    d = d.sort_values(['subject_id', 'timestamp'], kind='stable')
    units = q.groupby('tag1').part.agg(lambda s: s.mode().iloc[0]).reset_index().rename(columns={'tag1': 'name', 'part': 'area'})
    units = units[units.name.isin(d.tag1.unique())].reset_index(drop=True)
    uidx = {s: i for i, s in enumerate(units.name)}
    u = pd.factorize(d.subject_id)[0]; e = d.tag1.map(uidx).values; t = d.timestamp.values; c = d.is_correct.values.astype(bool)
    meta = dict(source='EdNet-KT1, first shard of the Hugging Face mirror mgor/EDNet (kt1/train-00000-of-00011.parquet)', rows_raw=int(len(d)))
import os; os.makedirs(out, exist_ok=True)
u = pd.factorize(u)[0]
chg = np.r_[True, u[1:] != u[:-1]]; st = np.flatnonzero(chg); cnt = np.diff(np.r_[st, len(u)])
lid = (np.cumsum(chg) - 1).astype(np.int32); pos = np.arange(len(u)) - st[lid]; nn = cnt[lid]
np.savez(f'{out}/sorted.npz', u=u, e=e.astype(np.int32), t=t, c=c, lid=lid, frac=(pos / nn).astype(np.float32), n=nn.astype(np.int32))
units.to_csv(f'{out}/units.csv', index_label='unit')
NC = len(units)
# ---- mined relations (training segments only)
TR = ((pos / nn) < 0.6) & (nn >= 10)
lt = lid[TR]; et = e[TR].astype(np.int64)
key = lt.astype(np.int64) * NC + et; uk, fi = np.unique(key, return_index=True)   # first training attempt per (learner, unit)
Lr = (uk // NC).astype(np.int64); Er = (uk % NC).astype(np.int64); Pr = np.flatnonzero(TR)[fi]
before = np.zeros((NC, NC)); both = np.zeros((NC, NC)); att = np.bincount(Er, minlength=NC).astype(float)
for l in np.unique(Lr):
    m = Lr == l
    if m.sum() < 2: continue
    E_ = Er[m]; P_ = Pr[m]; o = np.argsort(P_); E_ = E_[o]
    k = len(E_); idx = np.arange(k)
    B = (idx[:, None] < idx[None, :])                        # row unit attempted before column unit
    np.add.at(before, (E_[:, None].repeat(k, 1)[B], E_[None, :].repeat(k, 0)[B]), 1)
    np.add.at(both, (E_[:, None].repeat(k, 1), E_[None, :].repeat(k, 0)), 1)
share = np.where(both >= 30, before / np.maximum(both, 1), 0)
pi = before / np.maximum(att[None, :], 1)                        # pre-exposure rate of i before j
rows = []
for j in range(NC):
    cand = [i for i in range(NC) if i != j and share[i, j] >= 0.6]
    cand = sorted(cand, key=lambda i: -pi[i, j])[:5]
    for i in cand: rows.append(dict(src=i, dst=j, w=float(pi[i, j]), share=float(share[i, j]), both=int(both[i, j])))
R = pd.DataFrame(rows)
if kind == 'xes':
    R.to_csv(f'{out}/relations_mined.csv', index=False); TREE.to_csv(f'{out}/relations.csv', index=False)
    meta.update(mined_relations=int(len(R)), mined_targets=int(R.dst.nunique()), mined_weight_mean=float(R.w.mean()),
                tree_siblings=int((TREE.level == 'sibling').sum()), tree_cousins=int((TREE.level == 'cousin').sum()))
    R = TREE
else: R.to_csv(f'{out}/relations.csv', index=False)
meta.update(rows=int(len(u)), learners=int(lid.max() + 1), units=int(NC), learners_ge10=int((cnt >= 10).sum()),
            relations=int(len(R)), targets_with_candidates=int(R.dst.nunique()) if len(R) else 0,
            weight_mean=float(R.w.mean()) if len(R) else None)
json.dump(meta, open(f'{out}/prep.json', 'w'), indent=1); print(meta)
