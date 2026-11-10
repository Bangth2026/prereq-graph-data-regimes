"""Convert junyi_ProblemLog_original.csv (streamed on stdin) to data/log.npz.
Stops with an error if row, learner or exercise counts differ from the published release."""
import sys, pandas as pd, numpy as np
EXPECTED = dict(rows=25_925_992, learners=247_606, exercises=722)
ex = pd.read_csv('data/junyi_Exercise_table.csv', encoding='utf-8-sig')
idx = {n: i for i, n in enumerate(ex['name'])}
parts = []
for ch in pd.read_csv(sys.stdin, usecols=['user_id','exercise','time_done','correct','suggested','review_mode'],
                      chunksize=2_000_000, dtype={'user_id':'int32','exercise':'str','time_done':'int64'}):
    e = ch['exercise'].map(idx)
    if e.isna().any():
        sys.exit(f'ERROR: {int(e.isna().sum())} attempts reference unknown exercises; download may be corrupt.')
    parts.append(pd.DataFrame({'u': ch.user_id.values, 'e': e.astype('int16').values, 't': ch.time_done.values,
        'c': (ch.correct.astype(str).str.lower() == 'true').values,
        's': (ch.suggested.astype(str).str.lower() == 'true').values,
        'r': (ch.review_mode.astype(str).str.lower() == 'true').values}))
    print(f'{sum(len(p) for p in parts):,} rows read', file=sys.stderr)
df = pd.concat(parts, ignore_index=True)
got = dict(rows=len(df), learners=int(df.u.nunique()), exercises=int(df.e.nunique()))
if got != EXPECTED:
    sys.exit(f'ERROR: counts {got} differ from expected {EXPECTED}; stopping on partial or altered data.')
np.savez('data/log.npz', **{k: df[k].values for k in df.columns})
print('OK', got)
