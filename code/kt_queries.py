"""List every (row, exercise) pair for which benchmark.py needs a KT prediction: prerequisites and targets at window
start and immediately before each case, validation and held-out windows. Output: data/kt_queries.npz"""
from evalcore import *
Av, Cv, val = load('val'); At, Ct, te = load('test')
val = val[val.prior_att == 0].reset_index(drop=True); te = te[te.prior_att == 0].reset_index(drop=True)
BIG = np.int64(1) << 27
R = []; X = []
for df, A_, C_, lo, hi in [(val, Av, Cv, 0.6, 0.8), (te, At, Ct, 0.8, 1.01)]:
    _, (rep, eidx, _) = load_scores(df, A_, C_, np.ones(NE)); I = SRC[eidx]
    wm = (n >= 10) & (frac >= lo) & (frac < hi); wi = np.flatnonzero(wm)
    wkey = lid[wi].astype(np.int64) * NC + e[wi]; uk, first = np.unique(wkey, return_index=True); fi = wi[first]
    ws = wi[np.r_[True, lid[wi][1:] != lid[wi][:-1]]]
    fcase = fi[np.searchsorted(uk, df.L.values.astype(np.int64) * NC + df['T'].values)]
    wsL = ws[np.searchsorted(lid[ws], df.L.values)]
    for rr, xx in [(wsL[rep], I), (fcase[rep], I), (wsL, df['T'].values), (fcase, df['T'].values)]:
        R.append(rr); X.append(xx)
R = np.concatenate(R).astype(np.int64); X = np.concatenate(X).astype(np.int64)
k = np.unique(R * 4096 + X); np.savez('data/kt_queries.npz', rows=k // 4096, ex=k % 4096); print('queries', len(k))
