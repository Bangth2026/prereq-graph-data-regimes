"""Estimator values for the worked case (learner 167, target division_0.5, held-out window start). Output: out_card.json"""
from evalcore import *
import json, pickle, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
ex_tab = pd.read_csv('data/junyi_Exercise_table.csv', encoding='utf-8-sig')
topic = pd.factorize(ex_tab.topic.fillna('none'))[0]; NT = topic.max() + 1
areaI = pd.factorize(ex_tab.area.fillna('none'))[0]; NA_ = areaI.max() + 1
NL = lid.max() + 1
Ttop = sp.csr_matrix((np.ones(NC), (np.arange(NC), topic)), shape=(NC, NT)); Tare = sp.csr_matrix((np.ones(NC), (np.arange(NC), areaI)), shape=(NC, NA_))
def prof_mats(mask):
    A = sp.coo_matrix((np.ones(mask.sum(), np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr()
    C = sp.coo_matrix((c[mask].astype(np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr(); return A, C
def lkt_feats(A, C, L, I, dif, frq):
    hn = np.asarray(A.sum(1)).ravel(); hc = np.asarray(C.sum(1)).ravel()
    tn = (A @ Ttop).toarray(); tc = (C @ Ttop).toarray(); an = (A @ Tare).toarray(); ac = (C @ Tare).toarray()
    ha = np.where(hn > 0, hc / np.maximum(hn, 1), 0.6)[L]
    tN = tn[L, topic[I]]; tA = np.where(tN > 0, tc[L, topic[I]] / np.maximum(tN, 1), ha)
    aN = an[L, areaI[I]]; aA = np.where(aN > 0, ac[L, areaI[I]] / np.maximum(aN, 1), ha)
    return np.c_[ha, np.log1p(hn[L]), 1 - dif[I], np.log1p(frq[I]), tA, np.log1p(tN), aA, np.log1p(aN), tN == 0]
m04 = (frac < 0.4) & (n >= 10); A4, C4 = prof_mats(m04)
f4 = np.bincount(e[m04], minlength=NC).astype(float); er4 = np.bincount(e[m04], weights=(~c[m04]).astype(float), minlength=NC)
d4 = np.where(f4 > 0, er4 / np.maximum(f4, 1), np.nanmean(er4 / np.maximum(f4, 1)))
lm = (frac >= 0.4) & (frac < 0.6) & (n >= 10); li = np.flatnonzero(lm)
k4 = lid[li].astype(np.int64) * NC + e[li]; _, fst = np.unique(k4, return_index=True); fr = li[fst]
fr = fr[np.asarray(A4[lid[fr], e[fr]]).ravel() == 0]; fr = np.random.default_rng(11).choice(fr, min(600_000, len(fr)), replace=False)
Xl = lkt_feats(A4, C4, lid[fr], e[fr], d4, f4); mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
LK = LogisticRegression(max_iter=500).fit((Xl - mu_l) / sd_l, c[fr].astype(int))
At, Ct, te = load('test')
bkt = pickle.load(open('data/bkt_params_train.pkl', 'rb'))
dk = np.load('data/dkt_states.npz'); drows = dk['rows']
L = 167; T = idx['division_0.5']
r = np.flatnonzero((lid == L) & (frac >= 0.8))[0]                   # first row of the held-out window
h = dk['H'][np.searchsorted(drows, r)].astype(np.float32)
dkt = lambda ex_: float(1 / (1 + np.exp(-(h @ dk['W'][ex_] + dk['b'][ex_]))))
rows = []
for k in np.flatnonzero(DST == T):
    i = int(SRC[k]); x = lkt_feats(At, Ct, np.array([L]), np.array([i]), diff, freq)
    rows.append(dict(prereq=names[i], mu=float(cr.mu[k]), attempts=int(At[L, i]), pop=float(1 - diff[i]),
                     bkt_L0=float(bkt[i][0]) if i in bkt else None,
                     lkt=float(LK.predict_proba((x - mu_l) / sd_l)[0, 1]), dkt=dkt(i)))
out = dict(learner=L, target='division_0.5', dkt_target=dkt(T), rows=rows)
json.dump(out, open('out_card.json', 'w'), indent=1); print(json.dumps(out, indent=1))
