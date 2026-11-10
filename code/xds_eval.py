"""Replication of the evidence-state analyses on a second or third dataset (ASSISTments 2009, EdNet-KT1).

python xds_eval.py queries DIR   -> DIR/kt_queries.npz (rows, ex) for the KT models
python xds_eval.py eval DIR      -> DIR/out_xds.json

Same design as benchmark.py: positional 60/20/20 split per learner (learners with >= 10 attempts), cases = first
attempt at a unit never attempted before with at least one candidate relation and >= 5 history attempts, logistic
layer fitted on validation cases and scored once on held-out cases, one shared 300-replicate learner bootstrap.
Graphs: G = mined relations with graded weight pi_ij, U = same candidates with weight 1 (see xds_prep.py).
"""
import sys, json, time, pickle, itertools, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
MODE, DIR = sys.argv[1], sys.argv[2]; T0 = time.time()
z = np.load(f'{DIR}/sorted.npz'); e = z['e'].astype(np.int64); c = z['c'].astype(bool); lid = z['lid']; frac = z['frac']; n = z['n']; tt = z['t']
HAS_TIME = 'ASSISTments' not in json.load(open(f'{DIR}/prep.json'))['source']
units = pd.read_csv(f'{DIR}/units.csv'); NC = len(units); areaI = pd.factorize(units.area.astype(str))[0]; NA_ = areaI.max() + 1
R = pd.read_csv(f'{DIR}/relations.csv'); SRC = R.src.values.astype(np.int64); DST = R.dst.values.astype(np.int64); NE = len(R)
GR = {'G': R.w.values.astype(float), 'U': np.ones(NE)}
NL = lid.max() + 1; LSTART = np.r_[0, np.flatnonzero(lid[1:] != lid[:-1]) + 1]
TR = (frac < 0.6) & (n >= 10)
freq = np.bincount(e[TR], minlength=NC).astype(float); err = np.bincount(e[TR], weights=(~c[TR]).astype(float), minlength=NC)
diff = np.where(freq > 0, err / np.maximum(freq, 1), np.nanmean(np.where(freq > 0, err / np.maximum(freq, 1), np.nan)))
indeg = np.bincount(DST, minlength=NC); Dset = set(DST)
def mats(mask):
    A = sp.coo_matrix((np.ones(mask.sum(), np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr()
    C = sp.coo_matrix((c[mask].astype(np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr(); return A, C
def build(lo, hi):
    elig = n >= 10; A, C = mats(elig & (frac < lo))
    wi = np.flatnonzero(elig & (frac >= lo) & (frac < hi)); key = lid[wi].astype(np.int64) * NC + e[wi]
    uk, first = np.unique(key, return_index=True); fi = wi[first]
    df = pd.DataFrame(dict(L=lid[fi], T=e[fi], y=(~c[fi]).astype(np.int8), row=fi))
    df['prior_att'] = np.asarray(A[df.L.values, df['T'].values]).ravel()
    hn = np.asarray(A.sum(1)).ravel(); hc = np.asarray(C.sum(1)).ravel()
    df['h_n'] = hn[df.L]; df['h_acc'] = hc[df.L] / np.maximum(hn[df.L], 1)
    df = df[(df.prior_att == 0) & (df.h_n >= 5) & df['T'].isin(Dset)].reset_index(drop=True)
    ws = wi[np.r_[True, lid[wi][1:] != lid[wi][:-1]]]; df['ws'] = ws[np.searchsorted(lid[ws], df.L.values)]
    order = np.argsort(DST, kind='stable'); ds = DST[order]
    st_ = np.searchsorted(ds, df['T'].values, 'left'); en_ = np.searchsorted(ds, df['T'].values, 'right'); k = en_ - st_
    rep = np.repeat(np.arange(len(df)), k); eidx = order[np.repeat(st_, k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))]
    L = df.L.values[rep]; I = SRC[eidx]; na = np.asarray(A[L, I]).ravel(); nc_ = np.asarray(C[L, I]).ravel()
    # sequential counts: attempts of I by L in the window before the case row
    BIG = np.int64(1) << 27; comp = key * BIG + wi; o = np.argsort(comp, kind='stable'); cs = comp[o]; ccum = np.r_[0, np.cumsum(c[wi][o].astype(np.int64))]
    kp = L.astype(np.int64) * NC + I; fcase = df.row.values[rep]
    a_ = np.searchsorted(cs, kp * BIG); b_ = np.searchsorted(cs, kp * BIG + fcase)
    P = dict(rep=rep, eidx=eidx, L=L, I=I, na=na, nc=nc_, nq=na + (b_ - a_), cq=nc_ + (ccum[b_] - ccum[a_]), N=len(df), A=A, C=C)
    return df, P
val, PV = build(0.6, 0.8); te, PT = build(0.8, 1.01)
print('cases', len(val), len(te), 'pairs', len(PV['rep']), len(PT['rep']), round(time.time() - T0), flush=True)

if MODE == 'queries':
    Rr = []; X = []
    for df, P_ in [(val, PV), (te, PT)]:
        for rr, xx in [(df.ws.values[P_['rep']], P_['I']), (df.row.values[P_['rep']], P_['I']), (df.ws.values, df['T'].values), (df.row.values, df['T'].values)]:
            Rr.append(rr); X.append(xx)
    Rr = np.concatenate(Rr).astype(np.int64); X = np.concatenate(X).astype(np.int64)
    k = np.unique(Rr * 4096 + X); np.savez(f'{DIR}/kt_queries.npz', rows=k // 4096, ex=k % 4096); print('queries', len(k)); sys.exit()

out = dict(n_val=int(len(val)), n_test=int(len(te)), learners_test=int(te.L.nunique()), fail_rate_test=float(te.y.mean()),
           pairs_test=int(len(PT['rep'])), relations=int(NE), units=int(NC))
yv = val.y.values; yt = te.y.values; pos = yt == 1
# ---------------- estimators
pop_acc = 1 - diff
def fit_bkt():
    rng = np.random.default_rng(3); H_ = TR; Ls = np.unique(lid[H_]); samp = rng.choice(Ls, min(20000, len(Ls)), replace=False)
    m = H_ & np.isin(lid, samp); ee = e[m]; ll = lid[m]; cc = c[m].astype(np.int8)
    grid = np.array(list(itertools.product([0.1, 0.3, 0.5, 0.7, 0.9], [0.02, 0.05, 0.1, 0.2, 0.3], [0.05, 0.15, 0.25], [0.05, 0.1, 0.2])))
    L0, Tg, G, S = [grid[:, k][:, None] for k in range(4)]; par = {}
    for k in np.unique(ee):
        mk = ee == k
        if mk.sum() < 50: continue
        l = ll[mk]; x = cc[mk]; st = np.r_[0, np.flatnonzero(l[1:] != l[:-1]) + 1]; ln = np.minimum(np.diff(np.r_[st, len(l)]), 50)
        Xm = np.full((len(st), ln.max()), -1, np.int8)
        for j in range(ln.max()): v = ln > j; Xm[v, j] = x[st[v] + j]
        pL = np.repeat(L0, len(st), 1).astype(float); llk = np.zeros(len(grid))
        for j in range(Xm.shape[1]):
            o_ = Xm[:, j]; v = o_ >= 0; pc = pL * (1 - S) + (1 - pL) * G
            llk += np.where(v, np.where(o_ == 1, np.log(pc), np.log(1 - pc)), 0).sum(1)
            post = np.where(o_ == 1, pL * (1 - S) / pc, pL * S / (1 - pc)); pL = np.where(v, post + (1 - post) * Tg, pL)
        par[k] = grid[np.nanargmax(llk)]
    return par
bkt = fit_bkt(); bL0 = np.array([bkt[k][0] if k in bkt else np.nan for k in range(NC)]); bL0 = np.where(np.isnan(bL0), pop_acc, bL0)
Tare = sp.csr_matrix((np.ones(NC), (np.arange(NC), areaI)), shape=(NC, NA_))
def lkt_feats(A, C, L, I, dif, frq):
    hn = np.asarray(A.sum(1)).ravel(); hc = np.asarray(C.sum(1)).ravel(); an = (A @ Tare).toarray(); ac = (C @ Tare).toarray()
    ha = np.where(hn > 0, hc / np.maximum(hn, 1), 0.6)[L]; aN = an[L, areaI[I]]; aA = np.where(aN > 0, ac[L, areaI[I]] / np.maximum(aN, 1), ha)
    return np.c_[ha, np.log1p(hn[L]), 1 - dif[I], np.log1p(frq[I]), aA, np.log1p(aN)]
m04 = (frac < 0.4) & (n >= 10); A4, C4 = mats(m04)
f4 = np.bincount(e[m04], minlength=NC).astype(float); er4 = np.bincount(e[m04], weights=(~c[m04]).astype(float), minlength=NC)
d4 = np.where(f4 > 0, er4 / np.maximum(f4, 1), np.nanmean(np.where(f4 > 0, er4 / np.maximum(f4, 1), np.nan)))
li = np.flatnonzero((frac >= 0.4) & (frac < 0.6) & (n >= 10)); k4 = lid[li].astype(np.int64) * NC + e[li]; _, fst = np.unique(k4, return_index=True); fr = li[fst]
fr = fr[np.asarray(A4[lid[fr], e[fr]]).ravel() == 0]; fr = np.random.default_rng(11).choice(fr, min(600_000, len(fr)), replace=False)
Xl = lkt_feats(A4, C4, lid[fr], e[fr], d4, f4); mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
LK = LogisticRegression(max_iter=500).fit((Xl - mu_l) / sd_l, c[fr].astype(int))
dk = np.load(f'{DIR}/dkt_states.npz'); drows = dk['rows']; DH = dk['H'].astype(np.float32)
def dkt_at(rows, ex_):
    k = np.searchsorted(drows, rows); assert np.all(drows[k] == rows)
    return 1 / (1 + np.exp(-((DH[k] * dk['W'][ex_]).sum(1) + dk['b'][ex_])))
KTQ = {}
for nm in ['sakt', 'akt']:
    q = np.load(f'{DIR}/kt_{nm}.npz'); kq = q['rows'].astype(np.int64) * 4096 + q['ex']; o = np.argsort(kq); KTQ[nm] = (kq[o], q['p'][o]); del kq, o
def att_at(nm, rows, ex_):
    keys, p = KTQ[nm]; kk = rows.astype(np.int64) * 4096 + ex_; j = np.searchsorted(keys, kk)
    assert np.all(keys[j] == kk); return p[j].astype(float)
KT = {'DKT': dkt_at, 'SAKT': lambda r, x: att_at('sakt', r, x), 'AKT': lambda r, x: att_at('akt', r, x)}
for df, P_ in [(val, PV), (te, PT)]:
    P_['th_s'] = np.where(P_['na'] > 0, P_['nc'] / np.maximum(P_['na'], 1), np.nan); P_['th_q'] = np.where(P_['nq'] > 0, P_['cq'] / np.maximum(P_['nq'], 1), np.nan)
    P_['ZERO'] = np.zeros(len(P_['I'])); P_['POP'] = pop_acc[P_['I']]; P_['BKT'] = bL0[P_['I']]
    P_['LKT'] = LK.predict_proba((lkt_feats(P_['A'], P_['C'], P_['L'], P_['I'], diff, freq) - mu_l) / sd_l)[:, 1]
    for k_, f_ in KT.items():
        P_[k_ + '_s'] = f_(df.ws.values[P_['rep']], P_['I']); P_[k_ + '_q'] = f_(df.row.values[P_['rep']], P_['I'])
EST = ['ZERO', 'POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']
def est(P_, k, prof): return P_[f'{k}_{prof}'] if k in KT else P_[k]
# estimator accuracy on unobserved-at-window-start prerequisites attempted later in the held-out window
wi = np.flatnonzero((n >= 10) & (frac >= 0.8)); wk = lid[wi].astype(np.int64) * NC + e[wi]; uk, f_ = np.unique(wk, return_index=True); fi = wi[f_]
kp = PT['L'].astype(np.int64) * NC + PT['I']; j = np.minimum(np.searchsorted(uk, kp), len(uk) - 1); later = (uk[j] == kp) & (PT['na'] == 0)
lab = c[fi[j]].astype(int); sel = np.flatnonzero(later)[np.unique(kp[later], return_index=True)[1]]
out['estimator_accuracy'] = {k: dict(n=int(len(sel)), auc=(float(roc_auc_score(lab[sel], est(PT, k, 's')[sel])) if k != 'ZERO' else 0.5),
                                     brier=float(np.mean((est(PT, k, 's')[sel] - lab[sel]) ** 2)), observed=float(lab[sel].mean())) for k in EST}
print('acc', out['estimator_accuracy'], flush=True)
# ---------------- shared bootstrap
Lu, inv = np.unique(te.L.values, return_inverse=True); RB = np.random.default_rng(2027)
BW = np.stack([np.bincount(RB.integers(0, len(Lu), len(Lu)), minlength=len(Lu))[inv] for _ in range(300)]).astype(np.float32)
CA = {}
def wauc(p):
    if id(p) in CA: return CA[id(p)][1]
    o = np.argsort(p, kind='mergesort'); ps = p[o]; grp = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]; y = pos[o]; ng = grp[-1] + 1; r = np.empty(300)
    for b0 in range(0, 300, 25):                     # replicates in blocks of 25 to bound memory; same result as all at once
        Wb = BW[b0:b0 + 25][:, o]
        gp = np.stack([np.bincount(grp, Wb[b] * y, ng) for b in range(len(Wb))]); gn = np.stack([np.bincount(grp, Wb[b] * (~y), ng) for b in range(len(Wb))])
        r[b0:b0 + 25] = (gp * (np.cumsum(gn, 1) - gn + 0.5 * gn)).sum(1) / (gp.sum(1) * gn.sum(1))
    CA[id(p)] = (p, r); return r
def contrast(p1, p0):
    b = wauc(p1) - wauc(p0); return [float(roc_auc_score(yt, p1) - roc_auc_score(yt, p0)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def delong(p1, p2):
    m = pos.sum(); nn_ = (~pos).sum(); V = []
    for p in (p1, p2):
        X_ = p[pos]; Y_ = p[~pos]; tz = stats.rankdata(np.r_[X_, Y_])
        V.append(((tz[:m].sum() / m - (m + 1) / 2) / nn_, (tz[:m] - stats.rankdata(X_)) / nn_, 1 - (tz[m:] - stats.rankdata(Y_)) / m))
    S = np.cov(np.c_[V[0][1], V[1][1]].T) / m + np.cov(np.c_[V[0][2], V[1][2]].T) / nn_; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    zz = (V[0][0] - V[1][0]) / se; return dict(dAUC=float(V[0][0] - V[1][0]), z=float(zz), p=float(2 * stats.norm.sf(abs(zz))))
def flags(p, q=0.2):
    k = int(round(q * len(p))); top = np.zeros(len(p), bool); top[np.argsort(-p, kind='mergesort')[:k]] = True
    return float(yt[top].sum() / len(p) * 1000), (BW * (top & pos)).sum(1) / BW.sum(1) * 1000
def fdiff(p1, p0):
    a, ab = flags(p1); b, bb = flags(p0); d = ab - bb; return [a - b, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
# ---------------- features
cum = np.r_[0, np.cumsum(c.astype(np.int64))]
def gfeat(P_, w, k, prof, kind):
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = w[eidx]; th_o = P_['th_s'] if prof == 's' else P_['th_q']; obs = ~np.isnan(th_o)
    thh = est(P_, k, prof); th = np.where(obs, th_o, thh)
    if kind == 'coupled': return np.bincount(rep, np.maximum(0, ww - th), N)[:, None]
    return np.c_[np.bincount(rep, np.where(obs, ww * (1 - th), 0), N), np.bincount(rep, np.where(~obs, ww * (1 - thh), 0), N)]
def counts(P_, prof):
    nn_ = P_['na'] if prof == 's' else P_['nq']
    return np.c_[np.bincount(P_['rep'], (nn_ < 1).astype(float), P_['N']), np.bincount(P_['rep'], (nn_ >= 1).astype(float), P_['N'])]
def conf(df, base, ref, kt=None):
    X = np.c_[df.h_acc.values, np.log1p(df.h_n.values), diff[df['T']], np.log1p(freq[df['T']]), indeg[df['T']]]
    r = df.ws.values if ref == 's' else df.row.values
    if ref == 'q':
        k_ = r - df.ws.values; kc = cum[r] - cum[df.ws.values]; hn = df.h_n.values + k_; X[:, 0] = (df.h_acc.values * df.h_n.values + kc) / hn; X[:, 1] = np.log1p(hn)
    if 'R' in base:
        kk = np.minimum(10, r - LSTART[lid[r]]); X = np.c_[X, (cum[r] - cum[r - kk]) / np.maximum(kk, 1), (np.log1p(np.maximum(tt[df.row.values] - tt[r - 1], 0) / 1e3) if HAS_TIME else np.zeros(len(r)))]
    if kt is not None:
        pk = np.clip(KT[kt](r, df['T'].values), 1e-4, 1 - 1e-4); X = np.c_[X, np.log(pk / (1 - pk))]
    return X
def fit(Xv, Xt):
    mu = Xv.mean(0); sd = Xv.std(0) + 1e-9; m = LogisticRegression(max_iter=500).fit((Xv - mu) / sd, yv); return m.predict_proba((Xt - mu) / sd)[:, 1]
# ---------------- composition and estimator x use x protocol
E = {}
import os
ONLY_LM = os.environ.get('ONLY_LM') == '1'; FORCE_KT = os.environ.get('FORCE_KT')   # training-length check: learner models only, graph over B1RK with a fixed KT model
for pn, prof, ref, base in ([] if ONLY_LM else [('snapshot', 's', 's', 'B1'), ('fully_sequential', 'q', 'q', 'B1R')]):
    nn_ = PT['na'] if prof == 's' else PT['nq']; th = PT['th_s'] if prof == 's' else PT['th_q']; w = GR['G'][PT['eidx']]
    un = nn_ < 1; g0 = np.maximum(0, w - np.where(un, 0, th))
    Rz = dict(share_unobserved=float(un.mean()), score_mass_unobserved=float(g0[un].sum() / g0.sum()),
              cases_all_unobserved=float((np.bincount(PT['rep'], (~un).astype(float), PT['N']) == 0).mean()))
    Xbv = conf(val, base, ref); Xbt = conf(te, base, ref); XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
    p0 = fit(Xbv, Xbt); pC = fit(XCv, XCt); Rz.update(base_auc=float(roc_auc_score(yt, p0)), counts_vs_base=contrast(pC, p0))
    pp = {}
    for k in EST:
        for g in ['G', 'U']:
            pp[(k, 'c', g)] = fit(np.c_[Xbv, gfeat(PV, GR[g], k, prof, 'coupled')], np.c_[Xbt, gfeat(PT, GR[g], k, prof, 'coupled')])
            pp[(k, 'e', g)] = fit(np.c_[XCv, gfeat(PV, GR[g], k, prof, 'esd')], np.c_[XCt, gfeat(PT, GR[g], k, prof, 'esd')])
            Rz[f'{k}|pooled|{g}'] = contrast(pp[(k, 'c', g)], p0); Rz[f'{k}|esd|{g}'] = contrast(pp[(k, 'e', g)], pC)
        Rz[f'{k}|pooled|G-U'] = contrast(pp[(k, 'c', 'G')], pp[(k, 'c', 'U')]); Rz[f'{k}|esd|G-U'] = contrast(pp[(k, 'e', 'G')], pp[(k, 'e', 'U')])
    E[pn] = Rz; print(pn, {k: v for k, v in Rz.items() if 'esd|G' in k or 'pooled|G' in k}, round(time.time() - T0), flush=True)
out['estimator_use_protocol'] = E
# ---------------- stronger learner models and graph evidence over them
E2 = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    Rz = {}; P0 = {}
    P0['B1'] = fit(conf(val, 'B1', ref), conf(te, 'B1', ref)); P0['B1R'] = fit(conf(val, 'B1R', ref), conf(te, 'B1R', ref))
    for kt in KT: P0['B1RK_' + kt] = fit(conf(val, 'B1R', ref, kt), conf(te, 'B1R', ref, kt))
    for k_, p in P0.items(): Rz[f'{k_}_auc'] = float(roc_auc_score(yt, p))
    for k_ in P0:
        if k_ != 'B1': Rz[f'{k_}_vs_B1'] = contrast(P0[k_], P0['B1'])
    best = FORCE_KT or max(KT, key=lambda kt: Rz[f'B1RK_{kt}_auc']); Rz['best_kt'] = best
    Xbv = conf(val, 'B1R', ref, best); Xbt = conf(te, 'B1R', ref, best); XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
    pC = fit(XCv, XCt); Rz['counts_vs_B1RK'] = contrast(pC, P0['B1RK_' + best])
    for k in ['ZERO', best]:
        pg = {g: fit(np.c_[XCv, gfeat(PV, GR[g], k, prof, 'esd')], np.c_[XCt, gfeat(PT, GR[g], k, prof, 'esd')]) for g in ['G', 'U']}
        Rz[f'ESD-{k}|G_total_vs_B1RK'] = contrast(pg['G'], P0['B1RK_' + best]); Rz[f'ESD-{k}|G-U'] = contrast(pg['G'], pg['U'])
        Rz[f'ESD-{k}|G_flags20_vs_B1RK'] = fdiff(pg['G'], P0['B1RK_' + best])
        Rz[f'ESD-{k}|G_delong_vs_B1RK'] = delong(pg['G'], P0['B1RK_' + best])
    Rz['B1RK_flags20_vs_B1'] = fdiff(P0['B1RK_' + best], P0['B1']); Rz['B1_flags20'] = flags(P0['B1'])[0]
    E2[pn] = Rz; print('E2', pn, Rz, flush=True)
out['learner_models'] = E2; out['seconds'] = time.time() - T0
json.dump(out, open(f'{DIR}/out_xds' + (f'_{FORCE_KT}' if FORCE_KT else '') + '.json', 'w'), indent=1, default=float); print('done', round(time.time() - T0))
