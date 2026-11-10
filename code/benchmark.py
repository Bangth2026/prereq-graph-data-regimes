"""Estimator benchmark, learner models and alert budgets (run inside code/ after the base pipeline, bkt_prior.py and kt_dkt.py).

All analyses use the reused held-out window and are exploratory. They answer three questions:
  E1  How much does the estimator chosen for unobserved prerequisites matter?  Five estimators
      (zero, population prior, BKT prior P(L0), cold-start logistic KT, DKT) x three ways of using them
      (coupled single score, decoupled single score, evidence-state decomposition ESD).
  E2  Do the conclusions survive stronger learner models?  Confounders + recency + DKT prediction for the target.
  E3  Do they survive the fully sequential protocol (prerequisite evidence AND confounders before each case)?
  E4  Practical size: failures flagged per 1,000 cases at fixed alert budgets; DeLong tests.
  E5  Specification selected on validation only, held-out evaluated once.
  E6  Exposure: platform-suggested versus self-selected target attempts.
All learner-bootstrap intervals in this file share ONE set of 300 resampling weights (seed 2027), models fixed,
so every interval reported for the same estimate is identical wherever it appears.
Output: out_benchmark.json
"""
from evalcore import *
import json, pickle, time
from scipy import stats
T0 = time.time()
z = np.load('data/sorted.npz'); tt = z['t']; sug = z['s']
ex_tab = pd.read_csv('data/junyi_Exercise_table.csv', encoding='utf-8-sig')
topic = pd.factorize(ex_tab.topic.fillna('none'))[0]; NT = topic.max() + 1
areaI = pd.factorize(ex_tab.area.fillna('none'))[0]; NA_ = areaI.max() + 1
out = {}
LSTART = np.r_[0, np.flatnonzero(lid[1:] != lid[:-1]) + 1]       # first row of every learner
Av, Cv, val = load('val'); At, Ct, te = load('test')
val = val[val.prior_att == 0].reset_index(drop=True); te = te[te.prior_att == 0].reset_index(drop=True)
yv = val.y.values; yt = te.y.values
W['FPKG'] = gated(0.80)
GR = {'B3': W['BIN_EXP'], 'B4': W['BIN_MAJ'], 'B5': W['UNG'], 'B6': W['FPKG'], 'B9': W['ALL1']}

# ------------------------------------------------------------------ shared bootstrap (fast weighted AUC)
Lu, inv = np.unique(te.L.values, return_inverse=True)
RB = np.random.default_rng(2027); NB = 300
BW = np.stack([np.bincount(RB.integers(0, len(Lu), len(Lu)), minlength=len(Lu))[inv] for _ in range(NB)]).astype(np.float32)
pos = yt == 1
def _prep(p):
    o = np.argsort(p, kind='mergesort'); ps = p[o]; grp = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]
    return o, grp
def wauc_all(p):
    """AUC under each of the NB bootstrap weightings (ties counted as 1/2)."""
    o, grp = _prep(p); y = pos[o]; Wb = BW[:, o]
    wp = Wb * y; wn = Wb * (~y)
    ng = grp[-1] + 1
    gp = np.zeros((NB, ng)); gn = np.zeros((NB, ng))
    for b in range(NB):
        gp[b] = np.bincount(grp, wp[b], ng); gn[b] = np.bincount(grp, wn[b], ng)
    cn = np.cumsum(gn, 1) - gn
    num = (gp * (cn + 0.5 * gn)).sum(1)
    return num / (gp.sum(1) * gn.sum(1))
CACHE = {}
def bs(p):
    k = id(p)
    if k not in CACHE: CACHE[k] = (p, wauc_all(p))
    return CACHE[k][1]
def contrast(p1, p0):
    d = float(roc_auc_score(yt, p1) - roc_auc_score(yt, p0)); b = bs(p1) - bs(p0)
    return [d, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
# DeLong (fast, Sun & Xu 2014) for paired AUCs
def _midrank(x): return stats.rankdata(x)
def delong(p1, p2):
    m = pos.sum(); nn_ = (~pos).sum(); V = []
    for p in (p1, p2):
        X = p[pos]; Y = p[~pos]; tx = _midrank(X); ty = _midrank(Y); tz = _midrank(np.r_[X, Y])
        auc = (tz[:m].sum() / m - (m + 1) / 2) / nn_
        V.append((auc, (tz[:m] - tx) / nn_, 1 - (tz[m:] - ty) / m))
    a = np.array([V[0][0], V[1][0]]); v10 = np.c_[V[0][1], V[1][1]]; v01 = np.c_[V[0][2], V[1][2]]
    S = np.cov(v10.T) / m + np.cov(v01.T) / nn_
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]); zz = (a[0] - a[1]) / se
    return dict(dAUC=float(a[0] - a[1]), se=float(se), z=float(zz), p=float(2 * stats.norm.sf(abs(zz))))

# ------------------------------------------------------------------ pair expansion and protocols
BIG = np.int64(1) << 27
def window_index(lo, hi):
    wm = (n >= 10) & (frac >= lo) & (frac < hi); wi = np.flatnonzero(wm)
    wkey = lid[wi].astype(np.int64) * NC + e[wi]
    comp = wkey * BIG + wi; order = np.argsort(comp, kind='stable'); cs = comp[order]
    ccum = np.r_[0, np.cumsum(c[wi][order].astype(np.int64))]
    uk, first = np.unique(wkey, return_index=True); fi = wi[first]
    ws = wi[np.r_[True, lid[wi][1:] != lid[wi][:-1]]]          # first row of each learner's window
    return cs, ccum, uk, fi, ws
def expand(df, A_, C_, lo, hi):
    _, (rep, eidx, _) = load_scores(df, A_, C_, np.ones(NE))
    L = df.L.values[rep]; I = SRC[eidx]
    na = np.asarray(A_[L, I]).ravel(); nc = np.asarray(C_[L, I]).ravel()
    cs, ccum, uk, fi, ws = window_index(lo, hi)
    kq = df.L.values.astype(np.int64) * NC + df['T'].values; p_ = np.searchsorted(uk, kq); assert np.all(uk[p_] == kq)
    fcase = fi[p_]                                              # row of the case (first attempt at the target)
    wsL = ws[np.searchsorted(lid[ws], df.L.values)]; assert np.all(lid[wsL] == df.L.values)
    kp = L.astype(np.int64) * NC + I
    lo_ = np.searchsorted(cs, kp * BIG); hi_ = np.searchsorted(cs, kp * BIG + fcase[rep])
    nq = na + (hi_ - lo_); cq = nc + (ccum[hi_] - ccum[lo_])
    return dict(rep=rep, eidx=eidx, L=L, I=I, na=na, nc=nc, nq=nq, cq=cq, fcase=fcase, ws=wsL, N=len(df))
t1 = time.time(); PV = expand(val, Av, Cv, 0.6, 0.8); PT = expand(te, At, Ct, 0.8, 1.01); print('expand', round(time.time() - t1), flush=True)

# ------------------------------------------------------------------ estimators for unobserved prerequisites
TR = (frac < 0.6) & (n >= 10)
pop_acc = 1 - diff                                              # population first-attempt accuracy, training segments
bkt = pickle.load(open('data/bkt_params_train.pkl', 'rb'))      # per-exercise (L0, T, G, S), fitted on training segments
bL0 = np.array([bkt[k][0] if k in bkt else np.nan for k in range(NC)])
bL0 = np.where(np.isnan(bL0), pop_acc, bL0)
# cold-start logistic KT (LKT): P(correct | learner profile, exercise) for an exercise the learner has not attempted.
# Trained on the training segments only: profile = attempts with frac < 0.4, labels = first attempts at new exercises in [0.4, 0.6).
import scipy.sparse as sp
NL = lid.max() + 1
Ttop = sp.csr_matrix((np.ones(NC), (np.arange(NC), topic)), shape=(NC, NT))
Tare = sp.csr_matrix((np.ones(NC), (np.arange(NC), areaI)), shape=(NC, NA_))
def prof_mats(mask):
    A = sp.coo_matrix((np.ones(mask.sum(), np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr()
    C = sp.coo_matrix((c[mask].astype(np.float32), (lid[mask], e[mask])), shape=(NL, NC)).tocsr()
    return A, C
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
newm = np.asarray(A4[lid[fr], e[fr]]).ravel() == 0; fr = fr[newm]
fr = np.random.default_rng(11).choice(fr, min(600_000, len(fr)), replace=False)
Xl = lkt_feats(A4, C4, lid[fr], e[fr], d4, f4); yl = c[fr].astype(int)
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
from sklearn.linear_model import LogisticRegression
LK = LogisticRegression(max_iter=500, C=1.0).fit((Xl - mu_l) / sd_l, yl)
out['lkt'] = dict(n_train=int(len(fr)), train_auc=float(roc_auc_score(yl, LK.predict_proba((Xl - mu_l) / sd_l)[:, 1])))
def lkt_theta(A, C, L, I):
    return LK.predict_proba((lkt_feats(A, C, L, I, diff, freq) - mu_l) / sd_l)[:, 1]
# DKT
dk = np.load('data/dkt_states.npz'); drows = dk['rows']; DH = dk['H'].astype(np.float32); DW = dk['W']; Db = dk['b']
def dkt_at(rows, ex_):
    k = np.searchsorted(drows, rows); assert np.all(drows[k] == rows)
    return 1 / (1 + np.exp(-((DH[k] * DW[ex_]).sum(1) + Db[ex_])))
KTQ = {}
for nm_ in ['sakt', 'akt']:
    q_ = np.load(f'data/kt_{nm_}.npz'); kk_ = q_['rows'].astype(np.int64) * 4096 + q_['ex']; oo_ = np.argsort(kk_); KTQ[nm_] = (kk_[oo_], q_['p'][oo_])
def att_at(nm_, rows, ex_):
    keys, p = KTQ[nm_]; kk = rows.astype(np.int64) * 4096 + ex_; j = np.searchsorted(keys, kk); assert np.all(keys[j] == kk); return p[j].astype(float)
KTF = {'DKT': dkt_at, 'SAKT': lambda r, x: att_at('sakt', r, x), 'AKT': lambda r, x: att_at('akt', r, x)}
t1 = time.time()
for P_, A_, C_ in [(PV, Av, Cv), (PT, At, Ct)]:
    P_['th_obs_s'] = np.where(P_['na'] > 0, P_['nc'] / np.maximum(P_['na'], 1), np.nan)
    P_['th_obs_q'] = np.where(P_['nq'] > 0, P_['cq'] / np.maximum(P_['nq'], 1), np.nan)
    P_['ZERO'] = np.zeros(len(P_['L'])); P_['POP'] = pop_acc[P_['I']]; P_['BKT'] = bL0[P_['I']]
    P_['LKT'] = lkt_theta(A_, C_, P_['L'], P_['I'])
    P_['DKT_s'] = dkt_at(P_['ws'][P_['rep']], P_['I'])            # state at window start
    P_['DKT_q'] = dkt_at(P_['fcase'][P_['rep']], P_['I'])         # state before the case
    for k_ in ['SAKT', 'AKT']:
        P_[k_ + '_s'] = KTF[k_](P_['ws'][P_['rep']], P_['I']); P_[k_ + '_q'] = KTF[k_](P_['fcase'][P_['rep']], P_['I'])
print('estimators', round(time.time() - t1), flush=True)
EST = ['ZERO', 'POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']
def est(P_, k, prof):
    if k in ('DKT', 'SAKT', 'AKT'): return P_[k + '_s'] if prof == 's' else P_[k + '_q']
    return P_[k]
# accuracy of each estimator as a predictor of the learner's eventual first attempt at the unobserved prerequisite
# (pairs whose prerequisite is attempted later in the held-out window; selected by future behaviour, descriptive only)
cs, ccum, uk, fi, _ = window_index(0.8, 1.01)
kp = PT['L'].astype(np.int64) * NC + PT['I']; j = np.searchsorted(uk, kp); j = np.minimum(j, len(uk) - 1)
later = (uk[j] == kp) & (PT['na'] == 0); lab = c[fi[j]].astype(int)
u_once = np.unique(kp[later], return_index=True)[1]; sel = np.flatnonzero(later)[u_once]
out['estimator_calibration_unobserved'] = {k: dict(n=int(len(sel)), auc=float(roc_auc_score(lab[sel], est(PT, k, 's')[sel])) if k != 'ZERO' else 0.5,
                                                   mean_pred=float(est(PT, k, 's')[sel].mean()), mean_obs=float(lab[sel].mean()),
                                                   brier=float(np.mean((est(PT, k, 's')[sel] - lab[sel]) ** 2))) for k in EST}
print('calib', out['estimator_calibration_unobserved'], flush=True)

# ------------------------------------------------------------------ features
def gfeat(P_, w, k, prof, kind):
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = w[eidx]; on = ww > 0
    th_o = P_['th_obs_s'] if prof == 's' else P_['th_obs_q']; obs = ~np.isnan(th_o)
    thh = est(P_, k, prof); th = np.where(obs, th_o, thh)
    if kind == 'coupled':   return np.bincount(rep, np.maximum(0, ww - th) * on, N)[:, None]
    if kind == 'decoupled': return np.bincount(rep, ww * (1 - th) * on, N)[:, None]
    if kind == 'esd':       return np.c_[np.bincount(rep, np.where(obs, ww * (1 - th), 0) * on, N),
                                         np.bincount(rep, np.where(~obs, ww * (1 - thh), 0) * on, N)]
def counts(P_, prof):
    nn_ = P_['na'] if prof == 's' else P_['nq']
    return np.c_[np.bincount(P_['rep'], (nn_ < 1).astype(float), P_['N']), np.bincount(P_['rep'], (nn_ >= 1).astype(float), P_['N'])]
def conf(df, P_, kind, kt='DKT'):
    """confounder sets. kind: B1 | B1R | B1K | B1RK, each at window start ('_s') or before the case ('_q')."""
    base, ref = kind.split('_'); X = base_X(df).copy()
    r = P_['ws'] if ref == 's' else P_['fcase']
    if ref == 'q':                                             # update history accuracy and length before the case
        wsr = P_['ws']; cum = np.r_[0, np.cumsum(c.astype(np.int64))]
        k_ = r - wsr; kc = cum[r] - cum[wsr]
        hn = df.h_n.values + k_; hc = df.h_acc.values * df.h_n.values + kc
        X[:, 0] = hc / hn; X[:, 1] = np.log1p(hn)
    if 'R' in base:
        cum = np.r_[0, np.cumsum(c.astype(np.int64))]; kk = np.minimum(10, r - LSTART[lid[r]])
        ra = (cum[r] - cum[r - kk]) / kk
        gap = np.log1p(np.maximum(tt[P_['fcase']] - tt[r - 1], 0) / 1e6)
        X = np.c_[X, ra, gap]
    if 'K' in base:
        pk = np.clip(KTF[kt](r, df['T'].values), 1e-6, 1 - 1e-6); X = np.c_[X, np.log(pk / (1 - pk))]
    return X
def fit(Xv, Xt): return fit_auc(Xv, yv, Xt, yt)

# ------------------------------------------------------------------ E1 + E3: estimator x usage x graph x protocol
PROT = {'snapshot': ('s', 'B1_s'), 'prereq_sequential': ('q', 'B1_s'), 'fully_sequential': ('q', 'B1R_q')}
E = {}; preds = {}
for pn, (prof, ck) in PROT.items():
    Xbv = conf(val, PV, ck); Xbt = conf(te, PT, ck)
    XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
    p0 = fit(Xbv, Xbt); pC = fit(XCv, XCt); preds[(pn, 'base')] = p0; preds[(pn, 'counts')] = pC
    R = dict(base_auc=float(roc_auc_score(yt, p0)), counts_auc=float(roc_auc_score(yt, pC)), counts_vs_base=contrast(pC, p0))
    for k in EST:
        for kind in ['coupled', 'decoupled', 'esd']:
            for g in ['B5', 'B6', 'B9']:
                fv = gfeat(PV, GR[g], k, prof, kind); ft = gfeat(PT, GR[g], k, prof, kind)
                if kind == 'coupled': p = fit(np.c_[Xbv, fv], np.c_[Xbt, ft]); ref = p0
                else: p = fit(np.c_[XCv, fv], np.c_[XCt, ft]); ref = pC
                preds[(pn, k, kind, g)] = p
                R[f'{k}|{kind}|{g}'] = contrast(p, ref)
            R[f'{k}|{kind}|B5-B9'] = contrast(preds[(pn, k, kind, 'B5')], preds[(pn, k, kind, 'B9')])
            R[f'{k}|{kind}|B6-B9'] = contrast(preds[(pn, k, kind, 'B6')], preds[(pn, k, kind, 'B9')])
        print(pn, k, {kk: np.round(v, 4).tolist() for kk, v in R.items() if kk.startswith(k + '|') and 'B5' in kk}, round(time.time() - T0), flush=True)
    # evidence-state composition under this protocol (admitted 713)
    P_ = PT; on = GR['B6'][P_['eidx']] > 0; nn_ = P_['na'] if prof == 's' else P_['nq']
    R['share_unobserved_713'] = float(((nn_ < 1) & on).sum() / on.sum())
    E[pn] = R
out['E1_E3_estimators_protocols'] = E

# ------------------------------------------------------------------ E2: stronger learner models (snapshot and fully sequential)
E2 = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    R = {}; P0 = {}
    for ck in ['B1', 'B1R', 'B1K', 'B1RK']:
        Xbv = conf(val, PV, f'{ck}_{ref}'); Xbt = conf(te, PT, f'{ck}_{ref}'); P0[ck] = fit(Xbv, Xbt)
        R[f'{ck}_auc'] = float(roc_auc_score(yt, P0[ck]))
    for ck in ['B1R', 'B1K', 'B1RK']: R[f'{ck}_vs_B1'] = contrast(P0[ck], P0['B1'])
    # graph evidence over the strongest learner model
    Xbv = conf(val, PV, f'B1RK_{ref}'); Xbt = conf(te, PT, f'B1RK_{ref}')
    XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]; pC = fit(XCv, XCt)
    R['B1RK+counts_vs_B1RK'] = contrast(pC, P0['B1RK'])
    pp = {}
    for k in ['ZERO', 'POP', 'DKT']:
        for g in ['B5', 'B6', 'B9']:
            pz = fit(np.c_[Xbv, gfeat(PV, GR[g], k, prof, 'coupled')], np.c_[Xbt, gfeat(PT, GR[g], k, prof, 'coupled')])
            pe = fit(np.c_[XCv, gfeat(PV, GR[g], k, prof, 'esd')], np.c_[XCt, gfeat(PT, GR[g], k, prof, 'esd')])
            pp[(k, 'c', g)] = pz; pp[(k, 'e', g)] = pe
            R[f'{k}|coupled|{g}'] = contrast(pz, P0['B1RK']); R[f'{k}|esd|{g}'] = contrast(pe, pC)
        R[f'{k}|esd|B5-B9'] = contrast(pp[(k, 'e', 'B5')], pp[(k, 'e', 'B9')])
        R[f'{k}|esd|B5|total_vs_B1RK'] = contrast(pp[(k, 'e', 'B5')], P0['B1RK'])
    E2[pn] = R; preds[(pn, 'E2', 'B1RK')] = P0['B1RK']; preds[(pn, 'E2', 'best')] = pp[('DKT', 'e', 'B5')]
    preds[(pn, 'E2', 'B1')] = P0['B1']
    print('E2', pn, {kk: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for kk, v in R.items()}, flush=True)
out['E2_stronger_learner_models'] = E2
# E2b: attention-based KT learner models (SAKT, AKT) and graph evidence over them
E2b = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    R = {}; pB1 = preds[(pn, 'E2', 'B1')]
    for kt in ['SAKT', 'AKT']:
        Xbv = conf(val, PV, f'B1RK_{ref}', kt); Xbt = conf(te, PT, f'B1RK_{ref}', kt); pK = fit(Xbv, Xbt)
        R[f'B1RK_{kt}_auc'] = float(roc_auc_score(yt, pK)); R[f'B1RK_{kt}_vs_B1'] = contrast(pK, pB1)
        R[f'B1RK_{kt}_vs_B1RK_DKT'] = contrast(pK, preds[(pn, 'E2', 'B1RK')]); R[f'B1RK_{kt}_vs_B1RK_DKT_delong'] = delong(pK, preds[(pn, 'E2', 'B1RK')])
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
        for k in ['ZERO', kt]:
            pg = {g: fit(np.c_[XCv, gfeat(PV, GR[g], k, prof, 'esd')], np.c_[XCt, gfeat(PT, GR[g], k, prof, 'esd')]) for g in ['B5', 'B9']}
            R[f'{kt}|ESD-{k}|B5_total_vs_B1RK'] = contrast(pg['B5'], pK); R[f'{kt}|ESD-{k}|B5-B9'] = contrast(pg['B5'], pg['B9'])
            R[f'{kt}|ESD-{k}|B5_delong'] = delong(pg['B5'], pK)
            if k == kt: preds[(pn, 'E2b', kt, 'B1RK')] = pK; preds[(pn, 'E2b', kt, 'best')] = pg['B5']
    E2b[pn] = R; print('E2b', pn, R, flush=True)
out['E2b_attention_kt'] = E2b

# ------------------------------------------------------------------ canonical intervals for the original tables (one bootstrap)
CAN = {}
Xb0v = conf(val, PV, 'B1_s'); Xb0t = conf(te, PT, 'B1_s'); p0 = fit(Xb0v, Xb0t)
for prof, pn in [('s', 'snapshot'), ('q', 'prereq_sequential')]:
    pz = {}
    for g in ['B3', 'B4', 'B5', 'B6', 'B9']:
        pz[g] = fit(np.c_[Xb0v, gfeat(PV, GR[g], 'ZERO', prof, 'coupled')], np.c_[Xb0t, gfeat(PT, GR[g], 'ZERO', prof, 'coupled')])
        CAN[f'{pn}|{g}'] = contrast(pz[g], p0) + [float(roc_auc_score(yt, pz[g]))]
    CAN[f'{pn}|B5-B9'] = contrast(pz['B5'], pz['B9']); CAN[f'{pn}|B6-B9'] = contrast(pz['B6'], pz['B9'])
    CAN[f'{pn}|DeLong B6 vs B1'] = delong(pz['B6'], p0); CAN[f'{pn}|DeLong B5 vs B9'] = delong(pz['B5'], pz['B9'])
CAN['B1_auc'] = float(roc_auc_score(yt, p0))
out['canonical_primary'] = CAN; print('CAN', CAN, flush=True)

# ladder on the canonical bootstrap + selection on validation only (E5)
def ladder_feats(P_, w, step, prof='s'):
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = w[eidx]; on = ww > 0
    th = P_['th_obs_s']; obs = ~np.isnan(th); th0 = np.where(obs, th, 0)
    if step == '1':  return np.bincount(rep, np.maximum(0, ww - th0) * on, N)[:, None]
    if step in ('2a', '2b'): return np.bincount(rep, np.where(obs, np.maximum(0, ww - th0), 0) * on, N)[:, None]
    if step == '2c': return np.bincount(rep, np.where(obs, ww * (1 - th0), 0) * on, N)[:, None]
    D = np.bincount(rep, np.where(obs & (th0 < ww), ww - th0, 0) * on, N); M = np.bincount(rep, np.where(obs & (th0 >= ww), ww, 0) * on, N)
    U = np.bincount(rep, np.where(~obs, ww, 0) * on, N)
    if step == '3': return np.c_[D, M, U]
    Dp = np.bincount(rep, np.where(obs, ww * (1 - th0), 0) * on, N); return np.c_[Dp, M, U]
XCv0 = np.c_[Xb0v, counts(PV, 's')]; XCt0 = np.c_[Xb0t, counts(PT, 's')]; pC0 = fit(XCv0, XCt0)
h = ((val.L.values.astype(np.int64) * 2654435761) % 7) < 4      # the validation split used for gate selection
LAD = {}; vsel = {}
for g in ['B5', 'B6', 'B9', 'B3']:
    for st in ['1', '2a', '2b', '2c', '3', '4']:
        bv, bt = (Xb0v, Xb0t) if st in ('1', '2a') else (XCv0, XCt0); ref = p0 if st in ('1', '2a') else pC0
        Fv = np.c_[bv, ladder_feats(PV, GR[g], st)]; Ft = np.c_[bt, ladder_feats(PT, GR[g], st)]
        p = fit(Fv, Ft); preds[('lad', g, st)] = p; LAD[f'{g}|{st}'] = contrast(p, ref)
        pv = fit_auc(Fv[h], yv[h], Fv[~h], yv[~h]); vsel[(g, st)] = float(roc_auc_score(yv[~h], pv))
for st in ['1', '2a', '2b', '2c', '3', '4']:
    LAD[f'B5-B9|{st}'] = contrast(preds[('lad', 'B5', st)], preds[('lad', 'B9', st)])
best = {g: max(['1', '2a', '2b', '2c', '3', '4'], key=lambda s: vsel[(g, s)]) for g in ['B5', 'B6', 'B9']}
E5 = dict(validation_auc={f'{g}|{s}': v for (g, s), v in vsel.items()}, selected=best,
          heldout_selected={g: LAD[f'{g}|{best[g]}'] for g in best},
          heldout_B5sel_minus_B9sel=contrast(preds[('lad', 'B5', best['B5'])], preds[('lad', 'B9', best['B9'])]))
out['ladder_canonical'] = LAD; out['E5_validation_selection'] = E5; print('E5', E5, flush=True)

# ------------------------------------------------------------------ E4: practical size at fixed alert budgets
def flags(p, q):
    k = int(round(q * len(p))); top = np.argsort(-p, kind='mergesort')[:k]; return yt[top].sum() / len(p) * 1000, yt[top].mean()
def flags_boot(p, q):
    k = int(round(q * len(p))); top = np.zeros(len(p), bool); top[np.argsort(-p, kind='mergesort')[:k]] = True
    return (BW * (top & pos)).sum(1) / BW.sum(1) * 1000
E4 = {}
cands = {'snapshot B1': p0, 'snapshot primary B6 (zero)': None, 'snapshot best ESD B5 (DKT)': preds[('snapshot', 'DKT', 'esd', 'B5')],
         'snapshot unweighted B9 coupled': preds[('snapshot', 'ZERO', 'coupled', 'B9')],
         'fully sequential B1R': preds[('fully_sequential', 'base')], 'fully sequential best ESD B5 (DKT)': preds[('fully_sequential', 'DKT', 'esd', 'B5')],
         'snapshot B1RK': preds[('snapshot', 'E2', 'B1RK')], 'snapshot B1RK + ESD B5 (DKT)': preds[('snapshot', 'E2', 'best')],
         'fully sequential B1RK': preds[('fully_sequential', 'E2', 'B1RK')], 'fully sequential B1RK + ESD B5 (DKT)': preds[('fully_sequential', 'E2', 'best')]}
cands['snapshot primary B6 (zero)'] = preds[('snapshot', 'ZERO', 'coupled', 'B6')]
for q in [0.10, 0.20, 0.30]:
    base_f = flags_boot(p0, q)
    for nm, p in cands.items():
        f, prec = flags(p, q); fb = flags_boot(p, q) - base_f
        E4[f'{nm}|top{int(q*100)}%'] = dict(failures_flagged_per_1000=float(f), precision=float(prec),
                                            diff_vs_snapshot_B1=[float(f - flags(p0, q)[0]), float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))])
E4['DeLong'] = {'snapshot ESD B5 (DKT) vs counts baseline': delong(preds[('snapshot', 'DKT', 'esd', 'B5')], preds[('snapshot', 'counts')]),
                'snapshot ESD B5 (DKT) vs ESD B9 (DKT)': delong(preds[('snapshot', 'DKT', 'esd', 'B5')], preds[('snapshot', 'DKT', 'esd', 'B9')]),
                'fully seq B1RK+ESD B5 (DKT) vs B1RK': delong(preds[('fully_sequential', 'E2', 'best')], preds[('fully_sequential', 'E2', 'B1RK')]),
                'snapshot B1RK vs B1': delong(preds[('snapshot', 'E2', 'B1RK')], preds[('snapshot', 'E2', 'B1')])}
out['E4_practical'] = E4; print('E4', {k: v for k, v in E4.items() if 'top20' in k}, flush=True)

# E4b: what graph evidence adds on top of the strongest learner model, at fixed alert budgets
E4b = {}
for pn in ['snapshot', 'fully_sequential']:
    pB = preds[(pn, 'E2', 'B1RK')]; pG = preds[(pn, 'E2', 'best')]
    for q in [0.10, 0.20, 0.30]:
        d0 = flags(pG, q)[0] - flags(pB, q)[0]; db = flags_boot(pG, q) - flags_boot(pB, q)
        E4b[f'{pn}|top{int(q*100)}%'] = [float(d0), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
for pn in ['snapshot', 'fully_sequential']:
    for kt in ['SAKT', 'AKT']:
        pB = preds[(pn, 'E2b', kt, 'B1RK')]; pG = preds[(pn, 'E2b', kt, 'best')]
        d0 = flags(pG, 0.2)[0] - flags(pB, 0.2)[0]; db = flags_boot(pG, 0.2) - flags_boot(pB, 0.2)
        E4b[f'{pn}|{kt}|top20%'] = [float(d0), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
out['E4b_graph_over_B1RK_flags_per_1000'] = E4b; print('E4b', E4b, flush=True)
# E6 dropped: in the release the 'suggested' flag is never true on a failed first attempt (fail rate 0.000 among
# 12.99% of cases), so it is outcome-linked and cannot serve as an exposure indicator.
sg = sug[PT['fcase']]
out['E6_suggested_flag_check'] = dict(share_cases_suggested=float(sg.mean()), fail_rate_suggested=float(yt[sg].mean()),
                                      fail_rate_other=float(yt[~sg].mean()))
np.savez_compressed('data/benchmark_preds.npz', **{'|'.join(map(str, k)): v for k, v in preds.items()}, yt=yt, L=te.L.values)
out['n_val'] = int(len(val)); out['n_test'] = int(len(te)); out['seconds'] = time.time() - T0
json.dump(out, open('out_benchmark.json', 'w'), indent=1, default=float)
print('done', round(time.time() - T0))
