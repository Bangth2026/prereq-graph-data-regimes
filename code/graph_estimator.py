"""Graph-propagated estimator (GPE) for unattempted prerequisites, and the graph in two roles (Junyi; post hoc).

GPE adds one-hop evidence from the prerequisite graph to a base estimator (POP, LKT or DKT):
  logit pi_G(s,i) = logit pi_base(s,i) + b1 * UP + b2 * DOWN + b3 * log(1 + n_up + n_down)
  UP   = sum_{k: i->k, k != target, observed} w_ik (theta_sk - pbar_k) / (1 + sum_{k: i->k} w_ik)   (successes on items that need i)
  DOWN = sum_{h: h->i, observed}           w_hi (theta_sh - pbar_h) / (1 + sum_{h: h->i} w_hi)   (failures on items i needs)
theta = proportion correct on the learner's attempts before the scoring point, pbar = population first-attempt accuracy.
b1..b3 are fitted by logistic regression with the base logit as offset (no intercept, so b = 0 gives the base back) on
validation-window pairs: prerequisites unattempted at the validation window start and attempted later in that window.
Graph = the 1,131 crowd relations read B -> A with graded weights mu* (B5).

Modes
  python graph_estimator.py full            estimator accuracy, ESD with GPE (estimation role), weighting role, same-topic controls
  python graph_estimator.py regime F SEED   data-regime point: POP, LKT, difficulty and frequency refitted on the training segments of a
                                fraction F of learners (sample seed SEED), DKT read from data/regime/f{F}_s{SEED}
Output: out_gpe.json (full) or data/regime/out_f{F}_s{SEED}.json
"""
import sys, json, time, numpy as np
from scipy.optimize import minimize
MODE = sys.argv[1]
src = open('benchmark.py').read().split('# ------------------------------------------------------------------ E1 + E3')[0]
if MODE == 'regime':
    F = float(sys.argv[2]); SEED = int(sys.argv[3]); RD = f'data/regime/f{F:g}_s{SEED}'
    inj = f"""
_sub = np.load('{RD}/learners.npz')['L']; INSUB = np.isin(lid, _sub)
_TRs = (frac < 0.6) & (n >= 10) & INSUB
_fq = np.bincount(e[_TRs], minlength=NC).astype(float); _er = np.bincount(e[_TRs], weights=(~c[_TRs]).astype(float), minlength=NC)
_df = np.where(_fq > 0, _er / np.maximum(_fq, 1), np.nanmean(np.where(_fq > 0, _er / np.maximum(_fq, 1), np.nan)))
diff[:] = _df; freq[:] = _fq                                      # shared with evalcore: B1 features use the subsample too
"""
    src = src.replace("T0 = time.time()\n", "T0 = time.time()\n" + inj, 1)
    for a, b in [("m04 = (frac < 0.4) & (n >= 10);", "m04 = (frac < 0.4) & (n >= 10) & INSUB;"),
                 ("lm = (frac >= 0.4) & (frac < 0.6) & (n >= 10);", "lm = (frac >= 0.4) & (frac < 0.6) & (n >= 10) & INSUB;"),
                 ("np.load('data/dkt_states.npz')", f"np.load('{RD}/dkt_states.npz')")]:
        assert a in src, a; src = src.replace(a, b)
exec(src)
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
import pandas as pd
w5 = GR['B5']; R0 = {}

# ---------------- counts of attempts and correct answers of learner L on unit K before a reference row
def counts_at(L, K, A_, C_, ref_rows, lo, hi, prof):
    na = np.asarray(A_[L, K]).ravel(); nc = np.asarray(C_[L, K]).ravel()
    if prof == 's': return na, nc
    cs, ccum, _, _, _ = window_index(lo, hi); kp = L.astype(np.int64) * NC + K
    a = np.searchsorted(cs, kp * BIG); b = np.searchsorted(cs, kp * BIG + ref_rows)
    return na + (b - a), nc + (ccum[b] - ccum[a])
# ---------------- one-hop features
def neighbours(src_, dst_, w_):
    """for each unit: successors (i -> k) and predecessors (h -> i) with weights, as CSR arrays"""
    out = {}
    for name, a, b in [('succ', src_, dst_), ('pred', dst_, src_)]:
        o = np.argsort(a, kind='stable'); ptr = np.r_[0, np.cumsum(np.bincount(a, minlength=NC))]
        out[name] = (ptr, b[o], w_[o]); out[name + '_wsum'] = np.bincount(a, weights=w_, minlength=NC)
    return out
NB_REAL = neighbours(SRC, DST, w5)
def gpe_feats(P_, A_, C_, lo, hi, prof, NB):
    L = P_['L']; I = P_['I']; T = (val if P_ is PV else te)['T'].values[P_['rep']]
    ref = P_['ws'][P_['rep']] if prof == 's' else P_['fcase'][P_['rep']]
    res = {}
    for name in ['succ', 'pred']:
        ptr, nb, ww = NB[name]; deg = ptr[I + 1] - ptr[I]; rp = np.repeat(np.arange(len(I)), deg)
        off = np.arange(deg.sum()) - np.repeat(np.cumsum(deg) - deg, deg); k = nb[np.repeat(ptr[I], deg) + off]; wk = ww[np.repeat(ptr[I], deg) + off]
        keep = k != T[rp]                                            # never use the target itself
        rp, k, wk = rp[keep], k[keep], wk[keep]
        na, nc = counts_at(L[rp], k, A_, C_, ref[rp], lo, hi, prof); ob = na > 0
        dev = np.where(ob, nc / np.maximum(na, 1) - pop_acc[k], 0)
        res[name] = np.bincount(rp, wk * dev, len(I)) / (1 + NB[name + '_wsum'][I]); res[name + '_n'] = np.bincount(rp, ob.astype(float), len(I))
    return np.c_[res['succ'], res['pred'], np.log1p(res['succ_n'] + res['pred_n'])]
def fit_offset(X, off, y, lam=1e-3):
    def f(b):
        z = off + X @ b; p = expit(z)
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + lam * b @ b, X.T @ (p - y) / len(y) + 2 * lam * b
    return minimize(f, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B').x
BASES = ['POP', 'LKT', 'DKT'] if MODE == 'regime' else ['POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']
def build_gpe(NB, tag):
    """fit b on validation pairs and add GPE-<base> estimates to PV and PT, for both profiles"""
    Fs = {(nm, pr): gpe_feats(P_, A_, C_, lo, hi, pr, NB) for nm, P_, A_, C_, lo, hi in [('V', PV, Av, Cv, 0.6, 0.8), ('T', PT, At, Ct, 0.8, 1.01)] for pr in ['s', 'q']}
    csv_, ccv, ukv, fiv, _ = window_index(0.6, 0.8)
    kpv = PV['L'].astype(np.int64) * NC + PV['I']; jv = np.minimum(np.searchsorted(ukv, kpv), len(ukv) - 1)
    latv = (ukv[jv] == kpv) & (PV['na'] == 0); labv = c[fiv[jv]].astype(float); selv = np.flatnonzero(latv)[np.unique(kpv[latv], return_index=True)[1]]
    betas = {}
    for b in BASES:
        base_s = np.clip(est(PV, b, 's'), 1e-4, 1 - 1e-4); bt = fit_offset(Fs[('V', 's')][selv], logit(base_s[selv]), labv[selv]); betas[b] = bt.tolist()
        for nm, P_ in [('V', PV), ('T', PT)]:
            for pr in ['s', 'q']:
                bb = np.clip(est(P_, b, pr), 1e-4, 1 - 1e-4); g = expit(logit(bb) + Fs[(nm, pr)] @ bt)
                key = f'G{tag}{b}'
                if b in ('DKT', 'SAKT', 'AKT'): P_[f'{key}_{pr}'] = g
                elif pr == 's': P_[key + '_s'] = g
                else: P_[key + '_q'] = g
    return betas, int(len(selv))
def est2(P_, k, prof):
    if k.startswith('G'): return P_[f'{k}_{prof}']
    return est(P_, k, prof)
def gfeat2(P_, w, k, prof):
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = w[eidx]; on = ww > 0
    th_o = P_['th_obs_s'] if prof == 's' else P_['th_obs_q']; obs = ~np.isnan(th_o); thh = est2(P_, k, prof)
    return np.c_[np.bincount(rep, np.where(obs, ww * (1 - th_o), 0) * on, N), np.bincount(rep, np.where(~obs, ww * (1 - thh), 0) * on, N)]
betas, nfit = build_gpe(NB_REAL, '')
R0['gpe_fit'] = dict(betas=betas, validation_pairs=nfit)
# ---------------- (i) estimator accuracy on held-out pairs unattempted at window start and attempted later
acc = {}
for k in BASES + ['G' + b for b in BASES]:
    v = est2(PT, k, 's')[sel]; acc[k] = dict(auc=float(roc_auc_score(lab[sel], v)), brier=float(np.mean((v - lab[sel]) ** 2)))
R0['accuracy'] = acc; print('acc', {k: (round(v['auc'], 4), round(v['brier'], 4)) for k, v in acc.items()}, flush=True)
# ---------------- (ii) roles over B1 and B1RK
ROLE = {}; PR = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    for base in ['B1', 'B1RK']:
        bk = f'{base}_{ref}'
        Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk); pB = fit(Xbv, Xbt)
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]; pC = fit(XCv, XCt)
        r = {'auc': float(roc_auc_score(yt, pB)), 'counts_vs_base': contrast(pC, pB)}; pe = {}
        for k in (['POP', 'LKT', 'DKT'] if MODE == 'regime' else BASES):
            for kk in [k, 'G' + k]:
                pe[kk] = fit(np.c_[XCv, gfeat2(PV, w5, kk, prof)], np.c_[XCt, gfeat2(PT, w5, kk, prof)])
                r[f'ESD-{kk}_vs_base'] = contrast(pe[kk], pB)
            r[f'estimation_role_{k}'] = contrast(pe['G' + k], pe[k])        # GPE-x minus x, both inside ESD
        ROLE[f'{pn}|{base}'] = r; PR[(pn, base)] = (pB, pC, pe)
        print(pn, base, {kk: np.round(v, 4).tolist() for kk, v in r.items() if 'role' in kk or 'DKT' in kk}, round(time.time() - T0), flush=True)
R0['roles'] = ROLE
if MODE == 'regime':
    R0['n_training_learners'] = int(len(np.load(f'{RD}/learners.npz')['L']))
    json.dump(R0, open(f'{RD}/out.json', 'w'), indent=1, default=float); print('done'); sys.exit()
json.dump(R0, open('out_gpe.json', 'w'), indent=1, default=float)

# ---------------- (iii) same-topic controls: candidates drawn from the target's topic, as many as the real ones, weights permuted
def expand_r(df, A_, C_, lo, hi, SRCr):
    _, (rep, eidx, _) = load_scores(df, A_, C_, np.ones(NE))
    L = df.L.values[rep]; I = SRCr[eidx]
    na = np.asarray(A_[L, I]).ravel(); nc = np.asarray(C_[L, I]).ravel()
    cs, ccum, uk, fi, ws = window_index(lo, hi)
    kq = df.L.values.astype(np.int64) * NC + df['T'].values; fcase = fi[np.searchsorted(uk, kq)]
    wsL = ws[np.searchsorted(lid[ws], df.L.values)]; kp = L.astype(np.int64) * NC + I
    a = np.searchsorted(cs, kp * BIG); b = np.searchsorted(cs, kp * BIG + fcase[rep])
    P_ = dict(rep=rep, eidx=eidx, L=L, I=I, na=na, nc=nc, nq=na + (b - a), cq=nc + (ccum[b] - ccum[a]), fcase=fcase, ws=wsL, N=len(df))
    P_['th_obs_s'] = np.where(na > 0, nc / np.maximum(na, 1), np.nan); P_['th_obs_q'] = np.where(P_['nq'] > 0, P_['cq'] / np.maximum(P_['nq'], 1), np.nan)
    P_['DKT_s'] = dkt_at(wsL[rep], I); P_['DKT_q'] = dkt_at(fcase[rep], I)
    return P_
rng = np.random.default_rng(41); CTRL = {}
bytopic = {t: np.flatnonzero(topic == t) for t in np.unique(topic)}
obs_rate = np.asarray((At > 0).sum(0)).ravel() / At.shape[0]
for d in range(3):
    SRCr = SRC.copy(); wr = w5.copy()
    for j in np.unique(DST):
        idx = np.flatnonzero(DST == j); pool = np.setdiff1d(bytopic[topic[j]], np.r_[j, SRC[idx]])
        if len(pool) < len(idx): pool = np.setdiff1d(np.arange(NC), np.r_[j, SRC[idx]])
        # proximity-matched: within the topic, prefer units whose observation rate is closest to the real candidates'
        tgt_rate = obs_rate[SRC[idx]]; cand = []
        for r_ in tgt_rate:
            pl = np.setdiff1d(pool, cand); dd = np.abs(obs_rate[pl] - r_); near = pl[np.argsort(dd, kind='stable')[:3]]; cand.append(rng.choice(near))
        SRCr[idx] = cand; wr[idx] = rng.permutation(w5[idx])
    PVr = expand_r(val, Av, Cv, 0.6, 0.8, SRCr); PTr = expand_r(te, At, Ct, 0.8, 1.01, SRCr)
    for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
        for base in ['B1', 'B1RK']:
            pB, pC, pe = PR[(pn, base)]; bk = f'{base}_{ref}'
            Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk); XCv = np.c_[Xbv, counts(PVr, prof)]; XCt = np.c_[Xbt, counts(PTr, prof)]
            pr_ = fit(np.c_[XCv, gfeat2(PVr, wr, 'DKT', prof)], np.c_[XCt, gfeat2(PTr, wr, 'DKT', prof)])
            CTRL.setdefault(f'{pn}|{base}|weighting: real minus same-topic ESD-DKT', []).append(contrast(pe['DKT'], pr_))
            CTRL.setdefault(f'{pn}|{base}|same-topic ESD-DKT vs base', []).append(contrast(pr_, pB))
    # estimation role with same-topic neighbours (real candidate sets, random same-topic neighbour graph)
    NB_R = neighbours(SRCr, DST, wr); build_gpe(NB_R, 'R')
    for k in ['POP', 'DKT']:
        v = est2(PT, 'GR' + k, 's')[sel]; CTRL.setdefault(f'accuracy GPE-{k} same-topic graph', []).append(dict(auc=float(roc_auc_score(lab[sel], v)), brier=float(np.mean((v - lab[sel]) ** 2))))
    for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
        for base in ['B1', 'B1RK']:
            pB, pC, pe = PR[(pn, base)]; bk = f'{base}_{ref}'
            Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk); XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
            pg = fit(np.c_[XCv, gfeat2(PV, w5, 'GRDKT', prof)], np.c_[XCt, gfeat2(PT, w5, 'GRDKT', prof)])
            CTRL.setdefault(f'{pn}|{base}|estimation: real minus same-topic GPE-DKT', []).append(contrast(pe['GDKT'], pg))
    print('ctrl', d, {k: np.round(v[-1], 4).tolist() if isinstance(v[-1], list) else v[-1] for k, v in CTRL.items()}, round(time.time() - T0), flush=True)
    R0['same_topic_controls'] = CTRL; json.dump(R0, open('out_gpe.json', 'w'), indent=1, default=float)
print('done', round(time.time() - T0))
