"""Recalibration baselines for the graph-propagated estimator (Junyi, full size; post hoc).
Fits, on the same validation pairs as GPE: (a) Platt recalibration, logit pi = a + c * logit pi_base; (b) Platt plus
log(1 + attempts of the learner before the window); (c) GPE without the count term; (d) GPE with an intercept. Reports AUC
and Brier on the held-out later-attempted pairs, and the GPE coefficients for each base. Output: out_gpe_recal.json

Graph-propagated estimator (GPE) for unattempted prerequisites, and the graph in two roles (Junyi; post hoc).

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
MODE = 'full'
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

def fit_free(X, y, lam=1e-3):
    """logistic regression with free coefficients (no offset), small L2 except on the intercept column 0"""
    def f(b):
        z = X @ b; p = expit(z); reg = lam * (b[1:] @ b[1:])
        g = X.T @ (p - y) / len(y); g[1:] += 2 * lam * b[1:]
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + reg, g
    return minimize(f, np.r_[0, 1, np.zeros(X.shape[1] - 2)], jac=True, method='L-BFGS-B').x
csv_, ccv, ukv, fiv, _ = window_index(0.6, 0.8)
kpv = PV['L'].astype(np.int64) * NC + PV['I']; jv = np.minimum(np.searchsorted(ukv, kpv), len(ukv) - 1)
latv = (ukv[jv] == kpv) & (PV['na'] == 0); labv = c[fiv[jv]].astype(float); selv = np.flatnonzero(latv)[np.unique(kpv[latv], return_index=True)[1]]
FV = gpe_feats(PV, Av, Cv, 0.6, 0.8, 's', NB_REAL); FT = gpe_feats(PT, At, Ct, 0.8, 1.01, 's', NB_REAL)
hv = np.log1p(np.asarray(Av.sum(1)).ravel()[PV['L']]); ht = np.log1p(np.asarray(At.sum(1)).ravel()[PT['L']])
OUT = {'gpe_betas': betas, 'n_fit_pairs': int(len(selv)), 'n_heldout_pairs': int(len(sel)), 'heldout_mean_observed': float(lab[sel].mean()),
       'validation_mean_observed': float(labv[selv].mean())}
def ev(v): return dict(auc=float(roc_auc_score(lab[sel], v)), brier=float(np.mean((v - lab[sel]) ** 2)), mean_pred=float(v.mean()))
for b in BASES:
    bv = logit(np.clip(est(PV, b, 's'), 1e-4, 1 - 1e-4)); bt = logit(np.clip(est(PT, b, 's'), 1e-4, 1 - 1e-4)); r = {}
    r['base'] = ev(expit(bt[sel])); r['GPE'] = ev(est2(PT, 'G' + b, 's')[sel])
    one_v = np.ones(len(bv)); one_t = np.ones(len(bt))
    co = fit_free(np.c_[one_v, bv][selv], labv[selv]); r['platt'] = ev(expit(np.c_[one_t, bt][sel] @ co)); r['platt_coef'] = co.tolist()
    co = fit_free(np.c_[one_v, bv, hv][selv], labv[selv]); r['platt_plus_history'] = ev(expit(np.c_[one_t, bt, ht][sel] @ co)); r['platt_plus_history_coef'] = co.tolist()
    bb = fit_offset(FV[selv][:, :2], bv[selv], labv[selv]); r['GPE_no_count'] = ev(expit(bt[sel] + FT[sel][:, :2] @ bb)); r['GPE_no_count_beta'] = bb.tolist()
    co = fit_free(np.c_[one_v, bv, FV][selv], labv[selv]); r['GPE_with_intercept_free_slope'] = ev(expit(np.c_[one_t, bt, FT][sel] @ co)); r['GPE_with_intercept_coef'] = co.tolist()
    OUT[b] = r; print(b, {k: (round(v['auc'], 4), round(v['brier'], 4)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
json.dump(OUT, open('out_gpe_recal.json', 'w'), indent=1); print('done', round(time.time() - T0))
