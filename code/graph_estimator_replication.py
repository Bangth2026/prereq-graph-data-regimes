"""Graph-propagated estimator (GPE) and the data-regime study on a replication dataset (XES3G5M; post hoc).

Same estimator as graph_estimator.py: one-hop UP/DOWN evidence from the relation graph added to a base estimator (POP, LKT, DKT)
with the base logit as offset, coefficients fitted on validation-window pairs. Graph = relations.csv with graded weights.

python graph_estimator_replication.py full DIR               all learners, DKT from DIR/dkt_states.npz        -> DIR/out_gpe.json
python graph_estimator_replication.py regime DIR F SEED      learners subsampled (DIR/regime/f{F}_s{SEED})    -> .../out.json
"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
MODE2, DIR = sys.argv[1], sys.argv[2]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xds_eval.py')).read()
src = src.split('# ---------------- composition and estimator x use x protocol')[0]
RD = DIR
if MODE2 == 'regime':
    F = float(sys.argv[3]); SEED = int(sys.argv[4]); RD = f'{DIR}/regime/f{F:g}_s{SEED}'
    inj = f"_sub = np.load('{RD}/learners.npz')['L']; INSUB = np.isin(lid, _sub)\n"
    for a, b in [("TR = (frac < 0.6) & (n >= 10)\n", inj + "TR = (frac < 0.6) & (n >= 10) & INSUB\n"),
                 ("m04 = (frac < 0.4) & (n >= 10);", "m04 = (frac < 0.4) & (n >= 10) & INSUB;"),
                 ("li = np.flatnonzero((frac >= 0.4) & (frac < 0.6) & (n >= 10));", "li = np.flatnonzero((frac >= 0.4) & (frac < 0.6) & (n >= 10) & INSUB);"),
                 ("bkt = fit_bkt();", "bkt = {};"),                 # BKT is not part of the data-regime study
                 ("np.load(f'{DIR}/dkt_states.npz')", f"np.load('{RD}/dkt_states.npz')")]:
        assert a in src, a; src = src.replace(a, b)
sys.argv = ['xds_eval.py', 'eval', DIR]
exec(src)
G = GR['G']; R0 = {}
def window(lo, hi):
    elig = n >= 10; wi = np.flatnonzero(elig & (frac >= lo) & (frac < hi)); key = lid[wi].astype(np.int64) * NC + e[wi]
    comp = key * BIG + wi; o = np.argsort(comp, kind='stable'); return comp[o], np.r_[0, np.cumsum(c[wi][o].astype(np.int64))]
BIG = np.int64(1) << 27; WIN = {'V': window(0.6, 0.8), 'T': window(0.8, 1.01)}
def neighbours(src_, dst_, w_):
    out = {}
    for name, a, b in [('succ', src_, dst_), ('pred', dst_, src_)]:
        o = np.argsort(a, kind='stable'); out[name] = (np.r_[0, np.cumsum(np.bincount(a, minlength=NC))], b[o], w_[o]); out[name + '_wsum'] = np.bincount(a, weights=w_, minlength=NC)
    return out
NB = neighbours(SRC, DST, G)
def gpe_feats(df, P_, tag, prof, CH=400_000):
    # chunked over pairs to bound memory (the curriculum tree gives about ten neighbours per unit)
    N = len(P_['I']); out = np.zeros((N, 3))
    for s0 in range(0, N, CH):
        sl = slice(s0, min(N, s0 + CH)); out[sl] = _gpe_feats(df, P_, tag, prof, sl)
    return out
def _gpe_feats(df, P_, tag, prof, sl):
    L = P_['L'][sl]; I = P_['I'][sl]; rep_ = P_['rep'][sl]; T = df['T'].values[rep_]; ref = (df.ws.values if prof == 's' else df.row.values)[rep_]
    res = {}
    for name in ['succ', 'pred']:
        ptr, nb, ww = NB[name]; deg = ptr[I + 1] - ptr[I]; rp = np.repeat(np.arange(len(I)), deg)
        pos_ = np.repeat(ptr[I], deg) + np.arange(deg.sum()) - np.repeat(np.cumsum(deg) - deg, deg); k = nb[pos_]; wk = ww[pos_]
        keep = k != T[rp]; rp, k, wk = rp[keep], k[keep], wk[keep]
        na = np.asarray(P_['A'][L[rp], k]).ravel(); nc_ = np.asarray(P_['C'][L[rp], k]).ravel()
        if prof == 'q':
            cs, ccum = WIN[tag]; kp = L[rp].astype(np.int64) * NC + k
            a_ = np.searchsorted(cs, kp * BIG); b_ = np.searchsorted(cs, kp * BIG + ref[rp]); na = na + (b_ - a_); nc_ = nc_ + (ccum[b_] - ccum[a_])
        ob = na > 0; dev = np.where(ob, nc_ / np.maximum(na, 1) - pop_acc[k], 0)
        res[name] = np.bincount(rp, wk * dev, len(I)) / (1 + NB[name + '_wsum'][I]); res[name + '_n'] = np.bincount(rp, ob.astype(float), len(I))
    return np.c_[res['succ'], res['pred'], np.log1p(res['succ_n'] + res['pred_n'])]
def fit_offset(X, off, y, lam=1e-3):
    def f(b):
        p = expit(off + X @ b)
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + lam * b @ b, X.T @ (p - y) / len(y) + 2 * lam * b
    return minimize(f, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B').x
BASES = ['POP', 'LKT', 'DKT']
FT = {(tg, pr): gpe_feats(df, P_, tg, pr) for tg, df, P_ in [('V', val, PV), ('T', te, PT)] for pr in ['s', 'q']}
# validation pairs: unobserved at validation window start, attempted later in the validation window
wi = np.flatnonzero((n >= 10) & (frac >= 0.6) & (frac < 0.8)); wk_ = lid[wi].astype(np.int64) * NC + e[wi]; ukv, fv = np.unique(wk_, return_index=True); fiv = wi[fv]
kpv = PV['L'].astype(np.int64) * NC + PV['I']; jv = np.minimum(np.searchsorted(ukv, kpv), len(ukv) - 1); latv = (ukv[jv] == kpv) & (PV['na'] == 0)
labv = c[fiv[jv]].astype(float); selv = np.flatnonzero(latv)[np.unique(kpv[latv], return_index=True)[1]]
betas = {}
for b in BASES:
    bt = fit_offset(FT[('V', 's')][selv], logit(np.clip(est(PV, b, 's')[selv], 1e-4, 1 - 1e-4)), labv[selv]); betas[b] = bt.tolist()
    for tg, P_ in [('V', PV), ('T', PT)]:
        for pr in ['s', 'q']:
            P_[f'G{b}_{pr}'] = expit(logit(np.clip(est(P_, b, pr), 1e-4, 1 - 1e-4)) + FT[(tg, pr)] @ bt)
R0['gpe_fit'] = dict(betas=betas, validation_pairs=int(len(selv)))
def est2(P_, k, prof): return P_[f'{k}_{prof}'] if k.startswith('G') else est(P_, k, prof)
acc = {}
for k in BASES + ['G' + b for b in BASES]:
    v = est2(PT, k, 's')[sel]; acc[k] = dict(auc=float(roc_auc_score(lab[sel], v)), brier=float(np.mean((v - lab[sel]) ** 2)))
R0['accuracy'] = acc; print('acc', {k: (round(v['auc'], 4), round(v['brier'], 4)) for k, v in acc.items()}, flush=True)
def gf(P_, k, prof):
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = G[eidx]; th_o = P_['th_s'] if prof == 's' else P_['th_q']; obs = ~np.isnan(th_o); thh = est2(P_, k, prof)
    return np.c_[np.bincount(rep, np.where(obs, ww * (1 - th_o), 0), N), np.bincount(rep, np.where(~obs, ww * (1 - thh), 0), N)]
ROLE = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    for base, bb, kt in [('B1', 'B1', None), ('B1RK', 'B1R', 'DKT')]:
        Xbv = conf(val, bb, ref, kt); Xbt = conf(te, bb, ref, kt); pB = fit(Xbv, Xbt)
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]; pC = fit(XCv, XCt)
        r = {'auc': float(roc_auc_score(yt, pB)), 'counts_vs_base': contrast(pC, pB)}; pe = {}
        for k in BASES:
            for kk in [k, 'G' + k]:
                pe[kk] = fit(np.c_[XCv, gf(PV, kk, prof)], np.c_[XCt, gf(PT, kk, prof)]); r[f'ESD-{kk}_vs_base'] = contrast(pe[kk], pB)
            r[f'estimation_role_{k}'] = contrast(pe['G' + k], pe[k])
        ROLE[f'{pn}|{base}'] = r
        print(pn, base, {kk: np.round(v, 4).tolist() for kk, v in r.items() if 'role' in kk or 'DKT' in kk}, round(time.time() - T0), flush=True)
R0['roles'] = ROLE
if MODE2 == 'regime': R0['n_training_learners'] = int(len(np.load(f'{RD}/learners.npz')['L']))
json.dump(R0, open(f'{RD}/out.json' if MODE2 == 'regime' else f'{DIR}/out_gpe.json', 'w'), indent=1, default=float); print('done', round(time.time() - T0))
