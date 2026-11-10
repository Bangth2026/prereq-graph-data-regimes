"""Control analyses (Junyi; post hoc; run after benchmark.py).

  X1  Exposure versus estimation inside ESD: B1(+RK) + counts + D_obs, then + sum of w over unobserved
      prerequisites (= ESD with ZERO), then + the estimator term sum w(1 - pi) on top of that.
  X2  Learner-level random-graph control: each target keeps its number of candidates and its weights
      (permuted), candidates drawn at random from the other exercises; five draws.
  X3  Calibration of the final models: Brier score, expected calibration error (10 equal-frequency bins),
      calibration intercept and slope.
  X4  Failures flagged per 1,000 cases at a 20% budget for every row of Table 6.
  X5  Estimator values on unobserved prerequisites that were attempted later in the window versus never.
Uses the same split, cases, estimators, feature builders and the same 300 bootstrap weights as benchmark.py.
Output: out_controls.json
"""
src = open('benchmark.py').read().split('# ------------------------------------------------------------------ E1 + E3')[0]
exec(src)
from scipy.special import logit
R3 = {}
def gfeat_x(P_, w, k, prof, part):
    """parts of ESD: 'Dobs' observed shortfall, 'Wun' summed weight of unobserved prerequisites, 'Uest' sum w(1-pi)."""
    rep, eidx, N = P_['rep'], P_['eidx'], P_['N']; ww = w[eidx]; on = ww > 0
    th_o = P_['th_obs_s'] if prof == 's' else P_['th_obs_q']; obs = ~np.isnan(th_o)
    if part == 'Dobs': return np.bincount(rep, np.where(obs, ww * (1 - th_o), 0) * on, N)
    if part == 'Wun':  return np.bincount(rep, np.where(~obs, ww, 0) * on, N)
    return np.bincount(rep, np.where(~obs, ww * (1 - est(P_, k, prof)), 0) * on, N)
def flags20(p):
    k = int(round(0.2 * len(p))); top = np.zeros(len(p), bool); top[np.argsort(-p, kind='mergesort')[:k]] = True
    return float(yt[top].sum() / len(p) * 1000), (BW * (top & pos)).sum(1) / BW.sum(1) * 1000
def fdiff(p1, p0):
    a, ab = flags20(p1); b, bb = flags20(p0); d = ab - bb
    return [a - b, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
def calib(p):
    q = np.clip(p, 1e-6, 1 - 1e-6); bins = np.quantile(q, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(bins, q, 'right') - 1, 0, 9)
    ece = sum(abs(yt[b == i].mean() - q[b == i].mean()) * (b == i).mean() for i in range(10))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(q)[:, None], yt)
    return dict(brier=float(np.mean((q - yt) ** 2)), ece=float(ece), cal_intercept=float(lr.intercept_[0]), cal_slope=float(lr.coef_[0, 0]),
                mean_pred=float(q.mean()), mean_obs=float(yt.mean()))
w5 = GR['B5']
# ---------------- X1, X3, X4
X1 = {}; X3 = {}; X4 = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    out1 = {}; cal = {}; fl = {}
    for base in ['B1', 'B1RK']:
        bk = f'{base}_s' if (pn == 'snapshot' or base == 'B1RK') else 'B1R_q'
        if base == 'B1RK': bk = f'B1RK_{ref}'
        Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk); pB = fit(Xbv, Xbt)
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]; pC = fit(XCv, XCt)
        Dv = gfeat_x(PV, w5, None, prof, 'Dobs'); Dt = gfeat_x(PT, w5, None, prof, 'Dobs')
        Wv = gfeat_x(PV, w5, None, prof, 'Wun'); Wt = gfeat_x(PT, w5, None, prof, 'Wun')
        p1 = fit(np.c_[XCv, Dv], np.c_[XCt, Dt]); p2 = fit(np.c_[XCv, Dv, Wv], np.c_[XCt, Dt, Wt])
        r = {'counts_vs_base': contrast(pC, pB), 'Dobs_vs_counts': contrast(p1, pC), 'Wun_given_Dobs': contrast(p2, p1),
             'ESD-ZERO_vs_counts': contrast(p2, pC)}
        for k in ['POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']:
            Uv = gfeat_x(PV, w5, k, prof, 'Uest'); Ut = gfeat_x(PT, w5, k, prof, 'Uest')
            p3 = fit(np.c_[XCv, Dv, Wv, Uv], np.c_[XCt, Dt, Wt, Ut]); p4 = fit(np.c_[XCv, Dv, Uv], np.c_[XCt, Dt, Ut])
            r[f'{k}|estimate_given_exposure'] = contrast(p3, p2); r[f'{k}|ESD_vs_counts'] = contrast(p4, pC)
            r[f'{k}|ESD+Wun_vs_counts'] = contrast(p3, pC)
            if k == 'DKT': pbest = p4
        out1[base] = r
        cal[base] = calib(pB); cal[base + '+counts'] = calib(pC); cal[base + '+ESD-ZERO'] = calib(p2); cal[base + '+ESD-DKT'] = calib(pbest)
        fl[base] = flags20(pB)[0]; fl[base + '+counts'] = fdiff(pC, pB); fl[base + '+ESD-ZERO'] = fdiff(p2, pB); fl[base + '+ESD-DKT'] = fdiff(pbest, pB)
    # learner models of Table 6 (B1 here is the Table 6 baseline: B1 at the same reference point as the KT models)
    pB1 = fit(conf(val, PV, f'B1_{ref}'), conf(te, PT, f'B1_{ref}')); fl['B1_table6'] = flags20(pB1)[0]
    fl['B1RK_table6'] = flags20(fit(conf(val, PV, f'B1RK_{ref}'), conf(te, PT, f'B1RK_{ref}')))[0]
    for bk in ['B1R', 'B1K']:
        p = fit(conf(val, PV, f'{bk}_{ref}'), conf(te, PT, f'{bk}_{ref}')); fl[bk] = flags20(p)[0]; fl[bk + '_vs_B1'] = fdiff(p, pB1)
    for kt in ['SAKT', 'AKT']:
        Xbv = conf(val, PV, f'B1RK_{ref}', kt); Xbt = conf(te, PT, f'B1RK_{ref}', kt); pK = fit(Xbv, Xbt)
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
        pz = fit(np.c_[XCv, gfeat(PV, w5, 'ZERO', prof, 'esd')], np.c_[XCt, gfeat(PT, w5, 'ZERO', prof, 'esd')])
        pk = fit(np.c_[XCv, gfeat(PV, w5, kt, prof, 'esd')], np.c_[XCt, gfeat(PT, w5, kt, prof, 'esd')])
        fl[f'B1RK-{kt}'] = flags20(pK)[0]; fl[f'B1RK-{kt}_vs_B1'] = fdiff(pK, pB1)
        fl[f'B1RK-{kt}+ESD-ZERO'] = fdiff(pz, pK); fl[f'B1RK-{kt}+ESD-{kt}'] = fdiff(pk, pK)
        cal[f'B1RK-{kt}'] = calib(pK); cal[f'B1RK-{kt}+ESD-{kt}'] = calib(pk)
    X1[pn] = out1; X3[pn] = cal; X4[pn] = fl
    print('X1', pn, {b: {k: np.round(v, 4).tolist() for k, v in r.items()} for b, r in out1.items()}, flush=True)
    print('X3', pn, cal, flush=True); print('X4', pn, fl, flush=True)
R3['X1_exposure_vs_estimation'] = X1; R3['X3_calibration'] = X3; R3['X4_flags20_table6'] = X4
json.dump(R3, open('out_controls.json', 'w'), indent=1, default=float)

# ---------------- X2: learner-level random-graph control
def expand_r(df, A_, C_, lo, hi, SRCr):
    _, (rep, eidx, _) = load_scores(df, A_, C_, np.ones(NE))
    L = df.L.values[rep]; I = SRCr[eidx]
    na = np.asarray(A_[L, I]).ravel(); nc = np.asarray(C_[L, I]).ravel()
    cs, ccum, uk, fi, ws = window_index(lo, hi)
    kq = df.L.values.astype(np.int64) * NC + df['T'].values; p_ = np.searchsorted(uk, kq); fcase = fi[p_]
    wsL = ws[np.searchsorted(lid[ws], df.L.values)]
    kp = L.astype(np.int64) * NC + I
    lo_ = np.searchsorted(cs, kp * BIG); hi_ = np.searchsorted(cs, kp * BIG + fcase[rep])
    nq = na + (hi_ - lo_); cq = nc + (ccum[hi_] - ccum[lo_])
    P_ = dict(rep=rep, eidx=eidx, L=L, I=I, na=na, nc=nc, nq=nq, cq=cq, fcase=fcase, ws=wsL, N=len(df))
    P_['th_obs_s'] = np.where(na > 0, nc / np.maximum(na, 1), np.nan); P_['th_obs_q'] = np.where(nq > 0, cq / np.maximum(nq, 1), np.nan)
    P_['ZERO'] = np.zeros(len(L)); P_['DKT_s'] = dkt_at(wsL[rep], I); P_['DKT_q'] = dkt_at(fcase[rep], I)
    return P_
X2 = {}; rng = np.random.default_rng(31)
real = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    for base in ['B1', 'B1RK']:
        bk = f'{base}_{ref}' if base == 'B1RK' else ('B1_s' if pn == 'snapshot' else 'B1R_q')
        Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk); XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
        pC = fit(XCv, XCt); real[(pn, base, 'C')] = pC
        for k in ['ZERO', 'DKT']:
            real[(pn, base, k)] = fit(np.c_[XCv, gfeat(PV, w5, k, prof, 'esd')], np.c_[XCt, gfeat(PT, w5, k, prof, 'esd')])
            X2[f'{pn}|{base}|real ESD-{k} vs counts'] = contrast(real[(pn, base, k)], pC)
for d in range(5):
    SRCr = SRC.copy(); wr = w5.copy()
    for j in np.unique(DST):
        idx = np.flatnonzero(DST == j); pool = np.setdiff1d(np.arange(NC), [j])
        SRCr[idx] = rng.choice(pool, len(idx), replace=False); wr[idx] = rng.permutation(w5[idx])
    PVr = expand_r(val, Av, Cv, 0.6, 0.8, SRCr); PTr = expand_r(te, At, Ct, 0.8, 1.01, SRCr)
    for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
        for base in ['B1', 'B1RK']:
            bk = f'{base}_{ref}' if base == 'B1RK' else ('B1_s' if pn == 'snapshot' else 'B1R_q')
            Xbv = conf(val, PV, bk); Xbt = conf(te, PT, bk)
            XCv = np.c_[Xbv, counts(PVr, prof)]; XCt = np.c_[Xbt, counts(PTr, prof)]; pCr = fit(XCv, XCt)
            X2.setdefault(f'{pn}|{base}|random counts vs real counts', []).append(contrast(pCr, real[(pn, base, 'C')]))
            for k in ['ZERO', 'DKT']:
                pr = fit(np.c_[XCv, gfeat(PVr, wr, k, prof, 'esd')], np.c_[XCt, gfeat(PTr, wr, k, prof, 'esd')])
                X2.setdefault(f'{pn}|{base}|random ESD-{k} vs random counts', []).append(contrast(pr, pCr))
                X2.setdefault(f'{pn}|{base}|real minus random ESD-{k}', []).append(contrast(real[(pn, base, k)], pr))
    X2.setdefault('share_unobserved_random_snapshot', []).append(float((PTr['na'] < 1).mean()))
    print('X2 draw', d, {k: (np.round(v[-1], 4).tolist() if isinstance(v, list) and isinstance(v[-1], list) else v) for k, v in X2.items()}, round(time.time() - T0), flush=True)
    R3['X2_random_graph'] = X2; json.dump(R3, open('out_controls.json', 'w'), indent=1, default=float)
CACHE.clear()

# ---------------- X5: estimator values, later-attempted versus never-attempted unobserved prerequisites
unobs = PT['na'] == 0; never = unobs & ~later
X5 = {}
for k in ['POP', 'BKT', 'LKT', 'DKT', 'SAKT', 'AKT']:
    v = est(PT, k, 's'); a = v[later]; b = v[never]
    X5[k] = dict(later_mean=float(a.mean()), never_mean=float(b.mean()), later_q=np.percentile(a, [10, 50, 90]).tolist(),
                 never_q=np.percentile(b, [10, 50, 90]).tolist(), ks=float(stats.ks_2samp(a[::7], b[::7]).statistic))
X5['n_later_pairs'] = int(later.sum()); X5['n_never_pairs'] = int(never.sum())
R3['X5_later_vs_never'] = X5; print('X5', X5, flush=True)
R3['seconds'] = time.time() - T0
json.dump(R3, open('out_controls.json', 'w'), indent=1, default=float); print('done', round(time.time() - T0))
