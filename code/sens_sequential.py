"""Supplementary sensitivity analyses (exploratory; same held-out window). Run inside code/ after run_all.sh.
Writes sens_sequential.json. Nothing here was run before package v1.4.5 was assembled; see README.

A. Sequential profile: theta and n_obs recomputed from every attempt before the case (history + earlier
   attempts in the same window) instead of the window-start snapshot. Reports the share of (case, relation)
   pairs unobserved at window start but attempted before the target, evidence-state shares, score mass,
   primary dAUC of B3, B4, B5, B6, B9 and the B5 - B9, B6 - B9 contrasts under both profiles, and per-case
   distributions of the unobserved share (713 admitted and 1,131 candidate relations).
B. Count encoding under L2: counts baseline with (n_unobs, n_obs) versus n_obs alone; step-2b and step-3
   increments of B5, B6, B9 and B5 - B9 under both encodings.
C. Linked refitting bootstrap: learners resampled once from the union of validation and held-out learners,
   both windows of a resampled learner kept together; primary and three-state B5, B9, B5 - B9.
D. Direction trace for the relations of the teacher card (target division_0.5): learners attempting both,
   learners attempting B first, log share, expert flags, state under the Table 2 rule.
E. Signed Pearson r for the graph-level criterion (B5, B4, B3).
F. Ranking: NDCG@3 averaged over 200 random orderings, separately for cases with exactly two and with at
   least three labelled candidates; the seeded single ordering is reported alongside.
G. Direct comparison of the main contrasts with expected_outputs at absolute tolerance 1e-6.
"""
from evalcore import *
import json, os
from sklearn.linear_model import LogisticRegression
out = {}
Av, Cv, val = load('val'); At, Ct, te = load('test')
val = val[val.prior_att == 0].reset_index(drop=True); te = te[te.prior_att == 0].reset_index(drop=True)
yv = val.y.values; yt = te.y.values
W['FPKG'] = gated(0.80)
Lu, inv = np.unique(te.L.values, return_inverse=True)
Lvu, invv = np.unique(val.L.values, return_inverse=True)

def lboot(p1, p2, B=300, seed=808):
    r = np.random.default_rng(seed); o = []
    for _ in range(B):
        sw = np.bincount(r.integers(0, len(Lu), len(Lu)), minlength=len(Lu))[inv]; k = sw > 0
        o.append(roc_auc_score(yt[k], p1[k], sample_weight=sw[k]) - roc_auc_score(yt[k], p2[k], sample_weight=sw[k]))
    return [float(roc_auc_score(yt, p1) - roc_auc_score(yt, p2)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]

def expand(df, A_, C_):
    _, (rep, eidx, th) = load_scores(df, A_, C_, np.ones(NE))
    L = df.L.values[rep]; I = SRC[eidx]
    na = np.asarray(A_[L, I]).ravel(); nc = np.asarray(C_[L, I]).ravel()
    return rep, eidx, L, I, na, nc

# ---------------- A. sequential profile ----------------
BIG = np.int64(1) << 27          # > number of rows in the log (25.9 M)
def window_index(lo, hi):
    wm = (n >= 10) & (frac >= lo) & (frac < hi); wi = np.flatnonzero(wm)
    wkey = lid[wi].astype(np.int64) * NC + e[wi]
    comp = wkey * BIG + wi; order = np.argsort(comp, kind='stable'); cs = comp[order]
    ccum = np.r_[0, np.cumsum(c[wi][order].astype(np.int64))]
    uk, first = np.unique(wkey, return_index=True); fi = wi[first]      # first attempt of each (learner, exercise) in the window
    return cs, ccum, uk, fi

def seq_profile(df, A_, C_, lo, hi):
    """theta, n_obs for every (case, candidate) pair from all attempts before the case's first attempt at the target."""
    cs, ccum, uk, fi = window_index(lo, hi)
    rep, eidx, L, I, na, nc = expand(df, A_, C_)
    kq = df.L.values.astype(np.int64) * NC + df['T'].values
    pos = np.searchsorted(uk, kq); assert np.all(uk[pos] == kq), 'case not found in window'
    fcase = fi[pos][rep]                                                   # row index of the case's target attempt
    kp = L.astype(np.int64) * NC + I
    lo_ = np.searchsorted(cs, kp * BIG); hi_ = np.searchsorted(cs, kp * BIG + fcase)
    cnt = hi_ - lo_; cor = ccum[hi_] - ccum[lo_]
    n_seq = na + cnt; c_seq = nc + cor
    th_seq = np.where(n_seq > 0, c_seq / np.maximum(n_seq, 1), 0.0)
    th_snap = np.where(na > 0, nc / np.maximum(na, 1), 0.0)
    return rep, eidx, na, n_seq, th_snap, th_seq

def score_theta0(rep, eidx, th, w, ncase):
    ww = w[eidx]; return np.bincount(rep, np.maximum(0, ww - th) * (ww > 0), ncase)[:, None]

EV = seq_profile(val, Av, Cv, 0.6, 0.8); ET = seq_profile(te, At, Ct, 0.8, 1.01)
repT, eidxT, naT, nseqT, thS_T, thQ_T = ET
A = {}
for name, w in [('fpkg_713', W['FPKG']), ('all_1131', W['UNG'])]:
    on = w[eidxT] > 0
    snap_unobs = (naT < 1) & on; seq_unobs = (nseqT < 1) & on
    newly = snap_unobs & (nseqT >= 1)
    def shares(th, unobs):
        ww = w[eidxT]; obs = ~unobs & on
        gps = np.maximum(0, ww - th) * on
        return dict(share_unobserved=float(unobs.sum() / on.sum()),
                    share_observed_deficit=float((obs & (th < ww)).sum() / on.sum()),
                    share_observed_no_deficit=float((obs & (th >= ww)).sum() / on.sum()),
                    gps_mass_share_unobserved=float(gps[unobs].sum() / gps.sum()))
    def percase(unobs):
        n_on = np.bincount(repT, on.astype(float), len(te)); n_un = np.bincount(repT, unobs.astype(float), len(te))
        s = n_un[n_on > 0] / n_on[n_on > 0]
        return dict(n_cases=int((n_on > 0).sum()), median=float(np.median(s)), q25=float(np.percentile(s, 25)),
                    q75=float(np.percentile(s, 75)), mean=float(s.mean()), share_cases_all_unobserved=float((s == 1).mean()))
    A[name] = dict(pairs=int(on.sum()),
                   unobserved_at_window_start_but_attempted_before_target=int(newly.sum()),
                   share_of_pairs=float(newly.sum() / on.sum()),
                   share_of_window_start_unobserved=float(newly.sum() / max(snap_unobs.sum(), 1)),
                   snapshot=shares(thS_T, snap_unobs), sequential=shares(thQ_T, seq_unobs),
                   percase_snapshot=percase(snap_unobs), percase_sequential=percase(seq_unobs))
    print('A', name, json.dumps(A[name])[:400])
Xb0v = base_X(val); Xb0t = base_X(te)
p0 = fit_auc(Xb0v, yv, Xb0t, yt)
repV, eidxV, naV, nseqV, thS_V, thQ_V = EV
res = {}; preds = {}
for prof, thv, tht in [('snapshot', thS_V, thS_T), ('sequential', thQ_V, thQ_T)]:
    for m in ['BIN_EXP', 'BIN_MAJ', 'UNG', 'FPKG', 'ALL1']:
        p = fit_auc(np.c_[Xb0v, score_theta0(repV, eidxV, thv, W[m], len(val))], yv,
                    np.c_[Xb0t, score_theta0(repT, eidxT, tht, W[m], len(te))], yt)
        preds[(prof, m)] = p; res[f'{prof}_{m}'] = lboot(p, p0)
    res[f'{prof}_B5_minus_B9'] = lboot(preds[(prof, 'UNG')], preds[(prof, 'ALL1')])
    res[f'{prof}_B6_minus_B9'] = lboot(preds[(prof, 'FPKG')], preds[(prof, 'ALL1')])
A['primary_dAUC'] = res; A['confounder_auc'] = float(roc_auc_score(yt, p0))
out['A_sequential_profile'] = A; print('A dAUC', {k: np.round(v, 5).tolist() for k, v in res.items()})

# ---------------- B. count encoding under L2 ----------------
def counts_both(rep, na, ncase): return np.c_[np.bincount(rep, (na < 1).astype(float), ncase), np.bincount(rep, (na >= 1).astype(float), ncase)]
def counts_obs(rep, na, ncase): return np.bincount(rep, (na >= 1).astype(float), ncase)[:, None]
def sc_obs(rep, eidx, na, th, w, ncase):
    ww = w[eidx]; g = np.maximum(0, ww - th) * (ww > 0); return np.bincount(rep, np.where(na >= 1, g, 0), ncase)[:, None]
def sc_three(rep, eidx, na, th, w, ncase):
    ww = w[eidx]; on = ww > 0; obs = na > 0
    return np.c_[np.bincount(rep, np.where(obs & (th < ww), ww - th, 0) * on, ncase),
                 np.bincount(rep, np.where(obs & (th >= ww), ww, 0) * on, ncase),
                 np.bincount(rep, np.where(~obs, ww, 0) * on, ncase)]
Bres = {}
for enc, fv, ft in [('n_unobs_and_n_obs', counts_both(repV, naV, len(val)), counts_both(repT, naT, len(te))),
                    ('n_obs_only', counts_obs(repV, naV, len(val)), counts_obs(repT, naT, len(te)))]:
    XbCv = np.c_[Xb0v, fv]; XbCt = np.c_[Xb0t, ft]
    pC = fit_auc(XbCv, yv, XbCt, yt)
    r = {'counts_baseline_auc': float(roc_auc_score(yt, pC)), 'counts_vs_confounders': lboot(pC, p0)}
    pp = {}
    for m in ['UNG', 'FPKG', 'ALL1']:
        p2b = fit_auc(np.c_[XbCv, sc_obs(repV, eidxV, naV, thS_V, W[m], len(val))], yv, np.c_[XbCt, sc_obs(repT, eidxT, naT, thS_T, W[m], len(te))], yt)
        p3 = fit_auc(np.c_[XbCv, sc_three(repV, eidxV, naV, thS_V, W[m], len(val))], yv, np.c_[XbCt, sc_three(repT, eidxT, naT, thS_T, W[m], len(te))], yt)
        pp[(m, '2b')] = p2b; pp[(m, '3')] = p3
        r[f'{m}_step2b'] = lboot(p2b, pC); r[f'{m}_step3'] = lboot(p3, pC)
    r['B5_minus_B9_step2b'] = lboot(pp[('UNG', '2b')], pp[('ALL1', '2b')]); r['B5_minus_B9_step3'] = lboot(pp[('UNG', '3')], pp[('ALL1', '3')])
    Bres[enc] = r; print('B', enc, {k: (np.round(v, 5).tolist() if isinstance(v, list) else v) for k, v in r.items()})
# prediction-level comparison between the two encodings for the B5 step-3 model
pa = fit_auc(np.c_[Xb0v, counts_both(repV, naV, len(val)), sc_three(repV, eidxV, naV, thS_V, W['UNG'], len(val))], yv,
             np.c_[Xb0t, counts_both(repT, naT, len(te)), sc_three(repT, eidxT, naT, thS_T, W['UNG'], len(te))], yt)
pb = fit_auc(np.c_[Xb0v, counts_obs(repV, naV, len(val)), sc_three(repV, eidxV, naV, thS_V, W['UNG'], len(val))], yv,
             np.c_[Xb0t, counts_obs(repT, naT, len(te)), sc_three(repT, eidxT, naT, thS_T, W['UNG'], len(te))], yt)
Bres['B5_step3_pred_difference'] = dict(max_abs=float(np.abs(pa - pb).max()), mean_abs=float(np.abs(pa - pb).mean()),
                                        auc_both=float(roc_auc_score(yt, pa)), auc_obs_only=float(roc_auc_score(yt, pb)))
out['B_count_encoding'] = Bres

# ---------------- C. linked refitting bootstrap ----------------
XbCv = np.c_[Xb0v, counts_both(repV, naV, len(val))]; XbCt = np.c_[Xb0t, counts_both(repT, naT, len(te))]
Lall = np.unique(np.r_[Lvu, Lu]); posv = np.searchsorted(Lall, val.L.values); post = np.searchsorted(Lall, te.L.values)
def linked_refit(fv, ft, fv_base, ft_base, B=100, seed=909):
    r = np.random.default_rng(seed); o = []
    for _ in range(B):
        cnt = np.bincount(r.integers(0, len(Lall), len(Lall)), minlength=len(Lall))
        wv = cnt[posv]; wt = cnt[post]; kv = wv > 0; kt = wt > 0
        def fit(Xv, Xt):
            mu_ = Xv[kv].mean(0); sd = Xv[kv].std(0) + 1e-9
            m_ = LogisticRegression(max_iter=500).fit((Xv[kv] - mu_) / sd, yv[kv], sample_weight=wv[kv]); return m_.predict_proba((Xt - mu_) / sd)[:, 1]
        p1 = fit(fv, ft); p2 = fit(fv_base, ft_base)
        o.append(roc_auc_score(yt[kt], p1[kt], sample_weight=wt[kt]) - roc_auc_score(yt[kt], p2[kt], sample_weight=wt[kt]))
    o = np.array(o); return [float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)), float(o.std(ddof=1))]
Cres = {}
f5p = (np.c_[Xb0v, score_theta0(repV, eidxV, thS_V, W['UNG'], len(val))], np.c_[Xb0t, score_theta0(repT, eidxT, thS_T, W['UNG'], len(te))])
f9p = (np.c_[Xb0v, score_theta0(repV, eidxV, thS_V, W['ALL1'], len(val))], np.c_[Xb0t, score_theta0(repT, eidxT, thS_T, W['ALL1'], len(te))])
f53 = (np.c_[XbCv, sc_three(repV, eidxV, naV, thS_V, W['UNG'], len(val))], np.c_[XbCt, sc_three(repT, eidxT, naT, thS_T, W['UNG'], len(te))])
f93 = (np.c_[XbCv, sc_three(repV, eidxV, naV, thS_V, W['ALL1'], len(val))], np.c_[XbCt, sc_three(repT, eidxT, naT, thS_T, W['ALL1'], len(te))])
Cres['B5_primary'] = linked_refit(*f5p, Xb0v, Xb0t); Cres['B9_primary'] = linked_refit(*f9p, Xb0v, Xb0t)
Cres['B5_minus_B9_primary'] = linked_refit(*f5p, *f9p)
Cres['B5_three_state'] = linked_refit(*f53, XbCv, XbCt); Cres['B9_three_state'] = linked_refit(*f93, XbCv, XbCt)
Cres['B5_minus_B9_three_state'] = linked_refit(*f53, *f93)
Cres['learners_in_union'] = int(len(Lall)); Cres['learners_in_both_windows'] = int(len(np.intersect1d(Lvu, Lu)))
Cres['note'] = 'values: [mean, 2.5%, 97.5%, sd over 100 replicates]; learners resampled once from the union, both windows kept together'
out['C_linked_refit_bootstrap'] = Cres; print('C', Cres)

# ---------------- D. direction trace for the card relations ----------------
TRm = (frac < 0.6) & (n >= 10); lt = lid[TRm]; et = e[TRm]
key = lt.astype(np.int64) * NC + et; uk, fi = np.unique(key, return_index=True)
Lr = uk // NC; Cr = uk % NC
byc = {}
for cc in set(SRC) | set(DST):
    sel = Cr == cc; byc[cc] = pd.Series(fi[sel], index=Lr[sel])
expdir = W['BIN_EXP'] == 1; exprev = np.array([(d, s) in Ex for s, d in zip(SRC, DST)])
def state(share, same, rev):
    if np.isnan(share):
        return 'supported (expert only)' if same else ('disputed' if rev else 'unresolved')
    if 0.4 <= share <= 0.6: return 'ambiguous'
    if share > 0.6: return 'disputed' if rev else ('supported (both agree)' if same else 'supported (log only)')
    return 'disputed'
Dres = []
tgt = idx['division_0.5'] if 'division_0.5' in idx else int(DST[0])   # the card's target; fallback only for smoke tests
for k in np.flatnonzero(DST == tgt):
    a_ = byc.get(SRC[k]); b_ = byc.get(DST[k])
    j = a_.index.intersection(b_.index) if (a_ is not None and b_ is not None) else []
    nb = int(len(j)); nfirst = int((a_.loc[j].values < b_.loc[j].values).sum()) if nb else 0
    share = nfirst / nb if nb >= 30 else float('nan')
    Dres.append(dict(source=names[SRC[k]], target=names[DST[k]], learners_attempting_both=nb, learners_source_first=nfirst,
                     log_share=None if np.isnan(share) else float(share), expert_same=bool(expdir[k]), expert_reverse=bool(exprev[k]),
                     state=state(share, expdir[k], exprev[k]), rule='Table 2, package v1.4.5'))
out['D_direction_trace_division_0.5'] = Dres; print('D', Dres)

# ---------------- E. signed r ----------------
TP = (frac >= 0.8) & (n >= 10); ltp = lid[TP]; etp = e[TP]; same = ltp[1:] == ltp[:-1]
a = etp[:-1][same]; b = etp[1:][same]; mm = a != b
M = np.zeros((NC, NC)); np.add.at(M, (np.minimum(a[mm], b[mm]), np.maximum(a[mm], b[mm])), 1)
ycc = np.log1p((M + M.T)[SRC, DST])
bin_und = ((W['BIN_EXP'] + exprev) > 0).astype(float)
Eres = {}
rng = np.random.default_rng(5)
for nm, x in [('B5_graded', W['UNG']), ('B4_majority', W['BIN_MAJ']), ('B3_expert_either', bin_und)]:
    r_ = float(np.corrcoef(x, ycc)[0, 1])
    bs = []
    for _ in range(1000):   # two-way resampling of sources and targets, as in run_graph.py
        s_ = rng.choice(np.unique(SRC), len(np.unique(SRC)), replace=True); t_ = rng.choice(np.unique(DST), len(np.unique(DST)), replace=True)
        ms = np.isin(SRC, s_) & np.isin(DST, t_)
        bs.append(np.corrcoef(x[ms], ycc[ms])[0, 1] if x[ms].std() > 0 else 0.0)
    Eres[nm] = dict(r=r_, r2=r_ * r_, n_relations=int(NE), ci_two_way=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
out['E_signed_r'] = Eres; print('E', Eres)

# ---------------- F. ranking: permutation-averaged random control ----------------
wm = (n >= 10) & (frac >= 0.8); wi = np.flatnonzero(wm)
key = lid[wi].astype(np.int64) * NC + e[wi]; uk2, first = np.unique(key, return_index=True); fcor = c[wi[first]]
_, (rep, eidx, th) = load_scores(te, At, Ct, np.ones(NE))
L = te.L.values[rep]; I = SRC[eidx]; k2 = L.astype(np.int64) * NC + I; pos = np.minimum(np.searchsorted(uk2, k2), len(uk2) - 1); obs = uk2[pos] == k2
df = pd.DataFrame(dict(case=rep, L=L, I=I, e=eidx, th=th, rel=np.where(obs, (~fcor[pos]).astype(float), np.nan), obs=obs))
df = df[df.obs]; g = df.groupby('case'); ok = g.rel.transform('size') >= 2
df = df[ok & (g.rel.transform('max') > 0)]; df = df[te.y.values[df.case] == 1]
cases = df.case.values; relv = df.rel.values; ncand = df.groupby('case').size()
def ndcg(s, K=3):
    d = pd.DataFrame(dict(c=cases, s=s, r=relv)).sort_values(['c', 's'], ascending=[True, False]); d['rk'] = d.groupby('c').cumcount()
    dcg = (d.r / np.log2(d.rk + 2) * (d.rk < K)).groupby(d.c).sum()
    ideal = d.sort_values(['c', 'r'], ascending=[True, False]); ideal['rk'] = ideal.groupby('c').cumcount()
    idcg = (ideal.r / np.log2(ideal.rk + 2) * (ideal.rk < K)).groupby(ideal.c).sum()
    return dcg / idcg
rng = np.random.default_rng(11); jit = rng.random(len(df)) * 1e-6
sc = {'DEFICIT': 1 - df.th.values, 'BIN_EXP': np.maximum(0, W['BIN_EXP'][df.e.values] - df.th.values) * (W['BIN_EXP'][df.e.values] > 0),
      'UNG': np.maximum(0, W['UNG'][df.e.values] - df.th.values), 'FPKG': np.maximum(0, W['FPKG'][df.e.values] - df.th.values) * (W['FPKG'][df.e.values] > 0)}
grp = {'two_labelled': ncand[ncand == 2].index.values, 'three_or_more_labelled': ncand[ncand >= 3].index.values}
Fres = {'cases': int(len(ncand)), 'cases_two_labelled': int((ncand == 2).sum()), 'cases_three_or_more': int((ncand >= 3).sum())}
perm = np.stack([ndcg(rng.random(len(df))).values for _ in range(200)])   # rows: permutations; cols: cases in sorted order
cidx = ndcg(rng.random(len(df))).index.values
for gname, gc in grp.items():
    msk = np.isin(cidx, gc)
    Fres[gname] = {'random_mean_over_200_perms': float(perm[:, msk].mean()), 'random_sd_across_perms': float(perm[:, msk].mean(1).std(ddof=1))}
    for m, s in sc.items(): Fres[gname][m] = float(ndcg(s + jit)[msk].mean())
Fres['all'] = {'random_mean_over_200_perms': float(perm.mean()), 'random_single_seeded_ordering_run_rank': 'see out_rank.json RANDOM'}
for m, s in sc.items(): Fres['all'][m] = float(ndcg(s + jit).mean())
out['F_ranking_random_control'] = Fres; print('F', Fres)

# ---------------- G. direct comparison of main contrasts at 1e-6 ----------------
Gres = {}
try:
    exp_ = json.load(open(os.path.join('..', 'expected_outputs', 'out_main_new.json')))['contrasts']
    got = json.load(open('out_main_new.json'))['contrasts']
    diffs = {k: abs(exp_[k][0] - got[k][0]) for k in exp_ if k in got}
    Gres = dict(max_abs_difference_point_estimates=float(max(diffs.values())), n_contrasts=len(diffs),
                all_within_1e_6=bool(max(diffs.values()) <= 1e-6), per_contrast=diffs)
except Exception as ex:
    Gres = dict(error=str(ex))
out['G_contrast_comparison_1e-6'] = Gres; print('G', Gres)

json.dump(out, open('sens_sequential.json', 'w'), indent=1, default=float)
print('wrote sens_sequential.json')
