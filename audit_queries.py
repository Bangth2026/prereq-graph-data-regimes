"""Audit queries over the evidence representation (ESM 1, v1.4.2 proposal).

Q1  Score mass by learner evidence state  -> reads expected_outputs/analysis_f.json (produced by analysis_f.py; needs the log).
Q2  Relations whose admission status changes when Q_min changes -> reads edge_list/fpkg_edges_crowd.csv,
    which v1.4.4 no longer distributes; rebuild it with code/make_edge_list.py after run_all.sh.
Q3  (requires the log; run inside code/ after run_all.sh)  per-case distribution of the unobserved share and
    signed Pearson r for the graph-level criterion.

Usage:  python audit_queries.py q1 | q2 <Q_min_old> <Q_min_new> | q3
"""
import sys, os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

def q1():
    d = json.load(open(os.path.join(ROOT, 'expected_outputs', 'analysis_f.json')))
    print('graph, (case,relation) pairs, unobserved, obs-deficit, obs-no-deficit, score mass from unobserved (theta=0)')
    for g, v in d.items():
        print(f"{g:15s} {v['n_relations_in_cases']:>8d}  {v['share_unobserved']:.3f}  {v['share_observed_deficit']:.3f}  "
              f"{v['share_observed_mastered']:.3f}  {v['gps_mass_share_unobserved']:.3f}")

def q2(q_old, q_new):
    f = os.path.join(ROOT, 'edge_list', 'fpkg_edges_crowd.csv')
    if not os.path.exists(f):
        raise SystemExit('edge_list/fpkg_edges_crowd.csv is not distributed in this record; rebuild it with: cd code && python make_edge_list.py (after run_all.sh)')
    e = pd.read_csv(f)
    a_old = e.Q >= q_old - 1e-9; a_new = e.Q >= q_new - 1e-9
    gained = e[a_new & ~a_old]; lost = e[a_old & ~a_new]
    print(f'Q_min {q_old} -> {q_new}: admitted {int(a_old.sum())} -> {int(a_new.sum())}; '
          f'newly admitted {len(gained)} (feeding {gained.target.nunique()} targets); '
          f'newly rejected {len(lost)} (affecting {lost.target.nunique()} targets)')
    cols = ['source', 'target', 'weight', 'Q', 'n_raters', 'ratings_0to9']
    out = pd.concat([gained[cols].assign(change='admitted'), lost[cols].assign(change='rejected')])
    out.to_csv(os.path.join(ROOT, f'audit_q2_{q_old}_to_{q_new}.csv'), index=False)
    print(out.head(10).to_string(index=False)); print('... full list written to', f'audit_q2_{q_old}_to_{q_new}.csv')

def q3():
    # run from inside code/ after run_all.sh (needs data/ built from the log)
    sys.path.insert(0, os.getcwd())
    import evalcore as ev
    NE, SRC, DST, W, Ex, frac, n, lid, e, NC, load_scores = (ev.NE, ev.SRC, ev.DST, ev.W, ev.Ex, ev.frac, ev.n,
                                                            ev.lid, ev.e, ev.NC, ev.load_scores)
    At, Ct, te = ev.load('test'); te = te[te.prior_att == 0].reset_index(drop=True)   # new-target cases, as in run_main.py
    _, (rep, eidx, th) = load_scores(te, At, Ct, np.ones(NE))
    L = te.L.values[rep]; I = SRC[eidx]; na = np.asarray(At[L, I]).ravel()
    for label, wv in [('713 admitted (FPKG)', ev.gated(0.80)), ('1,131 candidates (B5)', W['UNG'])]:
        w = wv[eidx]; on = w > 0
        unobs = (na < 1) & on
        n_on = np.bincount(rep, on.astype(float), len(te)); n_un = np.bincount(rep, unobs.astype(float), len(te))
        share = n_un[n_on > 0] / n_on[n_on > 0]
        print('per-case unobserved share, %s: n = %d  median %.3f  IQR [%.3f, %.3f]  mean %.3f  share of cases with all candidates unobserved %.3f'
              % (label, len(share), np.median(share), *np.percentile(share, [25, 75]), share.mean(), (share == 1).mean()))
    # signed Pearson r with held-out adjacent-attempt co-occurrence (same construction as run_graph.py)
    TP = (frac >= 0.8) & (n >= 10); lt = lid[TP]; et = e[TP]; same = lt[1:] == lt[:-1]
    a = et[:-1][same]; b = et[1:][same]; mm = a != b
    M = np.zeros((NC, NC)); np.add.at(M, (np.minimum(a[mm], b[mm]), np.maximum(a[mm], b[mm])), 1)
    y = np.log1p((M + M.T)[SRC, DST])
    for k in ['UNG', 'BIN_MAJ', 'BIN_EXP']:
        x = W[k] if k != 'BIN_EXP' else (W['BIN_EXP'] + np.array([(d, s) in Ex for s, d in zip(SRC, DST)]) > 0).astype(float)
        r = np.corrcoef(x, y)[0, 1]; print(f'signed r {k}: {r:+.3f}  (r^2 {r*r:.3f})')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'q1'
    if cmd == 'q1': q1()
    elif cmd == 'q2': q2(float(sys.argv[2]), float(sys.argv[3]))
    elif cmd == 'q3': q3()
