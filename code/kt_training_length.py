"""Does the graph increment depend on how long the KT model is trained? (Junyi, post hoc)

For DKT models trained for 1, 2 (main analysis), 5 epochs and to early stopping (patience 2 on held-back training
pieces), recompute the B1RK learner model and the graph evidence over it, snapshot and fully sequential.
Usage: python kt_training_length.py DIR1 [DIR2 ...]   (folders written by kt_dkt.py; 'data' = main model)
Output: out_kt_training_length.json
"""
import sys, json
dirs = sys.argv[1:]
base_src = open('benchmark.py').read().split('# ------------------------------------------------------------------ E1 + E3')[0]
import os
RES = json.load(open('out_kt_training_length.json')) if os.path.exists('out_kt_training_length.json') else {}   # one folder per process keeps memory bounded
for d in dirs:
    g = {}
    exec(base_src.replace("np.load('data/dkt_states.npz')", f"np.load('{d}/dkt_states.npz')"), g)
    conf, fit, counts, gfeat, contrast, delong = g['conf'], g['fit'], g['counts'], g['gfeat'], g['contrast'], g['delong']
    val, te, PV, PT, GR, yt, est = g['val'], g['te'], g['PV'], g['PT'], g['GR'], g['yt'], g['est']
    rs = {}
    sel = g['sel']; lab = g['lab']; from sklearn.metrics import roc_auc_score
    import numpy as np
    rs['dkt_estimator_auc'] = float(roc_auc_score(lab[sel], est(PT, 'DKT', 's')[sel]))
    rs['dkt_estimator_brier'] = float(np.mean((est(PT, 'DKT', 's')[sel] - lab[sel]) ** 2))
    try: rs['heldback_piece_auc'] = json.load(open(f'{d}/out_dkt.json'))['heldout_piece_auc_by_epoch']
    except Exception: rs['heldback_piece_auc'] = json.load(open('out_dkt.json'))['heldout_piece_auc_by_epoch']
    for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
        pB1 = fit(conf(val, PV, f'B1_{ref}'), conf(te, PT, f'B1_{ref}'))
        Xbv = conf(val, PV, f'B1RK_{ref}'); Xbt = conf(te, PT, f'B1RK_{ref}'); pK = fit(Xbv, Xbt)
        XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
        r = {'B1RK_auc': float(roc_auc_score(yt, pK)), 'B1RK_vs_B1': contrast(pK, pB1)}
        for k in ['ZERO', 'DKT']:
            pg = fit(np.c_[XCv, gfeat(PV, GR['B5'], k, prof, 'esd')], np.c_[XCt, gfeat(PT, GR['B5'], k, prof, 'esd')])
            r[f'ESD-{k}_B5_vs_B1RK'] = contrast(pg, pK)
        # graph evidence without a KT learner model, for reference
        Xb1v = conf(val, PV, f'B1_{ref}'); Xb1t = conf(te, PT, f'B1_{ref}')
        pe = fit(np.c_[Xb1v, counts(PV, prof), gfeat(PV, GR['B5'], 'DKT', prof, 'esd')], np.c_[Xb1t, counts(PT, prof), gfeat(PT, GR['B5'], 'DKT', prof, 'esd')])
        r['ESD-DKT_B5_vs_B1'] = contrast(pe, pB1)
        rs[pn] = r
    RES[d] = rs; print(d, json.dumps(rs)[:900], flush=True)
    json.dump(RES, open('out_kt_training_length.json', 'w'), indent=1, default=float)
