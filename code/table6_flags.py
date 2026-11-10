"""Remaining Table 6 flag differences (20% budget): B1RK vs B1 and ESD-ZERO B5 - B9 over B1RK, both protocols."""
src = open('benchmark.py').read().split('# ------------------------------------------------------------------ E1 + E3')[0]
exec(src)
def flags20(p):
    k = int(round(0.2 * len(p))); top = np.zeros(len(p), bool); top[np.argsort(-p, kind='mergesort')[:k]] = True
    return float(yt[top].sum() / len(p) * 1000), (BW * (top & pos)).sum(1) / BW.sum(1) * 1000
def fdiff(p1, p0):
    a, ab = flags20(p1); b, bb = flags20(p0); d = ab - bb; return [a - b, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
R = {}
for pn, prof, ref in [('snapshot', 's', 's'), ('fully_sequential', 'q', 'q')]:
    pB1 = fit(conf(val, PV, f'B1_{ref}'), conf(te, PT, f'B1_{ref}'))
    Xbv = conf(val, PV, f'B1RK_{ref}'); Xbt = conf(te, PT, f'B1RK_{ref}'); pK = fit(Xbv, Xbt)
    XCv = np.c_[Xbv, counts(PV, prof)]; XCt = np.c_[Xbt, counts(PT, prof)]
    p5 = fit(np.c_[XCv, gfeat(PV, GR['B5'], 'ZERO', prof, 'esd')], np.c_[XCt, gfeat(PT, GR['B5'], 'ZERO', prof, 'esd')])
    p9 = fit(np.c_[XCv, gfeat(PV, GR['B9'], 'ZERO', prof, 'esd')], np.c_[XCt, gfeat(PT, GR['B9'], 'ZERO', prof, 'esd')])
    R[pn] = {'B1': flags20(pB1)[0], 'B1RK': flags20(pK)[0], 'B1RK_vs_B1': fdiff(pK, pB1), 'ESD-ZERO_B5-B9_over_B1RK': fdiff(p5, p9)}
    print(pn, R[pn], flush=True)
json.dump(R, open('out_table6_flags.json', 'w'), indent=1)
