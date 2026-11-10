"""Engineering cost of evidence-state scoring (Algorithm 2 with an estimator plug-in).
Measures (i) batch scoring of every held-out case with vectorised numpy and (ii) a per-case online path that
reads one learner's counters from a dict-based store, as a remediation service would. One CPU core.
Output: out_latency.json"""
from evalcore import *
import time, json
At, Ct, te = load('test'); te = te[te.prior_att == 0].reset_index(drop=True)
W['FPKG'] = gated(0.80); w = W['UNG']; prior = 1 - diff
res = {}
# (i) batch
t0 = time.perf_counter()
_, (rep, eidx, _) = load_scores(te, At, Ct, np.ones(NE))
L = te.L.values[rep]; I = SRC[eidx]; na = np.asarray(At[L, I]).ravel(); nc = np.asarray(Ct[L, I]).ravel()
obs = na > 0; th = np.where(obs, nc / np.maximum(na, 1), 0); ww = w[eidx]
Dobs = np.bincount(rep, np.where(obs, ww * (1 - th), 0), len(te)); Uexp = np.bincount(rep, np.where(~obs, ww * (1 - prior[I]), 0), len(te))
nu = np.bincount(rep, (~obs).astype(float), len(te)); no = np.bincount(rep, obs.astype(float), len(te))
t1 = time.perf_counter()
res['batch_cases'] = int(len(te)); res['batch_pairs'] = int(len(rep)); res['batch_seconds'] = t1 - t0
res['batch_microseconds_per_case'] = (t1 - t0) / len(te) * 1e6
# (ii) online: dict store {(learner, exercise): (n, k)}, in-edge lists per target
inn = {}
for k in range(NE): inn.setdefault(int(DST[k]), []).append((int(SRC[k]), float(w[k])))
sample = te.sample(20000, random_state=1)
Ls = sample.L.values; Ts = sample['T'].values
store = {}
for l in np.unique(Ls):
    row = At.getrow(l); rc = Ct.getrow(l)
    for j, v in zip(row.indices, row.data): store[(int(l), int(j))] = (float(v), float(rc[0, j]))
lat = []
for l, t in zip(Ls, Ts):
    s = time.perf_counter(); d = u = 0.0; a = b = 0
    for (i, wi) in inn[int(t)]:
        r = store.get((int(l), i))
        if r is None: u += wi * (1 - prior[i]); a += 1
        else: d += wi * (1 - r[1] / r[0]); b += 1
    lat.append(time.perf_counter() - s)
lat = np.array(lat) * 1e6
res['online_cases'] = 20000; res['online_us_median'] = float(np.median(lat)); res['online_us_p99'] = float(np.percentile(lat, 99))
res['indegree_median'] = float(np.median([len(inn[int(t)]) for t in Ts])); res['indegree_max'] = int(max(len(v) for v in inn.values()))
json.dump(res, open('out_latency.json', 'w'), indent=1); print(res)
