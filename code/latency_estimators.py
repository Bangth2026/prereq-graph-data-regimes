"""End-to-end online cost per case with each estimator for unattempted prerequisites (one CPU thread).

A case is a learner reaching a target with K candidate prerequisites (Junyi: median 5). For each estimator the
timed path is what a remediation service runs when the case arrives:
  POP, BKT : table lookup per unattempted prerequisite + ESD features + logistic layer
  LKT      : 9 learner/exercise features per unattempted prerequisite + logistic estimate + ESD + logistic layer
  DKT      : output layer on the learner's current GRU state for the K prerequisites and the target + ESD + layer;
             the GRU state is advanced once per attempt (timed separately, it is part of running a DKT learner model)
  SAKT/AKT : encode the last 100 interactions and answer K + 1 queries (prerequisites and target) + ESD + layer;
             no caching of window encodings between cases
Weights are random: the cost does not depend on their values. Output: out_latency_estimators.json
"""
import time, json, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)
NC = 837; K = 5; REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; Wn = 100
def timeit(f, reps=REPS):
    for _ in range(50): f()
    t = []
    for _ in range(reps):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    t = np.array(t) * 1e6; return dict(median_us=float(np.median(t)), p99_us=float(np.percentile(t, 99)))
res = dict(K=K, threads=1)
pop = rng.random(NC); w = rng.random(K); obs = np.array([True, False, False, True, False]); th_o = rng.random(K)
coef = rng.standard_normal(9); b0 = 0.1; I = rng.choice(NC, K, replace=False); tgt = int(rng.integers(NC))
def esd_layer(theta_un):
    D = (w * (1 - th_o) * obs).sum(); U = (w * (1 - theta_un) * ~obs).sum()
    z = coef[0] * D + coef[1] * U + coef[2] * (~obs).sum() + coef[3] * obs.sum() + b0
    return 1 / (1 + np.exp(-z))
res['POP_or_BKT'] = timeit(lambda: esd_layer(pop[I]))
lk = rng.standard_normal(9); feats = rng.random((K, 9))
res['LKT'] = timeit(lambda: esd_layer(1 / (1 + np.exp(-(feats @ lk)))))
# DKT
H = 64; gru = nn.GRUCell(64, H); emb = nn.Embedding(2 * NC, 64); Wout = rng.standard_normal((NC, H)).astype(np.float32); bout = rng.standard_normal(NC).astype(np.float32)
h = torch.zeros(1, H); hn = np.zeros(H, np.float32)
def dkt_case():
    p = 1 / (1 + np.exp(-(Wout[np.r_[I, tgt]] @ hn + bout[np.r_[I, tgt]]))); return esd_layer(p[:K])
res['DKT_case'] = timeit(dkt_case)
x = torch.tensor([5])
def dkt_step():
    with torch.no_grad(): return gru(emb(x), h)
res['DKT_state_update_per_attempt'] = timeit(dkt_step)
# SAKT / AKT: class definitions from kt_attn.py
src = open('kt_attn.py').read(); cls = src[src.index('class Block'):src.index('def pieces')]
for KIND in ['sakt', 'akt']:
    g = dict(nn=nn, torch=torch, F=F, np=np, KIND=KIND, W=100, T=100, D=64, H=4, NC=NC); exec(cls, g); model = g['KT']().eval()
    Q = torch.randint(0, NC, (1, Wn)); X = Q * 2 + torch.randint(0, 2, (1, Wn)); valid = torch.ones(1, Wn, dtype=torch.bool)
    qex = torch.tensor([list(I) + [tgt]]); distw = (Wn - torch.arange(Wn)).float().unsqueeze(0)
    def att_case():
        with torch.no_grad():
            _, Eq, Ex, qe, ke = model.encode(Q, X, valid)
            lg = model.query_rows(qex, None, Eq, Ex, qe, ke, valid, distw)
        return esd_layer(torch.sigmoid(lg[0, :K]).numpy())
    res[f'{KIND.upper()}_case'] = timeit(att_case, 500)
json.dump(res, open('out_latency_estimators.json', 'w'), indent=1); print(json.dumps(res, indent=1))
