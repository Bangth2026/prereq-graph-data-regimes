"""Deep knowledge tracing (DKT; Piech et al., 2015) baseline for the EAAI extension.

Trained only on training segments (first 60 % of every retained learner's attempts). Writes the hidden
state that precedes (i) the first attempt of every (learner, exercise) pair in the validation and held-out
windows and (ii) the start of each window, so that
    theta_DKT(s, c) = sigmoid(h . W_c + b_c)
can be read for any prerequisite c and any target c under the snapshot and sequential protocols.
Output: data/dkt_states.npz, out_dkt.json. Seeds fixed. CPU only.
"""
import numpy as np, torch, torch.nn as nn, time, json, sys
from sklearn.metrics import roc_auc_score
import os
SEED = int(os.environ.get('DKT_SEED', 2026)); SUBSET = os.environ.get('DKT_SUBSET')   # data-regime study: training learners and seed
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(2)
DATA = sys.argv[2] if len(sys.argv) > 2 else 'data/sorted.npz'
ODIR = sys.argv[4] if len(sys.argv) > 4 else 'data'
OJSON = 'out_dkt.json' if len(sys.argv) <= 4 else f'{ODIR}/out_dkt.json'
z = np.load(DATA); e = z['e'].astype(np.int64); c = z['c'].astype(np.int64)
lid = z['lid']; frac = z['frac']; n = z['n']
import pandas as pd
NC = int(sys.argv[3]) if len(sys.argv) > 3 else len(pd.read_csv("data/junyi_Exercise_table.csv", encoding="utf-8-sig")); H = 64; EMB = 64
elig = n >= 10
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PATIENCE = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # >0: EPOCHS is a maximum; stop after PATIENCE epochs without gain, keep best

class DKT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(2 * NC + 1, EMB, padding_idx=2 * NC)
        s.gru = nn.GRU(EMB, H, batch_first=True)
        s.out = nn.Linear(H, NC)
    def forward(s, x, h0=None):
        o, h = s.gru(s.emb(x), h0); return o, h

# learner boundaries (data sorted by learner, then time)
chg = np.r_[True, lid[1:] != lid[:-1]]; start = np.flatnonzero(chg); end = np.r_[start[1:], len(lid)]
L_elig = np.flatnonzero(elig[start])

def segs(lo_frac):
    """for each eligible learner: rows with frac < lo_frac (training part)"""
    out = []
    for k in L_elig:
        a, b = start[k], end[k]
        m = a + int(np.searchsorted(frac[a:b], lo_frac, 'left'))
        if m - a >= 2: out.append((a, m))
    return out

model = DKT(); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
lossf = nn.BCEWithLogitsLoss()
tr = segs(0.6)
if SUBSET:                                                       # keep only the training segments of the sampled learners
    keepL = set(np.load(SUBSET)['L'].tolist()); tr = [(a_, m_) for a_, m_ in tr if int(lid[a_]) in keepL]
# chunk training sequences into pieces of length <= 200 (truncated BPTT without carry-over)
CH = 200; pieces = []
for a, m in tr:
    for s0 in range(a, m - 1, CH): pieces.append((s0, min(s0 + CH + 1, m)))
pieces = np.array(pieces); print('training pieces', len(pieces), 'steps', int((pieces[:, 1] - pieces[:, 0]).sum()), flush=True)
# hold out 5 % of pieces for early stopping / sanity AUC
rng = np.random.default_rng(1 if SEED == 2026 else SEED); perm = rng.permutation(len(pieces)); nv = max(2, len(pieces) // 20)
vpieces = pieces[perm[:nv]]; tpieces = pieces[perm[nv:]]
tok = e * 2 + c
def batch(pcs):
    Lm = int((pcs[:, 1] - pcs[:, 0]).max())
    X = np.full((len(pcs), Lm - 1), 2 * NC); Y = np.full((len(pcs), Lm - 1), -1); Q = np.zeros((len(pcs), Lm - 1), np.int64)
    for i, (a, b) in enumerate(pcs):
        k = b - a - 1; X[i, :k] = tok[a:b - 1]; Q[i, :k] = e[a + 1:b]; Y[i, :k] = c[a + 1:b]
    return torch.from_numpy(X), torch.from_numpy(Q), torch.from_numpy(Y)
def eval_auc(pcs):
    model.eval(); P = []; T = []
    with torch.no_grad():
        for i in range(0, len(pcs), 512):
            X, Q, Y = batch(pcs[i:i + 512]); o, _ = model(X)
            lg = (o * model.out.weight[Q]).sum(-1) + model.out.bias[Q]; m = Y >= 0
            P.append(torch.sigmoid(lg[m]).numpy()); T.append(Y[m].numpy())
    model.train(); return roc_auc_score(np.concatenate(T), np.concatenate(P))
t0 = time.time(); hist = []
import os
TRAINED = os.path.exists(f'{ODIR}/dkt_model.pt')
if TRAINED:
    ck = torch.load(f'{ODIR}/dkt_model.pt'); model.load_state_dict(ck['state']); hist = ck['hist']
for ep in (range(0) if TRAINED else range(EPOCHS)):
    order = rng.permutation(len(tpieces)); tot = 0
    for bi, i in enumerate(range(0, len(order), 256)):
        X, Q, Y = batch(tpieces[order[i:i + 256]]); o, _ = model(X)
        lg = (o * model.out.weight[Q]).sum(-1) + model.out.bias[Q]; m = Y >= 0
        loss = lossf(lg[m], Y[m].float()); opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        if bi % 500 == 0: print(ep, bi, round(loss.item(), 4), round(time.time() - t0), flush=True)
    a = eval_auc(vpieces); hist.append(a); print('epoch', ep, 'held-out-piece AUC', round(a, 4), round(time.time() - t0), flush=True)
    if PATIENCE:
        if a >= max(hist): best_state = {k: v.clone() for k, v in model.state_dict().items()}; best_ep = ep
        elif ep - best_ep >= PATIENCE: break
if PATIENCE and not TRAINED: model.load_state_dict(best_state); print('early stopping: best epoch', best_ep + 1, flush=True)

if not TRAINED: torch.save(dict(state=model.state_dict(), hist=hist), f'{ODIR}/dkt_model.pt')
# ---- states: run every eligible learner's full sequence, keep hidden state before needed rows ----
need = np.zeros(len(lid), bool)
for lo, hi in [(0.6, 0.8), (0.8, 1.01)]:
    wm = elig & (frac >= lo) & (frac < hi); wi = np.flatnonzero(wm)
    key = lid[wi].astype(np.int64) * NC + e[wi]; _, first = np.unique(key, return_index=True); need[wi[first]] = True
    # window start = first row of the window for each learner
    ws = wi[np.r_[True, lid[wi][1:] != lid[wi][:-1]]]; need[ws] = True
rows = np.flatnonzero(need); print('rows needing a preceding state', len(rows), flush=True)
Hs = np.zeros((len(rows), H), np.float16)
rowpos = {}
model.eval()
lens = end[L_elig] - start[L_elig]; ordL = L_elig[np.argsort(lens)]
CHK = 1000
with torch.no_grad():
    i = 0; done = 0
    while i < len(ordL):
        B = min(256, len(ordL) - i); grp = ordL[i:i + B]
        ln = end[grp] - start[grp]; Lm = int(ln.max())
        X = np.full((B, Lm), 2 * NC)
        for j, k in enumerate(grp): X[j, :ln[j]] = tok[start[k]:end[k]]
        lo_ = np.searchsorted(rows, start[grp]); hi_ = np.searchsorted(rows, end[grp])
        need_p = [rows[lo_[j]:hi_[j]] - start[grp[j]] - 1 for j in range(B)]   # state after the previous attempt
        h = None
        for c0 in range(0, Lm, CHK):
            o, h = model(torch.from_numpy(X[:, c0:c0 + CHK]), h); o = o.numpy()
            for j in range(B):
                p = need_p[j]; m = (p >= c0) & (p < c0 + CHK)
                if m.any():
                    idx_ = np.arange(lo_[j], hi_[j])[m]; Hs[idx_] = o[j, p[m] - c0].astype(np.float16)
        i += B; done += B
        if (done // B) % 50 == 0: print('states', done, '/', len(ordL), round(time.time() - t0), flush=True)
W = model.out.weight.detach().numpy(); b = model.out.bias.detach().numpy()
np.savez(f'{ODIR}/dkt_states.npz', rows=rows, H=Hs, W=W, b=b)
json.dump(dict(epochs=(best_ep + 1 if PATIENCE and not TRAINED else EPOCHS), patience=PATIENCE, hidden=H, emb=EMB, chunk=CH, lr=2e-3, batch=256, train_pieces=int(len(tpieces)),
               heldout_piece_auc_by_epoch=hist, seconds=time.time() - t0), open(OJSON, 'w'), indent=1)
print('done', time.time() - t0)
