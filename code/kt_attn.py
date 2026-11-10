"""Attention-based knowledge tracing models (SAKT- and AKT-style) with a query interface.

Usage:  python kt_attn.py {sakt|akt} DATA_NPZ NUM_UNITS QUERY_NPZ OUT_NPZ [EPOCHS] [PIECE_FRACTION]

DATA_NPZ   : sorted log with arrays e (unit id), c (correct), lid (learner, sorted), frac (position share), n (attempts)
QUERY_NPZ  : arrays rows (int64 row index r) and ex (unit id); the model returns P(correct on ex | attempts before row r)
Training uses only rows with frac < 0.6 (training segments) of learners with n >= 10.

Both models attend over a window of the last W = 100 interactions.
SAKT (Pandey and Karypis, 2019): the query exercise attends over past interaction embeddings; positions enter as a
  learned per-head bias by distance (relative rather than absolute positions, so that a query can sit at any row).
AKT (Ghosh et al., 2020), without Rasch embeddings because items are single-level here: a question encoder and a
  knowledge encoder (self-attention with monotonic exponential decay by distance), and a knowledge retriever in which
  the current question attends over past question encodings to read past knowledge encodings.
Seeds fixed; CPU.
"""
import sys, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(2026); np.random.seed(2026); torch.set_num_threads(2)
KIND, DATA, NC, QF, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
EPOCHS = int(sys.argv[6]) if len(sys.argv) > 6 else 1
PFRAC = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
W = 100; T = 100; D = 64; H = 4
z = np.load(DATA); e = z['e'].astype(np.int64); c = z['c'].astype(np.int64); lid = z['lid']; frac = z['frac']; n = z['n']
chg = np.r_[True, lid[1:] != lid[:-1]]; start = np.flatnonzero(chg); end = np.r_[start[1:], len(lid)]
elig_l = np.flatnonzero(n[start] >= 10)
xtok = e * 2 + c

class Block(nn.Module):
    def __init__(s, decay, relbias):
        super().__init__()
        s.q = nn.Linear(D, D); s.k = nn.Linear(D, D); s.v = nn.Linear(D, D); s.o = nn.Linear(D, D)
        s.nk = nn.Parameter(torch.zeros(H, 1, D // H)); s.nv = nn.Parameter(torch.zeros(H, 1, D // H))
        s.ln1 = nn.LayerNorm(D); s.ln2 = nn.LayerNorm(D); s.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        s.decay = decay; s.relbias = relbias
        if decay: s.theta = nn.Parameter(torch.full((H,), -2.0))
        if relbias: s.rb = nn.Parameter(torch.zeros(H, W + 2))
        s.drop = nn.Dropout(0.05)
    def forward(s, xq, xk, xv, allowed, dist):
        # xq [B,Lq,D]; xk,xv [B,Lk,D]; allowed [B,Lq,Lk] bool; dist [B,Lq,Lk] float (query pos - key pos)
        B, Lq, _ = xq.shape; Lk = xk.shape[1]; dh = D // H
        Q = s.q(xq).view(B, Lq, H, dh).transpose(1, 2); K = s.k(xk).view(B, Lk, H, dh).transpose(1, 2); V = s.v(xv).view(B, Lk, H, dh).transpose(1, 2)
        sc = Q @ K.transpose(-1, -2) / dh ** 0.5                                  # [B,H,Lq,Lk]
        dd = dist.unsqueeze(1).clamp(min=0)
        if s.decay: sc = sc * torch.exp(-F.softplus(s.theta).view(1, H, 1, 1) * dd / 10.0)
        if s.relbias: sc = sc + s.rb[:, dd.squeeze(1).clamp(max=W + 1).long()].permute(1, 0, 2, 3)
        sc = sc.masked_fill(~allowed.unsqueeze(1), float('-inf'))
        sn = (Q * s.nk.unsqueeze(0)).sum(-1, keepdim=True) / dh ** 0.5           # null key, always allowed
        a = torch.softmax(torch.cat([sc, sn], -1), -1); a = s.drop(a)
        out = a[..., :-1] @ V + a[..., -1:] * s.nv.unsqueeze(0)
        out = s.o(out.transpose(1, 2).reshape(B, Lq, D))
        h = s.ln1(xq + s.drop(out)); return s.ln2(h + s.drop(s.ff(h)))

class KT(nn.Module):
    def __init__(s):
        super().__init__()
        s.eq = nn.Embedding(NC + 1, D, padding_idx=NC); s.ex = nn.Embedding(2 * NC + 1, D, padding_idx=2 * NC)
        if KIND == 'sakt': s.b = Block(False, True)
        else: s.bq = Block(True, False); s.bk = Block(True, False); s.br = Block(True, False)
        s.out = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Dropout(0.05), nn.Linear(D, 1))
    def encode(s, Qt, Xt, valid):
        """chunk forward: per-position question/knowledge encodings and next-response logits"""
        B, L = Qt.shape; Eq = s.eq(Qt); Ex = s.ex(Xt)
        i = torch.arange(L); dist = (i.view(-1, 1) - i.view(1, -1)).float().unsqueeze(0).expand(B, L, L)
        vj = valid.unsqueeze(1)
        past = vj & (dist > 0) & (dist <= W); inc = vj & (dist >= 0) & (dist <= W)
        if KIND == 'sakt':
            h = s.b(Eq, Ex, Ex, past, dist); qe = ke = None
        else:
            qe = s.bq(Eq, Eq, Eq, inc, dist); ke = s.bk(Ex, Ex, Ex, inc, dist); h = s.br(qe, qe, ke, past, dist)
        return s.out(torch.cat([h, Eq], -1)).squeeze(-1), Eq, Ex, qe, ke
    def query(s, qex, Eqw, Exw, qew, kew, vw, distw):
        """qex [N] query units; *w [N,W,D] window tensors ending just before the query row; vw [N,W] valid; distw [N,W]"""
        Eqq = s.eq(qex).unsqueeze(1)
        if KIND == 'sakt':
            h = s.b(Eqq, Exw, Exw, vw.unsqueeze(1), distw.unsqueeze(1))
        else:
            keys = torch.cat([Eqw, Eqq], 1); al = torch.cat([vw, torch.ones_like(vw[:, :1])], 1).unsqueeze(1)
            dq = torch.cat([distw, torch.zeros_like(distw[:, :1])], 1).unsqueeze(1)
            qq = s.bq(Eqq, keys, keys, al, dq)
            h = s.br(qq, qew, kew, vw.unsqueeze(1), distw.unsqueeze(1))
        return s.out(torch.cat([h, Eqq], -1)).squeeze(-1).squeeze(-1)
    def query_rows(s, qex, qm, Eqw, Exw, qew, kew, vw, distw):
        """same as query, grouped by row: qex,qm [R,Qm] query units and mask; window tensors [R,W,*] shared by the
        Qm queries of a row. Each query is processed independently (per-position blocks), so the result equals query."""
        Eqq = s.eq(qex); R, Qm = qex.shape
        al = vw.unsqueeze(1).expand(R, Qm, W); dq = distw.unsqueeze(1).expand(R, Qm, W)
        if KIND == 'sakt':
            h = s.b(Eqq, Exw, Exw, al, dq)
        else:
            keys = torch.cat([Eqw, Eqq], 1)
            al2 = torch.cat([al, torch.eye(Qm, dtype=torch.bool).unsqueeze(0).expand(R, Qm, Qm)], 2)
            dq2 = torch.cat([dq, torch.zeros(R, Qm, Qm)], 2)
            qq = s.bq(Eqq, keys, keys, al2, dq2)
            h = s.br(qq, qew, kew, al, dq)
        return s.out(torch.cat([h, Eqq], -1)).squeeze(-1)

def pieces(lo_frac=None):
    """training pieces: (prefix start, target start, target end) inside training segments"""
    P = []
    for k in elig_l:
        a, b = start[k], end[k]
        m = a + int(np.searchsorted(frac[a:b], 0.6, 'left')) if lo_frac is None else b
        for s0 in range(a, m, T):
            P.append((max(a, s0 - W), s0, min(s0 + T, m)))
    return np.array(P, dtype=np.int64)
def batch(P):
    L = int((P[:, 2] - P[:, 0]).max())
    Qt = np.full((len(P), L), NC); Xt = np.full((len(P), L), 2 * NC); Y = np.full((len(P), L), -1); V = np.zeros((len(P), L), bool)
    for j, (p0, s0, s1) in enumerate(P):
        k = s1 - p0; Qt[j, :k] = e[p0:s1]; Xt[j, :k] = xtok[p0:s1]; V[j, :k] = True; Y[j, s0 - p0:k] = c[s0:s1]
    return torch.from_numpy(Qt), torch.from_numpy(Xt), torch.from_numpy(V), torch.from_numpy(Y)

t0 = time.time(); model = KT(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
P = pieces(); rng = np.random.default_rng(1); perm = rng.permutation(len(P)); nv = len(P) // 40
PV = P[perm[:nv]]; PT = P[perm[nv:]]
print(KIND, 'pieces', len(P), 'tokens', int((P[:, 2] - P[:, 1]).sum()), flush=True)
from sklearn.metrics import roc_auc_score
def val_auc():
    model.eval(); pr = []; ys = []
    with torch.no_grad():
        for i in range(0, len(PV), 128):
            Qt, Xt, V, Y = batch(PV[i:i + 128]); lg = model.encode(Qt, Xt, V)[0]; m = Y >= 0
            pr.append(torch.sigmoid(lg[m]).numpy()); ys.append(Y[m].numpy())
    model.train(); return float(roc_auc_score(np.concatenate(ys), np.concatenate(pr)))
hist = []
for ep in range(EPOCHS):
    order = rng.permutation(len(PT))[: int(PFRAC * len(PT))]
    for bi, i in enumerate(range(0, len(order), 64)):
        Qt, Xt, V, Y = batch(PT[order[i:i + 64]]); lg = model.encode(Qt, Xt, V)[0]; m = Y >= 0
        loss = F.binary_cross_entropy_with_logits(lg[m], Y[m].float()); opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if bi % 300 == 0: print(ep, bi, len(order) // 64, round(loss.item(), 4), round(time.time() - t0), flush=True)
    hist.append(val_auc()); print('epoch', ep, 'held-back piece AUC', round(hist[-1], 4), round(time.time() - t0), flush=True)

# ---------------- queries: process each learner's whole sequence in chunks of T rows with a W-row prefix
qz = np.load(QF); qrows = qz['rows'].astype(np.int64); qex = qz['ex'].astype(np.int64)
o = np.argsort(qrows, kind='stable'); qrows_s = qrows[o]; qex_s = qex[o]; prob = np.full(len(qrows), np.nan, np.float32)
model.eval()
CH = 200
import os; CHECK = os.environ.get('KT_CHECK') == '1'
with torch.no_grad():
    # chunks covering every row r-1 for which a query exists
    need_prev = np.unique(qrows_s - 1)
    L_of = lid[need_prev]; li = np.searchsorted(start, need_prev, 'right') - 1
    a_of = start[li]; kch = (need_prev - a_of) // CH
    chunks = np.unique(np.c_[a_of + kch * CH, end[li]], axis=0)        # (target start, learner end)
    C = np.c_[np.maximum(start[np.searchsorted(start, chunks[:, 0], 'right') - 1], chunks[:, 0] - W), chunks[:, 0], np.minimum(chunks[:, 0] + CH, chunks[:, 1])]
    done = 0
    for i in range(0, len(C), 32):
        Cb = C[i:i + 32]; Qt, Xt, V, _ = batch(Cb)
        _, Eq, Ex, qe, ke = model.encode(Qt, Xt, V)
        # queries whose r-1 lies in the target range of a chunk of this batch
        lo = np.searchsorted(qrows_s - 1, Cb[:, 1], 'left'); hi = np.searchsorted(qrows_s - 1, Cb[:, 2], 'left')
        bidx = np.concatenate([np.full(h_ - l_, j) for j, (l_, h_) in enumerate(zip(lo, hi))]).astype(np.int64)
        qi = np.concatenate([np.arange(l_, h_) for l_, h_ in zip(lo, hi)]).astype(np.int64)
        if len(qi) == 0: continue
        p = (qrows_s[qi] - 1) - Cb[bidx, 0]                                     # local index of r-1
        # group the queries of this batch by (chunk, row): window tensors are gathered once per row
        rk = bidx * 100000 + p; ur, rinv = np.unique(rk, return_inverse=True); rinv = rinv.ravel()
        cnt = np.bincount(rinv); slot = np.arange(len(qi)) - np.r_[0, np.cumsum(cnt)][:-1][rinv]
        rb_ = (ur // 100000).astype(np.int64); rp = (ur % 100000).astype(np.int64)
        widx = rp[:, None] - np.arange(W)[::-1][None, :]
        wv = widx >= 0; widx = np.maximum(widx, 0)
        distw = torch.from_numpy((rp[:, None] + 1 - widx).astype(np.float32))
        Qm = int(cnt.max()); QX = np.full((len(ur), Qm), NC, np.int64); QX[rinv, slot] = qex_s[qi]
        rows_per = max(1, 4096 // Qm)
        for s0 in range(0, len(ur), rows_per):
            sl = slice(s0, s0 + rows_per); bb = torch.from_numpy(rb_[sl]); ww = torch.from_numpy(widx[sl])
            g = lambda X: X[bb.unsqueeze(1), ww] if X is not None else None
            lg = model.query_rows(torch.from_numpy(QX[sl]), None, g(Eq), g(Ex), g(qe), g(ke), torch.from_numpy(wv[sl]), distw[sl])
            m = (rinv >= s0) & (rinv < s0 + rows_per)
            prob[o[qi[m]]] = torch.sigmoid(lg[rinv[m] - s0, slot[m]]).numpy()
            if CHECK and s0 == 0:
                ii = np.flatnonzero(m)[:300]; pp = p[ii]; wi_ = pp[:, None] - np.arange(W)[::-1][None, :]; wv_ = wi_ >= 0; wi_ = np.maximum(wi_, 0)
                b2 = torch.from_numpy(bidx[ii]); w2 = torch.from_numpy(wi_); g2 = lambda X: X[b2.unsqueeze(1), w2] if X is not None else None
                ref = torch.sigmoid(model.query(torch.from_numpy(qex_s[qi[ii]]), g2(Eq), g2(Ex), g2(qe), g2(ke), torch.from_numpy(wv_), torch.from_numpy((pp[:, None] + 1 - wi_).astype(np.float32)))).numpy()
                print('CHECK max abs diff', float(np.abs(ref - prob[o[qi[ii]]]).max()), flush=True); sys.exit()
        done += len(qi)
        if (i // 32) % 200 == 0: print('queries', done, '/', len(qrows), round(time.time() - t0), flush=True)
assert not np.isnan(prob).any(), 'unanswered queries'
np.savez(OUT, rows=qrows, ex=qex, p=prob)
json.dump(dict(kind=KIND, window=W, d=D, heads=H, epochs=EPOCHS, piece_fraction=PFRAC, lr=1e-3, batch=64,
               heldback_piece_auc=hist, n_queries=int(len(qrows))), open(OUT.replace('.npz', '.json'), 'w'), indent=1)
print('done', round(time.time() - t0), flush=True)
