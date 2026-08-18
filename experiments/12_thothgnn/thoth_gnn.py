"""ThothGNN Stage 4: bipolar heterogeneous readout with an anchored base score.

Implements the architecture plan. The three ideas that distinguish it from the
GraphSAGE baseline that failed (test AUC 0.6501 against 0.7128 for a plain
stance count):

1. Sign is structural, never learned. AGREE and DISAGREE are separate relations
   with separate weight matrices, so a debate's DISAGREE cannot become support.
   The learned edge MLP produces a magnitude in [0,1] and nothing else.

2. The class score is anchored on mal_share. Following Gradual AA-CBR's MLP
   semantics the base score enters additively in pre-activation:

       score_MAL - score_BEN = beta * logit(mal_share) + gamma * (w.z_MAL - w.z_BEN)

   with beta=1, gamma=0 at initialisation. The model therefore *starts* at the
   0.7128 baseline and can only move away from it if that reduces CV loss.
   gamma is then a direct scalar measurement of how much the graph holds beyond
   the stance count -- a result whatever its value.

3. Readout is bipolar attention over claims, signed by whether a claim's stance
   matches the class. Because h_i already encodes how much AGREE and DISAGREE a
   claim received, a claim that was successfully disputed contributes less than
   its raw stance implies. UNIFORM-ATT isolates exactly this.

Validation is 78 graphs, too small to select epochs on, so 5-fold CV over the
546 training graphs selects the epoch and every hyperparameter; test is touched
once per configuration.

Usage:
  python thoth_gnn.py --corpus corpus/v5q.pt --configs BASE FULL --seeds 5
"""
from __future__ import annotations

import argparse, json, logging, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, HeteroConv
from torch_geometric.utils import softmax as pyg_softmax

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-4
CLAIM_DIM, TRIPLE_RAW, REL_EMB = 12, 2, 8

# Each entry flips exactly one component relative to FULL.
CONFIGS = {
    "BASE":        dict(gamma_free=False, gamma0=0.0),
    "FULL":        dict(),
    "UNIFORM-ATT": dict(uniform_att=True),
    "NO-ANCHOR":   dict(beta_free=False, beta0=0.0, gamma_free=False, gamma0=1.0),
    "NO-XLAYER":   dict(xlayer=False),
    "EDGE-FIXED":  dict(learned_edge=False),
    "NO-SIGN":     dict(sign=False),
    "NO-KG-NODES": dict(kg_nodes=False),
    "NO-NEXT":     dict(next_turn=False),
    "FUSE-GATE":   dict(fuse="gate"),
    "BALANCE":     dict(balance=True),
}
DEFAULTS = dict(gamma_free=True, gamma0=0.0, beta_free=True, beta0=1.0,
                uniform_att=False, xlayer=True, learned_edge=True, sign=True,
                kg_nodes=True, next_turn=True, fuse="concat", balance=False)

REL_CC = [("claim", "agree", "claim"), ("claim", "disagree", "claim")]
REL_NX = [("claim", "next", "claim")]
REL_KG = [("claim", "cites", "triple"), ("triple", "cited_by", "claim")]
REL_BAL = [("claim", "balance", "claim")]


# --------------------------------------------------------------------------
# corpus -> HeteroData
# --------------------------------------------------------------------------
def _ei(pairs):
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def to_hetero(g: dict, cfg: dict) -> HeteroData:
    d = HeteroData()
    x = torch.tensor(g["x_claim"], dtype=torch.float)
    d["claim"].x = x
    n = x.size(0)

    agree, dis = list(g["agree"]), list(g["disagree"])
    if not cfg["sign"]:
        # NO-SIGN: one relation, one weight matrix, polarity discarded.
        agree, dis = agree + dis, []
    d["claim", "agree", "claim"].edge_index = _ei(agree)
    d["claim", "disagree", "claim"].edge_index = _ei(dis)

    nxt = list(g["next"]) if cfg["next_turn"] else []
    d["claim", "next", "claim"].edge_index = _ei(nxt)

    if cfg["balance"]:
        # SGCN balance theory: the enemy of my enemy. Composing two DISAGREE
        # hops yields a path that should behave like support.
        out = {}
        for s, t in g["disagree"]:
            out.setdefault(s, []).append(t)
        bal = [(s, u) for s, ts in out.items() for t in ts for u in out.get(t, [])
               if u != s]
        d["claim", "balance", "claim"].edge_index = _ei(sorted(set(bal)))
    else:
        d["claim", "balance", "claim"].edge_index = _ei([])

    if cfg["kg_nodes"] and g["x_triple"]:
        tx = torch.tensor(g["x_triple"], dtype=torch.float)
        d["triple"].x = tx
        d["triple"].rel = torch.tensor(g["rel_triple"], dtype=torch.long)
        ci = [(a, b) for a, b, _ in g["cites"]]
        w = torch.tensor([s for _, _, s in g["cites"]], dtype=torch.float)
        # Normalise per target so an add-aggregation is exactly the weighted
        # mean the plan specifies (SUM cos*h / SUM cos).
        d["claim", "cites", "triple"].edge_index = _ei(ci)
        d["claim", "cites", "triple"].edge_weight = _norm_by(w, _ei(ci)[1] if ci else None,
                                                             len(g["x_triple"]))
        rc = [(b, a) for a, b, _ in g["cites"]]
        d["triple", "cited_by", "claim"].edge_index = _ei(rc)
        d["triple", "cited_by", "claim"].edge_weight = _norm_by(w, _ei(rc)[1] if rc else None, n)
    else:
        d["triple"].x = torch.zeros(1, TRIPLE_RAW)
        d["triple"].rel = torch.zeros(1, dtype=torch.long)
        for r in REL_KG:
            d[r].edge_index = _ei([])
            d[r].edge_weight = torch.zeros(0)

    d["claim"].sign = torch.tensor(g["class_sign"], dtype=torch.float)
    d.mal_share = torch.tensor([g["mal_share"]], dtype=torch.float)
    d.y = torch.tensor([g["y"]], dtype=torch.long)
    return d


def _norm_by(w, dst, n):
    if dst is None or w.numel() == 0:
        return torch.zeros(0)
    s = torch.zeros(n).index_add_(0, dst, w).clamp(min=1e-6)
    return w / s[dst]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class EdgeGate(nn.Module):
    """m_ji = sigmoid(MLP([h_j, h_i, type])) in [0,1]; magnitude only."""

    def __init__(self, h, n_types=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * h + n_types, h // 2), nn.ReLU(),
                                 nn.Linear(h // 2, 1))
        self.n_types = n_types

    def forward(self, h, ei, t):
        if ei.size(1) == 0:
            return torch.zeros(0, device=h.device)
        oh = F.one_hot(torch.full((ei.size(1),), t, device=h.device),
                       self.n_types).float()
        m = torch.sigmoid(self.net(torch.cat([h[ei[0]], h[ei[1]], oh], -1))).squeeze(-1)
        assert torch.all(m >= 0), "edge magnitude must stay non-negative"
        return m


class ThothGNN(nn.Module):
    def __init__(self, cfg, n_rel, hidden=32, layers=2, dropout=0.5):
        super().__init__()
        self.cfg, self.L, self.p = cfg, layers, dropout
        self.rel_emb = nn.Embedding(n_rel, REL_EMB)
        self.claim_in = nn.Linear(CLAIM_DIM, hidden)
        self.triple_in = nn.Linear(TRIPLE_RAW + REL_EMB, hidden)

        rels = REL_CC + REL_NX + REL_KG + REL_BAL
        self.convs = nn.ModuleList([
            HeteroConv({r: GraphConv(hidden, hidden, aggr="mean" if r in REL_CC else "add")
                        for r in rels}, aggr="sum") for _ in range(layers)])
        self.self_lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.gate = nn.ModuleList([EdgeGate(hidden) for _ in range(layers)]) \
            if cfg["learned_edge"] else None
        if cfg["fuse"] == "gate":
            self.fuse_g = nn.Linear(2 * hidden, hidden)

        out_dim = hidden * (layers + 1) if cfg["xlayer"] else hidden
        self.jk = nn.Linear(out_dim, hidden)
        self.class_emb = nn.Embedding(2, hidden)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, 1))
        self.w = nn.Linear(hidden, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor(float(cfg["beta0"])),
                                 requires_grad=cfg["beta_free"])
        self.gamma = nn.Parameter(torch.tensor(float(cfg["gamma0"])),
                                  requires_grad=cfg["gamma_free"])

    def forward(self, b):
        cfg = self.cfg
        h = {"claim": self.claim_in(b["claim"].x),
             "triple": self.triple_in(torch.cat(
                 [b["triple"].x, self.rel_emb(b["triple"].rel)], -1))}
        hs = [h["claim"]]

        for l in range(self.L):
            ew = {}
            for t, r in enumerate(REL_CC + REL_NX + REL_BAL):
                ei = b[r].edge_index
                ew[r] = (self.gate[l](h["claim"], ei, t) if self.gate is not None
                         else torch.ones(ei.size(1), device=ei.device))
            for r in REL_KG:
                ew[r] = b[r].edge_weight            # fixed retrieval cosine

            hd = {k: F.dropout(v, self.p, self.training) for k, v in h.items()}
            agg = self.convs[l](hd, b.edge_index_dict, edge_weight_dict=ew)
            a = agg.get("claim", torch.zeros_like(h["claim"]))
            new = self.norm[l](F.relu(self.self_lin[l](h["claim"]) + a))
            h = {"claim": new, "triple": agg.get("triple", h["triple"])}
            hs.append(new)

        hf = self.jk(torch.cat(hs, -1) if cfg["xlayer"] else hs[-1])

        # Bipolar signed attention pooling toward the two class nodes.
        batch = b["claim"].batch
        sign = b["claim"].sign
        z = []
        for c in (0, 1):
            e = self.class_emb.weight[c].expand(hf.size(0), -1)
            if cfg["uniform_att"]:
                cnt = torch.zeros(int(batch.max()) + 1, device=hf.device)
                cnt.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
                al = (1.0 / cnt.clamp(min=1))[batch].unsqueeze(-1)
            else:
                al = pyg_softmax(self.att(torch.cat([hf, e], -1)), batch)
            s = sign if c == 1 else -sign        # class 1 = MALIGNANT
            zc = torch.zeros(int(batch.max()) + 1, hf.size(1), device=hf.device)
            zc.index_add_(0, batch, al * s.unsqueeze(-1) * hf)
            z.append(zc)

        delta = (self.w(z[1]) - self.w(z[0])).squeeze(-1).clamp(-1.0, 1.0)
        ms = b.mal_share.clamp(EPS, 1 - EPS)
        return self.beta * torch.log(ms / (1 - ms)) + self.gamma * delta


# --------------------------------------------------------------------------
# metrics / training
# --------------------------------------------------------------------------
def auc(s, y):
    p = [a for a, b in zip(s, y) if b == 1]
    n = [a for a, b in zip(s, y) if b == 0]
    if not p or not n:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))


def bacc(s, y, t):
    tp = sum(1 for a, b in zip(s, y) if b == 1 and a >= t)
    fn = sum(1 for a, b in zip(s, y) if b == 1 and a < t)
    tn = sum(1 for a, b in zip(s, y) if b == 0 and a < t)
    fp = sum(1 for a, b in zip(s, y) if b == 0 and a >= t)
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def collate(graphs, dev, batch=128, shuffle=False):
    """Pre-collate and move to the device ONCE.

    These graphs hold ~19 nodes each, so per-epoch collation and host-to-device
    transfer dominated the step time -- re-collating every epoch made a single
    seed take 4.5 minutes. The batches are fixed after this call, which trades
    per-epoch reshuffling for roughly an order of magnitude in throughput.
    """
    idx = list(range(len(graphs)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for i in range(0, len(idx), batch):
        chunk = [graphs[j] for j in idx[i:i + batch]]
        out.append(next(iter(DataLoader(chunk, batch_size=len(chunk)))).to(dev))
    return out


def run_epochs(model, tr_batches, ev_sets, epochs, lr, wd, l2g, pos_w, dev):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.tensor(pos_w, device=dev)
    hist = []
    for _ in range(epochs):
        model.train()
        for b in tr_batches:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(b), b.y.float(),
                                                      pos_weight=pw)
            loss = loss + l2g * model.gamma.pow(2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ev_sets:
            hist.append([evaluate(model, e)[0] for e in ev_sets])
    return hist


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    s, y = [], []
    for b in batches:
        s += model(b).cpu().tolist()
        y += b.y.cpu().tolist()
    return auc(s, y), s, y


def mal_share_cv(graphs):
    return auc([g["mal_share"] for g in graphs], [g["y"] for g in graphs])


def set_seed(k):
    random.seed(k); np.random.seed(k); torch.manual_seed(k)
    torch.cuda.manual_seed_all(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--configs", nargs="+", default=["BASE", "FULL"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--l2-gamma", type=float, nargs="+", default=[0.0, 1e-2])
    ap.add_argument("--augment", default=None, help="second corpus for E3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.corpus, weights_only=False)
    graphs, meta = blob["graphs"], blob["meta"]
    aug = torch.load(args.augment, weights_only=False)["graphs"] if args.augment else None
    logger.info("%s: %d graphs%s", Path(args.corpus).stem, len(graphs),
                f" (+{len(aug)} augment)" if aug else "")

    results = {}
    for name in args.configs:
        cfg = {**DEFAULTS, **CONFIGS[name]}
        tr = [g for g in graphs if g["split"] == "train"]
        te = [g for g in graphs if g["split"] == "test"]
        va = [g for g in graphs if g["split"] == "val"]
        pos_w = sum(1 for g in tr if g["y"] == 0) / max(1, sum(1 for g in tr if g["y"] == 1))

        D_tr = [to_hetero(g, cfg) for g in tr]
        D_te = [to_hetero(g, cfg) for g in te]
        D_va = [to_hetero(g, cfg) for g in va]
        aug_by_sid = {}
        if aug:
            for g in aug:
                if g["split"] == "train":
                    aug_by_sid[g["sample_id"]] = to_hetero(g, cfg)

        te_batches = collate(D_te, dev, 256)
        fitv_batches = collate(D_tr + D_va, dev, 256)
        seed_scores = []
        for seed in range(args.seeds):
            set_seed(seed)
            # ---- CV over the 546 training graphs: pick epoch and l2(gamma) ----
            best = (-1, args.epochs, args.l2_gamma[0])
            # BASE is gamma=0, so the score is beta*logit(mal_share): a positive
            # monotone map, and AUC is invariant to it. There is nothing for CV
            # to select, so skip it rather than burn an hour reselecting noise.
            pure_anchor = (not cfg["gamma_free"]) and cfg["gamma0"] == 0.0
            if pure_anchor:
                best = (mal_share_cv(tr), 30, args.l2_gamma[0])
            elif cfg["gamma_free"] or cfg["beta_free"]:
                folds = [list(range(i, len(D_tr), args.folds)) for i in range(args.folds)]
                for l2g in args.l2_gamma:
                    curves = []
                    for h in folds:
                        hs = set(h)
                        fit = [D_tr[i] for i in range(len(D_tr)) if i not in hs]
                        if aug_by_sid:
                            fit += [aug_by_sid[tr[i]["sample_id"]]
                                    for i in range(len(D_tr))
                                    if i not in hs and tr[i]["sample_id"] in aug_by_sid]
                        m = ThothGNN(cfg, meta["n_relations"], args.hidden).to(dev)
                        c = run_epochs(m, collate(fit, dev, args.batch, True),
                                       [collate([D_tr[i] for i in h], dev, 256)],
                                       args.epochs, args.lr, args.wd, l2g, pos_w, dev)
                        curves.append([r[0] for r in c])
                    mean = [sum(c[e] for c in curves) / len(curves)
                            for e in range(args.epochs)]
                    e = int(np.argmax(mean))
                    if mean[e] > best[0]:
                        best = (mean[e], e + 1, l2g)
            cv_auc, n_ep, l2g = best

            # ---- retrain on the full training split, evaluate test once ----
            set_seed(seed)
            fit = list(D_tr) + ([aug_by_sid[g["sample_id"]] for g in tr
                                 if g["sample_id"] in aug_by_sid] if aug_by_sid else [])
            m = ThothGNN(cfg, meta["n_relations"], args.hidden).to(dev)
            run_epochs(m, collate(fit, dev, args.batch, True), [],
                       n_ep, args.lr, args.wd, l2g, pos_w, dev)
            a_te, s_te, y_te = evaluate(m, te_batches)
            _, s_fit, y_fit = evaluate(m, fitv_batches)
            thr = max(sorted(set(s_fit)), key=lambda t: bacc(s_fit, y_fit, t))
            seed_scores.append(dict(seed=seed, cv=cv_auc, epochs=n_ep, l2g=l2g,
                                    test_auc=a_te, test_bacc=bacc(s_te, y_te, thr),
                                    gamma=float(m.gamma.detach()),
                                    beta=float(m.beta.detach())))
            logger.info("  %-12s seed %d  cv %.4f  ep %3d  test AUC %.4f  bAcc %.4f  "
                        "gamma %+.3f", name, seed, cv_auc, n_ep, a_te,
                        seed_scores[-1]["test_bacc"], seed_scores[-1]["gamma"])

        f = lambda k: (float(np.mean([s[k] for s in seed_scores])),
                       float(np.std([s[k] for s in seed_scores])))
        results[name] = dict(runs=seed_scores, test_auc=f("test_auc"),
                             test_bacc=f("test_bacc"), gamma=f("gamma"))
        logger.info("%-12s TEST AUC %.4f +- %.4f | bAcc %.4f +- %.4f | gamma %+.3f",
                    name, *f("test_auc"), *f("test_bacc"), f("gamma")[0])

    print("\n" + "=" * 78)
    print(f"{Path(args.corpus).stem}"
          f"{'  (+E3 augment)' if args.augment else ''}")
    print(f"{'config':14s} {'test AUC':>16s} {'test bAcc':>16s} {'gamma':>9s}")
    for k, v in results.items():
        print(f"{k:14s} {v['test_auc'][0]:8.4f} +-{v['test_auc'][1]:5.3f} "
              f"{v['test_bacc'][0]:8.4f} +-{v['test_bacc'][1]:5.3f} {v['gamma'][0]:+9.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
