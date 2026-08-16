"""Train a GNN on the v2 debate graphs and compare it to non-graph baselines.

Until now nothing in this repo consumed the graph structure. The "verdict
accuracy" reported for a dataset is just `evidence_score >= 0.5` thresholded,
and the per-feature AUCs are single-scalar diagnostics. This script produces an
actual model output from the graphs, and — more importantly — says whether the
structure earns its keep.

Three models on the same split, so the comparison is honest:

  probe-mlp   the 17/29 probe vector alone, no graph at all. This is the number
              the GNN has to beat; if it cannot, the graph is decoration.
  mean-pool   claim node features averaged, no message passing. Isolates how
              much comes from the nodes versus from the edges.
  gnn         GraphSAGE over the claim-claim and claim-triple edges.

Node features per claim: p_yes, stance_weight, their product, round index,
expert id, whether the label was explicit, and the claim's own label. Triple
nodes get a learned type embedding, since they carry no measurement.

Reports balanced accuracy alongside AUC throughout: the probe score is poorly
calibrated at 0.5, so AUC alone overstates how usable it is.
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

_FEATS_PER_CLAIM = 8


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def balanced_acc(scores, labels, th):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    tp = sum(1 for s in pos if s >= th)
    tn = sum(1 for s in neg if s < th)
    return 0.5 * (tp / len(pos) + tn / len(neg))


def best_balanced_acc(scores, labels):
    return max((balanced_acc(scores, labels, t), t) for t in sorted(set(scores)))


def load_graph(path: Path, feat_index: dict[str, int]) -> Data | None:
    g = json.loads(path.read_text())
    claims = g["nodes"]["claims"]
    triples = g["nodes"]["triples"]
    if not claims:
        return None

    idx, x = {}, []
    for c in claims:
        idx[c["node_id"]] = len(x)
        p = c.get("p_yes")
        w = c.get("stance_weight") or 0.0
        known = 0.0 if p is None else 1.0
        p = 0.5 if p is None else p
        lab = {"MALIGNANT": 1.0, "BENIGN": -1.0}.get(c.get("label"), 0.0)
        x.append([p, w, p * w, known,
                  c.get("round_idx", 0) / 3.0,
                  1.0 if c.get("expert_id") == "expert_a" else 0.0,
                  1.0 if c.get("label_explicit") else 0.0,
                  lab])
    n_claim = len(x)
    for t in triples:                       # triple nodes carry no measurement
        idx[t["node_id"]] = len(x)
        x.append([0.0] * _FEATS_PER_CLAIM)

    src, dst, eattr = [], [], []
    for e in g["edges"]["claim_claim"]:
        if e["src"] in idx and e["dst"] in idx:
            s, d = idx[e["src"]], idx[e["dst"]]
            sign = 1.0 if e.get("type") == "AGREE" else -1.0
            src += [s, d]; dst += [d, s]; eattr += [sign, sign]
    for e in g["edges"]["claim_triple"]:
        if e["src"] in idx and e["dst"] in idx:
            s, d = idx[e["src"]], idx[e["dst"]]
            src += [s, d]; dst += [d, s]; eattr += [0.0, 0.0]
    if not src:                              # keep isolated graphs usable
        src, dst, eattr = [0], [0], [0.0]

    # Graph-level probe vector, for the no-graph baseline.
    ev = {e["feature"]: e for e in (g.get("evidence") or [])}
    vec = [0.5] * len(feat_index)
    for f, i in feat_index.items():
        if f in ev and ev[f]["p_yes"] is not None:
            vec[i] = ev[f]["p_yes"]

    d = Data(x=torch.tensor(x, dtype=torch.float),
             edge_index=torch.tensor([src, dst], dtype=torch.long),
             edge_attr=torch.tensor(eattr, dtype=torch.float).view(-1, 1),
             y=torch.tensor([1 if g["gold_label"] == "MALIGNANT" else 0]))
    d.probe = torch.tensor([vec], dtype=torch.float)
    d.n_claim = n_claim
    return d


class GNN(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, hid)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, d):
        h = F.relu(self.c1(d.x, d.edge_index))
        h = F.dropout(h, 0.3, self.training)
        h = F.relu(self.c2(h, d.edge_index))
        g = torch.cat([global_mean_pool(h, d.batch),
                       global_max_pool(h, d.batch)], dim=1)
        return self.head(g).squeeze(-1)


class Hybrid(nn.Module):
    """GraphSAGE readout concatenated with the full probe vector.

    The decisive test. The graph only contains measurements the agents chose to
    cite, so it loses information relative to the raw probe vector. If structure
    carries anything the measurements do not, this beats probe-mlp; if it ties,
    the debate is adding nothing; if it loses, the structure is actively noise.
    """
    def __init__(self, in_dim, probe_dim, hid=64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, hid)
        self.head = nn.Sequential(
            nn.Linear(2 * hid + probe_dim, hid), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, d):
        h = F.relu(self.c1(d.x, d.edge_index))
        h = F.dropout(h, 0.3, self.training)
        h = F.relu(self.c2(h, d.edge_index))
        g = torch.cat([global_mean_pool(h, d.batch),
                       global_max_pool(h, d.batch), d.probe], dim=1)
        return self.head(g).squeeze(-1)


class MeanPool(nn.Module):
    """Node features averaged, no message passing."""
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, d):
        return self.net(global_mean_pool(d.x, d.batch)).squeeze(-1)


class ProbeMLP(nn.Module):
    """Probe vector only — the graph is never touched."""
    def __init__(self, in_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, d):
        return self.net(d.probe).squeeze(-1)


def run(model, tr, va, te, dev, epochs, pos_weight, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))

    def evaluate(loader):
        model.eval()
        s, y = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                s += model(b).cpu().tolist()
                y += b.y.cpu().tolist()
        return s, y

    best_va, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for b in tr:
            b = b.to(dev)
            opt.zero_grad()
            loss = lossf(model(b), b.y.float())
            loss.backward()
            opt.step()
        s, y = evaluate(va)
        a = auc([x for x, t in zip(s, y) if t == 1], [x for x, t in zip(s, y) if t == 0])
        if a > best_va:
            best_va = a
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)          # early stop on val AUC
    s, y = evaluate(te)
    a = auc([x for x, t in zip(s, y) if t == 1], [x for x, t in zip(s, y) if t == 0])
    ba, th = best_balanced_acc(s, y)
    sv, yv = evaluate(va)
    _, th_va = best_balanced_acc(sv, yv)       # threshold chosen on val, not test
    ba_honest = balanced_acc(s, y, th_va)
    return a, ba, ba_honest, best_va


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    root = Path(args.dataset)
    # Stable probe ordering across all graphs.
    feats = set()
    for f in (root / "train/graphs").glob("*.json"):
        for e in json.loads(f.read_text()).get("evidence") or []:
            feats.add(e["feature"])
    feat_index = {f: i for i, f in enumerate(sorted(feats))}

    splits = {}
    for sp in ("train", "val", "test"):
        ds = [load_graph(f, feat_index) for f in sorted((root / sp / "graphs").glob("*.json"))]
        splits[sp] = [d for d in ds if d is not None]
        print(f"{sp}: {len(splits[sp])} graphs")

    npos = sum(int(d.y.item()) for d in splits["train"])
    pos_weight = (len(splits["train"]) - npos) / max(1, npos)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"features={len(feat_index)}  pos_weight={pos_weight:.2f}  device={dev}\n")

    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
               for k, v in splits.items()}

    models = {
        "probe-mlp (no graph)": lambda: ProbeMLP(len(feat_index)),
        "mean-pool (nodes, no edges)": lambda: MeanPool(_FEATS_PER_CLAIM),
        "GraphSAGE (nodes + edges)": lambda: GNN(_FEATS_PER_CLAIM),
        "hybrid (graph + probes)": lambda: Hybrid(_FEATS_PER_CLAIM, len(feat_index)),
    }

    print(f"{'model':30s} {'test AUC':>18s} {'bAcc@best':>18s} {'bAcc@val-th':>18s}")
    print("-" * 88)
    for name, ctor in models.items():
        rs = [run(ctor(), loaders["train"], loaders["val"], loaders["test"],
                  dev, args.epochs, pos_weight, seed=s) for s in range(args.seeds)]
        for j, lab in ((0, "test AUC"), (1, "bAcc@best"), (2, "bAcc@val-th")):
            pass
        m = lambda j: sum(r[j] for r in rs) / len(rs)
        sd = lambda j: (sum((r[j] - m(j)) ** 2 for r in rs) / len(rs)) ** 0.5
        print(f"{name:30s} {m(0):9.4f}+-{sd(0):.3f} {m(1):9.4f}+-{sd(1):.3f} "
              f"{m(2):9.4f}+-{sd(2):.3f}")

    print(f"\nReference points on the same test split:")
    print(f"  zero-shot no KG          AUC 0.6013   bAcc 0.5883 (own th) / 0.6284 (best)")
    print(f"  unsupervised probe score AUC 0.6685   bAcc 0.5614 (own th) / 0.6497 (best)")
    print(f"  evidence_score readout   AUC 0.6733   (graph-level, untrained)")
    print("\nbAcc@val-th is the honest number: threshold picked on val, applied to test.")


if __name__ == "__main__":
    main()
