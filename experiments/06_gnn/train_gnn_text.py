"""Train a GNN on text-attributed debate graphs (v3) and compare fairly.

v2 graphs carry a measured probe value on every claim, so a GNN over them was
competing against the probe vector directly — and lost, because the debate only
cites a biased subset of the measurements.

v3 graphs have no measurements by construction: the agents never saw the KG.
Node features therefore come from sentence embeddings of the claim text, and KG
triples are attached afterwards by retrieval. The question this script answers
is whether that produces something a GNN can actually learn from.

Models, all on the same split:

  text-mean    claim embeddings averaged, no edges. How much is in the words
               alone, before any structure.
  gnn          GraphSAGE over claim-claim (AGREE/DISAGREE) and claim-triple
               (retrieval) edges. Beating text-mean means the structure helps.
  gnn+triples  same, but KG triple nodes carry their own text embedding rather
               than a zero vector. Isolates what the knowledge graph adds.

Embeddings are computed here from node text rather than stored in the graphs,
which keeps the dataset small and the encoder swappable.

Usage:
  python train_gnn_text.py --dataset .../dataset_v3
  python train_gnn_text.py --dataset .../dataset_v3 --compare-with .../dataset_v2
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

_META = 5          # label, expert, round, top_sim, n_links


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def balanced_acc(scores, labels, th):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return 0.5 * (sum(1 for s in pos if s >= th) / len(pos) +
                  sum(1 for s in neg if s < th) / len(neg))


def best_balanced_acc(scores, labels):
    return max((balanced_acc(scores, labels, t), t) for t in sorted(set(scores)))


def build(root: Path, split: str, enc, use_triple_text: bool) -> list[Data]:
    files = sorted((root / split / "graphs").glob("*.json"))
    graphs = [json.loads(f.read_text()) for f in files]

    # Encode every distinct string once across the whole split.
    texts = []
    for g in graphs:
        texts += [c["text"] for c in g["nodes"]["claims"]]
        if use_triple_text:
            texts += [t.get("text", "") for t in g["nodes"]["triples"]]
    uniq = sorted(set(t for t in texts if t))
    emb = enc.encode(uniq, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=256)
    lookup = {t: i for i, t in enumerate(uniq)}
    dim = emb.shape[1]

    out = []
    for g in graphs:
        claims, triples = g["nodes"]["claims"], g["nodes"]["triples"]
        if not claims:
            continue
        idx, rows = {}, []
        for c in claims:
            idx[c["node_id"]] = len(rows)
            e = emb[lookup[c["text"]]] if c["text"] in lookup else [0.0] * dim
            meta = [
                {"MALIGNANT": 1.0, "BENIGN": -1.0}.get(c.get("label"), 0.0),
                1.0 if c.get("expert_id") == "expert_a" else 0.0,
                c.get("round_idx", 0) / 3.0,
                float(c.get("top_sim", 0.0)),
                float(c.get("n_links", 0)) / 4.0,
            ]
            rows.append(list(e) + meta + [1.0])          # trailing flag: is-claim
        for t in triples:
            idx[t["node_id"]] = len(rows)
            if use_triple_text and t.get("text") in lookup:
                e = emb[lookup[t["text"]]]
            else:
                e = [0.0] * dim
            rows.append(list(e) + [0.0] * _META + [0.0])

        src, dst = [], []
        for e in g["edges"]["claim_claim"] + g["edges"]["claim_triple"]:
            if e["src"] in idx and e["dst"] in idx:
                a, b = idx[e["src"]], idx[e["dst"]]
                src += [a, b]; dst += [b, a]
        if not src:
            src, dst = [0], [0]

        out.append(Data(
            x=torch.tensor(rows, dtype=torch.float),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            y=torch.tensor([1 if g["gold_label"] == "MALIGNANT" else 0])))
    return out


class TextMean(nn.Module):
    """Node features averaged; edges ignored."""
    def __init__(self, dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                 nn.Dropout(0.4), nn.Linear(hid, 1))

    def forward(self, d):
        return self.net(global_mean_pool(d.x, d.batch)).squeeze(-1)


class GNN(nn.Module):
    def __init__(self, dim, hid=64):
        super().__init__()
        self.proj = nn.Linear(dim, hid)
        self.c1, self.c2 = SAGEConv(hid, hid), SAGEConv(hid, hid)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(hid, 1))

    def forward(self, d):
        h = F.relu(self.proj(d.x))
        h = F.dropout(F.relu(self.c1(h, d.edge_index)), 0.4, self.training)
        h = F.relu(self.c2(h, d.edge_index))
        g = torch.cat([global_mean_pool(h, d.batch),
                       global_max_pool(h, d.batch)], dim=1)
        return self.head(g).squeeze(-1)


def run(model, tr, va, te, dev, epochs, pw, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))

    def ev(loader):
        model.eval(); s, y = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                s += model(b).cpu().tolist(); y += b.y.cpu().tolist()
        return s, y

    best, state = -1, None
    for _ in range(epochs):
        model.train()
        for b in tr:
            b = b.to(dev)
            opt.zero_grad(); lossf(model(b), b.y.float()).backward(); opt.step()
        s, y = ev(va)
        a = auc([x for x, t in zip(s, y) if t == 1], [x for x, t in zip(s, y) if t == 0])
        if a > best:
            best, state = a, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(state)
    s, y = ev(te)
    a = auc([x for x, t in zip(s, y) if t == 1], [x for x, t in zip(s, y) if t == 0])
    ba, _ = best_balanced_acc(s, y)
    sv, yv = ev(va)
    _, th = best_balanced_acc(sv, yv)      # threshold from val, applied to test
    return a, ba, balanced_acc(s, y, th)


def evaluate_dataset(root: Path, enc, args, dev, label: str):
    print(f"\n{'='*90}\n{label}: {root}\n{'='*90}")
    results = {}
    for use_tt in (False, True):
        splits = {sp: build(root, sp, enc, use_tt) for sp in ("train", "val", "test")}
        if not splits["train"] or not splits["test"]:
            print("  missing splits"); return
        dim = splits["train"][0].x.shape[1]
        npos = sum(int(d.y.item()) for d in splits["train"])
        pw = (len(splits["train"]) - npos) / max(1, npos)
        ld = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
              for k, v in splits.items()}
        if not use_tt:
            print(f"  train={len(splits['train'])} val={len(splits['val'])} "
                  f"test={len(splits['test'])}  node_dim={dim}  pos_weight={pw:.2f}\n")
            print(f"  {'model':28s} {'test AUC':>17s} {'bAcc@best':>17s} {'bAcc@val-th':>17s}")
            print("  " + "-" * 84)
            models = {"text-mean (no edges)": lambda: TextMean(dim),
                      "GraphSAGE (claims only)": lambda: GNN(dim)}
        else:
            models = {"GraphSAGE + KG triple text": lambda: GNN(dim)}
        for name, ctor in models.items():
            rs = [run(ctor(), ld["train"], ld["val"], ld["test"], dev,
                      args.epochs, pw, seed=s) for s in range(args.seeds)]
            m = lambda j: sum(r[j] for r in rs) / len(rs)
            sd = lambda j: (sum((r[j] - m(j)) ** 2 for r in rs) / len(rs)) ** 0.5
            print(f"  {name:28s} {m(0):8.4f}+-{sd(0):.3f} {m(1):8.4f}+-{sd(1):.3f} "
                  f"{m(2):8.4f}+-{sd(2):.3f}")
            results[name] = m(0)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--compare-with", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--encoder", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_dataset(Path(args.dataset), enc, args, dev, "v3 (KG-free debate)")
    if args.compare_with:
        evaluate_dataset(Path(args.compare_with), enc, args, dev,
                         "v2 (KG-conditioned debate), same text features")

    print("\nReference points, same test split:")
    print("  zero-shot no KG            AUC 0.6013   bAcc 0.5883 (own th)")
    print("  probe vector MLP, no graph AUC 0.6985   bAcc 0.6305 (val th)")
    print("  v2 GraphSAGE               AUC 0.6037   bAcc 0.5654 (val th)")
    print("  v2 hybrid (graph + probes) AUC 0.6772   bAcc 0.6323 (val th)")


if __name__ == "__main__":
    main()
