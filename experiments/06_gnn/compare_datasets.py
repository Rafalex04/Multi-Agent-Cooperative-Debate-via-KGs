"""Train GraphSAGE on several debate-graph datasets and compare how they learn.

Final AUC alone hides the thing that matters when judging a dataset: whether
the model is learning anything from the graph or just fitting the class prior.
Two datasets can land on the same test AUC while one climbs steadily on
validation and the other memorises the training set from epoch five. So this
records the whole curve — train and validation AUC at every epoch, for every
seed — and reports the shape alongside the endpoint.

Metrics per dataset:

  test AUC / bAcc      endpoint, with the threshold chosen on validation
  best val AUC         how far it got before overfitting
  epoch of best val    how quickly; early peaks mean little signal to exploit
  train-val gap        overfitting, measured at the best-val epoch
  val AUC slope        early learning rate over the first 15 epochs
  curve                per-epoch means, written to JSON for plotting

Node features are sentence embeddings of the claim text plus label, expert,
round and citation count, so datasets with and without measurements can be
compared on equal terms.

Usage:
  python compare_datasets.py --datasets v2=.../dataset_v2 v3=.../dataset_v3 \
      v4=.../dataset_v4 --out curves.json
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

_META = 10     # label, expert, round, n_cited, against_side, is_repeat,
               # p_yes, p_yes*stance_weight, measured, is_claim


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def split_auc(scores, labels):
    return auc([s for s, y in zip(scores, labels) if y == 1],
               [s for s, y in zip(scores, labels) if y == 0])


def bacc(scores, labels, th):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return 0.5 * (sum(1 for s in pos if s >= th) / len(pos) +
                  sum(1 for s in neg if s < th) / len(neg))


def _num(v, default):
    """v2 stores p_yes as null when a probe failed; treat that as unmeasured."""
    return default if v is None else float(v)


def build(root: Path, split: str, enc):
    files = sorted((root / split / "graphs").glob("*.json"))
    graphs = [json.loads(f.read_text()) for f in files]
    texts = sorted({c["text"] for g in graphs for c in g["nodes"]["claims"]} |
                   {t.get("text", "") for g in graphs for t in g["nodes"]["triples"]})
    texts = [t for t in texts if t]
    emb = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=256)
    look = {t: i for i, t in enumerate(texts)}
    dim = emb.shape[1]

    out = []
    for g in graphs:
        claims, triples = g["nodes"]["claims"], g["nodes"]["triples"]
        if not claims:
            continue
        idx, rows = {}, []
        for c in claims:
            idx[c["node_id"]] = len(rows)
            e = emb[look[c["text"]]] if c["text"] in look else [0.0] * dim
            rows.append(list(e) + [
                {"MALIGNANT": 1.0, "BENIGN": -1.0}.get(c.get("label"), 0.0),
                1.0 if c.get("expert_id") == "expert_a" else 0.0,
                c.get("round_idx", 0) / 3.0,
                float(c.get("n_cited", c.get("n_links", 0))) / 4.0,
                1.0 if c.get("against_side") else 0.0,
                1.0 if c.get("is_repeat") else 0.0,
                # The measured probe value for the finding this claim cites.
                # Omitting these was the flaw in the first comparison: v2 had
                # them and the trainer ignored them, so a GNN over its graph
                # scored 0.6027 against 0.6535 for a plain average of the same
                # numbers. `measured` distinguishes a real 0.5 from a missing one.
                _num(c.get("p_yes"), 0.5),
                _num(c.get("p_yes"), 0.5) * _num(c.get("stance_weight"), 0.0),
                1.0 if c.get("measured") else 0.0,
                1.0,
            ])
        for t in triples:
            idx[t["node_id"]] = len(rows)
            e = emb[look[t["text"]]] if t.get("text") in look else [0.0] * dim
            rows.append(list(e) + [0.0] * _META)

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
        return self.head(torch.cat([global_mean_pool(h, d.batch),
                                    global_max_pool(h, d.batch)], 1)).squeeze(-1)


def run_seed(loaders, dim, dev, epochs, pw, seed):
    torch.manual_seed(seed)
    model = GNN(dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))

    def ev(loader):
        model.eval(); s, y = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                s += model(b).cpu().tolist(); y += b.y.cpu().tolist()
        return s, y

    curve = {"train": [], "val": [], "loss": []}
    best = (-1, None, 0)
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in loaders["train"]:
            b = b.to(dev)
            opt.zero_grad()
            l = lossf(model(b), b.y.float())
            l.backward(); opt.step(); tot += l.item()
        strn, ytrn = ev(loaders["train"]); sval, yval = ev(loaders["val"])
        a_tr, a_va = split_auc(strn, ytrn), split_auc(sval, yval)
        curve["train"].append(a_tr); curve["val"].append(a_va)
        curve["loss"].append(tot / max(1, len(loaders["train"])))
        if a_va > best[0]:
            best = (a_va, {k: v.clone() for k, v in model.state_dict().items()}, ep)

    model.load_state_dict(best[1])
    ste, yte = ev(loaders["test"]); sva, yva = ev(loaders["val"])
    th = max((bacc(sva, yva, t), t) for t in sorted(set(sva)))[1]
    return {
        "test_auc": split_auc(ste, yte),
        "test_bacc": bacc(ste, yte, th),
        "best_val_auc": best[0],
        "best_epoch": best[2],
        "train_auc_at_best": curve["train"][best[2]],
        "gap_at_best": curve["train"][best[2]] - best[0],
        "val_slope_15": (curve["val"][min(14, len(curve["val"]) - 1)] - curve["val"][0]),
        "curve": curve,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, help="name=path ...")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--out", default="curves.json")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for spec in args.datasets:
        name, path = spec.split("=", 1)
        root = Path(path)
        splits = {s: build(root, s, enc) for s in ("train", "val", "test")}
        if not splits["train"] or not splits["test"]:
            print(f"{name}: missing splits, skipping")
            continue
        dim = splits["train"][0].x.shape[1]
        npos = sum(int(d.y.item()) for d in splits["train"])
        pw = (len(splits["train"]) - npos) / max(1, npos)
        loaders = {k: DataLoader(v, batch_size=args.batch_size,
                                 shuffle=(k == "train")) for k, v in splits.items()}
        rs = [run_seed(loaders, dim, dev, args.epochs, pw, s) for s in range(args.seeds)]
        results[name] = {
            "n_train": len(splits["train"]), "n_test": len(splits["test"]),
            "seeds": [{k: v for k, v in r.items() if k != "curve"} for r in rs],
            "curve_mean": {
                k: [st.mean(r["curve"][k][e] for r in rs)
                    for e in range(args.epochs)] for k in ("train", "val", "loss")},
        }
        m = lambda k: st.mean(r[k] for r in rs)
        sd = lambda k: st.pstdev([r[k] for r in rs])
        print(f"\n{name}  (train={len(splits['train'])} test={len(splits['test'])})")
        print(f"  test AUC        {m('test_auc'):.4f} +- {sd('test_auc'):.3f}")
        print(f"  test bAcc       {m('test_bacc'):.4f} +- {sd('test_bacc'):.3f}")
        print(f"  best val AUC    {m('best_val_auc'):.4f}  at epoch {m('best_epoch'):.1f}")
        print(f"  train AUC there {m('train_auc_at_best'):.4f}  "
              f"(gap {m('gap_at_best'):+.4f})")
        print(f"  val gain by ep15 {m('val_slope_15'):+.4f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\ncurves -> {args.out}")

    if len(results) > 1:
        print(f"\n{'='*78}\n{'dataset':10s} {'test AUC':>10s} {'test bAcc':>10s} "
              f"{'best val':>10s} {'epoch':>7s} {'gap':>8s} {'ep15 gain':>10s}")
        for n, r in results.items():
            m = lambda k: st.mean(s[k] for s in r["seeds"])
            print(f"{n:10s} {m('test_auc'):10.4f} {m('test_bacc'):10.4f} "
                  f"{m('best_val_auc'):10.4f} {m('best_epoch'):7.1f} "
                  f"{m('gap_at_best'):+8.4f} {m('val_slope_15'):+10.4f}")


if __name__ == "__main__":
    main()
