"""Section 10: is the attention an explanation, or decoration?

Two measurements, not assertions.

FAITHFULNESS. Delete the top-attention claim, re-score, and record the shift in
p_mal. Compare against deleting a random claim from the same graph. If the two
shifts are the same size, the weights carry no information about what the model
actually used.

DIVERSITY. Distribution of attention mass over claims and triples across the
test set. If a handful of triples dominate every explanation, the
interpretability claim fails regardless of faithfulness.

Usage:
  python interpret.py --corpus corpus/v5qE2.pt --config FULL --seed 0
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thoth_gnn import (CONFIGS, DEFAULTS, ThothGNN, auc, collate, run_epochs,  # noqa
                       set_seed, to_hetero)
from torch_geometric.utils import softmax as pyg_softmax


@torch.no_grad()
def attention(model, batch):
    """Re-run the encoder and return per-claim attention for the MAL class."""
    cfg = model.cfg
    h = {"claim": model.claim_in(batch["claim"].x),
         "triple": model.triple_in(torch.cat(
             [batch["triple"].x, model.rel_emb(batch["triple"].rel)], -1))}
    hs = [h["claim"]]
    from thoth_gnn import REL_BAL, REL_CC, REL_KG, REL_NX
    for l in range(model.L):
        ew = {}
        for t, r in enumerate(REL_CC + REL_NX + REL_BAL):
            ei = batch[r].edge_index
            ew[r] = (model.gate[l](h["claim"], ei, t) if model.gate is not None
                     else torch.ones(ei.size(1), device=ei.device))
        for r in REL_KG:
            ew[r] = batch[r].edge_weight
        agg = model.convs[l](h, batch.edge_index_dict, edge_weight_dict=ew)
        a = agg.get("claim", torch.zeros_like(h["claim"]))
        new = model.norm[l](torch.relu(model.self_lin[l](h["claim"]) + a))
        h = {"claim": new, "triple": agg.get("triple", h["triple"])}
        hs.append(new)
    hf = model.jk(torch.cat(hs, -1) if cfg["xlayer"] else hs[-1])
    e = model.class_emb.weight[1].expand(hf.size(0), -1)
    return pyg_softmax(model.att(torch.cat([hf, e], -1)), batch["claim"].batch).squeeze(-1)


def drop_claim(g: dict, i: int) -> dict:
    """Remove claim i and every edge touching it, re-indexing the rest."""
    keep = [j for j in range(len(g["x_claim"])) if j != i]
    remap = {j: k for k, j in enumerate(keep)}
    f = lambda es: [(remap[a], remap[b]) for a, b in es if a in remap and b in remap]
    out = dict(g)
    out["x_claim"] = [g["x_claim"][j] for j in keep]
    out["class_sign"] = [g["class_sign"][j] for j in keep]
    out["agree"], out["disagree"] = f(g["agree"]), f(g["disagree"])
    out["next"] = f(g["next"])
    out["cites"] = [(remap[a], b, s) for a, b, s in g["cites"] if a in remap]
    mal = sum(1 for r in out["x_claim"] if r[0] == 1.0)
    out["mal_share"] = mal / max(1, len(out["x_claim"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", default="FULL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.corpus, weights_only=False)
    graphs, meta = blob["graphs"], blob["meta"]
    cfg = {**DEFAULTS, **CONFIGS[args.config]}
    tr = [g for g in graphs if g["split"] == "train"]
    te = [g for g in graphs if g["split"] == "test"]
    pos_w = sum(1 for g in tr if g["y"] == 0) / max(1, sum(1 for g in tr if g["y"] == 1))

    set_seed(args.seed)
    m = ThothGNN(cfg, meta["n_relations"]).to(dev)
    run_epochs(m, collate([to_hetero(g, cfg) for g in tr], dev, 128, True), [],
               args.epochs, 1e-3, 5e-4, 0.0, pos_w, dev)
    m.eval()

    # ---------------- faithfulness ----------------
    rng = random.Random(0)
    top_shift, rnd_shift, tops, rel_mass = [], [], [], Counter()
    for g in te:
        if len(g["x_claim"]) < 4:
            continue
        b = collate([to_hetero(g, cfg)], dev, 1)[0]
        with torch.no_grad():
            p0 = torch.sigmoid(m(b)).item()
        al = attention(m, b).cpu().numpy()
        i_top = int(np.argmax(al))
        i_rnd = rng.choice([j for j in range(len(al)) if j != i_top])
        for i, bucket in ((i_top, top_shift), (i_rnd, rnd_shift)):
            bb = collate([to_hetero(drop_claim(g, i), cfg)], dev, 1)[0]
            with torch.no_grad():
                bucket.append(abs(torch.sigmoid(m(bb)).item() - p0))
        tops.append(i_top / len(al))
        # attention mass reaching each KG triple through its citing claims
        for c, t, s in g["cites"]:
            if c < len(al):
                rel_mass[g["triple_gid"][t]] += float(al[c]) * s

    t_mean, r_mean = float(np.mean(top_shift)), float(np.mean(rnd_shift))
    wins = sum(1 for a, b in zip(top_shift, rnd_shift) if a > b) / len(top_shift)
    print(f"\n=== faithfulness ({len(top_shift)} test graphs) ===")
    print(f"  drop top-attention claim   mean |dp| {t_mean:.4f}")
    print(f"  drop random claim          mean |dp| {r_mean:.4f}")
    print(f"  ratio {t_mean / max(1e-9, r_mean):.2f}x   top-beats-random on "
          f"{100*wins:.0f}% of graphs")
    # The mean ratio alone is not enough: a handful of large shifts can carry it
    # while the top claim loses to a random one on most graphs. Faithfulness
    # requires the per-graph comparison to hold too.
    if t_mean > r_mean * 1.05 and wins > 0.5:
        print("  -> attention identifies claims the model actually uses")
    elif t_mean > r_mean * 1.05:
        print(f"  -> WEAK: mean is carried by a few graphs; the top claim loses "
              f"to a random one on {100*(1-wins):.0f}% of them")
    else:
        print("  -> attention is decoration")

    # ---------------- diversity ----------------
    tot = sum(rel_mass.values())
    top10 = sum(v for _, v in rel_mass.most_common(10)) / max(1e-9, tot)
    print(f"\n=== diversity ===")
    print(f"  distinct triples receiving attention: {len(rel_mass)}")
    print(f"  share of all attention mass on the top 10 triples: {top10:.3f}")
    print(f"  mean normalised position of the top claim: {np.mean(tops):.3f} "
          f"(0.5 = no positional bias)")
    print("  -> a few triples dominate every explanation" if top10 > 0.5
          else "  -> attention spreads across the knowledge graph")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"top_shift": t_mean, "rnd_shift": r_mean, "ratio": t_mean / max(1e-9, r_mean),
             "win_rate": wins, "distinct_triples": len(rel_mass),
             "top10_mass": top10, "mean_top_pos": float(np.mean(tops))}, indent=1))


if __name__ == "__main__":
    main()
