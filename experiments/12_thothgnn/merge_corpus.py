"""Section 5, E2: merge two independent debate runs into one graph per sample.

The single-run anchor is mal_share = 0.7128 on test; averaging two runs gives
0.7416, the best score in the project. E2 makes that gain available to the GNN
rather than sitting outside it: claims from both runs live in one graph, with no
claim->claim edges across runs, but sharing triple nodes so the knowledge graph
is the bridge between the two debates.

The merged mal_share is the pooled ratio over all claims from both runs, which
is what the BASE configuration will now anchor on. So the merged BASE should
land at the ensemble number instead of the single-run one, and everything the
GNN learns is measured on top of the stronger baseline.

Usage:
  python merge_corpus.py --corpora corpus/v5q.pt corpus/v5q_r1.pt --out corpus/v5qE2.pt
"""
from __future__ import annotations

import argparse, logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge(gs: list[dict]) -> dict:
    """Concatenate claim sets; re-index edges; union triples by global id."""
    x_claim, agree, dis, nxt, sign, cites = [], [], [], [], [], []
    tri_order, tri_feat, tri_rel = [], {}, {}
    coff = 0
    for g in gs:
        n = len(g["x_claim"])
        x_claim += g["x_claim"]
        sign += g["class_sign"]
        agree += [(a + coff, b + coff) for a, b in g["agree"]]
        dis += [(a + coff, b + coff) for a, b in g["disagree"]]
        nxt += [(a + coff, b + coff) for a, b in g["next"]]
        # Triples are shared across runs, so index them by their global KG id.
        for local, gid in enumerate(g["triple_gid"]):
            if gid not in tri_feat:
                tri_feat[gid] = g["x_triple"][local]
                tri_rel[gid] = g["rel_triple"][local]
                tri_order.append(gid)
        pos = {gid: i for i, gid in enumerate(tri_order)}
        cites += [(c + coff, pos[g["triple_gid"][t]], s) for c, t, s in g["cites"]]
        coff += n

    mal = sum(1 for r in x_claim if r[0] == 1.0)
    return {
        "sample_id": gs[0]["sample_id"], "split": gs[0]["split"], "y": gs[0]["y"],
        "mal_share": mal / len(x_claim),
        "x_claim": x_claim, "class_sign": sign,
        "x_triple": [tri_feat[g] for g in tri_order],
        "rel_triple": [tri_rel[g] for g in tri_order],
        "triple_gid": tri_order,
        "agree": agree, "disagree": dis, "next": nxt, "cites": cites,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    blobs = [torch.load(c, weights_only=False) for c in args.corpora]
    idx = [{(g["split"], g["sample_id"]): g for g in b["graphs"]} for b in blobs]
    keys = sorted(set.intersection(*[set(i) for i in idx]))
    logger.info("%d corpora, %d samples present in all of them", len(blobs), len(keys))

    graphs = [merge([i[k] for i in idx]) for k in keys]
    meta = dict(blobs[0]["meta"])
    meta["merged_from"] = args.corpora
    torch.save({"graphs": graphs, "meta": meta}, args.out)

    def auc(m, b):
        return sum((x > y) + 0.5 * (x == y) for x in m for y in b) / (len(m) * len(b))
    for sp in ("train", "val", "test"):
        g = [x for x in graphs if x["split"] == sp]
        m = [x["mal_share"] for x in g if x["y"] == 1]
        b = [x["mal_share"] for x in g if x["y"] == 0]
        logger.info("  merged mal_share %-5s AUC %.4f  (n=%d)", sp, auc(m, b), len(g))
    logger.info("wrote %d graphs -> %s | mean claims %.1f triples %.1f cites %.1f",
                len(graphs), args.out,
                sum(len(g["x_claim"]) for g in graphs) / len(graphs),
                sum(len(g["x_triple"]) for g in graphs) / len(graphs),
                sum(len(g["cites"]) for g in graphs) / len(graphs))


if __name__ == "__main__":
    main()
