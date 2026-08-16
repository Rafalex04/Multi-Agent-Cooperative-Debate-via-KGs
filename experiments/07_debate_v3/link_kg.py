"""Attach KG triples to a finished KG-free debate, producing trainable graphs.

The v3 debates were produced without the model ever seeing the knowledge graph.
This links them afterwards: each claim is embedded, compared against every
verbalised triple, and connected to its top-k matches.

The KG therefore supplies structure over what the debate actually said, instead
of dictating the vocabulary the debate is allowed to use — which is what made
the v2 graphs a lossy compression of their own measurements.

Node features are text embeddings (claims and triples alike) plus the claim's
label, expert, round and retrieval score. Emits the same graph JSON schema as
dataset_full and dataset_v2, so the GNN trainer reads it unchanged.

Usage:
  python link_kg.py --debates .../debates_v3 --out-dir .../dataset_v3 --k 4
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "breastMnist" / "src"))

# Structural relations describe the ontology, not a finding, so linking a claim
# to them adds edges without adding evidence.
_TAXONOMY = {"is_a", "subtype_of", "birads_subcategory_of", "category_descriptor",
             "also_known_as"}


def verbalise(t: dict) -> str:
    s = t.get("subject", "").replace("_", " ")
    r = t.get("relation", "").replace("_", " ")
    o = str(t.get("object", "")).replace("_", " ")
    if o in ("true", ""):
        return f"{s} {r}".strip()
    return f"{s} {r} {o}".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debates", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=4, help="triples linked per claim")
    p.add_argument("--min-sim", type=float, default=0.20,
                   help="drop links below this cosine similarity")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--kg-root", default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    import numpy as np

    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    triples = [t for t in triples if t.get("relation") not in _TAXONOMY]
    texts = [verbalise(t) for t in triples]

    enc = SentenceTransformer(args.model)
    T = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info("encoded %d non-taxonomy triples", len(triples))

    dbg_root = Path(args.debates)
    out_root = Path(args.out_dir)
    n_done = 0

    for split_dir in sorted(d for d in dbg_root.iterdir() if d.is_dir()):
        files = sorted(split_dir.glob("debate_*.json"))
        if not files:
            continue
        gdir = out_root / split_dir.name / "graphs"
        gdir.mkdir(parents=True, exist_ok=True)

        for f in files:
            d = json.loads(f.read_text())
            claims = d["claims"]
            if not claims:
                continue
            C = enc.encode([c["text"] for c in claims],
                           normalize_embeddings=True, show_progress_bar=False)
            sim = C @ T.T                                   # cosine, both unit norm

            claim_nodes, used, ct_edges = [], {}, []
            for i, c in enumerate(claims):
                order = np.argsort(-sim[i])[:args.k]
                links = [(int(j), float(sim[i][j])) for j in order
                         if sim[i][j] >= args.min_sim]
                for rank, (j, s) in enumerate(links):
                    used[triples[j]["id"]] = triples[j]
                    ct_edges.append({"src": c["node_id"], "dst": triples[j]["id"],
                                     "rank": rank, "score": round(s, 4),
                                     "type": "retrieval"})
                claim_nodes.append({
                    "node_id": c["node_id"], "type": "claim", "text": c["text"],
                    "label": c["label"],
                    "label_int": {"MALIGNANT": 1, "BENIGN": 0}.get(c["label"], -1),
                    "label_explicit": c["label_explicit"],
                    "expert_id": c["expert_id"], "round_idx": c["round_idx"],
                    "top_sim": round(float(links[0][1]), 4) if links else 0.0,
                    "n_links": len(links),
                })

            triple_nodes = []
            for tid, t in used.items():
                k = texts[[x["id"] for x in triples].index(tid)]
                triple_nodes.append({
                    "node_id": tid, "type": "triple",
                    "subject": t.get("subject", ""), "predicate": t.get("relation", ""),
                    "object": t.get("object", ""), "text": k,
                })

            by_id = {c["node_id"] for c in claim_nodes}
            lab = {c["node_id"]: c["label"] for c in claim_nodes}
            cc, seen = [], set()
            for c in claims:
                for tgt in set(c["addressed_ids"]):
                    if tgt not in by_id or tgt == c["node_id"] or \
                       (c["node_id"], tgt) in seen:
                        continue
                    seen.add((c["node_id"], tgt))
                    same = lab[c["node_id"]] and lab[c["node_id"]] == lab[tgt]
                    cc.append({"src": c["node_id"], "dst": tgt,
                               "sign": "+" if same else "-",
                               "type": "AGREE" if same else "DISAGREE"})

            tt, ids = [], list(used)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    ta, tb = used[ids[a]], used[ids[b]]
                    if ta.get("subject") == tb.get("subject") or \
                       ta.get("object") == tb.get("object"):
                        tt.append({"src": ids[a], "dst": ids[b],
                                   "type": "shared_entity"})

            labs = [c["label"] for c in claim_nodes
                    if c["label"] in ("MALIGNANT", "BENIGN")]
            share = sum(1 for l in labs if l == "MALIGNANT") / len(labs) if labs else 0.5
            verdict = "MALIGNANT" if share > 0.5 else "BENIGN"

            (gdir / f"sample_{d['sample_id']}.json").write_text(json.dumps({
                "sample_id": d["sample_id"], "gold_label": d["gold_label"],
                "gold_label_int": 1 if d["gold_label"] == "MALIGNANT" else 0,
                "verdict": verdict, "correct": verdict == d["gold_label"],
                "rounds_used": d.get("rounds_used", 3),
                "debate_mal_share": share,
                "nodes": {"claims": claim_nodes, "triples": triple_nodes},
                "edges": {"claim_claim": cc, "claim_triple": ct_edges,
                          "triple_triple": tt},
                "stats": {
                    "num_claims": len(claim_nodes), "num_triples": len(triple_nodes),
                    "num_claim_claim_edges": len(cc),
                    "num_claim_triple_edges": len(ct_edges),
                    "num_triple_triple_edges": len(tt),
                },
            }, indent=1))
            n_done += 1

        logger.info("%s: %d graphs", split_dir.name, len(files))

    logger.info("linked %d graphs -> %s", n_done, out_root)


if __name__ == "__main__":
    main()
