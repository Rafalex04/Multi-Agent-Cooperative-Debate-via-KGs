"""ThothGNN Stage 4, step 1: build and freeze the heterogeneous graph corpus.

Reads the debate transcripts directly rather than the existing dataset_* graph
JSONs. Those were built by two different scripts and drop fields Stage 4 needs:
`is_repeat`, `addressed_ids` and the explicit AGREE/DISAGREE verdict. Building
from the transcripts gives one code path for every corpus.

Emits, per sample, the node/edge tensors described in Section 1 of the plan:

  claim nodes   12 features (stance, round, agent, 4 degrees, KG grounding,
                n triples, repeat flag)
  triple nodes  stance sign, relation id (embedded later), citation frequency
  edges         AGREE / DISAGREE / CITES / CITED_BY / NEXT_TURN
  class signs   +1 when a claim's stance matches the class, -1 when it opposes

mal_share is stored per graph and is the quantity the BASE configuration must
reproduce, so it is computed exactly as the existing results were: malignant
claims over ALL claims, matching the audit and ensemble scripts.

Usage (must run where sentence-transformers lives, i.e. gpu22):
  python build_corpus.py --debates .../debates_v5q --out corpus/v5q.pt
"""
from __future__ import annotations

import argparse, json, logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TAXONOMY = {"is_a", "subtype_of", "birads_subcategory_of", "category_descriptor",
             "also_known_as"}
_NEGATION = ("_no_", "no_", "not_", "non_", "absence", "lack_", "without")
_MAL = ("malignan", "suspicious", "birads_5", "birads_4")
_BEN = ("benign", "birads_1", "birads_2", "birads_3")


def polarity(relation: str, obj: str) -> float:
    """Diagnostic sign of a triple, read from the ontology's own vocabulary.

    Negation is tested first so `has_no_malignant_potential` reads benign rather
    than matching the substring "malignan".
    """
    s = f"{relation} {obj}".lower()
    flip = -1.0 if any(n in relation.lower() for n in _NEGATION) else 1.0
    if any(k in s for k in _BEN):
        return -1.0 * flip
    if any(k in s for k in _MAL):
        return +1.0 * flip
    return 0.0


def verbalise(t: dict) -> str:
    s = t.get("subject", "").replace("_", " ")
    r = t.get("relation", "").replace("_", " ")
    o = str(t.get("object", "")).replace("_", " ")
    return (f"{s} {r}" if o in ("true", "") else f"{s} {r} {o}").strip()


def stance_edges(claims: list[dict]) -> tuple[list, list]:
    """AGREE and DISAGREE claim->claim edges.

    v5 states the verdict per claim. v3/v4 do not, so it is inferred from
    whether the two claims carry the same stance -- the same rule
    build_graphs_v4.py used, kept identical so older corpora stay comparable.
    """
    by_id = {c["node_id"]: c for c in claims}
    agree, disagree = [], []
    for c in claims:
        for tgt in c.get("addressed_ids") or []:
            if tgt not in by_id or tgt == c["node_id"]:
                continue
            v = c.get("stance_verdict")
            if v is None:
                v = "AGREE" if c.get("label") == by_id[tgt].get("label") else "DISAGREE"
            (agree if v == "AGREE" else disagree).append((c["node_id"], tgt))
    return agree, disagree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4, help="triples retrieved per claim")
    p.add_argument("--min-sim", type=float, default=0.20)
    p.add_argument("--encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--kg-root", default="breastMnist")
    args = p.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    kg = json.loads(Path(args.kg_root, "data/breast/knowledge_graph.json").read_text())["triples"]
    kg = [t for t in kg if t.get("relation") not in _TAXONOMY]
    rel_vocab = sorted({t.get("relation", "") for t in kg})
    rel_id = {r: i for i, r in enumerate(rel_vocab)}
    tri_pol = [polarity(t.get("relation", ""), str(t.get("object", ""))) for t in kg]

    enc = SentenceTransformer(args.encoder)
    T = enc.encode([verbalise(t) for t in kg], normalize_embeddings=True,
                   show_progress_bar=False)
    logger.info("%d non-taxonomy triples, %d relations, %d with a polarity",
                len(kg), len(rel_vocab), sum(1 for x in tri_pol if x))

    root = Path(args.debates)
    files = [(sp.name, f) for sp in sorted(d for d in root.iterdir() if d.is_dir())
             for f in sorted(sp.glob("debate_*.json"))]
    logger.info("%d transcripts under %s", len(files), root)

    raw = []
    for split, f in files:
        d = json.loads(f.read_text())
        if d["claims"]:
            raw.append((split, d))
    logger.info("%d with at least one claim (%d empty, skipped)",
                len(raw), len(files) - len(raw))

    # One encoder pass over every claim in the corpus.
    texts = [c["text"] for _, d in raw for c in d["claims"]]
    C = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                   batch_size=256)
    logger.info("encoded %d claims", len(C))

    # Corpus-wide citation frequency needs a first pass over retrieval.
    off, hits = 0, []
    for _, d in raw:
        n = len(d["claims"])
        sims = C[off:off + n] @ T.T
        off += n
        idx = np.argsort(-sims, axis=1)[:, :args.k]
        hits.append([[(int(j), float(sims[i, j])) for j in row
                      if sims[i, j] >= args.min_sim] for i, row in enumerate(idx)])
    freq = Counter(j for g in hits for row in g for j, _ in row)
    fv = np.array([freq.get(i, 0) for i in range(len(kg))], dtype=np.float64)
    fz = (fv - fv.mean()) / max(1e-6, fv.std())

    graphs = []
    for (split, d), hit in zip(raw, hits):
        claims = d["claims"]
        cid = {c["node_id"]: i for i, c in enumerate(claims)}
        agree, disagree = stance_edges(claims)

        deg = {k: [0.0] * len(claims) for k in ("ia", "id", "oa", "od")}
        for s, t in agree:
            deg["oa"][cid[s]] += 1; deg["ia"][cid[t]] += 1
        for s, t in disagree:
            deg["od"][cid[s]] += 1; deg["id"][cid[t]] += 1

        # Local triple index: only triples this graph actually cites.
        used = sorted({j for row in hit for j, _ in row})
        tpos = {j: i for i, j in enumerate(used)}
        cites = [(cid[claims[i]["node_id"]], tpos[j], s)
                 for i, row in enumerate(hit) for j, s in row]

        rounds = max((c.get("round_idx", 0) for c in claims), default=0) or 1
        x = []
        for i, c in enumerate(claims):
            lab = c.get("label")
            sims = [s for k, row in enumerate(hit) if k == i for _, s in row]
            x.append([
                1.0 if lab == "MALIGNANT" else 0.0,
                1.0 if lab == "BENIGN" else 0.0,
                c.get("round_idx", 0) / rounds,
                1.0 if c.get("expert_id") in ("agent_1", "expert_a") else 0.0,
                1.0 if c.get("expert_id") in ("agent_2", "expert_b") else 0.0,
                deg["ia"][i] / 4.0, deg["id"][i] / 4.0,
                deg["oa"][i] / 4.0, deg["od"][i] / 4.0,
                (sum(sims) / len(sims)) if sims else 0.0,
                len(sims) / 4.0,
                1.0 if c.get("is_repeat") else 0.0,
            ])

        # NEXT_TURN: consecutive claims by the same agent (SMAGDi continuity).
        nxt = []
        for a in {c.get("expert_id") for c in claims}:
            seq = [i for i, c in enumerate(claims) if c.get("expert_id") == a]
            nxt += list(zip(seq, seq[1:]))

        # Bipolar class attachment: +1 when the claim's stance is the class,
        # -1 when it opposes, 0 when the claim carries no stance.
        sgn = [(1.0 if c.get("label") == "MALIGNANT" else
                -1.0 if c.get("label") == "BENIGN" else 0.0) for c in claims]

        mal = sum(1 for c in claims if c.get("label") == "MALIGNANT")
        graphs.append({
            "sample_id": d["sample_id"], "split": split,
            "y": 1 if d["gold_label"] == "MALIGNANT" else 0,
            "mal_share": mal / len(claims),
            "x_claim": x,
            "x_triple": [[tri_pol[j], fz[j]] for j in used],
            "rel_triple": [rel_id.get(kg[j].get("relation", ""), 0) for j in used],
            "triple_gid": used,
            "agree": [(cid[s], cid[t]) for s, t in agree],
            "disagree": [(cid[s], cid[t]) for s, t in disagree],
            "cites": cites,
            "next": nxt,
            "class_sign": sgn,
        })

    meta = {"n_relations": len(rel_vocab), "claim_dim": 12,
            "encoder": args.encoder, "k": args.k, "debates": str(root)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "meta": meta}, out)

    from collections import Counter as Ct
    logger.info("wrote %d graphs -> %s", len(graphs), out)
    logger.info("splits %s", dict(Ct(g["split"] for g in graphs)))
    logger.info("mean claims %.1f | agree %.1f | disagree %.1f | cites %.1f | triples %.1f",
                *[sum(len(g[k]) for g in graphs) / len(graphs)
                  for k in ("x_claim", "agree", "disagree", "cites", "x_triple")])


if __name__ == "__main__":
    main()
