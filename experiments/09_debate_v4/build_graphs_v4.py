"""Build graphs from the stance-matched debates.

Unlike v3, claim-to-triple edges are not recovered by embedding similarity —
each claim states which finding it invoked, so the edge is what the agent
actually cited. Every triple carrying that finding as its subject is attached,
which also brings in its appearance and classification triples, not just the
stance one.

Emits the same schema as dataset_full / dataset_v2 / dataset_v3 so the GNN
trainers read it unchanged.

Usage:
  python build_graphs_v4.py --debates .../debates_v4 --out-dir .../dataset_v4
"""
from __future__ import annotations

import argparse, json, logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_TAXONOMY = {"is_a", "subtype_of", "birads_subcategory_of", "category_descriptor",
             "also_known_as"}
# v4 assigns sides; v5 does not, so agents there are agent_1/agent_2 and no
# claim can be "against side". Both formats build the same graph schema.
_SIDE = {"expert_a": "MALIGNANT", "expert_b": "BENIGN"}
# Literal objects, not entities — expanding through them would join every
# `suggests_benign true` triple into one hub.
_STOP = {"true", "false", "possible", ""}


def neighbourhood(seed: str, incident: dict, hops: int) -> dict:
    """Triples within `hops` of a cited feature, following subject and object.

    Measured on this graph, expansion buys very little: 14 of the 16 stance
    features are leaves carrying only their own stance assertion, so 1-hop and
    3-hop return the same single triple. Only architectural_distortion (18 -> 69)
    and clustered_microcysts (11 -> 60) have real neighbourhoods, because the
    graph's 113 appearance triples hang off lesion types rather than features.
    The expansion is implemented properly regardless, so a denser graph would
    benefit without code changes.
    """
    ents, used = {seed}, {}
    for _ in range(max(1, hops)):
        for e in list(ents):
            if e in _STOP:
                continue
            for t in incident.get(e, []):
                used[t["id"]] = t
                ents |= {t.get("subject", ""), str(t.get("object", ""))}
    return used


def verbalise(t):
    s = t.get("subject", "").replace("_", " ")
    r = t.get("relation", "").replace("_", " ")
    o = str(t.get("object", "")).replace("_", " ")
    return f"{s} {r}".strip() if o in ("true", "") else f"{s} {r} {o}".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debates", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kg-root", default=str(_HERE.parents[2] / "breastMnist"))
    p.add_argument("--hops", type=int, default=1,
                   help="KG expansion depth around each cited feature")
    p.add_argument("--drop-repeats", action="store_true",
                   help="discard claims flagged is_repeat. Debate-time dedup is "
                        "irreversible and cost 9 of every 12 rebuttal edges, so "
                        "repeats are kept and flagged during generation and "
                        "filtered here instead, which makes the choice a "
                        "build-time option rather than a commitment.")
    p.add_argument("--max-round0", type=int, default=0,
                   help="cap opening claims per agent (0 = keep all). The prompt "
                        "asks for at most 3 but the model emits ~5.4, and the "
                        "opening round is where the repetition lives: measured "
                        "over 217 graphs, round 0 has uniqueness 0.184 against "
                        "0.622 and 0.731 for the two rebuttal rounds, while "
                        "supplying 80% of all claims. Capping at 3 lifts overall "
                        "uniqueness from 0.204 to 0.283 at the cost of dropping "
                        "claims, so both variants are worth training.")
    args = p.parse_args()

    root = Path(args.kg_root)
    all_triples = json.loads(
        (root / "data/breast/knowledge_graph.json").read_text())["triples"]
    incident = defaultdict(list)
    for t in all_triples:
        if t.get("relation") in _TAXONOMY:
            continue
        incident[t.get("subject", "")].append(t)
        incident[str(t.get("object", ""))].append(t)

    dbg_root, out_root = Path(args.debates), Path(args.out_dir)
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
            if args.drop_repeats:
                claims = [c for c in claims if not c.get("is_repeat")]
            if args.max_round0:
                per, kept = defaultdict(int), []
                for c in claims:
                    if c["round_idx"] == 0:
                        if per[c["expert_id"]] >= args.max_round0:
                            continue
                        per[c["expert_id"]] += 1
                    kept.append(c)
                claims = kept
            if not claims:
                continue

            claim_nodes, used, ct = [], {}, []
            for c in claims:
                for feat in c.get("cited_features", []):
                    for t in neighbourhood(feat, incident, args.hops).values():
                        used[t["id"]] = t
                        # Direct edge only for the triple naming the feature;
                        # anything reached by expansion is context, and marking
                        # it as "cited" would overstate what the agent said.
                        direct = (t.get("subject") == feat or
                                  str(t.get("object")) == feat)
                        ct.append({"src": c["node_id"], "dst": t["id"],
                                   "type": "cited" if direct else "expanded",
                                   "feature": feat})
                side = _SIDE.get(c["expert_id"], "")
                claim_nodes.append({
                    "node_id": c["node_id"], "type": "claim", "text": c["text"],
                    "label": c["label"],
                    "label_int": {"MALIGNANT": 1, "BENIGN": 0}.get(c["label"], -1),
                    "label_explicit": c["label_explicit"],
                    "expert_id": c["expert_id"], "round_idx": c["round_idx"],
                    "cited_features": c.get("cited_features", []),
                    "n_cited": len(c.get("cited_features", [])),
                    # Repeats are kept for the ADDRESSED edge they carry; the
                    # flag lets a model discount their text without losing the
                    # structure. See run_debate_v4._dedupe.
                    "is_repeat": bool(c.get("is_repeat")),
                    # True when the agent labelled against the side it was told
                    # to argue, which v1 and v2 had no way to express.
                    "against_side": bool(c["label"]) and c["label"] != side,
                })

            triple_nodes = [{
                "node_id": tid, "type": "triple",
                "subject": t.get("subject", ""), "predicate": t.get("relation", ""),
                "object": t.get("object", ""), "text": verbalise(t),
            } for tid, t in used.items()]

            ids = {c["node_id"] for c in claim_nodes}
            lab = {c["node_id"]: c["label"] for c in claim_nodes}
            cc, seen = [], set()
            for c in claims:
                for tgt in set(c["addressed_ids"]):
                    if tgt not in ids or tgt == c["node_id"] or \
                       (c["node_id"], tgt) in seen:
                        continue
                    seen.add((c["node_id"], tgt))
                    # v5 states AGREE/DISAGREE explicitly, which is what the
                    # agent meant; v4 has to infer it from labels matching.
                    v = c.get("stance_verdict")
                    if v in ("AGREE", "DISAGREE"):
                        etype = v
                    else:
                        same = lab[c["node_id"]] and lab[c["node_id"]] == lab[tgt]
                        etype = "AGREE" if same else "DISAGREE"
                    cc.append({"src": c["node_id"], "dst": tgt,
                               "sign": "+" if etype == "AGREE" else "-",
                               "type": etype,
                               "explicit": v in ("AGREE", "DISAGREE")})

            tt, tids = [], list(used)
            for a in range(len(tids)):
                for b in range(a + 1, len(tids)):
                    ta, tb = used[tids[a]], used[tids[b]]
                    if ta.get("subject") == tb.get("subject") or \
                       ta.get("object") == tb.get("object"):
                        tt.append({"src": tids[a], "dst": tids[b],
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
                "n_against_side": sum(1 for c in claim_nodes if c["against_side"]),
                "n_repeat": sum(1 for c in claim_nodes if c["is_repeat"]),
                "nodes": {"claims": claim_nodes, "triples": triple_nodes},
                "edges": {"claim_claim": cc, "claim_triple": ct, "triple_triple": tt},
                "stats": {
                    "num_claims": len(claim_nodes), "num_triples": len(triple_nodes),
                    "num_claim_claim_edges": len(cc),
                    "num_claim_triple_edges": len(ct),
                    "num_triple_triple_edges": len(tt),
                },
            }, indent=1))
            n_done += 1

        logger.info("%s: %d graphs", split_dir.name, len(files))

    logger.info("built %d graphs -> %s", n_done, out_root)


if __name__ == "__main__":
    main()
