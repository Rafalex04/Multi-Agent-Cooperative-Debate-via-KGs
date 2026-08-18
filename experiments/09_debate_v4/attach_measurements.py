"""Attach measured probe values to debate-graph claim nodes.

WHY
---
The GNN comparison was handicapped in a way that only showed up on inspection.
`compare_datasets.py` builds node features from claim text plus seven small
meta fields, and never reads `p_yes`. So v2's measurements were discarded, and
v3/v4/v5 never had any. Every dataset was compared on claim text — which every
experiment in this project says does not track the image.

The consequence is visible in v2, the one dataset that did carry measurements:

  average of its node measurements   AUC 0.6535
  GraphSAGE on its text embeddings   AUC 0.6027

The GNN scored below a plain mean of values it was never shown.

WHAT THIS DOES
--------------
Each v4/v5 claim already names the finding it cites, and the v2 graphs already
hold a measured p(yes) for all 17 findings on all 780 images. So the measurement
for a claim's cited feature can simply be looked up and written onto the node —
no new inference, the probes were run once and are reused.

That produces the combination nothing has tested yet: debate structure for the
edges, and per-node values that demonstrably carry image signal.

Usage:
  python attach_measurements.py --dataset .../dataset_v5 \
      --evidence-from .../dataset_v2 --out-dir .../dataset_v5m
"""
from __future__ import annotations

import argparse, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_evidence(root: Path) -> dict[str, dict[str, dict]]:
    """sample_id -> feature -> {p_yes, weight}."""
    out = {}
    for f in sorted(root.glob("*/graphs/*.json")):
        g = json.loads(f.read_text())
        ev = g.get("evidence")
        if ev:
            out[g["sample_id"]] = {
                e["feature"]: {"p_yes": e.get("p_yes"), "weight": e.get("weight", 0.0)}
                for e in ev}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--evidence-from", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ev_by_sample = load_evidence(Path(args.evidence_from))
    logger.info("loaded measurements for %d samples", len(ev_by_sample))

    src, dst = Path(args.dataset), Path(args.out_dir)
    n_done = n_hit = n_claim = 0

    for f in sorted(src.glob("*/graphs/*.json")):
        g = json.loads(f.read_text())
        ev = ev_by_sample.get(g["sample_id"], {})

        for c in g["nodes"]["claims"]:
            feats = c.get("cited_features") or ([c["cited_feature"]]
                                                if c.get("cited_feature") else [])
            vals = [ev[x] for x in feats if x in ev and ev[x]["p_yes"] is not None]
            n_claim += 1
            if vals:
                n_hit += 1
                # Mean when a claim cites several findings; in practice it cites one.
                c["p_yes"] = sum(v["p_yes"] for v in vals) / len(vals)
                c["stance_weight"] = sum(v["weight"] for v in vals) / len(vals)
                c["measured"] = True
            else:
                # 0.5 is the uninformative value, and `measured` lets the model
                # tell a real 0.5 from a missing one.
                c["p_yes"] = 0.5
                c["stance_weight"] = 0.0
                c["measured"] = False

        # Graph-level readout over the measurements the debate actually cited,
        # which is not the same as the full probe vector: the agents choose
        # which findings to raise.
        cited = [(c["p_yes"], c["stance_weight"]) for c in g["nodes"]["claims"]
                 if c.get("measured")]
        if cited:
            den = sum(abs(w) for _, w in cited)
            g["cited_evidence_score"] = (
                sum(w * p for p, w in cited) / den if den else None)
        else:
            g["cited_evidence_score"] = None
        g["evidence"] = [{"feature": k, "p_yes": v["p_yes"], "weight": v["weight"]}
                         for k, v in ev.items()]

        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(g, indent=1))
        n_done += 1

    logger.info("wrote %d graphs -> %s", n_done, dst)
    logger.info("claims with a measurement: %d/%d (%.1f%%)",
                n_hit, n_claim, 100 * n_hit / max(1, n_claim))


if __name__ == "__main__":
    main()
