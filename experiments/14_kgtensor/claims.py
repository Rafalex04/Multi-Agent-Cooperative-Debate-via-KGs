"""Shared loader: debate transcripts -> per-sample claim records, KG-indexed.

Two things the ThothGNN corpus builder threw away and this keeps:

INCOMING VERDICTS.  A claim's own `stance_verdict` says whether IT was agreeing
or disagreeing with someone. It says nothing about whether anyone agreed with
IT. Those are different quantities and only the first has ever been measured
(section 11 of the ledger weights claims by their own verdict). Here every claim
also carries `in_agree` / `in_disagree`, counted by scanning who addressed it.

THE FINDING INDEX.  Every claim in the v5q protocol cites exactly one of the 16
BI-RADS stance findings (measured: 15219 of 15220 claims, mean 1.00 findings per
claim). ThothGNN pushed that through a text encoder; here it stays a categorical
index into the knowledge graph, which is what makes the F x 6 evidence tensor
possible.

The KG also supplies each finding's stance polarity, used to initialise the
readout rather than to filter anything.
"""
from __future__ import annotations

import glob, json, sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "10_debate_v5"))
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

SPLITS = ("train", "val", "test")
_ROOT = _HERE.parents[2] / "breastMnist"


def kg_findings():
    """[(feature, +1 if suggests malignancy else -1)] straight from the graph."""
    from run_debate_v5 import all_findings
    triples = json.loads((_ROOT / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema = json.loads((_ROOT / "data/breast/schema.json").read_text())
    return [(feat, 1.0 if side == "MALIGNANT" else -1.0)
            for feat, _desc, side in all_findings(triples, schema)]


def load(roots, findings=None):
    """(split, sample_id) -> dict(split, y, claims[]), claims pooled over runs.

    A claim is a plain dict so downstream code stays readable:
      stance      1 MALIGNANT / 0 BENIGN
      verdict     the claim's OWN stance: AGREE / DISAGREE / OPEN
      round       0,1,2 (capped)
      finding     index into `findings`, or -1 when the cited feature is unknown
      in_agree    how many later claims AGREED with this one
      in_disagree how many later claims DISAGREED with this one
      is_repeat   flagged duplicate of an earlier claim
    """
    findings = findings or kg_findings()
    fidx = {f: i for i, (f, _) in enumerate(findings)}
    out = {}
    for run, root in enumerate(roots):
        for f in sorted(glob.glob(f"{root}/*/debate_*.json")):
            d = json.loads(Path(f).read_text())
            cl = d.get("claims") or []
            if not cl:
                continue
            split = f.split("/")[-2]
            key = (split, d["sample_id"])

            # who received what, by node id
            inc = defaultdict(lambda: [0, 0])
            for c in cl:
                v = c.get("stance_verdict")
                if v not in ("AGREE", "DISAGREE"):
                    continue
                for t in c.get("addressed_ids") or []:
                    inc[t][0 if v == "AGREE" else 1] += 1

            rec = out.setdefault(
                key, {"split": split, "sid": d["sample_id"],
                      "y": 1 if d["gold_label"] == "MALIGNANT" else 0,
                      "claims": []})
            for c in cl:
                if c.get("label") not in ("MALIGNANT", "BENIGN"):
                    continue
                feats = c.get("cited_features") or []
                a, dis = inc[c["node_id"]]
                rec["claims"].append({
                    "run": run,
                    "stance": 1 if c["label"] == "MALIGNANT" else 0,
                    "verdict": c.get("stance_verdict") or "OPEN",
                    "round": min(int(c.get("round_idx", 0)), 2),
                    "finding": fidx.get(feats[0], -1) if feats else -1,
                    "in_agree": a,
                    "in_disagree": dis,
                    "is_repeat": bool(c.get("is_repeat")),
                })
    return out


def by_split(data):
    d = {s: [] for s in SPLITS}
    for rec in data.values():
        if rec["split"] in d:
            d[rec["split"]].append(rec)
    return d


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def bacc(scores, golds, t):
    tp = sum(1 for s, g in zip(scores, golds) if g == 1 and s >= t)
    fn = sum(1 for s, g in zip(scores, golds) if g == 1 and s < t)
    tn = sum(1 for s, g in zip(scores, golds) if g == 0 and s < t)
    fp = sum(1 for s, g in zip(scores, golds) if g == 0 and s >= t)
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def report(name, by, score_fn, out=None):
    """AUC per split plus bAcc at a threshold swept on train+val only."""
    cell = {}
    for s in SPLITS:
        v = [(score_fn(r), r["y"]) for r in by[s]]
        cell[s] = [(a, g) for a, g in v if a is not None]
    a = {s: auc([x for x, g in cell[s] if g], [x for x, g in cell[s] if not g]) for s in SPLITS}
    fit = cell["train"] + cell["val"]
    fs, fg = [x for x, _ in fit], [g for _, g in fit]
    ts, tg = [x for x, _ in cell["test"]], [g for _, g in cell["test"]]
    thr = max(sorted(set(fs)), key=lambda t: bacc(fs, fg, t)) if fs else 0.0
    b = bacc(ts, tg, thr) if ts else float("nan")
    print(f"  {name:34s} train {a['train']:.4f}  val {a['val']:.4f}  "
          f"TEST {a['test']:.4f}   bAcc {b:.4f}")
    if out is not None:
        out[name] = {"train": a["train"], "val": a["val"], "test": a["test"], "bacc": b}
    return a["test"], b
