"""Why does debate structure add nothing beyond the stance count?

The GNN keeps landing on the anchor: FULL scored 0.7136 / 0.7090 / 0.7044
against BASE's 0.7128, with gamma swinging +0.279 / -0.274 / -0.343. That is
noise, not a small gain, so the question is whether the structure it is being
offered carries any label information at all.

The decisive quantity is claim-level precision. mal_share treats every
malignant claim as equally informative. Attention can only beat it by finding
subsets of claims whose stance is MORE predictive than average. So for each
subgroup this measures:

    P(image is malignant | a claim in this subgroup says MALIGNANT)

If that probability is flat across rounds, challenged/unchallenged, agents and
KG-grounding, then no reweighting exists that beats a uniform count, and the
architecture cannot help however it is trained. If it varies sharply, the
signal is there and the failure is optimisation.

Usage:
  python diagnose.py --debates .../debates_v5q [.../debates_v5q_r1]
"""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict


def prec(rows):
    """P(gold malignant | claim asserts malignant), and the benign mirror."""
    m = [g for lab, g in rows if lab == "MALIGNANT"]
    b = [g for lab, g in rows if lab == "BENIGN"]
    return (sum(m) / len(m) if m else float("nan"), len(m),
            1 - (sum(b) / len(b)) if b else float("nan"), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    args = ap.parse_args()

    for root in args.debates:
        files = sorted(glob.glob(f"{root}/*/debate_*.json"))
        groups = defaultdict(list)
        base = []
        for f in files:
            d = json.loads(open(f).read())
            if not d["claims"]:
                continue
            g = 1 if d["gold_label"] == "MALIGNANT" else 0
            challenged = {t for c in d["claims"] for t in (c.get("addressed_ids") or [])
                          if c.get("stance_verdict") == "DISAGREE"}
            for c in d["claims"]:
                lab = c.get("label")
                if lab not in ("MALIGNANT", "BENIGN"):
                    continue
                base.append((lab, g))
                groups[f"round {c.get('round_idx', 0)}"].append((lab, g))
                groups["challenged" if c["node_id"] in challenged
                       else "unchallenged"].append((lab, g))
                groups[c.get("expert_id", "?")].append((lab, g))
                groups["repeat" if c.get("is_repeat") else "fresh"].append((lab, g))
                v = c.get("stance_verdict")
                groups[f"verdict {v}"].append((lab, g))
                n = len(c.get("cited_features") or [])
                groups[f"cites {'yes' if n else 'no'}"].append((lab, g))

        pm, nm, pb, nb = prec(base)
        print(f"\n=== {root.split('/')[-1]} ===")
        print(f"  overall: P(mal | claim=MAL) {pm:.3f} (n={nm})   "
              f"P(ben | claim=BEN) {pb:.3f} (n={nb})")
        print(f"  {'subgroup':18s} {'P(mal|MAL)':>11s} {'n':>6s} "
              f"{'P(ben|BEN)':>11s} {'n':>6s} {'lift':>7s}")
        for k in sorted(groups):
            a, na, b, nb2 = prec(groups[k])
            if na < 30 and nb2 < 30:
                continue
            lift = ((a - pm) if a == a else 0) + ((b - pb) if b == b else 0)
            print(f"  {k:18s} {a:11.3f} {na:6d} {b:11.3f} {nb2:6d} {lift:+7.3f}")


if __name__ == "__main__":
    main()
