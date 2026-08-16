"""Score the graded two-run contrast, and verify it did not saturate.

The binary variant of this experiment failed silently in the sense that its AUC
looked merely mediocre; only the per-class means revealed that every context
answered identically for all 156 images. So this reports the saturation
diagnostics first: the spread of E[d] and the digit-distribution entropy. If
those are near zero the AUC below is meaningless regardless of its value.

Usage:
  python evaluate_scored.py --results 'results/scored_test_*.jsonl'
"""
from __future__ import annotations

import argparse, glob, json, statistics as st
from pathlib import Path


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def report(rows, key, name):
    vals = [(r[key], r["gold"]) for r in rows if r.get(key) is not None]
    if not vals:
        print(f"  {name:26s} no data")
        return None
    m = [v for v, g in vals if g == "MALIGNANT"]
    b = [v for v, g in vals if g == "BENIGN"]
    a = auc(m, b)
    best, th = max((0.5 * (sum(1 for x in m if x >= t) / len(m) +
                           sum(1 for x in b if x < t) / len(b)), t)
                   for t in sorted({v for v, _ in vals}))
    print(f"  {name:26s} AUC={a:.4f}  bAcc@best={best:.4f}  "
          f"mean(mal)={st.mean(m):+.3f} mean(ben)={st.mean(b):+.3f}  "
          f"sd={st.pstdev([v for v, _ in vals]):.3f}")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    args = p.parse_args()

    rows = []
    for pat in args.results:
        for f in sorted(glob.glob(pat)):
            rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    rows = list({r["index"]: r for r in rows}.values())
    nm = sum(1 for r in rows if r["gold"] == "MALIGNANT")
    print(f"N={len(rows)}  malignant={nm}  benign={len(rows)-nm}\n")

    print("--- saturation check (the binary variant died here) ---")
    for k in ("e_mal", "e_ben"):
        v = [r[k] for r in rows if r.get(k) is not None]
        print(f"  {k:8s} range {min(v):.2f}..{max(v):.2f}  mean {st.mean(v):.2f}  "
              f"sd {st.pstdev(v):.2f}  distinct {len(set(round(x,2) for x in v))}")
    for k in ("entropy_mal", "entropy_ben"):
        v = [r[k] for r in rows if r.get(k) is not None]
        print(f"  {k:8s} mean {st.mean(v):.3f}  (0 = collapsed onto one digit)")

    print("\n--- single context ---")
    report(rows, "e_mal", "E[evidence | mal rules]")
    report(rows, "e_ben", "E[evidence | ben rules]")

    print("\n--- contrast ---")
    a = report(rows, "score", "E[mal] - E[ben]")

    # A sign flip is still signal; report the usable magnitude either way.
    if a is not None and a < 0.5:
        for r in rows:
            if r.get("score") is not None:
                r["score_flipped"] = -r["score"]
        print("\n  AUC below 0.5 means the ranking is inverted, not absent:")
        a = report(rows, "score_flipped", "inverted")

    print(f"\n{'='*74}")
    print("reference, same 156-sample test split:")
    print("  no KG baseline                  AUC 0.6013   bAcc@best 0.6284")
    print("  binary two-run contrast         AUC 0.5587   bAcc@best 0.5840")
    print("  KG feature probes (per-feature) AUC 0.6685   bAcc@best 0.6497")
    if a is not None:
        if a > 0.6685:
            print(f"\n=> {a:.4f}: best KG method so far")
        elif a > 0.6013:
            print(f"\n=> {a:.4f}: beats the no-KG baseline")
        elif a > 0.5587:
            print(f"\n=> {a:.4f}: better than the binary contrast, still under baseline")
        else:
            print(f"\n=> {a:.4f}: no better than the binary contrast")


if __name__ == "__main__":
    main()
