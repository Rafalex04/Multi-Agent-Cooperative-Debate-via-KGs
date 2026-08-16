"""Evaluate the two-run 0-10 scoring.

Reports the emitted-integer difference, the digit-expectation difference, and
the plain "higher score wins" decision, plus each side on its own. The integer
tends to quantise onto a couple of values, so the expectation is where any
ranking signal will show up first.

Usage:
  python evaluate_score10.py --results 'results/score10_test_*.jsonl'
"""
from __future__ import annotations

import argparse, glob, json, statistics as st
from collections import Counter
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
    best = max(0.5 * (sum(1 for x in m if x >= t) / len(m) +
                      sum(1 for x in b if x < t) / len(b))
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

    print("--- what the model actually emitted ---")
    for k in ("score_mal", "score_ben"):
        c = Counter(r[k] for r in rows if r.get(k) is not None)
        print(f"  {k}: {dict(sorted(c.items()))}")
    for k in ("exp_mal", "exp_ben"):
        v = [r[k] for r in rows if r.get(k) is not None]
        print(f"  {k}: range {min(v):.2f}..{max(v):.2f} sd {st.pstdev(v):.2f} "
              f"distinct {len(set(round(x,2) for x in v))}")

    print("\n--- single side ---")
    report(rows, "exp_mal", "E[malignant score]")
    report(rows, "exp_ben", "E[benign score]")

    print("\n--- contrast ---")
    a_i = report(rows, "diff_int", "integer difference")
    a_e = report(rows, "diff_exp", "expectation difference")

    # "Higher score wins" as a hard decision, which is how it was posed.
    wins = [(r["winner"], r["gold"]) for r in rows if r.get("winner")]
    if wins:
        tp = sum(1 for w, g in wins if w == "MALIGNANT" and g == "MALIGNANT")
        tn = sum(1 for w, g in wins if w == "BENIGN" and g == "BENIGN")
        nmw = sum(1 for _, g in wins if g == "MALIGNANT")
        nbw = len(wins) - nmw
        print(f"\n  higher-score-wins: bAcc={0.5*(tp/nmw+tn/nbw):.4f}  "
              f"decisions={dict(Counter(w for w, _ in wins))}")

    best = max([x for x in (a_i, a_e) if x is not None] or [0])
    if best < 0.5:
        for r in rows:
            if r.get("diff_exp") is not None:
                r["flipped"] = -r["diff_exp"]
        print("\n  (ranking inverted — reporting the flipped magnitude)")
        best = report(rows, "flipped", "expectation, inverted") or best

    print(f"\n{'='*74}")
    print("reference, same 156-sample test split:")
    print("  no KG baseline                  AUC 0.6013  bAcc@best 0.6284")
    print("  binary two-run contrast         AUC 0.5587  bAcc@best 0.5840")
    print("  graded two-run contrast (0-9)   AUC 0.4994  bAcc@best 0.5301")
    print("  KG feature probes (per-feature) AUC 0.6685  bAcc@best 0.6497")
    print(f"\n=> this run: {best:.4f}")


if __name__ == "__main__":
    main()
