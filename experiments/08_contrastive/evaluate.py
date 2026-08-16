"""Score the two-run contrast and check whether subtraction did what it should.

Reports each single-context run on its own alongside the difference. That is the
point of the experiment: if the contrast works, each context alone should look
like the earlier prompt-based failures while the difference beats them, which
would show the gain comes from cancelling the shared bias rather than from the
rules themselves.

Usage:
  python evaluate.py --results results/contrastive_test_*.jsonl
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
        print(f"{name:34s} no data")
        return
    m = [v for v, g in vals if g == "MALIGNANT"]
    b = [v for v, g in vals if g == "BENIGN"]
    a = auc(m, b)
    best, th = max((0.5 * (sum(1 for x in m if x >= t) / len(m) +
                           sum(1 for x in b if x < t) / len(b)), t)
                   for t in sorted({v for v, _ in vals}))
    ties = sum(x == y for x in m for y in b) / (len(m) * len(b))
    print(f"{name:34s} AUC={a:.4f}  bAcc@best={best:.4f}  "
          f"mean(mal)={st.mean(m):+.3f} mean(ben)={st.mean(b):+.3f}  ties={ties:.1%}")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    args = p.parse_args()

    rows = []
    for pat in args.results:
        for f in sorted(glob.glob(pat)):
            rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    # Shards may overlap if a sweeper ran; keep one row per sample.
    rows = list({r["index"]: r for r in rows}.values())
    nm = sum(1 for r in rows if r["gold"] == "MALIGNANT")
    print(f"N={len(rows)}  malignant={nm}  benign={len(rows)-nm}\n")

    print("--- single context (expected to behave like the earlier failures) ---")
    report(rows, "verdict_mal_ctx", "verdict | malignant rules")
    report(rows, "verdict_ben_ctx", "verdict | benign rules")
    report(rows, "match_mal_ctx",   "match   | malignant rules")
    report(rows, "match_ben_ctx",   "match   | benign rules")

    print("\n--- contrast (the experiment) ---")
    a1 = report(rows, "score_verdict", "verdict difference")
    a2 = report(rows, "score_match",   "match difference")

    # Sum of both contrasts, in case they carry complementary signal.
    for r in rows:
        if r.get("score_verdict") is not None and r.get("score_match") is not None:
            r["score_both"] = r["score_verdict"] + r["score_match"]
    a3 = report(rows, "score_both", "verdict + match")

    print(f"\n{'='*72}")
    print("reference, same 156-sample test split:")
    print("  no KG baseline                 AUC 0.6013   bAcc@best 0.6284")
    print("  best KG-in-prompt (kg_v3)      AUC 0.5825   bAcc@best 0.6228")
    print("  KG feature probes (17, manual) AUC 0.6685   bAcc@best 0.6497")
    best = max(x for x in (a1, a2, a3) if x is not None)
    print(f"\ncontrast best = {best:.4f}")
    if best > 0.6685:
        print("=> beats the feature probes; the contrast is the strongest KG use so far")
    elif best > 0.6013:
        print("=> beats the no-KG baseline, unlike every single-run prompt method")
    else:
        print("=> still below baseline; subtraction did not recover the bias")


if __name__ == "__main__":
    main()
