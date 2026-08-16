"""Quality audit for a debate run, written to be pessimistic.

Every failure this run has hit so far was invisible in aggregate counts and only
showed up when the actual text was read: agents copying the toolkit descriptions
verbatim, then copying the worked examples verbatim, then dropping the format
tags once the prompt grew. v3 produced 780 graphs before anyone noticed that one
sentence accounted for 12% of all claims.

So each check below is phrased as a failure to look for, with a threshold that
prints FAIL rather than a number to interpret, and the audit ends by dumping raw
claims for the reader to judge.

Usage:
  python audit.py --debates .../debates_v4
  python audit.py --debates .../debates_v4 --compare .../debates_v3
"""
from __future__ import annotations

import argparse, json, re, statistics as st
from collections import Counter
from pathlib import Path

_SIDE = {"expert_a": "MALIGNANT", "expert_b": "BENIGN"}


def load(root: Path):
    return [json.loads(f.read_text())
            for f in sorted(root.glob("*/debate_*.json"))]


def _norm(t):
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def audit(debates, label, verbose=True):
    if not debates:
        print(f"{label}: no debates yet")
        return {}
    claims = [c for d in debates for c in d["claims"]]
    if not claims:
        print(f"{label}: no claims")
        return {}
    texts = [_norm(c["text"]) for c in claims]
    uniq = len(set(texts)) / len(texts)
    per_graph = [len(d["claims"]) for d in debates]
    cited = sum(1 for c in claims if c.get("cited_features"))
    addressed = sum(1 for c in claims if c.get("addressed_ids"))
    labelled = sum(1 for c in claims if c.get("label"))
    against = sum(1 for c in claims
                  if c.get("label") and c["label"] != _SIDE.get(c["expert_id"]))
    rounds = Counter(c["round_idx"] for c in claims)
    top = Counter(c["text"].strip() for c in claims).most_common(5)
    top10share = sum(n for _, n in Counter(texts).most_common(10)) / len(texts)

    # How much does one image's claim set differ from another's? If the debate
    # is not reading the image, these overlap almost completely.
    sets = [{_norm(c["text"]) for c in d["claims"]} for d in debates[:60]]
    jac = [len(a & b) / len(a | b)
           for i, a in enumerate(sets) for b in sets[i + 1:i + 6] if a | b]
    cross = st.mean(jac) if jac else float("nan")

    m = {
        "n_debates": len(debates), "n_claims": len(claims),
        "claims_per_graph": st.mean(per_graph),
        "uniqueness": uniq, "top10_share": top10share,
        "cross_sample_overlap": cross,
        "pct_cited": cited / len(claims), "pct_addressed": addressed / len(claims),
        "pct_labelled": labelled / len(claims), "pct_against_side": against / len(claims),
    }

    if not verbose:
        return m

    print(f"\n{'='*74}\n{label}   ({len(debates)} debates, {len(claims)} claims)\n{'='*74}")

    def check(name, val, good, fmt="{:.3f}", worse_is_low=True):
        ok = (val >= good) if worse_is_low else (val <= good)
        flag = "ok  " if ok else "FAIL"
        arrow = ">=" if worse_is_low else "<="
        print(f"  [{flag}] {name:34s} {fmt.format(val)}   want {arrow} {good}")

    check("claim uniqueness", uniq, 0.60)
    check("top-10 claims share of all", top10share, 0.25, worse_is_low=False)
    check("cross-sample claim overlap", cross, 0.35, worse_is_low=False)
    check("claims per graph", m["claims_per_graph"], 10.0, "{:.1f}")
    check("% claims citing a finding", m["pct_cited"], 0.80)
    check("% claims with a label", m["pct_labelled"], 0.90)
    check("% rebuttals with an edge", addressed / max(1, len(claims) - rounds[0]), 0.70)
    check("% labelled against own side", m["pct_against_side"], 0.05)

    print(f"\n  claims per round: {dict(sorted(rounds.items()))}")
    print(f"  most repeated claims:")
    for t, n in top:
        print(f"     {n:4d}x  {t[:78]}")
    return m


def show_examples(debates, n=2):
    print(f"\n{'='*74}\nRAW DEBATES — read these, the metrics above will not catch everything\n{'='*74}")
    for d in debates[:n]:
        print(f"\n--- sample {d['sample_id']}  gold={d['gold_label']} ---")
        for c in d["claims"]:
            side = "M" if c["expert_id"] == "expert_a" else "B"
            flag = " <AGAINST-SIDE>" if (c.get("label") and
                                         c["label"] != _SIDE.get(c["expert_id"])) else ""
            cite = ",".join(c.get("cited_features", [])) or "-"
            addr = ",".join(c.get("addressed_ids", [])) or "-"
            print(f"  {c['node_id']:4s} {side} r{c['round_idx']} "
                  f"{str(c.get('label'))[:4]:4s} cite={cite[:28]:28s} ->{addr:4s}{flag}")
            print(f"        {c['text'][:96]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debates", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--examples", type=int, default=2)
    args = p.parse_args()

    cur = load(Path(args.debates))
    a = audit(cur, Path(args.debates).name)
    if args.examples:
        show_examples(cur, args.examples)

    if args.compare:
        other = load(Path(args.compare))
        b = audit(other, Path(args.compare).name)
        if a and b:
            print(f"\n{'='*74}\n{'metric':32s} {Path(args.debates).name:>14s} "
                  f"{Path(args.compare).name:>14s}\n{'='*74}")
            for k in ("claims_per_graph", "uniqueness", "top10_share",
                      "cross_sample_overlap", "pct_cited", "pct_addressed",
                      "pct_labelled", "pct_against_side"):
                print(f"{k:32s} {a.get(k, float('nan')):14.3f} {b.get(k, float('nan')):14.3f}")


if __name__ == "__main__":
    main()
