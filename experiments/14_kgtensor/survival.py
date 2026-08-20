"""Step 1: does being disagreed WITH carry information?

The ledger's evidence weights index a claim by its OWN verdict. Nobody has
measured the other direction -- whether a claim that survived the debate
unchallenged predicts better than one that got attacked. The two are different
quantities, and if incoming disagreement carries nothing then dropping DISAGREE
edges from the topology is confirmed by a second, independent route.

Survival weighting, no fitted parameters:

    d_i = in_disagree / (in_agree + in_disagree)        0 when nobody replied
    mal_share_surv = sum_i s_i (1 - d_i) / sum_i (1 - d_i)

A claim everyone attacked contributes nothing; an unchallenged claim contributes
fully. Reported next to the diagnostic that says whether the ingredient is even
present: P(gold = MAL | stance, incoming) by subgroup.

Usage:
  python survival.py --debates .../debates_v5q [.../debates_v5q_r1 ...]
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import SPLITS, by_split, load, report                     # noqa: E402


def mal_share(r):
    c = r["claims"]
    return sum(x["stance"] for x in c) / len(c) if c else 0.5


def survival(r, eps=0.0):
    """Weight each claim by the fraction of its replies that were NOT attacks."""
    num = den = 0.0
    for x in r["claims"]:
        tot = x["in_agree"] + x["in_disagree"]
        d = x["in_disagree"] / tot if tot else 0.0
        w = (1.0 - d) + eps
        num += x["stance"] * w
        den += w
    return num / den if den else 0.5


def endorsed(r):
    """The opposite emphasis: weight by how much agreement a claim attracted."""
    num = den = 0.0
    for x in r["claims"]:
        w = 1.0 + x["in_agree"]
        num += x["stance"] * w
        den += w
    return num / den if den else 0.5


def unchallenged_only(r):
    """Hardest version: keep only claims nobody disagreed with."""
    c = [x for x in r["claims"] if x["in_disagree"] == 0]
    return sum(x["stance"] for x in c) / len(c) if c else mal_share(r)


def diagnose(train):
    """Is the ingredient present at all? Claim-level precision by incoming verdict."""
    tot = defaultdict(lambda: [0.0, 0.0])
    for r in train:
        for x in r["claims"]:
            t = x["in_agree"] + x["in_disagree"]
            grp = ("none" if t == 0 else
                   "attacked" if x["in_disagree"] > x["in_agree"] else "endorsed")
            k = ("MAL" if x["stance"] else "BEN", grp)
            tot[k][0] += r["y"]; tot[k][1] += 1
    print("\n  P(gold = MALIGNANT | claim stance, incoming verdicts), train split")
    print(f"    {'stance':7s} {'incoming':10s} {'n':>7s} {'P(mal)':>8s}")
    for k in sorted(tot):
        s, n = tot[k]
        print(f"    {k[0]:7s} {k[1]:10s} {int(n):7d} {s/n:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load(args.debates)
    by = by_split(data)
    print(f"pooled {len(args.debates)} run(s): "
          + "  ".join(f"{s} {len(by[s])}" for s in SPLITS))
    nrep = sum(1 for r in by["train"] for x in r["claims"]
               if x["in_agree"] + x["in_disagree"] > 0)
    ntot = sum(len(r["claims"]) for r in by["train"])
    print(f"claims that received a reply: {nrep}/{ntot} ({100*nrep/max(1,ntot):.1f}%)")

    diagnose(by["train"])

    res = {}
    print("\n  readout                              train      val      TEST     bAcc")
    report("mal_share (baseline)", by, mal_share, res)
    report("survival-weighted", by, survival, res)
    report("survival-weighted (eps=0.25)", by, lambda r: survival(r, 0.25), res)
    report("endorsement-weighted", by, endorsed, res)
    report("unchallenged claims only", by, unchallenged_only, res)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
