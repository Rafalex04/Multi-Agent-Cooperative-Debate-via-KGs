"""Do rebuttals track truth?

Propagation on an attack graph can only help if being attacked is evidence of
being wrong. That is an assumption about the debate protocol, not a theorem, and
it is directly measurable: compare the credibility of claims whose stance matches
the gold label against those whose stance does not.

If credible claims are no likelier to be correct, then no amount of message
passing over the attack graph can recover anything -- the topology is independent
of the quantity being predicted, and every architecture over it is a null.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))


def auc(pos, neg):
    """Rank-based AUC with tie correction. The O(n_pos*n_neg) version in claims.py
    is fine for 156 test samples and hopeless for the ~15k claims pooled here."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

from diag_credibility import credibility                              # noqa: E402
from hetgraph import build                                            # noqa: E402


def main():
    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    g = {k: np.concatenate([d[s][k] for s in ("train", "val", "test")])
         for k in ("Xc", "Acc", "Acf", "mask", "y")}
    m = g["mask"] > 0
    stance = g["Xc"][:, :, 0]                       # +1 MALIGNANT, -1 BENIGN
    gold = np.where(g["y"][:, None] == 1, 1.0, -1.0) * np.ones_like(stance)
    correct = (stance == gold)

    print(f"claims: {int(m.sum())}   correct stance: {correct[m].mean():.1%}\n")

    att = np.maximum(0.0, -g["Acc"])
    n_att = att.sum(1)                              # attacks received
    print(f"{'':22s} {'correct':>9s} {'wrong':>9s} {'diff':>8s}")
    print(f"{'attacks received':22s} {n_att[m & correct].mean():9.3f} "
          f"{n_att[m & ~correct].mean():9.3f} "
          f"{n_att[m & correct].mean() - n_att[m & ~correct].mean():+8.3f}")
    for k in (1, 2, 3):
        r = credibility(g["Acc"], g["mask"], k)
        a, b = r[m & correct].mean(), r[m & ~correct].mean()
        # AUC of credibility as a classifier of "is this claim correct"
        u = auc(list(r[m & correct]), list(r[m & ~correct]))
        print(f"{'credibility k=' + str(k):22s} {a:9.3f} {b:9.3f} {a - b:+8.3f}"
              f"    AUC(credibility -> correct) {u:.4f}")

    print("\nIf that AUC sits at 0.500, being attacked is independent of being wrong,")
    print("and propagation over the attack graph is a null by construction.")

    # is the attack graph even about the CONTENT, or just about turn order?
    r1 = np.maximum(0.0, -g["Acc"]).sum(1)
    rnd = g["Xc"][:, :, 2] * 0 + g["Xc"][:, :, 3] * 1 + g["Xc"][:, :, 4] * 2
    print(f"\nattacks received by round: " + "  ".join(
        f"r{i}={r1[m & (rnd == i)].mean():.2f}" for i in (0, 1, 2)))
    opp = (stance[:, :, None] != stance[:, None, :])
    both = (m[:, :, None] & m[:, None, :])
    e = g["Acc"] != 0
    print(f"edges between OPPOSING stances: {(e & opp & both).sum() / max(1, (e & both).sum()):.1%}"
          f"   (chance {(opp & both).sum() / max(1, both.sum()):.1%})")


if __name__ == "__main__":
    main()
