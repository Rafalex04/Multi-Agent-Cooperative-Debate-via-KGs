"""Does rebuttal propagation on the claim graph carry information a mean does not?

The cheapest decisive question before building anything. Score a sample by the
credibility-weighted stance of its claims, aggregated through the citation edges
onto findings, and compare k=0 (every claim equally credible -- i.e. the plain
mean that every previous version of this pipeline used) against k=1,2,3 rounds of
gradual-argumentation propagation.

Credibility follows the h-categoriser of gradual argumentation semantics: a claim
is worth less when credible claims attack it, and the attackers' own credibility
is settled by the same rule, which is what makes it a propagation rather than a
count of incoming arrows.

    r_j = (1 + delta * endorse_j) / (1 + sum_i attack_ij * r_i)

If k>0 does not beat k=0 here, the debate graph's topology carries nothing beyond
its degree statistics and no architecture over it will help.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import auc                                                # noqa: E402
from hetgraph import SPLITS, build                                    # noqa: E402


def credibility(Acc, mask, k, delta=0.0):
    """h-categoriser credibility after k propagation rounds. k=0 -> all ones."""
    att = np.maximum(0.0, -Acc)          # att[i,j]: i attacks j
    end = np.maximum(0.0, Acc)
    r = mask.copy()
    for _ in range(k):
        inc_att = np.einsum("nij,ni->nj", att, r)
        inc_end = np.einsum("nij,ni->nj", end, r)
        r = mask * (1.0 + delta * inc_end) / (1.0 + inc_att)
    return r


def finding_evidence(g, r, channel):
    """Push claim evidence along citation edges onto findings.

    Two channels, and they must not be multiplied together -- doing so squares the
    sign and turns a benign claim about a benign finding into malignant evidence:

      stance    what the AGENT concluded, credibility weighted. The debate's own
                verdict, localised to the finding it was arguing about.
      attention how much credible argument a finding attracted at all, regardless
                of what was said. The KG prior then reads that mass.
    """
    w = r * (g["Xc"][:, :, 0] if channel == "stance" else 1.0)
    e = np.einsum("nc,ncf->nf", w, g["Acf"])
    return e / np.maximum(1.0, g["Acf"].sum(1))


def main():
    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    for channel, wgt, tag in (("stance", np.ones(len(prior)), "debate verdict, summed over findings"),
                              ("stance", prior, "debate verdict x KG prior (sign-squared: expect junk)"),
                              ("attention", prior, "argument mass read by the KG prior")):
        print(f"\n--- {tag} ---")
        print(f"{'k':>2s} {'delta':>6s} " + "".join(f"{s:>9s}" for s in SPLITS))
        for k in (0, 1, 2, 3):
            for delta in ((0.0,) if k == 0 else (0.0, 1.0)):
                row = []
                for s in SPLITS:
                    g = d[s]
                    r = credibility(g["Acc"], g["mask"], k, delta)
                    sc = finding_evidence(g, r, channel) @ wgt
                    row.append(auc(list(sc[g["y"] == 1]), list(sc[g["y"] == 0])))
                print(f"{k:2d} {delta:6.1f} " + "".join(f"{v:9.4f}" for v in row))

    # how much does propagation actually MOVE the per-claim weights?
    g = d["train"]
    print()
    for k in (1, 2, 3):
        rk = credibility(g["Acc"], g["mask"], k)
        m = g["mask"] > 0
        frac = float((np.abs(rk[m] - 1.0) > 1e-9).mean())
        print(f"k={k}: credibility range [{rk[m].min():.3f},{rk[m].max():.3f}] "
              f"mean {rk[m].mean():.3f} sd {rk[m].std():.3f}  moved off 1.0: {frac:.1%}")


if __name__ == "__main__":
    main()
