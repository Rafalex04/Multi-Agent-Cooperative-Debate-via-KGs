"""Is the combination gain real, or 156-sample noise?

The test split has 42 malignant and 114 benign images. Differences of 0.01 AUC
on that are easy to manufacture by chance, and the ledger already contains one
case (3-run 0.7694 -> 4-run 0.7506) where a swing that size was almost certainly
noise. So the paired bootstrap is reported for every claim of improvement:
resample test samples with replacement, recompute BOTH scores on the SAME
resample, and record how often the challenger beats the incumbent.

Paired matters. The two readouts share the images, so their errors are
correlated, and an unpaired comparison would overstate the uncertainty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import SPLITS, auc, bacc, by_split, kg_findings, load     # noqa: E402
from combine import birads_scores, zfit                                # noqa: E402
from kg_tensor import evidence_score, fit_evidence, mal_share          # noqa: E402

_D = Path(__file__).resolve().parents[2] / "breastMnist/data/breast"


def main(B=4000):
    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    by = by_split(load(runs, kg_findings()))
    ev_w = fit_evidence(by["train"])
    bir = birads_scores()

    col = {s: {"ev": [], "bi": [], "ms": []} for s in SPLITS}
    Y = {}
    for s in SPLITS:
        ys = []
        for r in by[s]:
            b = bir.get((s, int(r["sid"])))
            if b is None:
                continue
            col[s]["ev"].append(evidence_score(r["claims"], ev_w))
            col[s]["ms"].append(mal_share(r["claims"]))
            col[s]["bi"].append(b)
            ys.append(r["y"])
        Y[s] = np.array(ys, dtype=float)

    Z = {}
    for k in ("ev", "bi", "ms"):
        m, sd = zfit(col["train"][k])
        Z[k] = {s: (np.array(col[s][k]) - m) / sd for s in SPLITS}

    y = Y["test"]
    cand = {
        "evidence (incumbent)": Z["ev"]["test"],
        "birads": Z["bi"]["test"],
        "evidence+birads": (Z["ev"]["test"] + Z["bi"]["test"]) / 2,
        "all three": (Z["ev"]["test"] + Z["bi"]["test"] + Z["ms"]["test"]) / 3,
    }
    base = cand["evidence (incumbent)"]

    print(f"test n={len(y)}  ({int(y.sum())} malignant, {int((1-y).sum())} benign)")
    print(f"\n  {'candidate':24s} {'AUC':>7s}  {'vs incumbent':>13s}  "
          f"{'95% CI of delta':>22s}  {'P(better)':>9s}")
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    for name, sc in cand.items():
        a = auc(list(sc[y == 1]), list(sc[y == 0]))
        if name.startswith("evidence (in"):
            print(f"  {name:24s} {a:7.4f}  {'-':>13s}  {'-':>22s}  {'-':>9s}")
            continue
        d = []
        for ii in idx:
            yy = y[ii]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            d.append(auc(list(sc[ii][yy == 1]), list(sc[ii][yy == 0]))
                     - auc(list(base[ii][yy == 1]), list(base[ii][yy == 0])))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name:24s} {a:7.4f}  {a - auc(list(base[y==1]),list(base[y==0])):+13.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {(d > 0).mean():9.3f}")


if __name__ == "__main__":
    main()
