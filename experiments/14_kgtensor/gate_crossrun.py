"""Gate 2: does per-finding cross-run reproducibility carry signal?

gate.py tested pairwise products of POOLED per-finding counts and found nothing.
That does not close section 4.1, because pooling destroys exactly the quantity
that section is about: a pooled count cannot distinguish one run flagging a
finding twice from two independent runs flagging it once each. Run-to-run
mal_share correlation is only 0.308, so agreement across independent debates is
genuinely different information -- and it is the plan's strongest remaining
argument for a graph model at all.

So test it directly, without building the model:

    base       pooled per-finding shares + mal_share            (additive)
    +crossrun  base plus, for each finding, the fraction of runs
               that mentioned it and the spread of its stance across runs

If reproducibility carried signal a graph could exploit, adding these must beat
base under the same CV protocol. Same positive control as gate.py applies -- the
protocol is already known to detect a real effect at +0.19.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import auc, by_split, kg_findings, load                   # noqa: E402
from gate import base_feats, cv_auc                                   # noqa: E402


def crossrun_feats(claims, F, nruns):
    """Per finding: how many runs mentioned it, and how much they disagreed.

    `presence` is reproducibility of the observation; `spread` is reproducibility
    of its interpretation. Both are zero for a finding only one run raised, which
    is the case a pooled count cannot tell apart from a repeated one.
    """
    per = defaultdict(lambda: defaultdict(list))
    for c in claims:
        if c["finding"] >= 0:
            per[c["finding"]][c["run"]].append(c["stance"])
    pres = np.zeros(F); spread = np.zeros(F)
    for f, runs in per.items():
        pres[f] = len(runs) / max(1, nruns)
        means = [np.mean(v) for v in runs.values()]
        spread[f] = float(np.std(means)) if len(means) > 1 else 0.0
    return np.concatenate([pres, spread])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    findings = kg_findings()
    F = len(findings)
    nruns = len(args.debates)
    by = by_split(load(args.debates, findings))
    tr = by["train"]
    y = np.array([r["y"] for r in tr], dtype=float)

    Xb = np.array([base_feats(r["claims"], F) for r in tr])
    Xc = np.array([np.concatenate([base_feats(r["claims"], F),
                                   crossrun_feats(r["claims"], F, nruns)]) for r in tr])
    print(f"train graphs {len(y)}  runs {nruns}")
    print(f"  base        {Xb.shape[1]:3d} features")
    print(f"  +crossrun   {Xc.shape[1]:3d} features")

    L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
    print(f"\n  {'l2':>8s}   {'base':>8s}   {'+crossrun':>10s}")
    rb = rc = -1
    for l2 in L2S:
        a, b = cv_auc(Xb, y, l2), cv_auc(Xc, y, l2)
        rb, rc = max(rb, a), max(rc, b)
        print(f"  {l2:8g}   {a:8.4f}   {b:10.4f}")
    print(f"\n  best base      CV AUC {rb:.4f}")
    print(f"  best +crossrun CV AUC {rc:.4f}")
    print(f"  delta          {rc - rb:+.4f}")
    print("\n  -> cross-run reproducibility carries signal; section 4.1 is justified"
          if rc > rb + 0.005 else
          "\n  -> cross-run reproducibility adds nothing beyond the pooled count")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"base_cv": rb, "crossrun_cv": rc, "delta": rc - rb}, indent=1))


if __name__ == "__main__":
    main()
