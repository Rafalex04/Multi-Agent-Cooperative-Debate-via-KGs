"""In-domain score predicts the transfer gap almost exactly, and predicts the
external score in the WRONG DIRECTION.

Round 3 found this once, between two models: ResNet-18 leads zero-shot by 0.13
AUC on BreastMNIST and trails it by 0.08 on BUS-BRA. One reversal between two
very different model classes is an anecdote. This is the same comparison run
across six feature blocks that share an architecture, a readout, a protocol and
a training split, and differ only in WHICH ZERO-SHOT CHANNELS feed them.

Every arm here is fitted on BreastMNIST train (546), selected by CV on train,
evaluated once on BreastMNIST test (156), then applied unchanged to BUS-BRA
(1875 images / 1064 cases, different country, four scanners). Nothing is refitted
externally.

The blocks were not chosen to make this point -- each was built to answer its own
question, and the ordering fell out afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# (label, params, BreastMNIST test AUC, BUS-BRA case AUC, where it came from)
BLOCKS = [
    ("BiomedCLIP contrastive",      16, 0.8745, 0.7027, "21_biomedclip/hybrid.py"),
    ("hybrid, scorer per finding",  14, 0.8241, 0.7202, "21_biomedclip/hybrid.py"),
    ("VLM both phrasings pooled",   32, 0.8133, 0.7325, "20_perception/phrasing_arms.py"),
    ("VLM negated phrasing",        16, 0.8089, 0.7171, "20_perception/phrasing_arms.py"),
    ("VLM verification phrasing",   16, 0.7822, 0.7456, "20_perception/phrasing_arms.py"),
    ("VLM verification, KG rule",    0, 0.7467, 0.7499, "20_perception/phrasing_arms.py"),
]


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[o] = np.arange(len(v))
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    ind = np.array([b[2] for b in BLOCKS])
    ext = np.array([b[3] for b in BLOCKS])
    gap = ind - ext

    print(f"{'feature block':30s} {'params':>6s} {'BreastMNIST':>11s} "
          f"{'BUS-BRA':>9s} {'gap':>8s}")
    for lab, p, i, e, _ in sorted(BLOCKS, key=lambda r: -r[2]):
        print(f"{lab:30s} {p:6d} {i:11.4f} {e:9.4f} {i - e:+8.4f}")

    print(f"\n  Spearman(in-domain, external)      {spearman(ind, ext):+.3f}")
    print(f"  Spearman(in-domain, gap)           {spearman(ind, gap):+.3f}")
    print(f"  Pearson (in-domain, gap)           {np.corrcoef(ind, gap)[0, 1]:+.3f}")
    sl, ic = np.polyfit(ind, gap, 1)
    print(f"  fit: gap = {sl:.3f} * indomain {ic:+.3f}   "
          f"-> break-even at in-domain {-ic / sl:.4f}")
    print(f"  Spearman(params, gap)              {spearman([b[1] for b in BLOCKS], gap):+.3f}")

    print("\n  The block with the BEST in-domain score has the WORST external score.")
    print("  The block with the WORST in-domain score has the BEST external score,")
    print("  and it is the one with zero fitted parameters.")

    (_HERE / "results/gap_law.json").write_text(json.dumps(
        {"blocks": [{"label": l, "params": p, "indomain": i, "external": e,
                     "gap": i - e, "source": s} for l, p, i, e, s in BLOCKS],
         "spearman_indomain_external": spearman(ind, ext),
         "pearson_indomain_gap": float(np.corrcoef(ind, gap)[0, 1]),
         "slope": float(sl), "intercept": float(ic),
         "spearman_params_gap": spearman([b[1] for b in BLOCKS], gap)},
        indent=1, default=float))
    print("\nwrote results/gap_law.json")


if __name__ == "__main__":
    main()
