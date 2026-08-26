"""Is a probe inverted, or is ONE OF ITS TWO WORDINGS inverted?

sign_audit.py pooled P2 and P3 before measuring, which can only report the
average of the two. That matters for what the repair should be. If both wordings
of `irregular_shape` anti-correlate with the radiologist, the question itself is
broken and dropping it is right. If P2 is aligned and P3 is reversed, then
nothing is broken except the pooling, and the repair is to SELECT the wording per
finding -- strictly better than dropping, because it keeps the signal.

Selection would then be fitted on BrEaST descriptors and tested on BUS-BRA
malignancy, the same separation sign_audit.py uses.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from analyse_e1 import auc, load_probes                                # noqa: E402
from analyse_e2 import load as load_breast_probes                      # noqa: E402
from claims import kg_findings                                         # noqa: E402
from sign_audit import zscore                                          # noqa: E402
from sign_control import auc_many, case_reduce                         # noqa: E402

_DATA = _HERE.parents[1] / "data/external"
PH = ("P2 (negated)", "P3 (verification)")


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)

    probes = load_breast_probes(names)
    z = np.load(_DATA / "breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]
    idx = sorted(probes)
    P = np.array([probes[i] for i in idx])              # n x F x 2

    print(f"per-phrasing perception on BrEaST (n={len(idx)})")
    print(f"{'finding':46s} {PH[0]:>14s} {PH[1]:>16s} {'pooled':>9s}   agree?")
    rows = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
            continue
        a = [auc(col[m], P[m, j, t]) for t in range(2)]
        ap = auc(col[m], P[m, j].mean(1))
        same = (a[0] - .5) * (a[1] - .5) > 0
        rows[f] = {"p2": a[0], "p3": a[1], "pooled": ap, "consistent": bool(same)}
        print(f"{f:46s} {a[0]:14.4f} {a[1]:16.4f} {ap:9.4f}   "
              f"{'yes' if same else 'NO -- wordings disagree'}")
    incons = [f for f, r in rows.items() if not r["consistent"]]
    print(f"\nwordings disagree on {len(incons)}/{len(rows)} findings: {incons}")

    # ---- the repair this suggests ------------------------------------------
    # KEEP the readings that agree with the radiologist, average them, and drop
    # the ones that do not. A first attempt selected by |AUC - 0.5| and then
    # flipped, on the reasoning that an anti-correlated probe still carries
    # signal; that picks the 0.19 reading of `irregular_shape` over the 0.75 one
    # and transfers badly (0.5832 case AUC). Anti-correlation on 252 images is
    # not a reliable enough basis to invert a channel -- agreement is.
    #
    # Every decision is a bit read off BrEaST descriptors, never off malignancy.
    keepw = np.ones((len(names), 2))
    for j, f in enumerate(names):
        r = rows.get(f)
        if r is None:
            continue                       # unannotated: no evidence, keep both
        for t, k in enumerate(("p2", "p3")):
            if r[k] < 0.5:
                keepw[j, t] = 0.0
    dead = [names[j] for j in range(len(names)) if keepw[j].sum() == 0]
    onew = [names[j] for j in range(len(names)) if keepw[j].sum() == 1]
    print(f"\nwordings kept: {int(keepw.sum())}/{2*len(names)}")
    print(f"  both wordings dropped ({len(dead)}): {dead}")
    print(f"  one wording dropped  ({len(onew)}): {onew}")

    ext = load_probes(names, ("busbra_p2", "busbra_p3"),
                      _HERE.parents[0] / "16_external/results")
    zb = np.load(_DATA / "busbra_pad2_224.npz", allow_pickle=True)
    ei = sorted(ext)
    E = np.array([ext[i] for i in ei])                   # n x F x 2
    y = (zb["labels"][ei, 0] == 0).astype(float)
    cs = zb["cases"][ei]

    def case_auc(M, w):
        ZC, ys = case_reduce(M, y, cs)
        return float(auc_many(ys, ZC @ np.asarray(w).reshape(-1, 1)))

    pooled = E.mean(2)
    w = keepw / np.where(keepw.sum(1, keepdims=True) == 0, 1, keepw.sum(1, keepdims=True))
    kept = (E * w[None]).sum(2)                     # weighted mean of surviving wordings
    prior_kept = prior * (keepw.sum(1) > 0)         # a finding with no wording left is out
    drop5 = prior * np.array([0.0 if (f in rows and rows[f]["pooled"] < .5) else 1.0
                              for f in names])
    print("\nBUS-BRA transfer (biopsy label; every choice made on BrEaST descriptors)")
    out = {
        "pooled, KG as written":        case_auc(pooled, prior),
        "pooled, drop 5 inverted":      case_auc(pooled, drop5),
        "keep agreeing wordings":       case_auc(kept, prior_kept),
    }
    for k, v in out.items():
        print(f"  {k:34s} {v:.4f}")

    (_HERE / "results/per_phrasing.json").write_text(json.dumps(
        {"per_finding": rows, "inconsistent": incons,
         "keep_wordings": keepw.tolist(), "dead": dead, "one_wording": onew,
         "busbra": out}, indent=1, default=float))
    print("\nwrote results/per_phrasing.json")


if __name__ == "__main__":
    main()
