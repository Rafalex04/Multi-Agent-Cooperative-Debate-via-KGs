"""E2 -- do the probes see what a radiologist sees?

Every probe number in this project so far is an agreement with the LABEL. That
cannot distinguish a probe that genuinely detects spiculated margins from one
that has learned some other correlate of malignancy and answers "yes" to the
spiculation question whenever the mass looks bad. BrEaST annotates the findings
themselves, so the two can finally be separated.

Three quantities:

  perception   AUC of each probe against the radiologist's annotation OF THAT
               DESCRIPTOR. Nothing is fitted. This is the first direct test of
               whether the probes are perceptually faithful.
  ceiling      the radiologist's own descriptors pushed through the SAME
               KG-signed sum that the probes are pushed through. If perfect
               perception with this inference rule scores well, the rule is
               sound and perception is the bottleneck; if it scores poorly, the
               ontology's stance mapping is the bottleneck and better probes
               cannot help.
  actual       the probes through that same rule, on the same images.

The gap between `actual` and `ceiling` is perception error; the gap between
`ceiling` and 1.0 is inference error. That decomposition is the single most
useful number for anyone extending this work, because it says which half to fix.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
from analyse_e1 import auc                                            # noqa: E402
from claims import bacc, kg_findings                                  # noqa: E402

TAGS = ("brst_p2", "brst_p3")


def load(names):
    per = {}
    for ti, t in enumerate(TAGS):
        for f in sorted((_HERE / "results").glob(f"{t}_*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                per.setdefault(int(r["index"]), {})[ti] = r["p_yes"] or {}
    out = {}
    for i, d in per.items():
        if len(d) != len(TAGS):
            continue
        M = np.full((len(names), len(TAGS)), 0.5)
        for ti, v in d.items():
            for j, n in enumerate(names):
                if v.get(n) is not None:
                    M[j, ti] = float(v[n])
        out[i] = M
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    probes = load(names)
    z = np.load(_HERE.parents[1] / "data/external/breast_pad2_224.npz")
    dn = list(z["descriptor_names"])
    D = z["descriptors"]                       # n x len(dn), NaN where unannotated
    y_all = (z["labels"][:, 0] == 0).astype(float)
    idx = sorted(probes)
    print(f"BrEaST: {len(idx)}/{len(y_all)} images probed in both arms")
    if len(idx) < 60:
        print("not enough yet"); return
    P = np.array([probes[i] for i in idx])     # n x F x 2
    Pm = P.mean(2)                             # pool the two phrasings
    y = y_all[idx]
    res = {"n": len(idx)}

    print("\n=== perception: probe vs radiologist annotation of the same descriptor ===")
    print(f"{'finding':46s} {'n+/n':>10s} {'AUC':>7s}")
    per = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or col[m].sum() < 5 or col[m].sum() == m.sum():
            continue
        a = auc(col[m], Pm[m, j])
        per[f] = {"auc": a, "n_pos": int(col[m].sum()), "n": int(m.sum())}
        print(f"{f:46s} {int(col[m].sum()):4d}/{int(m.sum()):4d} {a:7.4f}")
    if per:
        v = [d["auc"] for d in per.values()]
        print(f"{'MEAN':46s} {'':>10s} {np.mean(v):7.4f}   "
              f"({sum(1 for x in v if x > 0.6)}/{len(v)} above 0.60, "
              f"{sum(1 for x in v if x < 0.5)}/{len(v)} below chance)")
    res["perception"] = per

    print("\n=== perception -> inference decomposition (KG-signed sum, nothing fitted) ===")
    zscore = (Pm - Pm.mean(0)) / np.where(Pm.std(0) < 1e-9, 1, Pm.std(0))
    actual = zscore @ prior
    # ceiling: the radiologist's own descriptors through the identical rule
    cols = [dn.index(f) for f in names if f in dn]
    sub = [j for j, f in enumerate(names) if f in dn]
    R = D[np.ix_(idx, cols)]
    keep = ~np.isnan(R).any(1)
    ceil_sc = R[keep] @ prior[sub]
    act_sub = zscore[np.ix_(keep, sub)] @ prior[sub]
    a_act_all = auc(y, actual)
    a_act = auc(y[keep], act_sub)
    a_ceil = auc(y[keep], ceil_sc)
    print(f"  probes,      all 16 findings   n={len(y):4d}   AUC {a_act_all:.4f}")
    print(f"  probes,      {len(sub):2d} annotated ones  n={int(keep.sum()):4d}   AUC {a_act:.4f}")
    print(f"  RADIOLOGIST, {len(sub):2d} annotated ones  n={int(keep.sum()):4d}   AUC {a_ceil:.4f}   <- ceiling")
    print(f"\n  perception error (ceiling - probes): {a_ceil - a_act:+.4f}")
    print(f"  inference  error (1.0 - ceiling):    {1.0 - a_ceil:+.4f}")
    verdict = ("PERCEPTION is the bottleneck: the inference rule on perfect inputs "
               "beats the probes by a clear margin" if a_ceil - a_act > 0.05 else
               "INFERENCE is the bottleneck: even perfect descriptors scored through "
               "the ontology's stance mapping do not do much better than the probes")
    print(f"  -> {verdict}")
    res["decomposition"] = {"probes_all": a_act_all, "probes_sub": a_act,
                            "radiologist_ceiling": a_ceil,
                            "perception_error": a_ceil - a_act,
                            "inference_error": 1.0 - a_ceil, "n_complete": int(keep.sum())}

    print("\n=== E4 on BrEaST: agreement with radiologist BI-RADS ===")
    br = z["birads"][idx]
    order = {"2": 2, "3": 3, "4a": 4.1, "4b": 4.2, "4c": 4.3, "5": 5}
    bv = np.array([order.get(str(b), np.nan) for b in br])
    m = ~np.isnan(bv)
    hi = np.isin([str(b) for b in br], ["4a", "4b", "4c", "5"])
    print(f"  AUC(BI-RADS 4/5 vs 2/3) from KG-signed probe sum: {auc(hi[m].astype(float), actual[m]):.4f}")
    res["E4_breast"] = {"auc_birads45": auc(hi[m].astype(float), actual[m])}

    (_HERE / "results/e2_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/e2_analysis.json")


if __name__ == "__main__":
    main()
