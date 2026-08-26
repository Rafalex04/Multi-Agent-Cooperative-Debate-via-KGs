"""The negated phrasing arm is anti-predictive, and it is half of every probe.

Every probe number in this project is an average over two phrasings of the same
question -- P2 negates the finding, P3 asks the model to verify it -- pooled
because the 2x2 factorial study found a large acquiescence bias (0.346 vs 0.608)
and averaging a negated arm against a direct one cancels it. That is true and it
is also beside the point: the negated arm cancels the yes-bias by inverting the
CONTENT along with it.

The selection chain, in the order it was actually run, each step blind to the
next:

  1. BrEaST descriptors. Mean AUC against the radiologist's annotation of the
     descriptor each probe names: P2 0.4882 (chance), P3 0.6531. P2 is inverted
     on 6 of 13 findings, P3 on 3. This alone picks P3, on a dataset and a label
     that appear nowhere else in the chain.
  2. BreastMNIST test, malignancy. P3 0.7467, P2 0.5478, pooled 0.7270.
  3. BUS-BRA, malignancy, 1064 cases. P3 0.7499, P2 0.4452, pooled 0.6919.

Steps 2 and 3 are confirmations of a choice already made at step 1, not the
choice itself.

The sharpest single number: the negated arm alone scores 0.4452 externally,
BELOW chance, while carrying half the probe budget of the whole project.

The second sharpest: P3 through the zero-parameter KG-signed sum transfers with
essentially no gap -- 0.7467 in domain, 0.7499 out -- which no fitted model in
this project does.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
sys.path.insert(0, str(_HERE))
from analyse_e1 import auc, bacc, load_probes, paired_bootstrap        # noqa: E402
from analyse_e2 import load as load_breast_probes                     # noqa: E402
from claims import kg_findings                                        # noqa: E402
from freeze_models import breastmnist_matrix                          # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                     # noqa: E402
from sign_audit import zscore                                         # noqa: E402

_DATA = _HERE.parents[1] / "data/external"
SP = ("train", "val", "test")
ARMS = ("P2 negated", "P3 verification", "pooled")


def descriptor_faithfulness(names):
    """Step 1: how well does each phrasing arm see what it claims to see?"""
    probes = load_breast_probes(names)
    z = np.load(_DATA / "breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]
    idx = sorted(probes)
    P = np.array([probes[i] for i in idx])
    per = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
            continue
        per[f] = [float(auc(col[m], P[m, j, t])) for t in range(2)]
    return per


def busbra(names):
    ext = load_probes(names, ("busbra_p2", "busbra_p3"),
                      _HERE.parents[0] / "16_external/results")
    z = np.load(_DATA / "busbra_pad2_224.npz", allow_pickle=True)
    ei = sorted(ext)
    return np.array([ext[i] for i in ei]), \
        (z["labels"][ei, 0] == 0).astype(float), z["cases"][ei]


def case_view(y, s, c):
    u = np.unique(c)
    return np.array([y[c == k][0] for k in u]), np.array([s[c == k].mean() for k in u])


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    F = len(names)
    prior = np.array([s for _, s in findings], float)
    res = {}

    per = descriptor_faithfulness(names)
    a = np.array(list(per.values()))
    print("=== 1. BrEaST: agreement with the radiologist's own descriptor ===")
    for t, lab in enumerate(ARMS[:2]):
        print(f"  {lab:18s} mean AUC {a[:, t].mean():.4f}   "
              f"inverted on {(a[:, t] < .5).sum()}/{len(a)} findings")
    print(f"  -> this step alone selects {'P3' if a[:,1].mean() > a[:,0].mean() else 'P2'}, "
          f"using no malignancy label anywhere")
    res["breast_descriptor"] = {"per_finding": per,
                                "mean": {ARMS[t]: float(a[:, t].mean()) for t in range(2)},
                                "n_inverted": {ARMS[t]: int((a[:, t] < .5).sum())
                                               for t in range(2)}}

    X, Y = breastmnist_matrix(findings)
    E, y_e, cases = busbra(names)
    sel = {"P2 negated": [slice(0, F)], "P3 verification": [slice(F, 2 * F)],
           "pooled": [slice(0, F), slice(F, 2 * F)]}
    esel = {"P2 negated": E[:, :, 0], "P3 verification": E[:, :, 1],
            "pooled": E.mean(2)}

    print("\n=== 2+3. malignancy, ZERO fitted parameters (KG-signed sum) ===")
    print(f"{'arm':20s} {'BreastMNIST 156':>16s} {'BUS-BRA 1064 cases':>20s} {'gap':>8s}")
    res["zero_param"] = {}
    for lab in ARMS:
        ind = auc(Y["test"], zscore(np.mean([X["test"][:, s] for s in sel[lab]], 0)) @ prior)
        yc, sc = case_view(y_e, zscore(esel[lab]) @ prior, cases)
        print(f"{lab:20s} {ind:16.4f} {auc(yc, sc):20.4f} {ind - auc(yc, sc):+8.4f}")
        res["zero_param"][lab] = {"indomain": ind, "ext_case": auc(yc, sc),
                                  "ext_bacc": bacc(yc, sc, np.median(sc)),
                                  "gap": ind - auc(yc, sc)}

    print("\n=== the same three arms with a FITTED readout, frozen on BreastMNIST train ===")
    print(f"{'arm':20s} {'params':>7s} {'BreastMNIST':>12s} {'BUS-BRA case':>13s} {'bAcc':>7s} {'gap':>8s}")
    res["fitted"] = {}
    keepc = {}
    for lab in ARMS:
        cols = [i for s in sel[lab] for i in range(*s.indices(2 * F))]
        Xb = {s: X[s][:, cols] for s in SP}
        mu, sd = Xb["train"].mean(0), Xb["train"].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = {s: (Xb[s] - mu) / sd for s in SP}
        best = None
        for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
            for l2 in L2S:
                c = cv_score(Z["train"], Y["train"], l2, f)[0]
                if best is None or c > best[0]:
                    best = (c, l2, f)
        _, l2, f = best
        w, b = f(Z["train"], Y["train"], l2)
        ind = auc(Y["test"], Z["test"] @ w + b)
        Xe = E.transpose(0, 2, 1).reshape(len(E), -1)[:, cols]
        yc, sc = case_view(y_e, ((Xe - mu) / sd) @ w + b, cases)
        keepc[lab] = sc
        print(f"{lab:20s} {len(cols):7d} {ind:12.4f} {auc(yc, sc):13.4f} "
              f"{bacc(yc, sc, np.median(sc)):7.4f} {ind - auc(yc, sc):+8.4f}")
        res["fitted"][lab] = {"n_params": len(cols), "indomain": ind,
                              "ext_case": auc(yc, sc),
                              "ext_bacc": bacc(yc, sc, np.median(sc)),
                              "gap": ind - auc(yc, sc)}

    yc, _ = case_view(y_e, esel["pooled"] @ prior, cases)
    p_fit = paired_bootstrap(yc, keepc["P3 verification"], keepc["pooled"])
    _, s_zero_p3 = case_view(y_e, zscore(esel["P3 verification"]) @ prior, cases)
    p_zero = paired_bootstrap(yc, s_zero_p3, keepc["pooled"])
    print(f"\n  P3 vs pooled, fitted readout, external : P(better) {p_fit:.3f}")
    print(f"  ZERO-PARAMETER P3 vs the 32-param pooled fitted model, external: "
          f"P(better) {p_zero:.3f}")
    res["P_p3_beats_pooled_fitted"] = float(p_fit)
    res["P_zeroparam_p3_beats_fitted_pooled"] = float(p_zero)

    (_HERE / "results/phrasing_arms.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/phrasing_arms.json")


if __name__ == "__main__":
    main()
