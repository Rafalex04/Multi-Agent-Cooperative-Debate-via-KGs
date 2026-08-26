"""Is the perception gap a resolution problem or a WORDING problem?

E2 reported one number -- probes 0.7020 against a radiologist ceiling of 0.8515,
a perception gap of 0.1495 -- and the obvious reading was that the model cannot
see the findings well enough. The per-finding table says something different and
much more actionable.

BrEaST annotates 13 of the 16 findings, but not with 13 independent columns.
Three of the probes ask about the SAME radiologist column:

    irregular_shape  ==  spiculated_or_irregular_mass     (identical, n+=140)
    oval_shape       ==  1 - irregular_shape              (complement)
    echogenic_rind   ==  echogenic_pseudocapsule  ==  thin_uniform_pseudocapsule

so the spread WITHIN each group is a measurement of nothing but phrasing. It is
large. On identical ground truth the two shape wordings score 0.3352 and 0.7769.
A model that could not see mass shape would score 0.5 on both.

So the probes are not uniformly blind. Several of them are POINTED THE WRONG WAY:
five of thirteen sit below chance, and the KG then multiplies each one by a fixed
+/-1 stance, which means an inverted probe is subtracted when it should be added.

THE EXPERIMENT. Learn one bit per finding -- is this probe aligned or inverted --
from BrEaST DESCRIPTOR agreement, and test whether that transfers to MALIGNANCY
on two other datasets.

The separation is the point. Signs are chosen against radiologist descriptors on
BrEaST; they are scored against biopsy labels on BUS-BRA and BreastMNIST. A
different label on a different dataset, so a gain cannot be selection.

PREDICTION, recorded before running the transfer. The shape and halo probes are
inverted on BrEaST by a wide margin and those are 5 of 16 channels, so I expect
the correction to help: +0.02 to +0.05 external case AUC. The honest risk is that
the inversion is a BrEaST annotation convention rather than a property of the
probe, in which case transfer is flat or negative and the wording story is wrong.
Either outcome is reportable; a null here kills the repair, not the diagnosis.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from analyse_e1 import auc, bacc, load_probes, paired_bootstrap        # noqa: E402
from analyse_e2 import load as load_breast_probes                      # noqa: E402
from claims import kg_findings                                         # noqa: E402
from freeze_models import breastmnist_matrix                           # noqa: E402

_DATA = _HERE.parents[1] / "data/external"


def zscore(P):
    s = P.std(0)
    return (P - P.mean(0)) / np.where(s < 1e-9, 1.0, s)


def descriptor_signs(names):
    """One bit per finding, fitted ONLY on BrEaST radiologist descriptors.

    Returns (sign, table). sign[f] = -1 when the probe's answer anti-correlates
    with the radiologist's annotation of the descriptor it names -- i.e. the
    probe measures the finding but reports it with the wrong polarity.
    Findings BrEaST does not annotate keep +1: no evidence, no change.
    """
    probes = load_breast_probes(names)
    z = np.load(_DATA / "breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]
    idx = sorted(probes)
    Pm = np.array([probes[i] for i in idx]).mean(2)          # pool P2 and P3

    sign = np.ones(len(names))
    table = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
            continue
        a = auc(col[m], Pm[m, j])
        sign[j] = 1.0 if a >= 0.5 else -1.0
        table[f] = {"auc": float(a), "sign": float(sign[j]),
                    "n_pos": int(col[m].sum()), "n": int(m.sum())}
    return sign, table, dn, D, idx, Pm


def wording_groups(names, table, dn, D):
    """Probes that ask about the same radiologist column, so their spread is phrasing."""
    groups, seen = [], set()
    ann = [f for f in names if f in table]
    for i, a in enumerate(ann):
        if a in seen:
            continue
        g = [a]
        for b in ann[i + 1:]:
            x, y = D[:, dn.index(a)], D[:, dn.index(b)]
            if np.array_equal(x, y):
                g.append(b); seen.add(b)
            elif np.array_equal(x, 1 - y):
                g.append(b + " (complement)"); seen.add(b)
        seen.add(a)
        if len(g) > 1:
            groups.append(g)
    return groups


def evaluate(tag, y, score, groupby=None):
    out = {"n": int(len(y)), "auc": float(auc(y, score)),
           "bacc": float(bacc(y, score, np.median(score)))}
    if groupby is not None:
        u = np.unique(groupby)
        ys = np.array([y[groupby == c][0] for c in u])
        ss = np.array([score[groupby == c].mean() for c in u])
        out.update(n_cases=int(len(u)), auc_case=float(auc(ys, ss)),
                   bacc_case=float(bacc(ys, ss, np.median(ss))))
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)

    sign, table, dn, D, bidx, bPm = descriptor_signs(names)
    print(f"=== per-finding perception on BrEaST (n={len(bidx)}) ===")
    print(f"{'finding':46s} {'n+/n':>10s} {'AUC':>7s}  sign")
    for f in names:
        t = table.get(f)
        if t is None:
            print(f"{f:46s} {'--':>10s} {'--':>7s}   +1  (not annotated)")
        else:
            print(f"{f:46s} {t['n_pos']:4d}/{t['n']:4d} {t['auc']:7.4f}  {int(t['sign']):+d}")
    flipped = [f for f in names if table.get(f, {}).get("sign", 1) < 0]
    print(f"\ninverted: {len(flipped)}/{len(table)} annotated findings -> {flipped}")

    print("\n=== wording sensitivity: probes sharing one radiologist column ===")
    groups = wording_groups(names, table, dn, D)
    for g in groups:
        vals = []
        for m in g:
            f = m.replace(" (complement)", "")
            a = table[f]["auc"]
            a = 1 - a if m.endswith("(complement)") else a
            vals.append(a)
            print(f"  {m:52s} {a:7.4f}")
        print(f"  {'SPREAD (same ground truth, wording only)':52s} {max(vals)-min(vals):7.4f}\n")

    res = {"perception": table, "inverted": flipped,
           "wording_spread": [{"group": g,
                               "spread": float(max(v) - min(v))}
                              for g in groups
                              for v in [[(1 - table[m.replace(' (complement)','')]['auc'])
                                         if m.endswith('(complement)')
                                         else table[m.replace(' (complement)','')]['auc']
                                         for m in g]]]}

    # ---------------- transfer: BUS-BRA, biopsy label, never seen by the fit ----
    print("=== TRANSFER 1 / BUS-BRA (biopsy label; signs fitted on BrEaST descriptors) ===")
    probes = load_probes(names, ("busbra_p2", "busbra_p3"),
                         _HERE.parents[0] / "16_external/results")
    z = np.load(_DATA / "busbra_pad2_224.npz", allow_pickle=True)
    idx = sorted(probes)
    P = np.array([probes[i] for i in idx]).mean(2)
    y = (z["labels"][idx, 0] == 0).astype(float)
    cases = z["cases"][idx]
    Z = zscore(P)
    base, corr = Z @ prior, Z @ (prior * sign)
    res["busbra"] = {"baseline": evaluate("busbra", y, base, cases),
                     "corrected": evaluate("busbra", y, corr, cases)}
    u = np.unique(cases)
    ys = np.array([y[cases == c][0] for c in u])
    cs = lambda v: np.array([v[cases == c].mean() for c in u])
    p = paired_bootstrap(ys, cs(corr), cs(base))
    res["busbra"]["P_corrected_better"] = float(p)
    for k in ("baseline", "corrected"):
        r = res["busbra"][k]
        print(f"  {k:10s} image AUC {r['auc']:.4f}  case AUC {r['auc_case']:.4f}  "
              f"case bacc {r['bacc_case']:.4f}")
    d = res["busbra"]["corrected"]["auc_case"] - res["busbra"]["baseline"]["auc_case"]
    print(f"  delta (case AUC) {d:+.4f}   P(corrected better) = {p:.3f}   "
          f"n={res['busbra']['baseline']['n']} images / {res['busbra']['baseline']['n_cases']} cases")

    # ---------------- transfer: BreastMNIST test, the in-domain 156 -------------
    print("\n=== TRANSFER 2 / BreastMNIST test (biopsy label) ===")
    X, Y = breastmnist_matrix(findings)
    F = len(names)
    Xt, yt = X["test"], Y["test"]
    Pt = np.stack([Xt[:, :F], Xt[:, F:]], 2).mean(2)
    Zt = zscore(Pt)
    b2, c2 = Zt @ prior, Zt @ (prior * sign)
    res["breastmnist"] = {"baseline": evaluate("bm", yt, b2),
                          "corrected": evaluate("bm", yt, c2),
                          "P_corrected_better": float(paired_bootstrap(yt, c2, b2))}
    for k in ("baseline", "corrected"):
        r = res["breastmnist"][k]
        print(f"  {k:10s} AUC {r['auc']:.4f}  bacc {r['bacc']:.4f}")
    print(f"  delta {res['breastmnist']['corrected']['auc']-res['breastmnist']['baseline']['auc']:+.4f}"
          f"   P = {res['breastmnist']['P_corrected_better']:.3f}   n={len(yt)}")

    (_HERE / "results/sign_audit.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/sign_audit.json")


if __name__ == "__main__":
    main()
