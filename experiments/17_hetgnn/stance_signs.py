"""Two checks on the debate's two surviving channels.

CHECK 1 -- is net stance below chance because of a SIGN ERROR?
Net stance through the KG prior sits at 0.4842 while argument mass reaches
0.6347, and a systematically inverted channel usually means a sign problem. Five
KG signs were repaired elsewhere (0.6919 -> 0.7223 on the probes).

There is a distinction that decides what a positive result would mean. The
repair was diagnosed on the PROBES: those five probes anti-correlate with the
radiologist's annotation of the descriptor they name. That says the probe reads
the finding backwards. It does not say the ONTOLOGY assigns the finding to the
wrong side. Net stance uses no probe at all -- it is agents citing findings and
taking a position -- so:

  if the repair lifts net stance too, the two share a cause and the ontology's
    stance assignment is what is wrong;
  if it does not, the ontology is fine, only the probes were backwards, and the
    "debate direction carries nothing" claim survives the obvious objection.

The decisive form is not the repair but the ORACLE: the best of all 2^16 sign
vectors, chosen on the very labels being scored. If even that leaves net stance
near chance, no sign assignment can rescue the channel.

CHECK 2 -- is argument mass just a noisier probe?
Mass at finding f is how much the agents talked about f. The probe at f is
p(f is present). If those are strongly correlated then mass is a degraded copy
of something already in the vector and the debate stage contributes nothing new.
If they are not, mass is the one debate-derived channel that is both independent
and above chance, and it is the only surviving argument for running debates.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _d in ("14_kgtensor", "15_kggnn", "16_external", "20_perception"):
    sys.path.insert(0, str(_HERE.parents[1] / _d))
from analyse_e1 import auc, load_probes, paired_bootstrap                # noqa: E402
from claims import kg_findings                                           # noqa: E402
from hetgraph import load_sample                                         # noqa: E402
from sign_control import auc_many, case_reduce                           # noqa: E402

_DEB = _HERE.parents[2] / "breastMnist/data/breast/debates_busbra/all"
_EXT = _HERE.parents[1] / "16_external/results"
INVERTED = ("irregular_shape", "echogenic_pseudocapsule", "oval_shape",
            "thin_uniform_pseudocapsule", "echogenic_rind")


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[o] = np.arange(len(v))
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    F = len(names)

    deb = {}
    for f in sorted(glob.glob(str(_DEB / "*.json"))):
        try:
            X, Acc, Acf, mask, y, sid = load_sample(f, names)
        except Exception:
            continue
        deb[int(sid)] = (X, Acf, mask)
    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(deb) & set(probes))
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    y = (z["labels"][common, 0] == 0).astype(float)
    cases = z["cases"][common]
    n = len(common)

    MASS = np.zeros((n, F)); NET = np.zeros((n, F)); P3 = np.zeros((n, F))
    for k, i in enumerate(common):
        Xc, Acf, m = deb[i]
        MASS[k] = (m[:, None] * Acf).sum(0)
        NET[k] = ((m * Xc[:, 0])[:, None] * Acf).sum(0)
        P3[k] = probes[i][:, 1]
    print(f"complete corpus: {n} images / {len(np.unique(cases))} cases\n")

    keep = np.array([0.0 if f in INVERTED else 1.0 for f in names])
    flip = np.array([-1.0 if f in INVERTED else 1.0 for f in names])

    def case_auc_of(M, w):
        ZC, yc = case_reduce(M, y, cases)
        return float(auc_many(yc, ZC @ np.asarray(w, float).reshape(-1, 1))[0])

    print("=== CHECK 1: net stance under the repaired KG signs ===")
    print(f"{'channel':26s} {'KG as written':>14s} {'drop 5':>9s} {'flip 5':>9s}")
    for lab, M in (("net stance", NET), ("argument mass", MASS),
                   ("probe P3", P3)):
        print(f"{lab:26s} {case_auc_of(M, prior):14.4f} "
              f"{case_auc_of(M, prior*keep):9.4f} {case_auc_of(M, prior*flip):9.4f}")

    # oracle over every sign vector, chosen on the labels being scored
    Mn = np.arange(1 << F)
    A = np.ones((F, len(Mn)))
    for b in range(F):
        A[b] = 1.0 - 2.0 * ((Mn >> b) & 1)
    print("\n  oracle over all 2^16 sign vectors, fitted on the scored labels:")
    for lab, M in (("net stance", NET), ("argument mass", MASS)):
        ZC, yc = case_reduce(M, y, cases)
        v = auc_many(yc, ZC @ (A * np.ones((F, 1))))
        print(f"    {lab:16s} best {v.max():.4f}   worst {v.min():.4f}   "
              f"KG-as-written sits at the {100*(v < case_auc_of(M, prior)).mean():.1f}th pct")

    print("\n=== CHECK 2: is argument mass a noisier probe? ===")
    print(f"{'finding':46s} {'rho(mass, p3)':>14s} {'AUC mass':>9s} {'AUC p3':>8s}")
    rhos = []
    for j, f in enumerate(names):
        r = spearman(MASS[:, j], P3[:, j])
        rhos.append(r)
        print(f"{f:46s} {r:14.4f} {auc(y, MASS[:, j]):9.4f} {auc(y, P3[:, j]):8.4f}")
    print(f"{'MEAN |rho|':46s} {np.nanmean(np.abs(rhos)):14.4f}")

    ZC_m, yc = case_reduce(MASS, y, cases)
    ZC_p, _ = case_reduce(P3, y, cases)
    a_m, a_p = case_auc_of(MASS, prior), case_auc_of(P3, prior)
    B = np.stack([(ZC_p @ prior), (ZC_m @ prior)], 1)
    B = (B - B.mean(0)) / B.std(0)
    w = np.linalg.lstsq(B, yc - yc.mean(), rcond=None)[0]
    print(f"\n  probe P3 alone                  {a_p:.4f}")
    print(f"  argument mass alone             {a_m:.4f}")
    print(f"  both, weights fitted on the scored cases   {auc(yc, B @ w):.4f}"
          f"   (+{auc(yc, B @ w) - a_p:.4f} over the probe)")

    (_HERE.parent / "results/stance_signs.json").write_text(json.dumps(
        {"n": n, "net_stance": {"kg": case_auc_of(NET, prior),
                                "drop5": case_auc_of(NET, prior*keep),
                                "flip5": case_auc_of(NET, prior*flip)},
         "mass": {"kg": a_m, "drop5": case_auc_of(MASS, prior*keep)},
         "mean_abs_rho_mass_probe": float(np.nanmean(np.abs(rhos))),
         "combined_optimistic": float(auc(yc, B @ w))}, indent=1, default=float))
    print("\nwrote results/stance_signs.json")


if __name__ == "__main__":
    main()
