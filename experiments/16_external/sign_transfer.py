"""Fix the inverted probes using radiologist agreement, not labels.

E2 showed that five of thirteen probes score BELOW chance against the
radiologist's annotation of the very descriptor they ask about -- `oval_shape` at
0.2745, `irregular_shape` at 0.3352. That is not blindness, it is a sign error:
an AUC of 0.27 flipped is 0.73. The model can see the feature and is answering
the question backwards.

That suggests a correction which costs no labels at all. For each finding, ask
BrEaST's radiologist annotations which way the probe points, and flip the ones
that point the wrong way. The correction is estimated from **descriptor
agreement on BrEaST** and applied to **malignancy prediction on BUS-BRA**:
different dataset, different country, different task, and no malignancy label
touched anywhere in the fitting. The zero-parameter KG-signed sum stays
zero-parameter with respect to the outcome.

Two controls, because a 13-way sign search will find something in noise:
  random signs   the same number of flips placed at random (20 draws)
  label-fitted   signs taken from BreastMNIST train labels instead, which is the
                 ordinary supervised thing and shows what the descriptor route
                 gives up
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
from analyse_e1 import auc, bacc, load_probes                         # noqa: E402
from analyse_e2 import load as load_breast                            # noqa: E402
from claims import kg_findings                                        # noqa: E402


def main():
    F = json.loads((_HERE / "results/frozen_models.json").read_text())
    names, prior = F["findings"], np.array(F["prior"], float)
    mu, sd = np.array(F["mu"]), np.array(F["sd"])

    # ---- estimate the sign on BrEaST, from descriptor agreement only ---------
    zb = np.load(_HERE.parents[1] / "data/external/breast_pad2_224.npz")
    dn = list(zb["descriptor_names"]); D = zb["descriptors"]
    bp = load_breast(names)
    bidx = sorted(bp)
    Pb = np.array([bp[i] for i in bidx]).mean(2)          # pool phrasings
    flip = np.ones(len(names))
    detail = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[bidx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not (0 < col[m].sum() < m.sum()):
            continue
        a = auc(col[m], Pb[m, j])
        detail[f] = a
        if a < 0.5:
            flip[j] = -1.0
    n_flip = int((flip < 0).sum())
    print(f"signs estimated on BrEaST descriptor agreement: {n_flip} of "
          f"{len(detail)} annotated findings flipped")
    for f, a in sorted(detail.items(), key=lambda kv: kv[1]):
        print(f"   {f:46s} {a:.4f} {'FLIP' if a < 0.5 else ''}")

    # ---- apply to BUS-BRA, per the frozen standardisation --------------------
    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _HERE / "results")
    z = np.load(_HERE.parents[1] / "data/external/busbra_pad2_224.npz")
    idx = sorted(probes)
    X = np.array([probes[i].T.reshape(-1) for i in idx])
    y = (z["labels"][:, 0] == 0).astype(int)[idx]
    cases = z["cases"][idx]
    Z = (X - mu) / sd
    u = np.unique(cases)
    y_case = np.array([y[cases == c][0] for c in u])

    def score(fl):
        w = np.tile(prior * fl, 2) / (len(names) * 2)
        s = Z @ w
        return np.array([s[cases == c].mean() for c in u]), s

    base_c, base_i = score(np.ones(len(names)))
    fix_c, fix_i = score(flip)
    a0, a1 = auc(y_case, base_c), auc(y_case, fix_c)
    print(f"\nBUS-BRA, 0-parameter KG-signed sum (case level, n={len(u)})")
    print(f"  as registered                    {a0:.4f}")
    print(f"  sign-corrected from BrEaST       {a1:.4f}   delta {a1-a0:+.4f}")

    rng = np.random.default_rng(0)
    rnd = []
    for _ in range(20):
        fl = np.ones(len(names))
        fl[rng.choice(len(names), n_flip, replace=False)] = -1.0
        rnd.append(auc(y_case, score(fl)[0]))
    print(f"  {n_flip} RANDOM flips (20 draws)          "
          f"{np.mean(rnd):.4f} +- {np.std(rnd):.4f}"
          f"   -> real is {(a1-np.mean(rnd))/(np.std(rnd)+1e-9):+.2f} sd better")

    # what a label-fitted sign would give, for reference
    from freeze_models import breastmnist_matrix
    Xb, Yb = breastmnist_matrix([(n, p) for n, p in zip(names, prior)])
    Zb = (Xb["train"] - mu) / sd
    lab_flip = np.ones(len(names))
    for j in range(len(names)):
        col = Zb[:, j] + Zb[:, j + len(names)]
        lab_flip[j] = 1.0 if (auc(Yb["train"], col * prior[j]) >= 0.5) else -1.0
    a2 = auc(y_case, score(lab_flip)[0])
    agree = int((lab_flip == flip).sum())
    print(f"  signs from BreastMNIST LABELS    {a2:.4f}   delta {a2-a0:+.4f}"
          f"   (agrees with the descriptor route on {agree}/{len(names)})")

    out = {"n_flipped": n_flip, "perception_auc": detail,
           "base_case": a0, "sign_corrected_case": a1, "delta": a1 - a0,
           "random_mean": float(np.mean(rnd)), "random_sd": float(np.std(rnd)),
           "label_fitted_case": a2, "sign_agreement": agree,
           "flip": {n: float(f) for n, f in zip(names, flip)}}
    (_HERE / "results/sign_transfer.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote results/sign_transfer.json")


if __name__ == "__main__":
    main()
