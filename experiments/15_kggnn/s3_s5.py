"""S5 (calibration-layer Laplacian) and S3 (uncertainty-gated debate expert).

S5. The earlier Laplacian test penalised the READOUT weights and CV set lambda
to zero. This is a distinct hypothesis: penalise the per-finding CALIBRATION
(a scale and bias per finding, 2F parameters) so that KG-adjacent same-stance
findings share statistical strength, while the readout stays free. Features are
never mixed.

S3. The diagnostic fired but with the sign REVERSED: the debate correlates with
the probe residual where probes are most CONFIDENT (+0.260) and not at all where
they are least confident (-0.015). The spec predicted the opposite. So the gate
coefficient is left free-signed and what it learns is reported, rather than
assuming the motivating story.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc, kg_findings                     # noqa: E402
from features import build                                            # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank, full_eval          # noqa: E402
from kg_graph import build_adjacency                                  # noqa: E402
from kg_tensor import evidence_score, fit_evidence                    # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"
TAGS = ["probe", "probeneg", "probep3", "probep4"]


def fit_calib(X, y, F, nb, L, lam, l2, epochs=2000, lr=0.3):
    """Per-finding scale/bias (shared across phrasing blocks) + free readout.

    The Laplacian penalty lam * (a'La + c'Lc) acts ONLY on the calibration
    parameters, so KG-adjacent same-stance findings borrow calibration strength
    while every finding keeps its own readout weight.
    """
    n, d = X.shape
    a = np.ones(F); c = np.zeros(F); w = np.zeros(d); b = 0.0
    idx = np.tile(np.arange(F), nb)
    step = lr / (1.0 + lam * np.abs(L).max() + l2)
    for _ in range(epochs):
        Z = X * a[idx] + c[idx]
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        e = p - y
        gw = (Z.T @ e) / n + l2 * w
        gz = np.outer(e, w) / n
        ga = np.zeros(F); gc = np.zeros(F)
        np.add.at(ga, idx, (gz * X).sum(0))
        np.add.at(gc, idx, gz.sum(0))
        w -= step * gw
        b -= step * float(e.mean())
        a -= step * (ga + lam * (L @ a))
        c -= step * (gc + lam * (L @ c))
    return a, c, w, b, idx


def main():
    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings(); F = len(findings)
    prior = np.array([s for _, s in findings])
    B = {}
    for t in TAGS:
        rows, Xt, Y, _ = build(runs, tag=t)
        B[t] = {s: Xt[s][:, :, 0] for s in SPLITS}
    X = {s: np.hstack([B[t][s] for t in TAGS]) for s in SPLITS}
    nb = len(TAGS)
    mu, sd = X["train"].mean(0), X["train"].std(0); sd = np.where(sd < 1e-9, 1, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    res = {}

    A, _ = build_adjacency(_KG, [f for f, _ in findings])
    same = (prior[:, None] * prior[None, :]) > 0
    As = A * same - A * (~same)
    Lap = np.diag(np.abs(As).sum(1)) - As

    print("=== S5: Laplacian on the CALIBRATION layer only ===")
    y = Y["train"]
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(len(y)), 5)
    best = None
    for lam in (0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        for l2 in (1e-2, 1e-1, 3e-1):
            sc = np.zeros(len(y))
            for fo in folds:
                m = np.zeros(len(y), bool); m[fo] = True
                a, c, w, b, idx = fit_calib(Z["train"][~m], y[~m], F, nb, Lap, lam, l2)
                sc[m] = (Z["train"][m] * a[idx] + c[idx]) @ w + b
            cv = auc(list(sc[y == 1]), list(sc[y == 0]))
            if best is None or cv > best[0]:
                best = (cv, lam, l2)
    cv, lam, l2 = best
    a, c, w, b, idx = fit_calib(Z["train"], y, F, nb, Lap, lam, l2)
    s = {k: (Z[k] * a[idx] + c[idx]) @ w + b for k in SPLITS}
    at = auc(list(s["test"][Y["test"] == 1]), list(s["test"][Y["test"] == 0]))
    sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    bb = bacc(s["test"].tolist(), Y["test"].tolist(), t)
    print(f"  CV-selected lambda {lam:g}, l2 {l2:g}   cv {cv:.4f}  TEST {at:.4f}  bAcc {bb:.4f}")
    print("  -> " + ("KILLED: CV selects lambda=0, the KG calibration prior is rejected"
                     if lam == 0 else "KG calibration prior selected; check vs test"))
    res["S5"] = {"lambda": lam, "l2": l2, "cv": cv, "test": at, "bacc": bb}

    print("\n=== S3: uncertainty-gated debate expert (free-sign gate) ===")
    ev_w = fit_evidence(rows["train"])
    deb = {s: np.array([evidence_score(r["claims"], ev_w) for r in rows[s]]) for s in SPLITS}
    P = {s: np.mean([B[t][s] for t in TAGS], axis=0) for s in SPLITS}
    unc = {s: 1.0 - 2 * np.abs(P[s] - 0.5).mean(1) for s in SPLITS}
    um, us = unc["train"].mean(), unc["train"].std() or 1
    unc = {s: (unc[s] - um) / us for s in SPLITS}
    dm, ds = deb["train"].mean(), deb["train"].std() or 1
    dz = {s: (deb[s] - dm) / ds for s in SPLITS}
    bestf = max(((cv_score(Z["train"], Y["train"], l2, fit_rank)[0], l2) for l2 in L2S))
    _, l2f = bestf
    aa, ab, asc = full_eval(Z, Y, l2f, fit_rank)
    print(f"  anchor probe-64      cv {bestf[0]:.4f}  TEST {aa['test']:.4f}  bAcc {ab:.4f}")
    bs = None
    for g in np.arange(-0.6, 0.61, 0.05):
        for aq in (-1.0, -0.5, 0.0, 0.5, 1.0):
            sc = {s: asc[s] + g * (1 / (1 + np.exp(-(aq * unc[s])))) * dz[s] for s in SPLITS}
            cv = auc(list(sc["train"][Y["train"] == 1]), list(sc["train"][Y["train"] == 0]))
            if bs is None or cv > bs[0]:
                bs = (cv, g, aq, sc)
    cv, g, aq, sc = bs
    att = auc(list(sc["test"][Y["test"] == 1]), list(sc["test"][Y["test"] == 0]))
    sf = np.concatenate([sc["train"], sc["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    bb2 = bacc(sc["test"].tolist(), Y["test"].tolist(), t)
    print(f"  gated (train-fit g={g:+.2f}, slope={aq:+.1f})  train-cv {cv:.4f}  "
          f"TEST {att:.4f}  bAcc {bb2:.4f}")
    print("  -> " + ("gate weight -> 0; S3 KILLED" if abs(g) < 0.05
                     else f"gate retained, delta TEST {att-aa['test']:+.4f} "
                          f"bAcc {bb2-ab:+.4f}"))
    res["S3"] = {"gamma": g, "slope": aq, "test": att, "bacc": bb2,
                 "anchor_test": aa["test"], "anchor_bacc": ab}
    Path("experiments/15_kggnn/results/s3_s5.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
