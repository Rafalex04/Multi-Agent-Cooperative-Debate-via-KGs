"""S5 decided properly: same code path, only the penalty matrix changes.

The first S5 run compared a calibration model fitted by fit_calib against an
anchor fitted by fit_rank. Different optimisers, so the delta confounded "the KG
prior helps" with "this code path differs". Here every arm goes through
fit_calib with an identical CV sweep and the ONLY thing that changes is L:

    lambda = 0     no penalty at all -- the honest no-KG control
    KG signed      KG-adjacent same-stance findings share calibration
    KG unsigned    adjacency without the stance split
    shuffled KG    same matrix, node identities permuted (5 draws)

If the KG arm does not beat both lambda=0 and the shuffled distribution, the
ontology is not the thing helping.
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
from kg_graph import build_adjacency                                  # noqa: E402
from s3_s5 import fit_calib                                           # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"
TAGS = ["probe", "probeneg", "probep3", "probep4"]
LAMS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)
L2S = (1e-2, 1e-1, 3e-1)


def lap(A, prior, signed=True):
    if signed:
        same = (prior[:, None] * prior[None, :]) > 0
        As = A * same - A * (~same)
    else:
        As = A.copy()
    return np.diag(np.abs(As).sum(1)) - As


def run(Z, Y, F, nb, L, lams, tag, res):
    y = Y["train"]
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(y)), 5)
    best = None
    for lam in lams:
        for l2 in L2S:
            sc = np.zeros(len(y))
            for fo in folds:
                m = np.zeros(len(y), bool); m[fo] = True
                a, c, w, b, idx = fit_calib(Z["train"][~m], y[~m], F, nb, L, lam, l2)
                sc[m] = (Z["train"][m] * a[idx] + c[idx]) @ w + b
            cv = auc(list(sc[y == 1]), list(sc[y == 0]))
            if best is None or cv > best[0]:
                best = (cv, lam, l2)
    cv, lam, l2 = best
    a, c, w, b, idx = fit_calib(Z["train"], y, F, nb, L, lam, l2)
    s = {k: (Z[k] * a[idx] + c[idx]) @ w + b for k in SPLITS}
    at = auc(list(s["test"][Y["test"] == 1]), list(s["test"][Y["test"] == 0]))
    av = auc(list(s["val"][Y["val"] == 1]), list(s["val"][Y["val"] == 0]))
    sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    bb = bacc(s["test"].tolist(), Y["test"].tolist(), t)
    print(f"  {tag:24s} cv {cv:.4f}  TEST {at:.4f}  bAcc {bb:.4f}   "
          f"(lam {lam:g}, l2 {l2:g}, val {av:.4f})")
    res[tag] = {"cv": cv, "test": at, "bacc": bb, "val": av, "lam": lam, "l2": l2}
    return cv, at, bb, s


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
    mu, sd = X["train"].mean(0), X["train"].std(0); sd = np.where(sd < 1e-9, 1, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    nb = len(TAGS)
    A, _ = build_adjacency(_KG, [f for f, _ in findings])
    Zero = np.zeros((F, F))
    res = {}
    print("all arms fitted by the SAME fit_calib path; only L changes\n")
    _, t0, b0, s0 = run(Z, Y, F, nb, Zero, (0.0,), "lambda = 0 (no KG)", res)
    ck, tk, bk, sk = run(Z, Y, F, nb, lap(A, prior, True), LAMS, "KG signed", res)
    run(Z, Y, F, nb, lap(A, prior, False), LAMS, "KG unsigned", res)
    print()
    sh = []
    rng = np.random.default_rng(7)
    for k in range(5):
        p = rng.permutation(F)
        sh.append(run(Z, Y, F, nb, lap(A[np.ix_(p, p)], prior[p], True), LAMS,
                      f"shuffled KG #{k}", res))
    scv = np.array([x[0] for x in sh]); ste = np.array([x[1] for x in sh])
    sbb = np.array([x[2] for x in sh])
    print(f"\n  shuffled: cv {scv.mean():.4f}+-{scv.std():.4f}  "
          f"TEST {ste.mean():.4f}+-{ste.std():.4f}  bAcc {sbb.mean():.4f}+-{sbb.std():.4f}")
    print(f"  KG signed vs lambda=0 : TEST {tk-t0:+.4f}  bAcc {bk-b0:+.4f}")
    print(f"  KG signed vs shuffled : TEST {tk-ste.mean():+.4f} "
          f"({(tk-ste.mean())/max(ste.std(),1e-9):+.1f} sd)  "
          f"bAcc {bk-sbb.mean():+.4f} ({(bk-sbb.mean())/max(sbb.std(),1e-9):+.1f} sd)")
    res["shuffled"] = {"cv": float(scv.mean()), "test": float(ste.mean()),
                       "test_sd": float(ste.std()), "bacc": float(sbb.mean()),
                       "bacc_sd": float(sbb.std())}

    y = Y["test"]; rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(4000)]
    d = []
    for ii in idx:
        yy = y[ii]
        if yy.sum() in (0, len(yy)):
            continue
        d.append(auc(list(sk["test"][ii][yy == 1]), list(sk["test"][ii][yy == 0]))
                 - auc(list(s0["test"][ii][yy == 1]), list(s0["test"][ii][yy == 0])))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  bootstrap KG-signed vs lambda=0: {tk-t0:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(better) {(d>0).mean():.3f}")
    res["bootstrap"] = {"delta": tk - t0, "lo": float(lo), "hi": float(hi),
                        "p_better": float((d > 0).mean())}
    Path("experiments/15_kggnn/results/s5_control.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
