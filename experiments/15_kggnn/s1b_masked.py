"""S1b -- ontology-masked pairwise conjunctions, evaluated under the full protocol.

G0 found the only interaction signal this project has seen: products of probe
values restricted to finding pairs adjacent in the lesion graph beat both no
products and ALL products. S1 tried the higher-order version of that idea (a
lesion presents up to seven findings) and improved CV without moving test, so
the pairwise level is where the signal actually sits.

This evaluates it properly. The control that matters is not "all pairs" -- that
changes the parameter count as well as the mask -- but a SHUFFLED mask with
exactly the same number of pairs, so density and dimensionality are held fixed
and only the identity of the pairs changes. Five shuffles, so the comparison is
against a distribution rather than one draw.

Reported: CV, one test evaluation for the CV-selected configuration, and a
paired bootstrap against the flat anchor on the same test split.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc, kg_findings                     # noqa: E402
from features import build                                            # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank, full_eval          # noqa: E402
from kg_graph import build_adjacency                                  # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def products(P, pairs):
    return {s: np.column_stack([P[s][:, i] * P[s][:, j] for i, j in pairs]) for s in SPLITS}


def evaluate(X, Y, label, res, losses=("ce", "rank")):
    best = None
    for lname, fitter in (("ce", fit_ce), ("rank", fit_rank)):
        if lname not in losses:
            continue
        for l2 in L2S:
            cv, _ = cv_score(X["train"], Y["train"], l2, fitter)
            if best is None or cv > best[0]:
                best = (cv, l2, fitter, lname)
    cv, l2, fitter, lname = best
    a, bb, sc = full_eval(X, Y, l2, fitter)
    print(f"  {label:30s} cv {cv:.4f}  TEST {a['test']:.4f}  bAcc {bb:.4f}   "
          f"({lname}, l2 {l2}, feats {X['train'].shape[1]})")
    res[label] = {"cv": cv, "test": a["test"], "bacc": bb, "val": a["val"],
                  "loss": lname, "l2": l2}
    return sc, cv, a["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["probe", "probeneg"])
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings()
    blocks = []
    for tag in args.tags:
        rows, Xt, Y, _ = build(runs, tag=tag)
        blocks.append({s: Xt[s][:, :, 0] for s in SPLITS})
    X0 = {s: np.hstack([b[s] for b in blocks]) for s in SPLITS}
    P = {s: np.mean([b[s] for b in blocks], axis=0) for s in SPLITS}
    F = len(findings)
    print(f"phrasings {len(blocks)}  anchor features {X0['train'].shape[1]}")

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    allp = list(itertools.combinations(range(F), 2))
    kgp = [(i, j) for i, j in allp if A[i, j] > 0]
    print(f"KG-adjacent pairs {len(kgp)} of {len(allp)}\n")

    res = {}
    base_sc, base_cv, base_te = evaluate(X0, Y, "base (no products)", res)
    kg_pr = products(P, kgp)
    kg_sc, kg_cv, kg_te = evaluate({s: np.hstack([X0[s], kg_pr[s]]) for s in SPLITS},
                                   Y, "+ KG-masked products", res)
    all_pr = products(P, allp)
    evaluate({s: np.hstack([X0[s], all_pr[s]]) for s in SPLITS}, Y, "+ ALL products", res)

    print()
    sh_cv, sh_te = [], []
    for k in range(args.shuffles):
        rng = np.random.default_rng(100 + k)
        pick = [allp[i] for i in rng.choice(len(allp), len(kgp), replace=False)]
        pr = products(P, pick)
        _, c, t = evaluate({s: np.hstack([X0[s], pr[s]]) for s in SPLITS}, Y,
                           f"+ shuffled mask #{k}", res)
        sh_cv.append(c); sh_te.append(t)
    print(f"\n  shuffled-mask control: cv {np.mean(sh_cv):.4f}+-{np.std(sh_cv):.4f}   "
          f"TEST {np.mean(sh_te):.4f}+-{np.std(sh_te):.4f}")
    print(f"  KG mask vs shuffled:   cv {kg_cv-np.mean(sh_cv):+.4f}   "
          f"TEST {kg_te-np.mean(sh_te):+.4f}")
    res["shuffled_mean"] = {"cv": float(np.mean(sh_cv)), "test": float(np.mean(sh_te)),
                            "cv_sd": float(np.std(sh_cv)), "test_sd": float(np.std(sh_te))}

    # ---------------- paired bootstrap vs the flat anchor ----------------
    y = Y["test"]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(4000)]
    d = []
    for ii in idx:
        yy = y[ii]
        if yy.sum() in (0, len(yy)):
            continue
        d.append(auc(list(kg_sc["test"][ii][yy == 1]), list(kg_sc["test"][ii][yy == 0]))
                 - auc(list(base_sc["test"][ii][yy == 1]), list(base_sc["test"][ii][yy == 0])))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  paired bootstrap, KG-masked vs base: {kg_te-base_te:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(better) {(d>0).mean():.3f}")
    res["bootstrap"] = {"delta": kg_te - base_te, "lo": float(lo), "hi": float(hi),
                        "p_better": float((d > 0).mean())}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
