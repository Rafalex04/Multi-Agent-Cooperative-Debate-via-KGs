"""S0 result and the re-anchored candidate sweep.

Four probe phrasings now exist, a 2x2 of polarity (present / free-of) crossed
with framing (observe yourself / adjudicate another reader's report). This
measures what pooling them is worth, then re-runs the surviving candidate
(ontology-masked pairwise conjunctions) against the strongest anchor, with the
shuffled-mask control that decides whether the ontology or merely the
dimensionality is doing the work.
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
TAGS = ["probe", "probeneg", "probep3", "probep4"]
LAB = {"probe": "P1 direct/positive", "probeneg": "P2 direct/negative",
       "probep3": "P3 verify/positive", "probep4": "P4 verify/negative"}


def pick(X, Y, label, res, losses=("ce", "rank")):
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
    print(f"  {label:32s} cv {cv:.4f}   TEST {a['test']:.4f}  bAcc {bb:.4f}   "
          f"({lname}, l2 {l2}, {X['train'].shape[1]} feats)")
    res[label] = {"cv": cv, "test": a["test"], "bacc": bb, "val": a["val"],
                  "loss": lname, "l2": l2, "n_feats": int(X["train"].shape[1])}
    return sc, cv, a["test"], bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings()
    B = {}
    for tag in TAGS:
        rows, Xt, Y, _ = build(runs, tag=tag)
        B[tag] = {s: Xt[s][:, :, 0] for s in SPLITS}
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    res = {}
    print("\n=== S0: each phrasing alone, and pooled ===")
    for t in TAGS:
        pick(B[t], Y, LAB[t], res)
    print()
    p32 = {s: np.hstack([B["probe"][s], B["probeneg"][s]]) for s in SPLITS}
    p48 = {s: np.hstack([B[t][s] for t in TAGS[:3]]) for s in SPLITS}
    p64 = {s: np.hstack([B[t][s] for t in TAGS]) for s in SPLITS}
    a32 = pick(p32, Y, "probe-32 (P1+P2)", res)
    pick(p48, Y, "probe-48 (P1..P3)", res)
    a64 = pick(p64, Y, "probe-64 (all four)", res)

    anchors = {"probe-32": (p32, a32), "probe-64": (p64, a64)}
    best_key = max(anchors, key=lambda k: anchors[k][1][1])
    Xa, (asc, acv, ate, abb) = anchors[best_key]
    print(f"\n  CV selects {best_key} as the anchor "
          f"(cv {acv:.4f}, TEST {ate:.4f}, bAcc {abb:.4f})")

    # ---- ontology-masked conjunctions on the chosen anchor ----
    F = len(findings)
    A, _ = build_adjacency(_KG, [f for f, _ in findings])
    allp = list(itertools.combinations(range(F), 2))
    kgp = [(i, j) for i, j in allp if A[i, j] > 0]
    P = {s: np.mean([B[t][s] for t in TAGS], axis=0) for s in SPLITS}
    prod = lambda pairs: {s: np.column_stack([P[s][:, i] * P[s][:, j] for i, j in pairs])
                          for s in SPLITS}

    print(f"\n=== conjunctions on {best_key} ({len(kgp)}/{len(allp)} KG-adjacent pairs) ===")
    kg_pr = prod(kgp)
    kg_sc, kg_cv, kg_te, kg_bb = pick({s: np.hstack([Xa[s], kg_pr[s]]) for s in SPLITS},
                                      Y, "+ KG-masked products", res)
    ap_ = prod(allp)
    pick({s: np.hstack([Xa[s], ap_[s]]) for s in SPLITS}, Y, "+ ALL products", res)
    sh = []
    for k in range(args.shuffles):
        rng = np.random.default_rng(200 + k)
        pk = [allp[i] for i in rng.choice(len(allp), len(kgp), replace=False)]
        pr = prod(pk)
        sh.append(pick({s: np.hstack([Xa[s], pr[s]]) for s in SPLITS}, Y,
                       f"+ shuffled mask #{k}", res))
    scv = np.array([x[1] for x in sh]); ste = np.array([x[2] for x in sh])
    print(f"\n  shuffled control: cv {scv.mean():.4f}+-{scv.std():.4f}  "
          f"TEST {ste.mean():.4f}+-{ste.std():.4f}")
    print(f"  KG mask vs shuffled: cv {kg_cv-scv.mean():+.4f} "
          f"({(kg_cv-scv.mean())/max(scv.std(),1e-9):.1f} sd)   "
          f"TEST {kg_te-ste.mean():+.4f} ({(kg_te-ste.mean())/max(ste.std(),1e-9):.1f} sd)")
    res["shuffled"] = {"cv": float(scv.mean()), "cv_sd": float(scv.std()),
                       "test": float(ste.mean()), "test_sd": float(ste.std())}

    # ---- bootstrap: best conjunction model vs the flat anchor ----
    y = Y["test"]; rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(4000)]
    d = []
    for ii in idx:
        yy = y[ii]
        if yy.sum() in (0, len(yy)):
            continue
        d.append(auc(list(kg_sc["test"][ii][yy == 1]), list(kg_sc["test"][ii][yy == 0]))
                 - auc(list(asc["test"][ii][yy == 1]), list(asc["test"][ii][yy == 0])))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  bootstrap, +KG products vs {best_key}: {kg_te-ate:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(better) {(d>0).mean():.3f}")
    res["bootstrap"] = {"anchor": best_key, "delta": kg_te - ate, "lo": float(lo),
                        "hi": float(hi), "p_better": float((d > 0).mean())}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
