"""Sign-separated KG propagation over probe-measured finding nodes.

The trained logistic on the 16 raw probes reaches 0.7909 / 0.7713, better than
the entire debate apparatus. So the question this answers is narrow and fair:
given those measurements sitting on KG finding nodes, does propagating them
along the ontology's own edges add anything?

Propagation is sign-separated, because the unsigned test failed for a legible
reason -- the strongest KG links join findings of OPPOSITE polarity (a lesion
description names both what is present and what is absent), so a single averaging
matrix mixes opposing evidence. A+ and A- get separate blocks:

    features = [X, A+ X, A- X]      (and A+^2, A-^2 at K=2)

Controls, all with matched density and degree:
    identity   no propagation at all
    shuffled   the same signed matrices with node identities permuted, so the
               edges are wrong but the graph statistics are right
    complete   every pair connected

If `signed KG` does not beat `shuffled`, the ontology is not doing the work.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc                                  # noqa: E402
from features import COLS, build                                      # noqa: E402
from kg_graph import build_adjacency, normalised                      # noqa: E402
from kg_gnn import fit_eval                                           # noqa: E402
from kg_tensor import anchor_of, fit_evidence                         # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"
L2S = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def prop_signed(X, Ap, An, K):
    outs, cur = [X], X
    for _ in range(K):
        p = np.einsum("ij,njc->nic", Ap, cur)
        n = np.einsum("ij,njc->nic", An, cur)
        outs += [p, n]
        cur = p + n
    return np.concatenate(outs, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, X, Y, findings = build(args.debates)
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    ev_w = fit_evidence(rows["train"])
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    An_ = normalised(A, self_loops=False)
    same = (prior[:, None] * prior[None, :]) > 0
    Ap, Am = An_ * same, An_ * (~same)
    rng = np.random.default_rng(0); perm = rng.permutation(F)
    Sp, Sm = Ap[np.ix_(perm, perm)], Am[np.ix_(perm, perm)]
    Cp = normalised(np.ones((F, F)) - np.eye(F), self_loops=False)
    Z = np.zeros((F, F))

    # feature sets: probes only, and probes + debate
    sets = {
        "probe-only": [COLS.index("p_yes"), COLS.index("prior")],
        "probe+debate": list(range(len(COLS))),
    }
    arms = {"signed KG": (Ap, Am), "shuffled KG": (Sp, Sm),
            "complete": (Cp, Z), "identity": (Z, Z)}
    anchors = {
        "none": {s: np.zeros(len(Y[s])) for s in SPLITS},
        "evidence": {s: anchor_of(rows[s], "evidence", ev_w, False) for s in SPLITS},
    }

    res = {}
    for fname, cols in sets.items():
        Xf = {s: X[s][:, :, cols] for s in SPLITS}
        for aname, anc in anchors.items():
            print(f"\n=== features={fname}  anchor={aname} ===")
            print(f"  {'arm':12s} {'K':>2s}  {'TEST':>7s} {'bAcc':>7s}  {'cv':>7s} {'val':>7s}")
            for K in (0, 1, 2):
                for arm, (P, M) in arms.items():
                    if K == 0 and arm != "identity":
                        continue
                    Xs = {s: prop_signed(Xf[s], P, M, K).reshape(len(Xf[s]), -1)
                          for s in SPLITS}
                    at, bb, cv, av, l2, _ = fit_eval(Xs, Y, anc, L2S)
                    tag = "no-prop" if K == 0 else arm
                    print(f"  {tag:12s} {K:2d}  {at:7.4f} {bb:7.4f}  {cv:7.4f} {av:7.4f}")
                    res[f"{fname}|{aname}|{tag}|K{K}"] = {
                        "test": at, "bacc": bb, "cv": cv, "val": av, "l2": l2}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
