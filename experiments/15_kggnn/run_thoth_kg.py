"""Fit ThothKG: sign-separated message passing over the KG finding graph.

The unsigned propagation test showed KG propagation LOSING to no propagation
(0.7703 against 0.7751), and the adjacency says why: the strongest KG links join
findings of OPPOSITE polarity, because a lesion description names both what is
present and what is absent. `circumscribed_margin` (benign) is tied to
`posterior_shadowing` (malignant) at weight 0.897. Averaging across that edge
mixes opposing evidence and dilutes it.

Sign separation is the fix and it is the one learned component the Stage 4
ablation said earned its parameters. Split the adjacency by whether an edge joins
findings the KG gives the SAME stance or OPPOSING stances, and give each its own
weight matrix, so the model can aggregate agreement and contrast differently
instead of averaging them together.

Arms:
  signed      A+ and A- separate            (the architecture)
  unsigned    one matrix over A+ + A-       (ablates sign separation)
  no-prop     A+ = A- = 0                   (ablates the KG topology entirely)
  shuffled    signed, but node identities permuted, so the KG edges are wrong
              while density and degree are preserved -- the honest control for
              "is it the ontology or just having edges"
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc                                  # noqa: E402
from features import build                                            # noqa: E402
from kg_graph import build_adjacency, normalised                      # noqa: E402
from kg_tensor import anchor_of, fit_evidence                         # noqa: E402
from thoth_kg import ThothKG, fit                                     # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def split_sign(A, prior):
    """A+ joins findings of equal KG stance, A- joins opposing stances."""
    same = (prior[:, None] * prior[None, :]) > 0
    return A * same, A * (~same)


def evaluate(m, X, anchors, Y):
    s = {k: m.forward(X[k], anchors[k])[0] for k in SPLITS}
    a = {k: auc(list(s[k][Y[k] == 1]), list(s[k][Y[k] == 0])) for k in SPLITS}
    sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    return a, bacc(s["test"].tolist(), Y["test"].tolist(), t), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--node-id", action="store_true",
                    help="append a one-hot node identity to every node's features")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, X, Y, findings = build(args.debates)
    prior = np.array([s for _, s in findings], dtype=float)
    F, C = len(findings), X["train"].shape[2]
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS)
          + f" | nodes {F} feats {C}")

    mu, sd = X["train"].mean((0, 1)), X["train"].std((0, 1))
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xz = {s: (X[s] - mu) / sd for s in SPLITS}
    if args.node_id:
        # A single shared W0 forces every finding through the same transform, so
        # the network cannot learn that `spiculated` and `oval_shape` deserve
        # different treatment -- which is why the flat linear model over the same
        # features beat it. A one-hot identity restores per-node parameters while
        # leaving the message passing intact.
        eye = np.eye(F)
        Xz = {s: np.concatenate([Xz[s], np.broadcast_to(eye, (len(Xz[s]), F, F))], axis=2)
              for s in SPLITS}
        C = C + F
        print(f"node identity appended -> feats {C}")

    ev_w = fit_evidence(rows["train"])
    anchors = {s: anchor_of(rows[s], "evidence", ev_w, False) for s in SPLITS}
    a_te = anchors["test"]
    print(f"anchor alone TEST {auc(list(a_te[Y['test']==1]), list(a_te[Y['test']==0])):.4f}")

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    Ap, An = split_sign(normalised(A, self_loops=False), prior)
    print(f"KG edges {info['n_edges']}/{info['n_pairs']}  "
          f"same-stance {int((Ap>0).sum()//2)}  opposing {int((An>0).sum()//2)}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(F)
    As, An_s = Ap[np.ix_(perm, perm)], An[np.ix_(perm, perm)]
    Z = np.zeros((F, F))
    arms = {"signed (KG)": (Ap, An),
            "unsigned (KG)": (Ap + An, Z),
            "shuffled KG": (As, An_s),
            "no-prop": (Z, Z)}

    res = {}
    print(f"\n  {'arm':16s} {'TEST':>16s} {'bAcc':>16s}  {'val':>7s} {'gamma':>7s}")
    for name, (P, N) in arms.items():
        ts, bs, vs, gs = [], [], [], []
        for seed in range(args.seeds):
            best = None
            for l2 in (1e-3, 1e-2, 1e-1):
                m = ThothKG(C, args.hidden, P, N, seed=seed, gamma0=0.0)
                fit(m, Xz["train"], anchors["train"], Y["train"], l2, args.epochs)
                a, bb, _ = evaluate(m, Xz, anchors, Y)
                if best is None or a["val"] > best[0]:
                    best = (a["val"], a["test"], bb, m.gamma)
            vs.append(best[0]); ts.append(best[1]); bs.append(best[2]); gs.append(best[3])
        print(f"  {name:16s} {np.mean(ts):.4f}+-{np.std(ts):.4f} "
              f"{np.mean(bs):.4f}+-{np.std(bs):.4f}  {np.mean(vs):7.4f} {np.mean(gs):+7.3f}")
        res[name] = {"test": float(np.mean(ts)), "test_sd": float(np.std(ts)),
                     "bacc": float(np.mean(bs)), "bacc_sd": float(np.std(bs)),
                     "val": float(np.mean(vs)), "gamma": float(np.mean(gs))}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
