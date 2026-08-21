"""ThothGNN, re-derived with the knowledge graph as the TOPOLOGY.

What changed from Stage 4, and why.

ThothGNN's nodes were CLAIMS and its edges were AGREE/DISAGREE. That failed for a
measured reason: 87% of the edges were DISAGREE and DISAGREE claims carry zero
evidence, so message passing redistributed noise. The knowledge graph appeared as
extra nodes to aggregate from, and deleting it changed nothing (NO-KG-NODES
+0.0049).

Here the nodes are the 16 BI-RADS FINDINGS and the edges come from the knowledge
graph -- two findings are adjacent when some lesion in the ontology presents
both (kg_graph.py). Each node carries what is known about that finding: a visual
probe measurement, how the debate argued it, and the KG's own stance prior. So
the KG is no longer a passive neighbour; it decides what talks to what, and the
debate graph is attached to it node by node.

Retained from the original design, because the ablation said they were the parts
that mattered: the anchored score `beta * logit(anchor) + graph term`, with the
anchor being the strongest known readout, and sign structure on the readout.

Propagation is the Simplified Graph Convolution of Wu et al. (2019): the linear
part of a GCN, with the nonlinearity between layers removed. On a fixed 16-node
graph this is exact, convex to fit, and -- the reason it is used here -- makes
the topology ablation clean, because the ONLY thing that changes between arms is
the propagation matrix.

    K = 0   no propagation (each finding stands alone)
    K = 1   one hop over the KG
    K = 2   two hops

Adjacency arms: the real KG, identity, a random graph of matched density, and
the complete graph. If the KG arm does not beat random and complete, the
ontology is decorative and this must be reported as such.
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
from kg_tensor import anchor_of, fit_evidence, predict, train_head    # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def propagate(X, Ahat, K):
    """[X, AX, ..., A^K X] concatenated on the feature axis."""
    outs, cur = [X], X
    for _ in range(K):
        cur = np.einsum("ij,njc->nic", Ahat, cur)
        outs.append(cur)
    return np.concatenate(outs, axis=2)


def random_adj(F, n_edges, seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((F, F))
    pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]
    for k in rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False):
        i, j = pairs[k]
        A[i, j] = A[j, i] = rng.random()
    return A


def fit_eval(Xs, Ys, anchors, l2s, folds=5, epochs=1500, seed=0):
    """CV-select L2 on train, refit, return (test AUC, test bAcc, cv, val)."""
    Xtr, ytr, atr = Xs["train"], Ys["train"], anchors["train"]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (Xs[s] - mu) / sd for s in SPLITS}
    rng = np.random.default_rng(seed)
    folds_ix = np.array_split(rng.permutation(len(ytr)), folds)
    best = None
    for l2 in l2s:
        sc = np.zeros(len(ytr))
        for fo in folds_ix:
            m = np.zeros(len(ytr), dtype=bool); m[fo] = True
            w, b, be = train_head(Z["train"][~m], atr[~m], ytr[~m],
                                  np.zeros(Z["train"].shape[1]), l2, epochs)
            sc[m] = predict(Z["train"][m], atr[m], w, b, be)
        c = auc(list(sc[ytr == 1]), list(sc[ytr == 0]))
        if best is None or c > best[0]:
            best = (c, l2)
    cv, l2 = best
    w, b, be = train_head(Z["train"], atr, ytr, np.zeros(Z["train"].shape[1]), l2, epochs)
    s = {k: predict(Z[k], anchors[k], w, b, be) for k in SPLITS}
    at = auc(list(s["test"][Ys["test"] == 1]), list(s["test"][Ys["test"] == 0]))
    av = auc(list(s["val"][Ys["val"] == 1]), list(s["val"][Ys["val"] == 0]))
    sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Ys["train"], Ys["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    bb = bacc(s["test"].tolist(), Ys["test"].tolist(), t)
    return at, bb, cv, av, l2, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--anchor", default="evidence")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, X, Y, findings = build(args.debates)
    F = len(findings)
    print("aligned samples: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS)
          + f" | nodes {F} feats {len(COLS)}")
    if min(len(Y[s]) for s in SPLITS) == 0:
        print("not enough data yet"); return

    ev_w = fit_evidence(rows["train"])
    anchors = {s: anchor_of(rows[s], args.anchor, ev_w, False) for s in SPLITS}
    a_te = anchors["test"]
    print(f"anchor ({args.anchor}) alone  TEST "
          f"{auc(list(a_te[Y['test']==1]), list(a_te[Y['test']==0])):.4f}")

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    print(f"KG adjacency: {info['n_edges']}/{info['n_pairs']} pairs, "
          f"isolated {info['isolated']}")

    arms = {"KG": normalised(A),
            "identity": np.eye(F),
            "complete": normalised(np.ones((F, F)) - np.eye(F))}
    for s in range(3):
        arms[f"random{s}"] = normalised(random_adj(F, info["n_edges"], s))

    L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
    res = {}
    print(f"\n  {'arm':10s} {'K':>2s}  {'TEST':>7s} {'bAcc':>7s}  {'cv':>7s} {'val':>7s}  l2")
    for K in (0, 1, 2):
        for name, Ah in arms.items():
            if K == 0 and name != "identity":
                continue
            Xs = {s: propagate(X[s], Ah, K).reshape(len(X[s]), -1) for s in SPLITS}
            at, bb, cv, av, l2, _ = fit_eval(Xs, Y, anchors, L2S)
            tag = "none" if K == 0 else name
            print(f"  {tag:10s} {K:2d}  {at:7.4f} {bb:7.4f}  {cv:7.4f} {av:7.4f}  {l2}")
            res[f"{tag}|K{K}"] = {"test": at, "bacc": bb, "cv": cv, "val": av, "l2": l2}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
