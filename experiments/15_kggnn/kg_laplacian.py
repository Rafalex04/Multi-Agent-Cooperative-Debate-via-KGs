"""The KG as a signed-Laplacian prior on the weights, not as a propagation path.

Every attempt to make the ontology useful by PROPAGATING along it failed, and the
reason is consistent across arms: smoothing features between findings destroys the
per-finding discrimination that the probes provide. The flat 32-feature model wins
because each finding keeps its own weight.

But there is a second way a graph can inform a model, and it has not been tried:
constrain the WEIGHTS instead of mixing the FEATURES.

    penalty = lam * w^T L_signed w,     L_signed = D - (A_same - A_opposite)

Two findings the KG links with the same stance are pushed toward similar weights;
two it links with opposing stances toward opposite weights. Nothing is averaged,
so per-finding discrimination survives, and the ontology still shapes the
solution -- on 546 samples with 32 free parameters, a good prior is worth more
than a good architecture.

This is the graph-Laplacian regularisation that underlies GCNs, applied at the
parameter level. The ablation is exact: same model, same CV protocol, only the
penalty matrix changes.

    identity      plain L2, the KG absent
    KG signed     the ontology, with stance signs
    KG unsigned   the ontology, signs discarded
    shuffled KG   the same matrix with node identities permuted
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
from kg_graph import build_adjacency                                  # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def signed_laplacian(A, prior, signed=True):
    """L = D - A_s where A_s carries +A on same-stance and -A on opposing edges."""
    if signed:
        same = (prior[:, None] * prior[None, :]) > 0
        As = A * same - A * (~same)
    else:
        As = A.copy()
    D = np.diag(np.abs(As).sum(1))
    return D - As


def blockify(L, n_blocks):
    """Repeat the finding-level Laplacian once per feature block."""
    F = len(L)
    M = np.zeros((F * n_blocks, F * n_blocks))
    for b in range(n_blocks):
        M[b * F:(b + 1) * F, b * F:(b + 1) * F] = L
    return M


def fit_pen(X, y, P, lam, l2, epochs=3000, lr=0.3):
    """Logistic regression with penalty lam * w'Pw + l2 * ||w||^2."""
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(y)
    step = lr / (1.0 + lam * np.abs(P).max() + l2)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        e = p - y
        w -= step * ((X.T @ e) / n + lam * (P @ w) + l2 * w)
        b -= step * float(e.mean())
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, Xp, Y, findings = build(args.debates, tag="probe")
    _, Xn, _, _ = build(args.debates, tag="probeneg")
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    X = {s: np.hstack([Xp[s][:, :, 0], Xn[s][:, :, 0]]) for s in SPLITS}
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS)
          + f"  features {Z['train'].shape[1]}")

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    rng = np.random.default_rng(0); perm = rng.permutation(F)
    pens = {
        "plain L2 (no KG)":  np.zeros((2 * F, 2 * F)),
        "KG signed":         blockify(signed_laplacian(A, prior, True), 2),
        "KG unsigned":       blockify(signed_laplacian(A, prior, False), 2),
        "shuffled KG signed": blockify(signed_laplacian(A[np.ix_(perm, perm)],
                                                        prior[perm], True), 2),
    }

    LAMS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
    L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)
    kf = np.array_split(np.random.default_rng(0).permutation(len(Y["train"])), 5)

    res = {}
    print(f"\n  {'penalty':20s} {'TEST':>7s} {'bAcc':>7s}  {'cv':>7s} {'val':>7s}  lam    l2")
    for name, P in pens.items():
        best = None
        lams = (0.0,) if name.startswith("plain") else LAMS
        for lam in lams:
            for l2 in L2S:
                sc = np.zeros(len(Y["train"]))
                for fo in kf:
                    m = np.zeros(len(Y["train"]), dtype=bool); m[fo] = True
                    w, b = fit_pen(Z["train"][~m], Y["train"][~m], P, lam, l2)
                    sc[m] = Z["train"][m] @ w + b
                c = auc(list(sc[Y["train"] == 1]), list(sc[Y["train"] == 0]))
                if best is None or c > best[0]:
                    best = (c, lam, l2)
        cv, lam, l2 = best
        w, b = fit_pen(Z["train"], Y["train"], P, lam, l2)
        s = {k: Z[k] @ w + b for k in SPLITS}
        a = {k: auc(list(s[k][Y[k] == 1]), list(s[k][Y[k] == 0])) for k in SPLITS}
        sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
        t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
        bb = bacc(s["test"].tolist(), Y["test"].tolist(), t)
        print(f"  {name:20s} {a['test']:7.4f} {bb:7.4f}  {cv:7.4f} {a['val']:7.4f}  "
              f"{lam:<6g} {l2:g}")
        res[name] = {"test": a["test"], "bacc": bb, "cv": cv, "val": a["val"],
                     "lam": lam, "l2": l2, "train": a["train"]}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
