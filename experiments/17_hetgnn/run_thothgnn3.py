"""Evaluate ThothGNN v3 against the controls that killed its predecessors.

The comparison is not "does the GNN score well" -- a dense 16-node graph smooths
features and that helps a little whatever the edges say. The comparisons that
decide anything are:

  no-graph      same network, A+ = A- = 0. Isolates message passing from capacity.
  shuffled-KG   same edge count and weight distribution, edges rewired at random,
                5 draws. Isolates the ONTOLOGY from graph-ness.
  no-debate     probe and prior channels only. Isolates the citation merge.

Hyperparameters are chosen by 5-fold CV on train. Test is evaluated once per arm.
bAcc thresholds are swept on train+val and applied unchanged.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
from claims import auc, bacc                                          # noqa: E402
from gates import L2S, fit_ce, fit_rank                               # noqa: E402
from hetgraph import SPLITS, build, kg_operators                      # noqa: E402
import thothgnn3                                                       # noqa: E402
from thothgnn3 import check_grad, fit, forward, node_tensor, probe_block  # noqa: E402


def shuffle_kg(Ap, An, rng):
    """Rewire while preserving the multiset of weights and symmetry."""
    F = Ap.shape[0]
    out = []
    for A in (Ap, An):
        iu = np.triu_indices(F, 1)
        w = A[iu].copy()
        rng.shuffle(w)
        B = np.zeros_like(A); B[iu] = w
        out.append(B + B.T)
    return out


def standardise(X):
    mu = X["train"].reshape(-1, X["train"].shape[-1]).mean(0)
    sd = X["train"].reshape(-1, X["train"].shape[-1]).std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {s: (X[s] - mu) / sd for s in X}


def evaluate(Z, Y, Ap, An, prior, l2, kdim, seed=0):
    P = fit(Z["train"], Y["train"], Ap, An, l2=l2, kdim=kdim, seed=seed,
            prior_centre=prior)
    sc = {s: forward(P, Z[s], Ap, An)[2] for s in SPLITS}
    return sc


def cv_auc(Ztr, ytr, Ap, An, prior, l2, kdim, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    out = np.zeros(len(ytr))
    for fo in np.array_split(idx, folds):
        m = np.zeros(len(ytr), bool); m[fo] = True
        P = fit(Ztr[~m], ytr[~m], Ap, An, l2=l2, kdim=kdim, seed=seed, prior_centre=prior)
        out[m] = forward(P, Ztr[m], Ap, An)[2]
    return auc(list(out[ytr == 1]), list(out[ytr == 0]))


def report(name, sc, Y, res):
    a = {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS}
    f = np.concatenate([sc["train"], sc["val"]]); g = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(f.tolist())), key=lambda t: bacc(f.tolist(), g.tolist(), t))
    bb = bacc(sc["test"].tolist(), Y["test"].tolist(), t)
    print(f"  {name:26s} train {a['train']:.4f}  val {a['val']:.4f}  "
          f"TEST {a['test']:.4f}  bAcc {bb:.4f}")
    res[name] = {"auc": a, "bacc": bb}
    return a["test"], bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrasings", default="both", choices=("both", "p2", "p3"))
    ap.add_argument("--out", default="thothgnn3.json")
    a = ap.parse_args()
    thothgnn3.PHRASE_COLS = {"both": (0, 1), "p2": (0,), "p3": (1,)}[a.phrasings]
    print(f"probe phrasing arms in the node tensor: {a.phrasings}")

    w = check_grad()
    print(f"gradient check {w:.3e}  {'PASS' if w < 1e-6 else 'FAIL'}\n")

    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    probes = probe_block(names)
    Ap, An = kg_operators(names, prior)
    X = {s: node_tensor(d[s], names, prior, probes, s) for s in SPLITS}
    Y = {s: d[s]["y"] for s in SPLITS}
    Z = standardise(X)
    print(f"nodes {len(names)}  features {X['train'].shape[-1]}  "
          f"n={[len(Y[s]) for s in SPLITS]}\n")
    res = {}

    # ---- the anchor the GNN has to beat: flat readout, no graph at all -----
    flat = {s: Z[s].reshape(len(Y[s]), -1) for s in SPLITS}
    best = None
    for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
        for l2 in L2S:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(Y["train"])); o = np.zeros(len(Y["train"]))
            for fo in np.array_split(idx, 5):
                m = np.zeros(len(Y["train"]), bool); m[fo] = True
                ww, bb = f(flat["train"][~m], Y["train"][~m], l2)
                o[m] = flat["train"][m] @ ww + bb
            c = auc(list(o[Y["train"] == 1]), list(o[Y["train"] == 0]))
            if best is None or c > best[0]:
                best = (c, l2, nm, f)
    cv, l2f, nmf, ff = best
    ww, bb2 = ff(flat["train"], Y["train"], l2f)
    print(f"=== anchor: flat linear readout (cv {cv:.4f}, {nmf}, l2={l2f}) ===")
    a_test, a_bacc = report("linear flat", {s: flat[s] @ ww + bb2 for s in SPLITS}, Y, res)

    # ---- GNN, hyperparameters by CV on train only -------------------------
    print("\n=== ThothGNN v3: message passing on the KG ===")
    bestg = None
    for l2 in (0.03, 0.1, 0.3):
        for kdim in (2, 4):
            c = cv_auc(Z["train"], Y["train"], Ap, An, prior, l2, kdim)
            print(f"  cv l2={l2:<5g} k={kdim}  {c:.4f}")
            if bestg is None or c > bestg[0]:
                bestg = (c, l2, kdim)
    cvg, l2g, kg = bestg
    print(f"  -> selected l2={l2g}, k={kg} (cv {cvg:.4f})")
    g_test, g_bacc = report("GNN (real KG)", evaluate(Z, Y, Ap, An, prior, l2g, kg), Y, res)

    # ---- controls ---------------------------------------------------------
    print("\n=== controls ===")
    Z0 = np.zeros_like(Ap)
    report("GNN no-graph (A=0)", evaluate(Z, Y, Z0, Z0, prior, l2g, kg), Y, res)

    sh = []
    for s_ in range(5):
        rng = np.random.default_rng(100 + s_)
        Sp, Sn = shuffle_kg(Ap, An, rng)
        sc = evaluate(Z, Y, Sp, Sn, prior, l2g, kg)
        sh.append(auc(list(sc["test"][Y["test"] == 1]), list(sc["test"][Y["test"] == 0])))
    print(f"  {'GNN shuffled-KG (5 draws)':26s} TEST {np.mean(sh):.4f} +- {np.std(sh):.4f}"
          f"   [{min(sh):.4f}, {max(sh):.4f}]")
    res["shuffled"] = {"mean": float(np.mean(sh)), "sd": float(np.std(sh)), "draws": sh}
    print(f"  {'-> KG minus shuffled':26s} {g_test - np.mean(sh):+.4f}"
          f"  ({(g_test - np.mean(sh)) / (np.std(sh) + 1e-9):.2f} sd)")

    npc = len(thothgnn3.PHRASE_COLS)
    Znd = {s: Z[s].copy() for s in SPLITS}
    for s in SPLITS:
        Znd[s][:, :, npc:npc + 2] = 0.0
    report("GNN no-debate channel", evaluate(Znd, Y, Ap, An, prior, l2g, kg), Y, res)

    res["selected"] = {"l2": l2g, "kdim": kg, "cv": cvg}
    res["anchor"] = {"test": a_test, "bacc": a_bacc, "l2": l2f, "loss": nmf}
    (_HERE.parent / "results" / a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nGNN - anchor: {g_test - a_test:+.4f} AUC, {g_bacc - a_bacc:+.4f} bAcc")


if __name__ == "__main__":
    main()
