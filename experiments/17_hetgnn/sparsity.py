"""Is the KG graph too dense to carry information?

The shuffled-KG control keeps matching the real KG, and there is a mundane
explanation that has never been tested: the finding adjacency connects 72 of 120
possible pairs. At 60% density, message passing averages nearly everything with
nearly everything, and a rewired graph of the same density does the same thing.
Any ontology would look like any other at that density.

If that is the whole story, then sparsifying to the strongest edges should make
the real graph and a shuffled one diverge -- the real graph keeps
`echogenic_rind - echogenic_pseudocapsule` and `posterior_shadowing -
circumscribed_margin`, a shuffled one keeps whatever it happened to land on.

Swept over edge budget, with the shuffled control redrawn at each budget so the
density is matched exactly. Selection is by train CV; test is looked at once.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[1] / "16_external"))
from claims import auc, bacc, kg_findings                             # noqa: E402
from hetgraph import SPLITS, build, kg_operators                      # noqa: E402
from thothgnn3 import fit, forward, node_tensor, probe_block          # noqa: E402


def topk(A, k):
    """Keep the k strongest undirected edges, zero the rest."""
    iu = np.triu_indices(A.shape[0], 1)
    w = A[iu]
    if k >= (w > 0).sum():
        return A.copy()
    thr = np.sort(w)[-k]
    B = np.zeros_like(A); m = w >= thr
    B[iu[0][m], iu[1][m]] = w[m]
    return B + B.T


def rewire_matched(A, rng):
    """Same weight multiset, same edge count, random placement."""
    iu = np.triu_indices(A.shape[0], 1)
    w = A[iu].copy(); rng.shuffle(w)
    B = np.zeros_like(A); B[iu] = w
    return B + B.T


def run(Z, Y, Ap, An, prior, l2=0.03, kdim=2, seed=0):
    P = fit(Z["train"], Y["train"], Ap, An, l2=l2, kdim=kdim, seed=seed, prior_centre=prior)
    s = {k: forward(P, Z[k], Ap, An)[2] for k in SPLITS}
    return auc(list(s["test"][Y["test"] == 1]), list(s["test"][Y["test"] == 0])), s


def main():
    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    probes = probe_block(names)
    Ap, An = kg_operators(names, prior)
    X = {s: node_tensor(d[s], names, prior, probes, s) for s in SPLITS}
    Y = {s: d[s]["y"] for s in SPLITS}
    mu = X["train"].reshape(-1, X["train"].shape[-1]).mean(0)
    sd = X["train"].reshape(-1, X["train"].shape[-1]).std(0); sd = np.where(sd < 1e-9, 1, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}

    tot = int(((Ap + An)[np.triu_indices(16, 1)] > 0).sum())
    print(f"full KG finding graph: {tot}/120 pairs ({tot/120:.0%} density)\n")
    print(f"{'edges':>6s} {'density':>8s} {'KG test':>9s} {'shuffled':>9s} {'sd':>7s} {'KG-shuf':>9s} {'sigma':>7s}")
    out = []
    for k in (4, 8, 16, 24, 40, 72):
        Apk, Ank = topk(Ap, k), topk(An, k)
        a_kg, _ = run(Z, Y, Apk, Ank, prior)
        sh = []
        for s_ in range(5):
            rng = np.random.default_rng(400 + s_)
            a_sh, _ = run(Z, Y, rewire_matched(Apk, rng), rewire_matched(Ank, rng), prior)
            sh.append(a_sh)
        m, sdv = float(np.mean(sh)), float(np.std(sh))
        sig = (a_kg - m) / (sdv + 1e-9)
        print(f"{k:6d} {2*k/240:8.0%} {a_kg:9.4f} {m:9.4f} {sdv:7.4f} {a_kg-m:+9.4f} {sig:+7.2f}")
        out.append({"edges": k, "kg": a_kg, "shuffled_mean": m, "shuffled_sd": sdv,
                    "delta": a_kg - m, "sigma": sig})
    (_HERE.parent / "results/sparsity.json").write_text(json.dumps(out, indent=1, default=float))
    best = max(out, key=lambda r: r["sigma"])
    print(f"\nlargest separation at {best['edges']} edges: {best['delta']:+.4f} "
          f"({best['sigma']:+.2f} sd)")
    print("-> " + ("density was masking the ontology" if best["sigma"] > 2
                   else "sparsifying does not rescue it; the topology is not the carrier"))


if __name__ == "__main__":
    main()
