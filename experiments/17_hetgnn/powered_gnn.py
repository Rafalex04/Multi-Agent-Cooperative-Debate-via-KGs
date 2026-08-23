"""The KG-vs-shuffled question, finally at a sample size that can decide it.

Every graph result in this project has been decided at 1-2 sd on 156 test
samples, which is why eight of them came back "null" and none came back
"definitely null". BUS-BRA supplies 1875 images over 1064 cases -- 12x the test
set, and roughly 3.5x tighter error bars on an AUC.

The protocol is BUS-BRA's own 5-fold cross-validation, which is defined at CASE
level. That matters here more than usual: 811 of the 1064 cases contribute two
views of the same lesion, so an image-level split would put the same lesion on
both sides of the fold boundary and inflate every arm equally.

Three arms per fold, identical in every respect except the graph:

    linear      no message passing at all
    GNN + KG    message passing on the ontology's signed adjacency
    GNN + shuffled   same edge count and weight multiset, rewired (5 draws)

If the KG arm does not separate from the shuffled arm at n=1875, the KG's
*topology* carries nothing for this task, and that is a result rather than a
failure -- the ontology would still be supplying the finding set, the
descriptions and the stance priors, which is where its measured value lives.
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
from analyse_e1 import auc, load_probes, paired_bootstrap                # noqa: E402
from claims import kg_findings                                           # noqa: E402
from gates import fit_ce, fit_rank                                       # noqa: E402
from hetgraph import kg_operators                                        # noqa: E402
from run_thothgnn3 import shuffle_kg                                     # noqa: E402
from thothgnn3 import fit, forward                                       # noqa: E402

_EXT = _HERE.parents[1] / "16_external/results"
TAGS = ("busbra_p2", "busbra_p3")


def build_tensor(names, prior):
    """(n x F x 3 node tensor, y, case ids). Channels: probe P2, probe P3, prior."""
    probes = load_probes(names, TAGS, _EXT)
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz")
    idx = sorted(probes)
    X = np.zeros((len(idx), len(names), 3))
    for i, k in enumerate(idx):
        X[i, :, 0:2] = probes[k]
    X[:, :, 2] = prior[None, :]
    y = (z["labels"][:, 0] == 0).astype(float)[idx]
    return X, y, z["cases"][idx]


def case_folds(cases, k=5, seed=0):
    """k folds split on CASE, so both views of a lesion land on the same side."""
    u = np.unique(cases)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    out = []
    for part in np.array_split(u, k):
        out.append(np.isin(cases, part))
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings])
    X, y, cases = build_tensor(names, prior)
    print(f"n={len(y)} images, {len(np.unique(cases))} cases, "
          f"malignant {y.mean():.3f}")
    if len(y) < 800:
        print("E1 not far enough along for a powered test yet")
        return
    Ap, An = kg_operators(names, prior)
    mu = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0)
    Z = (X - np.where(sd < 1e-9, 0, mu)) / np.where(sd < 1e-9, 1, sd)
    folds = case_folds(cases)
    Zn = np.zeros_like(Ap)

    shuf = [shuffle_kg(Ap, An, np.random.default_rng(300 + s)) for s in range(5)]
    arms = {"linear": None, "GNN+KG": (Ap, An), "GNN+nograph": (Zn, Zn)}
    for s in range(5):
        arms[f"GNN+shuf{s}"] = shuf[s]
    oof = {k: np.zeros(len(y)) for k in arms}

    for te in folds:
        tr = ~te
        flat_tr = Z[tr].reshape(tr.sum(), -1)
        w, b = fit_rank(flat_tr, y[tr], 0.3)
        oof["linear"][te] = Z[te].reshape(te.sum(), -1) @ w + b
        for k, g in arms.items():
            if g is None:
                continue
            P = fit(Z[tr], y[tr], g[0], g[1], l2=0.03, kdim=2, prior_centre=prior)
            oof[k][te] = forward(P, Z[te], g[0], g[1])[2]

    u = np.unique(cases)
    def to_case(v):
        return np.array([v[cases == c].mean() for c in u])
    y_case = np.array([y[cases == c][0] for c in u])

    print(f"\n{'arm':16s} {'image AUC':>10s} {'case AUC':>10s}")
    res = {}
    for k in ("linear", "GNN+nograph", "GNN+KG"):
        ai, ac = auc(y, oof[k]), auc(y_case, to_case(oof[k]))
        res[k] = {"auc_image": ai, "auc_case": ac}
        print(f"{k:16s} {ai:10.4f} {ac:10.4f}")
    sh_i = [auc(y, oof[f"GNN+shuf{s}"]) for s in range(5)]
    sh_c = [auc(y_case, to_case(oof[f"GNN+shuf{s}"])) for s in range(5)]
    print(f"{'GNN+shuffled':16s} {np.mean(sh_i):10.4f} {np.mean(sh_c):10.4f}"
          f"   (sd {np.std(sh_c):.4f}, 5 draws)")

    d = res["GNN+KG"]["auc_case"] - np.mean(sh_c)
    p = paired_bootstrap(y_case, to_case(oof["GNN+KG"]),
                         to_case(np.mean([oof[f"GNN+shuf{s}"] for s in range(5)], 0)))
    print(f"\nKG minus shuffled (case-level): {d:+.4f}"
          f"   ({d/(np.std(sh_c)+1e-9):+.2f} sd)   P(KG better) {p:.3f}")
    print(f"KG minus linear:                "
          f"{res['GNN+KG']['auc_case'] - res['linear']['auc_case']:+.4f}")
    print("\n-> " + ("KG topology CONFIRMED at P>=0.95" if p >= 0.95 else
                     "KG topology not distinguishable from a random graph of equal density"))
    res.update({"shuffled_case_mean": float(np.mean(sh_c)),
                "shuffled_case_sd": float(np.std(sh_c)),
                "P_kg_better_than_shuffled": p, "n_images": int(len(y)),
                "n_cases": int(len(u))})
    (_HERE.parent / "results/powered_gnn.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
