"""Is the architecture failing because the dataset is small? And does it converge?

Two diagnostics that between them answer 'would more data rescue this'.

LEARNING CURVE. Refit the merged GNN on an increasing fraction of the training
folds and track two things: absolute performance, and the gap between the real KG
and a weight-matched rewiring. Absolute performance rising with n means the model
is data-limited. The GAP is the one that matters for the KG question:

  gap grows with n   -> the ontology effect is real and we are power-limited
  gap flat in n      -> more data will not produce it; it is a ceiling, not a
                        sample-size problem

CONVERGENCE. Train AUC and held-out AUC every 25 epochs, to check the fit is
stable rather than still moving or already overfitting when we stop at 800.

Subsampling is at CASE level, never image level: BUS-BRA gives two views of most
lesions and splitting them across the boundary would leak.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[1] / "16_external"))
from analyse_e1 import auc, load_probes                                  # noqa: E402
from claims import kg_findings                                           # noqa: E402
from external_merged import load_debates                                 # noqa: E402
from hetgraph import kg_operators                                        # noqa: E402
from powered_gnn import case_folds                                       # noqa: E402
from sparsity import rewire_matched                                      # noqa: E402
from thothgnn3 import fit, forward, grads, init                          # noqa: E402

_EXT = _HERE.parents[1] / "16_external/results"


def build():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    deb = load_debates(names)
    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(deb) & set(probes))
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz")
    y = (z["labels"][:, 0] == 0).astype(float)[common]
    cases = z["cases"][common]
    F, n = len(names), len(common)
    X = np.zeros((n, F, 5))
    for k, i in enumerate(common):
        Xc, Acc, Acf, mask, _ = deb[i]
        X[k, :, 0:2] = probes[i]
        X[k, :, 2] = (mask[:, None] * Acf).sum(0)
        X[k, :, 3] = ((mask * Xc[:, 0])[:, None] * Acf).sum(0)
        X[k, :, 4] = prior
    mu, sd = X.reshape(-1, 5).mean(0), X.reshape(-1, 5).std(0)
    Z = (X - mu) / np.where(sd < 1e-9, 1, sd)
    return Z, y, cases, names, prior


def main():
    Z, y, cases, names, prior = build()
    Ap, An = kg_operators(names, prior)
    folds = case_folds(cases, k=5, seed=0)
    u = np.unique(cases)
    y_case = np.array([y[cases == c][0] for c in u])
    def to_case(v):
        return np.array([v[cases == c].mean() for c in u])
    print(f"n={len(y)} images, {len(u)} cases\n")

    # ------------------------------------------------ learning curve --------
    fracs = (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)
    rows = []
    print(f"{'frac':>5s} {'train cases':>12s} {'KG':>8s} {'shuffled':>10s} {'gap':>8s}")
    for f in fracs:
        oof_kg = np.zeros(len(y))
        oof_sh = [np.zeros(len(y)) for _ in range(3)]
        ntr = 0
        for fi, te in enumerate(folds):
            tr_cases = np.setdiff1d(u, np.unique(cases[te]))
            rng = np.random.default_rng(1000 + fi)
            keep = rng.permutation(tr_cases)[:max(8, int(round(len(tr_cases) * f)))]
            tr = np.isin(cases, keep)
            ntr += len(keep)
            P = fit(Z[tr], y[tr], Ap, An, l2=0.03, kdim=2, prior_centre=prior)
            oof_kg[te] = forward(P, Z[te], Ap, An)[2]
            for s in range(3):
                rg = np.random.default_rng(2000 + s)
                Sp, Sn = rewire_matched(Ap, rg), rewire_matched(An, rg)
                Ps = fit(Z[tr], y[tr], Sp, Sn, l2=0.03, kdim=2, prior_centre=prior)
                oof_sh[s][te] = forward(Ps, Z[te], Sp, Sn)[2]
        a_kg = auc(y_case, to_case(oof_kg))
        a_sh = float(np.mean([auc(y_case, to_case(o)) for o in oof_sh]))
        rows.append({"frac": f, "train_cases": ntr // 5, "kg": a_kg,
                     "shuffled": a_sh, "gap": a_kg - a_sh})
        print(f"{f:5.2f} {ntr//5:12d} {a_kg:8.4f} {a_sh:10.4f} {a_kg-a_sh:+8.4f}")

    # ------------------------------------------------ convergence -----------
    print("\nconvergence on one fold (train vs held-out AUC by epoch)")
    te = folds[0]; tr = ~te
    F_, d = Z.shape[1], Z.shape[2]
    rng = np.random.default_rng(0)
    P = init(F_, d, 2, rng, prior)
    centre = np.zeros((F_, 2)); centre[:, 0] = prior
    m = {k: np.zeros_like(v) if not np.isscalar(v) else 0.0 for k, v in P.items()}
    curve = []
    for ep in range(1, 801):
        G, _ = grads(P, Z[tr], Ap, An, y[tr], 0.03, centre)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            P[k] = P[k] - 0.15 * m[k]
        if ep % 25 == 0:
            a_tr = auc(y[tr], forward(P, Z[tr], Ap, An)[2])
            a_te = auc(y[te], forward(P, Z[te], Ap, An)[2])
            curve.append({"epoch": ep, "train": a_tr, "test": a_te})
            if ep % 100 == 0:
                print(f"  epoch {ep:4d}  train {a_tr:.4f}  held-out {a_te:.4f}")

    out = {"learning_curve": rows, "convergence": curve,
           "n_images": int(len(y)), "n_cases": int(len(u))}
    (_HERE.parent / "results/learning_curve.json").write_text(json.dumps(out, indent=1, default=float))
    g = [r["gap"] for r in rows]
    print(f"\ngap at smallest n {g[0]:+.4f}  ->  at full n {g[-1]:+.4f}")
    print("-> " + ("gap GROWS with data: power-limited, more data would help"
                   if g[-1] - g[0] > 0.01 else
                   "gap is FLAT in data: not a sample-size problem"))


if __name__ == "__main__":
    main()
