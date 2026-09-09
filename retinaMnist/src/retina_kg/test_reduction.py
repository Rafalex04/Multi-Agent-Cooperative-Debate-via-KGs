"""Option A at C=2 must BE thothgnn3, on the real breast corpus.

The reparameterisation is c = -theta_0: v3 puts a scalar bias inside s, CORAL
puts it in the cut point. Neither is regularised, so with theta starting at zero
the two gradient trajectories are mirror images and the fitted scores must agree
to floating point. If this ever fails, the ordinal head has drifted away from
the model whose numbers we are trying to extend.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("17_hetgnn", "14_kgtensor", "15_kggnn")

import thothgnn3                                                   # noqa: E402
from thothgnn3 import build, kg_operators, node_tensor, probe_block  # noqa: E402
from hetgraph import SPLITS                                         # noqa: E402
from claims import auc                                              # noqa: E402
import ordinal                                                      # noqa: E402


def load_breast(phrasings="p3"):
    thothgnn3.PHRASE_COLS = {"both": (0, 1), "p2": (0,), "p3": (1,)}[phrasings]
    root = str(P.BREAST_DEBATES)
    d, names, prior = build(root)
    probes = probe_block(names)
    X = {s: node_tensor(d[s], names, prior, probes, s) for s in SPLITS}
    Y = {s: d[s]["y"] for s in SPLITS}
    mu = X["train"].reshape(-1, X["train"].shape[-1]).mean(0)
    sd = X["train"].reshape(-1, X["train"].shape[-1]).std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    Ap, An = kg_operators(names, prior)
    return Z, Y, Ap, An, names, prior


def main():
    Z, Y, Ap, An, names, prior = load_breast("p3")
    print(f"corpus: {len(names)} findings, d={Z['train'].shape[2]}, "
          f"n={[len(Y[s]) for s in SPLITS]}")

    kw = dict(l2=0.03, kdim=2, epochs=800, lr=0.15, seed=0, prior_centre=prior)

    P3 = thothgnn3.fit(Z["train"], Y["train"], Ap, An, **kw)
    s_v3 = {s: thothgnn3.forward(P3, Z[s], Ap, An)[2] for s in SPLITS}

    PA = ordinal.fit(Z["train"], Y["train"], Ap, An, n_cut=1, **kw)
    s_A = {s: ordinal.predict(PA, Z[s], Ap, An)[0] - PA["theta"][0] for s in SPLITS}

    print(f"\n  v3 bias c        = {P3['c']:+.9f}")
    print(f"  A  -theta_0      = {-PA['theta'][0]:+.9f}")

    worst = 0.0
    for s in SPLITS:
        dv = float(np.abs(s_v3[s] - s_A[s]).max())
        worst = max(worst, dv)
        a3 = auc(list(s_v3[s][Y[s] == 1]), list(s_v3[s][Y[s] == 0]))
        aA = auc(list(s_A[s][Y[s] == 1]), list(s_A[s][Y[s] == 0]))
        print(f"  {s:5s} max|s_v3 - s_A| = {dv:.3e}   AUC v3 {a3:.6f}  A {aA:.6f}")

    ok = worst < 1e-8
    print(f"\n  REDUCTION {'PASS' if ok else 'FAIL'}  (worst {worst:.3e})")

    # Option C at n_cut=1 is literally one thothgnn3, so it must match too.
    PC = ordinal.fit_stacked(Z["train"], Y["train"], Ap, An, n_cut=1, **kw)
    SC, pC, gC = ordinal.predict_stacked(PC, Z["test"], Ap, An)
    dC = float(np.abs(SC[:, 0] - s_v3["test"]).max())
    print(f"  C at 1 cut point: max|s_C - s_v3| = {dC:.3e}  "
          f"{'PASS' if dC < 1e-12 else 'FAIL'}")
    return 0 if ok and dC < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
