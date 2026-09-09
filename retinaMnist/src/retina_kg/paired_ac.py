"""Paired bootstrap between the two ordinal heads on the same test samples.

The point estimates differ by less than 0.01 on referable AUC, which is inside
the shuffled-KG noise scale, so a paired test on identical samples is the only
way to say whether A and C actually differ. 4000 resamples, matching the
26_baselines pre-registration.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
_H = Path(__file__).resolve()
sys.path.insert(0, str(_H.parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")
import yaml, ordinal, metrics_ordinal as M, retina_features as RF
from kg_adjacency import build_adjacency, kg_operators
from run_ordinal import fit_predict, SPLITS

cfg = yaml.safe_load((P.ONTOLOGY).read_text())
n_class = len(cfg["levels"]); n_cut = n_class - 1
X, Y, IDS, names, prior, level, meta = RF.build(
    P.PACK, P.NPZ,
    P.RESULTS, ("retina_p3",), P.DEBATES, SPLITS)
Z = RF.standardise(X, SPLITS)
A, info = build_adjacency(P.PACK, cfg)
Ap, An = kg_operators(A, prior)
sel = json.loads((P.RESULTS / "ordinal_full.json").read_text())["selected"]

out = {}
for head in ("A", "C"):
    s, p, g, _ = fit_predict(head, Z["train"], Y["train"], Z["test"], Ap, An, n_cut,
                             sel[head]["l2"], sel[head]["kdim"], 0, prior)
    out[head] = {"ref": (s if np.ndim(s) == 1 else s[:, 1]), "grade": g}

y = Y["test"]; yr = M.referable(y)
rng = np.random.default_rng(0)
n = len(y)
dA = out["A"]; dC = out["C"]
bs = {"refauc": [], "qwk": [], "macroR": [], "mae": []}
for _ in range(4000):
    i = rng.integers(0, n, n)
    if yr[i].sum() in (0, len(i)):
        continue
    a, c = dA["ref"][i], dC["ref"][i]
    t = yr[i] == 1
    bs["refauc"].append(M.auc(list(a[t]), list(a[~t])) - M.auc(list(c[t]), list(c[~t])))
    bs["qwk"].append(M.qwk(y[i], dA["grade"][i], n_class) - M.qwk(y[i], dC["grade"][i], n_class))
    bs["macroR"].append(M.macro_recall(y[i], dA["grade"][i], n_class)
                        - M.macro_recall(y[i], dC["grade"][i], n_class))
    bs["mae"].append(M.mae(y[i], dA["grade"][i]) - M.mae(y[i], dC["grade"][i]))

print(f"paired bootstrap, A - C, n={n}, {len(bs['refauc'])} resamples\n")
print(f"{'metric':10s} {'A':>8s} {'C':>8s} {'A-C':>8s} {'95% CI':>20s}  P(A>C)")
for k, fa, fc in (("refauc", M.auc(list(dA['ref'][yr==1]), list(dA['ref'][yr==0])),
                             M.auc(list(dC['ref'][yr==1]), list(dC['ref'][yr==0]))),
                  ("qwk", M.qwk(y, dA['grade'], n_class), M.qwk(y, dC['grade'], n_class)),
                  ("macroR", M.macro_recall(y, dA['grade'], n_class), M.macro_recall(y, dC['grade'], n_class)),
                  ("mae", M.mae(y, dA['grade']), M.mae(y, dC['grade']))):
    v = np.array(bs[k]); lo, hi = np.percentile(v, [2.5, 97.5])
    better = float((v > 0).mean()) if k != "mae" else float((v < 0).mean())
    print(f"{k:10s} {fa:8.4f} {fc:8.4f} {fa-fc:+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {better:.3f}")
print("\n(P(A>C) for mae is P(A has LOWER error))")
json.dump({k: [float(np.percentile(bs[k],2.5)), float(np.percentile(bs[k],97.5))] for k in bs},
          open(P.RESULTS / "paired_ac.json", "w"), indent=1)
