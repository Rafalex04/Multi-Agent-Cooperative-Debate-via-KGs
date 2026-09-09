"""Paired bootstrap of A vs C on the 5-class axes MedMNIST reports.

The referable-DR bootstrap favoured A; ACC and macro-OVR AUC favour C. Both
cannot be dismissed as noise without testing them on the same samples.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
_H = Path(__file__).resolve()
sys.path.insert(0, str(_H.parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")
import yaml, metrics_ordinal as M, retina_features as RF
from kg_adjacency import build_adjacency, kg_operators
from run_ordinal import fit_predict, SPLITS

cfg = yaml.safe_load((P.ONTOLOGY).read_text())
NC = len(cfg["levels"]); n_cut = NC - 1
X, Y, *_rest = RF.build(P.PACK,
                        P.NPZ,
                        P.RESULTS, ("retina_p3",),
                        P.DEBATES, SPLITS)
names, prior = _rest[1], _rest[2]
Z = RF.standardise(X, SPLITS)
A_, _ = build_adjacency(P.PACK, cfg)
Ap, An = kg_operators(A_, prior)
sel = json.loads((P.RESULTS / "ordinal_full.json").read_text())["selected"]
y = Y["test"]

def class_probs(p):
    ext = np.concatenate([np.ones((len(p), 1)), p, np.zeros((len(p), 1))], axis=1)
    q = np.clip(ext[:, :-1] - ext[:, 1:], 1e-9, None)
    return q / q.sum(1, keepdims=True)

D = {}
for head in ("A", "C"):
    s, p, g, _ = fit_predict(head, Z["train"], Y["train"], Z["test"], Ap, An, n_cut,
                             sel[head]["l2"], sel[head]["kdim"], 0, prior)
    D[head] = {"P": class_probs(p), "g": g}

def _fast_auc(score, pos):
    """Midrank AUC, O(n log n). Identical to claims.auc (verified to 0.0 on
    300 tie-heavy cases); the pure-Python double loop there is hours at
    4000 resamples x 5 classes."""
    order = np.argsort(score, kind="mergesort"); s = score[order]
    ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s), float); r[order] = ranks
    npos = int(pos.sum()); nneg = len(pos) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ovr(P, yy):
    out = [_fast_auc(P[:, c], yy == c) for c in range(NC)
           if (yy == c).any() and (yy != c).any()]
    return float(np.mean(out))

rng = np.random.default_rng(0); n = len(y)
bs = {"acc": [], "auc_ovr": []}
for _ in range(4000):
    i = rng.integers(0, n, n)
    bs["acc"].append(float((D["A"]["g"][i] == y[i]).mean()) - float((D["C"]["g"][i] == y[i]).mean()))
    bs["auc_ovr"].append(ovr(D["A"]["P"][i], y[i]) - ovr(D["C"]["P"][i], y[i]))

print(f"paired bootstrap, A - C, 5-class axes, n={n}, 4000 resamples\n")
print(f"{'metric':10s} {'A':>8s} {'C':>8s} {'A-C':>8s} {'95% CI':>20s}  P(C>A)")
for k, fa, fc in (("acc", float((D['A']['g']==y).mean()), float((D['C']['g']==y).mean())),
                  ("auc_ovr", ovr(D['A']['P'], y), ovr(D['C']['P'], y))):
    v = np.array(bs[k]); lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"{k:10s} {fa:8.4f} {fc:8.4f} {fa-fc:+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {float((v<0).mean()):.3f}")
json.dump({k: {"ci": [float(np.percentile(bs[k],2.5)), float(np.percentile(bs[k],97.5))],
               "p_c_better": float((np.array(bs[k])<0).mean())} for k in bs},
          open(P.RESULTS / "paired_multiclass.json", "w"), indent=1)
