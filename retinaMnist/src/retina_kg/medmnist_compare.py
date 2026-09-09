"""Put the ordinal heads on the axes MedMNIST's own RetinaMNIST baselines use.

The headline elsewhere is referable-DR AUC (grade >= 2), which is a binary
reduction and is NOT what the published table reports. MedMNIST reports 5-class
ACC and a macro one-vs-rest AUC. CORAL gives class probabilities for free:

    P(y=c) = P(y>c-1) - P(y>c),   with P(y>-1)=1 and P(y>C-1)=0

so both are computable without refitting anything differently.
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
NC = len(cfg["levels"]); n_cut = NC - 1
X, Y, IDS, names, prior, level, meta = RF.build(
    P.PACK, P.NPZ,
    P.RESULTS, ("retina_p3",), P.DEBATES, SPLITS)
Z = RF.standardise(X, SPLITS)
A, _ = build_adjacency(P.PACK, cfg)
Ap, An = kg_operators(A, prior)
sel = json.loads((P.RESULTS / "ordinal_full.json").read_text())["selected"]
y = Y["test"]

def class_probs(p):
    """(n, n_cut) cumulative P(y>c) -> (n, NC) class probabilities."""
    ext = np.concatenate([np.ones((len(p), 1)), p, np.zeros((len(p), 1))], axis=1)
    q = ext[:, :-1] - ext[:, 1:]
    return np.clip(q, 1e-9, None) / np.clip(q, 1e-9, None).sum(1, keepdims=True)

def macro_ovr_auc(P, y):
    out = []
    for c in range(NC):
        t = (y == c)
        if t.any() and (~t).any():
            out.append(M.auc(list(P[t, c]), list(P[~t, c])))
    return float(np.mean(out)), [round(x, 4) for x in out]

print(f"{'arm':22s} {'ACC':>7s} {'AUC(ovr)':>9s} {'QWK':>8s} {'refAUC':>8s} {'macroR':>7s}")
res = {}
for head in ("A", "C"):
    s, p, g, _ = fit_predict(head, Z["train"], Y["train"], Z["test"], Ap, An, n_cut,
                             sel[head]["l2"], sel[head]["kdim"], 0, prior)
    P = class_probs(p)
    acc = float((g == y).mean())
    auc_m, per = macro_ovr_auc(P, y)
    ref = s if np.ndim(s) == 1 else s[:, 1]
    yr = M.referable(y)
    res[head] = {"acc": acc, "auc_ovr": auc_m, "auc_per_class": per,
                 "qwk": M.qwk(y, g, NC),
                 "referable_auc": M.auc(list(ref[yr == 1]), list(ref[yr == 0])),
                 "macro_recall": M.macro_recall(y, g, NC)}
    r = res[head]
    print(f"{head+' (real KG)':22s} {acc:7.4f} {auc_m:9.4f} {r['qwk']:+8.4f} "
          f"{r['referable_auc']:8.4f} {r['macro_recall']:7.4f}")
    print(f"{'  per-class AUC':22s} {per}")

maj = float((y == np.bincount(Y['train'], minlength=NC).argmax()).mean())
print(f"\n{'majority-class':22s} {maj:7.4f}  (predict grade "
      f"{np.bincount(Y['train'],minlength=NC).argmax()} always)")
print(f"{'test grade counts':22s} {np.bincount(y, minlength=NC).tolist()}")
json.dump(res, open(P.RESULTS / "medmnist_compare.json", "w"), indent=1)
