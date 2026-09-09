"""Five fixed seeds for heads A and C, at the CV-selected hyperparameters.

`run_ordinal.py` reports a single seed, which is v3's convention but not
`26_baselines`', and several A-vs-C gaps are small enough that seed variance
matters. Hyperparameters are NOT re-selected here -- they stay at what 5-fold CV
on train chose, so this measures seed variance, not a second selection pass.

Reports mean +- sd across seeds for every metric, and the controls alongside, so
the KG-topology delta gets an honest error bar too.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")
import yaml, metrics_ordinal as M, retina_features as RF               # noqa: E402
from kg_adjacency import build_adjacency, kg_operators                 # noqa: E402
from run_ordinal import fit_predict, SPLITS                            # noqa: E402
from run_thothgnn3 import shuffle_kg                                   # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def stat(vals):
    v = np.array(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std()), "vals": v.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P.RESULTS / "ordinal_seeds.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(P.ONTOLOGY.read_text())
    NC = len(cfg["levels"]); n_cut = NC - 1
    X, Y, IDS, names, prior, level, meta = RF.build(
        P.PACK, P.NPZ, P.RESULTS, ("retina_p3",), P.DEBATES, SPLITS)
    Z = RF.standardise(X, SPLITS)
    A_, _ = build_adjacency(P.PACK, cfg); Ap, An = kg_operators(A_, prior)
    sel = json.loads((P.RESULTS / "ordinal_full.json").read_text())["selected"]
    y = Y["test"].astype(int); yr = M.referable(y)
    res = {"seeds": list(SEEDS), "selected": sel}

    for head in ("A", "C"):
        l2, k = sel[head]["l2"], sel[head]["kdim"]
        print(f"\n=== option {head} (l2={l2}, k={k}), {len(SEEDS)} seeds ===")
        acc = {m: [] for m in ("refauc", "qwk", "acc", "macroR", "mae")}
        nog, shuf = [], []
        for sd in SEEDS:
            s, p, g, _ = fit_predict(head, Z["train"], Y["train"], Z["test"],
                                     Ap, An, n_cut, l2, k, sd, prior)
            ref = s if np.ndim(s) == 1 else s[:, 1]
            acc["refauc"].append(M.auc(list(ref[yr == 1]), list(ref[yr == 0])))
            acc["qwk"].append(M.qwk(y, g, NC))
            acc["acc"].append(float((g == y).mean()))
            acc["macroR"].append(M.macro_recall(y, g, NC))
            acc["mae"].append(M.mae(y, g))
            Zr = np.zeros_like(Ap)
            s0, _, _, _ = fit_predict(head, Z["train"], Y["train"], Z["test"],
                                      Zr, Zr, n_cut, l2, k, sd, prior)
            r0 = s0 if np.ndim(s0) == 1 else s0[:, 1]
            nog.append(M.auc(list(r0[yr == 1]), list(r0[yr == 0])))
            rng = np.random.default_rng(sd)
            Sp, Sn = shuffle_kg(Ap, An, rng)
            ss, _, _, _ = fit_predict(head, Z["train"], Y["train"], Z["test"],
                                      Sp, Sn, n_cut, l2, k, sd, prior)
            rs = ss if np.ndim(ss) == 1 else ss[:, 1]
            shuf.append(M.auc(list(rs[yr == 1]), list(rs[yr == 0])))
            print(f"  seed {sd}: refAUC {acc['refauc'][-1]:.4f}  QWK {acc['qwk'][-1]:+.4f}  "
                  f"ACC {acc['acc'][-1]:.4f}  | no-graph {nog[-1]:.4f}  shuffled {shuf[-1]:.4f}")
        res[head] = {m: stat(v) for m, v in acc.items()}
        res[head]["no_graph_refauc"] = stat(nog)
        res[head]["shuffled_refauc"] = stat(shuf)
        d = np.array(acc["refauc"]) - np.array(shuf)
        res[head]["kg_minus_shuffled"] = stat(d)
        print(f"  --- refAUC {res[head]['refauc']['mean']:.4f} +- {res[head]['refauc']['sd']:.4f}"
              f"   QWK {res[head]['qwk']['mean']:+.4f} +- {res[head]['qwk']['sd']:.4f}"
              f"   ACC {res[head]['acc']['mean']:.4f} +- {res[head]['acc']['sd']:.4f}")
        print(f"  --- KG minus shuffled (paired per seed) {d.mean():+.4f} +- {d.std():.4f}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
