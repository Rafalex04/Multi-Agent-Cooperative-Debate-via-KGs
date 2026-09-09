"""Learning curves for ThothGNN v3 AND both baselines, on identical axes.

A curve per method answers a question a single test number cannot: is a method
behind because it is worse, or because it is data-starved? B2 has ~10,700
parameters against v3's 63, so if B2's curve is still climbing at n=1080 its tie
with v3 is a small-data artefact; if it has plateaued, the tie is real.

  v3-A       our cumulative-link head, 5-fold-CV-selected hyperparameters
  v3-nograph the same with A+ = A- = 0, to see whether the graph ever helps at
             ANY training size -- the null could in principle be size-dependent
  B2         our implementation of GraphGeo, ordinal edge rule, its own CV tau
  B1         our implementation of Catfish -- NO learned head, so its curve moves
             only through tone/tau_conf selection. Flat by construction is the
             expected result and is worth showing rather than omitting.

All arms are scored with referable-DR AUC on the SAME 400 test images, and all
subsample the same stratified training indices per (fraction, seed).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn", "26_baselines")
import metrics_ordinal as M, retina_features as RF                   # noqa: E402
from kg_adjacency import build_adjacency, kg_operators               # noqa: E402
from run_ordinal import fit_predict, SPLITS                          # noqa: E402

FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)


def strat(y, frac, rng):
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(2, int(round(len(idx) * frac))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P.RESULTS / "baseline_curves.json"))
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    cfg = yaml.safe_load(P.ONTOLOGY.read_text())
    NC = len(cfg["levels"]); n_cut = NC - 1
    X, Y, IDS, names, prior, level, meta = RF.build(
        P.PACK, P.NPZ, P.RESULTS, ("retina_p3",), P.DEBATES, SPLITS)
    Z = RF.standardise(X, SPLITS)
    A_, _ = build_adjacency(P.PACK, cfg); Ap, An = kg_operators(A_, prior)
    sel = json.loads((P.RESULTS / "ordinal_full.json").read_text())["selected"]["A"]
    ytr, yte = Y["train"].astype(int), Y["test"].astype(int)
    ref = M.referable(yte) == 1

    # B2 graphs, keyed by sample id so the same subsample can be applied
    import torch
    torch.set_num_threads(4)
    from b2_graph_retina import build as b2build
    import b2_ordinal as B2
    b2cfg = json.loads((P.RESULTS / "b2_retina.json").read_text())
    tau = b2cfg["ordinal"]["_config"]["tau_transfer"]
    G = b2build(str(P.PACK / "multi6"), tau, rule="ordinal")
    gtr = {g["sid"]: g for g in G if g["split"] == "train"}
    gte = [g for g in G if g["split"] == "test"]
    yte_b2 = np.array([int(g["y"]) for g in gte])
    ref_b2 = M.referable(yte_b2) == 1

    # B1 scores are fixed per image (no learned head); its curve moves only
    # through the tone/tau grid re-selected at each training size.
    import analyse_b1_retina as B1
    b1 = B1.load_corpus(str(P.PACK / "catfish_b1"))
    b1tr, b1te = b1["train"], b1["test"]
    yb1 = np.array([d["gold_grade"] for d in b1te], int)
    ref_b1 = M.referable(yb1) == 1
    confs = [d["trigger"]["mean_conf"] for d in b1tr if d["trigger"]["mean_conf"] is not None]
    grid = [None] + list(np.quantile(confs, [0.25, 0.5, 0.75]))

    print(f"{'n_train':>8s} {'v3-A':>16s} {'v3-nograph':>16s} {'B2':>16s} {'B1':>16s}")
    rows = []
    for frac in FRACTIONS:
        acc = {k: [] for k in ("v3", "v3ng", "b2", "b1")}
        ns = []
        for sd in SEEDS:
            rng = np.random.default_rng(100 + sd)
            keep = strat(ytr, frac, rng); ns.append(len(keep))
            ids = set(int(IDS["train"][i]) for i in keep)

            s, _, _, _ = fit_predict("A", Z["train"][keep], Y["train"][keep],
                                     Z["test"], Ap, An, n_cut,
                                     sel["l2"], sel["kdim"], sd, prior)
            acc["v3"].append(M.fast_auc(s, ref))
            Zr = np.zeros_like(Ap)
            s0, _, _, _ = fit_predict("A", Z["train"][keep], Y["train"][keep],
                                      Z["test"], Zr, Zr, n_cut,
                                      sel["l2"], sel["kdim"], sd, prior)
            acc["v3ng"].append(M.fast_auc(s0, ref))

            sub = [g for sid, g in gtr.items() if sid in ids]
            sb, _, _ = B2.train_eval_coral(sub, gte, n_cut, sd, epochs=a.epochs,
                                           anchored=False)
            acc["b2"].append(M.fast_auc(np.asarray(sb), ref_b2))

            sub1 = [d for d in b1tr if int(d["sample_id"]) in ids]
            best = max(((B1.cv_ref_auc(sub1, "gated", t, q), t, q)
                        for t in B1.TONES for q in grid), key=lambda x: x[0]) \
                if len(sub1) > 20 else (0, "collaborative", None)
            sc = B1.scores(b1te, "gated", best[1], best[2])
            acc["b1"].append(M.fast_auc(sc, ref_b1))

        row = {"frac": frac, "n_train": int(np.mean(ns))}
        for k in acc:
            v = np.array(acc[k]); row[k] = [float(v.mean()), float(v.std())]
        rows.append(row)
        print(f"{row['n_train']:8d} " + " ".join(
            f"{row[k][0]:9.4f}+-{row[k][1]:5.4f}" for k in ("v3", "v3ng", "b2", "b1")))

    n = np.array([r["n_train"] for r in rows], float)
    slopes = {k: float(np.polyfit(np.log(n), [r[k][0] for r in rows], 1)[0])
              for k in ("v3", "v3ng", "b2", "b1")}
    print("\n  slope vs log n:", {k: round(v, 4) for k, v in slopes.items()})
    print("  A method still climbing is data-starved; a plateau means the tie is real.")
    Path(a.out).write_text(json.dumps(
        {"curve": rows, "slopes": slopes, "metric": "referable AUC",
         "n_test": int(len(yte))}, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
