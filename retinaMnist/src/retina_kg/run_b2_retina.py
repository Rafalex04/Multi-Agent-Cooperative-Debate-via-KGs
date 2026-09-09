"""B2 GraphGeo on retina, with the same two ordinal heads as ThothGNN v3.

Our implementation of GraphGeo (Zheng et al., arXiv 2511.00908); no released code
used. Label every result "our implementation of GraphGeo".

Arms mirror `26_baselines/b2_train.py`, with the ordinal head substituted for the
single logit:

  B2-full          relation-specific encoder, sum readout, NO anchor
  B2-anchored      + beta * mean claim grade. The key row: if it matches
                   ThothGNN v3, our contribution is the anchor, not the nodes
  B2-no-relation   one shared W for every edge type (relation ablation)

Each is run under BOTH edge rules from b2_graph_retina -- the faithful binary
stance rule and the ordinal grade-proximity rule -- because the binary rule
leaves 36% of graphs with no conflict edge at all on this single-lineage corpus.

tau_transfer is selected by 5-fold CV on TRAIN. 5 fixed seeds, mean +- sd.
ONE test evaluation per frozen config.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "26_baselines")
import metrics_ordinal as M                                          # noqa: E402
import b2_ordinal as B2                                              # noqa: E402
from b2_graph_retina import build, edge_stats, SPLITS, REL           # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
NC = 5
NCUT = NC - 1
TAUS = (0.0, 0.001, 0.01, 0.05, 0.2)


def split_graphs(g):
    return {s: [x for x in g if x["split"] == s] for s in SPLITS}


def ref_auc(y, s):
    return M.fast_auc(s, M.referable(np.asarray(y, int)) == 1)


def evaluate(tr, te, seed, head, **kw):
    if head == "A":
        s, p, g = B2.train_eval_coral(tr, te, NCUT, seed, **kw)
    else:
        s, p, g = B2.train_eval_stacked(tr, te, NCUT, seed, **kw)
    return (s if np.ndim(s) == 1 else np.asarray(s)[:, 1]), g


def folds(n, k=5, seed=0):
    o = np.random.default_rng(seed).permutation(n)
    return [np.isin(np.arange(n), o[i::k]) for i in range(k)]


def cv_ref_auc(graphs, head, seeds=(0, 1), **kw):
    y = np.array([g["y"] for g in graphs], int)
    vals = []
    for sd in seeds:
        oof = np.zeros(len(graphs))
        for te in folds(len(graphs), 5, seed=0):
            tr = ~te
            s, _ = evaluate([g for g, m in zip(graphs, tr) if m],
                            [g for g, m in zip(graphs, te) if m], sd, head, **kw)
            oof[te] = s
        vals.append(ref_auc(y, oof))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(P.PACK / "multi6"))
    ap.add_argument("--heads", default="A")
    ap.add_argument("--rules", default="ordinal,binary")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default=str(P.RESULTS / "b2_retina.json"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    res = {}

    for rule in a.rules.split(","):
        print(f"\n################  edge rule: {rule}  ################")
        # --- select tau_transfer on TRAIN only -------------------------------
        best, tab = None, {}
        for tau in TAUS:
            g = build(a.corpus, tau, rule=rule)
            tr = [x for x in g if x["split"] == "train"]
            if not tr:
                print("  no training graphs"); return
            v = cv_ref_auc(tr, "A", epochs=a.epochs)
            tab[str(tau)] = v
            print(f"  cv tau_transfer={tau:<6} refAUC {v:.4f}")
            if best is None or v > best[0]:
                best = (v, tau)
        cv, tau = best
        print(f"  -> selected tau_transfer={tau} (cv {cv:.4f})")

        g = build(a.corpus, tau, rule=rule)
        S = split_graphs(g)
        es = edge_stats(g)
        print(f"  graphs {[len(S[s]) for s in SPLITS]}   "
              f"conflict {es['conflict']['mean_per_graph']:.2f}/graph, "
              f"{es['conflict']['graphs_with_none']} with none")
        y = np.array([x["y"] for x in S["test"]], int)

        arms = {"B2-full":        dict(anchored=False),
                "B2-anchored":    dict(anchored=True),
                "B2-no-relation": dict(anchored=False, shared_rel=True)}
        rr = {"_config": {"tau_transfer": tau, "cv_ref_auc": cv,
                          "edge_stats": es, "epochs": a.epochs,
                          "n": {s: len(S[s]) for s in SPLITS}}}
        for head in a.heads.split(","):
            print(f"\n  --- head {head} ({'cumulative-link' if head=='A' else 'stacked'}) ---")
            print(f"  {'arm':18s} {'refAUC':>16s} {'QWK':>16s} {'ACC':>14s}")
            for lab, kw in arms.items():
                ra, qw, ac = [], [], []
                for sd in SEEDS:
                    s, gh = evaluate(S["train"], S["test"], sd, head,
                                     epochs=a.epochs, **kw)
                    ra.append(ref_auc(y, s)); qw.append(M.qwk(y, gh, NC))
                    ac.append(float((np.asarray(gh) == y).mean()))
                key = f"{head}:{lab}"
                rr[key] = {"referable_auc": [float(np.mean(ra)), float(np.std(ra))],
                           "qwk": [float(np.mean(qw)), float(np.std(qw))],
                           "acc": [float(np.mean(ac)), float(np.std(ac))],
                           "seeds": list(SEEDS)}
                print(f"  {lab:18s} {np.mean(ra):8.4f} +-{np.std(ra):6.4f} "
                      f"{np.mean(qw):+8.4f} +-{np.std(qw):6.4f} "
                      f"{np.mean(ac):6.4f} +-{np.std(ac):6.4f}")
        res[rule] = rr

    try:
        of = json.loads((P.RESULTS / "ordinal_full.json").read_text())
        v3 = of["A: GNN (real KG)"]["referable_auc"]
        print(f"\nThothGNN v3 (head A) reference: refAUC {v3:.4f}")
        res["_thothgnn3_A_referable_auc"] = v3
    except Exception:
        pass
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
