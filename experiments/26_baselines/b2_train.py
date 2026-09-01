"""B2 training, CV selection and ablations. One test evaluation per frozen config.

tau_transfer - the only new hyperparameter - is selected by 5-fold CV on the 546
BreastMNIST TRAINING graphs. 5 fixed seeds everywhere, mean +- sd reported.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc, bacc, paired_bootstrap          # noqa: E402
from b2_graph import build, roundtrip_test, summarise        # noqa: E402
from b2_model import GraphGeo, collate, drop_edge, REL       # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def train_eval(tr, te, seed, epochs=300, lr=3e-3, wd=5e-4, dropedge=0.2, **kw):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    B, T = collate(tr), collate(te)
    m = GraphGeo(d_img=B["x"].shape[1], **kw)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        e = drop_edge(B["edges"], dropedge, gen)
        s = m(B["x"], e, B["batch"], B["n"], B["slot"], B["anchor"])
        lossf(s, B["y"]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(T["x"], T["edges"], T["batch"], T["n"], T["slot"],
                 T["anchor"]).numpy()


def folds(n, k=5, seed=0):
    o = np.random.default_rng(seed).permutation(n)
    return [np.isin(np.arange(n), o[i::k]) for i in range(k)]


def cv_auc(graphs, k=5, **kw):
    y = np.array([g["y"] for g in graphs])
    vals = []
    for seed in SEEDS[:3]:                        # 3 seeds during selection
        oof = np.zeros(len(graphs))
        for te in folds(len(graphs), k, seed=0):
            tr = ~te
            oof[te] = train_eval([g for g, m in zip(graphs, tr) if m],
                                 [g for g, m in zip(graphs, te) if m], seed, **kw)
        vals.append(auc(y, oof))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()

    print("=== graph build + integrity ===")
    best_tau, best_cv, cvtab = None, -1, {}
    for tau in (0.0, 0.001, 0.01, 0.05, 0.2):
        g = build(a.corpus, tau)
        tr = [x for x in g if x["split"] == "train"]
        if len(tr) < 50:
            print(f"only {len(tr)} training graphs -- corpus not ready"); return
        if best_tau is None:
            print(summarise(g))
            ok, tot = roundtrip_test(g)
            print(f"round-trip: {ok}/{tot} {'OK' if ok == tot else 'FAILED'}")
            assert ok == tot, "graph serialisation is lossy"
            print(f"\n=== CV selection on {len(tr)} training graphs (test untouched) ===")
            print(f"{'tau_transfer':>13s} {'CV AUC':>9s} {'sd':>7s} {'transfer edges/graph':>21s}")
        m, s = cv_auc(tr, epochs=a.epochs)
        ne = float(np.mean([len(x["edges"]["transfer"][0]) for x in g]))
        cvtab[tau] = {"cv_auc": m, "sd": s, "edges": ne}
        print(f"{tau:13.3f} {m:9.4f} {s:7.4f} {ne:21.2f}")
        if m > best_cv:
            best_cv, best_tau = m, tau
    print(f"\nFROZEN: tau_transfer={best_tau}  CV AUC {best_cv:.4f}")

    g = build(a.corpus, best_tau)
    tr = [x for x in g if x["split"] in ("train", "val")]
    te = [x for x in g if x["split"] == "test"]
    y = np.array([x["y"] for x in te])
    print(f"\n=== TEST (n={len(te)}), one evaluation per frozen config ===")

    arms = {
        "B2-full":          dict(),
        "B2-no-transfer":   dict(relations=("agree", "conflict")),
        "B2-no-relation":   dict(shared_rel=True),
        "B2-no-agent-emb":  dict(use_agent_emb=False),
        "B2-mean-readout":  dict(readout="mean"),
        "B2-anchored":      dict(anchored=True),
    }
    res, S = {}, {}
    print(f"{'arm':20s} {'AUC':>16s} {'bAcc':>8s}  params")
    for lab, kw in arms.items():
        sc = np.array([train_eval(tr, te, s, epochs=a.epochs, **kw) for s in SEEDS])
        aucs = [auc(y, r) for r in sc]
        mean_s = sc.mean(0); S[lab] = mean_s
        nparam = sum(p.numel() for p in
                     GraphGeo(d_img=collate(tr)["x"].shape[1], **kw).parameters())
        thr = float(np.median(mean_s))
        res[lab] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                    "auc_ens": float(auc(y, mean_s)),
                    "bacc": float(bacc(y, mean_s, thr)), "params": int(nparam)}
        print(f"{lab:20s} {np.mean(aucs):8.4f} +- {np.std(aucs):.4f} "
              f"{res[lab]['bacc']:8.4f}  {nparam:6d}")

    print("\nKEY CONTRAST")
    p = paired_bootstrap(y, S["B2-anchored"], S["B2-full"], n=4000)
    print(f"  B2-anchored vs B2-full   delta "
          f"{res['B2-anchored']['auc_ens']-res['B2-full']['auc_ens']:+.4f}  P {p:.3f}")
    res["anchored_vs_full"] = {"P": float(p)}
    res["frozen"] = {"tau_transfer": best_tau, "cv_auc": best_cv, "cv_table": cvtab,
                     "n_test": len(te), "n_train": len(tr), "seeds": list(SEEDS)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
