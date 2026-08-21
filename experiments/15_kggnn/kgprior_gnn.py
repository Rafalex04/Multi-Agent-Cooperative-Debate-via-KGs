"""KG-prior anchor + KG-structured graph refinement.

Anchoring on the trained 32-feature probe model made the graph term redundant --
gamma settled at -0.04 and the model reported the anchor back, because the anchor
already fits the training data well enough that there is nothing left to learn.
That is a real finding, but it makes the architecture untestable.

So anchor on the knowledge graph's own unsupervised prior instead:

    anchor_i = sum_f  stance(f) * p_present(f, image i)

Zero fitted parameters -- purely the KG's statement of what each finding means,
applied to the measured probes. It scores 0.7629 on its own. Everything above
that has to come from the graph term, which is where the debate features, the
sign-separated message passing and the ontology edges live.

This is also the architecture the project has been arguing for: the KG supplies
the prior AND the topology, the debate supplies per-finding evidence, and the
learned part is a correction to the prior rather than a replacement for it.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc                                  # noqa: E402
from features import COLS, build                                      # noqa: E402
from kg_graph import build_adjacency, normalised                      # noqa: E402
from thoth_kg import ThothKG, fit                                     # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def score_stats(sc, Y):
    a = {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS}
    sf = np.concatenate([sc["train"], sc["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    return a, bacc(sc["test"].tolist(), Y["test"].tolist(), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--hiddens", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--readout", default="sum")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, Xp, Y, findings = build(args.debates, tag="probe")
    _, Xn, _, _ = build(args.debates, tag="probeneg")
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    # ---- KG-prior anchor: no fitted parameters at all ----
    pm = {s: (Xp[s][:, :, 0] + Xn[s][:, :, 0]) / 2 for s in SPLITS}
    anc = {s: (pm[s] * prior).sum(1) for s in SPLITS}
    am, asd = anc["train"].mean(), anc["train"].std()
    anc = {s: (anc[s] - am) / asd for s in SPLITS}
    a, b = score_stats(anc, Y)
    print(f"\n  {'KG-prior anchor (0 params)':30s} train {a['train']:.4f} "
          f"val {a['val']:.4f} TEST {a['test']:.4f}  bAcc {b:.4f}")

    dc = [COLS.index(c) for c in ("net", "presence", "mass", "endorsed", "disputed", "open")]
    eye = np.eye(F)
    X = {s: np.concatenate([
        Xp[s][:, :, 0:1], Xn[s][:, :, 0:1], Xp[s][:, :, dc],
        np.broadcast_to(prior[None, :, None], (len(Y[s]), F, 1)),
        np.broadcast_to(eye, (len(Y[s]), F, F))], axis=2) for s in SPLITS}
    mu, sd = X["train"].mean((0, 1)), X["train"].std((0, 1))
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xz = {s: (X[s] - mu) / sd for s in SPLITS}
    C = X["train"].shape[2]

    A, info = build_adjacency(_KG, [f for f, _ in findings])
    An_ = normalised(A, self_loops=False)
    same = (prior[:, None] * prior[None, :]) > 0
    Ap, Am = An_ * same, An_ * (~same)
    rng = np.random.default_rng(0); perm = rng.permutation(F)
    Z0 = np.zeros((F, F))
    arms = {"signed KG": (Ap, Am),
            "shuffled KG": (Ap[np.ix_(perm, perm)], Am[np.ix_(perm, perm)]),
            "unsigned KG": (Ap + Am, Z0),
            "no-prop": (Z0, Z0)}

    res = {"anchor": {"test": a["test"], "bacc": b}}
    print(f"\n  {'arm':14s} {'h':>3s}  {'TEST':>16s} {'bAcc':>16s}  {'val':>7s} {'gamma':>7s}")
    for h in args.hiddens:
        for name, (P, M) in arms.items():
            ts, bs, vs, gs = [], [], [], []
            for seed in range(args.seeds):
                best = None
                for l2 in (1e-2, 3e-2, 1e-1):
                    m = ThothKG(C, h, P, M, seed=seed, gamma0=0.0, readout=args.readout)
                    fit(m, Xz["train"], anc["train"], Y["train"], l2, args.epochs)
                    sc = {k: m.forward(Xz[k], anc[k])[0] for k in SPLITS}
                    av = auc(list(sc["val"][Y["val"] == 1]), list(sc["val"][Y["val"] == 0]))
                    if best is None or av > best[0]:
                        best = (av, sc, m.gamma)
                av, sc, g = best
                aa, bb = score_stats(sc, Y)
                ts.append(aa["test"]); bs.append(bb); vs.append(av); gs.append(g)
            print(f"  {name:14s} {h:3d}  {np.mean(ts):.4f}+-{np.std(ts):.4f} "
                  f"{np.mean(bs):.4f}+-{np.std(bs):.4f}  {np.mean(vs):7.4f} {np.mean(gs):+7.3f}")
            res[f"{name}|h{h}"] = {"test": float(np.mean(ts)), "test_sd": float(np.std(ts)),
                                   "bacc": float(np.mean(bs)), "bacc_sd": float(np.std(bs)),
                                   "val": float(np.mean(vs)), "gamma": float(np.mean(gs))}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
