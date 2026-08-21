"""The full model: KG finding graph, sign-separated message passing, probe anchor.

Node set        the 16 BI-RADS findings from the knowledge graph
Node features   both probe phrasings (present / free-of), the debate's treatment
                of that finding, the KG's stance prior, and a node identity
Edges           two findings are adjacent when a lesion in the KG presents both,
                split by whether the KG gives them the same or opposing stance
Readout         bipolar attention over findings
Score           beta * logit(anchor) + gamma * graph,  gamma init 0

The anchor is the strongest available readout, per the Stage 4 finding that
NO-ANCHOR was the only measurable effect in the whole ablation table. Here that
is the 32-feature probe model (0.8137), fitted on train only.

Controls, every one with matched density and degree:
  shuffled KG   the same signed matrices with node identities permuted
  no-prop       no message passing at all
  unsigned      one matrix over A+ + A-, ablating sign separation

If `signed KG` does not beat `shuffled KG`, the ontology contributes nothing and
that is what gets reported.
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
from kg_tensor import predict, train_head                             # noqa: E402
from thoth_kg import ThothKG, fit                                     # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cvfit(Xd, Y, l2s=(1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0), seed=0):
    mu, sd = Xd["train"].mean(0), Xd["train"].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (Xd[s] - mu) / sd for s in SPLITS}
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(Y["train"])), 5)
    best = None
    for l2 in l2s:
        sc = np.zeros(len(Y["train"]))
        for fo in folds:
            m = np.zeros(len(Y["train"]), dtype=bool); m[fo] = True
            w, b, be = train_head(Z["train"][~m], np.zeros((~m).sum()), Y["train"][~m],
                                  np.zeros(Z["train"].shape[1]), l2, 1500)
            sc[m] = predict(Z["train"][m], np.zeros(int(m.sum())), w, b, be)
        c = auc(list(sc[Y["train"] == 1]), list(sc[Y["train"] == 0]))
        if best is None or c > best[0]:
            best = (c, l2)
    cv, l2 = best
    w, b, be = train_head(Z["train"], np.zeros(len(Y["train"])), Y["train"],
                          np.zeros(Z["train"].shape[1]), l2, 1500)
    return {s: predict(Z[s], np.zeros(len(Y[s])), w, b, be) for s in SPLITS}, cv


def report(name, sc, Y, out=None):
    a = {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS}
    sf = np.concatenate([sc["train"], sc["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    b = bacc(sc["test"].tolist(), Y["test"].tolist(), t)
    print(f"  {name:26s} train {a['train']:.4f} val {a['val']:.4f} "
          f"TEST {a['test']:.4f}  bAcc {b:.4f}")
    if out is not None:
        out[name] = {"train": a["train"], "val": a["val"], "test": a["test"], "bacc": b}
    return a["test"], b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--readout", default="attention", choices=("attention", "sum"))
    ap.add_argument("--no-anchor", action="store_true",
                    help="zero the anchor so the graph must carry the whole score")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, Xp, Y, findings = build(args.debates, tag="probe")
    _, Xn, _, _ = build(args.debates, tag="probeneg")
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    # ---- anchor: the 32-feature probe model, train-only ----
    A32 = {s: np.hstack([Xp[s][:, :, 0], Xn[s][:, :, 0]]) for s in SPLITS}
    res = {}
    anc_sc, cv = cvfit(A32, Y)
    print(f"\n(anchor probe model, cv {cv:.4f})")
    report("ANCHOR probe-32 (no GNN)", anc_sc, Y, res)
    anchors = ({s: np.zeros(len(Y[s])) for s in SPLITS} if args.no_anchor
               else {s: anc_sc[s] for s in SPLITS})
    if args.no_anchor:
        # With a strong anchor the optimiser has no pressure to move gamma off
        # zero -- it reports the anchor back. Zeroing it makes the graph carry
        # the entire score, which is the only way to see what the topology is
        # actually worth.
        print("ANCHOR DISABLED: the graph term carries the whole score")

    # ---- node features ----
    dcols = [COLS.index(c) for c in ("net", "presence", "mass", "endorsed", "disputed", "open")]
    eye = np.eye(F)
    X = {}
    for s in SPLITS:
        n = len(Y[s])
        X[s] = np.concatenate([
            Xp[s][:, :, 0:1], Xn[s][:, :, 0:1],
            Xp[s][:, :, dcols],
            np.broadcast_to(prior[None, :, None], (n, F, 1)),
            np.broadcast_to(eye, (n, F, F)),
        ], axis=2)
    C = X["train"].shape[2]
    mu, sd = X["train"].mean((0, 1)), X["train"].std((0, 1))
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xz = {s: (X[s] - mu) / sd for s in SPLITS}
    print(f"node features: {C}  (2 probe, {len(dcols)} debate, 1 prior, {F} identity)")

    # ---- KG topology ----
    A, info = build_adjacency(_KG, [f for f, _ in findings])
    An_ = normalised(A, self_loops=False)
    same = (prior[:, None] * prior[None, :]) > 0
    Ap, Am = An_ * same, An_ * (~same)
    rng = np.random.default_rng(0); perm = rng.permutation(F)
    Z0 = np.zeros((F, F))
    print(f"KG edges {info['n_edges']}/{info['n_pairs']}  "
          f"same-stance {int((Ap>0).sum()//2)}  opposing {int((Am>0).sum()//2)}")

    arms = {
        "GNN signed KG": (Ap, Am),
        "GNN unsigned KG": (Ap + Am, Z0),
        "GNN shuffled KG": (Ap[np.ix_(perm, perm)], Am[np.ix_(perm, perm)]),
        "GNN no-prop": (Z0, Z0),
    }
    print()
    for name, (P, M) in arms.items():
        ts, bs, vs, gs = [], [], [], []
        for seed in range(args.seeds):
            best = None
            for l2 in (1e-2, 1e-1):
                m = ThothKG(C, args.hidden, P, M, seed=seed, readout=args.readout,
                            gamma0=1.0 if args.no_anchor else 0.0)
                fit(m, Xz["train"], anchors["train"], Y["train"], l2, args.epochs)
                sc = {k: m.forward(Xz[k], anchors[k])[0] for k in SPLITS}
                av = auc(list(sc["val"][Y["val"] == 1]), list(sc["val"][Y["val"] == 0]))
                if best is None or av > best[0]:
                    best = (av, sc, m.gamma)
            av, sc, g = best
            at = auc(list(sc["test"][Y["test"] == 1]), list(sc["test"][Y["test"] == 0]))
            sf = np.concatenate([sc["train"], sc["val"]])
            yf = np.concatenate([Y["train"], Y["val"]])
            t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
            ts.append(at); bs.append(bacc(sc["test"].tolist(), Y["test"].tolist(), t))
            vs.append(av); gs.append(g)
        print(f"  {name:26s} TEST {np.mean(ts):.4f}+-{np.std(ts):.4f}  "
              f"bAcc {np.mean(bs):.4f}+-{np.std(bs):.4f}  val {np.mean(vs):.4f}  "
              f"gamma {np.mean(gs):+.3f}")
        res[name] = {"test": float(np.mean(ts)), "test_sd": float(np.std(ts)),
                     "bacc": float(np.mean(bs)), "bacc_sd": float(np.std(bs)),
                     "val": float(np.mean(vs)), "gamma": float(np.mean(gs))}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
