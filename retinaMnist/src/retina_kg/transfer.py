"""Cross-ontology transfer: train the trunk on one dataset, read it out on the other.

The architecture splits into two parts with different index sets, and only one of
them is dataset-specific:

    W0, Wp, Wn (d x k), b (k)   indexed by FEATURE CHANNEL   -- 26 of 65 params
    u (F x k), theta            indexed by FINDING IDENTITY  -- 39 of 65 params

The four channels mean the same thing in both trees: [probe p(present), argument
mass, net stance, ontology prior]. `u` does not -- breast finding 0 is
`architectural_distortion`, retina finding 0 is `microaneurysm`, and F=16 in both
is coincidence. So the trunk can cross and the readout cannot, which makes one
question askable: is "how to weigh probe evidence against argument mass against
an ontology prior" a skill independent of the ontology it was learned on?

This is a stronger transfer than 16_external/22_transfer, which moved between
datasets sharing the BI-RADS ontology. Here the ontology, the modality and the
label space all change.

FOUR ARMS per direction. The third is the one that makes the result mean anything:

  ceiling        trunk trained on the TARGET, readout on the target
  transfer       trunk trained on the SOURCE, FROZEN, readout on the target
  random trunk   trunk drawn from the init distribution, FROZEN, readout on the
                 target, 5 draws. A frozen random projection followed by a fitted
                 per-node readout is already expressive; without this control a
                 good transfer number says nothing about the source ontology.
  0-parameter    sum_f prior_f * p_yes_f. Nothing is trained at all, so it
                 transfers trivially and is the floor any learned arm must beat.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")

import ordinal, metrics_ordinal as M, retina_features as RF          # noqa: E402
import thothgnn3                                                     # noqa: E402
from kg_adjacency import build_adjacency, kg_operators               # noqa: E402
from hetgraph import SPLITS                                          # noqa: E402

TRUNK = ("W0", "Wp", "Wn", "b")
READOUT = ("u", "theta")


# ------------------------------------------------------------------- corpora --
def load_retina():
    cfg = yaml.safe_load(P.ONTOLOGY.read_text())
    X, Y, IDS, names, prior, level, meta = RF.build(
        P.PACK, P.NPZ, P.RESULTS, ("retina_p3",), P.DEBATES, SPLITS)
    Z = RF.standardise(X, SPLITS)
    A, _ = build_adjacency(P.PACK, cfg)
    Ap, An = kg_operators(A, prior)
    return dict(name="retina", Z=Z, Y={s: Y[s].astype(float) for s in SPLITS},
                Ap=Ap, An=An, prior=prior, n_cut=len(cfg["levels"]) - 1,
                n_class=len(cfg["levels"]), names=names, Xraw=X)


def load_breast():
    thothgnn3.PHRASE_COLS = (1,)                       # P3 alone, per round 4
    d, names, prior = thothgnn3.build(str(P.BREAST_DEBATES))
    probes = thothgnn3.probe_block(names)
    X = {s: thothgnn3.node_tensor(d[s], names, prior, probes, s) for s in SPLITS}
    Y = {s: d[s]["y"] for s in SPLITS}
    mu = X["train"].reshape(-1, X["train"].shape[-1]).mean(0)
    sd = X["train"].reshape(-1, X["train"].shape[-1]).std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    Ap, An = thothgnn3.kg_operators(names, prior)
    return dict(name="breast", Z=Z, Y=Y, Ap=Ap, An=An, prior=prior,
                n_cut=1, n_class=2, names=names, Xraw=X)


# ------------------------------------------------------------------- fitting --
def fit_readout_only(D, trunk, l2=0.1, kdim=2, epochs=800, lr=0.15, seed=0):
    """Same optimiser as ordinal.fit, but the trunk is held fixed.

    Only `u` and `theta` receive updates; the trunk keys keep the values handed
    in. The L2 penalty centre on u[:,0] is the TARGET ontology's prior, because
    that is what the target's findings mean.
    """
    Z, y = D["Z"]["train"], D["Y"]["train"]
    F, d = Z.shape[1], Z.shape[2]
    rng = np.random.default_rng(seed)
    Q = ordinal.init(F, d, kdim, D["n_cut"], rng)
    for k in TRUNK:
        Q[k] = np.array(trunk[k], dtype=float, copy=True)
    centre = np.zeros((F, kdim)); centre[:, 0] = D["prior"]
    m = {k: np.zeros_like(Q[k]) for k in READOUT}
    for _ in range(epochs):
        G, _ = ordinal.grads(Q, Z, D["Ap"], D["An"], y, l2, centre, D["n_cut"])
        for k in READOUT:
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            Q[k] = Q[k] - lr * m[k]
    return Q


def train_trunk(D, l2=0.1, kdim=2, seed=0):
    """Fit the whole model on D and hand back only its trunk."""
    Q = ordinal.fit(D["Z"]["train"], D["Y"]["train"], D["Ap"], D["An"], D["n_cut"],
                    l2=l2, kdim=kdim, seed=seed, prior_centre=D["prior"])
    return {k: Q[k] for k in TRUNK}


def zero_param(D, split="test"):
    """sum_f prior_f * p_yes_f on the RAW (unstandardised) probe channel."""
    return (D["Xraw"][split][:, :, 0] * D["prior"][None, :]).sum(axis=1)


# ------------------------------------------------------------------ reporting --
def score(D, Q):
    s, p, g = ordinal.predict(Q, D["Z"]["test"], D["Ap"], D["An"])
    return s, p, g


def summarise(D, s, g):
    y = D["Y"]["test"].astype(int)
    if D["n_class"] == 2:
        return {"auc": M.auc(list(s[y == 1]), list(s[y == 0]))}
    yr = M.referable(y)
    return {"refauc": M.auc(list(s[yr == 1]), list(s[yr == 0])),
            "qwk": M.qwk(y, g, D["n_class"]),
            "acc": float((g == y).mean()),
            "macroR": M.macro_recall(y, g, D["n_class"])}


def fmt(D, m):
    if D["n_class"] == 2:
        return f"AUC {m['auc']:.4f}"
    return (f"refAUC {m['refauc']:.4f}  QWK {m['qwk']:+.4f}  "
            f"ACC {m['acc']:.4f}  macroR {m['macroR']:.4f}")


def run(src, tgt, l2, kdim, seed, res):
    print(f"\n=== {src['name']} -> {tgt['name']} "
          f"(F {len(src['names'])}->{len(tgt['names'])}, d={src['Z']['train'].shape[2]}, k={kdim}) ===")

    Q = ordinal.fit(tgt["Z"]["train"], tgt["Y"]["train"], tgt["Ap"], tgt["An"],
                    tgt["n_cut"], l2=l2, kdim=kdim, seed=seed, prior_centre=tgt["prior"])
    s, p, g = score(tgt, Q); mc = summarise(tgt, s, g)
    print(f"  {'ceiling (target trunk)':28s} {fmt(tgt, mc)}")

    trunk = train_trunk(src, l2=l2, kdim=kdim, seed=seed)
    Qt = fit_readout_only(tgt, trunk, l2=l2, kdim=kdim, seed=seed)
    st, pt, gt = score(tgt, Qt); mt = summarise(tgt, st, gt)
    print(f"  {'TRANSFER (source trunk)':28s} {fmt(tgt, mt)}")

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(5):
        rt = {k: rng.normal(0, 0.3, np.shape(trunk[k])) for k in TRUNK}
        rt["b"] = np.zeros_like(trunk["b"])
        Qr = fit_readout_only(tgt, rt, l2=l2, kdim=kdim, seed=seed)
        sr, pr, gr = score(tgt, Qr)
        draws.append(summarise(tgt, sr, gr))
    key = "auc" if tgt["n_class"] == 2 else "refauc"
    dv = np.array([d[key] for d in draws])
    print(f"  {'random trunk (5 draws)':28s} {key} {dv.mean():.4f} +- {dv.std():.4f} "
          f"[{dv.min():.4f}, {dv.max():.4f}]")
    z = (mt[key] - dv.mean()) / dv.std() if dv.std() > 0 else float("nan")
    print(f"  {'-> transfer minus random':28s} {mt[key]-dv.mean():+.4f}  ({z:+.2f} sd)")

    z0 = zero_param(tgt)
    y = tgt["Y"]["test"].astype(int)
    t0 = (y == 1) if tgt["n_class"] == 2 else M.referable(y)
    a0 = M.auc(list(z0[t0 == 1]), list(z0[t0 == 0]))
    print(f"  {'0-parameter KG-signed sum':28s} {key} {a0:.4f}")

    res[f"{src['name']}->{tgt['name']}"] = {
        "ceiling": mc, "transfer": mt,
        "random_trunk": {"mean": float(dv.mean()), "sd": float(dv.std()),
                         "draws": dv.tolist()},
        "transfer_minus_random": float(mt[key] - dv.mean()), "z": float(z),
        "zero_param_auc": a0,
        "retained": float(mt[key] / mc[key]) if mc[key] else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.03)
    ap.add_argument("--kdim", type=int, default=2, help="must match across trees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(P.RESULTS / "transfer.json"))
    a = ap.parse_args()

    R, B = load_retina(), load_breast()
    for D in (R, B):
        print(f"{D['name']:7s} F={len(D['names'])}  d={D['Z']['train'].shape[2]}  "
              f"n={[len(D['Y'][s]) for s in SPLITS]}  classes={D['n_class']}")
    assert R["Z"]["train"].shape[2] == B["Z"]["train"].shape[2], "channel counts differ"

    res = {"config": {"l2": a.l2, "kdim": a.kdim, "seed": a.seed,
                      "trunk_params": TRUNK, "readout_params": READOUT}}
    run(B, R, a.l2, a.kdim, a.seed, res)
    run(R, B, a.l2, a.kdim, a.seed, res)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
