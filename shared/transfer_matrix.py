"""Cross-ONTOLOGY transfer across all three datasets: breast, retina, derma.

The architecture splits by index set, and only one half is dataset-specific:

    W0, Wp, Wn (d x k), b (k)   FEATURE CHANNEL   -> crosses
    u, theta / c0               FINDING IDENTITY  -> cannot

The four channels mean the same thing in all three trees, and the trunk is free
in BOTH F and C. So a trunk trained on 16 BI-RADS findings for a binary target
can be frozen and read out over 55 dermoscopy findings for a 7-class one. That is
the strongest form of the generalisation claim this project can make: ontology,
modality, label space AND label cardinality all change.

  ceiling        trunk trained on the TARGET
  transfer       trunk trained on the SOURCE, FROZEN, readout refitted on target
  random trunk   trunk from the init distribution, FROZEN, 3 draws -- the control
                 that makes the number mean something, since a frozen random
                 projection plus a fitted per-node readout is already expressive
  0-parameter    the ontology rule alone, nothing trained

Reported on each target's own headline metric, because they are not comparable
across targets: breast AUC, retina referable AUC, derma macro-recall.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in ("experiments/14_kgtensor", "experiments/15_kggnn", "experiments/17_hetgnn",
          "retinaMnist/src/retina_kg", "dermaMnist/src/derma_kg"):
    sys.path.insert(0, str(REPO / p))

import thothgnn3, ordinal, nominal                                   # noqa: E402
import metrics_ordinal as MO, metrics_nominal as MN                  # noqa: E402
import retina_features as RF, derma_features as DF                   # noqa: E402
from hetgraph import SPLITS                                          # noqa: E402

TRUNK = ("W0", "Wp", "Wn", "b")
DERMA_CAP = 2500        # derma fits are 563 s at n=7007; the trunk question does
                        # not need all of it, and every arm uses the same cap


def _std(X):
    tr = X["train"].reshape(-1, X["train"].shape[-1])
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {s: (X[s] - mu) / sd for s in X}


def load_breast():
    thothgnn3.PHRASE_COLS = (1,)
    d, names, prior = thothgnn3.build(str(REPO / "breastMnist/data/breast/debates_v5q"))
    pb = thothgnn3.probe_block(names)
    X = {s: thothgnn3.node_tensor(d[s], names, prior, pb, s) for s in SPLITS}
    Ap, An = thothgnn3.kg_operators(names, prior)
    return dict(name="breast", kind="ordinal", n_cut=1, n_class=2,
                Z=_std(X), Y={s: d[s]["y"] for s in SPLITS},
                Ap=Ap, An=An, prior=prior, names=names, metric="AUC")


def load_retina():
    # By PATH, for the same reason load_derma is: both trees ship a
    # `kg_adjacency.py`, so which one a bare import returns depends on who
    # inserted their sys.path entry last. transfer_matrix standalone happened to
    # get retina's; prior_shuffle imports this module and THEN prepends derma's
    # dir, so the bare import silently returned derma's loader and died on
    # f["classes"]. Neither module may be reached by name.
    RKA = _load_module("retina_kg_adjacency",
                       REPO / "retinaMnist/src/retina_kg/kg_adjacency.py")
    rb, rk = RKA.build_adjacency, RKA.kg_operators
    cfg = yaml.safe_load((REPO / "retinaMnist/conf/ontology_retina.yaml").read_text())
    X, Y, IDS, names, prior, level, meta = RF.build(
        REPO / "retinaMnist/data/retina", REPO / "retinaMnist/data/retina/images_224",
        REPO / "retinaMnist/results", ("retina_p3",),
        REPO / "retinaMnist/data/retina/debates_r1", SPLITS)
    A, _ = rb(REPO / "retinaMnist/data/retina", cfg); Ap, An = rk(A, prior)
    return dict(name="retina", kind="ordinal", n_cut=4, n_class=5,
                Z=_std(X), Y={s: Y[s].astype(float) for s in SPLITS},
                Ap=Ap, An=An, prior=prior, names=names, metric="refAUC")


def _load_module(name, path):
    """Import by PATH, not by name.

    Both trees ship a `kg_adjacency.py`. Retina's is already on sys.path by the
    time load_derma runs, so a bare `import kg_adjacency` silently returns the
    RETINA loader, which then reads f["level"] off a derma finding and raises
    KeyError. Any module whose basename exists in more than one tree must be
    loaded this way.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_derma(debate=True):
    sys.path.insert(0, str(REPO / "dermaMnist/src/derma_kg"))
    DKA = _load_module("derma_kg_adjacency",
                       REPO / "dermaMnist/src/derma_kg/kg_adjacency.py")
    cfg = yaml.safe_load((REPO / "dermaMnist/conf/ontology_derma.yaml").read_text())
    droot = REPO / "dermaMnist/data/derma/debates_d1"
    X, Y, IDS, names, prior, classes, meta = DF.build(
        REPO / "dermaMnist/data/derma", REPO / "dermaMnist/data/derma/images_224",
        REPO / "dermaMnist/results", "derma_p3",
        str(droot) if (debate and droot.exists()) else None, SPLITS)
    A, _ = DKA.build_adjacency(REPO / "dermaMnist/data/derma", cfg)
    Ap, An = DKA.kg_operators(A, classes)
    Z = _std(X)
    rng = np.random.default_rng(0)
    y = Y["train"]
    if len(y) > DERMA_CAP:                       # stratified cap, all arms alike
        keep = np.sort(np.concatenate(
            [rng.choice(np.flatnonzero(y == c),
                        size=max(2, int(round((y == c).sum() * DERMA_CAP / len(y)))),
                        replace=False) for c in range(7) if (y == c).any()]))
        Z["train"], Y["train"] = Z["train"][keep], y[keep]
    return dict(name="derma", kind="nominal", n_cut=None, n_class=7,
                Z=Z, Y=Y, Ap=Ap, An=An, prior=prior, names=names,
                metric="macroR")


def train_trunk(D, l2, k, seed, epochs):
    if D["kind"] == "ordinal":
        Q = ordinal.fit(D["Z"]["train"], D["Y"]["train"], D["Ap"], D["An"],
                        D["n_cut"], l2=l2, kdim=k, epochs=epochs, seed=seed,
                        prior_centre=D["prior"])
    else:
        Q = nominal.fit(D["Z"]["train"], D["Y"]["train"], D["Ap"], D["An"],
                        D["n_class"], l2=l2, kdim=k, epochs=epochs, seed=seed,
                        prior_centre=D["prior"], class_weight="balanced")
    return {q: Q[q] for q in TRUNK}


def fit_readout(D, trunk, l2, k, seed, epochs):
    """Same optimiser, trunk held fixed; only the readout receives updates."""
    Z, y = D["Z"]["train"], D["Y"]["train"]
    F, d = Z.shape[1], Z.shape[2]
    rng = np.random.default_rng(seed)
    if D["kind"] == "ordinal":
        Q = ordinal.init(F, d, k, D["n_cut"], rng)
        centre = np.zeros((F, k)); centre[:, 0] = D["prior"]
        keys = ("u", "theta")
        gfun = lambda P: ordinal.grads(P, Z, D["Ap"], D["An"], y, l2, centre, D["n_cut"])
    else:
        Q = nominal.init(F, d, k, D["n_class"], rng)
        centre = np.zeros((F, k, D["n_class"])); centre[:, 0, :] = D["prior"]
        keys = ("u", "c0")
        gfun = lambda P: nominal.grads(P, Z, D["Ap"], D["An"], y, l2, centre)
    for q in TRUNK:
        Q[q] = np.array(trunk[q], float, copy=True)
    m = {q: np.zeros_like(Q[q]) for q in keys}
    for _ in range(epochs):
        G, _ = gfun(Q)
        for q in keys:
            m[q] = 0.9 * m[q] + 0.1 * G[q]
            Q[q] = Q[q] - 0.15 * m[q]
    return Q


def score(D, Q):
    y = np.asarray(D["Y"]["test"], int)
    if D["kind"] == "ordinal":
        s, p, g = ordinal.predict(Q, D["Z"]["test"], D["Ap"], D["An"])
        if D["n_class"] == 2:
            return MO.fast_auc(s, y == 1)
        return MO.fast_auc(s, MO.referable(y) == 1)
    s, p, g = nominal.predict(Q, D["Z"]["test"], D["Ap"], D["An"])
    return MN.macro_recall(y, g, D["n_class"])


def zero_param(D):
    y = np.asarray(D["Y"]["test"], int)
    Xp = D["Z"]["test"][:, :, 0]
    if D["kind"] == "ordinal":
        s = (Xp * np.asarray(D["prior"])[None, :]).sum(1)
        t = (y == 1) if D["n_class"] == 2 else (MO.referable(y) == 1)
        return MO.fast_auc(s, t)
    s = Xp @ np.asarray(D["prior"])          # n x C
    return MN.macro_recall(y, s.argmax(1), D["n_class"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--kdim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "shared/results/transfer_matrix.json"))
    a = ap.parse_args()

    D = {"breast": load_breast(), "retina": load_retina(), "derma": load_derma()}
    for k_, v in D.items():
        print(f"  {k_:7s} F={len(v['names']):3d} d={v['Z']['train'].shape[2]} "
              f"C={v['n_class']} n={[len(v['Y'][s]) for s in SPLITS]} metric={v['metric']}")

    trunks = {n: train_trunk(v, a.l2, a.kdim, a.seed, a.epochs) for n, v in D.items()}
    res = {"config": {"l2": a.l2, "kdim": a.kdim, "epochs": a.epochs,
                      "derma_cap": DERMA_CAP, "draws": a.draws}}
    print(f"\n{'target':8s} {'metric':8s} {'ceiling':>9s} "
          f"{'<-breast':>9s} {'<-retina':>9s} {'<-derma':>9s} {'random':>16s} {'0-param':>9s}")
    for tn, T in D.items():
        row = {}
        cq = fit_readout(T, trunks[tn], a.l2, a.kdim, a.seed, a.epochs)
        row["ceiling"] = score(T, cq)
        for sn in D:
            if sn == tn:
                continue
            q = fit_readout(T, trunks[sn], a.l2, a.kdim, a.seed, a.epochs)
            row[f"from_{sn}"] = score(T, q)
        rng = np.random.default_rng(a.seed)
        dv = []
        for _ in range(a.draws):
            rt = {q: (np.zeros_like(trunks[tn][q]) if q == "b"
                      else rng.normal(0, 0.3, np.shape(trunks[tn][q]))) for q in TRUNK}
            dv.append(score(T, fit_readout(T, rt, a.l2, a.kdim, a.seed, a.epochs)))
        row["random"] = [float(np.mean(dv)), float(np.std(dv))]
        row["zero_param"] = zero_param(T)
        res[tn] = row
        g = lambda k_: row.get(k_, float("nan"))
        print(f"{tn:8s} {T['metric']:8s} {row['ceiling']:9.4f} "
              f"{g('from_breast'):9.4f} {g('from_retina'):9.4f} {g('from_derma'):9.4f} "
              f"{row['random'][0]:8.4f}+-{row['random'][1]:5.4f} {row['zero_param']:9.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
