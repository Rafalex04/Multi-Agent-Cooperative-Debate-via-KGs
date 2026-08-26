"""Convergence curves for every learnable model in the project.

One protocol for all of them so the panels are comparable: BreastMNIST train
(546) split 80/20 stratified on label with a fixed seed, fit on the 80, and
train + held-out AUC recorded at regular epoch intervals. Nothing here is tuned;
each model runs at the hyperparameters its own experiment selected by CV.

Zero-parameter models (the KG-signed sum, the KG differential) are excluded --
they have nothing to learn, so a convergence curve for them is a flat line by
construction. ResNet-18 is read from its original training log rather than
retrained, and is on its own scale (100 epochs of SGD over images, not full-batch
gradient descent over features), so it is plotted but not compared epoch-for-epoch.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in ("14_kgtensor", "15_kggnn", "16_external", "17_hetgnn", "18_lesion"):
    sys.path.insert(0, str(_HERE.parents[0] / p))
from analyse_e1 import auc                                              # noqa: E402
from claims import kg_findings                                          # noqa: E402
from hetgraph import SPLITS, build, kg_operators                        # noqa: E402
from kg_graph import build_adjacency                                    # noqa: E402
from thothgnn3 import forward as v3_forward, grads as v3_grads, init as v3_init  # noqa: E402

REC = 20            # record every REC epochs


def split(y, frac=0.8, seed=7):
    """Stratified 80/20 so both halves carry the same malignant rate."""
    rng = np.random.default_rng(seed)
    tr = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        tr[idx[:int(round(len(idx) * frac))]] = True
    return tr


def curve_logistic(X, y, tr, l2, loss, epochs=2500, lr=0.5, seed=0):
    """fit_ce / fit_rank, instrumented. Same update rule, same defaults."""
    lr = min(lr, 0.5 / max(1e-9, l2))
    w = np.zeros(X.shape[1]); b = 0.0
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y[tr] == 1), np.flatnonzero(y[tr] == 0)
    Xtr, ytr = X[tr], y[tr]
    out = []
    for ep in range(1, epochs + 1):
        if loss == "ce":
            p = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
            e = p - ytr
            w -= lr * ((Xtr.T @ e) / len(ytr) + l2 * w); b -= lr * float(e.mean())
        else:
            i, j = rng.choice(pos, 512), rng.choice(neg, 512)
            s = Xtr @ w + b
            g = -1.0 / (1.0 + np.exp(np.clip(s[i] - s[j], -30, 30)))
            e = np.zeros(len(ytr)); np.add.at(e, i, g); np.add.at(e, j, -g); e /= 512
            w -= lr * ((Xtr.T @ e) + l2 * w); b -= lr * float(e.sum())
        if ep % REC == 0:
            sc = X @ w + b
            out.append({"epoch": ep, "train": auc(y[tr], sc[tr]), "test": auc(y[~tr], sc[~tr])})
    return out


def curve_calib(X, y, tr, F, nb, L, lam, l2, epochs=2000, lr=0.3):
    """S5: per-finding scale/bias under a KG-Laplacian penalty, free readout."""
    Xtr, ytr = X[tr], y[tr]
    n, d = Xtr.shape
    a = np.ones(F); c = np.zeros(F); w = np.zeros(d); b = 0.0
    idx = np.tile(np.arange(F), nb)
    step = lr / (1.0 + lam * np.abs(L).max() + l2)
    out = []
    for ep in range(1, epochs + 1):
        Z = Xtr * a[idx] + c[idx]
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        e = p - ytr
        gw = (Z.T @ e) / n + l2 * w
        gz = np.outer(e, w) / n
        ga = np.zeros(F); gc = np.zeros(F)
        np.add.at(ga, idx, (gz * Xtr).sum(0)); np.add.at(gc, idx, gz.sum(0))
        w -= step * gw; b -= step * float(e.mean())
        a -= step * (ga + lam * (L @ a)); c -= step * (gc + lam * (L @ c))
        if ep % REC == 0:
            sc = (X * a[idx] + c[idx]) @ w + b
            out.append({"epoch": ep, "train": auc(y[tr], sc[tr]), "test": auc(y[~tr], sc[~tr])})
    return out


def curve_v3(Z, y, tr, Ap, An, prior, l2=0.03, kdim=2, epochs=800, lr=0.15):
    F, d = Z.shape[1], Z.shape[2]
    P = v3_init(F, d, kdim, np.random.default_rng(0), prior)
    centre = np.zeros((F, kdim)); centre[:, 0] = prior
    m = {k: np.zeros_like(v) if not np.isscalar(v) else 0.0 for k, v in P.items()}
    out = []
    for ep in range(1, epochs + 1):
        G, _ = v3_grads(P, Z[tr], Ap, An, y[tr], l2, centre)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            P[k] = P[k] - lr * m[k]
        if ep % REC == 0:
            out.append({"epoch": ep,
                        "train": auc(y[tr], v3_forward(P, Z[tr], Ap, An)[2]),
                        "test": auc(y[~tr], v3_forward(P, Z[~tr], Ap, An)[2])})
    return out


def curve_v2(X, anchor, y, tr, Ap, An, l2=1e-3, epochs=400, lr=0.05, h=4):
    """ThothKG v2: sign-separated messages, attention readout, anchored score."""
    from thoth_kg import ThothKG, bce
    mdl = ThothKG(X.shape[2], h, Ap, An, seed=0, readout="attention")
    out = []
    for ep in range(1, epochs + 1):
        s, cache = mdl.forward(X[tr], anchor[tr])
        _, dS = bce(s, y[tr])
        mdl.step(mdl.backward(cache, dS, l2), lr)
        if ep % 10 == 0:
            str_, _ = mdl.forward(X[tr], anchor[tr])
            ste, _ = mdl.forward(X[~tr], anchor[~tr])
            out.append({"epoch": ep, "train": auc(y[tr], str_), "test": auc(y[~tr], ste)})
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    curves = {}

    # ---- feature matrices, all on BreastMNIST train ------------------------
    from freeze_models import breastmnist_matrix
    Xp, Yb = breastmnist_matrix([(n, 0) for n in names])          # P2+P3, 32 cols
    Xtr_p, y = Xp["train"], Yb["train"]
    mu, sd = Xtr_p.mean(0), Xtr_p.std(0); sd = np.where(sd < 1e-9, 1, sd)
    Zp = (Xtr_p - mu) / sd
    tr = split(y)
    print(f"protocol: {tr.sum()} fit / {(~tr).sum()} held-out  "
          f"(malignant {y[tr].mean():.3f} / {y[~tr].mean():.3f})\n")

    print("probe logistic, cross-entropy ...")
    curves["Probe logistic · cross-entropy"] = curve_logistic(Zp, y, tr, 0.3, "ce")
    print("probe logistic, ranking loss ...")
    curves["Probe logistic · ranking loss"] = curve_logistic(Zp, y, tr, 0.3, "rank")

    print("S5 KG-Laplacian calibration ...")
    A, _ = build_adjacency(_HERE.parents[1] / "breastMnist/data/breast/knowledge_graph.json", names)
    same = (prior[:, None] * prior[None, :]) > 0
    As = A * same - A * (~same)
    Lap = np.diag(np.abs(As).sum(1)) - As
    curves["S5 · KG-Laplacian calibration"] = curve_calib(Zp, y, tr, len(names), 2, Lap, 0.01, 0.3)

    # ---- findings + lesions -------------------------------------------------
    try:
        from analyse_lesion import load as load_les
        from lesions import lesion_table
        L = lesion_table(); ents = [e for e, *_ in L]
        tab = load_les("lesion", ents)
        keys = sorted(k for k in tab if k[0] == "train")
        idx = [k[1] for k in keys]
        XL = np.array([tab[k][0] for k in keys])
        if len(idx) == len(y):
            Xc = np.hstack([Xtr_p[idx], XL])
            m2, s2 = Xc.mean(0), Xc.std(0); s2 = np.where(s2 < 1e-9, 1, s2)
            print("findings + lesions logistic ...")
            curves["Findings + lesions · logistic"] = curve_logistic(
                (Xc - m2) / s2, y[idx], tr, 0.01, "ce")
        else:
            print(f"  lesion alignment {len(idx)} vs {len(y)} -- skipped")
    except Exception as exc:
        print("  lesions skipped:", exc)

    # ---- the two GNNs -------------------------------------------------------
    root = str(_HERE.parents[1] / "breastMnist/data/breast/debates_v5q")
    d, _, _ = build(root)
    from thothgnn3 import node_tensor, probe_block
    probes = probe_block(names)
    Xg = node_tensor(d["train"], names, prior, probes, "train")
    yg = d["train"]["y"]
    mug = Xg.reshape(-1, 5).mean(0); sdg = Xg.reshape(-1, 5).std(0)
    Zg = (Xg - mug) / np.where(sdg < 1e-9, 1, sdg)
    trg = split(yg)
    Ap, An = kg_operators(names, prior)
    print("ThothGNN v3 (merged) ...")
    curves["ThothGNN v3 · merged KG + debate"] = curve_v3(Zg, yg, trg, Ap, An, prior)

    print("ThothGNN v2 (ThothKG, anchored) ...")
    try:
        mal = np.array([np.mean([c["stance"] for c in r["claims"]]) if r["claims"] else 0.5
                        for r in d["train"]["ids"] and __import__("json") and []] or
                       [0.0] * len(yg))
        # anchor = the debate's malignant share, the quantity v2 was built around
        anc = Zg[:, :, 3].mean(1)
        curves["ThothGNN v2 · anchored, attention readout"] = curve_v2(
            Zg, anc, yg, trg, Ap, An)
    except Exception as exc:
        print("  v2 skipped:", exc)

    # ---- ResNet-18 from its original log ------------------------------------
    log = _HERE.parents[0] / "16_external/logs/resnet.log"
    if log.exists():
        rn = []
        for ln in log.read_text(errors="replace").splitlines():
            if ln.strip().startswith("epoch ") and "val_auc=" in ln:
                parts = ln.split()
                ep = int(parts[1].split("/")[0])
                va = float([p for p in parts if p.startswith("val_auc=")][0].split("=")[1])
                rn.append({"epoch": ep, "train": None, "test": va})
        if rn:
            curves["ResNet-18 · supervised baseline"] = rn
            print(f"ResNet-18 read from log ({len(rn)} points)")

    (_HERE / "results/train_curves.json").write_text(json.dumps(curves, indent=1, default=float))
    print("\n" + "=" * 66)
    for k, v in curves.items():
        last = v[-1]
        t = f"{last['train']:.4f}" if last["train"] is not None else "  n/a "
        best = max(x["test"] for x in v)
        print(f"{k:44s} train {t}  held-out {last['test']:.4f}  (best {best:.4f})")
    print(f"\nwrote results/train_curves.json  ({len(curves)} models)")


if __name__ == "__main__":
    main()
