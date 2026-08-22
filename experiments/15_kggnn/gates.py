"""G0, S7 and the S3 diagnostic -- the three cheap checks that set priors.

G0  Do KG-masked pairwise probe interactions add anything? The earlier
    interaction gate ran on DEBATE counts and found nothing; it has never run on
    probe features. Products are restricted to finding pairs adjacent in the
    lesion graph (72 of 120), which keeps the parameter count sane and makes the
    test specifically about ontology-linked conjunctions rather than all pairs.

S7  Does a pairwise ranking loss lift the flat probe model? If it does, every
    candidate's delta must be measured against the lifted number, not 0.8137.

S3d Does the debate carry signal exactly where the probes are UNCERTAIN? An
    average can hide conditional value. Correlate the debate score against the
    probe model's residual, binned by probe confidence, on training folds only.

Positive control on G0 is inherited from experiments/14_kgtensor/gate.py, which
established that this CV protocol detects a synthetic XOR interaction at +0.19.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc, kg_findings                     # noqa: E402
from features import build                                            # noqa: E402
from kg_graph import build_adjacency                                  # noqa: E402
from kg_tensor import evidence_score, fit_evidence                    # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"
L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def fit_ce(X, y, l2, epochs=2500, lr=0.5):
    lr = min(lr, 0.5 / max(1e-9, l2))
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        e = p - y
        w -= lr * ((X.T @ e) / n + l2 * w); b -= lr * float(e.mean())
    return w, b


def fit_rank(X, y, l2, epochs=2500, lr=0.5, npairs=512, seed=0):
    """softplus(-(s_mal - s_ben)) on sampled pairs; no class weighting needed."""
    lr = min(lr, 0.5 / max(1e-9, l2))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        i, j = rng.choice(pos, npairs), rng.choice(neg, npairs)
        s = X @ w + b
        g = -1.0 / (1.0 + np.exp(np.clip(s[i] - s[j], -30, 30)))
        e = np.zeros(len(y)); np.add.at(e, i, g); np.add.at(e, j, -g); e /= npairs
        w -= lr * ((X.T @ e) + l2 * w); b -= lr * float(e.sum())
    return w, b


def cv_score(X, y, l2, fitter, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(y)), folds)
    s = np.zeros(len(y))
    for p in parts:
        m = np.zeros(len(y), dtype=bool); m[p] = True
        mu, sd = X[~m].mean(0), X[~m].std(0); sd = np.where(sd < 1e-9, 1.0, sd)
        w, b = fitter((X[~m] - mu) / sd, y[~m], l2)
        s[m] = ((X[m] - mu) / sd) @ w + b
    return auc(list(s[y == 1]), list(s[y == 0])), s


def full_eval(X, Y, l2, fitter):
    mu, sd = X["train"].mean(0), X["train"].std(0); sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    w, b = fitter(Z["train"], Y["train"], l2)
    sc = {s: Z[s] @ w + b for s in SPLITS}
    a = {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS}
    sf = np.concatenate([sc["train"], sc["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
    return a, bacc(sc["test"].tolist(), Y["test"].tolist(), t), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["probe", "probeneg"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings()
    blocks, rows, Y = [], None, None
    for tag in args.tags:
        rows, Xt, Y, _ = build(runs, tag=tag)
        blocks.append({s: Xt[s][:, :, 0] for s in SPLITS})
    X = {s: np.hstack([b[s] for b in blocks]) for s in SPLITS}
    P = {s: np.mean([b[s] for b in blocks], axis=0) for s in SPLITS}   # per-finding mean
    print(f"anchor features: {X['train'].shape[1]}  ({len(args.tags)} phrasings x 16)")
    res = {}

    # ---------------- S7: ranking loss on the flat model ----------------
    print("\n=== S7: ranking loss vs cross-entropy on the flat probe model ===")
    for name, fitter in (("cross-entropy", fit_ce), ("ranking", fit_rank)):
        best = max(((cv_score(X["train"], Y["train"], l2, fitter)[0], l2) for l2 in L2S))
        cv, l2 = best
        a, bb, sc = full_eval(X, Y, l2, fitter)
        print(f"  {name:14s} TEST {a['test']:.4f}  bAcc {bb:.4f}   (cv {cv:.4f} "
              f"val {a['val']:.4f} l2 {l2})")
        res[f"S7|{name}"] = {"test": a["test"], "bacc": bb, "cv": cv, "l2": l2}
        if name == "cross-entropy":
            anchor_sc, anchor_cv = sc, cv

    # ---------------- G0: KG-masked pairwise interactions ----------------
    A, info = build_adjacency(_KG, [f for f, _ in findings])
    pairs = [(i, j) for i, j in itertools.combinations(range(len(findings)), 2) if A[i, j] > 0]
    print(f"\n=== G0: KG-masked probe interactions ({len(pairs)} adjacent pairs) ===")
    prod = {s: np.column_stack([P[s][:, i] * P[s][:, j] for i, j in pairs]) for s in SPLITS}
    allp = [(i, j) for i, j in itertools.combinations(range(len(findings)), 2)]
    prod_all = {s: np.column_stack([P[s][:, i] * P[s][:, j] for i, j in allp]) for s in SPLITS}
    variants = {"base (no products)": X,
                "+ KG-masked products": {s: np.hstack([X[s], prod[s]]) for s in SPLITS},
                "+ ALL products": {s: np.hstack([X[s], prod_all[s]]) for s in SPLITS}}
    for name, Xv in variants.items():
        best = max(((cv_score(Xv["train"], Y["train"], l2, fit_ce)[0], l2) for l2 in L2S))
        cv, l2 = best
        a, bb, _ = full_eval(Xv, Y, l2, fit_ce)
        print(f"  {name:24s} cv {cv:.4f}   TEST {a['test']:.4f}  bAcc {bb:.4f}  "
              f"(feats {Xv['train'].shape[1]}, l2 {l2})")
        res[f"G0|{name}"] = {"cv": cv, "test": a["test"], "bacc": bb}
    d = res["G0|+ KG-masked products"]["cv"] - res["G0|base (no products)"]["cv"]
    print(f"  -> KG-masked interaction delta on CV: {d:+.4f}  "
          + ("SIGNAL, S1/S2 keep their prior" if d > 0.005
             else "NOTHING; S1/S2 inherit a low prior"))
    res["G0|delta"] = d

    # ---------------- S3 diagnostic ----------------
    print("\n=== S3 diagnostic: does the debate help where probes are uncertain? ===")
    ev_w = fit_evidence(rows["train"])
    deb = {s: np.array([evidence_score(r["claims"], ev_w) for r in rows[s]]) for s in SPLITS}
    _, cv_s = cv_score(X["train"], Y["train"], res["S7|cross-entropy"]["l2"], fit_ce)
    p = 1.0 / (1.0 + np.exp(-cv_s))
    resid = Y["train"] - p
    # probe uncertainty: how close the pooled probes sit to 0.5, averaged
    unc = 1.0 - np.abs(P["train"] - 0.5).mean(1) * 2
    q = np.quantile(unc, [0, .25, .5, .75, 1.0])
    print(f"  {'confidence bin':22s} {'n':>5s} {'corr(debate, residual)':>24s}")
    corrs = []
    for k in range(4):
        m = (unc >= q[k]) & (unc <= q[k + 1])
        if m.sum() < 20:
            continue
        c = float(np.corrcoef(deb["train"][m], resid[m])[0, 1])
        corrs.append(c)
        lab = ["most confident", "", "", "least confident"][k]
        print(f"  Q{k+1} {lab:18s} {int(m.sum()):5d} {c:+24.3f}")
    res["S3d|corrs"] = corrs
    spread = max(corrs) - min(corrs) if corrs else 0.0
    print(f"  -> spread across bins {spread:.3f}  "
          + ("conditional value plausible, build S3" if spread > 0.15
             else "flat across bins; S3 killed before building"))
    res["S3d|spread"] = spread

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
