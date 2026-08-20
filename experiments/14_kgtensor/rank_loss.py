"""Section 1: does optimising the reported metric directly help?

Every model in this project is trained with cross-entropy, which optimises
calibration. Every number reported is AUC, which is a ranking statistic. The
plan's argument for closing that gap is also a supervision argument: 546 graphs
give 546 cross-entropy terms but roughly 147 x 399 = 58,000 malignant-benign
pairs, on a dataset whose binding constraint is sample count.

    L = mean over sampled (i in MAL, j in BEN) of softplus(-(s_i - s_j))

Run against the identical model, features, anchor and L2 sweep, changing only
the loss, so the comparison isolates it. Class weighting is dropped for the
ranking loss because every term already contains one of each class.

Usage:
  python rank_loss.py --debates ...
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import SPLITS, auc, bacc, by_split, kg_findings, load     # noqa: E402
from kg_tensor import (COLS, anchor_of, featurise, fit_evidence, kg_centre,  # noqa
                       pool_by_group, predict, train_head)


def train_rank(X, a, y, w0, l2, epochs=1500, lr=0.5, npairs=512, seed=0):
    """Same score function as train_head; pairwise ranking loss instead of BCE."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    w = w0.copy(); b, beta = 0.0, 1.0
    n = len(y)
    for _ in range(epochs):
        i = rng.choice(pos, npairs); j = rng.choice(neg, npairs)
        s = beta * a + (X @ w + b)
        d = s[i] - s[j]
        g = -1.0 / (1.0 + np.exp(np.clip(d, -30, 30)))       # dL/d(d)
        e = np.zeros(n)
        np.add.at(e, i, g); np.add.at(e, j, -g)
        e /= npairs
        w -= lr * ((X.T @ e) + l2 * (w - w0))
        b -= lr * float(e.sum())
        beta -= lr * float(e @ a)
    return w, b, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    findings = kg_findings()
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    by = by_split(load(args.debates, findings))
    ev_w = fit_evidence(by["train"])
    Y = {s: np.array([r["y"] for r in by[s]], dtype=float) for s in SPLITS}
    print("graphs: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    res = {}
    for kind in ("evidence", "mal_share", "none"):
        a_te = anchor_of(by["test"], kind, ev_w, False)
        print(f"\n=== anchor: {kind} "
              f"(alone TEST {auc(list(a_te[Y['test']==1]), list(a_te[Y['test']==0])):.4f}) ===")
        for aug in (False, True):
            Xtr, ytr, gtr = featurise(by["train"], F, COLS, aug, False)
            atr = anchor_of(by["train"], kind, ev_w, aug)
            w0 = kg_centre(COLS, False, True, F, prior)
            for lname, fn in (("cross-entropy", train_head), ("ranking", train_rank)):
                # L2 by grouped CV on train, same protocol for both losses
                rng = np.random.default_rng(0)
                folds = np.array_split(rng.permutation(np.unique(gtr)), 5)
                best = None
                for l2 in L2S:
                    sc, gs, ys = [], [], []
                    for fo in folds:
                        m = np.isin(gtr, fo)
                        w, b, be = fn(Xtr[~m], atr[~m], ytr[~m], w0, l2)
                        sc.append(predict(Xtr[m], atr[m], w, b, be))
                        gs.append(gtr[m]); ys.append(ytr[m])
                    s_ = np.concatenate(sc); g_ = np.concatenate(gs); y_ = np.concatenate(ys)
                    if aug:
                        nn = int(gtr.max()) + 1
                        ps = pool_by_group(s_, g_, nn)
                        py = np.zeros(nn); py[g_] = y_
                        k = np.unique(g_)
                        s_, y_ = ps[k], py[k]
                    c = auc(list(s_[y_ == 1]), list(s_[y_ == 0]))
                    if best is None or c > best[0]:
                        best = (c, l2)
                cvb, l2 = best
                w, b, be = fn(Xtr, atr, ytr, w0, l2)
                out = {}
                for s in SPLITS:
                    X, y, g = featurise(by[s], F, COLS, aug, False)
                    a = anchor_of(by[s], kind, ev_w, aug)
                    sc = predict(X, a, w, b, be)
                    out[s] = pool_by_group(sc, g, len(by[s])) if aug else sc
                at = auc(list(out["test"][Y["test"] == 1]), list(out["test"][Y["test"] == 0]))
                sf = np.concatenate([out["train"], out["val"]])
                yf = np.concatenate([Y["train"], Y["val"]])
                t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
                bb = bacc(out["test"].tolist(), Y["test"].tolist(), t)
                tag = f"{lname}{' +aug' if aug else ''}"
                print(f"  {tag:22s} TEST {at:.4f}  bAcc {bb:.4f}   (cv {cvb:.4f} l2 {l2})")
                res[f"{kind}|{tag}"] = {"test": at, "bacc": bb, "cv": cvb, "l2": l2}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
