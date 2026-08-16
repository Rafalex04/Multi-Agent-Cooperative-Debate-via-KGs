"""Fit a logistic head on the KG probe vector, train split only.

The unsupervised score (KG stance sign as the weight, equal magnitudes) reaches
AUC 0.6685 on test against a 0.6011 no-KG baseline. That fixes the sign of each
feature's contribution but not its magnitude, and the KG has no way to tell us
that `irregular_shape` discriminates while `spiculated` is saturated.

This fits those magnitudes on the training split and evaluates on test, never
touching test labels. The KG still chooses which features exist; only their
weights are learned. Reported alongside the unsupervised number, not instead of
it -- they are different settings and the zero-shot baseline is only comparable
to the unsupervised one.

Probe vectors are read out of the v2 debate graphs, which store the full
measurement array per sample, so this needs no extra inference.

Usage:
  python fit_probe_head.py --dataset ../../breastMnist/data/breast/dataset_v2
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path


def load(root: Path, split: str) -> tuple[list[list[float]], list[int], list[str]]:
    d = root / split / "graphs"
    X, y, feats = [], [], None
    for f in sorted(d.glob("*.json")):
        g = json.loads(f.read_text())
        ev = g.get("evidence")
        if not ev:
            continue
        ev = sorted(ev, key=lambda e: e["feature"])
        names = [e["feature"] for e in ev]
        if feats is None:
            feats = names
        elif names != feats:                      # probe set changed mid-run
            continue
        vals = [e["p_yes"] for e in ev]
        if any(v is None for v in vals):
            continue
        X.append(vals)
        y.append(1 if g["gold_label"] == "MALIGNANT" else 0)
    return X, y, (feats or [])


def standardise(X, mu=None, sd=None):
    n = len(X[0])
    if mu is None:
        mu = [sum(r[j] for r in X) / len(X) for j in range(n)]
        sd = [max(1e-6, (sum((r[j] - mu[j]) ** 2 for r in X) / len(X)) ** 0.5)
              for j in range(n)]
    return [[(r[j] - mu[j]) / sd[j] for j in range(n)] for r in X], mu, sd


def fit(X, y, epochs=4000, lr=0.05, l2=0.05):
    """Plain gradient-descent logistic regression with L2.

    Hand-rolled so the script runs on the worker nodes, which have numpy but no
    scikit-learn. Class weighting compensates for the 27% positive rate.
    """
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    npos = sum(y) or 1
    nneg = n - npos or 1
    wpos, wneg = n / (2 * npos), n / (2 * nneg)
    for _ in range(epochs):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[j] * xi[j] for j in range(d))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = (p - yi) * (wpos if yi else wneg)
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    root = Path(args.dataset)

    Xtr, ytr, feats = load(root, "train")
    Xte, yte, fte   = load(root, "test")
    if not Xtr or not Xte:
        print(f"not enough data yet: train={len(Xtr)} test={len(Xte)}")
        return
    if feats != fte:
        print("probe sets differ between splits; aborting")
        return

    print(f"train n={len(Xtr)} ({sum(ytr)} malignant)   "
          f"test n={len(Xte)} ({sum(yte)} malignant)   features={len(feats)}")

    Xtr_s, mu, sd = standardise(Xtr)
    Xte_s, _, _   = standardise(Xte, mu, sd)

    w, b = fit(Xtr_s, ytr)

    def score(X):
        return [b + sum(w[j] * r[j] for j in range(len(w))) for r in X]

    for name, X, y in (("train", Xtr_s, ytr), ("test", Xte_s, yte)):
        s = score(X)
        pos = [v for v, t in zip(s, y) if t == 1]
        neg = [v for v, t in zip(s, y) if t == 0]
        a = auc(pos, neg)
        best = max(
            (0.5 * (sum(1 for v in pos if v >= th) / len(pos) +
                    sum(1 for v in neg if v < th) / len(neg)), th)
            for th in sorted(set(s)))
        print(f"  {name:5s} AUC={a:.4f}  best_balanced_acc={best[0]:.4f}")

    print("\nLearned weights (positive = pushes toward malignant):")
    for f, wt in sorted(zip(feats, w), key=lambda kv: -abs(kv[1])):
        print(f"  {f[:46]:46s} {wt:+.3f}")

    print("\nReference: zero-shot no-KG baseline AUC = 0.6011 (unsupervised)")
    print("           unsupervised KG probe score  = 0.6685 (unsupervised)")
    print("           the number above is supervised and not directly comparable")


if __name__ == "__main__":
    main()
