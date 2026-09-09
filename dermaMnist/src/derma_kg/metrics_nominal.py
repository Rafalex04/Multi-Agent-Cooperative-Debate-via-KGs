"""Metrics for a 7-class NOMINAL target with extreme imbalance.

DermaMNIST train is 67% `nv` and 1.1% `df`. Accuracy is therefore close to
useless -- always predicting `nv` scores 0.669 on test -- so macro-recall and
macro one-vs-rest AUC are the headline, exactly as balanced accuracy was on
breast for the same reason.

`fast_auc` is the O(n log n) midrank AUC, identical to claims.auc but usable
inside a bootstrap; the pure-Python double loop there is hours at 2000 resamples.
"""
from __future__ import annotations

import numpy as np


def fast_auc(score, pos):
    score = np.asarray(score, float); pos = np.asarray(pos, bool)
    order = np.argsort(score, kind="mergesort"); srt = score[order]
    ranks = np.empty(len(srt), float); i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(srt), float); r[order] = ranks
    npos = int(pos.sum()); nneg = len(pos) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def macro_ovr_auc(P, y, n_class):
    """Mean per-class one-vs-rest AUC. The metric MedMNIST reports."""
    per = per_class_auc(P, y, n_class)
    v = [a for a in per if not np.isnan(a)]
    return (float(np.mean(v)) if v else float("nan")), per


def per_class_auc(P, y, n_class):
    y = np.asarray(y, int)
    return [fast_auc(P[:, c], y == c) if (y == c).any() and (y != c).any()
            else float("nan") for c in range(n_class)]


def macro_recall(y, yhat, n_class):
    """The n-class generalisation of balanced accuracy. Chance is 1/n_class."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    rec = [float((yhat[y == c] == c).mean()) for c in range(n_class) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def per_class_recall(y, yhat, n_class):
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    return [float((yhat[y == c] == c).mean()) if (y == c).any() else float("nan")
            for c in range(n_class)]


def macro_f1(y, yhat, n_class):
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    fs = []
    for c in range(n_class):
        tp = int(((yhat == c) & (y == c)).sum())
        fp = int(((yhat == c) & (y != c)).sum())
        fn = int(((yhat != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs)) if fs else float("nan")


def summarise(y, yhat, P, n_class):
    auc, per = macro_ovr_auc(P, y, n_class)
    return {"accuracy": float((np.asarray(y, int) == np.asarray(yhat, int)).mean()),
            "macro_recall": macro_recall(y, yhat, n_class),
            "macro_f1": macro_f1(y, yhat, n_class),
            "macro_ovr_auc": auc,
            "per_class_auc": per,
            "per_class_recall": per_class_recall(y, yhat, n_class)}
