"""Metrics for a 5-level ordinal target, alongside the binary ones v3 reports.

`experiments/README.md` fixes balanced accuracy as the headline because breast is
27% malignant. That definition -- (sensitivity + specificity)/2 -- is 2-class. On
a 5-level scale the analogue is macro-recall, and the ordinal-specific quantities
are quadratic-weighted kappa and MAE, neither of which existed in this repo.

`auc` and `bacc` are re-exported from claims.py rather than reimplemented, so the
binary numbers stay bit-comparable with every earlier experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_experiment_paths                              # noqa: E402
add_experiment_paths("14_kgtensor")
from claims import auc, bacc                                        # noqa: E402,F401


def fast_auc(score, pos):
    """Midrank AUC in O(n log n). Identical to claims.auc -- verified to 0.0 on
    tie-heavy random cases -- but usable inside a bootstrap.

    claims.auc is a pure-Python double loop over pos x neg. At 2000 resamples on
    n=400 that is ~400M comparisons per arm, i.e. hours. Any resampling loop in
    this tree should call this instead.
    """
    score = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    order = np.argsort(score, kind="mergesort")
    srt = score[order]
    ranks = np.empty(len(srt), float)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(srt), float)
    r[order] = ranks
    npos = int(pos.sum()); nneg = len(pos) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def referable(y, cut=2):
    """The clinical reduction: grade >= 2 is referable diabetic retinopathy."""
    return (np.asarray(y) >= cut).astype(float)


def threshold_auc(scores, y, n_cut):
    """AUC of the score for cut point c against 1[y > c], for each c.

    `scores` is (n,) -- option A has one shared score -- or (n, n_cut), where
    option C's head c is scored against its own threshold.
    """
    s = np.asarray(scores)
    y = np.asarray(y)
    out = []
    for c in range(n_cut):
        sc = s if s.ndim == 1 else s[:, c]
        t = (y > c)
        out.append(auc(list(sc[t]), list(sc[~t])) if t.any() and (~t).any() else float("nan"))
    return out


def qwk(y_true, y_pred, n_class):
    """Quadratic-weighted kappa. 0 is chance, 1 is perfect, negative is worse."""
    y_true = np.asarray(y_true, int)
    y_pred = np.clip(np.asarray(y_pred, int), 0, n_class - 1)
    O = np.zeros((n_class, n_class))
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1
    w = (np.arange(n_class)[:, None] - np.arange(n_class)[None, :]) ** 2
    w = w / ((n_class - 1) ** 2)
    ha = np.bincount(y_true, minlength=n_class).astype(float)
    hb = np.bincount(y_pred, minlength=n_class).astype(float)
    E = np.outer(ha, hb)
    E = E * (O.sum() / E.sum()) if E.sum() else E
    den = (w * E).sum()
    return float(1.0 - (w * O).sum() / den) if den else float("nan")


def mae(y_true, y_pred):
    return float(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)).mean())


def macro_recall(y_true, y_pred, n_class):
    """The n-class generalisation of balanced accuracy. Chance is 1/n_class."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    rec = [float((y_pred[y_true == c] == c).mean()) for c in range(n_class)
           if (y_true == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def adjacent_accuracy(y_true, y_pred):
    """Fraction predicted within one grade. Standard in DR grading papers."""
    return float((np.abs(np.asarray(y_true, int) - np.asarray(y_pred, int)) <= 1).mean())


def summarise(y_true, grade_hat, scores, n_class, ref_score=None):
    """Everything the ordinal arms report, in one dict."""
    n_cut = n_class - 1
    yr = referable(y_true)
    rs = ref_score if ref_score is not None else (
        scores if np.asarray(scores).ndim == 1 else np.asarray(scores)[:, 1])
    rs = np.asarray(rs)
    return {
        "referable_auc": auc(list(rs[yr == 1]), list(rs[yr == 0])),
        "threshold_auc": threshold_auc(scores, y_true, n_cut),
        "qwk": qwk(y_true, grade_hat, n_class),
        "mae": mae(y_true, grade_hat),
        "macro_recall": macro_recall(y_true, grade_hat, n_class),
        "adjacent_acc": adjacent_accuracy(y_true, grade_hat),
        "accuracy": float((np.asarray(y_true, int) == np.asarray(grade_hat, int)).mean()),
    }
