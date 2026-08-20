"""Section 0: is the label additive in independent per-claim evidence?

The evidence-weighted readout is a six-number lookup that treats every claim
independently, and it scores 0.7556 / 0.6930. A graph network can only beat that
by capturing something NON-additive -- an interaction between claims. So before
changing any architecture, test whether interactions exist at all.

    base          per-finding counts (n_mal, n_ben per finding)
    interaction   the same counts plus every pairwise product

Both get the identical protocol: standardised on the training folds only,
5-fold grouped cross-validation over the 546 training graphs, and the same L2
sweep, with the winner chosen by CV AUC. Giving the interaction model a weaker
regularisation sweep than the base model would rig the answer, so the sweep is
shared and extends far enough that the larger model can shrink itself back to
the smaller one if that is what the data wants.

If interactions do not beat counts under CV, the label is an additive function
of independent claim evidence and no GNN will beat the lookup table. That is a
clean, reportable result about the structure of LLM debate evidence.

Usage:
  python gate.py --debates .../debates_v5q [...]
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import auc, by_split, kg_findings, load                   # noqa: E402


def counts(claims, F, normalise=True):
    """n_mal and n_ben per finding -- the additive sufficient statistic.

    Normalised by claim count by default. A raw count vector cannot express
    mal_share, which is a RATIO, so an unnormalised base model is handicapped
    against the very readout it is supposed to represent -- measured at CV
    0.5811 against mal_share's 0.6476. Handicapping the base would rig the gate
    toward finding interactions, so the base gets the normalisation.
    """
    v = np.zeros(2 * F)
    for c in claims:
        f = c["finding"]
        if f >= 0:
            v[2 * f + (0 if c["stance"] else 1)] += 1.0
    if normalise:
        v = v / max(1.0, len(claims))
    return v


def base_feats(claims, F):
    """Per-finding shares, plus the two global statistics the readouts use.

    mal_share is included explicitly so the additive model starts from the
    strongest known additive readout rather than having to rediscover it from
    32 per-finding shares on 546 samples.
    """
    v = counts(claims, F)
    lab = [c for c in claims if c["finding"] >= 0]
    ms = (sum(c["stance"] for c in lab) / len(lab)) if lab else 0.5
    return np.concatenate([v, [ms, np.log1p(len(claims))]])


def with_products(X, d0):
    """Append every pairwise product of the first d0 (per-finding) columns."""
    idx = list(itertools.combinations(range(d0), 2))
    P = np.empty((len(X), len(idx)))
    for k, (i, j) in enumerate(idx):
        P[:, k] = X[:, i] * X[:, j]
    return np.hstack([X, P])


def fit_logreg(X, y, l2, epochs=3000, lr=0.5):
    """Gradient descent with the step scaled so the L2 shrinkage cannot diverge.

    The shrink factor per step is (1 - lr*l2); at lr 0.5 and l2 10 that is -4 and
    the weights blow up, which is what produced AUC 0.0000 rows rather than a
    heavily regularised model.
    """
    lr = min(lr, 0.5 / max(1e-9, l2))
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        e = p - y
        w -= lr * ((X.T @ e) / n + l2 * w)
        b -= lr * float(e.mean())
    return w, b


def cv_auc(X, y, l2, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    s = np.zeros(len(y))
    for p in parts:
        m = np.zeros(len(y), dtype=bool); m[p] = True
        mu, sd = X[~m].mean(0), X[~m].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        w, b = fit_logreg((X[~m] - mu) / sd, y[~m], l2)
        s[m] = ((X[m] - mu) / sd) @ w + b
    return auc(list(s[y == 1]), list(s[y == 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    findings = kg_findings()
    F = len(findings)
    by = by_split(load(args.debates, findings))
    tr = by["train"]
    y = np.array([r["y"] for r in tr], dtype=float)
    Xc = np.array([base_feats(r["claims"], F) for r in tr])
    Xi = with_products(Xc, 2 * F)
    print(f"train graphs {len(y)}  ({int(y.sum())} malignant)")
    print(f"  counts      {Xc.shape[1]:4d} features")
    print(f"  interaction {Xi.shape[1]:4d} features")

    ms = np.array([np.mean([c["stance"] for c in r["claims"] if c["finding"] >= 0] or [0.5])
                   for r in tr])
    print(f"  reference: unfitted mal_share on these graphs "
          f"AUC {auc(list(ms[y == 1]), list(ms[y == 0])):.4f}")
    L2S = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
    res = {}
    print(f"\n  {'l2':>8s}   {'counts':>8s}   {'interaction':>11s}")
    for l2 in L2S:
        a, b = cv_auc(Xc, y, l2), cv_auc(Xi, y, l2)
        res[l2] = (a, b)
        print(f"  {l2:8g}   {a:8.4f}   {b:11.4f}")

    ba = max(res.values(), key=lambda t: t[0])[0]
    bb = max(res.values(), key=lambda t: t[1])[1]
    la = max(res, key=lambda k: res[k][0]); lb = max(res, key=lambda k: res[k][1])
    print(f"\n  best counts       CV AUC {ba:.4f}  (l2 {la:g})")
    print(f"  best interaction  CV AUC {bb:.4f}  (l2 {lb:g})")
    print(f"  delta             {bb - ba:+.4f}")
    print("\n  -> interactions carry signal; a graph model is justified"
          if bb > ba + 0.005 else
          "\n  -> NO measurable interaction: the label is additive in independent\n"
          "     claim evidence, and no GNN will beat the weighted count.")

    # ---- positive control: can this protocol detect an interaction that IS there? ----
    # A null result is only meaningful if the test has power. Synthesise a label
    # that is pure XOR of two finding shares -- zero main effect, all interaction
    # -- and confirm the interaction model finds it while the base model cannot.
    f1, f2 = 2 * 0, 2 * 1
    a1 = Xc[:, f1] > np.median(Xc[:, f1])
    a2 = Xc[:, f2] > np.median(Xc[:, f2])
    y_x = (a1 ^ a2).astype(float)
    ca = max(cv_auc(Xc, y_x, l2) for l2 in (1e-3, 1e-2, 1e-1, 3e-1, 1.0))
    cb = max(cv_auc(Xi, y_x, l2) for l2 in (1e-3, 1e-2, 1e-1, 3e-1, 1.0))
    print(f"\n  positive control (synthetic XOR label, pure interaction)")
    print(f"    counts      CV AUC {ca:.4f}")
    print(f"    interaction CV AUC {cb:.4f}   delta {cb - ca:+.4f}")
    print("    -> the protocol CAN detect interactions; the null above is real"
          if cb > ca + 0.05 else
          "    -> WARNING: protocol lacks power; the null above proves nothing")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"counts_cv": ba, "interaction_cv": bb, "delta": bb - ba,
             "l2_counts": la, "l2_interaction": lb,
             "sweep": {str(k): v for k, v in res.items()}}, indent=1))


if __name__ == "__main__":
    main()
