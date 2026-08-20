"""Section 4: the KG-indexed evidence tensor.

ThothGNN gave the knowledge graph exactly one job -- be a neighbour to aggregate
messages from -- and deleting it changed nothing (NO-KG-NODES +0.0049). This
gives it a different job, the one the project's own governing finding says works:
the KG decides what the feature axes ARE.

Every claim cites exactly one of the F BI-RADS stance findings (measured: 15219
of 15220). So instead of pooling over claims, pool over FINDINGS, producing a
fixed F x C tensor per sample regardless of how long the debate ran:

    net        (n_mal - n_ben) / n     which way this finding was argued
    presence   1 if the finding was invoked at all
    mass       share of the debate spent on this finding
    endorsed   fraction of its claims that drew an incoming AGREE
    disputed   fraction that drew an incoming DISAGREE
    open       fraction asserted in round 0

The KG appears a second time, as the INITIALISATION of the readout: `net`'s
weight for finding f starts at the graph's own stance for f (+1 suggests
malignancy, -1 suggests benign). The model therefore starts as "count findings
weighted by what BI-RADS says they mean" and learns a deviation from it.

That gives two falsifiable tests of whether the KG is contributing structurally,
which is the point of the exercise:

    RAND-INIT   same tensor, random weights -> does the KG prior help?
    COLLAPSED   same columns summed over findings -> does the INDEX help?

Both are run as ablations below. If neither hurts, the KG is decorative here too
and the write-up must say so.

Usage:
  python kg_tensor.py --debates .../debates_v5q .../debates_v5q_r1 ...
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import SPLITS, auc, bacc, by_split, kg_findings, load     # noqa: E402

COLS = ("net", "presence", "mass", "endorsed", "disputed", "open")


# ----------------------------------------------------------------- tensor ----
def tensor(claims, F, cols=COLS):
    """F x len(cols) evidence tensor for one claim set."""
    g = defaultdict(list)
    for c in claims:
        if c["finding"] >= 0:
            g[c["finding"]].append(c)
    tot = max(1, len(claims))
    X = np.zeros((F, len(cols)), dtype=np.float64)
    for f, cl in g.items():
        n = len(cl)
        v = {"net": sum(1 if c["stance"] else -1 for c in cl) / n,
             "presence": 1.0,
             "mass": n / tot,
             "endorsed": sum(1 for c in cl if c["in_agree"] > 0) / n,
             "disputed": sum(1 for c in cl if c["in_disagree"] > 0) / n,
             "open": sum(1 for c in cl if c["round"] == 0) / n}
        for j, name in enumerate(cols):
            X[f, j] = v[name]
    return X


def collapse(X):
    """Sum the tensor over findings: same columns, index destroyed."""
    return X.sum(axis=0, keepdims=True)


# ------------------------------------------------------------ evidence wt ----
def fit_evidence(train, alpha=20.0):
    """The 4-run evidence-weighted readout, refit here so it can be an anchor."""
    tot, glob = defaultdict(lambda: [0.0, 0.0]), [0.0, 0.0]
    for r in train:
        for c in r["claims"]:
            k = (c["stance"], f"{c['verdict']}|r{c['round']}")
            tot[k][0] += r["y"]; tot[k][1] += 1
            glob[0] += r["y"]; glob[1] += 1
    prior = glob[0] / max(1.0, glob[1])
    lp = math.log(prior / (1 - prior))
    w = {}
    for k, (s, n) in tot.items():
        p = min(max((s + alpha * prior) / (n + alpha), 1e-4), 1 - 1e-4)
        w[k] = math.log(p / (1 - p)) - lp
    return w


def evidence_score(claims, w):
    v = [w.get((c["stance"], f"{c['verdict']}|r{c['round']}"), 0.0) for c in claims]
    return sum(v) / len(v) if v else 0.0


def mal_share(claims):
    return sum(c["stance"] for c in claims) / len(claims) if claims else 0.5


# ----------------------------------------------------------------- model -----
def train_head(Xtr, atr, ytr, w0, l2, epochs=1500, lr=0.5):
    """score = beta * anchor + (w.x + b), penalised toward the KG prior.

    Two design points, both forced by measurement rather than taste.

    NO MULTIPLICATIVE GAMMA. The first version wrote `gamma * (w.x + b)` with
    both gamma and w starting at zero. That is a bilinear dead point: dL/dgamma
    is proportional to w and dL/dw to gamma, so both stay zero forever and the
    model reports the anchor back with gamma = 0.000. Folding the scale into w
    keeps "starts at the anchor" (w = 0) while leaving the gradient alive.

    THE KG PRIOR IS A PENALTY CENTRE, NOT AN INITIALISATION. With an L2 penalty
    this objective is convex, so the optimum is unique and the starting point
    provably cannot survive to it -- initialising from the graph would have been
    untestable. Penalising ||w - w_kg|| instead shrinks the solution toward the
    graph's own stance for each finding, which does change the answer and can be
    ablated by moving the centre to zero.
    """
    w = w0.copy()
    b, beta = 0.0, 1.0
    n = len(ytr)
    for _ in range(epochs):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(beta * atr + z, -30, 30)))
        e = p - ytr
        w -= lr * ((Xtr.T @ e) / n + l2 * (w - w0))
        b -= lr * float(e.mean())
        beta -= lr * float((e * atr).mean())
    return w, b, beta


def predict(X, a, w, b, beta):
    return beta * a + (X @ w + b)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ main -----
def runs_of(r):
    g = defaultdict(list)
    for c in r["claims"]:
        g[c["run"]].append(c)
    return list(g.values())


def featurise(rows, F, cols, per_run, collapsed):
    """(X, y, group) -- group is the sample index, so CV folds never split a sample."""
    X, y, grp = [], [], []
    for i, r in enumerate(rows):
        sets = runs_of(r) if per_run else [r["claims"]]
        for cl in sets:
            T = tensor(cl, F, cols)
            X.append(collapse(T) if collapsed else T)
            y.append(r["y"]); grp.append(i)
    X = np.array(X).reshape(len(X), -1)
    return X, np.array(y, dtype=float), np.array(grp)


def anchor_of(rows, kind, ev_w, per_run):
    out = []
    for r in rows:
        for cl in (runs_of(r) if per_run else [r["claims"]]):
            out.append(0.0 if kind == "none" else
                       logit(mal_share(cl)) if kind == "mal_share" else
                       evidence_score(cl, ev_w))
    return np.array(out)


def kg_centre(cols, collapsed, use_prior, F, prior):
    """KG stance for each finding, placed on the `net` column of the weight vector."""
    C = len(cols)
    w = np.zeros((1 if collapsed else F) * C)
    if use_prior and "net" in cols and not collapsed:
        j = cols.index("net")
        for f in range(F):
            w[f * C + j] = prior[f]
    return w


def pool_by_group(s, grp, n):
    """Average per-run scores back to one score per sample."""
    out = np.zeros(n); cnt = np.zeros(n)
    np.add.at(out, grp, s); np.add.at(cnt, grp, 1.0)
    return out / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    findings = kg_findings()
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=np.float64)
    data = load(args.debates, findings)
    by = by_split(data)
    print(f"pooled {len(args.debates)} run(s): "
          + "  ".join(f"{s} {len(by[s])}" for s in SPLITS) + f" | F={F}")

    ev_w = fit_evidence(by["train"])

    def w_centre(cols, collapsed, use_prior):
        return kg_centre(cols, collapsed, use_prior, F, prior)

    CONFIGS = {
        "TENSOR + KG prior":     dict(cols=COLS, prior=True,  collapse=False, aug=False),
        "TENSOR, no KG prior":   dict(cols=COLS, prior=False, collapse=False, aug=False),
        "COLLAPSED (no index)":  dict(cols=COLS, prior=False, collapse=True,  aug=False),
        "TENSOR + prior + aug":  dict(cols=COLS, prior=True,  collapse=False, aug=True),
        "TENSOR net-only":       dict(cols=("net",), prior=True, collapse=False, aug=False),
        "TENSOR no-endorse":     dict(cols=("net","presence","mass","open"),
                                      prior=True, collapse=False, aug=False),
    }
    L2S = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)

    res = {}
    for kind in ("evidence", "mal_share", "none"):
        print(f"\n=== anchor: {kind} ===")
        a_te = anchor_of(by["test"], kind, ev_w, False)
        y_te = np.array([r["y"] for r in by["test"]], dtype=float)
        a_fit = np.concatenate([anchor_of(by["train"], kind, ev_w, False),
                                anchor_of(by["val"], kind, ev_w, False)])
        y_fit = np.array([r["y"] for r in by["train"]] + [r["y"] for r in by["val"]], dtype=float)
        thr = max(sorted(set(a_fit.tolist())),
                  key=lambda t: bacc(a_fit.tolist(), y_fit.tolist(), t))
        base = auc(list(a_te[y_te == 1]), list(a_te[y_te == 0]))
        print(f"  {'anchor alone':24s} TEST {base:.4f}   bAcc "
              f"{bacc(a_te.tolist(), y_te.tolist(), thr):.4f}")
        res[f"{kind}|anchor alone"] = {"test": base}

        for name, cfg in CONFIGS.items():
            cols, coll, aug = cfg["cols"], cfg["collapse"], cfg["aug"]
            Xtr, ytr, gtr = featurise(by["train"], F, cols, aug, coll)
            Xva, yva, gva = featurise(by["val"],   F, cols, aug, coll)
            Xte, yte, gte = featurise(by["test"],  F, cols, aug, coll)
            atr = anchor_of(by["train"], kind, ev_w, aug)
            ava = anchor_of(by["val"],   kind, ev_w, aug)
            ate = anchor_of(by["test"],  kind, ev_w, aug)
            w0 = w_centre(cols, coll, cfg["prior"])

            # ---- L2 by grouped k-fold CV on train, never on val or test ----
            usamp = np.unique(gtr)
            rng = np.random.default_rng(0); order = rng.permutation(usamp)
            folds = np.array_split(order, args.folds)
            cv = {}
            for l2 in L2S:
                sc, gs, ys = [], [], []
                for fo in folds:
                    m = np.isin(gtr, fo)
                    if m.sum() == 0 or (~m).sum() == 0:
                        continue
                    w, b, be = train_head(Xtr[~m], atr[~m], ytr[~m], w0, l2, args.epochs)
                    sc.append(predict(Xtr[m], atr[m], w, b, be))
                    gs.append(gtr[m]); ys.append(ytr[m])
                s_ = np.concatenate(sc); g_ = np.concatenate(gs); y_ = np.concatenate(ys)
                if aug:
                    n = int(gtr.max()) + 1
                    ps = pool_by_group(s_, g_, n)
                    py = np.zeros(n); py[g_] = y_
                    keep = np.isin(np.arange(n), np.unique(g_))
                    s_, y_ = ps[keep], py[keep]
                cv[l2] = auc(list(s_[y_ == 1]), list(s_[y_ == 0]))
            l2 = max(cv, key=cv.get)

            w, b, be = train_head(Xtr, atr, ytr, w0, l2, args.epochs)
            def sc_of(X, a, g, rows):
                s_ = predict(X, a, w, b, be)
                return pool_by_group(s_, g, len(rows)) if aug else s_
            s_tr = sc_of(Xtr, atr, gtr, by["train"])
            s_va = sc_of(Xva, ava, gva, by["val"])
            s_te = sc_of(Xte, ate, gte, by["test"])
            Y = {k: np.array([r["y"] for r in by[k]], dtype=float) for k in SPLITS}
            a_tr = auc(list(s_tr[Y["train"] == 1]), list(s_tr[Y["train"] == 0]))
            a_va = auc(list(s_va[Y["val"] == 1]), list(s_va[Y["val"] == 0]))
            a_t  = auc(list(s_te[Y["test"] == 1]), list(s_te[Y["test"] == 0]))
            sf = np.concatenate([s_tr, s_va]); yf = np.concatenate([Y["train"], Y["val"]])
            t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
            bb = bacc(s_te.tolist(), Y["test"].tolist(), t)
            print(f"  {name:24s} TEST {a_t:.4f}   bAcc {bb:.4f}   "
                  f"(train {a_tr:.4f} val {a_va:.4f} cv {cv[l2]:.4f} l2 {l2} "
                  f"beta {be:+.2f} ntr {len(ytr)})")
            res[f"{kind}|{name}"] = {"train": a_tr, "val": a_va, "test": a_t, "bacc": bb,
                                     "cv": cv[l2], "l2": l2, "beta": be, "n_train": len(ytr)}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
