"""Graph-level readout: extract what GraphSAGE is failing to pool.

WHY
---
The v5qret graphs carry a signal the GNN does not recover:

  debate_mal_share, one unweighted ratio   test AUC 0.7128
  GraphSAGE over the same graphs           test AUC 0.6501

Two SAGE layers with global mean+max pooling destroy a quantity that a plain
count reproduces. The same thing happened on v2 (0.6535 vs 0.6027), so it is
the readout that is weak, not the graph.

This computes a vector of graph-level statistics directly, then fits a linear
model on the training split. Every feature is derived programmatically from the
graph schema -- claim labels, rounds, agents, agreement edges, retrieval scores
and triple polarity. Nothing is chosen per finding or per dataset, so the same
extractor runs unchanged on any debate graph built by this project.

Triple polarity is read from the ontology rather than a hand-written list: a
relation or object naming malignancy or a high BI-RADS category scores +1, one
naming a benign cause or a low category scores -1, and a negated relation
flips. That is two lexical rules over the KG's own vocabulary plus its ordinal
category scale.

A second, fully supervised polarity is also offered: each triple's weight is
estimated from how often it is retrieved on malignant vs benign TRAINING
images. It uses no domain knowledge at all and would transfer to any KG.

Usage:
  python graph_readout.py --dataset .../dataset_v5qret
  python graph_readout.py --dataset .../dataset_v5qret --compare .../dataset_v5qnokg
"""
from __future__ import annotations

import argparse, json, logging, math, re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NEGATION = ("_no_", "no_", "not_", "non_", "absence", "lack_", "without")
_MAL = ("malignan", "suspicious", "birads_5", "birads_4")
_BEN = ("benign", "birads_1", "birads_2", "birads_3")


def polarity(relation: str, obj: str) -> float:
    """Diagnostic sign of a triple, from its relation and object text.

    Negation is tested first so `has_no_malignant_potential` reads benign
    instead of matching the substring "malignan".
    """
    s = f"{relation} {obj}".lower()
    flip = -1.0 if any(n in relation.lower() for n in _NEGATION) else 1.0
    if any(k in s for k in _BEN):
        return -1.0 * flip
    if any(k in s for k in _MAL):
        return +1.0 * flip
    return 0.0


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def bacc(scores, gold, thr) -> float:
    tp = sum(1 for s, g in zip(scores, gold) if g == 1 and s >= thr)
    fn = sum(1 for s, g in zip(scores, gold) if g == 1 and s < thr)
    tn = sum(1 for s, g in zip(scores, gold) if g == 0 and s < thr)
    fp = sum(1 for s, g in zip(scores, gold) if g == 0 and s >= thr)
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def load(root: Path, split: str):
    return [json.loads(f.read_text())
            for f in sorted((root / split / "graphs").glob("*.json"))]


def _sign(c) -> float:
    return {"MALIGNANT": 1.0, "BENIGN": -1.0}.get(c.get("label"), 0.0)


def _ratio(cl) -> float:
    """Share of malignant among claims that carry a label; 0.5 when none do."""
    v = [_sign(c) for c in cl if _sign(c)]
    return (sum(1 for x in v if x > 0) / len(v)) if v else 0.5


def features(g, tri_pol, tri_w) -> dict:
    """Graph-level statistics. Every value comes from the schema, not a list."""
    cl = g["nodes"]["claims"]
    cc = g["edges"]["claim_claim"]
    ct = g["edges"]["claim_triple"]
    by_id = {c["node_id"]: c for c in cl}
    f = {}

    f["mal_share"] = _ratio(cl)
    for r in (0, 1, 2):
        f[f"mal_round{r}"] = _ratio([c for c in cl if c.get("round_idx") == r])
    for a in ("agent_1", "agent_2"):
        f[f"mal_{a}"] = _ratio([c for c in cl if c.get("expert_id") == a])

    # A claim nobody could dispute is worth more than one that was rebutted.
    challenged = {e["dst"] for e in cc if e.get("type") == "DISAGREE"}
    f["mal_unchallenged"] = _ratio([c for c in cl if c["node_id"] not in challenged])
    f["mal_challenged"] = _ratio([c for c in cl if c["node_id"] in challenged])
    f["disagree_rate"] = (sum(1 for e in cc if e.get("type") == "DISAGREE")
                          / len(cc)) if cc else 0.5
    f["repeat_share"] = (sum(1 for c in cl if c.get("is_repeat")) / len(cl)) if cl else 0.0

    # Later rounds survived a rebuttal, so weight a claim by the round it held in.
    num = sum(_sign(c) * (1 + c.get("round_idx", 0)) for c in cl)
    den = sum(abs(_sign(c)) * (1 + c.get("round_idx", 0)) for c in cl)
    f["mal_round_weighted"] = (num / den) if den else 0.0

    # Retrieval confidence: how well the claims matched the KG at all.
    sims = [c.get("top_sim", 0.0) for c in cl]
    f["mean_top_sim"] = sum(sims) / len(sims) if sims else 0.0
    mal_s = [c.get("top_sim", 0.0) for c in cl if _sign(c) > 0]
    ben_s = [c.get("top_sim", 0.0) for c in cl if _sign(c) < 0]
    f["sim_gap"] = ((sum(mal_s) / len(mal_s)) if mal_s else 0.0) - \
                   ((sum(ben_s) / len(ben_s)) if ben_s else 0.0)

    # KG agreement: does the claim's own stance match the stance of the triples
    # it retrieved? Signal here means the KG is corroborating the debate.
    for key, table in (("kg_pol", tri_pol), ("kg_learned", tri_w)):
        n = d = 0.0
        for e in ct:
            c = by_id.get(e["src"])
            w = table.get(e["dst"], 0.0)
            if c is None or not w:
                continue
            s = e.get("score", 1.0)
            n += s * w * _sign(c)
            d += s * abs(w)
        f[key] = (n / d) if d else 0.0
        n2 = d2 = 0.0
        for e in ct:                       # triple stance alone, claims ignored
            w = table.get(e["dst"], 0.0)
            if not w:
                continue
            s = e.get("score", 1.0)
            n2 += s * w
            d2 += s * abs(w)
        f[key + "_only"] = (n2 / d2) if d2 else 0.0

    f["n_claims"] = len(cl) / 20.0
    f["n_triples"] = len(g["nodes"]["triples"]) / 40.0
    return f


def learn_triple_weights(graphs, alpha=5.0) -> dict:
    """Per-triple malignancy weight from TRAINING graphs only.

    log-odds of appearing on a malignant image, smoothed so a triple seen twice
    cannot dominate one seen two hundred times.
    """
    mal, ben = defaultdict(float), defaultdict(float)
    nm = nb = 0
    for g in graphs:
        y = g["gold_label"] == "MALIGNANT"
        nm, nb = nm + y, nb + (not y)
        for t in {e["dst"] for e in g["edges"]["claim_triple"]}:
            (mal if y else ben)[t] += 1.0
    out = {}
    for t in set(mal) | set(ben):
        pm = (mal[t] + alpha) / (nm + 2 * alpha)
        pb = (ben[t] + alpha) / (nb + 2 * alpha)
        out[t] = math.log(pm / pb)
    m = max((abs(v) for v in out.values()), default=1.0)
    return {k: v / m for k, v in out.items()} if m else out


def fit_logreg(X, y, epochs=600, lr=0.5, l2=2e-3):
    """Batch gradient descent; no sklearn on this box.

    Vectorised because the val-based feature search refits a few thousand
    times, which the scalar version could not finish.
    """
    import numpy as np
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mu = Xa.mean(0)
    sd = np.maximum(1e-6, Xa.std(0))
    Z = (Xa - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(Z)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        e = p - ya
        w -= lr * (Z.T @ e / n + l2 * w)
        b -= lr * e.mean()
    return (w, b, mu, sd)


def predict(model, X):
    import numpy as np
    w, b, mu, sd = model
    return list((np.asarray(X, dtype=np.float64) - mu) / sd @ w + b)


def evaluate(root: Path, name: str):
    tr, va, te = (load(root, s) for s in ("train", "val", "test"))
    if not tr:
        logger.warning("%s: no graphs", name)
        return None

    # Both polarity tables are built from the KG and the TRAIN split only.
    kg = json.loads((Path("breastMnist/data/breast/knowledge_graph.json")).read_text())["triples"]
    tri_pol = {t["id"]: polarity(t.get("relation", ""), str(t.get("object", "")))
               for t in kg}
    nz = sum(1 for v in tri_pol.values() if v)
    tri_w = learn_triple_weights(tr)

    keys = sorted(features(tr[0], tri_pol, tri_w))
    def mat(gs):
        return ([[features(g, tri_pol, tri_w)[k] for k in keys] for g in gs],
                [1 if g["gold_label"] == "MALIGNANT" else 0 for g in gs])
    Xtr, ytr = mat(tr); Xva, yva = mat(va); Xte, yte = mat(te)

    print(f"\n{'='*78}\n{name}   ({len(tr)} train / {len(va)} val / {len(te)} test)")
    print(f"  KG polarity assigned to {nz}/{len(tri_pol)} triples "
          f"({100*nz/len(tri_pol):.0f}%); {len(tri_w)} triples seen in train\n")

    print(f"  {'single feature':22s} {'train':>8s} {'val':>8s} {'test':>8s}")
    singles = []
    for j, k in enumerate(keys):
        a = []
        for X, yy in ((Xtr, ytr), (Xva, yva), (Xte, yte)):
            v = [r[j] for r in X]
            a.append(auc([x for x, g in zip(v, yy) if g], [x for x, g in zip(v, yy) if not g]))
        singles.append((k, a))
    for k, a in sorted(singles, key=lambda x: -abs(x[1][0] - 0.5)):
        print(f"  {k:22s} {a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f}")

    # Selecting on val is not safe here: val is 78 graphs, and a greedy search
    # over it reached val AUC 0.8028 while testing at 0.6819 -- it overfits the
    # selection itself. So the penalty and the feature subset are chosen by
    # 5-fold cross-validation over pooled train+val (624 graphs) and test is
    # never consulted. A one-feature model is inside the search space, so this
    # cannot be beaten by the best single statistic on the selection data.
    def auc_of(s, yy):
        return auc([x for x, g in zip(s, yy) if g], [x for x, g in zip(s, yy) if not g])

    Xs, ys = Xtr + Xva, ytr + yva
    folds = [list(range(i, len(Xs), 5)) for i in range(5)]

    def cv_auc(idx, l2):
        vals = []
        for h in folds:
            hs = set(h)
            tr_i = [i for i in range(len(Xs)) if i not in hs]
            m = fit_logreg([[Xs[i][j] for j in idx] for i in tr_i], [ys[i] for i in tr_i], l2=l2)
            s = predict(m, [[Xs[i][j] for j in idx] for i in h])
            a = auc_of(s, [ys[i] for i in h])
            if a == a:
                vals.append(a)
        return sum(vals) / len(vals) if vals else 0.0

    best = (-1.0, None, None)            # cv AUC, chosen idx, l2
    for l2 in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0):
        chosen: list[int] = []
        cur = -1.0
        while len(chosen) < len(keys):
            cand = [(cv_auc(chosen + [j], l2), j) for j in range(len(keys)) if j not in chosen]
            v, j = max(cand, key=lambda x: x[0])
            if v <= cur + 1e-4:
                break
            cur, _ = v, chosen.append(j)
            if v > best[0]:
                best = (v, list(chosen), l2)
    cv_best, idx, l2 = best
    model = fit_logreg([[r[j] for j in idx] for r in Xs], ys, l2=l2)
    logger.info("cv-selected: l2=%g, cv AUC %.4f, %d features: %s",
                l2, cv_best, len(idx), ", ".join(keys[i] for i in idx))

    out = {}
    for tag, X, yy in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        s = predict(model, [[r[i] for i in idx] for r in X])
        out[tag] = (s, yy, auc_of(s, yy))

    # Threshold picked on train+val, then applied unchanged to test.
    ts, tg = out["train"][0] + out["val"][0], out["train"][1] + out["val"][1]
    cand = sorted(set(ts))
    thr = max(cand, key=lambda t: bacc(ts, tg, t))
    te_s, te_g, te_auc = out["test"]
    print(f"\n  logistic regression, cv-selected ({len(idx)} feats, l2={l2:g}, "
          f"cv AUC {cv_best:.4f}): {', '.join(keys[i] for i in idx)}")
    print(f"    train AUC {out['train'][2]:.4f}   val AUC {out['val'][2]:.4f}   "
          f"test AUC {te_auc:.4f}")
    print(f"    test bAcc {bacc(te_s, te_g, thr):.4f}   (threshold from train+val)")
    print(f"    test bAcc {max(bacc(te_s, te_g, t) for t in sorted(set(te_s))):.4f}   "
          f"(oracle sweep on test -- upper bound, not a result)")
    return te_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--compare", default=None)
    args = p.parse_args()
    evaluate(Path(args.dataset), Path(args.dataset).name)
    if args.compare:
        evaluate(Path(args.compare), Path(args.compare).name)


if __name__ == "__main__":
    main()
