"""Score the judge arm against the aggregator it is meant to test.

Every debate number in this project comes from an aggregator WE chose -- mean
claim grade on retina, modal class on derma. So C2 ("debate adds nothing") is,
strictly, a claim about our readout of the debate rather than about the debate.
The judge arm removes that confound: a fresh model reads the transcript only,
never the image, and issues the verdict itself.

Three rows are therefore reported on identical samples:

  judge        a model reads the whole transcript and decides
  aggregator   the readout used everywhere else in this project
  GNN          the full framework, quoted from its own results file

If the judge beats the aggregator, C2 was partly an artefact of the readout and
the debate carries more than we were extracting. If it does not, C2 becomes a
statement about the debate CONTENT, which is the stronger claim.

The judge emits a discrete label, so its "score" for AUC is that label. That is
a coarse ranking (5 distinct values on retina) and it costs the judge some AUC
relative to a continuous score -- reported, not hidden, and the discrete metrics
(QWK, MAE, macro-recall, accuracy) are unaffected and carry the comparison.
"""
from __future__ import annotations

import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "retinaMnist/src/retina_kg"))
sys.path.insert(0, str(REPO / "experiments/26_baselines"))
from metrics_ordinal import qwk, mae, macro_recall, adjacent_accuracy, fast_auc  # noqa
from corpus_io import load_corpus                                                # noqa

TREES = {
    "retina": dict(judge=REPO / "retinaMnist/data/retina/judge",
                   debates=REPO / "retinaMnist/data/retina/debates_r1",
                   gold="gold_grade", kind="ordinal", n_class=5,
                   gnn=REPO / "retinaMnist/results/ordinal_full.json"),
    "derma": dict(judge=REPO / "dermaMnist/data/derma/judge",
                  debates=REPO / "dermaMnist/data/derma/debates_d1",
                  gold="gold_class", kind="nominal", n_class=7,
                  gnn=REPO / "dermaMnist/results/derma_nominal.json"),
}


def read_jsonl_dir(d, split):
    """Dedup by sample_id, first wins -- the judge fan-out can double-issue a
    shard, exactly as the B1 corpus did (236 extra records that time)."""
    seen, out = set(), {}
    for f in sorted(Path(d).glob(f"{split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            sid = r.get("sample_id")
            if sid in seen:
                continue
            seen.add(sid); out[sid] = r
    return out


def aggregate(rec, kind, classes=None):
    """The readout used everywhere else: mean claim grade / modal class."""
    cl = rec.get("claims") or []
    if kind == "ordinal":
        g = [c["grade"] for c in cl if c.get("grade") is not None]
        return float(np.mean(g)) if g else None
    lab = [c.get("label") for c in cl if c.get("label")]
    return Counter(lab).most_common(1)[0][0] if lab else None


def metrics(y, yhat, kind, n_class, score=None):
    y = np.asarray(y, int); yhat = np.asarray(yhat, int)
    m = {"accuracy": float((y == yhat).mean()),
         "macro_recall": macro_recall(y, yhat, n_class)}
    if kind == "ordinal":
        s = np.asarray(score if score is not None else yhat, float)
        ref = (y >= 2).astype(int)
        m |= {"qwk": qwk(y, yhat, n_class), "mae": mae(y, yhat),
              "adjacent_acc": adjacent_accuracy(y, yhat),
              "referable_auc": fast_auc(s, ref == 1)}
    return m


def paired_boot(y, a, b, kind, n_class, draws=2000, seed=0):
    """CI on (judge - aggregator) macro-recall over the SAME samples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int); a = np.asarray(a, int); b = np.asarray(b, int)
    n = len(y); d = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        d.append(macro_recall(y[i], a[i], n_class) - macro_recall(y[i], b[i], n_class))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run(tree, split="test"):
    T = TREES[tree]
    J = read_jsonl_dir(T["judge"], split)
    if not J:
        return None
    deb = {r["sample_id"]: r for r in load_corpus(T["debates"], (split,))[split]}

    classes = None
    if T["kind"] == "nominal":
        classes = sorted({r[T["gold"]] for r in deb.values()})
        cidx = {c: i for i, c in enumerate(classes)}

    y, jv, ag = [], [], []
    for sid, r in sorted(J.items()):
        d = deb.get(sid)
        if d is None or r.get("verdict") is None:
            continue
        a = aggregate(d, T["kind"], classes)
        if a is None:
            continue
        if T["kind"] == "ordinal":
            y.append(int(r["gold"])); jv.append(int(r["verdict"]))
            ag.append(int(round(float(a))))
        else:
            if r["verdict"] not in cidx or a not in cidx:
                continue
            y.append(cidx[d[T["gold"]]]); jv.append(cidx[r["verdict"]])
            ag.append(cidx[a])

    out = {"tree": tree, "split": split, "n": len(y),
           "judge_records": len(J), "coverage": len(y) / max(len(deb), 1),
           "judge": metrics(y, jv, T["kind"], T["n_class"]),
           "aggregator": metrics(y, ag, T["kind"], T["n_class"])}
    m, lo, hi = paired_boot(y, jv, ag, T["kind"], T["n_class"])
    out["judge_minus_aggregator_macroR"] = {"mean": m, "lo": lo, "hi": hi,
                                            "excludes_zero": bool(lo > 0 or hi < 0)}
    try:
        g = json.loads(Path(T["gnn"]).read_text())
        # retina stores per-head blocks ("A: GNN (real KG)"), derma a flat one
        blk = g.get("GNN (real KG)") or g.get("A: GNN (real KG)") or {}
        out["gnn_quoted"] = {k: blk[k] for k in
                             ("accuracy", "macro_recall", "qwk", "referable_auc")
                             if k in blk}
    except Exception as e:
        out["gnn_quoted"] = {"error": str(e)}
    return out


if __name__ == "__main__":
    res = {}
    for tree in ("retina", "derma"):
        for sp in ("test",):
            r = run(tree, sp)
            if r is None:
                print(f"{tree}/{sp}: no judge records yet"); continue
            res[f"{tree}_{sp}"] = r
            print(f"\n=== {tree}/{sp}  n={r['n']}  coverage={r['coverage']:.3f} ===")
            keys = sorted(set(r["judge"]) | set(r["aggregator"]))
            print(f"{'metric':16s} {'judge':>10s} {'aggregator':>12s} {'delta':>10s}")
            for k in keys:
                a, b = r["judge"].get(k), r["aggregator"].get(k)
                if a is None or b is None:
                    continue
                print(f"{k:16s} {a:10.4f} {b:12.4f} {a-b:+10.4f}")
            d = r["judge_minus_aggregator_macroR"]
            print(f"paired bootstrap macro-recall (judge - aggregator): "
                  f"{d['mean']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                  f"  {'EXCLUDES 0' if d['excludes_zero'] else 'includes 0'}")
            print(f"GNN quoted: {r['gnn_quoted']}")
    if res:
        p = REPO / "shared/results/judge_score.json"
        p.write_text(json.dumps(res, indent=1))
        print(f"\n-> {p}")
