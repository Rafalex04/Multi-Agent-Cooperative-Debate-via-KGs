"""Score the two-turn BI-RADS run, saturation diagnostics first.

The binary contrast in 08_contrastive failed silently: its AUC looked merely
mediocre, and only the per-class means revealed that every context answered
identically for all 156 images. So the category distribution and the spread are
reported BEFORE any AUC. If a turn emits one category for every image, its AUC
is a statement about the tie-correction rule, not about the model.

Two readouts are reported for the same run, because they answer differently:

  category   The BI-RADS the model committed to, converted to a malignancy
             probability by the knowledge graph, averaged over the two turns.
             This is the experiment as specified, and the verdict is the ladder
             rung nearest that average.
  digit      The expectation over the first-token digit distribution, averaged
             the same way. It keeps a gradient when the committed category
             quantises, which is exactly what happened to the 0-10 predecessor.

Thresholds, for balanced accuracy:
  honest     swept on train+val, applied unchanged to test.
  clinical   the biopsy threshold the graph itself defines -- the lowest rung
             above BI-RADS 3, i.e. the first band whose malignancy risk exceeds
             2 percent. Not fitted on anything.

Usage:
  python evaluate_birads2.py
"""
from __future__ import annotations

import argparse, glob, json, statistics as st, sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from birads_scale import load_scale, nearest                          # noqa: E402

SPLITS = ("train", "val", "test")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def bacc(scores, golds, t):
    tp = sum(1 for s, g in zip(scores, golds) if g == 1 and s >= t)
    fn = sum(1 for s, g in zip(scores, golds) if g == 1 and s < t)
    tn = sum(1 for s, g in zip(scores, golds) if g == 0 and s < t)
    fp = sum(1 for s, g in zip(scores, golds) if g == 0 and s >= t)
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def load(results_dir: Path, tag: str):
    by = {s: [] for s in SPLITS}
    for f in sorted(results_dir.glob(f"{tag}_*.jsonl")):
        split = f.stem.split("_")[1]
        if split not in by:
            continue
        for ln in f.read_text().splitlines():
            if ln.strip():
                by[split].append(json.loads(ln))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(_HERE.parent / "results"))
    ap.add_argument("--tag", default="birads2")
    ap.add_argument("--kg", default=str(_HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"))
    args = ap.parse_args()

    scale = load_scale(args.kg)
    by = load(Path(args.results), args.tag)
    print("samples: " + "  ".join(f"{s} {len(by[s])}" for s in SPLITS))

    te = by["test"]
    if not te:
        print("no test data yet"); return

    # ---------------- saturation, before any AUC ----------------
    print("\n=== saturation diagnostics (test) ===")
    print(f"  malignant-primed categories: {dict(Counter(r['cat_mal'] for r in te))}")
    print(f"  benign-primed categories   : {dict(Counter(r['cat_ben'] for r in te))}")
    for k in ("p_mal_turn", "p_ben_turn", "exp_mal", "exp_ben"):
        v = [r[k] for r in te if r.get(k) is not None]
        if v:
            print(f"  {k:11s} mean {st.mean(v):7.3f}  sd {st.pstdev(v):7.4f}  "
                  f"distinct {len(set(v))}")
    unparsed = sum(1 for r in te if r["cat_mal"] is None or r["cat_ben"] is None)
    print(f"  unparsed replies: {unparsed}/{len(te)}")

    # ---------------- readouts ----------------
    def cat_avg(r):
        a, b = r.get("p_mal_turn"), r.get("p_ben_turn")
        return None if a is None or b is None else (a + b) / 2

    def ord_avg(r):
        a, b = r.get("ord_mal"), r.get("ord_ben")
        return None if a is None or b is None else (a + b) / 2

    def dig_avg(r):
        a, b = r.get("exp_mal"), r.get("exp_ben")
        return None if a is None or b is None else (a + b) / 2

    readouts = [
        ("category avg  (as specified)", cat_avg),
        ("ordinal avg",                  ord_avg),
        ("digit-expectation avg",        dig_avg),
        ("malignant-primed turn alone",  lambda r: r.get("p_mal_turn")),
        ("benign-primed turn alone",     lambda r: r.get("p_ben_turn")),
        ("exp malignant-primed alone",   lambda r: r.get("exp_mal")),
        ("exp benign-primed alone",      lambda r: r.get("exp_ben")),
    ]

    # clinical cut: first rung above BI-RADS 3, read off the ladder
    below = [d for d in scale if d["p_mal"] <= 2.0]
    clinical_p = max(d["p_mal"] for d in below) if below else 0.0

    print("\n=== AUC (MALIGNANT = positive) ===")
    print(f"  {'readout':30s} {'train':>7s} {'val':>7s} {'TEST':>7s}   "
          f"{'bAcc(honest)':>12s} {'bAcc(clin)':>10s}")
    for name, fn in readouts:
        cell = {}
        for s in SPLITS:
            v = [(fn(r), 1 if r["gold"] == "MALIGNANT" else 0) for r in by[s]]
            v = [(a, g) for a, g in v if a is not None]
            cell[s] = v
        a_tr, a_va, a_te = (auc([x for x, g in cell[s] if g], [x for x, g in cell[s] if not g])
                            for s in SPLITS)
        fit = cell["train"] + cell["val"]
        ts, tg = [x for x, _ in cell["test"]], [g for _, g in cell["test"]]
        if fit and ts:
            fs, fg = [x for x, _ in fit], [g for _, g in fit]
            thr = max(sorted(set(fs)), key=lambda t: bacc(fs, fg, t))
            hb = bacc(ts, tg, thr)
        else:
            hb = float("nan")
        # The clinical cut lives on the probability scale, so it is only
        # meaningful for readouts expressed in those units.
        on_p_scale = fn in (cat_avg,) or name.endswith("turn alone") and "exp" not in name
        cb = bacc(ts, tg, clinical_p) if on_p_scale else None
        cbs = f"{cb:10.4f}" if cb is not None else f"{'-':>10s}"
        print(f"  {name:30s} {a_tr:7.4f} {a_va:7.4f} {a_te:7.4f}   "
              f"{hb:12.4f} {cbs}")

    # ---------------- the verdict the experiment asked for ----------------
    print(f"\n=== verdict distribution (category average -> nearest ladder rung) ===")
    vs = Counter()
    for r in te:
        a = cat_avg(r)
        if a is not None:
            vs[nearest(a, scale)["code"]] += 1
    print("  " + str(dict(vs)))
    print(f"  clinical cut used for bAcc(clin): score > {clinical_p}% "
          f"(above BI-RADS 3)")


if __name__ == "__main__":
    main()
