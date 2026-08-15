"""Score zero-shot BreastMNIST runs with imbalance-aware metrics.

BreastMNIST test is 27% malignant, so plain accuracy rewards a model that
always answers "benign" with 0.731. Every headline number here corrects for
that:

  balanced accuracy   (sensitivity + specificity) / 2 — the metric to read
                      first. Identical to accuracy re-weighted so each class
                      contributes equally regardless of its frequency.
  weighted accuracy   accuracy with per-sample weights inversely proportional
                      to class frequency. Mathematically the same as balanced
                      accuracy; reported separately because it is what
                      "weight by class fraction" literally means.
  MCC                 correlation between prediction and truth over the whole
                      confusion matrix; 0 = chance, robust under imbalance.
  AUC                 ranking quality from P(malignant); independent of the
                      yes/no threshold. Chance = 0.5.

Usage:
  python evaluate.py                          # every run in results/
  python evaluate.py --file results/x.jsonl   # one run
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
POS = "MALIGNANT"  # positive class


def _confusion(rows: list[dict]) -> tuple[int, int, int, int]:
    """Return (tp, fn, tn, fp) treating MALIGNANT as positive."""
    tp = sum(1 for r in rows if r["gold"] == POS and r["pred"] == POS)
    fn = sum(1 for r in rows if r["gold"] == POS and r["pred"] != POS)
    tn = sum(1 for r in rows if r["gold"] != POS and r["pred"] != POS)
    fp = sum(1 for r in rows if r["gold"] != POS and r["pred"] == POS)
    return tp, fn, tn, fp


def _auc(scores: list[float], positive: list[bool]) -> float | None:
    """AUC via rank statistic, with ties averaged. None if a class is absent."""
    pairs = [(s, p) for s, p in zip(scores, positive) if s is not None]
    if not pairs:
        return None
    n_pos = sum(1 for _, p in pairs if p)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ordered = sorted(pairs, key=lambda x: x[0])
    ranks: list[float] = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(r for r, (_, p) in zip(ranks, ordered) if p)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score(rows: list[dict]) -> dict:
    """Compute the full metric set for one run."""
    n = len(rows)
    unparsed = sum(1 for r in rows if r["pred"] is None)
    tp, fn, tn, fp = _confusion(rows)

    n_pos, n_neg = tp + fn, tn + fp
    sens = tp / n_pos if n_pos else float("nan")   # recall on MALIGNANT
    spec = tn / n_neg if n_neg else float("nan")   # recall on BENIGN
    bal_acc = (sens + spec) / 2

    # Weighted accuracy: each sample weighted by 1/frequency of its class.
    # Equals balanced accuracy; computed the literal way as a cross-check.
    w_pos = n / (2 * n_pos) if n_pos else 0.0
    w_neg = n / (2 * n_neg) if n_neg else 0.0
    weighted_acc = (tp * w_pos + tn * w_neg) / (n_pos * w_pos + n_neg * w_neg)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")

    auc = _auc([r.get("p_malignant") for r in rows], [r["gold"] == POS for r in rows])
    n_scored = sum(1 for r in rows if r.get("p_malignant") is not None)

    times = [r["time_s"] for r in rows if "time_s" in r]

    return {
        "n": n,
        "n_malignant": n_pos,
        "n_benign": n_neg,
        "unparsed": unparsed,
        "majority_baseline": max(n_pos, n_neg) / n if n else float("nan"),
        "accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": bal_acc,
        "weighted_accuracy": weighted_acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "n_with_scores": n_scored,
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "mean_time_s": sum(times) / len(times) if times else float("nan"),
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "   n/a"
    if isinstance(v, float) and math.isnan(v):
        return "   nan"
    return f"{v:6.4f}"


def report(name: str, m: dict) -> None:
    c = m["confusion"]
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    print(f"  samples {m['n']}  ({m['n_malignant']} malignant / {m['n_benign']} benign)"
          f"   unparsed: {m['unparsed']}")
    print(f"\n  {'BALANCED ACCURACY':<22} {_fmt(m['balanced_accuracy'])}   <- headline")
    print(f"  {'weighted accuracy':<22} {_fmt(m['weighted_accuracy'])}   (same by construction)")
    print(f"  {'AUC':<22} {_fmt(m['auc'])}   chance 0.5, n={m['n_with_scores']}")
    print(f"  {'MCC':<22} {_fmt(m['mcc'])}   chance 0.0")
    print(f"\n  {'raw accuracy':<22} {_fmt(m['accuracy'])}")
    print(f"  {'majority baseline':<22} {_fmt(m['majority_baseline'])}   <- beat this")
    print(f"\n  {'sensitivity (malig)':<22} {_fmt(m['sensitivity'])}")
    print(f"  {'specificity (benign)':<22} {_fmt(m['specificity'])}")
    print(f"  {'precision':<22} {_fmt(m['precision'])}")
    print(f"  {'F1':<22} {_fmt(m['f1'])}")
    print(f"\n  confusion:  TP {c['tp']:4d}   FN {c['fn']:4d}")
    print(f"              FP {c['fp']:4d}   TN {c['tn']:4d}")
    print(f"\n  mean inference {m['mean_time_s']:.1f}s/sample")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score zero-shot BreastMNIST runs")
    parser.add_argument("--file", default=None, help="Single .jsonl to score")
    args = parser.parse_args()

    files = [Path(args.file)] if args.file else sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No result files in {RESULTS_DIR}")

    summary: dict[str, dict] = {}
    for path in files:
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        if not rows:
            continue
        m = score(rows)
        summary[path.stem] = m
        report(path.stem, m)

    if len(summary) > 1:
        print(f"\n{'='*66}\nCOMPARISON\n{'='*66}")
        print(f"  {'run':<28} {'bal_acc':>8} {'AUC':>8} {'MCC':>8} {'sens':>8}")
        for name, m in summary.items():
            print(f"  {name:<28} {_fmt(m['balanced_accuracy']):>8} {_fmt(m['auc']):>8} "
                  f"{_fmt(m['mcc']):>8} {_fmt(m['sensitivity']):>8}")

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
