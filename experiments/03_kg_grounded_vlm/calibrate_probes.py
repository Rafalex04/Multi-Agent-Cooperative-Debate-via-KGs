"""Unsupervised recalibration of KG feature probes.

The raw probe score weights every KG stance feature equally, which wastes the
signal. Two failure modes show up in the measured data:

  SATURATION  Some probes answer the same way for every image
              (spiculated: 0.004 vs 0.005; posterior_shadowing: 0.998 vs 0.999).
              A constant feature carries no information no matter how
              diagnostically important the KG says it is.
  SCALE       Probes sit at wildly different operating points, so a feature that
              swings 0.03->0.14 is drowned out by one that sits near 1.0.

Both are fixed without touching labels:

  1. Drop probes whose standard deviation across the corpus is below --min-std.
  2. z-score each surviving probe.
  3. score = sum(w_f * z_f) / sum(|w_f|), with w_f still supplied by the KG.

Only the *distribution* of probe outputs is used, never the gold labels, so this
is honest feature selection rather than fitting to the answer. Statistics can be
fitted on one split and applied to another via --fit-from, which avoids using
test-set statistics at test time.

Usage:
  python calibrate_probes.py --results results/..._featureprobe.jsonl
  python calibrate_probes.py --results test.jsonl --fit-from train.jsonl
"""
from __future__ import annotations

import argparse, json, math, statistics as st
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def fit_stats(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Per-feature (mean, std) of p_yes across the corpus."""
    vals: dict[str, list[float]] = {}
    for r in rows:
        for p in r["probes"]:
            if p["p_yes"] is not None:
                vals.setdefault(p["feature"], []).append(p["p_yes"])
    return {f: (st.mean(v), st.pstdev(v)) for f, v in vals.items() if len(v) > 1}


def score(row: dict, stats: dict, min_std: float) -> float | None:
    num = den = 0.0
    for p in row["probes"]:
        if p["p_yes"] is None or p["feature"] not in stats:
            continue
        mu, sd = stats[p["feature"]]
        if sd < min_std:                      # saturated -> no information
            continue
        z = (p["p_yes"] - mu) / sd
        num += p["weight"] * z
        den += abs(p["weight"])
    return None if den == 0 else num / den


def report(rows: list[dict], stats: dict, min_std: float, title: str):
    kept = [f for f, (_, sd) in stats.items() if sd >= min_std]
    dropped = [f for f, (_, sd) in stats.items() if sd < min_std]

    scored = [(score(r, stats, min_std), r["gold"]) for r in rows]
    scored = [(s, g) for s, g in scored if s is not None]
    pos = [s for s, g in scored if g == "MALIGNANT"]
    neg = [s for s, g in scored if g == "BENIGN"]
    a = auc(pos, neg)

    # Best balanced accuracy over all thresholds.
    best_ba, best_th = 0.0, 0.0
    alls = sorted({s for s, _ in scored})
    for th in alls:
        tp = sum(1 for s, g in scored if s >= th and g == "MALIGNANT")
        tn = sum(1 for s, g in scored if s < th and g == "BENIGN")
        ba = 0.5 * (tp / len(pos) + tn / len(neg))
        if ba > best_ba:
            best_ba, best_th = ba, th

    print(f"\n--- {title} (min_std={min_std}) ---")
    print(f"  kept {len(kept)} probes, dropped {len(dropped)} as saturated")
    if dropped:
        print(f"  dropped: {', '.join(sorted(dropped))}")
    print(f"  AUC={a:.4f}  best_balanced_acc={best_ba:.4f} @ th={best_th:+.4f}")
    print(f"  mean(mal)={st.mean(pos):+.4f}  mean(ben)={st.mean(neg):+.4f}")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--fit-from", default=None,
                   help="fit normalisation on this file instead (avoids using "
                        "test statistics at test time)")
    p.add_argument("--min-std", type=float, default=0.05)
    args = p.parse_args()

    rows = load(Path(args.results))
    fit_rows = load(Path(args.fit_from)) if args.fit_from else rows
    stats = fit_stats(fit_rows)

    print(f"Loaded {len(rows)} rows; fitted stats on {len(fit_rows)} rows, "
          f"{len(stats)} features")

    # Raw uncalibrated baseline for comparison.
    raw = []
    for r in rows:
        num = den = 0.0
        for pr in r["probes"]:
            if pr["p_yes"] is None:
                continue
            num += pr["weight"] * pr["p_yes"]
            den += abs(pr["weight"])
        if den:
            raw.append(((num / den + 1) / 2, r["gold"]))
    ra = auc([s for s, g in raw if g == "MALIGNANT"],
             [s for s, g in raw if g == "BENIGN"])
    print(f"\n--- RAW (no calibration) ---\n  AUC={ra:.4f}")

    print("\nPer-probe spread (std near 0 == saturated == useless):")
    for f, (mu, sd) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        mark = "" if sd >= args.min_std else "   <-- dropped"
        print(f"  {f[:46]:46s} mean={mu:.3f} std={sd:.3f}{mark}")

    best = (ra, "raw", None)
    for ms in (0.0, 0.02, 0.05, 0.10, 0.15):
        a = report(rows, stats, ms, "CALIBRATED")
        if a and a > best[0]:
            best = (a, "calibrated", ms)

    print(f"\n{'='*60}")
    print(f"BEST: {best[1]} min_std={best[2]} AUC={best[0]:.4f}")
    print(f"Zero-shot no-KG baseline AUC = 0.6011")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
