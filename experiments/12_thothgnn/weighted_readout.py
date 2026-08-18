"""Evidence-weighted stance count, with per-claim weights estimated on train.

diagnose.py found the structure mal_share discards. A claim asserted in
AGREEMENT predicts the image far better than one asserted in disagreement:

    P(mal | claim=MAL)   AGREE 0.408   DISAGREE 0.263   overall 0.299
    P(ben | claim=BEN)   AGREE 0.836   DISAGREE 0.735   overall 0.769

mal_share weights every claim identically, so that spread is thrown away. This
replaces the uniform count with a log-odds weight per claim, estimated from the
TRAINING split only:

    e_c = logit P(gold = MAL | stance_c, group_c) - logit P(gold = MAL)

and scores a graph by the mean of e_c over its claims. The mean rather than the
sum because claims inside a debate are heavily correlated -- summing lets a
verbose debate outvote a decisive one, which is the failure mode a raw
log-likelihood sum would have.

Groups are formed automatically from fields the schema already carries
(verdict, round, challenged), never from a hand-chosen list, and any group too
rare in train to estimate falls back to the global rate.

Usage:
  python weighted_readout.py --debates .../debates_v5q
  python weighted_readout.py --debates .../debates_v5q .../debates_v5q_r1
"""
from __future__ import annotations

import argparse, glob, json, math
from collections import defaultdict

SPLITS = ("train", "val", "test")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def bacc(s, y, t):
    tp = sum(1 for a, b in zip(s, y) if b == 1 and a >= t)
    fn = sum(1 for a, b in zip(s, y) if b == 1 and a < t)
    tn = sum(1 for a, b in zip(s, y) if b == 0 and a < t)
    fp = sum(1 for a, b in zip(s, y) if b == 0 and a >= t)
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def load(roots):
    """sample -> (split, gold, [claims]) pooled across runs."""
    out = {}
    for root in roots:
        for f in sorted(glob.glob(f"{root}/*/debate_*.json")):
            d = json.loads(open(f).read())
            if not d["claims"]:
                continue
            sp = f.split("/")[-2]
            key = (sp, d["sample_id"])
            challenged = {t for c in d["claims"] for t in (c.get("addressed_ids") or [])
                          if c.get("stance_verdict") == "DISAGREE"}
            rec = out.setdefault(key, (sp, 1 if d["gold_label"] == "MALIGNANT" else 0, []))
            for c in d["claims"]:
                if c.get("label") in ("MALIGNANT", "BENIGN"):
                    rec[2].append({
                        "stance": 1 if c["label"] == "MALIGNANT" else 0,
                        "verdict": c.get("stance_verdict") or "OPEN",
                        "round": min(c.get("round_idx", 0), 2),
                        "challenged": c["node_id"] in challenged,
                    })
    return out


def group_of(c, mode):
    if mode == "verdict":
        return c["verdict"]
    if mode == "verdict+round":
        return f"{c['verdict']}|r{c['round']}"
    if mode == "verdict+chal":
        return f"{c['verdict']}|{'C' if c['challenged'] else 'U'}"
    return "all"


def fit(train, mode, alpha=20.0):
    """P(gold=MAL | stance, group) with a pull toward the global rate."""
    tot = defaultdict(lambda: [0.0, 0.0])
    glob_ = [0.0, 0.0]
    for _, y, claims in train:
        for c in claims:
            k = (c["stance"], group_of(c, mode))
            tot[k][0] += y; tot[k][1] += 1
            glob_[0] += y; glob_[1] += 1
    prior = glob_[0] / max(1.0, glob_[1])
    lp = math.log(prior / (1 - prior))
    w = {}
    for k, (s, n) in tot.items():
        p = (s + alpha * prior) / (n + alpha)
        p = min(max(p, 1e-4), 1 - 1e-4)
        w[k] = math.log(p / (1 - p)) - lp
    return w, prior, lp


def score(claims, w, mode, default=0.0):
    v = [w.get((c["stance"], group_of(c, mode)), default) for c in claims]
    return sum(v) / len(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    args = ap.parse_args()

    data = load(args.debates)
    by = {s: [v for v in data.values() if v[0] == s] for s in SPLITS}
    print(f"pooled {len(args.debates)} run(s): "
          + "  ".join(f"{s} {len(by[s])}" for s in SPLITS))

    def report(name, fn):
        row = {}
        for s in SPLITS:
            sc = [fn(c) for _, _, c in by[s]]
            y = [y for _, y, _ in by[s]]
            row[s] = (auc([a for a, b in zip(sc, y) if b], [a for a, b in zip(sc, y) if not b]),
                      sc, y)
        fs = row["train"][1] + row["val"][1]
        fy = row["train"][2] + row["val"][2]
        thr = max(sorted(set(fs)), key=lambda t: bacc(fs, fy, t))
        ts, ty = row["test"][1], row["test"][2]
        print(f"  {name:24s} train {row['train'][0]:.4f}  val {row['val'][0]:.4f}  "
              f"TEST {row['test'][0]:.4f}   bAcc {bacc(ts, ty, thr):.4f}")
        return row["test"][0]

    print("\n  readout                    train      val      TEST      bAcc")
    report("mal_share (baseline)",
           lambda c: sum(x["stance"] for x in c) / len(c) if c else 0.5)
    for mode in ("verdict", "verdict+round", "verdict+chal"):
        w, prior, lp = fit(by["train"], mode)
        report(f"evidence-weighted [{mode}]", lambda c, w=w, m=mode: score(c, w, m))
        if mode == "verdict":
            print("       learned weights (log-odds vs prior):")
            for k in sorted(w, key=lambda k: (k[0], k[1])):
                print(f"         stance={'MAL' if k[0] else 'BEN'} {k[1]:9s} {w[k]:+.3f}")


if __name__ == "__main__":
    main()
