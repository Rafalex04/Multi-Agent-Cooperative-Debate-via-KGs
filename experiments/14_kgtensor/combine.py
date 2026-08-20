"""Do the debate and the BI-RADS logprob signal carry the SAME information?

The project's biggest measured lever has been ensembling, and the reason it works
is stated in the ledger: run-to-run correlation is only 0.308, so pooling
independent debates cancels variance. Every pooled run so far is the same
mechanism repeated -- two agents arguing over claims.

The BI-RADS two-turn run (experiments/13_birads2) is a genuinely different
mechanism: no claims, no rounds, no rebuttals, just the first-token digit
distribution over an assessment category, two calls per image. It scores 0.7598
on its own. If it is weakly correlated with the debate score, combining them
should beat either, and for the usual reason rather than a new one.

Everything is z-scored on TRAIN statistics only, and the combination weights are
fit on train, so val and test stay untouched.

Usage:
  python combine.py
"""
from __future__ import annotations

import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from claims import SPLITS, auc, bacc, by_split, kg_findings, load     # noqa: E402
from kg_tensor import (COLS, evidence_score, featurise, fit_evidence, kg_centre,  # noqa
                       mal_share, pool_by_group, predict, train_head)


def tensor_scores(by, findings, keys, prior_on=True, collapse=False, aug=True, l2=1e-3):
    """KG-indexed evidence tensor as a standalone component (no anchor).

    Trained on train only, then applied unchanged to val and test, so it can sit
    beside the other components without leaking.
    """
    F = len(findings)
    prior = np.array([s for _, s in findings], dtype=float)
    Xtr, ytr, gtr = featurise(by["train"], F, COLS, aug, collapse)
    w0 = kg_centre(COLS, collapse, prior_on, F, prior)
    w, b, be = train_head(Xtr, np.zeros(len(ytr)), ytr, w0, l2, 1500)
    out = {}
    for s in SPLITS:
        X, y, g = featurise(by[s], F, COLS, aug, collapse)
        sc = predict(X, np.zeros(len(y)), w, b, be)
        full = pool_by_group(sc, g, len(by[s])) if aug else sc
        idx = {int(r["sid"]): i for i, r in enumerate(by[s])}
        out[s] = np.array([full[idx[k]] for k in keys[s]])
    return out

_D = _HERE.parents[2] / "breastMnist/data/breast"


def birads_scores():
    """(split, sample_id) -> mean of the two turns' first-token digit expectation."""
    out = {}
    for f in glob.glob(str(_HERE.parents[1] / "13_birads2/results/birads2_*.jsonl")):
        split = Path(f).name.split("_")[1]
        for ln in Path(f).read_text().splitlines():
            if not ln.strip():
                continue
            d = json.loads(ln)
            if d.get("exp_mal") is None or d.get("exp_ben") is None:
                continue
            out[(split, int(d["index"]))] = (d["exp_mal"] + d["exp_ben"]) / 2
    return out


def zfit(v):
    v = np.asarray(v, dtype=float)
    m, s = float(v.mean()), float(v.std())
    return m, (s if s > 1e-9 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings()
    data = load(runs, findings)
    by = by_split(data)
    ev_w = fit_evidence(by["train"])
    bir = birads_scores()

    # ---- per-sample component scores, aligned by (split, int sample id) ----
    comp = {s: defaultdict(list) for s in SPLITS}
    Y = {s: [] for s in SPLITS}
    keys = {s: [] for s in SPLITS}
    for s in SPLITS:
        for r in by[s]:
            sid = int(r["sid"])
            b = bir.get((s, sid))
            if b is None:
                continue
            comp[s]["evidence"].append(evidence_score(r["claims"], ev_w))
            comp[s]["mal_share"].append(mal_share(r["claims"]))
            comp[s]["birads"].append(b)
            Y[s].append(r["y"]); keys[s].append(sid)
        Y[s] = np.array(Y[s], dtype=float)
    print("aligned samples: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    tens = tensor_scores(by, findings, keys)
    for s in SPLITS:
        comp[s]["tensor"] = list(tens[s])
    names = ["evidence", "mal_share", "birads", "tensor"]
    Z = {}
    for n in names:
        m, sd = zfit(comp["train"][n])
        Z[n] = {s: (np.array(comp[s][n]) - m) / sd for s in SPLITS}

    def rep(name, sc):
        a = {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS}
        sf = np.concatenate([sc["train"], sc["val"]])
        yf = np.concatenate([Y["train"], Y["val"]])
        t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
        b = bacc(sc["test"].tolist(), Y["test"].tolist(), t)
        print(f"  {name:34s} train {a['train']:.4f}  val {a['val']:.4f}  "
              f"TEST {a['test']:.4f}   bAcc {b:.4f}")
        return a["test"], b

    print("\n=== correlation on train (Pearson) ===")
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            r = float(np.corrcoef(Z[a]["train"], Z[b_]["train"])[0, 1])
            print(f"  {a:10s} vs {b_:10s}  r = {r:+.3f}")

    print("\n  readout                              train      val      TEST     bAcc")
    res = {}
    for n in names:
        res[n] = rep(n + " alone", Z[n])
    res["ev+bir"] = rep("evidence + birads (equal)",
                        {s: (Z["evidence"][s] + Z["birads"][s]) / 2 for s in SPLITS})
    res["ms+bir"] = rep("mal_share + birads (equal)",
                        {s: (Z["mal_share"][s] + Z["birads"][s]) / 2 for s in SPLITS})
    res["ev+bir+tens"] = rep("evidence + birads + KG tensor",
                             {s: (Z["evidence"][s] + Z["birads"][s] + Z["tensor"][s]) / 3
                              for s in SPLITS})
    res["bir+tens"] = rep("birads + KG tensor",
                          {s: (Z["birads"][s] + Z["tensor"][s]) / 2 for s in SPLITS})
    res["all4"] = rep("all four (equal)",
                      {s: (Z["evidence"][s] + Z["mal_share"][s] + Z["birads"][s]
                           + Z["tensor"][s]) / 4 for s in SPLITS})

    # weight fitted on train only, swept coarsely
    best = None
    for wgt in np.arange(0.0, 1.01, 0.05):
        sc = {s: wgt * Z["evidence"][s] + (1 - wgt) * Z["birads"][s] for s in SPLITS}
        a = auc(list(sc["train"][Y["train"] == 1]), list(sc["train"][Y["train"] == 0]))
        if best is None or a > best[0]:
            best = (a, wgt)
    wgt = best[1]
    res["ev+bir_fit"] = rep(f"evidence + birads (w={wgt:.2f} on train)",
                            {s: wgt * Z["evidence"][s] + (1 - wgt) * Z["birads"][s]
                             for s in SPLITS})

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: {"test": v[0], "bacc": v[1]} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
