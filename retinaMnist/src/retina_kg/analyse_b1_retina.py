"""B1 Catfish on retina: analysis and ablations. Our implementation of the paper.

Readout is the MEAN CLAIM GRADE over the final claim set -- the ordinal analogue
of breast's `mal_share`, and identical in spirit: no learned head anywhere in B1,
so only the protocol varies between arms.

Two label-space changes from `26_baselines/analyse_b1.py`:
  * score is a mean ICDR grade, not a malignant fraction;
  * a grade prediction needs four cut points. They are set by matching the TRAIN
    grade distribution's quantiles (train labels only, never val or test), then
    applied unchanged, so QWK/ACC are reported without a test-fitted threshold.

Referable-DR AUC is the headline because it is the one number directly comparable
to the breast B1 AUC and to ThothGNN v3's retina arm.

tone and tau_conf are selected by 5-fold CV on the 1080 TRAINING records. ONE
test evaluation, after the configuration is frozen and printed.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "26_baselines")
import metrics_ordinal as M                                          # noqa: E402
from corpus_io import load_corpus                                    # noqa: E402

TONES = ("collaborative", "adversarial")
NC = 5


def mean_grade(grades):
    g = [x for x in grades if x is not None]
    return float(np.mean(g)) if g else 1.5      # neutral prior when nothing parsed


def score_base(d):
    return mean_grade([c.get("grade") for c in d["base_claims"]])


def score_branch(d, tone):
    b = d["branches"][tone]
    mod = b.get("moderator") or {}
    if mod:
        return mean_grade(list(mod.values()))
    allc = d["base_claims"] + b["catfish"] + b["response"]
    return mean_grade([c.get("grade") for c in allc])


def fires(d, tau):
    t = d["trigger"]
    if t["silent_agreement"]:
        return True
    c = t.get("mean_conf")
    return c is not None and tau is not None and c < tau


def scores(recs, mode, tone, tau):
    out = []
    for d in recs:
        if mode == "base":
            out.append(score_base(d))
        elif mode == "always":
            out.append(score_branch(d, tone))
        else:
            out.append(score_branch(d, tone) if fires(d, tau) else score_base(d))
    return np.array(out)


def ys(recs):
    return np.array([d["gold_grade"] for d in recs], dtype=int)


def cutpoints(train_scores, train_y):
    """Four thresholds matching the TRAIN grade distribution. Train labels only."""
    q = np.cumsum(np.bincount(train_y, minlength=NC)[:-1]) / len(train_y)
    return np.quantile(train_scores, q)


def to_grade(s, cuts):
    return (np.asarray(s)[:, None] > np.asarray(cuts)[None, :]).sum(1)


def ref_auc(y, s):
    """Fast midrank AUC -- the bootstrap below calls this 10,000 times."""
    return M.fast_auc(s, M.referable(y) == 1)


def cv_ref_auc(recs, mode, tone, tau, k=5, seed=0):
    y, s = ys(recs), scores(recs, mode, tone, tau)
    rng = np.random.default_rng(seed); o = rng.permutation(len(y))
    vals = []
    for i in range(k):
        te = np.isin(np.arange(len(y)), o[i::k])
        if len(np.unique(M.referable(y[te]))) < 2:
            continue
        vals.append(ref_auc(y[te], s[te]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(P.PACK / "catfish_b1"))
    ap.add_argument("--out", default=str(P.RESULTS / "b1_retina.json"))
    a = ap.parse_args()

    data = load_corpus(a.corpus)
    tr, va, te = data["train"], data["val"], data["test"]
    print(f"B1 retina corpus: train {len(tr)}  val {len(va)}  test {len(te)}")
    allc = [c for d in tr for c in d["base_claims"]]
    nocat = sum(1 for d in tr for t in TONES if not d["branches"][t]["catfish"])
    print(f"  base claims/image {len(allc)/max(1,len(tr)):.1f}   "
          f"branches with no parsed catfish {nocat}/{2*len(tr)} "
          f"({nocat/max(1,2*len(tr)):.1%})")

    confs = [d["trigger"]["mean_conf"] for d in tr
             if d["trigger"]["mean_conf"] is not None]
    grid = [None] + list(np.quantile(confs, [0.1, 0.25, 0.5, 0.75, 0.9]))

    print("\nSELECTION (5-fold CV on train only; test untouched)")
    print(f"{'tone':16s} {'tau_conf':>10s} {'CV refAUC':>10s} {'trigger rate':>13s}")
    best = None
    for tone in TONES:
        for tau in grid:
            v = cv_ref_auc(tr, "gated", tone, tau)
            rate = float(np.mean([fires(d, tau) for d in tr]))
            print(f"{tone:16s} {'none' if tau is None else f'{tau:.4f}':>10s} "
                  f"{v:10.4f} {rate:13.4f}")
            if best is None or v > best[0]:
                best = (v, tone, tau, rate)
    cv_best, tone_s, tau_s, rate_s = best
    print(f"\nFROZEN: tone={tone_s}  tau_conf="
          f"{'none' if tau_s is None else f'{tau_s:.4f}'}  CV refAUC {cv_best:.4f}  "
          f"trigger rate {rate_s:.4f}")
    if rate_s > 0.9 or rate_s < 0.1:
        print("  NOTE: gate fires on >90% or <10% of cases -- close to inert.")

    cuts = cutpoints(scores(tr, "gated", tone_s, tau_s), ys(tr))
    print(f"  cut points from the TRAIN grade distribution: "
          f"{np.round(cuts, 3).tolist()}")

    y = ys(te)
    arms = {
        "B1-full": ("gated", tone_s, tau_s),
        "B1-no-catfish": ("base", tone_s, None),
        "B1-always-on": ("always", tone_s, None),
        "B1-adversarial": ("gated", "adversarial", tau_s),
        "B1-collaborative": ("gated", "collaborative", tau_s),
    }
    res, S = {}, {}
    print(f"\nTEST (n={len(te)}, one evaluation)")
    print(f"{'arm':20s} {'refAUC':>8s} {'QWK':>8s} {'ACC':>7s} {'macroR':>8s} {'95% CI (refAUC)':>20s}")
    rng = np.random.default_rng(0)
    for lab, (mode, tone, tau) in arms.items():
        s = scores(te, mode, tone, tau); S[lab] = s
        g = to_grade(s, cuts)
        ra, qw = ref_auc(y, s), M.qwk(y, g, NC)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(M.referable(y[i]))) < 2:
                continue
            bs.append(ref_auc(y[i], s[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[lab] = {"referable_auc": ra, "qwk": qw,
                    "acc": float((g == y).mean()),
                    "macro_recall": M.macro_recall(y, g, NC),
                    "ci": [float(lo), float(hi)]}
        print(f"{lab:20s} {ra:8.4f} {qw:+8.4f} {res[lab]['acc']:7.4f} "
              f"{res[lab]['macro_recall']:8.4f}  [{lo:.4f}, {hi:.4f}]")

    # paired against ThothGNN v3 head A, same test images
    try:
        of = json.loads((P.RESULTS / "ordinal_full.json").read_text())
        v3 = of["A: GNN (real KG)"]["referable_auc"]
        d = res["B1-full"]["referable_auc"] - v3
        print(f"\nB1-full minus ThothGNN v3 (head A, {v3:.4f}): {d:+.4f}")
        res["_vs_thothgnn3_A"] = {"v3_referable_auc": v3, "delta": d}
    except Exception as exc:
        print(f"  (v3 comparison unavailable: {exc})")

    res["_config"] = {"tone": tone_s, "tau_conf": tau_s, "cv_ref_auc": cv_best,
                      "trigger_rate": rate_s, "cutpoints": cuts.tolist(),
                      "n": {"train": len(tr), "val": len(va), "test": len(te)}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
