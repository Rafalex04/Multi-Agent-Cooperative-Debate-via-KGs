"""B1 - our implementation of the Catfish Agent. Analysis and ablations.

Readout is `mal_share` over the final claim set - IDENTICAL to ours, so only the
protocol varies. No learned head anywhere in B1.

tau_conf and tone are selected by 5-fold CV on the 546 BreastMNIST TRAINING
graphs. ONE test evaluation, after the configuration is frozen and printed.
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc, bacc, paired_bootstrap        # noqa: E402
from corpus_io import load_corpus                        # noqa: E402

SPLITS = ("train", "val", "test")
TONES = ("collaborative", "adversarial")


def load(root):
    """split -> list of per-sample records (jsonl or legacy per-image json)."""
    return load_corpus(root)


def mal_share(labels):
    labels = [l for l in labels if l in ("MALIGNANT", "BENIGN")]
    if not labels:
        return 0.5
    return sum(1 for l in labels if l == "MALIGNANT") / len(labels)


def score_base(d):
    return mal_share([c.get("label") for c in d["base_claims"]])


def score_branch(d, tone):
    """moderator's consolidated labels; falls back to raw claims if empty."""
    b = d["branches"][tone]
    mod = b.get("moderator") or {}
    if mod:
        return mal_share(list(mod.values()))
    allc = d["base_claims"] + b["catfish"] + b["response"]
    return mal_share([c.get("label") for c in allc])


def fires(d, tau):
    t = d["trigger"]
    if t["silent_agreement"]:
        return True
    c = t.get("mean_conf")
    return c is not None and tau is not None and c < tau


def scores(recs, mode, tone, tau):
    s = []
    for d in recs:
        if mode == "base":
            s.append(score_base(d))
        elif mode == "always":
            s.append(score_branch(d, tone))
        else:                                   # gated
            s.append(score_branch(d, tone) if fires(d, tau) else score_base(d))
    return np.array(s)


def ys(recs):
    return np.array([1.0 if d["gold_label"] == "MALIGNANT" else 0.0 for d in recs])


def cv_auc(recs, mode, tone, tau, k=5, seed=0):
    y = ys(recs); s = scores(recs, mode, tone, tau)
    rng = np.random.default_rng(seed); o = rng.permutation(len(y))
    vals = []
    for i in range(k):
        te = np.isin(np.arange(len(y)), o[i::k])
        if len(np.unique(y[te])) < 2:
            continue
        vals.append(auc(y[te], s[te]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--asym-corpus", default=None,
                    help="the --catfish-sees-image corpus, for B1-catfish-sees-image")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load(args.corpus)
    tr, te = data["train"], data["test"]
    print(f"B1 corpus: train {len(tr)}  val {len(data['val'])}  test {len(te)}")
    confs = [d["trigger"]["mean_conf"] for d in tr
             if d["trigger"]["mean_conf"] is not None]
    grid = [None] + list(np.quantile(confs, [0.1, 0.25, 0.5, 0.75, 0.9]))

    # ---- selection: tone x tau_conf, 5-fold CV on TRAIN only ----
    print("\nSELECTION (5-fold CV on the 546 training graphs, test untouched)")
    print(f"{'tone':16s} {'tau_conf':>10s} {'CV AUC':>8s} {'trigger rate':>13s}")
    best = None
    for tone in TONES:
        for tau in grid:
            a = cv_auc(tr, "gated", tone, tau)
            rate = float(np.mean([fires(d, tau) for d in tr]))
            tag = "none" if tau is None else f"{tau:.4f}"
            print(f"{tone:16s} {tag:>10s} {a:8.4f} {rate:13.4f}")
            if best is None or a > best[0]:
                best = (a, tone, tau, rate)
    cv_best, tone_s, tau_s, rate_s = best
    print(f"\nFROZEN: tone={tone_s}  tau_conf="
          f"{'none' if tau_s is None else f'{tau_s:.4f}'}  "
          f"CV AUC {cv_best:.4f}  trigger rate {rate_s:.4f}")
    if rate_s > 0.9 or rate_s < 0.1:
        print("  NOTE: gate fires on >90% or <10% of cases -- it is close to inert.")

    # ---- one test evaluation ----
    y = ys(te)
    thr_pool = np.concatenate([scores(tr, "gated", tone_s, tau_s),
                               scores(data["val"], "gated", tone_s, tau_s)])
    thr = float(np.median(thr_pool))          # fitted on train+val, applied unchanged

    arms = {
        "B1-full": ("gated", tone_s, tau_s),
        "B1-no-catfish": ("base", tone_s, None),
        "B1-always-on": ("always", tone_s, None),
        "B1-adversarial": ("gated", "adversarial", tau_s),
        "B1-collaborative": ("gated", "collaborative", tau_s),
    }
    res, S = {}, {}
    print(f"\nTEST (n={len(te)}, one evaluation, threshold {thr:.4f} from train+val)")
    print(f"{'arm':22s} {'AUC':>8s} {'bAcc':>8s} {'95% CI':>18s}")
    rng = np.random.default_rng(0)
    for lab, (mode, tone, tau) in arms.items():
        s = scores(te, mode, tone, tau); S[lab] = s
        a = auc(y, s); b = bacc(y, s, thr)
        bs = [auc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y))
                                        for _ in range(2000))]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res[lab] = {"auc": float(a), "bacc": float(b), "ci": [lo, hi]}
        print(f"{lab:22s} {a:8.4f} {b:8.4f} [{lo:.4f},{hi:.4f}]")

    if args.asym_corpus and Path(args.asym_corpus).exists():
        ad = load(args.asym_corpus)
        if ad["test"]:
            s = scores(ad["test"], "gated", tone_s, tau_s)
            ya = ys(ad["test"])
            res["B1-catfish-sees-image"] = {"auc": float(auc(ya, s)),
                                            "bacc": float(bacc(ya, s, thr))}
            print(f"{'B1-catfish-sees-image':22s} "
                  f"{res['B1-catfish-sees-image']['auc']:8.4f} "
                  f"{res['B1-catfish-sees-image']['bacc']:8.4f}")

    print("\nCONTRASTS (paired bootstrap on test)")
    for a, b in (("B1-full", "B1-no-catfish"), ("B1-full", "B1-always-on"),
                 ("B1-collaborative", "B1-adversarial")):
        p = paired_bootstrap(y, S[a], S[b], n=4000)
        d = res[a]["auc"] - res[b]["auc"]
        print(f"  {a:20s} vs {b:20s}  delta {d:+.4f}  P(better) {p:.3f}")
        res[f"{a}_vs_{b}"] = {"delta": d, "P": float(p)}

    res["frozen"] = {"tone": tone_s, "tau_conf": tau_s, "cv_auc": cv_best,
                     "trigger_rate": rate_s, "threshold": thr, "n_test": len(te)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
