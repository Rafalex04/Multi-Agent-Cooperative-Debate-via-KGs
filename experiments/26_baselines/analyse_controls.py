"""Controls: self-consistency and independent ensemble, against B1 and ours.

Neither control involves any interaction between agents. If the debate arms do
not beat them, debate buys nothing over repeated sampling - which is the single
most important thing these baselines can establish, in either direction.

Readout is `mal_share`, identical to every debate arm. Self-consistency
aggregates it as the mean over N draws (the continuous analogue of majority
vote, which a ranking metric needs); the majority-vote verdict is also reported
as an accuracy for completeness.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc, bacc, paired_bootstrap        # noqa: E402
from corpus_io import load_corpus                        # noqa: E402

D = _HERE.parents[1] / "breastMnist/data/breast"


def mal_share(labels):
    labels = [l for l in labels if l in ("MALIGNANT", "BENIGN")]
    return sum(1 for l in labels if l == "MALIGNANT") / len(labels) if labels else 0.5


def load_draws(split):
    recs = []
    for d in load_corpus(D / "controls_b1").get(split, []):
        recs.append((d["sample_id"],
                     1.0 if d["gold_label"] == "MALIGNANT" else 0.0,
                     [mal_share([c.get("label") for c in dr]) for dr in d["draws"]]))
    return recs


def b1_scores(split, corpus="catfish_b1"):
    """B1-no-catfish (base rounds 0-1) and B1-full, keyed by sample id."""
    base, full = {}, {}
    b1 = json.loads((_HERE / "results/b1.json").read_text())
    tone, tau = b1["frozen"]["tone"], b1["frozen"]["tau_conf"]
    for d in load_corpus(D / corpus).get(split, []):
        sid = d["sample_id"]
        base[sid] = mal_share([c.get("label") for c in d["base_claims"]])
        br = d["branches"][tone]
        mod = br.get("moderator") or {}
        s = (mal_share(list(mod.values())) if mod else
             mal_share([c.get("label") for c in
                        d["base_claims"] + br["catfish"] + br["response"]]))
        t = d["trigger"]
        fired = t["silent_agreement"] or (t.get("mean_conf") is not None
                                          and tau is not None
                                          and t["mean_conf"] < tau)
        full[sid] = s if fired else base[sid]
    return base, full


def main():
    recs = load_draws("test")
    tr = load_draws("train") + load_draws("val")
    print(f"controls corpus: test {len(recs)}  train+val {len(tr)}")
    y = np.array([r[1] for r in recs])
    sid = [r[0] for r in recs]

    arms = {}
    arms["single draw (N=1)"] = np.array([r[2][0] for r in recs])
    arms["independent ensemble (N=2)"] = np.array([np.mean(r[2][:2]) for r in recs])
    arms["self-consistency (N=5)"] = np.array([np.mean(r[2]) for r in recs])
    base, full = b1_scores("test")
    arms["B1-no-catfish (ours)"] = np.array([base[s] for s in sid])
    arms["B1-full (Catfish)"] = np.array([full[s] for s in sid])

    # thresholds from train+val, applied unchanged
    thr = {}
    thr["single draw (N=1)"] = float(np.median([r[2][0] for r in tr]))
    thr["independent ensemble (N=2)"] = float(np.median([np.mean(r[2][:2]) for r in tr]))
    thr["self-consistency (N=5)"] = float(np.median([np.mean(r[2]) for r in tr]))
    tb, tf = b1_scores("train"); vb, vf = b1_scores("val")
    thr["B1-no-catfish (ours)"] = float(np.median(list(tb.values()) + list(vb.values())))
    thr["B1-full (Catfish)"] = float(np.median(list(tf.values()) + list(vf.values())))

    rng = np.random.default_rng(0)
    print(f"\nTEST n={len(y)}, malignant {int(y.sum())}")
    print(f"{'arm':30s} {'AUC':>8s} {'bAcc':>8s} {'95% CI':>18s}  VLM calls/img")
    calls = {"single draw (N=1)": 1, "independent ensemble (N=2)": 2,
             "self-consistency (N=5)": 5, "B1-no-catfish (ours)": 4,
             "B1-full (Catfish)": 9}
    res = {}
    for lab, s in arms.items():
        a, b = auc(y, s), bacc(y, s, thr[lab])
        bs = [auc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y))
                                        for _ in range(2000))]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res[lab] = {"auc": float(a), "bacc": float(b), "ci": [lo, hi],
                    "calls": calls[lab]}
        print(f"{lab:30s} {a:8.4f} {b:8.4f} [{lo:.4f},{hi:.4f}]  {calls[lab]:>5d}")

    # majority-vote accuracy for self-consistency, as the paper-style readout
    mv = np.array([np.mean([1.0 if x > 0.5 else 0.0 for x in r[2]]) > 0.5 for r in recs])
    print(f"\nself-consistency majority-vote accuracy: {(mv == (y == 1)).mean():.4f}")

    print("\nTHE DECIDING CONTRASTS (paired bootstrap, 4000)")
    for a, b in (("B1-full (Catfish)", "independent ensemble (N=2)"),
                 ("B1-full (Catfish)", "self-consistency (N=5)"),
                 ("B1-no-catfish (ours)", "independent ensemble (N=2)"),
                 ("B1-no-catfish (ours)", "self-consistency (N=5)")):
        p = paired_bootstrap(y, arms[a], arms[b], n=4000)
        print(f"  {a:28s} vs {b:28s}  delta {res[a]['auc']-res[b]['auc']:+.4f}  P {p:.3f}")
        res[f"{a} vs {b}"] = {"delta": res[a]["auc"] - res[b]["auc"], "P": float(p)}
    (_HERE / "results/controls.json").write_text(json.dumps(res, indent=1))
    print("\n-> results/controls.json")


if __name__ == "__main__":
    main()
