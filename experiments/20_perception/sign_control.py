"""Controls for the sign correction. Three questions, in order of how badly a
positive result needs them answered.

1. PERMUTATION. Five of sixteen signs were flipped. If flipping five signs AT
   RANDOM lifts BUS-BRA about as often, the descriptor fit contributed nothing
   and the gain is just a lucky direction in a 2^16 space. This is the control
   that decides whether the result survives.

2. DROP vs FLIP. Within a wording group the probes share ground truth, so an
   inverted probe may be a broken question rather than a reversed one. Weighting
   it zero is the more conservative repair; if dropping does as well as flipping,
   the honest claim is "these probes are uninformative", not "these probes are
   backwards".

3. WHY BreastMNIST DISAGREES. Signs fitted on 224px lesion crops transfer to
   224px lesion crops and reverse on 28x28 whole frames. Either that is framing,
   or it is n=156 noise. The comparison has to be a RANK, not a gap: the maximum
   over 8192 sign vectors on 156 images clears +0.05 on noise alone, so "an
   oracle buys a lot here" says nothing. What does say something is where the
   descriptor signs SIT in each dataset's own 2^13 distribution. High on one and
   low on the other is a genuine reversal of preference; middling on the small
   one is just an underpowered read.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from analyse_e1 import auc, load_probes                                # noqa: E402
from claims import kg_findings                                        # noqa: E402
from freeze_models import breastmnist_matrix                          # noqa: E402
from sign_audit import descriptor_signs, zscore                       # noqa: E402

_DATA = _HERE.parents[1] / "data/external"


def auc_many(y, S):
    """AUC of every COLUMN of S at once. The exhaustive sweep is 2^13 sign
    vectors; scoring them one at a time re-averages 1064 cases 8192 times."""
    n, m = S.shape
    order = np.argsort(S, axis=0, kind="mergesort")
    ranks = np.empty_like(S, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[order, np.arange(m)] = (rows + 1.0)          # ties are vanishingly
    npos = y.sum(); nneg = n - npos                     # rare on continuous sums
    rpos = (ranks * y[:, None]).sum(0)
    return (rpos - npos * (npos + 1) / 2.0) / (npos * nneg)


def case_reduce(P, y, c):
    """Per-case mean of the standardised probes, so a sign vector is one matmul."""
    Z = zscore(P)
    u = np.unique(c)
    G = (c[None, :] == u[:, None]).astype(float)
    G /= G.sum(1, keepdims=True)
    return G @ Z, np.array([y[c == k][0] for k in u])


def busbra(names):
    probes = load_probes(names, ("busbra_p2", "busbra_p3"),
                         _HERE.parents[0] / "16_external/results")
    z = np.load(_DATA / "busbra_pad2_224.npz", allow_pickle=True)
    idx = sorted(probes)
    P = np.array([probes[i] for i in idx]).mean(2)
    y = (z["labels"][idx, 0] == 0).astype(float)
    ZC, ys = case_reduce(P, y, z["cases"][idx])
    return (lambda w: float(auc_many(ys, (ZC @ np.asarray(w).reshape(-1, 1))))), len(ys), ZC, ys


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    sign, table, dn, D, bidx, bPm = descriptor_signs(names)
    annotated = [j for j, f in enumerate(names) if f in table]
    flipped = [j for j in annotated if sign[j] < 0]

    case_auc, ncases, ZC, ys = busbra(names)
    base, corr = case_auc(prior), case_auc(prior * sign)
    print(f"BUS-BRA, {ncases} cases")
    print(f"  baseline                     {base:.4f}")
    print(f"  descriptor-fitted signs      {corr:.4f}   ({corr-base:+.4f})")

    # ---- 1. permutation over which 5 of the 13 annotated signs get flipped ----
    rng = np.random.default_rng(0)
    k = len(flipped)
    W = np.ones((len(names), 2000))
    for t in range(2000):
        W[rng.choice(annotated, k, replace=False), t] = -1.0
    draws = auc_many(ys, ZC @ (W * prior[:, None]))
    beat = float((draws >= corr).mean())
    print(f"\n  random {k}-of-{len(annotated)} flips: mean {draws.mean():.4f} "
          f"sd {draws.std():.4f}   max {draws.max():.4f}")
    print(f"  descriptor signs sit at {(corr-draws.mean())/draws.std():+.2f} sd; "
          f"{beat*100:.1f}% of random flip sets do as well or better")

    # exhaustive over all 2^13 annotated sign vectors: where does ours rank?
    M = np.arange(1 << len(annotated))
    A = np.ones((len(names), len(M)))
    for b, j in enumerate(annotated):
        A[j] = 1.0 - 2.0 * ((M >> b) & 1)
    allv = auc_many(ys, ZC @ (A * prior[:, None]))
    pct = float((allv < corr).mean())
    print(f"  exhaustive over all 2^{len(annotated)} sign vectors: ours is at the "
          f"{pct*100:.1f}th percentile (best possible {allv.max():.4f})")

    # ---- 2. drop the inverted probes instead of flipping them ------------------
    drop = prior.copy(); drop[flipped] = 0.0
    print(f"\n  DROP inverted instead of flipping  {case_auc(drop):.4f}   "
          f"({case_auc(drop)-base:+.4f})")

    # ---- 3. is BreastMNIST framing or noise? -----------------------------------
    X, Y = breastmnist_matrix(findings)
    F = len(names)
    Zt = zscore(np.stack([X["test"][:, :F], X["test"][:, F:]], 2).mean(2))
    yt = Y["test"]
    bm = lambda w: auc(yt, Zt @ w)
    oracle = float(auc_many(yt, Zt @ (A * prior[:, None])).max())
    bm_all = auc_many(yt, Zt @ (A * prior[:, None]))
    bm_pct = float((bm_all < bm(prior * sign)).mean())
    bmdrop = prior.copy(); bmdrop[flipped] = 0.0
    print(f"\nBreastMNIST test, {len(yt)} images (28x28 whole frames, not lesion crops)")
    print(f"  baseline {bm(prior):.4f}   descriptor signs {bm(prior*sign):.4f}   "
          f"drop-inverted {bm(bmdrop):.4f}   oracle {oracle:.4f}")
    print(f"  descriptor signs percentile in this dataset's 2^{len(annotated)} "
          f"distribution: {bm_pct*100:.1f}th   (BUS-BRA: {pct*100:.1f}th)")
    print(f"  -> {'the two datasets genuinely prefer different signs' if bm_pct < 0.5 else 'consistent ordering; the point estimate is just noisy'}")

    (_HERE / "results/sign_control.json").write_text(json.dumps(
        {"busbra_cases": ncases, "baseline": base, "corrected": corr,
         "drop_inverted": case_auc(drop), "n_flipped": k,
         "perm_mean": float(draws.mean()), "perm_sd": float(draws.std()),
         "perm_frac_at_least_as_good": beat,
         "exhaustive_percentile": pct, "exhaustive_best": float(allv.max()),
         "breastmnist": {"baseline": bm(prior), "corrected": bm(prior * sign),
                         "drop_inverted": bm(bmdrop), "oracle": oracle,
                         "percentile": bm_pct, "n": int(len(yt))}},
        indent=1, default=float))
    print("\nwrote results/sign_control.json")


if __name__ == "__main__":
    main()
