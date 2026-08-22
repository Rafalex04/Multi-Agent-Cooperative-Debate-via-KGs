"""S2 kill gate: does per-finding DEBATE evidence add anything to the probes?

The spec's kill criterion is permutation importance. Shuffle each debate column
within the training folds and re-run CV: if CV AUC is unchanged, the debate adds
nothing at the finding level and S2 reduces to S0.

Includes the cross-run reproducibility columns, which are the one debate signal
never tested at the finding level -- two independent runs flagging a finding is
different evidence from one run flagging it twice, and no pooled count expresses
that (run-to-run correlation is 0.308).
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, kg_findings, load, by_split            # noqa: E402
from features import COLS, build                                       # noqa: E402
from gates import L2S, cv_score, fit_ce                                # noqa: E402

_D = _HERE.parents[2] / "breastMnist/data/breast"


def crossrun(rows, F, nruns):
    """per finding: fraction of runs mentioning it, and stance spread across runs."""
    out = []
    for r in rows:
        per = defaultdict(lambda: defaultdict(list))
        for c in r["claims"]:
            if c["finding"] >= 0:
                per[c["finding"]][c["run"]].append(c["stance"])
        pres = np.zeros(F); spread = np.zeros(F)
        for f, runs in per.items():
            pres[f] = len(runs) / max(1, nruns)
            m = [np.mean(v) for v in runs.values()]
            spread[f] = float(np.std(m)) if len(m) > 1 else 0.0
        out.append(np.concatenate([pres, spread]))
    return np.array(out)


def main():
    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings(); F = len(findings)
    blocks = []
    for tag in ("probe", "probeneg"):
        rows, Xt, Y, _ = build(runs, tag=tag)
        blocks.append({s: Xt[s][:, :, 0] for s in SPLITS})
    rows_all, Xp, Y, _ = build(runs, tag="probe")

    dc = [COLS.index(c) for c in ("net", "presence", "mass", "endorsed", "disputed", "open")]
    P = {s: np.hstack([b[s] for b in blocks]) for s in SPLITS}
    Dbg = {s: Xp[s][:, :, dc].reshape(len(Y[s]), -1) for s in SPLITS}
    CR = {s: crossrun(rows_all[s], F, len(runs)) for s in SPLITS}

    def best_cv(X):
        return max(cv_score(X, Y["train"], l2, fit_ce)[0] for l2 in L2S)

    base = best_cv(P["train"])
    full = np.hstack([P["train"], Dbg["train"], CR["train"]])
    withd = best_cv(full)
    print(f"  probes only               CV {base:.4f}   ({P['train'].shape[1]} feats)")
    print(f"  + debate + cross-run      CV {withd:.4f}   ({full.shape[1]} feats)")
    print(f"  delta {withd-base:+.4f}\n")

    # permutation importance of the debate block, 5 shuffles
    rng = np.random.default_rng(0)
    drops = []
    nP = P["train"].shape[1]
    for k in range(5):
        Xs = full.copy()
        perm = rng.permutation(len(Y["train"]))
        Xs[:, nP:] = Xs[perm, nP:]
        drops.append(withd - best_cv(Xs))
    print(f"  permutation importance of the debate block: "
          f"{np.mean(drops):+.4f} +- {np.std(drops):.4f}")
    verdict = ("debate carries finding-level signal; build S2"
               if np.mean(drops) > 0.005 and withd > base
               else "debate adds nothing at the finding level; S2 KILLED -> reduces to S0")
    print(f"  -> {verdict}")
    Path("experiments/15_kggnn/results/s2_gate.json").write_text(json.dumps(
        {"cv_probes": base, "cv_with_debate": withd,
         "perm_importance": float(np.mean(drops)), "perm_sd": float(np.std(drops)),
         "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
