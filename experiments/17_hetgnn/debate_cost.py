"""What does adding the debate cost, on identical samples with identical fitting?

Every comparison so far has moved two things at once -- architecture and feature
set -- so "the GNN did not help" and "the debate did not help" were entangled.
This isolates the debate. Same samples, same standardisation, same fitter, same
CV-selected L2. Only the columns change.

  probes           16 findings x 2 phrasings                        32 features
  debate           per-finding argument mass and net stance         32 features
  probes + debate  both                                             64 features

A permutation control is included because 32 extra columns of anything will move
a fitted score on 546 training samples: the debate block is shuffled ACROSS
samples 20 times, which destroys its relationship to the label while preserving
its dimensionality, scale and correlation structure. If real debate features do
no better than shuffled debate features, the channel is decoration.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
from claims import auc, bacc                                          # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                     # noqa: E402
from hetgraph import SPLITS, build                                    # noqa: E402
from thothgnn3 import node_tensor, probe_block                        # noqa: E402


def best_fit(Ztr, ytr):
    best = None
    for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
        for l2 in L2S:
            c = cv_score(Ztr, ytr, l2, f)[0]
            if best is None or c > best[0]:
                best = (c, l2, nm, f)
    return best


def evaluate(Z, Y, label, out):
    cv, l2, nm, f = best_fit(Z["train"], Y["train"])
    w, b = f(Z["train"], Y["train"], l2)
    s = {k: Z[k] @ w + b for k in SPLITS}
    a = {k: auc(list(s[k][Y[k] == 1]), list(s[k][Y[k] == 0])) for k in SPLITS}
    fv = np.concatenate([s["train"], s["val"]]); gv = np.concatenate([Y["train"], Y["val"]])
    t = max(sorted(set(fv.tolist())), key=lambda t: bacc(fv.tolist(), gv.tolist(), t))
    bb = bacc(s["test"].tolist(), Y["test"].tolist(), t)
    print(f"  {label:28s} {Z['train'].shape[1]:3d}f  cv {cv:.4f}  "
          f"TEST {a['test']:.4f}  bAcc {bb:.4f}")
    out[label] = {"cv": cv, "auc": a, "bacc": bb, "n_features": int(Z["train"].shape[1])}
    return a["test"]


def main():
    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    probes = probe_block(names)
    X = {s: node_tensor(d[s], names, prior, probes, s) for s in SPLITS}
    Y = {s: d[s]["y"] for s in SPLITS}
    n = {s: len(Y[s]) for s in SPLITS}
    print(f"identical samples throughout: {n}\n")

    def std(blocks):
        A = {s: np.concatenate([X[s][:, :, c] for c in blocks], 1) for s in SPLITS}
        mu, sd = A["train"].mean(0), A["train"].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        return {s: (A[s] - mu) / sd for s in SPLITS}

    out = {}
    a_p = evaluate(std([0, 1]), Y, "probes only", out)
    a_d = evaluate(std([2, 3]), Y, "debate only", out)
    a_b = evaluate(std([0, 1, 2, 3]), Y, "probes + debate", out)
    print(f"\n  debate contribution on top of probes: {a_b - a_p:+.4f} AUC")

    print("\n  permutation control: debate block shuffled across samples (20 draws)")
    Zp = std([0, 1]); Zd = std([2, 3])
    _, l2_sel, _, f_sel = best_fit(np.hstack([Zp["train"], Zd["train"]]), Y["train"])
    deltas = []
    for r in range(20):
        rng = np.random.default_rng(900 + r)
        Zs = {s: np.hstack([Zp[s], Zd[s][rng.permutation(len(Y[s]))]]) for s in SPLITS}
        w, b = f_sel(Zs["train"], Y["train"], l2_sel)
        sc = Zs["test"] @ w + b
        deltas.append(auc(list(sc[Y["test"] == 1]), list(sc[Y["test"] == 0])) - a_p)
    m, sd_ = float(np.mean(deltas)), float(np.std(deltas))
    print(f"  shuffled-debate contribution: {m:+.4f} +- {sd_:.4f}")
    print(f"  real minus shuffled:          {(a_b - a_p) - m:+.4f} "
          f"({((a_b - a_p) - m) / (sd_ + 1e-9):+.2f} sd)")
    print("\n  -> " + ("the debate carries something beyond its dimensionality"
                       if ((a_b - a_p) - m) / (sd_ + 1e-9) > 2 else
                       "the debate channel is indistinguishable from noise of the "
                       "same shape once probes are present"))
    out["permutation"] = {"shuffled_delta_mean": m, "shuffled_delta_sd": sd_,
                          "real_delta": a_b - a_p}
    (_HERE.parent / "results/debate_cost.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
