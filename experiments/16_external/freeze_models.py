"""Freeze the three externally-testable models on BreastMNIST, before scoring BUS-BRA.

Everything here is fitted on BreastMNIST TRAIN alone and then written to disk with
a hash. Nothing is refitted on external data in the primary analysis, and the bAcc
thresholds are swept on train+val and applied unchanged -- which is what deployment
actually means.

Only probe-based models can cross the boundary: BUS-BRA has no debate corpus, so a
debate-dependent model has nothing to consume there. The three are therefore

  kgsum   0 fitted parameters. score = sum_f prior_f * z_f, the KG's own sign
          vector applied to standardised probes. The purest statement that the
          ontology carries the decision.
  logit   free 32-weight logistic, L2 by CV on train.
  s5      per-finding calibration (a_f, c_f) penalised by the KG-signed Laplacian
          at the registered lambda=0.01, with a free readout above it.

One deviation from the BreastMNIST-internal runs, recorded rather than hidden: those
aligned samples on having BOTH a debate and a probe vector. A model that must run
where no debate exists cannot inherit that intersection, so these are fitted on the
probe-only alignment. The refit is what gets frozen and what gets reported in-domain,
so the in-domain and external numbers come from the identical object.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from claims import auc, bacc, kg_findings                             # noqa: E402
from features import probe_table                                      # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                      # noqa: E402
from kg_graph import build_adjacency                                   # noqa: E402
from s3_s5 import fit_calib                                            # noqa: E402

_KG = _HERE.parents[1] / "breastMnist/data/breast/knowledge_graph.json"
_PROBES = _HERE.parents[0] / "15_kggnn/results"
TAGS = ("probeneg", "probep3")          # registered primary set: P2 + P3
SPLITS = ("train", "val", "test")
LAMBDA = 0.01                            # registered, from the BreastMNIST CV


def breastmnist_matrix(findings):
    """(split -> X of shape n x 32, split -> y). Column block order is TAGS."""
    names = [f for f, _ in findings]
    tabs = {t: probe_table(_PROBES, t) for t in TAGS}
    gold = {}
    for t in TAGS:
        for f in sorted(Path(_PROBES).glob(f"{t}_*.jsonl")):
            split = f.name.split("_")[1]
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                gold[(split, int(r["index"]))] = 1 if r["gold"] == "MALIGNANT" else 0
    X, Y = {}, {}
    for s in SPLITS:
        keys = sorted(k for k in gold if k[0] == s and all(k in tabs[t] for t in TAGS))
        X[s] = np.array([[float(tabs[t][k].get(n) if tabs[t][k].get(n) is not None else 0.5)
                          for t in TAGS for n in names] for k in keys])
        Y[s] = np.array([gold[k] for k in keys], dtype=float)
    return X, Y


def sweep_threshold(sc_tr, y_tr, sc_va, y_va):
    """bAcc threshold swept on train+val only, then frozen."""
    f = np.concatenate([sc_tr, sc_va]); g = np.concatenate([y_tr, y_va])
    return float(max(sorted(set(f.tolist())), key=lambda t: bacc(f.tolist(), g.tolist(), t)))


def main():
    findings = kg_findings()
    F = len(findings)
    prior = np.array([s for _, s in findings])
    X, Y = breastmnist_matrix(findings)
    print(f"probe-only alignment: " + "  ".join(f"{s} n={len(Y[s])}" for s in SPLITS))

    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    y = Y["train"]
    out = {"probe_tags": list(TAGS), "findings": [f for f, _ in findings],
           "prior": prior.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
           "n": {s: int(len(Y[s])) for s in SPLITS}, "models": {}}

    # ---- (a) kgsum: zero fitted parameters -------------------------------
    w_kg = np.tile(prior, len(TAGS)) / (F * len(TAGS))
    sc = {s: Z[s] @ w_kg for s in SPLITS}
    thr = sweep_threshold(sc["train"], y, sc["val"], Y["val"])
    out["models"]["kgsum"] = {
        "kind": "fixed", "w": w_kg.tolist(), "b": 0.0, "threshold": thr,
        "n_fitted_params": 0,
        "indomain": {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS},
        "indomain_bacc": bacc(sc["test"].tolist(), Y["test"].tolist(), thr)}

    # ---- (b) logit: free weights, L2 and loss by CV on train -------------
    best = None
    for name, fitter in (("ce", fit_ce), ("rank", fit_rank)):
        for l2 in L2S:
            cv = cv_score(Z["train"], y, l2, fitter)[0]
            if best is None or cv > best[0]:
                best = (cv, l2, name, fitter)
    cv, l2, lname, fitter = best
    w, b = fitter(Z["train"], y, l2)
    sc = {s: Z[s] @ w + b for s in SPLITS}
    thr = sweep_threshold(sc["train"], y, sc["val"], Y["val"])
    out["models"]["logit"] = {
        "kind": "logistic", "loss": lname, "l2": l2, "cv": cv,
        "w": w.tolist(), "b": float(b), "threshold": thr,
        "n_fitted_params": int(len(w) + 1),
        "indomain": {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS},
        "indomain_bacc": bacc(sc["test"].tolist(), Y["test"].tolist(), thr)}

    # ---- (c) s5: KG-signed Laplacian on the calibration layer ------------
    A, _ = build_adjacency(_KG, [f for f, _ in findings])
    same = (prior[:, None] * prior[None, :]) > 0
    As = A * same - A * (~same)
    Lap = np.diag(np.abs(As).sum(1)) - As
    bl2, bcv = None, None
    for l2 in (1e-2, 1e-1, 3e-1):
        s_cv = np.zeros(len(y))
        rng = np.random.default_rng(0)
        for fo in np.array_split(rng.permutation(len(y)), 5):
            m = np.zeros(len(y), bool); m[fo] = True
            a_, c_, w_, b_, idx = fit_calib(Z["train"][~m], y[~m], F, len(TAGS), Lap, LAMBDA, l2)
            s_cv[m] = (Z["train"][m] * a_[idx] + c_[idx]) @ w_ + b_
        v = auc(list(s_cv[y == 1]), list(s_cv[y == 0]))
        if bcv is None or v > bcv:
            bcv, bl2 = v, l2
    a_, c_, w_, b_, idx = fit_calib(Z["train"], y, F, len(TAGS), Lap, LAMBDA, bl2)
    sc = {s: (Z[s] * a_[idx] + c_[idx]) @ w_ + b_ for s in SPLITS}
    thr = sweep_threshold(sc["train"], y, sc["val"], Y["val"])
    out["models"]["s5"] = {
        "kind": "calib_laplacian", "lambda": LAMBDA, "l2": bl2, "cv": bcv,
        "a": a_.tolist(), "c": c_.tolist(), "w": w_.tolist(), "b": float(b_),
        "threshold": thr, "n_fitted_params": int(2 * F + len(w_) + 1),
        "indomain": {s: auc(list(sc[s][Y[s] == 1]), list(sc[s][Y[s] == 0])) for s in SPLITS},
        "indomain_bacc": bacc(sc["test"].tolist(), Y["test"].tolist(), thr)}

    p = _HERE / "results/frozen_models.json"
    p.write_text(json.dumps(out, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (_HERE / "results/frozen_models.sha256").write_text(h + "  frozen_models.json\n")

    print(f"\n{'model':8s} {'params':>7s} {'train':>7s} {'val':>7s} {'TEST':>7s} {'bAcc':>7s}")
    for k, m in out["models"].items():
        d = m["indomain"]
        print(f"{k:8s} {m['n_fitted_params']:7d} {d['train']:7.4f} {d['val']:7.4f} "
              f"{d['test']:7.4f} {m['indomain_bacc']:7.4f}")
    print(f"\nfrozen -> {p.name}\nsha256  {h}")


if __name__ == "__main__":
    main()
