"""Section 4.1 end-to-end: cross-run reproducibility on top of the best anchor.

gate_crossrun.py showed +0.0076 CV on train. CV gains that size have not always
survived to test in this project, so this measures it where it counts: as an
additive head over the 4-run evidence-weighted anchor, trained on train only.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims import SPLITS, auc, bacc, by_split, kg_findings, load     # noqa: E402
from gate import base_feats                                           # noqa: E402
from gate_crossrun import crossrun_feats                              # noqa: E402
from kg_tensor import anchor_of, fit_evidence, predict, train_head    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    findings = kg_findings(); F = len(findings); nr = len(args.debates)
    by = by_split(load(args.debates, findings))
    ev_w = fit_evidence(by["train"])
    Y = {s: np.array([r["y"] for r in by[s]], dtype=float) for s in SPLITS}

    def feats(rows, kind):
        if kind == "crossrun":
            return np.array([crossrun_feats(r["claims"], F, nr) for r in rows])
        if kind == "base":
            return np.array([base_feats(r["claims"], F) for r in rows])
        return np.array([np.concatenate([base_feats(r["claims"], F),
                                         crossrun_feats(r["claims"], F, nr)])
                         for r in rows])

    a = {s: anchor_of(by[s], "evidence", ev_w, False) for s in SPLITS}
    print(f"anchor alone  TEST {auc(list(a['test'][Y['test']==1]), list(a['test'][Y['test']==0])):.4f}")

    res = {}
    for kind in ("crossrun", "base", "both"):
        X = {s: feats(by[s], kind) for s in SPLITS}
        mu, sd = X["train"].mean(0), X["train"].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        X = {s: (X[s] - mu) / sd for s in SPLITS}
        best = None
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(len(Y["train"])), 5)
        for l2 in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0):
            sc = np.zeros(len(Y["train"]))
            for fo in folds:
                m = np.zeros(len(Y["train"]), dtype=bool); m[fo] = True
                w, b, be = train_head(X["train"][~m], a["train"][~m], Y["train"][~m],
                                      np.zeros(X["train"].shape[1]), l2, 1500)
                sc[m] = predict(X["train"][m], a["train"][m], w, b, be)
            c = auc(list(sc[Y["train"] == 1]), list(sc[Y["train"] == 0]))
            if best is None or c > best[0]:
                best = (c, l2)
        cvb, l2 = best
        w, b, be = train_head(X["train"], a["train"], Y["train"],
                              np.zeros(X["train"].shape[1]), l2, 1500)
        s = {k: predict(X[k], a[k], w, b, be) for k in SPLITS}
        at = auc(list(s["test"][Y["test"] == 1]), list(s["test"][Y["test"] == 0]))
        sf = np.concatenate([s["train"], s["val"]]); yf = np.concatenate([Y["train"], Y["val"]])
        t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
        bb = bacc(s["test"].tolist(), Y["test"].tolist(), t)
        av = auc(list(s["val"][Y["val"] == 1]), list(s["val"][Y["val"] == 0]))
        print(f"  anchor + {kind:9s} TEST {at:.4f}  bAcc {bb:.4f}   (cv {cvb:.4f} val {av:.4f} l2 {l2})")
        res[kind] = {"test": at, "bacc": bb, "cv": cvb, "val": av}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
