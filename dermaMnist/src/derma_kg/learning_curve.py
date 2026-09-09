"""Learning curves for the derma framework, and the power question behind them.

Two things a curve can answer here, and the second is the one that matters:

  1. How does performance scale with training set size? DermaMNIST has 7,007
     training images against BreastMNIST's 546, so this is the first dataset in
     the project where a curve has real range.

  2. DOES THE KG EFFECT GROW WITH n? This is the diagnostic that decided C3.
     Round 3 left KG topology at P=0.939 with a curve predicting ~820 cases would
     clear it; more data made the effect FALL, which is what a null does and what
     an underpowered positive does not. Plotting `real KG - shuffled KG` against n
     is therefore the honest test, not the headline curve.

Every point is 3 seeds, mean +- sd. Shuffled-KG gets its own draw per seed so the
gap has an error bar rather than a point estimate.

  python learning_curve.py --debate-root ../../data/derma/debates_d1
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
import nominal, metrics_nominal as M, derma_features as DF           # noqa: E402
from kg_adjacency import build_adjacency, kg_operators                # noqa: E402
from run_derma import fit_predict, shuffle_kg, SPLITS                 # noqa: E402

FRACTIONS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)


def subsample(y, frac, rng, n_class):
    """Stratified subsample: every class keeps its share, and never drops to zero.

    An unstratified draw at 5% of 7,007 would take ~4 `df` images and could take
    none, which makes macro-recall undefined for that class and the curve
    unreadable.
    """
    keep = []
    for c in range(n_class):
        idx = np.flatnonzero(np.asarray(y) == c)
        k = max(2, int(round(len(idx) * frac))) if len(idx) else 0
        k = min(k, len(idx))
        if k:
            keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--npz-dir", default=str(P.NPZ))
    ap.add_argument("--probe-dir", default=str(P.RESULTS))
    ap.add_argument("--tag", default="derma_p3")
    ap.add_argument("--debate-root", default=None)
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--kdim", type=int, default=None)
    ap.add_argument("--out", default=str(P.RESULTS / "derma_learning_curve.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.ontology).read_text())
    NC = len(cfg["classes"])
    X, Y, IDS, names, prior, classes, meta = DF.build(
        a.pack, a.npz_dir, a.probe_dir, a.tag, a.debate_root, SPLITS)
    Z = DF.standardise(X, SPLITS)
    A_, info = build_adjacency(a.pack, cfg)
    Ap, An = kg_operators(A_, classes)
    ytr, yte = Y["train"], Y["test"]

    sel = {"l2": a.l2, "kdim": a.kdim}
    if sel["l2"] is None:
        for f in ("derma_nominal.json", "derma_probeonly.json"):
            p = P.RESULTS / f
            if p.exists():
                s = json.loads(p.read_text()).get("selected")
                if s:
                    sel = {"l2": s["l2"], "kdim": s["kdim"]}
                    print(f"  hyperparameters from {f}: {sel}")
                    break
    if sel["l2"] is None:
        sel = {"l2": 0.1, "kdim": 2}
        print(f"  no selection file found, defaulting to {sel}")

    has_debate = bool(a.debate_root) and float(np.abs(X["train"][:, :, 1:3]).sum()) > 0
    print(f"\nlearning curve | {len(names)} findings | debate "
          f"{'present' if has_debate else 'ABSENT'} | {len(SEEDS)} seeds/point")
    print(f"{'n_train':>8s} {'macroR real':>18s} {'macroR shuffled':>18s} "
          f"{'real - shuffled':>18s} {'ovrAUC real':>14s}")

    rows = []
    for frac in FRACTIONS:
        real, shuf, aucs, ns = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(1000 + sd)
            keep = subsample(ytr, frac, rng, NC)
            ns.append(len(keep))
            Ztr, yt = Z["train"][keep], ytr[keep]
            _, Pt, g = fit_predict(Ztr, yt, Z["test"], Ap, An, NC,
                                   sel["l2"], sel["kdim"], sd, prior)
            real.append(M.macro_recall(yte, g, NC))
            aucs.append(M.macro_ovr_auc(Pt, yte, NC)[0])
            Sp, Sn = shuffle_kg(Ap, An, np.random.default_rng(sd))
            _, _, gs = fit_predict(Ztr, yt, Z["test"], Sp, Sn, NC,
                                   sel["l2"], sel["kdim"], sd, prior)
            shuf.append(M.macro_recall(yte, gs, NC))
        r, s_, au = np.array(real), np.array(shuf), np.array(aucs)
        d = r - s_
        rows.append({"frac": frac, "n_train": int(np.mean(ns)),
                     "macro_recall": [float(r.mean()), float(r.std())],
                     "shuffled": [float(s_.mean()), float(s_.std())],
                     "delta": [float(d.mean()), float(d.std())],
                     "macro_ovr_auc": [float(au.mean()), float(au.std())],
                     "seeds": list(SEEDS)})
        print(f"{int(np.mean(ns)):8d} {r.mean():10.4f} +-{r.std():5.4f} "
              f"{s_.mean():10.4f} +-{s_.std():5.4f} "
              f"{d.mean():+10.4f} +-{d.std():5.4f} {au.mean():9.4f} +-{au.std():5.4f}")

    # does the KG gap grow with n? slope of delta vs log n
    n = np.array([r["n_train"] for r in rows], float)
    d = np.array([r["delta"][0] for r in rows])
    slope = float(np.polyfit(np.log(n), d, 1)[0])
    print(f"\n  slope of (real - shuffled) vs log n: {slope:+.5f}")
    print("  A real-but-underpowered effect RISES with n; a null does not.")
    out = {"curve": rows, "selected": sel, "has_debate": has_debate,
           "delta_slope_vs_logn": slope, "kg": info}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
