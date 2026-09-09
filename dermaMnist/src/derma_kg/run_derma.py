"""ThothGNN v3 with the multinomial head on DermaMNIST, against its controls.

Same three controls that decided the question on breast and retina:
  no-graph      A+ = A- = 0.  Isolates message passing from capacity.
  shuffled-KG   same edge count and weight multiset, rewired, 5 draws.
                Isolates the ONTOLOGY from graph-ness.
  no-debate     probe and prior channels only. Isolates the citation merge.

Plus both adjacency constructions (structural and the breast-style appearance
one), because on ICDR the appearance construction collapsed to a single edge and
on dermoscopy it does not -- so which one the null holds under is a real question
rather than a formality.

Headline is MACRO-RECALL and MACRO-OVR AUC, not accuracy: DermaMNIST test is 67%
`nv`, so always predicting the majority scores 0.669 accuracy and 0.143
macro-recall. Selection is 5-fold CV on train only; test evaluated once per arm.
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

SPLITS = ("train", "val", "test")
L2S = (0.03, 0.1, 0.3)
KDIMS = (2, 4)


def shuffle_kg(Ap, An, rng):
    """Rewire preserving the multiset of weights and symmetry."""
    F = Ap.shape[0]; out = []
    for A in (Ap, An):
        iu = np.triu_indices(F, 1)
        w = A[iu].copy(); rng.shuffle(w)
        B = np.zeros_like(A); B[iu] = w
        out.append(B + B.T)
    return out


def fit_predict(Ztr, ytr, Zev, Ap, An, NC, l2, k, seed, prior, balanced=True):
    Q = nominal.fit(Ztr, ytr, Ap, An, NC, l2=l2, kdim=k, seed=seed,
                    prior_centre=prior,
                    class_weight="balanced" if balanced else None)
    return nominal.predict(Q, Zev, Ap, An)


# Selection only has to rank (l2, kdim); it does not have to be the final fit.
# One fit at n=7007 with 55 findings is 563 s -- the trunk einsum scales with
# n x F^2 -- so a full-data 6x5 grid is 281 min before a single test arm runs.
# A stratified 2,000-image subsample at 300 epochs ranks the same grid in ~25 min.
# The REPORTED arms are still fitted on all 7,007 at 800 epochs.
CV_N = 2000
CV_EPOCHS = 300


def _stratified(y, n, rng, n_class):
    """Keep every class's share, and never let a class drop below 2 examples."""
    y = np.asarray(y); keep = []
    frac = min(1.0, n / len(y))
    for c in range(n_class):
        idx = np.flatnonzero(y == c)
        if not len(idx):
            continue
        k = min(len(idx), max(2, int(round(len(idx) * frac))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def cv_macro_recall(Z, y, Ap, An, NC, l2, k, prior, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    sub = _stratified(y, CV_N, rng, NC) if len(y) > CV_N else np.arange(len(y))
    Zs, ys = Z[sub], np.asarray(y)[sub]
    idx = rng.permutation(len(ys))
    pred = np.zeros(len(ys), int)
    for fo in np.array_split(idx, folds):
        m = np.zeros(len(ys), bool); m[fo] = True
        Q = nominal.fit(Zs[~m], ys[~m], Ap, An, NC, l2=l2, kdim=k,
                        epochs=CV_EPOCHS, seed=seed, prior_centre=prior,
                        class_weight="balanced")
        pred[m] = nominal.predict(Q, Zs[m], Ap, An)[2]
    return M.macro_recall(ys, pred, NC)


def report(name, y, P_, g, NC, res):
    m = M.summarise(y, g, P_, NC)
    print(f"  {name:30s} macroR {m['macro_recall']:.4f}  ovrAUC {m['macro_ovr_auc']:.4f}  "
          f"F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")
    res[name] = m
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--npz-dir", default=str(P.NPZ))
    ap.add_argument("--probe-dir", default=str(P.RESULTS))
    ap.add_argument("--tag", default="derma_p3")
    ap.add_argument("--debate-root", default=None)
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--mediators", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(P.RESULTS / "derma_nominal.json"))
    a = ap.parse_args()

    print(f"gradient check (nominal): {nominal.check_grad():.3e}")
    cfg = yaml.safe_load(Path(a.ontology).read_text())
    NC = len(cfg["classes"])
    X, Y, IDS, names, prior, classes, meta = DF.build(
        a.pack, a.npz_dir, a.probe_dir, a.tag, a.debate_root, SPLITS)
    Z = DF.standardise(X, SPLITS)
    has_debate = bool(a.debate_root) and float(np.abs(X["train"][:, :, 1:3]).sum()) > 0

    meds = a.mediators.split(",") if a.mediators else None
    A_, info = build_adjacency(a.pack, cfg, meds)
    Ap, An = kg_operators(A_, classes)

    print(f"\nnodes {len(names)}  features {Z['train'].shape[2]}  classes {NC}  "
          f"n={[len(Y[s]) for s in SPLITS]}")
    print(f"KG: {info['n_edges']}/{info['n_pairs']} pairs via {info['mediators']}, "
          f"A+ {int((Ap>0).sum()//2)}  A- {int((An>0).sum()//2)}, "
          f"isolated {len(info['isolated'])}")
    print(f"debate channel: {'present' if has_debate else 'ABSENT (probe-only arm)'}")
    y = Y["test"]
    maj = np.bincount(Y["train"], minlength=NC).argmax()
    print(f"  test classes {np.bincount(y, minlength=NC).tolist()}")
    print(f"  majority-class ({cfg['classes'][maj]}): acc "
          f"{(y == maj).mean():.4f}  macroR {M.macro_recall(y, np.full(len(y), maj), NC):.4f}")

    res = {"config": {"n_findings": len(names), "kg": info, "n_class": NC,
                      "has_debate": has_debate, "tag": a.tag,
                      "n": {s: len(Y[s]) for s in SPLITS}}, "cv": []}
    best = None
    for l2 in L2S:
        for k in KDIMS:
            v = cv_macro_recall(Z["train"], Y["train"], Ap, An, NC, l2, k, prior)
            res["cv"].append({"l2": l2, "kdim": k, "macro_recall": v})
            print(f"    cv l2={l2:<5} k={k}  macroR {v:.4f}")
            if best is None or v > best[1]:
                best = ((l2, k), v)
    (l2, k), cv = best
    print(f"    -> selected l2={l2}, k={k} (cv macroR {cv:.4f})")
    res["selected"] = {"l2": l2, "kdim": k, "cv_macro_recall": cv}

    print()
    _, Pt, g = fit_predict(Z["train"], Y["train"], Z["test"], Ap, An, NC, l2, k, a.seed, prior)
    report("GNN (real KG)", y, Pt, g, NC, res)

    Z0 = np.zeros_like(Ap)
    _, P0, g0 = fit_predict(Z["train"], Y["train"], Z["test"], Z0, Z0, NC, l2, k, a.seed, prior)
    report("no-graph (A=0)", y, P0, g0, NC, res)

    rng = np.random.default_rng(a.seed); draws = []
    for _ in range(5):
        Sp, Sn = shuffle_kg(Ap, An, rng)
        _, Ps, gs = fit_predict(Z["train"], Y["train"], Z["test"], Sp, Sn, NC, l2, k, a.seed, prior)
        draws.append(M.macro_recall(y, gs, NC))
    real = res["GNN (real KG)"]["macro_recall"]
    mu, sd = float(np.mean(draws)), float(np.std(draws))
    z = (real - mu) / sd if sd > 0 else float("nan")
    print(f"  {'shuffled-KG (5 draws)':30s} macroR {mu:.4f} +- {sd:.4f} "
          f"[{min(draws):.4f}, {max(draws):.4f}]")
    print(f"  {'-> real KG minus shuffled':30s} {real-mu:+.4f}  ({z:+.2f} sd)")
    res["shuffled-KG"] = {"mean": mu, "sd": sd, "draws": draws,
                          "delta": real - mu, "z": z}

    if has_debate:
        Znd = {s: Z[s].copy() for s in SPLITS}
        for s in SPLITS:
            Znd[s][:, :, 1:3] = 0.0
        _, Pn, gn = fit_predict(Znd["train"], Y["train"], Znd["test"], Ap, An,
                                NC, l2, k, a.seed, prior)
        report("no-debate channel", y, Pn, gn, NC, res)

    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
