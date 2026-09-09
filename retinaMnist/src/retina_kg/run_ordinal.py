"""Option A vs option C on RetinaMNIST, against the controls that decide anything.

Both arms decompose the 5-level ICDR scale the same way -- into the four
questions "is the grade above c?" -- and differ only in parameter sharing:

  A  cumulative-link   ONE trunk + 4 cut points          63 params
  C  stacked           FOUR independent trunks          236 params

So the comparison isolates one question: does message passing need to differ per
severity threshold, or does one shared representation with four cut points do?
A is C with proportional odds imposed.

Controls, per arm, exactly as in run_thothgnn3.py:
  no-graph      A+ = A- = 0.  Isolates message passing from capacity.
  shuffled-KG   same edge count and weight multiset, rewired, 5 draws.
                Isolates the ONTOLOGY from graph-ness.
  no-debate     probe and prior channels only. Isolates the citation merge.

Hyperparameters by 5-fold CV on train. Test evaluated once per arm. bAcc-style
thresholds are not swept here: the ordinal prediction comes from the cut points,
which are fitted, not chosen on a held-out sweep.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")

import ordinal                                                      # noqa: E402
import metrics_ordinal as M                                         # noqa: E402
import retina_features as RF                                        # noqa: E402
from kg_adjacency import build_adjacency, kg_operators              # noqa: E402
from run_thothgnn3 import shuffle_kg                                # noqa: E402

SPLITS = ("train", "val", "test")
L2S = (0.03, 0.1, 0.3)
KDIMS = (2, 4)


# ------------------------------------------------------------------- fitting --
def fit_predict(head, Ztr, ytr, Zev, Ap, An, n_cut, l2, kdim, seed, prior):
    if head == "A":
        P = ordinal.fit(Ztr, ytr, Ap, An, n_cut, l2=l2, kdim=kdim, seed=seed,
                        prior_centre=prior)
        s, p, g = ordinal.predict(P, Zev, Ap, An)
        return s, p, g, P
    Ps = ordinal.fit_stacked(Ztr, ytr, Ap, An, n_cut, l2=l2, kdim=kdim, seed=seed,
                             prior_centre=prior)
    S, p, g = ordinal.predict_stacked(Ps, Zev, Ap, An)
    return S, p, g, Ps


def cv_score(head, Z, y, Ap, An, n_cut, l2, kdim, prior, folds=5, seed=0):
    """Out-of-fold referable-DR AUC -- the quantity comparable to breast."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    ref = np.zeros(len(y))
    for fo in np.array_split(idx, folds):
        m = np.zeros(len(y), bool); m[fo] = True
        s, _, _, _ = fit_predict(head, Z[~m], y[~m], Z[m], Ap, An, n_cut,
                                 l2, kdim, seed, prior)
        ref[m] = s if np.ndim(s) == 1 else s[:, 1]
    t = M.referable(y)
    return M.auc(list(ref[t == 1]), list(ref[t == 0]))


def select(head, Z, y, Ap, An, n_cut, prior, log):
    best = (None, -1)
    for l2 in L2S:
        for k in KDIMS:
            v = cv_score(head, Z, y, Ap, An, n_cut, l2, k, prior)
            log.append({"head": head, "l2": l2, "kdim": k, "cv": v})
            print(f"    cv l2={l2:<5} k={k}  {v:.4f}")
            if v > best[1]:
                best = ((l2, k), v)
    print(f"    -> selected l2={best[0][0]}, k={best[0][1]} (cv {best[1]:.4f})")
    return best


def report(name, y, s, g, p, n_class, res):
    m = M.summarise(y, g, s, n_class)
    m["rank_violations"] = ordinal.rank_violations(p)
    ta = " ".join(f"{a:.3f}" for a in m["threshold_auc"])
    print(f"  {name:30s} refAUC {m['referable_auc']:.4f}  QWK {m['qwk']:+.4f}  "
          f"MAE {m['mae']:.3f}  macroR {m['macro_recall']:.4f}  adj {m['adjacent_acc']:.3f}"
          f"  [thr {ta}]")
    res[name] = m
    return m


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--npz-dir", default=str(P.NPZ))
    ap.add_argument("--probe-dir", default=str(P.RESULTS))
    ap.add_argument("--tags", default="retina_p3")
    ap.add_argument("--debate-root", default=str(P.DEBATES))
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--mediators", default=None)
    ap.add_argument("--heads", default="A,C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(P.RESULTS / "ordinal.json"))
    a = ap.parse_args()

    print(f"gradient check (ordinal): {ordinal.check_grad():.3e}")
    cfg = yaml.safe_load(Path(a.ontology).read_text())
    n_class = len(cfg["levels"]); n_cut = n_class - 1
    tags = tuple(a.tags.split(","))

    X, Y, IDS, names, prior, level, meta = RF.build(
        a.pack, a.npz_dir, a.probe_dir, tags, a.debate_root, SPLITS)
    if len(Y["train"]) == 0:
        raise SystemExit(f"no probe records under {a.probe_dir} for tags {tags}")
    Z = RF.standardise(X, SPLITS)
    meds = a.mediators.split(",") if a.mediators else None
    A, info = build_adjacency(a.pack, cfg, meds)
    Ap, An = kg_operators(A, prior)

    npc = len(tags)
    has_debate = bool(a.debate_root) and float(np.abs(X["train"][:, :, npc:npc+2]).sum()) > 0
    print(f"\nnodes {len(names)}  features {Z['train'].shape[2]}  "
          f"n={[len(Y[s]) for s in SPLITS]}  classes {n_class}")
    print(f"KG: {info['n_edges']}/{info['n_pairs']} pairs via {info['mediators']}, "
          f"A+ {int((Ap>0).sum()//2)}  A- {int((An>0).sum()//2)}, "
          f"isolated {len(info['isolated'])}")
    print(f"debate channel: {'present' if has_debate else 'ABSENT (probe-only arm)'}")
    for s in SPLITS:
        print(f"  {s:5s} grades {np.bincount(Y[s], minlength=n_class).tolist()}  "
              f"referable {M.referable(Y[s]).mean():.3f}")

    res = {"config": {"tags": list(tags), "mediators": info["mediators"],
                      "n_findings": len(names), "kg": info,
                      "has_debate": has_debate, "n_class": n_class,
                      "n": {s: len(Y[s]) for s in SPLITS}}, "cv": []}

    for head in a.heads.split(","):
        print(f"\n=== option {head}: "
              f"{'cumulative-link, one trunk + 4 cut points' if head=='A' else 'stacked, four independent trunks'} ===")
        (l2, k), cv = select(head, Z["train"], Y["train"], Ap, An, n_cut, prior, res["cv"])
        res.setdefault("selected", {})[head] = {"l2": l2, "kdim": k, "cv": cv}

        s, p, g, _ = fit_predict(head, Z["train"], Y["train"], Z["test"], Ap, An,
                                 n_cut, l2, k, a.seed, prior)
        report(f"{head}: GNN (real KG)", Y["test"], s, g, p, n_class, res)

        Z0 = np.zeros_like(Ap)
        s0, p0, g0, _ = fit_predict(head, Z["train"], Y["train"], Z["test"], Z0, Z0,
                                    n_cut, l2, k, a.seed, prior)
        report(f"{head}: no-graph (A=0)", Y["test"], s0, g0, p0, n_class, res)

        rng = np.random.default_rng(a.seed)
        draws = []
        for d in range(5):
            Sp, Sn = shuffle_kg(Ap, An, rng)
            sd_, pd_, gd_, _ = fit_predict(head, Z["train"], Y["train"], Z["test"],
                                           Sp, Sn, n_cut, l2, k, a.seed, prior)
            mm = M.summarise(Y["test"], gd_, sd_, n_class)
            draws.append(mm["referable_auc"])
        real = res[f"{head}: GNN (real KG)"]["referable_auc"]
        mu, sd = float(np.mean(draws)), float(np.std(draws))
        z = (real - mu) / sd if sd > 0 else float("nan")
        print(f"  {'shuffled-KG (5 draws)':30s} refAUC {mu:.4f} +- {sd:.4f}  "
              f"[{min(draws):.4f}, {max(draws):.4f}]")
        print(f"  {'-> real KG minus shuffled':30s} {real-mu:+.4f}  ({z:+.2f} sd)")
        res[f"{head}: shuffled-KG"] = {"mean": mu, "sd": sd, "draws": draws,
                                       "delta": real - mu, "z": z}

        if has_debate:
            Znd = {s_: Z[s_].copy() for s_ in SPLITS}
            for s_ in SPLITS:
                Znd[s_][:, :, npc:npc + 2] = 0.0
            snd, pnd, gnd, _ = fit_predict(head, Znd["train"], Y["train"], Znd["test"],
                                           Ap, An, n_cut, l2, k, a.seed, prior)
            report(f"{head}: no-debate channel", Y["test"], snd, gnd, pnd, n_class, res)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
