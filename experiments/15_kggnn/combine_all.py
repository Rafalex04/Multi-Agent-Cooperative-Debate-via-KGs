"""All mechanisms, combined, with the KG-topology GNN among them.

Four sources of evidence about the same image, deliberately built to be
different mechanisms rather than repeats of one:

  probe      16 yes/no visual probes, p(yes) from first-token logprobs
  debate     the 4-run evidence-weighted claim readout
  birads     the two-turn BI-RADS logprob score
  gnn        SGC over the KG finding graph, fusing probe and debate per node

Everything is z-scored on TRAIN statistics only and every combination weight is
either uniform or fitted on train, so val and test stay clean.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc                                  # noqa: E402
from features import build                                            # noqa: E402
from kg_graph import build_adjacency, normalised                      # noqa: E402
from kg_gnn import fit_eval, propagate                                # noqa: E402
from kg_tensor import anchor_of, evidence_score, fit_evidence, mal_share  # noqa: E402
from combine import birads_scores                                     # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


def z(train, all_):
    m, s = float(np.mean(train)), float(np.std(train))
    s = s if s > 1e-9 else 1.0
    return {k: (np.asarray(v) - m) / s for k, v in all_.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="+", default=[
        str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                              "debates_v5q_r2", "debates_v5q_r3")])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, X, Y, findings = build(args.debates)
    F = len(findings)
    prior = np.array([s for _, s in findings])
    print("aligned: " + "  ".join(f"{s} {len(Y[s])}" for s in SPLITS))

    ev_w = fit_evidence(rows["train"])
    bir = birads_scores()

    C = {}
    C["probe"] = {s: (X[s][:, :, 0] * prior).sum(1) for s in SPLITS}
    C["debate"] = {s: np.array([evidence_score(r["claims"], ev_w) for r in rows[s]])
                   for s in SPLITS}
    C["mal_share"] = {s: np.array([mal_share(r["claims"]) for r in rows[s]]) for s in SPLITS}
    C["birads"] = {s: np.array([bir.get((s, int(r["sid"])), np.nan) for r in rows[s]])
                   for s in SPLITS}

    # GNN component: SGC over the KG graph, anchored on the debate readout
    A, info = build_adjacency(_KG, [f for f, _ in findings])
    anchors = {s: anchor_of(rows[s], "evidence", ev_w, False) for s in SPLITS}
    Xs = {s: propagate(X[s], normalised(A), 1).reshape(len(X[s]), -1) for s in SPLITS}
    at, bb, cv, av, l2, gs = fit_eval(Xs, Y, anchors, (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0))
    print(f"KG-GNN component alone: TEST {at:.4f} bAcc {bb:.4f} (cv {cv:.4f} val {av:.4f})")
    C["gnn"] = gs

    # drop samples with a missing birads score
    ok = {s: ~np.isnan(C["birads"][s]) for s in SPLITS}
    for k in C:
        C[k] = {s: np.asarray(C[k][s])[ok[s]] for s in SPLITS}
    Yk = {s: Y[s][ok[s]] for s in SPLITS}
    print("after birads alignment: " + "  ".join(f"{s} {int(ok[s].sum())}" for s in SPLITS))

    Z = {k: z(C[k]["train"], C[k]) for k in C}

    def rep(name, sc, out):
        a = {s: auc(list(sc[s][Yk[s] == 1]), list(sc[s][Yk[s] == 0])) for s in SPLITS}
        sf = np.concatenate([sc["train"], sc["val"]]); yf = np.concatenate([Yk["train"], Yk["val"]])
        t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
        b = bacc(sc["test"].tolist(), Yk["test"].tolist(), t)
        print(f"  {name:38s} train {a['train']:.4f}  val {a['val']:.4f}  "
              f"TEST {a['test']:.4f}   bAcc {b:.4f}")
        out[name] = {"train": a["train"], "val": a["val"], "test": a["test"], "bacc": b}

    print("\n=== correlation on train ===")
    ks = list(Z)
    for i, a in enumerate(ks):
        for b_ in ks[i + 1:]:
            print(f"  {a:10s} vs {b_:10s}  r = "
                  f"{float(np.corrcoef(Z[a]['train'], Z[b_]['train'])[0,1]):+.3f}")

    res = {}
    print("\n  readout                                  train      val      TEST     bAcc")
    for k in ks:
        rep(f"{k} alone", Z[k], res)
    combos = {
        "probe + debate": ["probe", "debate"],
        "probe + birads": ["probe", "birads"],
        "debate + birads": ["debate", "birads"],
        "probe + debate + birads": ["probe", "debate", "birads"],
        "probe + gnn + birads": ["probe", "gnn", "birads"],
        "gnn + birads": ["gnn", "birads"],
        "all five": ks,
    }
    for name, parts in combos.items():
        rep(name, {s: np.mean([Z[p][s] for p in parts], axis=0) for s in SPLITS}, res)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
