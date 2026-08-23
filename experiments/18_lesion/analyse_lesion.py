"""Does the KG's differential diagnosis carry signal, and does it add to findings?

Three questions, in order of how much they would change the project:

  1. Alone. Score by the ontology's own mapping -- sum_L p(lesion L) * prior(L) --
     with zero fitted parameters. This is the differential-diagnosis claim in its
     purest form: the model names the lesion, the KG says what that lesion means.
  2. Fitted. A free readout over the 20 lesion probes, to separate "the ontology's
     mapping is wrong" from "the lesion probes see nothing".
  3. Added. Does it improve on the finding probes? The two channels ask different
     questions of the same ontology, so if they are near-independent views the
     pooling result predicts a gain; if the model is really answering both from one
     internal impression, they will be redundant.

The expected shape of the result is asymmetric and known in advance: 15 of the 20
lesions are is_a benign_breast_lesion and breast_cancer has a single appearance
triple, so this channel should rule OUT malignancy rather than rule it in.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc                                             # noqa: E402
from claims import bacc, kg_findings                                   # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                      # noqa: E402
from lesions import bipartite, lesion_table                            # noqa: E402

SPLITS = ("train", "val", "test")


def load(tag, ents):
    out = {}
    for f in sorted((_HERE / "results").glob(f"{tag}_*.jsonl")):
        split = f.name.split("_")[1]
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            v = r["p_yes"] or {}
            out[(split, int(r["index"]))] = (
                np.array([float(v[e]) if v.get(e) is not None else 0.5 for e in ents]),
                1.0 if r["gold"] == "MALIGNANT" else 0.0)
    return out


def finding_matrix(names):
    """P2+P3 finding probes, the current best channel, for the comparison."""
    from freeze_models import breastmnist_matrix
    return breastmnist_matrix([(n, 0) for n in names])


def sweep(sc, Y):
    f = np.concatenate([sc["train"], sc["val"]]); g = np.concatenate([Y["train"], Y["val"]])
    return max(sorted(set(f.tolist())), key=lambda t: bacc(f.tolist(), g.tolist(), t))


def row(name, sc, Y, res):
    a = {s: auc(Y[s], sc[s]) for s in SPLITS}
    t = sweep(sc, Y)
    bb = bacc(sc["test"].tolist(), Y["test"].tolist(), t)
    print(f"  {name:34s} train {a['train']:.4f}  val {a['val']:.4f}  "
          f"TEST {a['test']:.4f}  bAcc {bb:.4f}")
    res[name] = {"auc": a, "bacc": bb}
    return a["test"]


def main():
    L = lesion_table()
    ents = [e for e, *_ in L]
    prior = np.array([p for _, _, p, _, _ in L])
    tab = load("lesion", ents)
    if not tab:
        print("no lesion probes yet"); return
    n = {s: sum(1 for k in tab if k[0] == s) for s in SPLITS}
    print(f"lesion probes: {n}   ({len(ents)} lesions)")
    if min(n.values()) < 20:
        print("not enough yet"); return

    keys = {s: sorted(k for k in tab if k[0] == s) for s in SPLITS}
    X = {s: np.array([tab[k][0] for k in keys[s]]) for s in SPLITS}
    Y = {s: np.array([tab[k][1] for k in keys[s]]) for s in SPLITS}
    mu, sd = X["train"].mean(0), X["train"].std(0); sd = np.where(sd < 1e-9, 1, sd)
    Z = {s: (X[s] - mu) / sd for s in SPLITS}
    res = {}

    print("\n=== 1. the ontology's own mapping, zero fitted parameters ===")
    # centre the prior so a channel that is 15/20 benign is not a constant offset
    w0 = prior - prior.mean()
    row("KG differential (0 params)", {s: Z[s] @ w0 for s in SPLITS}, Y, res)
    print(f"     (raw p(lesion) means: benign-prior lesions "
          f"{X['train'][:, prior == 0].mean():.3f}, "
          f"malignant-prior {X['train'][:, prior > 0.5].mean():.3f})")

    print("\n=== 2. free readout over the lesion probes ===")
    best = None
    for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
        for l2 in L2S:
            c = cv_score(Z["train"], Y["train"], l2, f)[0]
            if best is None or c > best[0]:
                best = (c, l2, nm, f)
    cv, l2, nm, f = best
    w, b = f(Z["train"], Y["train"], l2)
    print(f"  (cv {cv:.4f}, {nm}, l2={l2})")
    a_les = row("lesion probes, fitted", {s: Z[s] @ w + b for s in SPLITS}, Y, res)

    print("\n=== 3. does it add to the finding probes? ===")
    names = [x for x, _ in kg_findings()]
    Xf, Yf = finding_matrix(names)
    # align on samples present in both channels
    al = {}
    for s in SPLITS:
        fk = sorted(k for k in tab if k[0] == s)
        idx = [int(k[1]) for k in fk]
        m = min(len(idx), len(Yf[s]))
        al[s] = (Xf[s][[i for i in idx if i < len(Yf[s])]],
                 X[s][[j for j, i in enumerate(idx) if i < len(Yf[s])]],
                 Yf[s][[i for i in idx if i < len(Yf[s])]])
    muf = al["train"][0].mean(0); sdf = al["train"][0].std(0); sdf = np.where(sdf < 1e-9, 1, sdf)
    Yc = {s: al[s][2] for s in SPLITS}
    Zf = {s: (al[s][0] - muf) / sdf for s in SPLITS}
    Zl = {s: (al[s][1] - mu) / sd for s in SPLITS}
    for tag, D in (("findings only (P2+P3)", Zf),
                   ("lesions only", Zl),
                   ("findings + lesions", {s: np.hstack([Zf[s], Zl[s]]) for s in SPLITS})):
        b2 = None
        for nm2, f2 in (("ce", fit_ce), ("rank", fit_rank)):
            for l22 in L2S:
                c2 = cv_score(D["train"], Yc["train"], l22, f2)[0]
                if b2 is None or c2 > b2[0]:
                    b2 = (c2, l22, f2)
        _, l22, f2 = b2
        w2, b3 = f2(D["train"], Yc["train"], l22)
        row(tag, {s: D[s] @ w2 + b3 for s in SPLITS}, Yc, res)

    (_HERE / "results/lesion_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/lesion_analysis.json")


if __name__ == "__main__":
    main()
