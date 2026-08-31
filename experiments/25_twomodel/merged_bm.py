"""Probes + debate on BreastMNIST (780), for one debate corpus.

The probe channel is qwen3-vl P3 (`probep3`, already measured on all 780) and is
IDENTICAL in every arm. The only thing that varies between corpora is who argued.
That makes the HET-vs-HOM comparison a clean test of the debate channel: a
stronger partner cannot win by improving perception, because perception is held
fixed by construction.

  python merged_bm.py --debates <dir> --label het --out het.json
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _d in ("17_hetgnn", "14_kgtensor", "15_kggnn", "16_external"):
    sys.path.insert(0, str(_HERE.parents[1] / _d))
from analyse_e1 import auc, bacc, paired_bootstrap        # noqa: E402
from claims import kg_findings                            # noqa: E402
from gates import fit_rank                                # noqa: E402
from hetgraph import kg_operators, load_sample            # noqa: E402

SPLITS = ("train", "val", "test")
_ROOT = _HERE.parents[1].parent / "breastMnist/data/breast"
_PROBES = _HERE.parents[1] / "15_kggnn/results"


def load_probe_split(tag, split, names):
    out = {}
    for f in sorted(_PROBES.glob(f"{tag}_{split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", required=True, help="corpus root holding train/val/test")
    ap.add_argument("--probe-tag", default="probep3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    F = len(names)

    rows, ys, keys = [], [], []
    for sp in SPLITS:
        pr = load_probe_split(args.probe_tag, sp, names)
        z = np.load(_ROOT / f"images_224/{sp}.npz")
        lab = (z["labels"][:, 0] == 0).astype(float)          # MALIGNANT=0 -> y=1
        for f in sorted(glob.glob(str(Path(args.debates) / sp / "*.json"))):
            try:
                Xc, Acc, Acf, mask, y, sid = load_sample(f, names)
            except Exception:
                continue
            i = int(sid)
            if i not in pr:
                continue
            assert y == lab[i], f"label mismatch {sp}:{i} debate={y} npz={lab[i]}"
            rows.append((pr[i], Xc, Acf, mask))
            ys.append(y); keys.append((sp, i))
    n = len(rows)
    print(f"corpus={args.label}  matched images: {n}  malignant {int(sum(ys))}")
    if n < 100:
        print("too few matched debates"); return
    y = np.array(ys)

    # channels: probe P3, argument mass, net stance, prior
    PMASS, PSTANCE, PPRIOR = 1, 2, 3
    X = np.zeros((n, F, 4))
    for k, (p, Xc, Acf, mask) in enumerate(rows):
        X[k, :, 0] = p
        X[k, :, PMASS] = (mask[:, None] * Acf).sum(0)
        X[k, :, PSTANCE] = ((mask * Xc[:, 0])[:, None] * Acf).sum(0)
        X[k, :, PPRIOR] = prior
    mu = X.reshape(-1, 4).mean(0); sd = X.reshape(-1, 4).std(0)
    Z = (X - mu) / np.where(sd < 1e-9, 1, sd)

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = [np.isin(np.arange(n), order[i::args.folds]) for i in range(args.folds)]

    arms = {"probes only": [0, PPRIOR],
            "probes + debate": [0, PMASS, PSTANCE, PPRIOR]}
    oof = {k: np.zeros(n) for k in arms}
    for te in folds:
        tr = ~te
        for lab, cols in arms.items():
            A = Z[:, :, cols].reshape(n, -1)
            w, b = fit_rank(A[tr], y[tr], 0.3)
            oof[lab][te] = A[te] @ w + b

    res = {"label": args.label, "debates": args.debates, "n": n,
           "n_malignant": int(y.sum()), "arms": {}}
    print(f"\n{'arm':22s} {'AUC':>8s} {'bAcc':>8s}")
    for lab in arms:
        a = float(auc(y, oof[lab])); b = float(bacc(y, oof[lab], float(np.median(oof[lab]))))
        res["arms"][lab] = {"auc": a, "bacc": b}
        print(f"{lab:22s} {a:8.4f} {b:8.4f}")
    d = res["arms"]["probes + debate"]["auc"] - res["arms"]["probes only"]["auc"]
    p = paired_bootstrap(y, oof["probes + debate"], oof["probes only"])
    res["debate_delta"] = d
    res["P_debate_helps"] = float(p)
    res["oof"] = {k: v.tolist() for k, v in oof.items()}
    res["y"] = y.tolist()
    res["keys"] = [f"{s}:{i}" for s, i in keys]
    print(f"\ndebate channel contribution: {d:+.4f}   P(debate helps) = {p:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
