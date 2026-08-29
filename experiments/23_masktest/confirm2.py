"""Confirmation 2 -- the negation lockstep, on the maskring subsample.

Mechanism claim: the model perceives the finding correctly and then applies the
negation backwards. It predicts that a clearer image moves the two phrasings in
OPPOSITE directions on the same finding. On BrEaST it did:
irregular_shape verification 0.7490->0.8925 while negated 0.1902->0.0941.

Measured here as probe->malignancy AUC, since BUS-BRA has no descriptor labels.
The no-mask negated arm is the archived busbra_p2 run on the identical npz; that
is a different run from tonight's, and is flagged rather than hidden. Everything
is restricted to the images present in all four cells, so the comparison is paired.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc, load_probes                                # noqa: E402
from claims import kg_findings                                        # noqa: E402


def load(tag, names, d):
    out = {}
    for f in sorted(Path(d).glob(f"{tag}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r["p_yes"] or {}
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    R, E = _HERE / "results", _HERE.parents[0] / "16_external/results"

    cells = {"ver_nomask": load("bbase_p3", names, R),
             "ver_mask": load("bring_p3", names, R),
             "neg_mask": load("bring_p2", names, R)}
    arch = load_probes(names, ("busbra_p2", "busbra_p3"), E)
    cells["neg_nomask"] = {i: v[:, 0] for i, v in arch.items()}

    common = sorted(set.intersection(*[set(c) for c in cells.values()]))
    z = np.load(_HERE.parents[1] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    y = (z["labels"][common, 0] == 0).astype(float)
    M = {k: np.array([c[i] for i in common]) for k, c in cells.items()}
    print(f"paired subsample: n = {len(common)} images  "
          f"(malignant {int(y.sum())})\n")

    print("probe -> malignancy AUC.  Prediction: the two columns move in OPPOSITE directions.")
    print(f"{'finding':46s} {'verification':>22s} {'negated':>22s}   lockstep?")
    print(f"{'':46s} {'no mask':>10s}{'ring':>12s} {'no mask':>10s}{'ring':>12s}")
    opp = same = 0
    rows = {}
    for j, f in enumerate(names):
        v0, v1 = auc(y, M["ver_nomask"][:, j]), auc(y, M["ver_mask"][:, j])
        n0, n1 = auc(y, M["neg_nomask"][:, j]), auc(y, M["neg_mask"][:, j])
        dv, dn = v1 - v0, n1 - n0
        o = dv * dn < 0
        opp += o; same += not o
        rows[f] = {"ver": [v0, v1], "neg": [n0, n1], "d_ver": dv, "d_neg": dn,
                   "opposite": bool(o)}
        print(f"{f:46s} {v0:10.4f}{v1:12.4f} {n0:10.4f}{n1:12.4f}   "
              f"{'OPPOSITE' if o else 'same dir'}")
    print(f"\n  opposite-direction findings: {opp}/16   same-direction: {same}/16")

    print("\n  the two findings the BrEaST lockstep was stated on:")
    for f in ("irregular_shape", "oval_shape"):
        r = rows[f]
        print(f"    {f:18s} verification {r['ver'][0]:.4f} -> {r['ver'][1]:.4f} "
              f"({r['d_ver']:+.4f})   negated {r['neg'][0]:.4f} -> {r['neg'][1]:.4f} "
              f"({r['d_neg']:+.4f})")

    print("\n  whole-arm, zero-parameter KG-signed sum on this subsample:")
    for k, lab in (("ver_nomask", "verification, no mask"), ("ver_mask", "verification, ring"),
                   ("neg_nomask", "negated, no mask"), ("neg_mask", "negated, ring")):
        A = M[k]
        Z = (A - A.mean(0)) / np.where(A.std(0) < 1e-9, 1, A.std(0))
        print(f"    {lab:26s} {auc(y, Z @ prior):.4f}")

    (_HERE / "results/confirm2.json").write_text(json.dumps(
        {"n": len(common), "per_finding": rows, "n_opposite": int(opp)},
        indent=1, default=float))
    print("\nwrote results/confirm2.json")


if __name__ == "__main__":
    main()
