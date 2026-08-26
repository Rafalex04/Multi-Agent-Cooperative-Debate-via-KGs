"""Does framing fix the probes that are pointed the wrong way?

Same 252 BrEaST images, same 32 probes, same radiologist ground truth. The only
thing that changes is how much of the frame the lesion fills:

    pad 1.25   the lesion nearly fills the frame
    pad 2.00   the registered framing
    pad 3.00   the lesion in context

The question is not "does cropping raise AUC". It is whether the SPECIFIC
failures move. Two are on the table and they predict different things:

  * The negated arm inverts shape (P2 reads irregular_shape at 0.1902 while P3
    reads it at 0.7490). If that is a framing artifact it should ease when the
    lesion fills the frame. If it is negation handling it will not move, because
    the window has nothing to do with the word "not".
  * The Halo group is at or below chance in BOTH wordings. A thin echogenic rim
    is a small structure at the lesion boundary, which is the one failure a
    tighter crop plausibly fixes.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc                                             # noqa: E402
from claims import kg_findings                                        # noqa: E402
from sign_audit import zscore                                         # noqa: E402

_DATA = _HERE.parents[1] / "data/external"
# (tag, npz, results dir, jsonl prefix per phrasing)
ARMS = (("pad 1.25", "breast_pad125_224.npz", _HERE / "results", ("pad125_p2", "pad125_p3")),
        ("pad 2.00", "breast_pad2_224.npz",
         _HERE.parents[0] / "16_external/results", ("brst_p2", "brst_p3")),
        ("pad 3.00", "breast_pad30_224.npz", _HERE / "results", ("pad30_p2", "pad30_p3")))


def load(names, tags, rdir):
    per = {}
    for ti, t in enumerate(tags):
        for f in sorted(Path(rdir).glob(f"{t}_*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                v = r["p_yes"] or {}
                if not v:
                    continue
                per.setdefault(int(r["index"]), {})[ti] = v
    out = {}
    for i, d in per.items():
        if len(d) != len(tags):
            continue
        M = np.full((len(names), len(tags)), 0.5)
        for ti, v in d.items():
            for j, nm in enumerate(names):
                if v.get(nm) is not None:
                    M[j, ti] = float(v[nm])
        out[i] = M
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    z = np.load(_DATA / "breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]
    y = (z["labels"][:, 0] == 0).astype(float)

    tables, cover = {}, {}
    for lab, _npz, rdir, tags in ARMS:
        p = load(names, tags, rdir)
        cover[lab] = len(p)
        if len(p) < 200:
            print(f"{lab}: only {len(p)}/252 probed in both arms -- skipping")
            continue
        idx = sorted(p)
        P = np.array([p[i] for i in idx])
        row = {}
        for j, f in enumerate(names):
            if f not in dn:
                continue
            col = D[idx, dn.index(f)]
            m = ~np.isnan(col)
            if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
                continue
            row[f] = {"P2": float(auc(col[m], P[m, j, 0])),
                      "P3": float(auc(col[m], P[m, j, 1])),
                      "pooled": float(auc(col[m], P[m, j].mean(1)))}
        row["_malignancy"] = {
            "P2": float(auc(y[idx], zscore(P[:, :, 0]) @ prior)),
            "P3": float(auc(y[idx], zscore(P[:, :, 1]) @ prior)),
            "pooled": float(auc(y[idx], zscore(P.mean(2)) @ prior))}
        tables[lab] = row

    if len(tables) < 2:
        print("need at least two framings; rerun when the probes finish")
        return

    labs = [l for l, *_ in ARMS if l in tables]
    keys = [f for f in names if f in tables[labs[0]]]
    for arm in ("P2", "P3"):
        print(f"\n=== descriptor agreement, {arm} phrasing ===")
        print(f"{'finding':46s} " + " ".join(f"{l:>10s}" for l in labs))
        for f in keys:
            print(f"{f:46s} " + " ".join(f"{tables[l][f][arm]:10.4f}" for l in labs))
        v = {l: np.array([tables[l][f][arm] for f in keys]) for l in labs}
        print(f"{'MEAN':46s} " + " ".join(f"{v[l].mean():10.4f}" for l in labs))

    print("\n=== malignancy through the zero-parameter KG rule (same 252 images) ===")
    print(f"{'arm':10s} " + " ".join(f"{l:>10s}" for l in labs))
    for arm in ("P2", "P3", "pooled"):
        print(f"{arm:10s} " + " ".join(f"{tables[l]['_malignancy'][arm]:10.4f}" for l in labs))

    print("\n=== the two specific failures ===")
    for f, why in (("irregular_shape", "negated arm inverts it"),
                   ("oval_shape", "negated arm inverts it"),
                   ("echogenic_rind", "both arms at chance"),
                   ("echogenic_pseudocapsule", "both arms at chance"),
                   ("thin_uniform_pseudocapsule", "both arms at chance")):
        if f not in keys:
            continue
        s = " ".join(f"{l} P2 {tables[l][f]['P2']:.3f}/P3 {tables[l][f]['P3']:.3f}"
                     for l in labs)
        print(f"  {f:30s} {why:24s} {s}")

    (_HERE / "results/crop_sweep.json").write_text(
        json.dumps({"coverage": cover, "tables": tables}, indent=1, default=float))
    print("\nwrote results/crop_sweep.json")


if __name__ == "__main__":
    main()
