"""Two mechanically different scorers on the same 16 findings, judged on whether
they SEE the finding rather than on whether they predict cancer.

BrEaST is the only dataset here that annotates the descriptors, so it is the only
place this comparison can be made. For each finding:

    VLM P3    first-token logprob p(yes) under the verification phrasing
    VLM P2    the same under the negated phrasing
    CLIP      cos(image, "...with <finding>") - cos(image, "...without <finding>")

MedCBR's hypothesis is specific and testable here: web-scale contrastive encoders
decline on ECHOGENICITY and POSTERIOR features in ultrasound, which depend on
imaging physics that figure-caption corpora under-represent, while GEOMETRY
survives. If that holds, the right architecture is not "pick a scorer" but "pick
a scorer per finding" -- and this table says which, using descriptor agreement
and never a malignancy label.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor", "20_perception"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc                                             # noqa: E402
from analyse_e2 import load as load_breast_probes                      # noqa: E402
from claims import kg_findings                                         # noqa: E402

# Which BI-RADS family each finding belongs to. MedCBR's claim is about these
# families, so the test needs them named in advance rather than read off the
# result.
FAMILY = {
    "irregular_shape": "geometry", "oval_shape": "geometry",
    "non_parallel_orientation": "geometry", "parallel_orientation": "geometry",
    "spiculated": "margin", "circumscribed_margin": "margin",
    "spiculated_or_irregular_mass": "margin", "architectural_distortion": "margin",
    "echogenic_rind": "margin", "echogenic_pseudocapsule": "margin",
    "thin_uniform_pseudocapsule": "margin",
    "anechoic_content": "echogenicity", "hyperechoic_mass": "echogenicity",
    "clustered_microcysts": "echogenicity",
    "microcalcifications_in_hypoechoic_mass": "echogenicity",
    "posterior_shadowing_with_solid_irregular_mass": "posterior",
}


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    z = np.load(_HERE.parents[1] / "data/external/breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]

    probes = load_breast_probes(names)
    idx = sorted(probes)
    P = np.array([probes[i] for i in idx])                 # n x F x 2
    C = np.load(_HERE / "results/clip_breast_pad2.npz", allow_pickle=True)
    assert list(C["findings"]) == names, "finding order mismatch"
    S = C["scores"][idx]

    print(f"BrEaST, n={len(idx)}: agreement with the radiologist's own descriptor")
    print(f"{'finding':46s} {'family':13s} {'VLM P2':>8s} {'VLM P3':>8s} {'CLIP':>8s}  best")
    rows, wins = {}, {}
    for j, f in enumerate(names):
        if f not in dn:
            continue
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
            continue
        a = {"VLM P2": auc(col[m], P[m, j, 0]), "VLM P3": auc(col[m], P[m, j, 1]),
             "CLIP": auc(col[m], S[m, j])}
        b = max(a, key=lambda k: a[k])
        rows[f] = {"family": FAMILY[f], **{k: float(v) for k, v in a.items()}, "best": b}
        wins[b] = wins.get(b, 0) + 1
        print(f"{f:46s} {FAMILY[f]:13s} {a['VLM P2']:8.4f} {a['VLM P3']:8.4f} "
              f"{a['CLIP']:8.4f}  {b}")

    print(f"\nper-scorer means over the {len(rows)} annotated findings:")
    for k in ("VLM P2", "VLM P3", "CLIP"):
        v = np.array([r[k] for r in rows.values()])
        print(f"  {k:8s} {v.mean():.4f}   above chance on {(v > .5).sum()}/{len(v)}")
    print(f"  wins: {wins}")

    print("\nMedCBR's hypothesis -- CLIP declines on echogenicity and posterior, "
          "holds on geometry:")
    print(f"{'family':14s} {'n':>3s} {'VLM P3':>8s} {'CLIP':>8s} {'CLIP - VLM P3':>14s}")
    fam = {}
    for f, r in rows.items():
        fam.setdefault(r["family"], []).append((r["VLM P3"], r["CLIP"]))
    for k in ("geometry", "margin", "echogenicity", "posterior"):
        if k not in fam:
            continue
        v = np.array(fam[k])
        print(f"{k:14s} {len(v):3d} {v[:,0].mean():8.4f} {v[:,1].mean():8.4f} "
              f"{v[:,1].mean()-v[:,0].mean():+14.4f}")
        fam[k] = {"n": len(v), "vlm_p3": float(v[:, 0].mean()), "clip": float(v[:, 1].mean())}

    (_HERE / "results/compare_scorers.json").write_text(
        json.dumps({"per_finding": rows, "by_family": fam, "wins": wins},
                   indent=1, default=float))
    print("\nwrote results/compare_scorers.json")


if __name__ == "__main__":
    main()
