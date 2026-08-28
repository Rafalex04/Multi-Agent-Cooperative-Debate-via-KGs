"""Primary analysis, exactly as pre-registered on 2026-08-27.

Zero-parameter KG-signed sum, case-level AUC over patients, paired bootstrap over
patients (4000 resamples) against the no-mask arm, decision bar P(better) >= 0.95.
Nothing is fitted anywhere in this file.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor", "20_perception"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc, bacc, paired_bootstrap                     # noqa: E402
from claims import kg_findings                                        # noqa: E402

ARMS = (("bbase", "no mask (2.0x box crop)"),
        ("bdim", "background attenuated to 35%"),
        ("bring", "2px boundary contour"))
HALO = ("echogenic_rind", "echogenic_pseudocapsule", "thin_uniform_pseudocapsule")


def load(tag, names):
    out = {}
    for f in sorted((_HERE / "results").glob(f"{tag}_p3_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r["p_yes"] or {}
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def boot_ci(y, s, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)

    P = {a: load(a, names) for a, _ in ARMS}
    common = sorted(set.intersection(*[set(P[a]) for a, _ in ARMS]))
    z = np.load(_HERE.parents[1] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    y = (z["labels"][common, 0] == 0).astype(float)
    cases = z["cases"][common]
    u = np.unique(cases)
    yc = np.array([y[cases == c][0] for c in u])
    print(f"n = {len(common)} images / {len(u)} patients   "
          f"malignant {int(yc.sum())} / {len(yc)} patients\n")

    S, res = {}, {"n_images": len(common), "n_patients": int(len(u)), "arms": {}}
    for a, desc in ARMS:
        M = np.array([P[a][i] for i in common])
        Z = (M - M.mean(0)) / np.where(M.std(0) < 1e-9, 1, M.std(0))
        s = Z @ prior
        sc = np.array([s[cases == c].mean() for c in u])
        S[a] = sc
        lo, hi = boot_ci(yc, sc)
        res["arms"][a] = {"desc": desc, "auc": auc(yc, sc), "ci": [lo, hi],
                          "bacc": bacc(yc, sc, np.median(sc))}

    print("PRIMARY — zero-parameter KG-signed sum, patient level")
    print(f"{'arm':32s} {'AUC':>7s} {'95% CI':>18s} {'bAcc':>7s} {'vs base':>9s} {'P(better)':>10s}")
    for a, desc in ARMS:
        r = res["arms"][a]
        d = "" if a == "bbase" else f"{r['auc']-res['arms']['bbase']['auc']:+9.4f}"
        p = "" if a == "bbase" else f"{paired_bootstrap(yc, S[a], S['bbase'], n=4000):10.3f}"
        if a != "bbase":
            res["arms"][a]["delta"] = r["auc"] - res["arms"]["bbase"]["auc"]
            res["arms"][a]["P_better"] = paired_bootstrap(yc, S[a], S["bbase"], n=4000)
        print(f"{desc:32s} {r['auc']:7.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] "
              f"{r['bacc']:7.4f} {d:>9s} {p:>10s}")

    print("\nDECISION (bar P >= 0.95)")
    for a, desc in ARMS[1:]:
        p = res["arms"][a]["P_better"]
        print(f"  {a:6s} {'CLEARS' if p >= 0.95 else 'DOES NOT CLEAR'} the bar  (P={p:.3f})")

    print("\nPER-FINDING probe->malignancy AUC (image level)")
    print(f"{'finding':46s}" + "".join(f"{a:>9s}" for a, _ in ARMS))
    pf = {}
    for j, f in enumerate(names):
        row = {}
        for a, _ in ARMS:
            M = np.array([P[a][i] for i in common])
            row[a] = auc(y, M[:, j])
        pf[f] = row
        mark = "  <- Halo" if f in HALO else ""
        print(f"{f:46s}" + "".join(f"{row[a]:9.4f}" for a, _ in ARMS) + mark)
    res["per_finding"] = pf

    print("\nCONFIRMATION 1 — Halo group (weaker test: BUS-BRA has no descriptor labels,")
    print("so this is probe->malignancy AUC, not whether the probe sees the finding)")
    for f in HALO:
        r = pf[f]
        print(f"  {f:30s} base {r['bbase']:.4f}   "
              f"dim {r['bdim']-r['bbase']:+.4f}   ring {r['bring']-r['bbase']:+.4f}")
    print(f"  {'MEAN of the three':30s} base "
          f"{np.mean([pf[f]['bbase'] for f in HALO]):.4f}   "
          f"dim {np.mean([pf[f]['bdim']-pf[f]['bbase'] for f in HALO]):+.4f}   "
          f"ring {np.mean([pf[f]['bring']-pf[f]['bbase'] for f in HALO]):+.4f}")

    (_HERE / "results/primary.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/primary.json")


if __name__ == "__main__":
    main()
