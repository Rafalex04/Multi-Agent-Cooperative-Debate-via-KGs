"""Does the lesion channel transfer? BreastMNIST-frozen, applied to BUS-BRA.

EXPLORATORY, and labelled as such. The lesion channel did not exist when the
Round 3 pre-registration was written and hashed, so it is not one of the
registered endpoints and cannot be reported as one. What it can do is answer the
obvious follow-up: the channel is worth +0.028 in domain at +2.82 sd against a
matched control -- does any of that survive a change of country and scanner?

The protocol mirrors E1 exactly so the numbers are comparable to it. Weights are
fitted on BreastMNIST train alone, the standardisation comes from BreastMNIST
train, the bAcc threshold is swept on train+val, and nothing is refitted on
BUS-BRA. Case-level aggregation is the primary view, as in E1.
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
from analyse_e1 import auc, bacc, load_probes, paired_bootstrap        # noqa: E402
from claims import kg_findings                                         # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                      # noqa: E402
from lesions import lesion_table                                       # noqa: E402

_EXT = _HERE.parents[0] / "16_external/results"


def lesion_probe_table(tag, ents, results_dir):
    out = {}
    for f in sorted(Path(results_dir).glob(f"{tag}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            v = r["p_yes"] or {}
            out[int(r["index"])] = np.array(
                [float(v[e]) if v.get(e) is not None else 0.5 for e in ents])
    return out


def bm_lesion(ents):
    tab = {}
    for f in sorted((_HERE / "results").glob("lesion_*.jsonl")):
        split = f.name.split("_")[1]
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            v = r["p_yes"] or {}
            tab[(split, int(r["index"]))] = (
                np.array([float(v[e]) if v.get(e) is not None else 0.5 for e in ents]),
                1.0 if r["gold"] == "MALIGNANT" else 0.0)
    return tab


def main():
    F = json.loads((_EXT / "frozen_models.json").read_text())
    names = F["findings"]
    L = lesion_table()
    ents = [e for e, *_ in L]
    prior_les = np.array([p for _, _, p, _, _ in L])

    bus_les = lesion_probe_table("leslbus", ents, _HERE / "results")
    print(f"BUS-BRA lesion probes: {len(bus_les)}/1875")
    if len(bus_les) < 400:
        print("not far enough along yet"); return

    # ---- BreastMNIST side: build the combined feature matrix ----------------
    from freeze_models import breastmnist_matrix
    Xf, Yb = breastmnist_matrix([(n, 0) for n in names])
    tabL = bm_lesion(ents)
    SP = ("train", "val", "test")
    keys = {s: sorted(k for k in tabL if k[0] == s) for s in SP}
    XL = {s: np.array([tabL[k][0] for k in keys[s]]) for s in SP}
    idxs = {s: [k[1] for k in keys[s]] for s in SP}
    Xfa = {s: Xf[s][idxs[s]] for s in SP}
    Y = {s: Yb[s][idxs[s]] for s in SP}
    print("BreastMNIST aligned: " + "  ".join(f"{s}={len(Y[s])}" for s in SP))

    def fit_and_transfer(Xb_, Xe_, label, res, y_ext, cases):
        mu, sd = Xb_["train"].mean(0), Xb_["train"].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = {s: (Xb_[s] - mu) / sd for s in SP}
        best = None
        for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
            for l2 in L2S:
                c = cv_score(Z["train"], Y["train"], l2, f)[0]
                if best is None or c > best[0]:
                    best = (c, l2, nm, f)
        cv, l2, nm, f = best
        w, b = f(Z["train"], Y["train"], l2)
        sc = {s: Z[s] @ w + b for s in SP}
        fv = np.concatenate([sc["train"], sc["val"]])
        gv = np.concatenate([Y["train"], Y["val"]])
        thr = max(sorted(set(fv.tolist())), key=lambda t: bacc(fv.tolist(), gv.tolist(), t))
        ind = auc(Y["test"], sc["test"])
        se = ((Xe_ - mu) / sd) @ w + b
        u = np.unique(cases)
        sec = np.array([se[cases == c].mean() for c in u])
        yc = np.array([y_ext[cases == c][0] for c in u])
        a_i, a_c = auc(y_ext, se), auc(yc, sec)
        print(f"  {label:26s} in-domain {ind:.4f}  BUS-BRA image {a_i:.4f}  "
              f"case {a_c:.4f}  bAcc {bacc(yc, sec, thr):.4f}  gap {ind-a_c:+.4f}")
        res[label] = {"indomain": ind, "ext_image": a_i, "ext_case": a_c,
                      "ext_bacc_case": bacc(yc, sec, thr), "gap": ind - a_c}
        return sec, yc

    # ---- BUS-BRA side -------------------------------------------------------
    fprobes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(fprobes) & set(bus_les))
    print(f"BUS-BRA with BOTH channels: {len(common)}")
    z = np.load(_HERE.parents[1] / "data/external/busbra_pad2_224.npz")
    y_ext = (z["labels"][:, 0] == 0).astype(int)[common]
    cases = z["cases"][common]
    XeF = np.array([fprobes[i].T.reshape(-1) for i in common])
    XeL = np.array([bus_les[i] for i in common])

    print("\nEXPLORATORY -- not a registered endpoint\n")
    res = {"n_ext": len(common), "n_cases": int(len(np.unique(cases)))}
    sf, yc = fit_and_transfer(Xfa, XeF, "findings only", res, y_ext, cases)
    sl, _ = fit_and_transfer(XL, XeL, "lesions only", res, y_ext, cases)
    sb, _ = fit_and_transfer({s: np.hstack([Xfa[s], XL[s]]) for s in SP},
                             np.hstack([XeF, XeL]), "findings + lesions", res, y_ext, cases)
    p = paired_bootstrap(yc, sb, sf)
    print(f"\n  combined vs findings-only, external case level: "
          f"{res['findings + lesions']['ext_case'] - res['findings only']['ext_case']:+.4f}"
          f"   P(better) {p:.3f}")
    res["P_combined_better"] = p
    (_HERE / "results/external_lesion.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote results/external_lesion.json")


if __name__ == "__main__":
    main()
