"""Pick the scorer per finding, on descriptor agreement, then test the transfer.

Three scorers see the same 16 findings by mechanically different routes: the VLM
under a negated phrasing, the VLM under a verification phrasing, and BiomedCLIP
by caption contrast. compare_scorers.py showed each wins somewhere -- VLM P3 on 7
findings, VLM P2 on 4, CLIP on 2 -- so the question is whether choosing per
finding beats choosing once.

The selection is a categorical choice per finding made against BrEaST radiologist
descriptors: which scorer sees this finding best, and is any of them above chance
at all. A finding no scorer can see is dropped rather than kept at whatever the
KG says it should mean. Malignancy is never consulted, and BrEaST appears nowhere
in the evaluation.

Evaluated two ways, because they answer different questions:
  zero parameters   the KG-signed sum. Tests the ONTOLOGY on repaired inputs.
  fitted            weights frozen on BreastMNIST train. Tests whether the
                    repair survives a readout that can learn signs by itself.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "15_kggnn", "14_kgtensor", "20_perception"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc, bacc, load_probes, paired_bootstrap        # noqa: E402
from analyse_e2 import load as load_breast_probes                      # noqa: E402
from claims import kg_findings                                        # noqa: E402
from gates import L2S, cv_score, fit_ce, fit_rank                     # noqa: E402
from sign_audit import zscore                                         # noqa: E402

_DATA = _HERE.parents[1] / "data/external"
_RES = _HERE / "results"
SC = ("VLM P2", "VLM P3", "CLIP")


def breast_stack(names):
    p = load_breast_probes(names)
    idx = sorted(p)
    P = np.array([p[i] for i in idx])                       # n x F x 2
    C = np.load(_RES / "clip_breast_pad2.npz", allow_pickle=True)["scores"][idx]
    return idx, np.concatenate([P, C[:, :, None]], axis=2)  # n x F x 3


def busbra_stack(names):
    e = load_probes(names, ("busbra_p2", "busbra_p3"), _HERE.parents[0] / "16_external/results")
    idx = sorted(e)
    P = np.array([e[i] for i in idx])
    C = np.load(_RES / "clip_busbra_pad2.npz", allow_pickle=True)["scores"][idx]
    z = np.load(_DATA / "busbra_pad2_224.npz", allow_pickle=True)
    return (np.concatenate([P, C[:, :, None]], axis=2),
            (z["labels"][idx, 0] == 0).astype(float), z["cases"][idx])


def bm_stack(names):
    from freeze_models import breastmnist_matrix
    X, Y = breastmnist_matrix([(n, 0) for n in names])
    F = len(names)
    out = {}
    for s in ("train", "val", "test"):
        C = np.load(_RES / f"clip_bm_{s}.npz", allow_pickle=True)["scores"]
        P = np.stack([X[s][:, :F], X[s][:, F:]], 2)
        assert len(C) == len(P), f"{s}: clip {len(C)} vs probes {len(P)}"
        out[s] = np.concatenate([P, C[:, :, None]], axis=2)
    return out, Y


def select(names):
    """finding -> scorer index, decided on BrEaST descriptors alone."""
    idx, S = breast_stack(names)
    z = np.load(_DATA / "breast_pad2_224.npz", allow_pickle=True)
    dn = list(z["descriptor_names"]); D = z["descriptors"]
    pick = np.full(len(names), 1)          # default: verification arm
    live = np.ones(len(names), bool)
    table = {}
    for j, f in enumerate(names):
        if f not in dn:
            continue                        # unannotated: keep the default
        col = D[idx, dn.index(f)]
        m = ~np.isnan(col)
        if m.sum() < 30 or not 5 <= col[m].sum() < m.sum():
            continue
        a = [auc(col[m], S[m, j, t]) for t in range(3)]
        b = int(np.argmax(a))
        table[f] = {"aucs": [float(x) for x in a], "pick": SC[b], "kept": a[b] >= 0.5}
        pick[j] = b
        live[j] = a[b] >= 0.5
    return pick, live, table


def gather(S, pick):
    return S[np.arange(len(S))[:, None], np.arange(S.shape[1])[None, :], pick[None, :]]


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    pick, live, table = select(names)

    print("scorer selected per finding (on BrEaST descriptors only)")
    for f in names:
        t = table.get(f)
        print(f"  {f:46s} {SC[pick[names.index(f)]]:8s}"
              + ("" if t is None else f"  best {max(t['aucs']):.4f}"
                 + ("" if t["kept"] else "   DROPPED (no scorer above chance)")))
    print(f"\n  kept {int(live.sum())}/{len(names)} findings; "
          f"picks: {dict((SC[i], int((pick == i).sum())) for i in range(3))}")

    E, y, cases = busbra_stack(names)
    u = np.unique(cases)
    yc = np.array([y[cases == c][0] for c in u])
    def to_case(v):
        return np.array([v[cases == c].mean() for c in u])

    print("\n=== BUS-BRA, 1064 cases, ZERO fitted parameters (KG-signed sum) ===")
    arms = {"VLM P2 only": E[:, :, 0], "VLM P3 only": E[:, :, 1],
            "CLIP only": E[:, :, 2], "hybrid per finding": gather(E, pick)}
    zp = {}
    for lab, M in arms.items():
        w = prior * (live if lab == "hybrid per finding" else 1.0)
        zp[lab] = to_case(zscore(M) @ w)
        print(f"  {lab:22s} case AUC {auc(yc, zp[lab]):.4f}  "
              f"bAcc {bacc(yc, zp[lab], np.median(zp[lab])):.4f}")
    p = paired_bootstrap(yc, zp["hybrid per finding"], zp["VLM P3 only"])
    print(f"  hybrid vs VLM P3 only: P(better) {p:.3f}")

    print("\n=== the same arms with weights frozen on BreastMNIST train ===")
    B, Y = bm_stack(names)
    SPL = ("train", "val", "test")
    fit_res = {}
    for lab, sel in (("VLM P2 only", 0), ("VLM P3 only", 1), ("CLIP only", 2),
                     ("hybrid per finding", None)):
        if sel is None:
            Xb = {s: gather(B[s], pick)[:, live] for s in SPL}
            Xe = gather(E, pick)[:, live]
        else:
            Xb = {s: B[s][:, :, sel] for s in SPL}
            Xe = E[:, :, sel]
        mu, sd = Xb["train"].mean(0), Xb["train"].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = {s: (Xb[s] - mu) / sd for s in SPL}
        best = None
        for nm, f in (("ce", fit_ce), ("rank", fit_rank)):
            for l2 in L2S:
                c = cv_score(Z["train"], Y["train"], l2, f)[0]
                if best is None or c > best[0]:
                    best = (c, l2, f)
        _, l2, f = best
        w, b = f(Z["train"], Y["train"], l2)
        ind = auc(Y["test"], Z["test"] @ w + b)
        s_ext = to_case(((Xe - mu) / sd) @ w + b)
        fit_res[lab] = s_ext
        print(f"  {lab:22s} {Xb['train'].shape[1]:2d} params  BreastMNIST {ind:.4f}  "
              f"BUS-BRA case {auc(yc, s_ext):.4f}  bAcc {bacc(yc, s_ext, np.median(s_ext)):.4f}"
              f"  gap {ind - auc(yc, s_ext):+.4f}")
    pf = paired_bootstrap(yc, fit_res["hybrid per finding"], fit_res["VLM P3 only"])
    print(f"  hybrid vs VLM P3 only: P(better) {pf:.3f}")

    (_HERE / "results/hybrid.json").write_text(json.dumps(
        {"selection": table, "kept": int(live.sum()),
         "zero_param": {k: float(auc(yc, v)) for k, v in zp.items()},
         "P_hybrid_zero": float(p),
         "fitted": {k: float(auc(yc, v)) for k, v in fit_res.items()},
         "P_hybrid_fitted": float(pf)}, indent=1, default=float))
    print("\nwrote results/hybrid.json")


if __name__ == "__main__":
    main()
