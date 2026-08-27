"""Debate only what is uncertain -- tested at FINDING level on the existing corpus.

The proposal: restrict the debate to findings the probes are unsure about, so the
debate becomes a targeted second opinion instead of a redundant first one. The
earlier uncertainty gate (S3) fired with the sign reversed, but that was at SAMPLE
level on the whole debate score; finding level is a different quantity.

Two things are being asked and only one of them needs new debates.

  Does the debate's contribution CONCENTRATE at uncertain findings?
      Answerable now. The corpus already records which finding every claim cites,
      so the debate channel can be gated after the fact: keep mass, net stance and
      contestation at the k most uncertain findings of each sample, zero them
      everywhere else, and see whether the gated channel beats the ungated one.
      This is the necessary condition. If the debate is no more useful where the
      probes are unsure than where they are confident, then telling the agents to
      argue about uncertain findings cannot help either.

  Would DIRECTING the agents at those findings produce better arguments?
      Not answerable without regenerating the corpus, and only worth the compute
      if the first question comes back positive.

Two uncertainty measures, because the obvious one is now compromised. "P2 and P3
disagree" was the natural choice, but the negated arm agrees with the radiologist
at 0.4882 -- chance -- so disagreement between the arms is substantially just P2
being wrong. `spread` is kept as the registered form of the proposal; `near5` is
the version that does not depend on a broken channel.

Run on the complete BUS-BRA corpus: 1875 debates, 1064 cases, folds split on case.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _d in ("14_kgtensor", "15_kggnn", "16_external"):
    sys.path.insert(0, str(_HERE.parents[1] / _d))
from analyse_e1 import auc, load_probes, paired_bootstrap                # noqa: E402
from claims import kg_findings                                           # noqa: E402
from gates import fit_rank                                               # noqa: E402
from hetgraph import load_sample                                         # noqa: E402
from powered_gnn import case_folds                                       # noqa: E402

_DEB = _HERE.parents[2] / "breastMnist/data/breast/debates_busbra/all"
_EXT = _HERE.parents[1] / "16_external/results"
KS = (2, 4, 8, 16)


def build():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    F = len(names)

    deb = {}
    for f in sorted(glob.glob(str(_DEB / "*.json"))):
        try:
            X, Acc, Acf, mask, y, sid = load_sample(f, names)
        except Exception:
            continue
        deb[int(sid)] = (X, Acf, mask)
    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(deb) & set(probes))
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    y = (z["labels"][common, 0] == 0).astype(float)
    cases = z["cases"][common]
    n = len(common)

    P = np.zeros((n, F, 2)); MASS = np.zeros((n, F))
    NET = np.zeros((n, F)); CON = np.zeros((n, F))
    for k, i in enumerate(common):
        Xc, Acf, m = deb[i]
        P[k] = probes[i]
        MASS[k] = (m[:, None] * Acf).sum(0)
        NET[k] = ((m * Xc[:, 0])[:, None] * Acf).sum(0)
        npos = ((m * (Xc[:, 0] > 0))[:, None] * Acf).sum(0)
        nneg = ((m * (Xc[:, 0] < 0))[:, None] * Acf).sum(0)
        tot = npos + nneg
        q = np.divide(npos, tot, out=np.zeros(F), where=tot > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        CON[k] = np.where((tot > 0) & (q > 0) & (q < 1), h, 0.0)
    return names, prior, P, MASS, NET, CON, y, cases, n


def topk_mask(U, k):
    """Boolean n x F, True at the k most uncertain findings of each sample."""
    idx = np.argsort(-U, axis=1)[:, :k]
    M = np.zeros(U.shape, bool)
    np.put_along_axis(M, idx, True, axis=1)
    return M


def zs(A):
    s = A.std(0)
    return (A - A.mean(0)) / np.where(s < 1e-9, 1.0, s)


def main():
    names, prior, P, MASS, NET, CON, y, cases, n = build()
    F = len(names)
    u = np.unique(cases)
    yc = np.array([y[cases == c][0] for c in u])
    to_case = lambda v: np.array([v[cases == c].mean() for c in u])
    folds = case_folds(cases, k=5, seed=0)
    print(f"complete corpus: {n} images / {len(u)} cases")

    U = {"near5": -np.abs(P[:, :, 1] - 0.5),          # verification arm near 0.5
         "spread": np.abs(P[:, :, 0] - P[:, :, 1])}   # the arms disagree

    def oof(cols):
        o = np.zeros(n)
        for te in folds:
            tr = ~te
            w, b = fit_rank(cols[tr], y[tr], 0.3)
            o[te] = cols[te] @ w + b
        return o

    Zp = zs(P.reshape(n, -1))
    Zprior = np.tile(prior, (n, 1))
    base = np.hstack([Zp, Zprior])
    full = np.hstack([Zp, Zprior, zs(MASS), zs(NET), zs(CON)])

    res = {"n": n, "n_cases": int(len(u))}
    a_base = auc(yc, to_case(oof(base)))
    a_full = auc(yc, to_case(oof(full)))
    print(f"\n{'arm':34s} {'case AUC':>9s} {'vs probes':>10s} {'P(better)':>10s}")
    print(f"{'probes only':34s} {a_base:9.4f}")
    s_base = to_case(oof(base))
    s_full = to_case(oof(full))
    print(f"{'+ debate, all 16 findings':34s} {a_full:9.4f} {a_full - a_base:+10.4f} "
          f"{paired_bootstrap(yc, s_full, s_base):10.3f}")
    res["probes_only"] = a_base
    res["debate_all"] = a_full
    res["gated"] = {}

    for lab, Uv in U.items():
        print(f"\n  gate = {lab}")
        res["gated"][lab] = {}
        for k in KS:
            G = topk_mask(Uv, k).astype(float)
            cols = np.hstack([Zp, Zprior, zs(MASS * G), zs(NET * G), zs(CON * G)])
            s = to_case(oof(cols))
            a = auc(yc, s)
            print(f"    debate at the {k:2d} most uncertain findings   {a:.4f}"
                  f"   {a - a_base:+.4f} vs probes   {a - a_full:+.4f} vs ungated"
                  f"   P {paired_bootstrap(yc, s, s_base):.3f}")
            res["gated"][lab][k] = {"auc": a, "vs_probes": a - a_base,
                                    "vs_ungated": a - a_full}

    # ---- is the debate channel itself any better where the probes are unsure? --
    print("\n=== does the debate's own signal concentrate at uncertain findings? ===")
    print("    (AUC of KG-signed net stance / mass, restricted to the gated findings)")
    for lab, Uv in U.items():
        for k in (4, 8):
            G = topk_mask(Uv, k).astype(float)
            an = auc(yc, to_case((NET * G) @ prior))
            am = auc(yc, to_case((MASS * G) @ prior))
            print(f"    {lab:7s} k={k:2d}   net stance {an:.4f}   mass {am:.4f}")
    print(f"    {'ALL 16':7s}        net stance {auc(yc, to_case(NET @ prior)):.4f}   "
          f"mass {auc(yc, to_case(MASS @ prior)):.4f}")

    (_HERE.parent / "results/uncertainty_gate.json").write_text(
        json.dumps(res, indent=1, default=float))
    print("\nwrote results/uncertainty_gate.json")


if __name__ == "__main__":
    main()
