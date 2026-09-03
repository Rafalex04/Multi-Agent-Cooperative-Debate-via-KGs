"""Cross-method paired bootstraps on the BUS-BRA external set.

Every comparison is PAIRED on the same patients, resampled 4000 times as the
protocol specifies. Our rows and the baseline rows are produced by different
scripts, so the two score files are aligned on case id rather than assumed to
be in the same order -- a silent misalignment here would compare a method
against a shuffled copy of its opponent and look like a real effect.

Agent counts differ and are printed with every B2 row: B2 GraphGeo runs SIX
agents and 10,737 fitted parameters against ThothGNN v3's TWO agents and 59.
That confound runs in B2's favour and is not corrected for.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc, paired_bootstrap            # noqa: E402

HET = _HERE.parents[0] / "17_hetgnn/results"
AGENTS = {"B2 GraphGeo (ours impl, no KG)": "6 agents, 10737 prm",
          "B2 GraphGeo + our anchor": "6 agents, 10738 prm",
          "B1 Catfish Agent (ours impl)": "2+1 agents, 0 prm",
          "GNN+KG": "2 agents, 59 prm (OURS)",
          "probes_only": "0 agents, 64 prm (OURS)",
          "probes_+_debate": "2 agents, 64 prm (OURS)"}


def load(arm):
    b = np.load(_HERE / f"results/external_{arm}.scores.npz", allow_pickle=True)
    o = np.load(HET / f"em_{'full' if arm == 'auto' else 'mask'}_both.scores.npz",
                allow_pickle=True)
    # align on case id
    bc, oc = b["_cases_b2"], o["cases"]
    common = np.intersect1d(bc, oc)
    bi = {c: k for k, c in enumerate(bc)}
    oi = {c: k for k, c in enumerate(oc)}
    bsel = np.array([bi[c] for c in common])
    osel = np.array([oi[c] for c in common])
    y = np.asarray(o["y_case"])[osel]
    yb = np.asarray(b["_y_case_b2"])[bsel]
    assert np.array_equal(y, yb), "label mismatch after case alignment"
    S = {}
    for k in b.files:
        if not k.startswith("_"):
            S[k] = np.asarray(b[k])[bsel]
    for k in o.files:
        if k not in ("y_case", "cases"):
            S[k] = np.asarray(o[k])[osel]
    return y, S, len(common)


def main():
    for arm in ("auto", "mask"):
        try:
            y, S, n = load(arm)
        except FileNotFoundError as e:
            print(f"\n=== {arm}: score file missing ({e.filename}) ==="); continue
        print(f"\n=== BUS-BRA external, {arm} condition | {n} patients, "
              f"malignant {y.mean():.3f} ===")
        for k in sorted(S):
            print(f"  {k:34s} AUC {auc(y, S[k]):.4f}   {AGENTS.get(k,'')}")
        ours = "GNN+KG"
        print(f"\n  paired bootstrap, 4000 resamples, vs ThothGNN v3 ({ours}):")
        for k in sorted(S):
            if k == ours:
                continue
            d = auc(y, S[k]) - auc(y, S[ours])
            p = paired_bootstrap(y, S[k], S[ours])
            verdict = ("BEATS ours" if p >= 0.95 else
                       "loses to ours" if p <= 0.05 else "indistinguishable")
            print(f"    {k:34s} {d:+.4f}  P(better than ours)={p:.3f}  {verdict}")


if __name__ == "__main__":
    main()
