"""Recompute every KG-in-prompt variant from round 03 straight from its jsonl.

Nothing here is quoted from a note. AUC uses the tie-corrected rank estimator
(midranks), which matters because several of these variants saturate p_malignant
at the extremes and produce large tie blocks.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_H = Path(__file__).resolve().parent
sys.path.insert(0, str(_H.parents[0] / "16_external"))
from analyse_e1 import auc, bacc                                       # noqa: E402

R = _H.parents[0] / "03_kg_grounded_vlm/results"


def read(p):
    y, s, ties = [], [], 0
    for ln in Path(p).read_text(errors="replace").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "p_malignant" not in r or r["p_malignant"] is None:
            continue
        y.append(1.0 if r["gold"] == "MALIGNANT" else 0.0)
        s.append(float(r["p_malignant"]))
    y, s = np.array(y), np.array(s)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    u, c = np.unique(s, return_counts=True)
    return {"n": int(len(y)), "n_mal": int(y.sum()), "auc": float(auc(y, s)),
            "bacc": float(bacc(y, s, float(np.median(s)))),
            "n_unique_scores": int(len(u)), "largest_tie_block": int(c.max()),
            "frac_in_ties": float((c[c > 1].sum()) / len(s)) if (c > 1).any() else 0.0}


def main():
    out = {}
    for p in sorted(R.glob("*.jsonl")):
        r = read(p)
        if r:
            out[p.name] = r
    print(f"{'variant':46s} {'n':>4s} {'AUC':>7s} {'bAcc':>7s} {'uniq':>6s} {'maxtie':>7s} {'%tied':>6s}")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"{k[:46]:46s} {v['n']:4d} {v['auc']:7.4f} {v['bacc']:7.4f} "
              f"{v['n_unique_scores']:6d} {v['largest_tie_block']:7d} "
              f"{100*v['frac_in_ties']:5.1f}%")
    (_H / "results/r03_variants.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(out)} variants recomputed -> results/r03_variants.json")


if __name__ == "__main__":
    main()
