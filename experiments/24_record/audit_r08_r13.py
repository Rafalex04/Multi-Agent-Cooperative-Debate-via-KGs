"""Recompute the two-run contrastive scoring family (round 08) and BI-RADS-2
(round 13) from their jsonl records. Both are two-VLM-call-per-sample designs
whose headline numbers were never written to a summary file."""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
_H = Path(__file__).resolve().parent
sys.path.insert(0, str(_H.parents[0] / "16_external"))
from analyse_e1 import auc, bacc                                       # noqa: E402


def read(pat, score_keys):
    y, S = [], {k: [] for k in score_keys}
    for f in sorted(glob.glob(pat)):
        for ln in Path(f).read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if not all(k in r and r[k] is not None for k in score_keys):
                continue
            y.append(1.0 if r["gold"] == "MALIGNANT" else 0.0)
            for k in score_keys:
                S[k].append(float(r[k]))
    return np.array(y), {k: np.array(v) for k, v in S.items()}


def report(name, y, S):
    print(f"\n{name}   n={len(y)}  malignant={int(y.sum())}")
    for k, s in S.items():
        u = len(np.unique(s))
        print(f"    {k:22s} AUC {auc(y, s):.4f}   bAcc {bacc(y, s, float(np.median(s))):.4f}"
              f"   unique scores {u}/{len(s)}")
    return {k: {"auc": float(auc(y, s)), "bacc": float(bacc(y, s, float(np.median(s)))),
                "n": int(len(y)), "unique": int(len(np.unique(s)))} for k, s in S.items()}


def main():
    out = {}
    R = _H.parents[0]
    y, S = read(str(R / "08_contrastive/results/contrastive_test_*.jsonl"),
                ["score_verdict", "score_match"])
    out["contrastive (verdict / match)"] = report("08 — two-run contrastive", y, S)
    y, S = read(str(R / "08_contrastive/results/scored_test_*.jsonl"), ["score"])
    out["contrastive scored (0-10 expectation)"] = report("08 — scored variant", y, S)
    y, S = read(str(R / "08_contrastive/results/score10_test_*.jsonl"), ["score"])
    out["contrastive score10"] = report("08 — score10 variant", y, S)

    # 13 BI-RADS-2: find the score field present in the records
    f = sorted(glob.glob(str(R / "13_birads2/results/birads2_test_*.jsonl")))[0]
    rec = json.loads(Path(f).read_text().splitlines()[0])
    print("\n13 — BI-RADS-2 record keys:", list(rec))
    keys = [k for k in rec if isinstance(rec[k], (int, float)) and k not in ("index", "time_s")]
    y, S = read(str(R / "13_birads2/results/birads2_test_*.jsonl"), keys)
    out["birads2"] = report("13 — BI-RADS-2 scale", y, S)
    (_H / "results/r08_r13.json").write_text(json.dumps(out, indent=1))
    print("\nwrote results/r08_r13.json")


if __name__ == "__main__":
    main()
