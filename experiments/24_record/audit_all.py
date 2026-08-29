"""Read every committed result file and emit one consolidated, traceable table.

Nothing is transcribed from a note. Each row records the file it came from so a
reader can check it. Keys differ between rounds (`test_auc` vs `test`), so the
extractor is explicit about which key it used.
"""
from __future__ import annotations
import json, glob, subprocess
from pathlib import Path

_H = Path(__file__).resolve().parent
_R = _H.parents[1]

AUC_KEYS = ("test_auc", "test", "auc", "ext_case", "indomain")
BACC_KEYS = ("test_bacc", "bacc", "ext_bacc_case", "balanced_accuracy")


def pick(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return k, float(d[k])
    return None, None


def commit_of(p):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(p)],
                              cwd=_R, capture_output=True, text=True,
                              timeout=20).stdout.strip() or "UNCOMMITTED"
    except Exception:
        return "?"


def main():
    files = sorted(glob.glob(str(_R / "experiments/*/results/*.json"))
                   + glob.glob(str(_R / "experiments/*/*.json")))
    out = {}
    for f in files:
        rel = str(Path(f).relative_to(_R))
        if "/24_record/" in rel:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rows = {}
        for arm, v in d.items():
            if not isinstance(v, dict):
                continue
            ka, a = pick(v, AUC_KEYS)
            kb, b = pick(v, BACC_KEYS)
            if a is None:
                continue
            rows[arm] = {"auc": a, "auc_key": ka, "bacc": b, "bacc_key": kb,
                         "sd": v.get("test_sd"), "n": v.get("n") or v.get("n_cases")}
        if rows:
            out[rel] = {"commit": commit_of(rel), "arms": rows}

    for rel, blk in out.items():
        print(f"\n### {rel}   [{blk['commit']}]")
        for arm, r in blk["arms"].items():
            sd = f" ± {r['sd']:.4f}" if isinstance(r.get("sd"), (int, float)) else ""
            bb = f"{r['bacc']:.4f}" if r["bacc"] is not None else "  --  "
            print(f"    {arm[:44]:44s} AUC {r['auc']:.4f}{sd:12s} bAcc {bb}")
    (_H / "results/audit_all.json").write_text(json.dumps(out, indent=1))
    n = sum(len(b["arms"]) for b in out.values())
    print(f"\n{n} arms across {len(out)} files -> results/audit_all.json")


if __name__ == "__main__":
    main()
