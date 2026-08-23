"""Integrity gate for probe corpora. Run before any analysis reads them.

Written after two nodes served HTTP 500 for every request for roughly two hours.
The runner logged a warning per failed probe and then wrote the record anyway,
so 452 rows entered the corpus with every value null. Those rows are worse than
missing rows in three separate ways: they look like data, resume-by-index treats
them as done so they are never retried, and the feature loader fills them with
0.5 defaults, which is 452 samples of pure noise silently averaged into an
external validation result.

Checks, in order of how badly each one would corrupt a conclusion:
  all-null records     a node that was not serving
  unparseable lines    interleaved appends from two writers on one file
  duplicate indices    two owners for one shard-group
  sparse records       partial failures worth knowing about
"""
from __future__ import annotations

import json, sys
from collections import Counter
from pathlib import Path


def validate(results_dir, patterns):
    ok = True
    for pat in patterns:
        files = sorted(Path(results_dir).glob(pat))
        if not files:
            print(f"  {pat:24s} NO FILES")
            continue
        tot = bad = nulls = sparse = 0
        idx = Counter()
        worst = []
        for f in files:
            fn = ftot = fnull = 0
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                tot += 1; ftot += 1
                try:
                    r = json.loads(ln)
                except Exception:
                    bad += 1
                    continue
                idx[(f.name, r["index"])] += 1
                vals = r.get("p_yes") or {}
                got = sum(1 for v in vals.values() if v is not None)
                if got == 0:
                    nulls += 1; fnull += 1
                elif got < len(vals):
                    sparse += 1
            if ftot and fnull / ftot > 0.5:
                worst.append(f"{f.name} ({fnull}/{ftot} null)")
        dups = sum(1 for v in idx.values() if v > 1)
        status = "OK" if (bad == 0 and nulls == 0 and dups == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {pat:24s} {tot:5d} recs  null={nulls:4d}  unparseable={bad:3d}  "
              f"dup={dups:3d}  partial={sparse:4d}  {status}")
        for w in worst:
            print(f"      -> {w}")
    return ok


if __name__ == "__main__":
    d = Path(__file__).resolve().parent / "results"
    print("probe corpus integrity")
    good = validate(d, ["busbra_p2_*.jsonl", "busbra_p3_*.jsonl",
                        "brst_p2_*.jsonl", "brst_p3_*.jsonl"])
    print("\n" + ("all corpora clean" if good else
                  "CORRUPT RECORDS PRESENT -- do not analyse until repaired"))
    sys.exit(0 if good else 1)
