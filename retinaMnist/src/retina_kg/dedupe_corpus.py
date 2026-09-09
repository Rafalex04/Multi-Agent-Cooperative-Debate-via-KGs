"""Remove duplicate sample_id records from a JSONL corpus, keeping the first.

Needed because a sweeper that re-issues a shard already in flight makes both
copies read `done_indices` before either writes, so both process the same
indices. `corpus_io.load_corpus` sorts by sample_id but does NOT deduplicate, so
a duplicated record is silently double-counted by every downstream analysis.

Rewrites in place only after verifying that unique coverage is unchanged.
"""
from __future__ import annotations
import argparse, glob, json, shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def dedupe(root, splits=SPLITS, apply=False):
    root = Path(root)
    report = []
    for sp in splits:
        files = sorted(glob.glob(str(root / f"{sp}_*.jsonl")))
        if not files:
            continue
        seen, before, kept = set(), 0, {f: [] for f in files}
        for f in files:                       # deterministic: first file wins
            for ln in open(f, errors="replace"):
                if not ln.strip():
                    continue
                before += 1
                try:
                    sid = json.loads(ln)["sample_id"]
                except Exception:
                    continue
                if sid in seen:
                    continue
                seen.add(sid)
                kept[f].append(ln.rstrip("\n"))
        after = sum(len(v) for v in kept.values())
        report.append((sp, before, after, len(seen)))
        if apply and after < before:
            for f in files:
                bak = f + ".predupe"
                shutil.copy2(f, bak)
                Path(f).write_text("\n".join(kept[f]) + ("\n" if kept[f] else ""))
            # verify
            back = set()
            n = 0
            for f in files:
                for ln in open(f, errors="replace"):
                    if ln.strip():
                        back.add(json.loads(ln)["sample_id"]); n += 1
            assert back == seen and n == after, f"{sp}: verification FAILED"
            for f in files:
                Path(f + ".predupe").unlink()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    for sp, b, af, u in dedupe(a.root, apply=a.apply):
        print(f"  {sp:6s} lines {b:5d} -> {af:5d}   unique {u}   "
              f"{'removed ' + str(b-af) if b != af else 'clean'}")
    print("  (dry run — pass --apply to rewrite)" if not a.apply else "  APPLIED, verified")
