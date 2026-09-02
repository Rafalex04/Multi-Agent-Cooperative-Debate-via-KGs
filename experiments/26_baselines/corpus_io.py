"""One place that knows how a debate corpus is stored.

Corpora were originally one JSON per image, which is convenient for
resume-by-file-existence and catastrophic for an inode quota: six more corpora
at that layout need ~9,060 files against ~1,355 of headroom. JSONL stores the
same records as one file per (split, shard) - ~48 files for everything - and
resumes by index the way run_probes.py already does.

`load_corpus` reads either layout, so every existing analysis keeps working and
already-generated corpora do not need regenerating.
"""
from __future__ import annotations

import glob, json
from pathlib import Path

SPLITS = ("train", "val", "test")


def jsonl_paths(out_dir, split):
    return sorted(Path(out_dir).glob(f"{split}_*.jsonl"))


def done_indices(out_dir, split, shard=None):
    """Every index already written for this split, by ANY shard.

    Scanning only this shard's own file means a shard that gets re-split across
    freed nodes redoes work another file already holds. Scanning all of them
    makes redistribution free, which is what the straggler rebalance needs.
    """
    done = set()
    for p in jsonl_paths(out_dir, split):
        for ln in p.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    done.add(int(json.loads(ln)["sample_id"]))
                except Exception:
                    pass
    return done


def append(out_dir, split, shard, rec):
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    with (p / f"{split}_{shard}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def load_corpus(root, splits=SPLITS):
    """-> {split: [record, ...]}. Accepts jsonl files OR a dir of per-image json."""
    root = Path(root)
    out = {}
    for sp in splits:
        recs = []
        for f in jsonl_paths(root, sp):
            for ln in f.read_text(errors="replace").splitlines():
                if ln.strip():
                    try:
                        recs.append(json.loads(ln))
                    except Exception:
                        pass
        if not recs:                                  # legacy per-image layout
            for f in sorted(glob.glob(str(root / sp / "*.json"))):
                try:
                    recs.append(json.loads(Path(f).read_text()))
                except Exception:
                    pass
        recs.sort(key=lambda r: int(r["sample_id"]))
        out[sp] = recs
    return out


def pack(root, splits=SPLITS, delete=True):
    """dir-of-json -> jsonl, verified record-for-record before anything is removed."""
    root = Path(root)
    report = []
    for sp in splits:
        files = sorted(glob.glob(str(root / sp / "*.json")))
        if not files:
            continue
        recs = [json.loads(Path(f).read_text()) for f in files]
        dest = root / f"{sp}_packed.jsonl"
        with dest.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        back = [json.loads(l) for l in dest.read_text().splitlines() if l.strip()]
        ok = len(back) == len(recs) and all(
            b == r for b, r in zip(back, recs))
        if ok and delete:
            for f in files:
                Path(f).unlink()
            try:
                (root / sp).rmdir()
            except OSError:
                pass
        elif not ok:
            dest.unlink()
        report.append((sp, len(recs), ok))
    return report


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--keep", action="store_true", help="do not delete originals")
    a = ap.parse_args()
    for r in a.roots:
        rep = pack(r, delete=not a.keep)
        for sp, n, ok in rep:
            print(f"{r:56s} {sp:6s} {n:5d} records  "
                  f"{'VERIFIED, loose files removed' if ok else 'MISMATCH - kept'}")
        if not rep:
            print(f"{r:56s} (nothing to pack)")
