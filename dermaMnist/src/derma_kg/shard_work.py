"""Shards with images left, most work first: "<split> <ns> <shard> <n>".

DermaMNIST split sizes; otherwise identical to the retina helper, including the
reason it exists: issuing the lowest-numbered free shard picks an EMPTY one after
a re-shard, the job exits in 0.0 min, and the issuer spins on it forever.
"""
from __future__ import annotations
import glob, json, sys

# train is 7007 images at 55 probes each, so it needs many more shards than
# retina's 16 to keep every fleet slot fed and to keep the tail short.
SPLITS = (("train", 7007, 64), ("val", 1003, 16), ("test", 2005, 32))
KEY = {"probe": "index", "debate": "sample_id", "b1": "sample_id", "b2": "sample_id"}


def main(out, stage="probe", tag="derma_p3"):
    key = KEY.get(stage, "sample_id")
    rows = []
    for sp, n, ns in SPLITS:
        done = set()
        pat = f"{out}/{tag}_{sp}_*.jsonl" if stage == "probe" else f"{out}/{sp}_*.jsonl"
        for f in glob.glob(pat):
            for l in open(f, errors="replace"):
                if l.strip():
                    try:
                        done.add(json.loads(l)[key])
                    except Exception:
                        pass
        miss = [i for i in range(n) if i not in done]
        for sh in range(ns):
            w = sum(1 for i in miss if i % ns == sh)
            if w:
                rows.append((w, sp, ns, sh))
    for w, sp, ns, sh in sorted(rows, reverse=True):
        print(f"{sp} {ns} {sh} {w}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "probe",
         sys.argv[3] if len(sys.argv) > 3 else "derma_p3")
