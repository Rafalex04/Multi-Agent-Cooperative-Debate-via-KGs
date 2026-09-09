"""Shards that still have images left, most work first: "<split> <ns> <shard> <n>".

Issuing the lowest-numbered free shard picks an EMPTY one after a re-shard; the
job exits in 0.0 min, the slot frees, and the issuer spins on it forever while
untouched shards sit full. Ranking by remaining work is what makes the loop
advance. Observed on B2: shards 0,1,2,7,8,9 empty and reissued every 90s while
shards 14 and 15 had 67 images each and had never started.
"""
from __future__ import annotations
import glob, json, sys

# train uses a FINER modulus than the 16 the corpus was started with. The tail of
# a run concentrates the last images into a few shards -- 184 images left sat in
# 4 of 16, capping the fleet at 4 of 14 slots -- and a finer modulus spreads them.
# Safe only because 64 is a multiple of 16 and NO NS=16 job is alive: index i is
# handled by shard i%64 now and was handled by i%16 before, so running both
# moduli at once would double-process. One modulus per corpus AT A TIME.
SPLITS = (("train", 1080, 64), ("val", 120, 4), ("test", 400, 8))


def main(out):
    rows = []
    for sp, n, ns in SPLITS:
        done = set()
        for f in glob.glob(f"{out}/{sp}_*.jsonl"):
            for l in open(f, errors="replace"):
                if l.strip():
                    try:
                        done.add(json.loads(l)["sample_id"])
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
    main(sys.argv[1])
