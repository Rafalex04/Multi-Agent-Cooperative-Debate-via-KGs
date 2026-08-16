#!/bin/bash
# Shard the v2 debate across GPU nodes.
#
# Each worker walks test -> val -> train sequentially so one node never runs
# three jobs at once. All workers write into the same NFS out-dir; a sample is a
# single file and a worker skips any file that already exists, so shards never
# collide, the job is resumable, and extra workers can be added at any time to
# help finish (run with --num-shards 1 as a sweeper).
#
# Usage: launch_debate_v2.sh <out-dir> <num-shards> <node:shard> [node:shard ...]
set -u
REPO="$HOME/Multi-Agent-Cooperative-Debate-via-KGs"
OUT="$1"; NSHARDS="$2"; shift 2

for SPEC in "$@"; do
  NODE="${SPEC%%:*}"; SHARD="${SPEC##*:}"
  ssh "$NODE" "cd $REPO && setsid nohup bash -c '
    for SPLIT in test val train; do
      python3 -u experiments/05_debate_v2/run_debate_v2.py \
        --split \$SPLIT --shard $SHARD --num-shards $NSHARDS --workers 6 \
        --out-dir \"$OUT\" >> /tmp/debate_v2_\$SPLIT.log 2>&1
    done' > /dev/null 2>&1 < /dev/null &" \
    && echo "launched $NODE shard $SHARD/$NSHARDS"
done
