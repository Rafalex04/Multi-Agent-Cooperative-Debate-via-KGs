#!/bin/bash
# launch_unit.sh <node> <shardgroup> <phrasings...>
# One (phrasing, shard-group) unit has exactly one owner. Two writers on one
# jsonl interleave their appends and produce truncated lines -- that corruption
# cost 26 re-runs earlier in this project, so ownership is enforced by hand.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
N=$1; SH=$2; shift 2
PL="$@"
CMD="cd $REPO/experiments/15_kggnn && for P in $PL; do \
  python3 -u run_probes.py --npz $REPO/data/external/busbra_pad2_224.npz \
    --split all --phrasing \$P --tag busbra_p\$P --num-shards 12 --shards $SH \
    --out-dir $REPO/experiments/16_external/results \
    >> $REPO/experiments/16_external/logs/e1_gpu${N}_p\$P.log 2>&1; done"
ssh -n -f gpu$N "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
echo "gpu$N -> shards $SH phrasings [$PL]"
