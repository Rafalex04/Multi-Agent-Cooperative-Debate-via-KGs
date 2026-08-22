#!/bin/bash
# E1 -- frozen probe pipeline over all 1875 BUS-BRA images, registered set P2+P3.
# Each node runs its shard for phrasing 2, then the same shard for phrasing 3.
# ssh -n -f + setsid keeps the channel from being held open (a past run launched
# only one node because ssh blocked); resume-by-index makes reassignment safe.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
NPZ=$REPO/data/external/busbra_pad2_224.npz
OUT=$REPO/experiments/16_external/results
LOG=$REPO/experiments/16_external/logs
NS=12
NODES=(09 10 11 12 13 26 27)
i=0
for n in "${NODES[@]}"; do
  a=$((i*2)); b=$((i*2+1)); i=$((i+1))
  [ $a -ge $NS ] && break
  SH="$a,$b"
  CMD="cd $REPO/experiments/15_kggnn && \
    for P in 2 3; do \
      TAG=busbra_p\$P; \
      python3 -u run_probes.py --npz $NPZ --split all --phrasing \$P \
        --tag \$TAG --num-shards $NS --shards $SH --out-dir $OUT \
        >> $LOG/e1_gpu${n}_p\$P.log 2>&1; \
    done"
  ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
  echo "gpu$n -> shards $SH"
done
