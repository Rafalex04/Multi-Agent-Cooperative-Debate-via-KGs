#!/bin/bash
# BUS-BRA debates, so the full architecture (debate + debate graph + KG + GNN)
# gets an external test rather than only the probe half.
#
# A debate is 6 long generations per sample against 2 short ones per probe, so the
# whole 1875 would cost roughly 5 GPU-hours per node-equivalent. Shards 0-3 of 12
# are run instead: a systematic 1-in-3 sample, unbiased with respect to the ID
# ordering (which tracks acquisition batch and device), giving ~625 samples --
# four times the BreastMNIST test set.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
NPZ=$REPO/data/external/busbra_pad2_224.npz
OUT=$REPO/breastMnist/data/breast/debates_busbra
LOG=$REPO/experiments/16_external/logs
i=0
for n in "$@"; do
  SH=$i; i=$((i+1))
  [ $SH -ge 12 ] && break
  CMD="cd $REPO/experiments/10_debate_v5 && python3 -u run_debate_v5.py \
    --model qwen3-vl:8b-instruct --npz $NPZ --split all --rounds 3 \
    --num-shards 12 --shard $SH --out-dir $OUT \
    >> $LOG/debate_gpu${n}.log 2>&1"
  ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
  echo "gpu$n -> debate shard $SH/12"
done
