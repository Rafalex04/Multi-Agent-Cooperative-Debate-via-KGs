#!/bin/bash
# Our own 2-agent v5 protocol on the MASK condition, so the external mask column
# has an "us" row to compare the baselines against. Settings are copied verbatim
# from the auto corpus's dispatcher (qwen3-vl:8b-instruct, rounds 3) with only
# the npz swapped, so auto and mask differ in the image and nothing else.
R=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
N=$1; S=$2
ssh -n -f $N "cd $R/experiments/10_debate_v5 && setsid nohup python3 -u run_debate_v5.py \
 --model qwen3-vl:8b-instruct --npz $R/data/external/busbra_bring_224.npz \
 --split all --rounds 3 --num-shards 20 --shard $S \
 --out-dir $R/breastMnist/data/breast/debates_busbra_mask \
 > /tmp/debmask_s$S.log 2>&1 < /dev/null &"
echo "launched $N shard $S/20"
