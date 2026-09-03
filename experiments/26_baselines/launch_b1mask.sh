#!/bin/bash
# b1-mask launcher. ONE modulus (24) for the whole corpus, forever.
# Usage: ./launch_b1mask.sh <node> <shard>
R=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
N=$1; S=$2
ssh -n -f $N "cd $R/experiments/26_baselines && setsid nohup python3 -u run_catfish.py \
 --split all --shard $S --num-shards 24 \
 --npz $R/data/external/busbra_bring_224.npz \
 --probe-dir $R/experiments/23_masktest/results --probe-tag bring_p3 \
 --out-dir $R/breastMnist/data/breast/catfish_busbra_mask \
 > /tmp/b1mask_s$S.log 2>&1 < /dev/null &"
echo "launched $N shard $S/24"
