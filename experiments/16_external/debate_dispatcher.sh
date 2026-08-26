#!/bin/bash
# Keep every node on a debate shard until the budget is spent.
# Shards are claimed by touching a lock file, so a node that finishes early takes
# the next unclaimed one instead of idling between manual checks. Debates write
# one JSON per sample and the runner skips existing files, so a re-run of any
# shard is a no-op rather than duplicated work.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
LOCK=$REPO/experiments/16_external/logs/claims
MAXSHARD=${1:-24}          # 24/36 of 1875 ~= 1250 samples
mkdir -p $LOCK
# Nodes are a parameter now: /data is node-local and was wiped on
# 21/26/27, so a node with no model store must not be handed a shard --
# it would claim the lock and then fail, silently losing that shard.
NODES=${NODES:-"10 13 21 23 24 26 27 11 12"}
while true; do
  claimed=$(ls $LOCK 2>/dev/null | wc -l)
  [ "$claimed" -ge "$MAXSHARD" ] && { echo "$(date +%H:%M) all $MAXSHARD shards claimed"; break; }
  for n in $NODES; do
    c=$(timeout 10 ssh -n gpu$n "ps -u \$USER -o args= | grep -c '^python3 -u run_debate'" 2>/dev/null)
    [ "${c:-1}" != "0" ] && continue
    for s in $(seq 0 $((MAXSHARD-1))); do
      [ -e "$LOCK/$s" ] && continue
      touch "$LOCK/$s"
      CMD="cd $REPO/experiments/10_debate_v5 && python3 -u run_debate_v5.py \
        --model qwen3-vl:8b-instruct --npz $REPO/data/external/busbra_pad2_224.npz \
        --split all --rounds 3 --num-shards 36 --shard $s \
        --out-dir $REPO/breastMnist/data/breast/debates_busbra \
        >> $REPO/experiments/16_external/logs/debate_s$s.log 2>&1"
      ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
      echo "$(date +%H:%M) gpu$n -> shard $s  (total debates $(ls $REPO/breastMnist/data/breast/debates_busbra/all/*.json 2>/dev/null | wc -l))"
      break
    done
  done
  sleep 120
done
