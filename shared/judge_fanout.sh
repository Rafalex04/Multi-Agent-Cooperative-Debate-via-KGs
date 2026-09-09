#!/bin/bash
# Direct fan-out for the judge. Replaces the polling dispatcher, which launched
# ONE shard per node per 90 s pass and left most slots idle.
#
# MAXPER is 6, not 2. The 2 came from breast PROBE measurements: image prefill on
# a shared GPU. The judge sends NO IMAGE and asks for 8 tokens, so it never
# touches the vision tower -- a node sustains far more of these concurrently.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
TREE=${TREE:?}; NODES=${NODES:-"10 21 13 23 24 11"}; PER=${PER:-4}
case $TREE in
  retina) OUT=$REPO/retinaMnist/data/retina/judge; SPL="test val train" ;;
  derma)  OUT=$REPO/dermaMnist/data/derma/judge;   SPL="test val train" ;;
esac
mkdir -p "$OUT" "$REPO/shared/logs"
NS=$(( $(echo $NODES | wc -w) * PER ))
for sp in $SPL; do
  echo "$(date +%H:%M) judge $TREE/$sp over $NS shards"
  i=0
  for n in $NODES; do
    for _ in $(seq 1 $PER); do
      ssh -n -f gpu$n "cd $REPO/shared && setsid nohup python3 -u run_judge.py \
        --tree $TREE --split $sp --num-shards $NS --shard $i \
        >> $REPO/shared/logs/judge_${TREE}_${sp}_${i}.log 2>&1 < /dev/null &" 2>/dev/null
      i=$((i+1))
    done
  done
  # wait for this split before starting the next, so shards never collide
  while true; do
    live=0
    for n in $NODES; do
      c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep -c 'run_judg[e].py'" 2>/dev/null)
      live=$((live + ${c:-0}))
    done
    [ "$live" -le 0 ] && break
    sleep 30
  done
  echo "$(date +%H:%M) judge $TREE/$sp done: $(cat $OUT/${sp}_*.jsonl 2>/dev/null | wc -l) records"
done
echo "$(date +%H:%M) judge $TREE COMPLETE"
