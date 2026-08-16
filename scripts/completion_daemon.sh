#!/bin/bash
# Keep the v2 debate running until every sample has a graph.
#
# The shard workers finish at very different times (the two GTX 1080 nodes are
# ~4x slower than the 2080 Ti nodes), so fixed sharding leaves gaps. This daemon
# watches each node and, whenever one goes idle while graphs are still missing,
# starts a sweeper there. Sweepers walk every index and skip files that already
# exist, so they safely fill whatever the shard workers did not reach.
#
# Alternating sweep direction per node keeps two sweepers from grinding through
# the same samples in the same order.
#
# Usage: completion_daemon.sh [out-dir]
set -u
REPO="$HOME/Multi-Agent-Cooperative-Debate-via-KGs"
OUT="${1:-$REPO/breastMnist/data/breast/dataset_v2}"
declare -A WANT=( [train]=546 [val]=78 [test]=156 )
NODES=(gpu21 gpu22 gpu23 gpu24)

have() { ls "$OUT/$1/graphs" 2>/dev/null | wc -l; }
total_have() { echo $(( $(have train) + $(have val) + $(have test) )); }
busy() { ssh -o ConnectTimeout=10 "$1" "ps -eo args | grep -q '^python3 -u experiments/05_debate_v2'" 2>/dev/null; }

i=0
while [ "$(total_have)" -lt 780 ]; do
  for NODE in "${NODES[@]}"; do
    [ "$(total_have)" -ge 780 ] && break
    if ! busy "$NODE"; then
      # Sweep the split that is furthest from complete first.
      ORDER=""
      for S in train val test; do
        [ "$(have $S)" -lt "${WANT[$S]}" ] && ORDER="$ORDER $S"
      done
      [ -z "$ORDER" ] && continue
      REV=""; [ $((i % 2)) -eq 1 ] && REV="--reverse"
      echo "$(date +%H:%M) starting sweeper on $NODE ($ORDER) $REV"
      # -f backgrounds ssh itself; without it the client can sit holding the
      # connection open and stall the whole daemon loop.
      ssh -f -o ConnectTimeout=10 "$NODE" "cd $REPO && setsid nohup bash -c '
        for S in $ORDER; do
          python3 -u experiments/05_debate_v2/run_debate_v2.py --split \$S \
            --num-shards 1 $REV --workers 6 --out-dir \"$OUT\" \
            >> /tmp/sweep_\$S.log 2>&1
        done' >/dev/null 2>&1 </dev/null"
      i=$((i + 1))
    fi
  done
  echo "$(date +%H:%M) train=$(have train)/546 val=$(have val)/78 test=$(have test)/156"
  sleep 180
done
echo "$(date +%H:%M) ALL 780 GRAPHS COMPLETE"
