#!/bin/bash
# Fires when a node's E1 unit is done, so the cluster is never idle overnight.
# Order is by value: BrEaST first (E2 is the round's most informative experiment
# and is only 252 images), then the lesion probes, then BUS-BRA debates.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
cd $REPO

busy() {  # a node is busy if it has any probe/debate python running
  local n=$1
  local c=$(timeout 12 ssh -n gpu$n "ps -u \$USER -o args= | grep -c '^python3 -u run_'" 2>/dev/null)
  [ "${c:-1}" -gt 0 ]
}

# --- wait for E1 -------------------------------------------------------------
while true; do
  P2=$(cat $REPO/experiments/16_external/results/busbra_p2_all_*.jsonl 2>/dev/null | wc -l)
  P3=$(cat $REPO/experiments/16_external/results/busbra_p3_all_*.jsonl 2>/dev/null | wc -l)
  echo "$(date +%H:%M) E1 P2=$P2 P3=$P3"
  [ "$P2" -ge 1875 ] && [ "$P3" -ge 1875 ] && break
  # opportunistically put any idle node on BrEaST while E1 finishes
  for n in 27 26 23 21 13 10 24; do
    if ! busy $n; then
      for s in 0 1 2 3; do
        f=$REPO/experiments/16_external/results/brst_p2_brst_$s.jsonl
        g=$REPO/experiments/16_external/results/brst_p3_brst_$s.jsonl
        if [ ! -f "$f.claim" ]; then
          touch "$f.claim"
          CMD="cd $REPO/experiments/15_kggnn && for P in 2 3; do python3 -u run_probes.py --npz $REPO/data/external/breast_pad2_224.npz --split brst --phrasing \$P --tag brst_p\$P --num-shards 8 --shards $((s+4)) --out-dir $REPO/experiments/16_external/results >> $REPO/experiments/16_external/logs/e2_gpu${n}_s$s.log 2>&1; done"
          ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
          echo "$(date +%H:%M) gpu$n -> BrEaST extra shard $((s+4))"
          break
        fi
      done
    fi
  done
  sleep 180
done
echo "$(date +%H:%M) E1 COMPLETE"
