#!/bin/bash
# SINGLE issuer for a DermaMNIST corpus stage. Same design as the retina issuer,
# carrying all four fixes that cost GPU time there:
#   1. one issuer only, PID-lockfile guarded
#   2. in-flight set updated in memory the instant a job is issued
#   3. which removes the resume race that duplicated 236 B1 records
#   4. shards ranked by remaining work, so empty ones are never re-issued
# gpu12 is excluded by default: it is DEAD (ollama serving nothing, ssh
# unreachable as of 2026-09-06). Add it back only after a real generation test.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
DER=$REPO/dermaMnist; SRC=$DER/src/derma_kg; LOG=$DER/logs
VENV=/data/rm2125/venv/bin/python
STAGE=${STAGE:?set STAGE=probe|debate|b1|b2}
NODES=${NODES:-"10 21 13 23 24 11"}
MAXPER=${MAXPER:-2}
TAG=${TAG:-derma_p3}
LOCK=$LOG/.issuer_$STAGE.lock

case $STAGE in
  probe)  OUT=$DER/results;              RUNNER=run_probes.py ;;
  debate) OUT=$DER/data/derma/debates_d1; RUNNER=run_debate_derma.py ;;
  b1)     OUT=$DER/data/derma/catfish_b1; RUNNER=run_catfish_derma.py ;;
  b2)     OUT=$DER/data/derma/multi6;     RUNNER=run_multiagent_derma.py ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac

if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "another issuer for $STAGE is live (pid $(cat $LOCK)); refusing"; exit 1
fi
mkdir -p "$LOG" "$OUT"; echo $$ > "$LOCK"
trap 'rm -f $LOCK' EXIT

echo "$(date +%H:%M) issuer up: stage=$STAGE out=$OUT nodes=[$NODES]"
while true; do
  WORK=$($VENV "$SRC/shard_work.py" "$OUT" "$STAGE" "$TAG")
  if [ -z "$WORK" ]; then echo "$(date +%H:%M) $STAGE COMPLETE"; break; fi
  echo "$(date +%H:%M) $(echo "$WORK" | wc -l) shards with work, $(echo "$WORK" | awk '{s+=$4} END{print s}') images left"

  INFLIGHT=" "
  for n in $NODES; do
    for s in $(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep '$RUNNER' | grep -v grep" 2>/dev/null \
               | sed -n 's/.*--split \([a-z]*\).*--shards\? \([0-9]*\).*/\1:\2/p'); do
      INFLIGHT="$INFLIGHT$s "
    done
  done

  for n in $NODES; do
    c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep 'python3 -u run_' | grep -vc grep" 2>/dev/null)
    [ "${c:-9}" -ge "$MAXPER" ] 2>/dev/null && continue
    while read -r row NS sh w; do
      [ -z "${row:-}" ] && continue
      case "$INFLIGHT" in *" $row:$sh "*) continue;; esac
      if [ "$STAGE" = probe ]; then
        CMD="cd $REPO/experiments/15_kggnn && python3 -u run_probes.py \
             --model qwen3-vl:8b-instruct \
             --npz $DER/data/derma/images_224/${row}.npz \
             --findings $DER/data/derma/probe_findings.json \
             --lexicon  $DER/data/derma/lexicon.json \
             --split $row --phrasing 3 --tag $TAG \
             --num-shards $NS --shards $sh --out-dir $OUT"
      else
        CMD="cd $SRC && python3 -u $RUNNER --model qwen3-vl:8b-instruct \
             --split $row --num-shards $NS --shard $sh --out-dir $OUT"
      fi
      echo "$(date +%H:%M) gpu$n <- $STAGE $row:$NS:$sh ($w left)"
      ssh -n -f gpu$n "setsid nohup bash -lc '$CMD' >> $LOG/${STAGE}_${row}_${sh}.log 2>&1 < /dev/null &"
      INFLIGHT="$INFLIGHT$row:$sh "
      sleep 4; break
    done <<< "$WORK"
  done
  sleep 90
done
