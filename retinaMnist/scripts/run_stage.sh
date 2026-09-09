#!/bin/bash
# SINGLE issuer for a corpus stage. Replaces dispatch_*.sh + sweep.sh, which were
# two independent issuers racing each other.
#
# Three bugs this fixes, all of which produced duplicate records on B1/B2:
#   1. TWO ISSUERS. dispatch_b2.sh and sweep.sh both handed out shards with no
#      shared state, so both issued train:7 within the same minute.
#   2. STALE LEDGER WITHIN A PASS. The in-flight set was rebuilt once per loop,
#      but a shard launched to node A did not appear in `ps` before node B was
#      considered in the SAME pass, so it was issued twice. The set is now
#      updated in memory the instant a job is issued.
#   3. RESUME RACE. Two jobs on one shard both read done_indices at startup,
#      before either had written, then reprocessed the same indices. Preventing
#      concurrent duplicates (1 and 2) is what actually removes this.
#
# A lockfile makes a second instance refuse to start. Re-issuing a shard whose
# job has EXITED is still safe and intended: the runners resume by scanning all
# shard files for the split.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
RET=$REPO/retinaMnist; SRC=$RET/src/retina_kg; LOG=$RET/logs
VENV=/data/rm2125/venv/bin/python
STAGE=${STAGE:?set STAGE=b1|b2|debate}
NODES=${NODES:-"10 21 13 23 24 11 12"}
MAXPER=${MAXPER:-2}
LOCK=$LOG/.issuer_$STAGE.lock

case $STAGE in
  b1)     OUT=$RET/data/retina/catfish_b1; RUNNER=run_catfish_retina.py ;;
  b2)     OUT=$RET/data/retina/multi6;     RUNNER=run_multiagent_retina.py ;;
  debate) OUT=$RET/data/retina/debates_r1; RUNNER=run_debate_retina.py ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac

if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "another issuer for $STAGE is live (pid $(cat $LOCK)); refusing"; exit 1
fi
mkdir -p $LOG "$OUT"; echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT


echo "$(date +%H:%M) issuer up: stage=$STAGE out=$OUT"
while true; do
  WORK=$($VENV "$SRC/shard_work.py" "$OUT")
  if [ -z "$WORK" ]; then echo "$(date +%H:%M) $STAGE COMPLETE"; break; fi
  echo "$(date +%H:%M) $(echo "$WORK" | wc -l) shards with work, $(echo "$WORK" | awk '{s+=$4} END{print s}') images left"

  # authoritative in-flight set, from the nodes themselves
  INFLIGHT=" "
  for n in $NODES; do
    for s in $(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep '$RUNNER' | grep -v grep" 2>/dev/null \
               | sed -n 's/.*--split \([a-z]*\).*--shard \([0-9]*\).*/\1:\2/p'); do
      INFLIGHT="$INFLIGHT$s "
    done
  done

  for n in $NODES; do
    c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep 'python3 -u run_' | grep -vc grep" 2>/dev/null)
    [ "${c:-9}" -ge "$MAXPER" ] 2>/dev/null && continue
    # take the shard with the MOST remaining work that is not already in flight
    while read -r row NS sh w; do
      [ -z "${row:-}" ] && continue
      case "$INFLIGHT" in *" $row:$sh "*) continue;; esac
        CMD="cd $SRC && python3 -u $RUNNER --model qwen3-vl:8b-instruct \
             --split $row --num-shards $NS --shard $sh --out-dir $OUT"
      echo "$(date +%H:%M) gpu$n <- $STAGE $row:$NS:$sh ($w left)"
      ssh -n -f gpu$n "setsid nohup bash -lc '$CMD' >> $LOG/${STAGE}_${row}_${sh}.log 2>&1 < /dev/null &"
      INFLIGHT="$INFLIGHT$row:$sh "   # updated in-pass: prevents double-issue
      sleep 4; break
    done <<< "$WORK"
  done
  sleep 90
done
