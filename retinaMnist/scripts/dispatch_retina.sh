#!/bin/bash
# Claim-based dispatch for the retina corpora, modelled on 23_masktest/dispatch.sh.
# A node that finishes early takes the next unclaimed shard instead of idling,
# because the fleet is heterogeneous (Maxwell TITAN X ~4x slower than TITAN Xp)
# and fixed round-robin leaves the slow nodes holding the tail.
# Both runners resume by index, so a re-issued shard is a no-op, never a repeat.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
RET=$REPO/retinaMnist
SRC=$RET/src/retina_kg
LOG=$RET/logs; LOCK=$LOG/claims
# SEGREGATED BY JOB KIND. run_probes.py asks for num_ctx 2048 and
# run_debate_retina.py for 4096; a different num_ctx is a different ollama runner,
# so a node given both tries to hold two 5.8GB copies of the model. On the 8GB
# cards it can hold only one, and every alternating request evicts and reloads
# the other -- 20-45s each, per 16_external/start_worker.sh. Measured cost of the
# mixed layout: 99-228 s/image against 10.8 s/image when a node runs one kind.
# The small-VRAM cards get the small-context job.
PROBE_NODES=${PROBE_NODES:-"10 13 23 24"}    # 12GB TITAN Xp + the three 8GB cards
DEBATE_NODES=${DEBATE_NODES:-"21 11 12"}     # 11GB 2080Ti + the two 12GB Maxwells
MAXPER=${MAXPER:-2}                          # within one kind: one runner, shared
mkdir -p $LOCK $LOG $RET/results

# split:shards -- val is only 120 images, so it gets fewer, larger shards
JOBS=""
for spec in train:16 val:4 test:8; do
  SP=${spec%:*}; NS=${spec#*:}
  for s in $(seq 0 $((NS-1))); do
    JOBS="$JOBS probe:$SP:$NS:$s debate:$SP:$NS:$s"
  done
done
NJOB=$(echo $JOBS | wc -w)
echo "$(date +%H:%M) dispatching $NJOB jobs | probes on [$PROBE_NODES] | debate on [$DEBATE_NODES]"

while true; do
  claimed=$(ls $LOCK 2>/dev/null | wc -l)
  [ "$claimed" -ge "$NJOB" ] && { echo "$(date +%H:%M) all $NJOB jobs claimed"; break; }
  for spec in "probe:$PROBE_NODES" "debate:$DEBATE_NODES"; do
   WANT=${spec%%:*}; NL=${spec#*:}
   for n in $NL; do
    c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o pid=,args= | grep 'python3 -u run_' | grep -vc grep" 2>/dev/null)
    [ "${c:-9}" -ge "$MAXPER" ] 2>/dev/null && continue
    for j in $JOBS; do
      K=${j//:/_}
      [ -e "$LOCK/$K" ] && continue
      KIND=$(echo $j|cut -d: -f1)
      [ "$KIND" != "$WANT" ] && continue
      touch "$LOCK/$K"
      SP=$(echo $j|cut -d: -f2)
      NS=$(echo $j|cut -d: -f3);   SH=$(echo $j|cut -d: -f4)
      if [ "$KIND" = probe ]; then
        CMD="cd $REPO/experiments/15_kggnn && python3 -u run_probes.py \
          --model qwen3-vl:8b-instruct \
          --npz $RET/data/retina/images_224/${SP}.npz \
          --findings $RET/data/retina/probe_findings.json \
          --lexicon $RET/data/retina/lexicon.json \
          --split $SP --phrasing 3 --tag retina_p3 \
          --num-shards $NS --shards $SH --out-dir $RET/results"
      else
        CMD="cd $SRC && python3 -u run_debate_retina.py \
          --model qwen3-vl:8b-instruct --split $SP --rounds 3 \
          --num-shards $NS --shard $SH \
          --out-dir $RET/data/retina/debates_r1"
      fi
      echo "$(date +%H:%M) gpu$n <- $j"
      ssh -n -f gpu$n "setsid nohup bash -lc '$CMD' >> $LOG/${K}.log 2>&1 < /dev/null &"
      sleep 3
      break
    done
   done
  done
  sleep 45
done
