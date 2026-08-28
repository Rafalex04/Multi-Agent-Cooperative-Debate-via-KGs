#!/bin/bash
# Claim-based dispatch: a node that finishes early takes the next unclaimed shard
# instead of idling. Necessary because this cluster is heterogeneous -- the
# Maxwell TITAN X nodes run this model roughly 4x slower than the TITAN Xp, and a
# fixed round-robin leaves them holding the tail (it did, twice, earlier).
# run_probes.py resumes by index, so a re-issued shard is a no-op, never a repeat.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
OUT=$REPO/experiments/23_masktest/results
LOG=$REPO/experiments/23_masktest/logs
LOCK=$LOG/claims
NSH=${NSH:-24}
PH=${PH:-3}
NODES=${NODES:-"10 13 21 23 24 11 12"}
# ollama runs with OLLAMA_NUM_PARALLEL=4, so a node sustains more than one
# probe stream. One job per node gave ~20 records/min across seven nodes,
# a 4.7h run for 5625 records; two per node roughly halves that.
MAXPER=${MAXPER:-2}
mkdir -p $LOCK $OUT
JOBS=""
for a in bbase bdim bring; do for s in $(seq 0 $((NSH-1))); do JOBS="$JOBS $a:$s"; done; done
NJOB=$(echo $JOBS | wc -w)
while true; do
  claimed=$(ls $LOCK 2>/dev/null | wc -l)
  [ "$claimed" -ge "$NJOB" ] && { echo "$(date +%H:%M) all $NJOB jobs claimed"; break; }
  for n in $NODES; do
    c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep -c '^python3 -u run_probe[s].py'" 2>/dev/null)
    [ "${c:-9}" -ge "${MAXPER:-1}" ] 2>/dev/null && continue
    for j in $JOBS; do
      [ -e "$LOCK/${j/:/_}" ] && continue
      touch "$LOCK/${j/:/_}"
      ARM=${j%:*}; SH=${j#*:}
      case $ARM in
        bbase) NPZ=busbra_pad2_224 ;;
        bdim)  NPZ=busbra_bdim_224 ;;
        bring) NPZ=busbra_bring_224 ;;
      esac
      CMD="cd $REPO/experiments/15_kggnn && python3 -u run_probes.py \
        --npz $REPO/data/external/${NPZ}.npz --split all --phrasing $PH \
        --tag ${ARM}_p${PH} --num-shards $NSH --shards $SH --out-dir $OUT \
        >> $LOG/${ARM}_p${PH}_s${SH}.log 2>&1"
      ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
      echo "$(date +%H:%M) gpu$n -> $ARM shard $SH/$NSH  (records $(cat $OUT/*_p${PH}_*.jsonl 2>/dev/null | wc -l))"
      break
    done
  done
  sleep 45
done
