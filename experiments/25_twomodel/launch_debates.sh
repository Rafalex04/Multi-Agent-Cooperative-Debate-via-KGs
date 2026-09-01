#!/bin/bash
# Launch one debate arm across the cluster.
#
#   launch_debates.sh <arch> <arm> <out-dir> <split> [partner-model]
#
# arch  v2|v3|v4|v5      arm  hom|het
#
# agent_1 always answers on the node the runner sits on (localhost, qwen3-vl).
# For `het`, agent_2 answers on a DIFFERENT node holding the partner model, so
# neither model is ever swapped off a GPU - two 6-8GB vision models on one
# 8-12GB card makes ollama reload weights between every turn.
set -u
R=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
ARCH=$1; ARM=$2; OUT=$3; SPLITS=${4:-test}; PARTNER=${5:-}

case $ARCH in
  v2) DIR=05_debate_v2; SCRIPT=run_debate_v2.py ;;
  v3) DIR=07_debate_v3; SCRIPT=run_debate_v3.py ;;
  v4) DIR=09_debate_v4; SCRIPT=run_debate_v4.py ;;
  v5) DIR=10_debate_v5; SCRIPT=run_debate_v5.py ;;
  *)  echo "unknown arch $ARCH"; exit 1 ;;
esac

QNODES=(${QNODES:-07 11 12 14 15 19 23 24})
PNODES=(${PNODES:-})
NQ=${#QNODES[@]}
[ "$ARM" = het ] && [ -z "$PARTNER" ] && { echo "het needs a partner model"; exit 1; }
[ "$ARM" = het ] && [ ${#PNODES[@]} -eq 0 ] && { echo "het needs PNODES"; exit 1; }

echo "$ARCH/$ARM -> $OUT  splits=$SPLITS  shards=$NQ"
for i in $(seq 0 $((NQ-1))); do
  n=${QNODES[$i]}
  EXTRA=""
  if [ "$ARM" = het ]; then
    pn=${PNODES[$((i % ${#PNODES[@]}))]}
    EXTRA="--model-b $PARTNER --url-b http://gpu$pn:11434/api/chat"
  fi
  # one shard walks every split in turn; the runner resumes by file existence
  CMD="cd $R/experiments/$DIR"
  for sp in $SPLITS; do
    CMD="$CMD && python3 -u $SCRIPT \
       --model qwen3-vl:8b-instruct --url http://localhost:11434/api/chat \
       $EXTRA --split $sp --rounds 3 \
       --shard $i --num-shards $NQ --out-dir $OUT"
  done
  ssh -n -f -o StrictHostKeyChecking=no gpu$n \
     "setsid nohup bash -c '$CMD' > /tmp/deb_${ARCH}_${ARM}_$i.log 2>&1 < /dev/null &"
  echo "  gpu$n shard $i/$NQ ${EXTRA:+-> agent_2 on gpu${PNODES[$((i % ${#PNODES[@]}))]}}"
done
