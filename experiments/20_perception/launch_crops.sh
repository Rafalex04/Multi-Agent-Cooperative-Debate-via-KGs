#!/bin/bash
# Re-probe BrEaST at two other framings, so the per-finding perception audit can
# be run three times on the SAME 252 images with the SAME 32 probes and the SAME
# radiologist ground truth, changing only how much of the frame the lesion fills.
#
# pad 1.25  lesion nearly fills the frame
# pad 2.00  the registered framing, already probed (brst_p2 / brst_p3)
# pad 3.00  lesion in context
#
# The question is specific, not "does cropping help": five probes anti-correlate
# with the descriptor they name, and shape is two of the five. If that is a
# framing artifact those AUCs move with the window. If it is the wording, they
# do not.
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
OUT=$REPO/experiments/20_perception/results
LOG=$REPO/experiments/20_perception/logs
NODES=${NODES:-"10 11 12 13 21 23 24 26 27"}
NSH=8
i=0
for TAG in pad125 pad30; do
  for s in $(seq 0 $((NSH-1))); do
    set -- $NODES; shift $((i % $#)); n=$1; i=$((i+1))
    CMD="cd $REPO/experiments/15_kggnn && for P in 2 3; do \
      python3 -u run_probes.py --npz $REPO/data/external/breast_${TAG}_224.npz \
        --split brst --phrasing \$P --tag ${TAG}_p\$P --num-shards $NSH --shards $s \
        --out-dir $OUT >> $LOG/crop_${TAG}_s${s}.log 2>&1; done"
    ssh -n -f gpu$n "setsid nohup bash -c '$CMD' < /dev/null > /dev/null 2>&1 &"
    echo "gpu$n -> $TAG shard $s/$NSH"
  done
done
