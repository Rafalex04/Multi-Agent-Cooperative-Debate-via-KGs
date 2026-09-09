#!/bin/bash
# Fan the judge across the fleet for one (tree, split); returns when complete.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
V=/data/rm2125/venv/bin/python
TREE=${TREE:?}; SPLIT=${SPLIT:?}
NODES=${NODES:-"10 21 13 23 24 11"}; MAXPER=${MAXPER:-2}; NS=${NS:-16}
case $TREE in
  retina) OUT=$REPO/retinaMnist/data/retina/judge ;;
  derma)  OUT=$REPO/dermaMnist/data/derma/judge ;;
esac
mkdir -p "$OUT"
total () { $V - "$OUT" "$SPLIT" "$TREE" <<'PY'
import glob,json,sys
out,sp,tree=sys.argv[1],sys.argv[2],sys.argv[3]
src={"retina":"retinaMnist/data/retina/debates_r1","derma":"dermaMnist/data/derma/debates_d1"}[tree]
base="/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs/"+src
want=set()
for f in glob.glob(f"{base}/{sp}_*.jsonl"):
    for l in open(f,errors="replace"):
        if l.strip():
            try: want.add(json.loads(l)["sample_id"])
            except Exception: pass
done=set()
for f in glob.glob(f"{out}/{sp}_*.jsonl"):
    for l in open(f,errors="replace"):
        if l.strip():
            try: done.add(json.loads(l)["sample_id"])
            except Exception: pass
print(len(want-done))
PY
}
while true; do
  M=$(total); [ "${M:-0}" -le 0 ] && { echo "$(date +%H:%M) judge $TREE/$SPLIT COMPLETE"; break; }
  for n in $NODES; do
    c=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep -c 'run_judg[e].py'" 2>/dev/null)
    [ "${c:-9}" -gt "$MAXPER" ] 2>/dev/null && continue
    for sh in $(seq 0 $((NS-1))); do
      busy=$(timeout 15 ssh -n gpu$n "ps -u \$USER -o args= | grep 'run_judg[e].py' | grep -c -- '--shard $sh '" 2>/dev/null)
      [ "${busy:-0}" -gt 0 ] && continue
      # python3, NOT $V: /data/rm2125 is NODE-LOCAL, so the venv exists only on
      # the submit host. Every other runner in this project uses system python3
      # on the workers for exactly this reason. With ssh -f the error went to
      # /dev/null and the dispatcher spun for two hours launching nothing.
      timeout 15 ssh -n -f gpu$n "cd $REPO/shared && setsid nohup python3 -u run_judge.py \
        --tree $TREE --split $SPLIT --num-shards $NS --shard $sh \
        >> $REPO/shared/logs/judge_${TREE}_${SPLIT}_${sh}.log 2>&1 < /dev/null &" 2>/dev/null
      sleep 3; break
    done
  done
  sleep 90
done
