#!/bin/bash
# b1-mask work queue. ONE modulus (24), forever.
#
# Two traps this guards against, both hit on the first attempt:
#  1. `pgrep -c -f PATTERN` over ssh counts its own shell wrapper ONLY when ssh
#     spawns one, so "1" means idle in one form and BUSY in the other. We count
#     real python3 processes explicitly instead, and idle == 0.
#  2. A gemma server runs NO python process, so an idle test alone would schedule
#     onto a node that is actively serving a running m6 debate. We derive the
#     gemma set dynamically from the live m6 runners and exclude it.
R=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
OUT=$R/breastMnist/data/breast/catfish_busbra_mask
# gpu11/gpu12 Maxwell excluded (5-6x slower); gpu17 is a 2GB GT1030; gpu25 has no ollama.
POOL="gpu03 gpu04 gpu05 gpu06 gpu07 gpu09 gpu13 gpu14 gpu15 gpu16 gpu20 gpu23 gpu24 gpu26 gpu27 gpu02 gpu08 gpu10 gpu18 gpu19 gpu21"

realprocs() { timeout 12 ssh -o ConnectTimeout=6 -o BatchMode=yes $1 \
  "pgrep -af 'python3 -u run_' | grep -cE '^[0-9]+ python3'" 2>/dev/null; }

for round in $(seq 1 80); do
  # --- gemma servers in use by live m6 runners (dynamic) ---
  GEMMA=""
  for n in $POOL; do
    g=$(timeout 12 ssh -o ConnectTimeout=6 -o BatchMode=yes $n \
        "pgrep -af 'python3 -u run_multiagent' | grep -oE 'gemma-url http://[a-z0-9]+' | grep -oE 'gpu[0-9]+'" 2>/dev/null)
    [ -n "$g" ] && GEMMA="$GEMMA $g"
  done
  # --- shards currently live anywhere ---
  LIVE=$(for n in $POOL; do
    timeout 12 ssh -o ConnectTimeout=6 -o BatchMode=yes $n \
      "pgrep -af 'run_catfish.py' | grep -oE 'shard [0-9]+ --num-shards 24'" 2>/dev/null \
      | grep -oE '[0-9]+ --' | grep -oE '^[0-9]+'
  done | tr '\n' ' ')
  # --- shards already complete ---
  DONE=$(python3 - <<PY
import json,glob,collections
cnt=collections.Counter()
for f in glob.glob("$OUT/all_*.jsonl"):
    for ln in open(f,errors='replace'):
        if ln.strip():
            try: cnt[int(json.loads(ln)['sample_id'])%24]+=1
            except Exception: pass
exp={k:len([i for i in range(1875) if i%24==k]) for k in range(24)}
print(' '.join(str(k) for k in range(24) if cnt.get(k,0)>=exp[k]))
PY
)
  echo "$(date +%H:%M) gemma:[$GEMMA] live:[$LIVE] done:[$DONE]"
  # --- assign ---
  for n in $POOL; do
    echo " $GEMMA " | grep -q " $n " && continue
    [ "$(realprocs $n)" = "0" ] || continue
    ok=$(timeout 110 ssh -o ConnectTimeout=8 $n "curl -s -m 90 http://localhost:11434/api/chat -d '{\"model\":\"qwen3-vl:8b-instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"stream\":false,\"options\":{\"num_predict\":3}}' | grep -c '\"content\"'" 2>/dev/null)
    [ "$ok" = "1" ] || { echo "  $n unhealthy, skip"; continue; }
    for k in $(seq 0 23); do
      echo " $LIVE " | grep -q " $k " && continue
      echo " $DONE " | grep -q " $k " && continue
      $R/experiments/26_baselines/launch_b1mask.sh $n $k && echo "  assigned shard $k -> $n"
      LIVE="$LIVE $k"
      break
    done
  done
  nu=$(python3 -c "
import json,glob
s=set()
for f in glob.glob('$OUT/all_*.jsonl'):
    for ln in open(f,errors='replace'):
        if ln.strip():
            try: s.add(int(json.loads(ln)['sample_id']))
            except Exception: pass
print(len(s))")
  echo "$(date +%H:%M) b1-mask unique=$nu/1875"
  [ "$nu" -ge 1875 ] && { echo "B1-MASK COMPLETE"; break; }
  sleep 480
done
