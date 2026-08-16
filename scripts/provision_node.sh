#!/bin/bash
# Copy ollama and the medgemma:4b blobs to a worker node and start the server.
#
# /data is node-local and $HOME is within ~2GB of its quota, so the 3.34GB model
# has to be pushed to each node individually. Sources are rotated across the
# nodes that already have it, otherwise a single source saturates while eight
# targets idle.
#
# Usage: provision_node.sh <target> <source>
set -u
T="$1"; SRC="$2"
D=/data/rm2125

ssh -o ConnectTimeout=10 "$T" "mkdir -p $D/bin $D/ollama_models/blobs \
    $D/ollama_models/manifests/registry.ollama.ai/library/medgemma" || exit 1

# Already provisioned? The model blob is the only large item worth checking.
if ssh "$T" "test -s $D/ollama_models/blobs/sha256-d1d201b9e957a15ee1c6c0e5b1e2e6b0cc2e9b0e0b0e0b0e0b0e0b0e0b0e0b0e" 2>/dev/null; then
  echo "$T already has blobs"
else
  # bin/lib holds llama-server and the CUDA runtimes; ollama is useless without
  # them and fails at request time, not at startup, so the server looks healthy.
  ssh "$SRC" "cd $D && tar cf - bin/ollama bin/lib bin/bin \
      ollama_models/manifests/registry.ollama.ai/library/medgemma/4b" \
    | ssh "$T" "cd $D && tar xf -" || { echo "$T: runtime copy failed"; exit 1; }

  # Blobs referenced by the 4b manifest only, not the whole store.
  BLOBS=$(ssh "$SRC" "python3 -c \"
import json
m=json.load(open('$D/ollama_models/manifests/registry.ollama.ai/library/medgemma/4b'))
print(' '.join(l['digest'].replace(':','-') for l in [m['config']]+m['layers']))\"")
  for b in $BLOBS; do
    ssh "$T" "test -s $D/ollama_models/blobs/$b" 2>/dev/null && continue
    ssh "$SRC" "cat $D/ollama_models/blobs/$b" \
      | ssh "$T" "cat > $D/ollama_models/blobs/$b" || { echo "$T: blob $b failed"; exit 1; }
  done
fi

ssh "$T" "chmod +x $D/bin/ollama; pkill -x ollama 2>/dev/null; sleep 1
  export OLLAMA_MODELS=$D/ollama_models OLLAMA_HOST=0.0.0.0:11434 \
         OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h
  setsid nohup $D/bin/ollama serve > /tmp/ollama.log 2>&1 </dev/null &" 2>/dev/null

for i in $(seq 1 20); do
  sleep 3
  if ssh -o ConnectTimeout=6 "$T" "curl -s -m 5 http://localhost:11434/api/tags | grep -q medgemma" 2>/dev/null; then
    echo "$T READY"; exit 0
  fi
done
echo "$T FAILED to serve"; exit 1
