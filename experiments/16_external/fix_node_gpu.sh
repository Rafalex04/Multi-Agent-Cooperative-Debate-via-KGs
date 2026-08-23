#!/bin/bash
# ollama ships its CUDA runners in lib/ollama/cuda_v1x, NOT in the binary.
# Copying only the 45MB binary silently yields "offloaded 0/37 layers to GPU" --
# the server starts, answers correctly, and runs entirely on CPU at roughly 1/50
# the throughput. Pull the library tree node-to-node so the source machine's
# uplink is the only bottleneck, then restart with LD_LIBRARY_PATH set.
N=$1; SRC=${2:-gpu13}
ssh -n gpu$N "mkdir -p /data/rm2125/bin/lib && \
  scp -q -o StrictHostKeyChecking=no -r $SRC:/data/rm2125/bin/lib/ollama /data/rm2125/bin/lib/ 2>&1 | tail -1; \
  du -sh /data/rm2125/bin/lib/ollama"
ssh -n -f gpu$N "cd /data/rm2125 && OLLAMA_MODELS=/data/rm2125/.ollama/models \
  LD_LIBRARY_PATH=/data/rm2125/bin/lib/ollama:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  setsid nohup /data/rm2125/bin/ollama serve > /tmp/ollama_$N.log 2>&1 < /dev/null &"
