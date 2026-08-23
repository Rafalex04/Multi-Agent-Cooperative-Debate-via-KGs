#!/bin/bash
# Match the environment of the nodes that already work (gpu10/13/...).
# Two things matter and both were wrong on the first attempt:
#   * NO LD_LIBRARY_PATH. ollama bundles a libcuda.so.1 next to its runners;
#     putting that directory on the loader path shadows the system driver's
#     libcuda, CUDA init fails silently, and the server falls back to CPU while
#     still answering every request correctly ("offloaded 0/37 layers to GPU").
#   * OLLAMA_KEEP_ALIVE=24h, or the model is evicted between probe batches and
#     every reload costs 20-45s.
N=$1
ssh -n gpu$N "pkill -u \$USER -f 'ollama serve' 2>/dev/null; sleep 2" 2>/dev/null
# OLLAMA_LIBRARY_PATH is the one that actually matters: ollama's GPU discovery
# looks for its CUDA runners beside the binary (it logged
# OLLAMA_LIBRARY_PATH=[/data/rm2125/bin]) and never descends into lib/ollama,
# so with the runners one directory down it finds no GPU in 28ms and loads on CPU.
ssh -n -f gpu$N "OLLAMA_MODELS=/data/rm2125/.ollama/models OLLAMA_HOST=0.0.0.0:11434 \
  OLLAMA_LIBRARY_PATH=/data/rm2125/bin/lib/ollama \
  OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h \
  setsid nohup /data/rm2125/bin/ollama serve > /tmp/ollama_$N.log 2>&1 < /dev/null &"
