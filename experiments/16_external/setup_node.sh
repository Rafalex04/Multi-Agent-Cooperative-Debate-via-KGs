#!/bin/bash
# Stand up an ollama worker on a bare node.
# OLLAMA_MODELS must point at /data: the default is ~/.ollama/models on the NFS
# home, which is inode- and space-limited (11.7G quota, ~4.8G free) and a 7GB
# model pull there would break every other job on this account.
N=$1
ssh -n gpu$N "mkdir -p /data/rm2125/bin /data/rm2125/.ollama/models" || exit 1
scp -q /data/rm2125/bin/ollama gpu$N:/data/rm2125/bin/ollama || exit 1
ssh -n gpu$N "chmod +x /data/rm2125/bin/ollama"
ssh -n -f gpu$N "cd /data/rm2125 && OLLAMA_MODELS=/data/rm2125/.ollama/models \
  setsid nohup /data/rm2125/bin/ollama serve > /tmp/ollama_$N.log 2>&1 < /dev/null &"
sleep 8
ssh -n gpu$N "OLLAMA_MODELS=/data/rm2125/.ollama/models OLLAMA_HOST=127.0.0.1:11434 \
  /data/rm2125/bin/ollama pull qwen3-vl:8b-instruct 2>&1 | tail -1"
