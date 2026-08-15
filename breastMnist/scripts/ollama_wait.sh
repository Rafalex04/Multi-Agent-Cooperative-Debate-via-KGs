#!/usr/bin/env bash
# Local stand-in for the cluster's OLLAMA_RESTART_SCRIPT.
# systemd already manages ollama here, so just wait for it to come back.
for _ in $(seq 1 30); do
    curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1 && exit 0
    sleep 5
done
exit 1
