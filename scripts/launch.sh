#!/bin/bash
# Fully detached launcher: survives the ssh session that starts it.
#   launch.sh <logfile> <command...>
LOG="$1"; shift
cd "$HOME/Multi-Agent-Cooperative-Debate-via-KGs" || exit 1
setsid nohup "$@" > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
