#!/bin/bash
cd /home/rafael/ICL/MSc/Thesis/Code/project/breastMnist
if ! pgrep -f "monitor_daemon.py" > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting watchdog..." >> outputs/monitor.log
    nohup bash -c '
while true; do
    .venv/bin/python monitor_daemon.py
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Daemon exited, restarting in 15s..." >> outputs/monitor.log
    sleep 15
done
' > /dev/null 2>&1 &
fi
