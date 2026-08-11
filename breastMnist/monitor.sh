#!/usr/bin/env bash
# Autonomous pipeline monitor for BreastMNIST debate dataset generation.
# Checks every 5 minutes, restarts on crash, builds dataset when done.
set -euo pipefail

WORKDIR="/home/rafael/ICL/MSc/Thesis/Code/project/breastMnist"
PYTHONPATH_VAL="/home/rafael/ICL/MSc/Thesis/Code/project"
VENV="$WORKDIR/.venv/bin/python"
RESUME="$WORKDIR/outputs/resume/debate_kg_adversarial_50_42.jsonl"
DATASET_DIR="$WORKDIR/data/breast/dataset"
LOG="$WORKDIR/outputs/monitor.log"
CHECK_INTERVAL=300   # seconds between checks

cd "$WORKDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

find_latest_output_dir() {
    # Return the most recently created Hydra output dir containing debate JSONs or a log
    find outputs -name "breast_main.log" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}' | xargs dirname 2>/dev/null || true
}

start_pipeline() {
    log "Starting debate pipeline (skip_judge=true, num_ctx=12288, k=3)..."
    PYTHONPATH="$PYTHONPATH_VAL" nohup "$VENV" -m debate_kg.breast_main \
        run=breast_adversarial data.num_samples=50 run.skip_judge=true \
        >> "$LOG" 2>&1 &
    echo $! > "$WORKDIR/outputs/pipeline.pid"
    log "Pipeline PID: $!"
}

pipeline_running() {
    if [[ -f "$WORKDIR/outputs/pipeline.pid" ]]; then
        local pid
        pid=$(cat "$WORKDIR/outputs/pipeline.pid")
        kill -0 "$pid" 2>/dev/null && return 0
    fi
    pgrep -f "debate_kg.breast_main" > /dev/null 2>&1 && return 0
    return 1
}

build_dataset() {
    local out_dir
    out_dir=$(find_latest_output_dir)
    log "Building dataset from $out_dir ..."
    PYTHONPATH="$PYTHONPATH_VAL" "$VENV" -m debate_kg.dataset.build_graph_dataset \
        --debate-dir "$out_dir" \
        --out-dir "$DATASET_DIR" \
        --k 3 --force 2>&1 | tee -a "$LOG"
    log "Dataset build complete. Output: $DATASET_DIR"
}

resume_count() {
    [[ -f "$RESUME" ]] && wc -l < "$RESUME" || echo 0
}

log "=== Monitor started ==="

# Start pipeline if not already running
if ! pipeline_running; then
    start_pipeline
fi

while true; do
    sleep "$CHECK_INTERVAL"

    done=$(resume_count)
    out_dir=$(find_latest_output_dir)
    log "Check: $done/50 samples done | output_dir=$out_dir"

    # Log last few lines of pipeline log
    if [[ -n "$out_dir" && -f "$out_dir/breast_main.log" ]]; then
        tail -5 "$out_dir/breast_main.log" | while read -r line; do
            log "  pipeline: $line"
        done
    fi

    # All done → build dataset and exit
    if [[ "$done" -ge 50 ]]; then
        log "All 50 samples complete!"
        build_dataset
        log "=== Monitor finished successfully ==="
        exit 0
    fi

    # Crashed → restart from checkpoint (resume file preserved)
    if ! pipeline_running; then
        log "Pipeline not running — restarting from checkpoint ($done/50 done)..."
        start_pipeline
    fi
done
