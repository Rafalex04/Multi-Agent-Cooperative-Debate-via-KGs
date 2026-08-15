#!/usr/bin/env bash
# 12-sample resolution probe: run debates at native 224px, build graphs, compare
# against the pre-fix (28px) baseline on the same 12 images.
#
# Single variable changed vs the original 780-graph run: image resolution.
# Prompts, model, KG, k, and max_rounds are all untouched.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(cd .. && pwd)"
VENV=".venv/bin/python"
export PYTHONPATH="$ROOT"

LOG="outputs/probe12.log"
OUT_DIR="data/breast/dataset_probe12"
BEFORE_DIR="data/breast/dataset_probe12_BEFORE/test/graphs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p outputs
log "=== probe12 START — 12 samples, native 224px, skip_judge ==="

# Marker to identify dirs created by this run (mtime comparison is unreliable
# because Hydra creates the dir before writing anything into it).
STAMP="$(mktemp)"

$VENV -m debate_kg.breast_main \
    run=breast_adversarial \
    data.split=test \
    data.indices_path=data/breast/probe12_indices.json \
    data.num_samples=12 \
    run.skip_judge=true >> "$LOG" 2>&1
RC=$?
log "pipeline exit: $RC"

# Locate the output dir this run produced: newest dir holding debate JSONs
# that is newer than STAMP.
RUN_DIR="$(find outputs -name 'debate_sample-*.json' -newer "$STAMP" -printf '%h\n' 2>/dev/null \
           | sort -u | tail -1)"
rm -f "$STAMP"

if [ -z "$RUN_DIR" ]; then
    log "ERROR: no debate JSONs produced — nothing to build. See $LOG"
    exit 1
fi
log "run dir: $RUN_DIR ($(ls "$RUN_DIR"/debate_sample-*.json 2>/dev/null | wc -l) debate JSONs)"

# The builder SKIPS any dir without a split marker — this is the step that
# silently produced zero graphs before.
touch "$RUN_DIR/.split_test"
log "tagged $RUN_DIR as split=test"

$VENV -m debate_kg.dataset.build_graph_dataset \
    --debate-dir "$RUN_DIR" \
    --out-dir "$OUT_DIR" \
    --k 3 --force >> "$LOG" 2>&1
log "builder exit: $? | graphs: $(ls "$OUT_DIR"/test/graphs 2>/dev/null | wc -l)"

log "=== COMPARISON ==="
$VENV scripts/compare_probe12.py \
    --before "$BEFORE_DIR" \
    --after  "$OUT_DIR/test/graphs" 2>&1 | tee -a "$LOG"

log "=== probe12 DONE ==="
