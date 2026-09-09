#!/bin/bash
# Autonomous overnight run. Two lanes so the fleet and the CPU never wait on
# each other: the judge arms need GPUs, everything else is NumPy/torch on CPU.
# Each step is skipped if its output already exists, so a restart resumes.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
V=/data/rm2125/venv/bin/python
L=$REPO/shared/logs; R=$REPO/shared/results
mkdir -p "$L" "$R"
have () { [ -s "$1" ]; }

# ---------------- GPU lane: judge arms -----------------------------------
gpu_lane () {
  echo "$(date +%H:%M) [gpu] waiting for the derma debate corpus"
  while [ "$(cat $REPO/dermaMnist/data/derma/debates_d1/*.jsonl 2>/dev/null | wc -l)" -lt 10015 ]; do
    sleep 300
  done
  echo "$(date +%H:%M) [gpu] debate corpus done"

  # retina first: 1600 images, cheap, and it validates the judge before the
  # 10,015-image derma run commits hours to it
  for sp in test val train; do
    echo "$(date +%H:%M) [gpu] judge retina $sp"
    NODES="10 21 13 23 24 11" STAGE=judge TREE=retina SPLIT=$sp \
      $REPO/shared/judge_dispatch.sh >> "$L/judge_retina.log" 2>&1
  done
  for sp in test val train; do
    echo "$(date +%H:%M) [gpu] judge derma $sp"
    NODES="10 21 13 23 24 11" STAGE=judge TREE=derma SPLIT=$sp \
      $REPO/shared/judge_dispatch.sh >> "$L/judge_derma.log" 2>&1
  done
  echo "$(date +%H:%M) [gpu] LANE DONE"
}

# ---------------- CPU lane: analyses -------------------------------------
cpu_lane () {
  # wait for the existing derma analysis chain so we do not fight it for cores
  while pgrep -f analysis_chain.sh >/dev/null; do sleep 300; done
  echo "$(date +%H:%M) [cpu] derma analysis chain finished"

  if ! have "$R/transfer_matrix.json"; then
    echo "$(date +%H:%M) [cpu] 3-way transfer matrix"
    $V -u "$REPO/shared/transfer_matrix.py" > "$L/transfer_matrix.log" 2>&1
  fi
  if ! have "$R/prior_shuffle.json"; then
    echo "$(date +%H:%M) [cpu] prior-shuffle control (vocabulary vs semantics)"
    $V -u "$REPO/shared/prior_shuffle.py" > "$L/prior_shuffle.log" 2>&1
  fi
  if ! have "$REPO/retinaMnist/results/baseline_curves.json"; then
    echo "$(date +%H:%M) [cpu] baseline learning curves (v3 / v3-nograph / B2 / B1)"
    (cd "$REPO/retinaMnist/src/retina_kg" && $V -u baseline_curves.py) \
      > "$L/baseline_curves.log" 2>&1
  fi
  echo "$(date +%H:%M) [cpu] LANE DONE"
}

gpu_lane & GP=$!
cpu_lane & CP=$!
wait $GP $CP
echo "$(date +%H:%M) NIGHT CHAIN COMPLETE"
