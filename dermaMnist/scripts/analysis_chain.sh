#!/bin/bash
# Run the framework analysis, then the learning curve, then re-run BOTH once the
# debate corpus completes. Chained so nothing waits on a human.
#
# The probe-only pass is not a placeholder: it IS v3's `no-debate` control and a
# complete result on its own. The debate pass adds the citation channel.
set -u
REPO=/homes/rm2125/Multi-Agent-Cooperative-Debate-via-KGs
DER=$REPO/dermaMnist; SRC=$DER/src/derma_kg; LOG=$DER/logs; RES=$DER/results
V=/data/rm2125/venv/bin/python
cd "$SRC"

echo "$(date +%H:%M) === pass 1: probe-only (no debate channel) ==="
$V -u run_derma.py --out "$RES/derma_probeonly.json" > "$LOG/derma_probeonly.log" 2>&1
echo "$(date +%H:%M) pass 1 analysis done"
$V -u learning_curve.py --out "$RES/derma_curve_probeonly.json" > "$LOG/curve_probeonly.log" 2>&1
echo "$(date +%H:%M) pass 1 curve done"

echo "$(date +%H:%M) waiting for the debate corpus..."
while true; do
  n=$(cat "$DER/data/derma/debates_d1"/*.jsonl 2>/dev/null | wc -l)
  [ "$n" -ge 10015 ] && break
  sleep 300
done
echo "$(date +%H:%M) debate corpus complete ($n)"

echo "$(date +%H:%M) === pass 2: probes + debate ==="
$V -u run_derma.py --debate-root "$DER/data/derma/debates_d1" \
   --out "$RES/derma_nominal.json" > "$LOG/derma_full.log" 2>&1
echo "$(date +%H:%M) pass 2 analysis done"
$V -u learning_curve.py --debate-root "$DER/data/derma/debates_d1" \
   --out "$RES/derma_curve.json" > "$LOG/curve_full.log" 2>&1
echo "$(date +%H:%M) pass 2 curve done"

echo "$(date +%H:%M) === appearance-adjacency arm (the ICDR construction, which works here) ==="
$V -u run_derma.py --debate-root "$DER/data/derma/debates_d1" --mediators appearance \
   --out "$RES/derma_nominal_appearance.json" > "$LOG/derma_appearance.log" 2>&1
echo "$(date +%H:%M) ALL DONE"
