#!/bin/bash
# Watch every running debate and flag quality regressions as they happen.
#
# Three separate defects in this project reached 780 graphs before anyone read
# the transcripts: v3's single sentence covering 12% of claims, v4's agents
# citing list position rather than image content, and v5's agents producing
# byte-identical claims. All were visible in the first fifty samples. This logs
# the diagnostics that would have caught each of them, every few minutes.
#
# Usage: monitor_all.sh [interval_seconds]
set -u
REPO="$HOME/Multi-Agent-Cooperative-Debate-via-KGs"
B="$REPO/breastMnist/data/breast"
LOG="$REPO/experiments/monitor.log"
INT="${1:-240}"

while true; do
  {
    echo "================ $(date +%H:%M) ================"
    for V in v4 v5; do
      D="$B/debates_$V"
      [ -d "$D" ] || continue
      python3 - "$D" "$V" <<'PY'
import json, sys, glob, statistics as st
from collections import Counter
root, name = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{root}/*/debate_*.json"))
if not files:
    print(f"{name}: no debates yet"); raise SystemExit
ds = [json.loads(open(f).read()) for f in files]
cl = [c for d in ds for c in d["claims"]]
if not cl:
    print(f"{name}: no claims"); raise SystemExit
t = [c["text"].lower().strip() for c in cl]
uniq = len(set(t)) / len(t)
rounds = Counter(c["round_idx"] for c in cl)
lab = Counter(c["label"] for c in cl)
mal_frac = lab.get("MALIGNANT", 0) / len(cl)

# Do different images produce different claim sets? This is the single check
# that separates a debate reading the image from one reciting a list.
sets = [{c["text"].lower().strip() for c in d["claims"]} for d in ds[-40:]]
ov = [len(a & b) / len(a | b)
      for i, a in enumerate(sets) for b in sets[i+1:i+4] if a | b]
cross = st.mean(ov) if ov else float("nan")

# v5 only: are the two agents actually independent?
agent = ""
ids = {c["expert_id"] for c in cl}
if "agent_1" in ids:
    s = []
    for d in ds:
        a = {c["text"].lower() for c in d["claims"]
             if c["expert_id"] == "agent_1" and c["round_idx"] == 0}
        b = {c["text"].lower() for c in d["claims"]
             if c["expert_id"] == "agent_2" and c["round_idx"] == 0}
        if a and b:
            s.append(len(a & b) / len(a | b))
    if s:
        agent = f" agent_overlap={st.mean(s):.3f}"

flags = []
if uniq < 0.10:            flags.append("UNIQUENESS_COLLAPSE")
if cross > 0.45:           flags.append("CLAIMS_NOT_IMAGE_DEPENDENT")
if rounds.get(0, 0) > 4 * (rounds.get(1, 0) + rounds.get(2, 1)):
                           flags.append("DEBATE_NOT_HAPPENING")
if mal_frac < 0.05 or mal_frac > 0.95:
                           flags.append("LABELS_DEGENERATE")
if agent and float(agent.split("=")[1]) > 0.5:
                           flags.append("AGENTS_IDENTICAL")

print(f"{name}: n={len(ds)} claims/graph={len(cl)/len(ds):.1f} uniq={uniq:.3f} "
      f"cross={cross:.3f} mal={mal_frac:.2f} rounds={dict(sorted(rounds.items()))}{agent}")
if flags:
    print(f"   *** {' '.join(flags)} ***")
    print("   most repeated:")
    for txt, n in Counter(c["text"].strip() for c in cl).most_common(3):
        print(f"      {n:4d}x {txt[:70]}")
PY
    done
  } >> "$LOG" 2>&1
  sleep "$INT"
done
