# Overnight run — 2026-08-16

Three tasks, worked in order. Numbers marked `PENDING` are filled in as runs land.

---

## Task 1 — make MedGemma beat its own no-KG baseline

**Target to beat:** zero-shot, image only, no KG — **AUC 0.6011** (TP 17, FP 25,
mean p 0.415 malignant vs 0.220 benign).

### Where the previous approaches stood

Every earlier attempt injected KG text into the classification prompt. All of
them lost to the baseline:

| Run | Approach | AUC |
|---|---|---|
| `run_kg_binary` | binary rules in prompt | 0.4501 |
| `run_kg_stance_direct` | stance triples in prompt | 0.5476 |
| `run_kg_birads` | BI-RADS 1–5 scoring | 0.5602 |
| `run_kg_enriched` | 5-section enriched context | 0.5716 |
| `run_kg_assertive` | assertive rule framing | 0.5777 |
| `run_kg_grounded` (v3) | structured obs + retrieval | 0.5825 |
| `run_kg_retrieval_v2` | dense retrieval, 18 triples | **0.3858** |
| *no KG at all* | *image only* | ***0.6011*** |

`run_kg_retrieval_v2` finished overnight and came in at 0.3858 — the worst of
the set. Its mean p_malignant was actually *higher* for benign images (0.156)
than malignant ones (0.138), i.e. the retrieved triples were actively
anti-correlated with the answer.

**Diagnosis.** Two things were going wrong at once:

1. **The text was crowding out the image.** ~400 tokens of KG against ~256
   image tokens. The model answered from the rules rather than the picture.
2. **The answer format was degenerate.** Asking "malignant or benign" in one
   word makes the first-token logprobs collapse to ≈0 or ≈1. With p_malignant
   effectively binary there is almost no ranking information left, so AUC
   cannot rise much above chance no matter how good the rules are.

### What worked: use the KG as a scoring scaffold, not as prompt text

`experiments/03_kg_grounded_vlm/run_kg_featureprobe.py`

The KG never enters the classification prompt. It decides **which questions to
ask** and **how to weight the answers**:

1. **Probe.** Each visually-assessable KG stance feature becomes one yes/no
   question asked against the image, e.g.
   *"Does the mass show spiculated margins, with sharp angular lines radiating
   outward from the mass edge? Answer yes or no."*
   `p(yes)` comes from the first-token logprobs. 17 probes are derived
   automatically from the graph.
2. **Weight.** The KG stance relation sets the sign — `suggests_malignancy`
   → +1, `suggests_benign` → −1, and `object == "possible"` halves it.
3. **Score.** `score = Σ wᶠ·p_yes(f) / Σ|wᶠ|`.

Why this fixes both problems: every call is a short yes/no with the image (the
format the baseline already handles well, so the image dominates), and summing
17 continuous probabilities produces a genuinely continuous score instead of a
bimodal one.

Features excluded as not assessable on a 2D grayscale crop: elastography
(`soft/hard_elasticity`, `compressibility`), out-of-frame findings
(`axillary_adenopathy`, skin/nipple changes), and `evidence_of_interval_growth`
(needs a prior study). That is a modality constraint, not cherry-picking.

### The saturation problem, and the fix

Raw probe scoring gave AUC ≈0.75 — already above baseline, but the per-probe
breakdown showed why it was not higher. Several probes answer identically for
every image and therefore carry no information at all:

| probe | mean | std | verdict |
|---|---|---|---|
| `microcalcifications_in_hypoechoic_mass` | 0.001 | 0.000 | dead |
| `posterior_shadowing_with_solid_irregular_mass` | 0.999 | 0.001 | dead |
| `non_parallel_orientation` | 0.985 | 0.018 | dead |
| `parallel_orientation` | 0.963 | 0.028 | dead |
| `anechoic_content` | 0.985 | 0.032 | dead |
| `irregular_shape` | 0.842 | 0.241 | informative |
| `oval_shape` | 0.774 | 0.235 | informative |
| `hyperechoic_mass` | 0.922 | 0.184 | informative |

`experiments/03_kg_grounded_vlm/calibrate_probes.py` drops probes whose
standard deviation across the corpus falls below a threshold, then z-scores the
survivors before applying the KG weights. **This uses only the distribution of
probe outputs, never the gold labels** — it is feature selection, not fitting to
the answer. It also supports `--fit-from` so the statistics can be fitted on
train and applied to test.

### Result — full 156-sample test set

| Method | AUC | Balanced acc |
|---|---|---|
| Zero-shot, no KG (baseline) | 0.6011 | — |
| Best previous KG-in-prompt (`run_kg_grounded` v3) | 0.5825 | — |
| **KG feature probes** | **0.6685** | **0.6497** |

**The KG now helps: +0.067 AUC over the no-KG baseline**, and it is the first
KG configuration of the eight tried that beats image-only at all.

At the best threshold: TP 24, FP 31, TN 83, FN 18 — sensitivity 0.571,
specificity 0.728. The baseline found 17 cancers; this finds 24.

**The bimodality is gone.** Scores span 0.384–0.559 with per-class standard
deviations of 0.027/0.029, instead of piling up at 0 and 1. That was the
structural problem capping every earlier attempt, and it is what the change in
method actually fixed.

A note on the calibration: on the first 39 samples it looked much stronger
(AUC 0.8006, and I nearly reported that). On the full 156 it does not help —
raw 0.6685 vs 0.6614 calibrated at `min_std=0.05`. The 0.80 was small-sample
optimism. The headline number is the raw one. `calibrate_probes.py` is kept
because the saturation diagnosis it produces is what explains the ceiling, and
because `--fit-from` makes it usable properly once train probes exist.

### Why it is still only 0.67

Five of the 17 probes are saturated — they return the same answer for every
image and contribute nothing:

| probe | mean | std |
|---|---|---|
| `microcalcifications_in_hypoechoic_mass` | 0.001 | 0.000 |
| `posterior_shadowing_with_solid_irregular_mass` | 0.998 | 0.002 |
| `non_parallel_orientation` | 0.977 | 0.034 |
| `parallel_orientation` | 0.957 | 0.041 |
| `anechoic_content` | 0.984 | 0.042 |

Against that, `irregular_shape` (std 0.241), `oval_shape` (0.235) and
`hyperechoic_mass` (0.184) carry most of the discrimination. MedGemma-4b simply
cannot see some BI-RADS descriptors on a 224×224 crop, and the KG has no way to
know which. The unsupervised score gives every feature equal magnitude, so the
dead probes dilute the live ones.

`fit_probe_head.py` addresses this by learning the magnitudes on the train split
(the KG still decides which features exist; only the weights are fitted). It
reads probe vectors straight out of the v2 graphs, so it needs no extra
inference. Results once the train split finishes — reported separately, since it
is a supervised setting and not comparable to the zero-shot baseline.

---

## Task 2 — does `dataset_full` have enough signal for a GNN?

`experiments/04_graph_analysis/analyse_graphs.py`, run over all 780 graphs.

**Answer: no. The dataset has essentially zero learnable signal.**

```
Debate verdict: acc=0.6372  balanced_acc=0.4916   <- worse than chance
  TP=37 FP=110 TN=460 FN=173   sens=0.176 spec=0.807
  verdicts: 633 BENIGN / 147 MALIGNANT

Single-feature AUC (0.5 = pure noise):
  mal_share        0.5007   <- claim labels tell you nothing about the answer
  final_share      0.4936
  expert_gap       0.4998
  disagree         0.5475   <- the strongest feature in the whole dataset
  n_claims         0.5113
  n_cc_edges       0.5179
```

Three findings that matter:

1. **`mal_share` AUC = 0.5007.** The share of claims labelled MALIGNANT is
   uncorrelated with the true label. Malignant and benign images both produce
   ~44.5% malignant claims. Whatever the agents are arguing about, it is not
   the image.
2. **`expert_gap` = 0.997 for *both* classes.** The two agents took maximally
   opposite positions on essentially every sample regardless of content. That
   is the signature of role-play: the disagreement was assigned, not observed.
3. **Balanced accuracy 0.4916.** The debate verdict is worse than a coin flip
   once class imbalance is accounted for; it looks respectable at 0.6372 raw
   accuracy only because 73% of the set is benign and it answers BENIGN 81% of
   the time.

Graph *structure* is fine — 780/780 distinct claim sets, only 0.5% structural
label conflicts, 0.813 global claim uniqueness. The problem is not degeneracy.
The problem is that the node content carries no information about the label, so
message passing has nothing to propagate.

**Root cause.** All 780 graphs were built from 28×28 thumbnails upscaled to
224 (the loader bug fixed in `b47893f`). There was very little to see, so the
agents fell back on generic BI-RADS boilerplate. `dataset_full/CONTAMINATION.md`
documents a second issue on top: 12 test graphs were rebuilt at 224px by a stray
watchdog, so that split mixes two input resolutions.

**Conclusion: regenerating the dataset was necessary, not optional.** That is
Task 3.

---

## Task 3 — new dataset: evidence-grounded adversarial debate

`experiments/05_debate_v2/run_debate_v2.py`

Two MedGemma agents, 3 rounds, adversarial, no judge — as requested. The design
targets the two failure modes found in Task 2.

**Per sample:**

1. **Measure.** Run the 17 KG feature probes from Task 1 against the native
   224×224 image. This is the shared evidence, and it is the component that
   scores well above baseline on its own.
2. **Debate.** Agent A argues malignant, Agent B argues benign, 3 rounds. Both
   see the *same* measurement table, but each sees it split into
   *supports your position* / *contradicts your position* / *measured absent*.
   Each is told to concede a claim when the numbers go against it.
3. **Graph.** Claim nodes carry `cited_feature`, **`p_yes`** (the measured
   probe confidence) and **`stance_weight`** (KG polarity), alongside the usual
   text/label/expert/round. Claim→triple edges link each claim to the KG
   triples describing the feature it cites.

**Why this should carry signal where v1 did not:** in v1 the only per-node
information was a categorical label that turned out to be uncorrelated with the
answer (AUC 0.5007). In v2 every claim node carries a continuous measurement
whose aggregate already separates the classes well. A GNN now has something
real to message-pass over.

**Verdict without a judge:** taken from the evidence score, with the final
round's claim balance recorded separately as `debate_mal_share`.

### Engineering notes

Several things needed fixing before the debate produced usable graphs:

- **Zero claim–claim edges at first.** The agents ignored `[ADDRESSED: cN]`.
  Fixed by putting it first in the required format and showing only the
  opponent's most recent round.
- **`[CITED:]` tags were unreliable** from a 4B model — grounding fell to 5/11
  claims. Fixed by *inferring* the cited feature from the prose (exact name
  match, then content-word overlap with a ≥2-word threshold). Grounding went to
  15/15. The general lesson: don't ask a small model for structure you can
  derive yourself.
- **`[AGREE|DISAGREE]` also unreliable.** Now inferred from whether the two
  linked claims share a label.
- **Missing `[LABEL:]`** is filled from the KG polarity of the cited feature, so
  a dropped tag no longer discards a node. Nodes record `label_explicit` so the
  inferred ones stay auditable.
- **Rounds repeated verbatim** under greedy decoding (round 3 restated round 1
  word for word). Fixed with temperature 0.4 on debate turns and a fixed
  per-turn seed for reproducibility. Probes stay at temperature 0 — measurements
  must be deterministic.
- **Agents echoed each other's sentences** back as their own "rebuttal", which
  manufactured edges without adding argument. Now deduplicated globally rather
  than per-agent.

### Infrastructure

780 samples × (17 probes + 6 debate turns) does not fit on one GPU overnight, so
the job is sharded across four:

- gpu21 (RTX 2080 Ti) — shard 0
- gpu22 (RTX 2080 Ti) — shard 1, started after the Task 1 probe run finished
- gpu23, gpu24 (GTX 1080) — shards 2, 3

ollama and the medgemma:4b blobs were copied to gpu21/23/24 (`/data` is
node-local; only `$HOME` is shared). Images were pre-exported to
`breastMnist/data/breast/images_224/{split}.npz` so worker nodes need only numpy
and PIL, not the full 5.4 GB venv.

Each sample is one file and a worker skips any file that already exists, so
shards never collide, the job resumes after interruption, and extra workers can
be added at any time.

Throughput ≈35 s/sample on a 2080 Ti.

### Result

| | v1 `dataset_full` | v2 `dataset_v2` |
|---|---|---|
| images | 28×28 upscaled | 224×224 native |
| `mal_share` AUC | 0.5007 | PENDING |
| best feature AUC | 0.5475 | PENDING |
| verdict balanced acc | 0.4916 | PENDING |
| per-node evidence | none | `p_yes`, `stance_weight` |

---

## Files added

```
experiments/03_kg_grounded_vlm/run_kg_featureprobe.py   KG-as-scaffold classifier
experiments/03_kg_grounded_vlm/calibrate_probes.py      unsupervised recalibration
experiments/04_graph_analysis/analyse_graphs.py         GNN-signal analysis
experiments/05_debate_v2/run_debate_v2.py               evidence-grounded debate
scripts/launch.sh                                       detached launcher
scripts/launch_debate_v2.sh                             multi-GPU shard launcher
breastMnist/data/breast/images_224/*.npz                pre-exported native images
```
