# Multi-Agent Cooperative Debate via Knowledge Graphs

**Do vision-language agents debating under a medical ontology classify images
better than the same model asked once?**

Measured across three MedMNIST datasets, three clinical ontologies, and three
label spaces (binary, 5-level ordinal, 7-class nominal), on a local GPU cluster,
at a hard $0 budget — no paid APIs.

**The answer is no**, and the interesting part is *why*. This repository holds the
full apparatus that established that: the debate orchestrators, the probe
perception layer, five graph-neural architectures, the controls that killed four
of them, two re-implemented baselines from the literature, and every result file.

---

## Table of contents

1. [What this project found](#1-what-this-project-found)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Repository layout](#3-repository-layout)
4. [Requirements and install](#4-requirements-and-install)
5. [Data preparation](#5-data-preparation)
6. [Framework A — the debate pipeline (Hydra)](#6-framework-a--the-debate-pipeline-hydra)
7. [Framework B — the measurement stack on BreastMNIST](#7-framework-b--the-measurement-stack-on-breastmnist)
8. [Framework C — RetinaMNIST / ICDR (ordinal)](#8-framework-c--retinamnist--icdr-ordinal)
9. [Framework D — DermaMNIST / dermoscopy (nominal)](#9-framework-d--dermamnist--dermoscopy-nominal)
10. [Framework E — cross-dataset experiments](#10-framework-e--cross-dataset-experiments)
11. [Framework F — HINPool graph-pooling benchmark](#11-framework-f--hinpool-graph-pooling-benchmark)
12. [The numbered experiment index (01–26)](#12-the-numbered-experiment-index-0126)
13. [Running on a GPU fleet](#13-running-on-a-gpu-fleet)
14. [Personalising the framework](#14-personalising-the-framework)
15. [Conventions, metrics and gotchas](#15-conventions-metrics-and-gotchas)
16. [Where every result lives](#16-where-every-result-lives)

---

## 1. What this project found

Five defensible claims, each traceable to a named results file (§16).

| # | Claim | Evidence | Strength |
|---|---|---|---|
| **C1** | The zero-shot KG pipeline **generalises ~3× better** than a supervised CNN | out-of-distribution AUC loss 0.093 vs 0.287; the ordering *reverses* off-distribution | Strong |
| **C2** | **Debate adds nothing.** It loses to sampling the same model five times | 0.6499 vs 0.7496, P=0.016 | Strong |
| **C3** | **KG topology is a well-powered null**, not an underpowered positive | 4 independent implementations, each matching its shuffled-KG control; the effect *falls* as n grows (P=0.873); on ICDR over 5 paired seeds it is **negative** | Strong |
| **C4** | **Perception dominates architecture** by an order of magnitude | +0.087 AUC from a segmentation prior vs ~+0.01 from any architectural choice | Strong |
| **C5** | The per-node **feature map transfers across ontologies** | breast→retina retains **99.4%** of the target-trained ceiling, **+4.35 sd** over a frozen random trunk | Moderate |

**The one-line summary.** The knowledge graph pays for itself through its
**finding vocabulary** — the list of things the probes ask the model to look for —
and *not* through its **edges**. Debate is neutral-to-harmful at every scale
tested. What matters most is what the model can actually see.

This is a negative-result project with a positive generalisation claim. Its
contribution is that all three findings were **measured to a decision** against
proper controls, rather than asserted.

---

## 2. How the pipeline works

```
                        medical image (224×224 RGB)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
  ① PERCEPTION               ② REASONING                  ③ ONTOLOGY
  first-token logprob        multi-agent debate            ACR BI-RADS / ICDR /
  probes, one per            over N rounds, agents         dermoscopy graph:
  ontology finding           cite [CITED:t_NNN]            triples, definitions,
        │                           │                      criteria, priors
        │                           │                           │
        └───────────┬───────────────┴───────────────┬───────────┘
                    ▼                               ▼
            (n, F, d) NODE TENSOR            FINDING ADJACENCY A±
            F = findings, d = channels:      signed, lesion-mediated
            p(present) · argument mass       Adamic-Adar over the KG
            · net stance · ontology prior
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                        ④ ThothGNN v3  (59 parameters)
                        message passing over the finding
                        graph, hand-written gradients
                                    │
                                    ▼
                        ⑤ READOUT → verdict + metrics
```

### ① The perception layer

Everything rests on one measurement primitive:

```
p_yes = P("yes") / (P("yes") + P("no"))      read from the top-20 logprobs
```

One probe per ontology finding, asked of a frozen vision-language model. Two
phrasing arms were tested — **P3 (verification)** and **P2 (negated)**. P2 was
included to cancel acquiescence bias, but it cancels the *content* along with the
bias and scored **0.4452 externally — below chance**. **Always use P3 alone.**
Any number in the record dated before 2026-08-26 used the pooled pair and is
therefore built on a feature block that is half anti-predictive.

### ② The debate

Two or more agents write structured `[CLAIM]` blocks with inline `[CITED:t_NNN]`
provenance and `[ADDRESSED:<id>][AGREE|DISAGREE]` stance tags. Each claim becomes
a `DebateNode`; edges are parsed from the tags, never inferred by a second LLM.
Five protocol generations exist; **v5 (open-stance) is the one to use**.

Note the naming collision: `debate v2/v3/v4/v5` are **corpus generators**;
`ThothGNN v1/v2/v3` is the **model**. Different axes.

### ③ The ontology

Each dataset carries a graph of triples plus definitions. From it are derived: the
finding set the probes ask about, a signed prior per finding, and a
finding→finding adjacency built by Adamic-Adar over shared mediators.

### ④ The model

**ThothGNN v3** — 59 parameters, hand-written gradients, 800 epochs, 12.8 seconds
on a single CPU core. Its whole point is that it is small enough that a null
result cannot be blamed on capacity.

Every architecture is measured against three controls, and this is the part that
matters most:

| Control | What it isolates |
|---|---|
| **no-graph** (`A± = 0`) | message passing, separated from capacity |
| **shuffled-KG** (× 5 draws, degree-matched) | the *ontology*, separated from mere graph-ness |
| **no-debate channel** | the citation merge, separated from the probes |

---

## 3. Repository layout

```
.
├── README.md                  ← this file, the only prose document
├── .gitignore
│
├── breastMnist/               Framework A+B — BreastMNIST + FEVER (Hydra app)
│   ├── conf/                  Hydra configs: run groups, judges, KG schema
│   ├── src/debate_kg/         the debate orchestrator, KG, judges, eval
│   ├── data/breast/           ontology, corpora, debate graphs, image npz
│   ├── tests/                 pytest suite
│   └── outputs/               Hydra run outputs (gitignored)
│
├── experiments/               Framework B — 26 self-contained studies
│   ├── 01_zeroshot_vlm_classification/  … through …
│   ├── 26_baselines/                    each with scripts + results/
│   └── artifacts/             rendered results ledger (HTML)
│
├── retinaMnist/               Framework C — RetinaMNIST / ICDR, 5-level ordinal
│   ├── conf/ontology_retina.yaml        every ICDR relation name, one file
│   ├── src/retina_kg/                   pack, probes, debate, ordinal heads
│   ├── data/retina/                     ICDR pack + corpora
│   ├── scripts/                         fleet dispatchers
│   └── results/                         all retina result JSON + corpora
│
├── dermaMnist/                Framework D — DermaMNIST, 7-class nominal
│   ├── conf/ontology_derma.yaml
│   ├── src/derma_kg/
│   ├── data/derma/
│   ├── scripts/
│   └── results/
│
├── shared/                    Framework E — experiments spanning all three trees
│   ├── transfer_matrix.py     3×3 cross-ontology transfer
│   ├── prior_shuffle.py       is the ontology prior doing any work?
│   ├── run_judge.py           LLM judge over any tree
│   └── results/
│
├── hinpool-bench/             Framework F — standalone HINPool reproduction
│
├── data/external/             BUS-BRA + BrEaST external validation npz
└── scripts/                   cluster launchers, monitors, node provisioning
```

**Path convention.** Experiment scripts resolve their own imports via
`sys.path` relative to `__file__`. **Run each script from its own directory**
(`cd experiments/17_hetgnn && python run_thothgnn3.py`). The retina and derma
trees centralise every path in `src/*/paths.py`, so moving those trees is a
one-line change.

---

## 4. Requirements and install

### 4.1 Software

- **Python 3.11+**
- **[Ollama](https://ollama.com)** — serves every model; there is no paid-API path
- **CUDA** optional. The analysis stack is CPU-only by design; only inference
  wants a GPU.

### 4.2 Install

```bash
git clone <this-repo> Multi-Agent-Cooperative-Debate-via-KGs
cd Multi-Agent-Cooperative-Debate-via-KGs/breastMnist

uv sync                  # recommended
# or:  pip install -e .

cp .env.example .env      # fill in any keys you need; none are required for $0 mode
```

`uv sync` installs everything the whole repository needs: `numpy`, `scikit-learn`,
`torch`, `torch-geometric`, `sentence-transformers`, `hydra-core`, `medmnist`,
`networkx`, `pydantic`, `requests`, `pillow`, `datasets`, `SPARQLWrapper`,
`pyyaml` (read by every ontology config under `retinaMnist`, `dermaMnist` and
`shared/`).

PyTorch is only needed for the ResNet baseline (`02`), the BiomedCLIP scorer
(`21`) and HINPool. Install it separately with the CUDA index URL matching your
cluster if the default wheel is wrong for you.

### 4.3 Models

```bash
ollama pull qwen3-vl:8b-instruct   # the workhorse: every probe and debate call
ollama pull minicpm-v:8b           # heterogeneous debate partner
ollama pull gemma3:12b             # second lineage for the B2 baseline
ollama pull medgemma:latest        # the MedGemma-4B-IT judge

# FEVER pipeline only (the three-judge panel):
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

Every runner takes `--url` (default `http://localhost:11434/api/chat`), so you can
point any script at any node.

### 4.4 Verify the install

```bash
cd breastMnist && pytest -m "not network"       # the unit suite, no LLM needed
cd ../retinaMnist/src/retina_kg
python ordinal.py            # gradient check, worst error must be ~1e-9
python test_reduction.py     # exact-reduction gate, must print 0.000e+00
```

Those two gates run against **real corpus data**, not fixtures. If
`test_reduction.py` fails, the ordinal head has drifted away from the binary model
whose result it extends, and no ordinal number is trustworthy.

---

## 5. Data preparation

### 5.1 BreastMNIST (in-domain, binary)

No separate step is needed: `debate_kg.data.breastmnist.load_samples` fetches the
split through `medmnist` on first use, and every runner calls it. To warm the
cache ahead of a fleet run, run any smoke command from §6.1. **Splits are fixed and never regenerated:**
train 546 / val 78 / test 156. The test split is 27% malignant, so always
answering "benign" scores 0.7308 accuracy — which is why **balanced accuracy is
the headline metric everywhere**.

Label convention, MedMNIST v2: **`0 → MALIGNANT`, `1 → BENIGN`. MALIGNANT is the
positive class.**

### 5.2 RetinaMNIST (5-level ordinal)

```bash
cd retinaMnist/src/retina_kg
python prepare_retina.py                # → data/retina/images_224/{train,val,test}.npz
python icdr_pack.py --print-findings    # → knowledge_graph.json, definitions.json,
                                        #   criteria.json, findings.json, lexicon.json
```

Labels are the **published ICDR grade 0–4**; nothing is binarised at export. The
binary "referable DR" reduction (grade ≥ 2) is applied at analysis time only.

```
train 1080  grades [486, 128, 206, 194, 66]   referable 0.431
val    120  grades [ 54,  12,  28,  20,  6]   referable 0.450
test   400  grades [174,  46,  92,  68, 20]   referable 0.450
```

### 5.3 DermaMNIST (7-class nominal)

```bash
cd dermaMnist/src/derma_kg
python prepare_derma.py --medmnist-root /path/to/cache
python derma_pack.py --print-findings
```

Splits: train 7007 / val 1003 / test 2005. 55 dermoscopic findings derived from
the ontology.

> **Cluster warning.** The DermaMNIST npz are 1,041 MB (train alone is 728 MB) and
> **must live on NFS home, not `/data`**. `/data` is node-local; compute nodes
> cannot see it, and the symlink dangles on every worker.

### 5.4 External validation sets (BUS-BRA, BrEaST)

```bash
cd experiments/16_external
python prepare_busbra.py --pads 2,15      # → data/external/busbra_pad{2,15}_224.npz
python prepare_breast.py --pad 2 --tag pad2
cd ../23_masktest && python prepare_busbra_masked.py --mode ring   # and: dim
```

BUS-BRA: 1875 images / 1064 patients, 4 scanners, biopsy-proven, from Brazil.
**Folds split on case**, so two views of one lesion cannot straddle a boundary,
and every AUC is reported at patient level as well as image level.

The raw 128 MB archive is gitignored (over GitHub's hard limit); the derived npz
are tracked.

### 5.5 Image size — the single most expensive bug in the project

Images **must** be loaded at native MedMNIST+ resolution and never upscaled.
`load_samples` passes `size=224` explicitly; omitting it silently yields 28×28 in
`medmnist ≥ 3.0`. An entire early run fed the VLM 28×28 thumbnails
LANCZOS-upscaled to 224 — 64× fewer real pixels — and BI-RADS margin character
and echo texture are simply unresolvable at that size. Every loader now hard-exits
on the wrong shape and logs `source WxH -> output WxH`.

---

## 6. Framework A — the debate pipeline (Hydra)

The original orchestrator. Two use cases share one core (debate loop,
`DebateNode`/`DebateEdge` graph, consensus check, LLM client layer).

### 6.1 BreastMNIST — six ablation modes

Each mode isolates one ingredient: debate, graph-structured verdicts, the domain
KG, adversarial stance assignment.

| Mode | Run config | Debate | KG | Verdict |
|---|---|---|---|---|
| `single_agent` | `breast_single` | no | no | parsed `[LABEL][CONFIDENCE]` |
| `debate_opinion` | `breast_debate_opinion` | free-text, multi-round | no | judge picks the winner |
| `debate_graph` | `breast_debate_graph` | structured `[CLAIM]` | no | weighted node-score vote |
| `debate_kg` | `breast_debate_kg` | structured `[CLAIM]` | **yes** | weighted node-score vote |
| `debate_kg_adversarial` | `breast_adversarial` | fixed opposing stances | yes | weighted node-score vote |
| `adaptive_adversarial` | `breast_adaptive` | conditional escalation | yes | confidence-gated |

```bash
cd breastMnist

python -m debate_kg.breast_main run=breast_single data.num_samples=2   # smoke test
python -m debate_kg.breast_main run=breast_single                      # the 50-sample set
python -m debate_kg.breast_main run=breast_debate_opinion
python -m debate_kg.breast_main run=breast_debate_graph
python -m debate_kg.breast_main run=breast_debate_kg
python -m debate_kg.breast_main run=breast_adversarial
python -m debate_kg.breast_main run=breast_adaptive
```

All six modes draw from the same fixed 50 test indices in
`data/breast/ablation_indices.json`. **Never regenerate that file** — a valid
cross-stage comparison depends on identical inputs.

Any Hydra key can be overridden inline:

```bash
python -m debate_kg.breast_main run=breast_debate_kg \
    debate.max_rounds=5 debate.temperature=0.6 \
    model.expert=qwen3-vl:8b-instruct judges=medgemma
```

**Output** (per run, to `outputs/<date>/<time>/`):

- `metrics.json` — accuracy, sensitivity, specificity, AUC, convergence rate,
  mean rounds, per-sample records
- `debate_sample-<id>.json` — nodes, edges, judge scores, verdict per sample
- `outputs/resume/<mode>_<n>_<seed>.jsonl` — incremental progress; re-running the
  same command **resumes** after a crash instead of restarting

### 6.2 FEVER fact-checking

Two agents grounded in **disjoint** Wikidata subgraphs debate a FEVER claim.
Disagreement signals incomplete knowledge, so the KGs are merged between rounds
and the loop repeats. A panel of three judges from different model families
scores each claim node; a weighted vote produces the verdict
(SUPPORTS = +1, NEI = 0, REFUTES = −1).

```bash
cd breastMnist

python -m debate_kg.main run=smoke                                      # 1 claim
python -m debate_kg.main run=fever_100 run.num_claims=5 run.mode=debate
python -m debate_kg.main run=fever_100 run.num_claims=5 run.mode=single_expert
python -m debate_kg.main run=fever_100 run.num_claims=5 \
    "hydra.verbose=[debate_kg.debate.orchestrator]"                     # verbose
```

Outputs: `claims_index.json`, `debate_<id>_<slug>.json`,
`judge_scores_<id>_<slug>.json`, `kgs/<id>_<slug>/round_<n>/{a,b}.json`,
`metrics.json`.

### 6.3 Generating a debate corpus at scale

The Hydra app is for interactive runs. Corpus generation uses the standalone,
shardable, resumable runners:

```bash
cd experiments/10_debate_v5
python run_debate_v5.py \
    --model qwen3-vl:8b-instruct \
    --split train --rounds 3 --temperature 0.8 --image-size 224 \
    --num-shards 16 --shard 0 \
    --url http://gpu07:11434/api/chat \
    --out-dir ../../breastMnist/data/breast/debates_v5q
```

One sample is one output record, and a worker **skips any index already written**.
So shards never collide, the job is resumable, and you can add workers at any time
— run with `--num-shards 1` as a sweeper to fill whatever the shards missed.

Earlier protocol generations are preserved for the record: `05_debate_v2`,
`07_debate_v3`, `09_debate_v4`. Use v5 for new work.

### 6.4 Tests

```bash
cd breastMnist
pytest                    # everything
pytest tests/debate       # one module
pytest -m "not network"   # skip live Wikidata calls
```

---

## 7. Framework B — the measurement stack on BreastMNIST

This is the path that produced the project's real results. Four stages.

### Stage 1 — measure the probes (needs a GPU fleet)

```bash
cd experiments/15_kggnn
python run_probes.py \
    --model qwen3-vl:8b-instruct \
    --split test --phrasing 3 --tag probep3 \
    --image-size 224 \
    --num-shards 16 --shard 0 \
    --url http://gpu07:11434/api/chat \
    --out-dir results
```

Repeat for `--split train` and `--split val`. `--phrasing 3` is P3
(verification); `--phrasing 2` is the negated arm, kept only for the record.

### Stage 2 — build the finding graph

```bash
python kg_graph.py         # lesion-mediated Adamic-Adar over the BI-RADS triples
python kg_structure.py     # inspect degree, isolation, edge weights
```

### Stage 3 — fit ThothGNN v3 against its controls

```bash
cd ../17_hetgnn
python run_thothgnn3.py --phrasings p3 --out results/thothgnn3_p3.json
```

Prints the real-KG arm next to `no-graph`, `shuffled-KG × 5` and `no-debate`.
Hyperparameters are selected by 5-fold CV **on train only**; test is evaluated
**once per arm**. bAcc thresholds are swept on train+val and applied unchanged.

Reference result: test AUC **0.8033** real KG vs **0.8051 ± 0.0109**
shuffled — the null that decided C3.

### Stage 4 — external validation

```bash
cd ../16_external
python freeze_models.py      # freeze on BreastMNIST, evaluate on BUS-BRA — TRUE transfer
python resnet_external.py    # the supervised ResNet-18 comparator
python analyse_e1.py && python analyse_e2.py

cd ../17_hetgnn
python external_merged.py --phrasings p3 --arm full --out results/em_full_both.json
python external_merged.py --phrasings p3 --arm mask --out results/em_mask_both.json
```

> **Never mix these two numbers.** `external_merged.py` reports **within-BUS-BRA
> 5-fold CV**; `freeze_models.py` reports **frozen-on-breast true transfer**. They
> rank the probe arms *oppositely*. That reversal is the transfer-gap law
> (`22_transfer/gap_law.py`, r=0.994), not a contradiction — but quoting one as
> the other is a reporting error.

### The alternative architectures

All four were measured and all four are nulls. Reproduce any of them:

```bash
cd experiments/15_kggnn
python kg_laplacian.py --debates ../../breastMnist/data/breast/debates_v5q --out results/lap.json
python probe_gnn.py    --debates ../../breastMnist/data/breast/debates_v5q --out results/probegnn.json
python run_thoth_kg.py --debates ../../breastMnist/data/breast/debates_v5q --hidden 16 --seeds 5 --out results/thothkg.json
python kgprior_gnn.py  --debates ../../breastMnist/data/breast/debates_v5q --hiddens 8,16 --seeds 5 --out results/kgprior.json
```

Each matches its own shuffled control almost exactly — 0.8137/0.8137,
0.7913/0.7903, 0.7863/0.7865, 0.7602/0.7573. Four implementations, one null.

---

## 8. Framework C — RetinaMNIST / ICDR (ordinal)

The replication that turns a single measurement into a finding: a second dataset,
a second ontology, a second modality, and a five-level ordinal label.

### 8.1 Build the pack, then check the gates

```bash
cd retinaMnist/src/retina_kg
python prepare_retina.py
python icdr_pack.py --print-findings
python kg_adjacency.py                       # inspect the derived adjacency
python ordinal.py && python test_reduction.py   # both correctness gates
```

**The 16-finding set is derived from the ontology, not hand-written.** The breast
set needed three hand-maintained tables (polarity, not-visible, phrasing);
automating them there cost 0.6685 → 0.5750. ICDR states enough about itself that
all three fall out of the graph:

- **polarity → ordinal level**, by relation priority `pathognomonic_for_level` >
  `defining_finding_of` > `establishes_level` > `minimum_level_for_presence` >
  `compatible_with_level`. The prior is `(level − 1.5)/2.5`, centred so the sign
  boundary *is* the referable-DR threshold and no finding sits at zero.
- **not-visible → `visible_on` / `located_in`.** Findings needing gonioscopy,
  slit-lamp or stereo imaging are dropped. Fluorescein is deliberately *not*
  excluded — removing microaneurysms would delete the grade-1 criterion outright.
- **phrasing → appearance triples, then definition lead, then bare name.**

**Label leakage is filtered.** 8 of 17 ICDR definitions lead with the grade the
lesion establishes ("sufficient for severe NPDR"), which would hand the VLM the
answer. Sentences carrying grading vocabulary are stripped; after filtering,
**0 of 16** descriptions mention a grade.

### 8.2 Generate the corpora (fleet)

```bash
# perception — reuses the breast probe runner, parameterised not forked
cd experiments/15_kggnn
python run_probes.py --model qwen3-vl:8b-instruct \
    --npz      ../../retinaMnist/data/retina/images_224/test.npz \
    --findings ../../retinaMnist/data/retina/probe_findings.json \
    --lexicon  ../../retinaMnist/data/retina/lexicon.json \
    --split test --phrasing 3 --tag retina_p3 \
    --num-shards 16 --shard 0 --out-dir ../../retinaMnist/results

# reasoning
cd ../../retinaMnist/src/retina_kg
python run_debate_retina.py --model qwen3-vl:8b-instruct \
    --split test --num-shards 16 --shard 0
```

### 8.3 The experiment

```bash
python run_ordinal.py --tags retina_p3 --heads A,C     # → results/ordinal.json
python run_seeds.py                                    # 5 paired seeds
python paired_ac.py && python paired_multiclass.py     # paired bootstraps
python medmnist_compare.py                             # vs published baselines
python baseline_curves.py
```

### Why heads A and C

Five ordered grades have **four** cut points, not five. Both heads decompose the
problem identically — into "is the grade above c?" for c = 0,1,2,3 — and differ
only in parameter sharing:

- **A — cumulative-link (CORAL).** One trunk, four cut points.
  `P(y>c) = σ(s − θ_c)`. **63 parameters.**
- **C — stacked.** Four independent trunks, nothing shared. **236 parameters.**
- **A is C with proportional odds imposed.** Five copies would be one-vs-rest — a
  different model that discards the ordering.

**Result: it depends on the metric, and C wins the dataset's own.** A leads the
binary/ranking measures (refAUC 0.9043 vs 0.8959); C leads all four multiclass and
class-balanced ones. Only one comparison resolves at 95%: **macro-OVR AUC — the
metric MedMNIST itself reports — where C is ahead by 0.0285 with the CI clear of
zero.** If you want a referable-DR screener, A at 63 parameters is as good. If you
want ICDR grading, C is the better model.

**Round 4 replicates.** KG topology is +1.08 sd (A) and +0.91 sd (C) against a
degree-matched shuffle — under 2 sd both — and `no-graph` matches or beats the real
KG in both. Over 5 paired seeds the effect goes **negative**.

### 8.4 Cross-ontology transfer

```bash
python transfer.py --kdim 2 --out ../../results/transfer.json
python transfer_decompose.py            # which parameter block carries it
```

The architecture splits into two blocks with different index sets:

```
W0, Wp, Wn (d×k), b (k)   indexed by FEATURE CHANNEL     26 of 65 params
u (F×k), theta            indexed by FINDING IDENTITY    39 of 65 params
```

The four channels mean the same thing in both trees, so the **trunk can cross**.
`u` cannot: breast finding 0 is `architectural_distortion`, retina finding 0 is
`microaneurysm`; F=16 in both is coincidence, not alignment.

**breast → retina is a clean positive:** 99.4% of the retina-trained ceiling,
**+4.35 sd** over a frozen random trunk. **retina → breast is weaker and honest
about it:** 93.4% of ceiling but only +1.73 sd, and it is *beaten* by the
0-parameter KG-signed sum, which trains nothing at all.

The random-trunk control is what makes this meaningful — a fixed random projection
plus a fitted per-node readout is already expressive (0.8399), so without it the
transfer number would say nothing about the source ontology.

### 8.5 Baselines on retina

```bash
python run_catfish_retina.py --model qwen3-vl:8b-instruct --split test --num-shards 16 --shard 0
python analyse_b1_retina.py --corpus ../../data/retina/catfish_b1 --out ../../results/b1_retina.json

python run_multiagent_retina.py --model qwen3-vl:8b-instruct --split test --num-shards 16 --shard 0
python b2_graph_retina.py --corpus ../../data/retina/multi6 --tau 0.858
python run_b2_retina.py --heads A,C --out ../../results/b2_retina.json
```

---

## 9. Framework D — DermaMNIST / dermoscopy (nominal)

The third instance, and the test of whether the port really is ontology-agnostic.
7 classes, **nominal not ordinal**, 55 findings.

```bash
cd dermaMnist/src/derma_kg

python prepare_derma.py --medmnist-root /path/to/cache
python derma_pack.py --print-findings
python kg_adjacency.py --mediators lesion,taxonomy,criterion,confusion
python nominal.py                                  # gradient gate

# probes (fleet) — same parameterised runner
cd ../../../experiments/15_kggnn
python run_probes.py --model qwen3-vl:8b-instruct \
    --npz      ../../dermaMnist/data/derma/images_224/test.npz \
    --findings ../../dermaMnist/data/derma/probe_findings.json \
    --lexicon  ../../dermaMnist/data/derma/lexicon.json \
    --split test --phrasing 3 --tag derma_p3 \
    --num-shards 32 --shard 0 --out-dir ../../dermaMnist/results

# debate (fleet)
cd ../../dermaMnist/src/derma_kg
python run_debate_derma.py --model qwen3-vl:8b-instruct --split test --num-shards 32 --shard 0

# the experiment
python run_derma.py --tag derma_p3                                    # probe-only arm
python run_derma.py --tag derma_p3 --debate-root ../../data/derma/debates_d1
python learning_curve.py --tag derma_p3 --out ../../results/derma_curve.json
```

**The probe-only pass is not a placeholder** — it *is* the `no-debate` control and
a complete result on its own. `scripts/analysis_chain.sh` runs the probe pass, the
learning curve, then re-runs both once the debate corpus lands, so nothing waits
on a human.

The adjacency uses four mediators — `lesion`, `taxonomy`, `criterion`,
`confusion` — giving 162 edges over 1485 pairs, with 11 pattern findings isolated.

---

## 10. Framework E — cross-dataset experiments

Studies that span all three trees at once.

```bash
cd shared

# 3×3 cross-ontology transfer matrix: every trunk into every target
python transfer_matrix.py --kdim 2 --l2 0.1 --epochs 400 --draws 3 \
    --out results/transfer_matrix.json

# Is the ontology PRIOR doing any work, or only the vocabulary?
python prior_shuffle.py --datasets breast,retina,derma --draws 5 \
    --out results/prior_shuffle.json

# An LLM judge over any tree, and how it compares to the aggregator
python run_judge.py --tree retina --split test --num-shards 16 --shard 0
python judge_score.py
```

`transfer_matrix.py` reproduces the headline transfer story on a wider grid — for
example, breast's ceiling is 0.7538 while a **derma-trained** trunk reaches 0.7341
on it, against a random-trunk floor of 0.6467 ± 0.0132.

`prior_shuffle.py` is the control that keeps C3 honest at the level below topology:
shuffling the per-finding sign prior does **not** hurt, and on retina the shuffled
prior is actually *better* (z = −3.16). The ontology's contribution really is
concentrated in which findings it names.

Fleet helpers: `judge_dispatch.sh`, `judge_fanout.sh`, `night_chain.sh`.

---

## 11. Framework F — HINPool graph-pooling benchmark

A **standalone** experiment, deliberately isolated from the rest of the repo. It
reproduces HINPool (type-aware heterogeneous graph pooling, AAAI 2026) on public
TUDataset benchmarks, where published numbers exist to compare against. Once it
works there, the model code transfers to the debate graphs.

```bash
cd hinpool-bench
pip install -r requirements.txt

python src/run.py --dataset MUTAG    --model hinpool --seeds 10 --out results/
python src/run.py --dataset MUTAG    --model rgcn    --seeds 10 --out results/
python src/run.py --dataset PROTEINS --model hinpool --seeds 10 --out results/
python src/run.py --dataset PROTEINS --model rgcn    --seeds 10 --out results/
```

Splits are frozen in `splits/*.json`. Results land as per-seed CSV with mean ± std
— the rigour protocol is non-negotiable here because the whole point is comparing
to published numbers.

---

## 12. The numbered experiment index (01–26)

Each directory is self-contained: scripts, `results/`, sometimes `logs/`. Run from
inside the directory. Chronological, so the index doubles as the project's
narrative.

### Phase 1 — where is the signal lost? (Aug 11–16)

**01 — Zero-shot VLM classification.** Strip away debate, KG and rounds; just ask
the model. Whatever it scores is the ceiling any debate on that model can reach.

```bash
cd experiments/01_zeroshot_vlm_classification
python run_zeroshot.py --model qwen3-vl:8b-instruct --split test --image-size 224
python evaluate.py --file results/<run>.json
```

**02 — Supervised ResNet baseline.** The comparator for C1.

```bash
cd ../02_resnet_baseline
python train_resnet.py --size 224 --seeds 5 --epochs 100 --class-weighted
```

**03 — KG-grounded single agent.** Observe → retrieve → classify. **The KG made it
worse**: AUC 0.4734 against 0.6013 without it, collapsing to a constant classifier
(0 malignant predictions). The failure is retrieval *quality*, and it is what sent
the project toward probes.

```bash
cd ../03_kg_grounded_vlm
python run_kg_grounded.py --model qwen3-vl:8b-instruct --split test --image-size 224
python run_kg_retrieval_v2.py --n-triples 30 --token-budget 6500
python run_kg_featureprobe.py --probe-set full --num-shards 8 --shard 0
python calibrate_probes.py --results results/ --min-std 0.01
python fit_probe_head.py --dataset ../../breastMnist/data/breast/dataset_v2
```

Ten `run_kg_*.py` variants sweep the injection strategy: `binary`, `natural`,
`freetext`, `assertive`, `enriched`, `birads`, `stance_direct`, `featureprobe`,
`retrieval_v2`, `grounded`.

**04 — Graph analysis.** Do the debate graphs carry label signal at all?
(TF-IDF over claim text: 0.5405. Structural features: 0.4933, i.e. chance.)

```bash
cd ../04_graph_analysis && python analyse_graphs.py --dataset ../../breastMnist/data/breast/dataset_v2 --splits train,val,test
```

**05, 07, 09, 10 — Debate protocols v2 → v5.** Corpus generators. See §6.3.
`09_debate_v4/audit.py` and `build_graphs_v4.py` add the positional-bias fix, the
forced-claim-count fix and the opening-round cap.

**06 — GNN over the debate graphs.** The first architecture attempt.

```bash
cd ../06_gnn
python train_gnn.py --dataset ../../breastMnist/data/breast/dataset_v5q --epochs 200 --seeds 5
python train_gnn_text.py --encoder tfidf --compare-with ../../breastMnist/data/breast/dataset_v4
python compare_datasets.py --datasets dataset_v4,dataset_v5q --out results/
```

**08 — Contrastive scoring.** Score-10 and scored variants.

```bash
cd ../08_contrastive
python run_contrastive.py --model qwen3-vl:8b-instruct --split test
python run_score10.py --n-findings 10 --split test
python evaluate.py --results results/
```

**11 — Graph readout.** Aggregation strategies over the claim graph.

```bash
cd ../11_readout && python graph_readout.py --dataset ../../breastMnist/data/breast/dataset_v5q --compare
```

**12 — ThothGNN v1.** First named model; corpus builder and interpreter.

```bash
cd ../12_thothgnn
python build_corpus.py --debates ../../breastMnist/data/breast/debates_v5q --out corpus/ --k 3
python merge_corpus.py --corpora corpus/ --out corpus/merged.json
python thoth_gnn.py --corpus corpus/merged.json --seeds 5 --folds 5 --epochs 800
python interpret.py --corpus corpus/merged.json --seed 0
python verify.py
```

**13 — BI-RADS 2.** The BI-RADS-scale readout and its evaluation.

```bash
cd ../13_birads2
python run_birads2.py --model qwen3-vl:8b-instruct --split test --n-findings 16 --tag birads2
python birads_scale.py
python evaluate_birads2.py --results results/ --tag birads2
```

**14 — KG-indexed evidence tensor.** Where the score actually improved, and where
the design plan was falsified by measurement. Two of its proposals were wrong:
down-weighting attacked claims *hurts* (0.6699 vs 0.7128), because **being
disagreed with carries nothing (spread 0.018) while being agreed with carries a
lot (0.189)** — the informative direction was inverted.

```bash
cd ../14_kgtensor
python kg_tensor.py --debates ../../breastMnist/data/breast/debates_v5q --out tensor.json --folds 5
python survival.py --debates ... --out survival_4run.json
python gate.py --debates ... --out gate_4run.json
python gate_crossrun.py --debates ... --out gate_crossrun.json
python crossrun_readout.py --debates ... --out crossrun_readout.json
python rank_loss.py --debates ... --out rank_loss.json
python combine.py --out combine.json
python bootstrap.py
```

**15 — KG finding graph + visual probes. The 0.80 result.** The project's biggest
single jump: **test AUC 0.8137, bAcc 0.7826** against a previous best of
0.7646/0.7036. Paired bootstrap: +0.0491, P(better) = 0.945. The method is a
logistic model on 32 features — 16 findings × 2 phrasings. **It does not use a
GNN.** See §7 for the full stack; the alternative architectures live here too.

### Phase 3 — external validation (Aug 22–24)

**16 — External / BUS-BRA.** The experiment behind C1. See §7 Stage 4.

**17 — Heterogeneous debate graph + ThothGNN v3.** Stops trying architectures and
measures the assumption underneath all of them: propagation over an attack graph
can only help if **being attacked is evidence of being wrong**. Measured over
15,220 claims: claim-level stance is correct 50.5% of the time, and attacks
received on correct vs wrong claims are 0.533 vs 0.536 — a difference of −0.003.
The assumption is simply false for this protocol.

```bash
cd ../17_hetgnn
python hetgraph.py                  # build the claim-resolution graph
python run_thothgnn3.py --phrasings p3 --out results/thothgnn3_p3.json
python diag_credibility.py          # h-categoriser credibility vs correctness
python diag_attacks.py
python powered_gnn.py               # is the null underpowered? (no)
python learning_curve.py
python contestation.py
python sparsity.py
python stance_signs.py
python uncertainty_gate.py
python debate_cost.py
python debate_when_split.py
python external_merged.py --phrasings p3 --arm full --out results/em_full_both.json
```

**18 — Lesion channel.** Asks "which lesion is this?" instead of "does it show
finding X?".

```bash
cd ../18_lesion
python run_lesion_probes.py --model qwen3-vl:8b-instruct --split test --phrasing 3 --tag lesion
python lesions.py && python analyse_lesion.py
python external_lesion.py --phrasings p3 --out results/ext_lesion.json
```

**19 — Learning curves.**  `cd ../19_curves && python train_curves.py`

### Phase 4 — the two reversals (Aug 26–29)

**20 — Perception.** The largest effect in the project (C4), and the experiment
that found P2 to be anti-predictive.

```bash
cd ../20_perception
python phrasing_arms.py             # P2 vs P3 vs pooled — run this before trusting any probe number
python per_phrasing.py
python prepare_masked.py --mode ring    # and: dim
python crop_sweep.py
python sign_audit.py && python sign_control.py
```

**21 — BiomedCLIP.** An alternative scorer, and its deflation.

```bash
cd ../21_biomedclip
python score_clip.py --npz ../../data/external/busbra_pad2_224.npz --out results/clip.json --batch 64
python compare_scorers.py && python hybrid.py
```

**22 — The transfer-gap law.**  `cd ../22_transfer && python gap_law.py`
(in-domain score vs external gap, r = 0.994).

**23 — Mask test.** Pre-registered before any run. The +0.087 that established C4.

```bash
cd ../23_masktest
python prepare_busbra_masked.py --mode ring
python analyse.py && python confirm2.py
```

**24 — Record audit.** The self-audit that caught three reporting faults, all of
which had flattered a result.

```bash
cd ../24_record
python audit_all.py && python audit_r03.py && python audit_r08_r13.py
python corpus_audit.py && python gap_correction.py
```

### Phase 5 — heterogeneous debate + two baselines (Aug 30 – Sep 3)

**25 — Two-model debate.** Pre-registered. Does a *heterogeneous* partner help?

```bash
cd ../25_twomodel
python screen.py                    # select the partner, bar AUC ≥ 0.60 → minicpm-v:8b
python merged_bm.py --debates ../../breastMnist/data/breast/debates_v5_het --probe-tag probep3 --folds 5 --out results/primary.json
python transcript_stats.py --corpora ../../breastMnist/data/breast/debates_v5_het --out results/stats.json
```

**26 — Re-implemented baselines.** Pre-registered, with a dated ERRATA recording
the B2 capacity confound. **Both are our implementations from the papers; no
released code was used.** Label every result "our implementation of X".

- **B1 — Catfish Agent** (arXiv 2505.21503). Four roles; the catfish sees the
  *transcript only* — information asymmetry. The trigger is recorded at generation
  and gated at analysis, so `tau_conf` is cross-validatable and `B1-always-on`
  falls out free. Frozen config: `tone=collaborative`, `tau_conf=0.858`.
- **B2 — GraphGeo** (arXiv 2511.00908). Six agent nodes; relations `r_agree`,
  `r_conflict`, `r_transfer`. 2 layers, hidden 32, dropout 0.5, DropEdge 0.2.

```bash
cd ../26_baselines
python run_catfish.py --model qwen3-vl:8b-instruct --split test --num-shards 16 --shard 0 \
    --probe-tag probep3 --out-dir ../../breastMnist/data/breast/catfish_b1
python analyse_b1.py --corpus ../../breastMnist/data/breast/catfish_b1 --out results/b1.json

python run_multiagent.py --qwen-model qwen3-vl:8b-instruct --gemma-model gemma3:12b \
    --qwen-url http://gpu07:11434/api/chat --gemma-url http://gpu11:11434/api/chat \
    --split test --num-shards 16 --shard 0 --out-dir ../../breastMnist/data/breast/multi6
python b2_graph.py --corpus ../../breastMnist/data/breast/multi6 --tau 0.858
python b2_train.py --corpus ../../breastMnist/data/breast/multi6 --out results/b2.json --epochs 300

python run_controls.py --draws 5 --split test    # self-consistency + independent ensemble
python analyse_controls.py
python external.py --arm auto --out results/external_auto.json      # and: --arm mask
python compare_paired.py                          # paired bootstraps vs ThothGNN v3
python assemble_table.py                          # → results/table.json, the consolidated table
```

**Neither baseline beats ThothGNN v3 in either condition**, despite B2 carrying
**182× the parameters** (10,737 vs 59) and 3× the agents. That confound runs in
B2's favour and is stated wherever B2 appears.

---

## 13. Running on a GPU fleet

Every corpus runner is **shardable and resumable**: `--num-shards N --shard i`, one
record per sample, and a worker skips indices already written. Shards cannot
collide; extra workers can join at any time.

```bash
# detached launch that survives the ssh session
scripts/launch.sh /path/to/log.txt python run_probes.py --split test --shard 0 --num-shards 16

# shard a corpus across nodes
scripts/launch_debate_v2.sh <out-dir> <num-shards> gpu07:0 gpu11:1 gpu13:2 ...

# fill whatever fixed sharding missed (nodes finish at very different speeds)
scripts/completion_daemon.sh

# watch running jobs and flag quality regressions live
scripts/monitor_all.sh 300

# push ollama + model blobs to a node whose /data is empty
scripts/provision_node.sh gpu17 gpu07
```

For the retina and derma trees use the **claim-based** dispatchers, which let a
node that finishes early take the next unclaimed shard instead of idling:

```bash
retinaMnist/scripts/run_stage.sh      # single-issuer, PID-lockfile guarded
retinaMnist/scripts/dispatch_retina.sh
dermaMnist/scripts/run_stage.sh
shared/judge_dispatch.sh
```

### Five operational rules, each paid for in lost GPU hours

1. **`/api/tags` is not a health check.** It answers happily while the GPU is
   dead. Use real inference *and* check `offloaded N/N layers` in the ollama log.
   One such trap produced 262 empty debates.

2. **Never mix job kinds with different `num_ctx` on one node.** `run_probes.py`
   asks for 2048 and the debate runners for 4096. A different `num_ctx` is a
   different ollama runner, so a node given both tries to hold two 5.8 GB copies of
   the model; 8 GB cards hold one, and every alternation evicts and reloads the
   other at 20–45 s. Measured: **99–228 s/image mixed against 10.8 s/image
   segregated — a 10.3× throughput difference** that cost ~14 hours before it was
   diagnosed. The dispatchers pin probe and debate jobs to disjoint node sets.

3. **Pilot the configuration you will actually dispatch.** The trap above was
   invisible in a pilot that ran probes on one node and debate on another — which
   is the working layout, not the dispatched one.

4. **One modulus per corpus, and count *unique* indices.** Mixing `--num-shards 2`
   with `--num-shards 3` silently duplicated 341 probe computations, and a raw line
   count reported 1874/1875 when unique coverage was 1536.

5. **Home directories cap on inodes (60k), not space.** Per-sample JSON corpora hit
   that ceiling twice, once hard enough that `tar` itself failed. **Store corpora
   as JSONL**, one record per line. The historical per-sample corpora are archived
   as tarballs in `breastMnist/data/breast/_archive/`.

**Node speed tiers** matter for sharding: TITAN Xp and 2080 Ti are fast; GTX 1080
and Quadro are mid; the **GTX TITAN X (Maxwell) is 5–6× slower** than the top tier.
Fixed round-robin leaves the slow nodes holding the tail — use the claim-based
dispatchers.

---

## 14. Personalising the framework

### 14.1 Add a new dataset and ontology

The retina and derma ports were built so this is **a data drop plus one YAML, not
a code fork**. Concretely, to add dataset *X*:

1. **Create the tree**, mirroring `retinaMnist/`:
   ```
   xMnist/
     conf/ontology_x.yaml       every relation name your ontology uses
     src/x_kg/paths.py          copy from retina; edit TREE and the pack paths
     data/x/source/             your raw ontology JSON
     results/  logs/  scripts/
   ```
   `paths.py` is the reason this is cheap — all thirty `Path(__file__).parents[N]`
   expressions live in one file, so relocating the tree is a one-line change.

2. **Write `x_pack.py`**, modelled on `icdr_pack.py`. It must emit
   `knowledge_graph.json` (triples with `t_NNN` ids), `definitions.json`
   (`d_NNN` ids — the loader raises `KeyError` without them), `criteria.json`,
   `findings.json`, `probe_findings.json` and `lexicon.json`.

3. **Derive the finding set from the graph**, don't hand-write it. You need three
   things, and the ICDR pack shows how to get each from relation names alone:
   polarity/prior, a not-visible filter, and probe phrasing. Hand tables are a
   maintenance trap — automating the breast tables cost 0.6685 → 0.5750 precisely
   *because* they had been hand-tuned.

4. **Filter label leakage.** Check whether your definitions lead with the class the
   finding establishes. In ICDR, 8 of 17 did. Strip sentences carrying grading
   vocabulary and assert that zero descriptions mention a class.

5. **Choose adjacency mediators** in `kg_adjacency.py`. Keep the Adamic-Adar
   formula; swap only what counts as a mediator. The breast lesion-mediated
   construction does **not** transfer blindly — ported directly to ICDR it gave
   **1 edge with 14 of 16 findings isolated**. Retina uses criterion, taxonomy,
   location and association (21 of 120 pairs, nothing isolated); derma uses lesion,
   taxonomy, criterion and confusion.

   > **Never use the class label as a mediator.** Joining two findings because they
   > establish the same grade is label homophily, not clinical structure, and the
   > prior already encodes it. `kg_graph.py` records the identical failure on breast.

6. **Reuse the runners.** `run_probes.py` is parameterised, not forked — point
   `--npz`, `--findings` and `--lexicon` at your pack. The breast defaults still
   render all four phrasing templates byte-for-byte, which is asserted by
   `test_reduction.py`'s sibling check.

7. **Pick a head** for your label space: `ordinal.py` (ordered) or `nominal.py`
   (unordered). Add a metrics module if neither `metrics_ordinal.py` nor
   `metrics_nominal.py` fits — always re-export `auc`/`bacc` from `claims.py` so
   binary numbers stay bit-comparable with every earlier experiment.

### 14.2 Swap the backbone model

Every runner takes `--model` and `--url`:

```bash
python run_probes.py --model llava:13b --url http://gpu09:11434/api/chat ...
```

The only requirement is that the backend returns **top-20 first-token logprobs** —
the probe primitive depends on it. A second backbone at scale is the single
biggest thing that would strengthen every claim in this project; only
`qwen3-vl:8b-instruct` has been run at full scale.

### 14.3 Change the probe phrasing

`--phrasing {1,2,3,4}` on `run_probes.py`. **Use 3.** Arm 2 (negated) is
anti-predictive — 0.4452 external, below chance. Adding a fifth arm means adding a
template to the `_PHRASING` table (breast) or extending the appearance-triple
derivation (retina/derma), then re-running `20_perception/phrasing_arms.py` to
validate it *before* it enters any feature block.

### 14.4 Tune the model

```bash
python run_thothgnn3.py --phrasings p3               # breast
python run_ordinal.py --heads A,C --seed 0           # retina
python run_derma.py --tag derma_p3                   # derma
```

Reference hyperparameters:

| Model | Params | Settings |
|---|---:|---|
| ThothGNN v3 | 59 | 800 epochs, momentum 0.9, lr 0.15, l2 0.03, kdim 2 |
| Ordinal head A | 63 | as above, 4 cut points |
| Ordinal head C | 236 | kdim 4, 4 independent trunks |
| B2 GraphGeo | 10,737 | 2 layers, hidden 32, dropout 0.5, DropEdge 0.2, wd 5e-4, lr 3e-3, 300 epochs |

Selection is by 5-fold CV **on train only**; test is evaluated **once per arm**.
Keep it that way — the whole record depends on it.

### 14.5 Change the debate protocol

`--rounds` (default 3), `--temperature` (default 0.8), `--no-kg` to drop ontology
injection, `--model-b` for a heterogeneous partner. Structural changes go in
`run_debate_v5.py`'s prompt builders.

**Watch the transcripts in the first fifty samples.** Three separate defects
reached 780 graphs before anyone read them: v3's single sentence covering 12% of
claims, v4's agents citing list position rather than image content, and v5's agents
producing byte-identical claims. All were visible early.
`scripts/monitor_all.sh` logs the diagnostics that would have caught each.

### 14.6 Add a control

Controls are the point of this repository. The four that matter:

```python
shuffle_kg(Ap, An, rng)   # 17_hetgnn/run_thothgnn3.py — rewires while preserving
                          # the weight multiset and symmetry (degree-matched)
A = 0                     # no-graph: capacity held, message passing removed
drop debate channel       # probes + prior only
shape-matched noise       # replaces the debate block with same-shape noise
```

Run at least 5 draws of any stochastic control and report **mean ± sd**. A single
draw cannot distinguish a null from a small effect — that is exactly how the
retina KG effect looked like +1.08 sd on one seed and turned **negative** over
five.

---

## 15. Conventions, metrics and gotchas

### Metrics

**Balanced accuracy is the headline**, always, because every dataset here is
imbalanced. On BreastMNIST test (27% malignant) always answering "benign" scores
0.7308 accuracy but 0.500 bAcc. AUC, MCC, sensitivity and specificity are reported
alongside; raw accuracy appears only for reference.

| Task | Metrics |
|---|---|
| binary | bAcc, AUC, MCC, sens, spec |
| ordinal (retina) | QWK, MAE, macro-recall, adjacent accuracy, per-threshold AUC, referable AUC |
| nominal (derma) | macro-recall, accuracy, macro-OVR AUC |

AUC handles ties correctly — `(a>b) + 0.5·(a==b)`. An earlier uncorrected version
understated one method by 0.18.

### Statistical protocol

- **5 fixed seeds** (0,1,2,3,4); report mean ± sd. Typical sd ≈ 0.005.
- **Paired bootstrap**, 4000 resamples, on **patients** not images.
- **5-fold CV**, folds split on **case**.
- Selection on train only. **One test evaluation per arm**, config frozen and
  logged first.
- The bar for calling an effect real in this project is **2 sd** over its control.

### Gotchas that have already cost real time

- **Pooled probe arms contaminate every number before 2026-08-26.** Quote P3-only
  for transfer; say "pooled" when quoting an older number.
- **Never compare `external_merged.py` with `freeze_models.py`.** Within-BUS-BRA CV
  vs true frozen transfer. They rank the arms oppositely; that is the gap law.
- **Never put a 5-fold-CV, in-domain, concept-supervised number in the same column
  as a zero-shot one.** Concretely: MedCBR's 94.2 on BUS-BRA is not comparable.
- **The MedMNIST comparison is not like-for-like.** 63–236 trained parameters on a
  frozen 8B VLM against CNNs trained from scratch. The gap measures what the
  backbone already knows, not an architectural win. Never report it as one.
- **The ThothGNN mask row is BUS-BRA only.** BreastMNIST ships no segmentation
  masks at all, so no in-domain mask claim exists or can exist.
- **Multiple comparisons are not formally corrected** across rounds. Pre-registration
  limits the endpoint count; that is not the same thing.

### Cost

**$0.** No paid APIs, ever. ~137k VLM calls for the breast work, ~93k for retina,
plus the derma corpora — all on a local GPU cluster.

---

## 16. Where every result lives

Nothing in this repository is quoted from memory; every number traces to a file.

**Start here: [`RESULTS.html`](RESULTS.html)** — the results compendium. Every
measured result in the project in one document: each experiment's question, what it
settled, and the result tables themselves, each read out of its source file at build
time and stamped with that file's SHA-256. It ends with a completeness audit that
walks the tree and names any result file the document fails to show. Rebuild it after
any new result with:

```bash
python scripts/build_results_compendium.py
```

The table below is the index to the underlying files.

| What | Path |
|---|---|
| **The results compendium (all of the below, rendered)** | `RESULTS.html` |
| **Consolidated master table** | `experiments/26_baselines/results/table.json` |
| Paired bootstraps vs ThothGNN v3 | `experiments/26_baselines/compare_paired.py` |
| External — baselines | `experiments/26_baselines/results/external_{auto,mask}.json` |
| External — our methods | `experiments/17_hetgnn/results/em_{full,mask}_both.json` |
| ThothGNN v3 + all controls | `experiments/17_hetgnn/results/` |
| The 0.80 probe result | `experiments/15_kggnn/results/` |
| Mask pre-registered result (C4) | `experiments/23_masktest/results/primary.json` |
| Transfer-gap law (r=0.994) | `experiments/22_transfer/results/gap_law.json` |
| Two-model debate | `experiments/25_twomodel/results/primary.json` |
| Perception / phrasing arms | `experiments/20_perception/results/` |
| Record audit + corrections | `experiments/24_record/results/` |
| Rendered results ledger (HTML) | `experiments/artifacts/breastmnist_kg_debate_ledger.html` |
| **Retina — ordinal + controls** | `retinaMnist/results/ordinal_full.json` |
| Retina — 5 paired seeds | `retinaMnist/results/ordinal_seeds.json` |
| Retina — paired bootstraps | `retinaMnist/results/paired_{ac,multiclass}.json` |
| Retina — MedMNIST comparison | `retinaMnist/results/medmnist_compare.json` |
| Retina — cross-ontology transfer | `retinaMnist/results/transfer{,_decompose}.json` |
| Retina — baselines B1 / B2 | `retinaMnist/results/b{1,2}_retina.json` |
| Retina — data provenance (SHA-256) | `retinaMnist/results/retina_prep.json` |
| **Derma — nominal + controls** | `dermaMnist/results/derma_nominal.json` |
| Derma — probe-only (no-debate) arm | `dermaMnist/results/derma_probeonly.json` |
| Derma — learning curves | `dermaMnist/results/derma_curve*.json` |
| Derma — data provenance | `dermaMnist/results/derma_prep.json` |
| **3×3 transfer matrix** | `shared/results/transfer_matrix.json` |
| Prior-shuffle control | `shared/results/prior_shuffle.json` |
| LLM judge scoring | `shared/results/judge_score.json` |
| HINPool benchmark | `hinpool-bench/results/*.csv` |
| ACR BI-RADS ontology | `breastMnist/data/breast/{knowledge_graph,definitions}.json` |
| ICDR ontology + relation names | `retinaMnist/data/retina/`, `retinaMnist/conf/ontology_retina.yaml` |
| Dermoscopy ontology | `dermaMnist/data/derma/`, `dermaMnist/conf/ontology_derma.yaml` |

### Known open items

- No retina or derma equivalent of the BUS-BRA external arm exists, so **C1 remains
  a single-dataset claim**.
- The §8.4 transfer arms are **single-seed**; only the random-trunk control is
  multi-draw. Roughly 30 minutes of CPU would fix this.
- **No pre-registration was committed before the retina run.** The procedure was
  clean — selection on train only, one test evaluation per arm — but the document
  does not exist. Write it as explicitly post-hoc; **never backdate it.**
- Token-level budget logging is incomplete; wall-clock and call counts are done.
- Only one backbone tested at scale.
