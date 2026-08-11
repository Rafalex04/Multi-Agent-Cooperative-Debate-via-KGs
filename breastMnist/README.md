# Multi-Agent Agreement and Decision-Making via Knowledge Graph Merging

Two use cases share the core data model (debate orchestrator, `DebateNode`/`DebateEdge` graph, consensus check) and LLM client layer:

- **FEVER fact-checking** (below) — two agents grounded in disjoint knowledge graphs debate FEVER claims; iterative KG merging drives consensus.
- **[BreastMNIST ablation study](#breastmnist-ablation-pipeline)** — VLM agents debate breast ultrasound images using a shared ACR BI-RADS knowledge graph, across 6 ablation modes.

See `SPEC.md` for the full spec of both pipelines.

## FEVER Pipeline

Two LLM agents grounded in disjoint knowledge graphs cooperatively debate FEVER claims. Disagreement between agents signals incomplete knowledge; iterative KG merging drives the agents toward consensus. A panel of three diverse LLM judges scores each claim node, and a weighted vote produces the final verdict.

## What it does

1. **KG Construction** — fetches Wikidata subgraphs for entities mentioned in each FEVER claim and splits them into two disjoint expert KGs.
2. **GraphRAG Retrieval** — retrieves the most relevant triples from each expert's KG per round using dense embeddings.
3. **Structured Debate** — two experts write `[CLAIM]` blocks with inline citations and `[ADDRESSED]` stance tags. Each `[CLAIM]` block becomes one `DebateNode`; edges are parsed from `[ADDRESSED:<ID>][AGREE|DISAGREE]` tags (no LLM edge classifier).
4. **Judge Panel** — three judges from different model families (Llama, Qwen, Mistral) score each claim node on KG-groundedness and factuality (0–100 each).
5. **Consensus Check** — stops when no disagreement edges exist, or when `max_rounds` is reached.
6. **KG Merger** — rule-based merger updates both KGs using judge scores as weights; loop continues with enriched KGs.
7. **Verdict** — weighted vote over all claim nodes: SUPPORTS=+1, NEI=0, REFUTES=−1, each weighted by its mean judge score.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with these models pulled:
  ```
  ollama pull llama3.1:8b
  ollama pull qwen2.5:7b
  ollama pull mistral:7b
  ```
- `uv` (recommended) or `pip`

## Install

```bash
uv sync        # or: pip install -e .
cp .env.example .env   # fill in any required keys
```

## Run

```bash
# Smoke test — 1 claim, full pipeline
python -m debate_kg.main run=smoke

# Debate mode — first N FEVER claims
python -m debate_kg.main run=fever_100 run.num_claims=5 run.mode=debate

# Single-expert baseline (no debate)
python -m debate_kg.main run=fever_100 run.num_claims=5 run.mode=single_expert

# With verbose orchestrator logging
python -m debate_kg.main run=fever_100 run.num_claims=5 "hydra.verbose=[debate_kg.debate.orchestrator]"
```

Outputs write to `outputs/<timestamp>/` (Hydra default). Each run produces:
- `claims_index.json` — claim IDs, slugs, and verdicts for the whole run
- `debate_<id>_<slug>.json` — debate graph (nodes + edges) per claim
- `judge_scores_<id>_<slug>.json` — per-node judge scores
- `kgs/<id>_<slug>/round_<n>/{a,b}.json` — KG snapshots per round
- `metrics.json` — aggregated accuracy, F1, convergence rate, KG drift

## Tests

```bash
pytest                   # all tests
pytest tests/debate      # debate module only
pytest -m "not network"  # skip live Wikidata calls
```

## Architecture

```
input claim
  ↓
[1 KG Construction] → [2 GraphRAG Retriever] → [3 Debate Orchestrator]
                                                       ↓
                                                  [4 Judge Panel]
                                                       ↓
                                                  [5 Consensus Check] ── stop → verdict
                                                       │ continue
                                                       ↓
                                                  [6 KG Merger] ──────────→ back to step 2
```

All models run locally via Ollama. No paid APIs. Strict $0 budget.

---

## BreastMNIST Ablation Pipeline

Six ablation modes isolating the contribution of debate, graph-based verdicts, a domain knowledge graph, and adversarial stances on binary classification (BENIGN / MALIGNANT) of breast ultrasound images from BreastMNIST (MedMNIST v2). VLM experts debate; a single medically fine-tuned judge (MedGemma-4B-IT) scores each claim or picks the winning expert.

| Mode | Run config | Debate | KG | Verdict |
|---|---|---|---|---|
| `single_agent` | `breast_single` | No | No | Parsed `[LABEL][CONFIDENCE]` |
| `debate_opinion` | `breast_debate_opinion` | Free-text, multi-round | No | Judge picks winning expert |
| `debate_graph` | `breast_debate_graph` | Structured `[CLAIM]` | No | Weighted node-score vote |
| `debate_kg` | `breast_debate_kg` | Structured `[CLAIM]` | Yes (ACR BI-RADS) | Weighted node-score vote |
| `debate_kg_adversarial` | `breast_adversarial` | Structured `[CLAIM]`, fixed opposing stances | Yes | Weighted node-score vote |
| `adaptive_adversarial` | `breast_adaptive` | Conditional escalation | Yes | Confidence-gated |

### Requirements

- Everything above, plus:
  - `ollama pull qwen2.5vl:7b` — expert model, **always served via Ollama** (CPU in dev, GPU on cluster; there is no vLLM path for experts).
  - Judge backend set in `conf/judges/medgemma.yaml`: `ollama pull medgemma:latest` (shares the expert's Ollama instance) or a vLLM server serving `google/medgemma-4b-it`.
  - `medmnist` (installed via `uv sync`) downloads the BreastMNIST test split automatically on first run.

### Run

```bash
python -m debate_kg.breast_main run=breast_single data.num_samples=2   # quick test
python -m debate_kg.breast_main run=breast_single                       # 50 samples
python -m debate_kg.breast_main run=breast_debate_opinion
python -m debate_kg.breast_main run=breast_debate_graph
python -m debate_kg.breast_main run=breast_debate_kg
python -m debate_kg.breast_main run=breast_adversarial
python -m debate_kg.breast_main run=breast_adaptive
```

All six modes draw from the same fixed 50 test-set indices in `data/breast/ablation_indices.json` (never regenerate — required for a valid cross-stage comparison).

### Output

Each run writes to `outputs/<timestamp>/`:
- `metrics.json` — accuracy, sensitivity, specificity, AUC-ROC, convergence rate, mean rounds/time, per-sample records.
- `debate_sample-<id>.json` — debate nodes, edges, judge scores, verdict per sample.
- `outputs/resume/<mode>_<num_samples>_<seed>.jsonl` — incremental per-sample progress; rerunning the same command resumes after a crash instead of restarting.

### Architecture

```
breast ultrasound image (base64 JPEG)
  ↓
[single_agent]  or  [opinion debate]  or  [structured [CLAIM] debate (+ ACR BI-RADS KG)]
                                                       ↓
                                            [MedGemma-4B-IT judge]
                                                       ↓
                                  [verdict: weighted node vote | judge pick_winner]
```

No retrieval (the ~370-triple KG is injected wholesale) and no KG merger (the KG is static). Provenance (`[CITED:t_NNN]`/`[CITED:d_NNN]`) is still tracked on every `DebateNode`.
