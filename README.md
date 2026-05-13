# Multi-Agent Fact-Checking via Knowledge Graph Merging

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
