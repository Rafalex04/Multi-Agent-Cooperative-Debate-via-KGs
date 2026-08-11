# Project: Multi-Agent Agreement and Decision-Making via KG Merging

## Read first

`SPEC.md` is the source of truth for what we're building. **Read it before any non-trivial work.** If a request contradicts the spec, ask before proceeding — do not silently reinterpret.

## Project structure

```
.
├── SPEC.md
├── CLAUDE.md
├── pyproject.toml
├── .env.example                    # template for required secrets (committed)
├── .env                            # actual secrets (NEVER committed)
├── conf/
│   ├── config.yaml                 # FEVER pipeline root config
│   ├── config_breast.yaml          # BreastMNIST pipeline root config
│   ├── judges/medgemma.yaml        # MedGemma judge config (backend: ollama|vllm)
│   └── run/                        # run group configs (breast_single, breast_debate_*, etc.)
├── data/breast/                    # ACR BI-RADS KG + fixed ablation sample set
│   ├── knowledge_graph.json        # 370 triples, IDs t_001–t_370
│   ├── definitions.json            # 100 entities, IDs d_001–d_100
│   └── ablation_indices.json       # 50 fixed test-set indices (seed=42, never regenerate)
├── src/
│   └── debate_kg/
│       ├── main.py                 # FEVER Hydra entrypoint
│       ├── breast_main.py          # BreastMNIST Hydra entrypoint (6 modes)
│       ├── kg/
│       │   ├── graph.py            # NetworkX KG class, stable triple UUIDs
│       │   ├── schema.py           # Pydantic models (DebateNode, JudgeScore, WinnerJudgment, …)
│       │   ├── loader.py           # load_kg_from_json, load_definitions, serialize_kg_for_prompt
│       │   ├── wikidata.py         # SPARQL helpers (FEVER only)
│       │   └── construct.py        # subgraph builder (FEVER only)
│       ├── retriever/              # GraphRAG (FEVER only)
│       ├── debate/
│       │   ├── breast_orchestrator.py  # all 6 breast modes
│       │   └── prompts/breast/     # Jinja2 templates (system_*.j2, expert_*.j2, judge_*.j2)
│       ├── judges/
│       │   └── medgemma_judge.py   # MedGemmaJudge: score_utterance + pick_winner
│       ├── consensus/              # stop check (shared)
│       ├── merger/                 # rule-based KG merger (FEVER only)
│       ├── data/
│       │   └── breastmnist.py      # loads BreastMNIST as base64 JPEG samples
│       ├── models/                 # OllamaClient, VLLMClient, StubClient
│       └── eval/breast_metrics.py  # accuracy, AUC-ROC, sensitivity, specificity
└── tests/
```

## Key data model

### Shared (both pipelines)
- **Each `DebateNode` IS one factual claim** — one `[CLAIM]` block from the LLM response. A single expert turn may produce N nodes.
- **Short IDs** (`c1`, `c2`, …) are globally sequential across all rounds. Assignment: `c{len(history) + i + 1}`.
- **Edges** come from parsing `[ADDRESSED:cN][AGREE|DISAGREE]` tags — no edge-classifier LLM call.
- **Opening expert alternates**: A opens even rounds, B opens odd rounds.

### BreastMNIST differences
- **Labels**: `BENIGN` / `MALIGNANT` (not SUPPORTS/REFUTES/NEI).
- **Opinion mode** (mode 2): free-text, one `DebateNode` per expert turn, no `[CLAIM]` structure. Judge picks winner.
- **Structured modes** (3–5): same `[CLAIM]` block structure. Every turn after round 0 must include at least one `[ADDRESSED:cN][AGREE|DISAGREE]` tag (prompt-enforced, not code-enforced). Verdict = weighted node-score vote.
- **Expert identity**: every system prompt receives `expert_letter` (A or B) and `other_letter` — experts must know who they are.
- **KG citations**: `[CITED:t_NNN]` for triples, `[CITED:d_NNN]` for definitions, parsed into `DebateNode.provenance`. By default only the 370 triples are serialized into prompts (`include_definitions=False`, ~6.5K tokens) — the 100 definitions are loaded but not shown to the model unless `include_definitions=True`.
- **Early stop (opinion mode)**: stops after round ≥ 1 if any expert writes `[FINISH]`, OR if both experts state the same verdict for 2 consecutive rounds (`consecutive_agreement >= 2`).
- **Early stop (structured modes)**: shared `is_consensus()` with FEVER — round 0 never stops early; consensus can only be declared from `round_idx >= 1`.
- **Fixed sample set**: `data/breast/ablation_indices.json` — 50 test-set indices. Never regenerate; all ablation stages must use the same samples.
- **Verdict (opinion)**: judge's `pick_winner` → winner's last stated label.
- **Verdict (structured)**: weighted vote — `score = Σ weight_i × vote_i`, `weight_i = (groundedness + factuality) / 200`, `vote_i = +1` (MALIGNANT) or `−1` (BENIGN).
- **GPU sharing**: the expert model (always Ollama — there is no vLLM path for experts) and an Ollama-backend judge share one Ollama instance. `breast_main` unloads each model (`keep_alive=0`) before the other runs, and caps the judge's `num_ctx` at 4096 vs the expert's 32768.

## Standing rules

- **KG class** lives in `kg/graph.py`. Triples have stable UUIDs. Don't re-implement elsewhere.
- **LLM access** goes through `models/`. Never call Ollama or vLLM HTTP directly from module code.
- **Prompts** are separate `.j2` files under `debate/prompts/`. No prompts as Python string literals.
- **Provenance is non-optional.** Every DebateNode must carry cited triple/definition IDs. Losing provenance is a bug.
- **Determinism:** all randomness through `numpy.random.default_rng(seed)`. Seed comes from Hydra config.
- **Paths** via `pathlib.Path`, never `os.path`.
- **Type hints** on every public function.
- **Secrets** live in `.env`, loaded via `python-dotenv`. Never commit `.env`. Maintain `.env.example`.
- **TODO format:** `# TODO(<module>): <thing>`. Bare `# TODO` is forbidden.

## Logging

- Use `logging`, not `print`. Configure once in `main.py` / `breast_main.py`. Library code: `logger = logging.getLogger(__name__)`.
- DEBUG: per-turn LLM responses. INFO: per-node labels/scores, round milestones, sample timing. WARNING: parse failures. ERROR: hard failures.

## Things to avoid

- Don't add the GNN merger to v1.
- Don't introduce closed-source or paid model dependencies. Strict $0 budget.
- Don't hardcode dataset paths.
- Don't write a "let me also build X" feature without asking. Stay in scope.
- Don't catch exceptions silently. If you catch, log and re-raise or handle visibly.
- Don't regenerate `ablation_indices.json` — it must stay fixed across all ablation stages.

## Code style

- Black (line length 100) + ruff. Run before committing.
- Tests with pytest. Every module gets at least a smoke test.
- Docstrings: short. One-line summary; expand only when behavior is non-obvious.

## Build, test, run

```bash
# install
uv sync

# test
pytest
pytest tests/kg

# FEVER pipeline
python -m debate_kg.main run=smoke
python -m debate_kg.main run=fever_100

# BreastMNIST pipeline — 6 ablation modes
python -m debate_kg.breast_main run=breast_single data.num_samples=2     # quick test
python -m debate_kg.breast_main run=breast_single                         # 50 samples
python -m debate_kg.breast_main run=breast_debate_opinion
python -m debate_kg.breast_main run=breast_debate_graph
python -m debate_kg.breast_main run=breast_debate_kg
python -m debate_kg.breast_main run=breast_adversarial
python -m debate_kg.breast_main run=breast_adaptive

# Judge backend (edit conf/judges/medgemma.yaml)
# backend: "ollama"  — shares the expert's Ollama instance (testing, or cluster if no vLLM server)
# backend: "vllm"   — separate vLLM server (real runs)
```

**Cluster reliability:**
- `OllamaClient` retries on connection errors, timeouts, and 5xx with backoff (~30 min total) and restarts Ollama via `$OLLAMA_RESTART_SCRIPT` (set in `.env`).
- Each run writes `outputs/resume/<mode>_<num_samples>_<seed>.jsonl` incrementally; rerunning the same command skips already-completed samples and deletes the file on success.

## Working preferences

- Use plan mode for anything touching multiple modules or the round loop.
- For single-module changes, normal mode is fine.
- When stuck, read the relevant paper section in `SPEC.md` rather than guessing.
- Update this file when you discover a convention that prevented a class of mistakes. Keep it under ~250 lines.
