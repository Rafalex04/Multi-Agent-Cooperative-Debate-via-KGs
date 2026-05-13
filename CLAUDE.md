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
├── conf/                           # Hydra configs (model.yaml, data.yaml, run/*.yaml, judges/*.yaml)
├── src/
│   └── debate_kg/
│       ├── __init__.py
│       ├── main.py                 # Hydra entrypoint, round loop, logging setup
│       ├── kg/                     # Module 1: KG construction
│       │   ├── graph.py            # NetworkX-backed KG class with stable triple UUIDs
│       │   ├── wikidata.py         # SPARQL helpers, cached
│       │   └── construct.py        # subgraph builder + split logic
│       ├── retriever/              # Module 2: GraphRAG
│       ├── debate/                 # Module 3: orchestrator
│       │   └── prompts/            # prompt templates as .j2 / .txt files
│       ├── judges/                 # Module 4: panel
│       ├── consensus/              # Module 5: stop check
│       ├── merger/                 # Module 6: rule-based merger
│       ├── data/                   # FEVER loading
│       ├── models/                 # Ollama / vLLM client wrappers
│       └── eval/                   # metrics + baselines
├── tests/                          # mirrors src/ layout
└── scripts/                        # one-off / cluster-launch scripts
```

## Key data model

- **Each `DebateNode` IS one factual claim** — one `[CLAIM]` block from the LLM response. A single expert turn (one LLM call) may produce N `DebateNode`s.
- **Short IDs** (`c1`, `c2`, …) are globally sequential across all rounds of a debate. Assignment: `c{len(history) + i + 1}` where `i` is the 0-based position within the current turn.
- **Edges come from parsing `[ADDRESSED:<ID>][AGREE|DISAGREE]` tags** — there is no edge-classifier LLM call.
- **Labels** (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`) are the expert's verdict on the FEVER claim, not on their own statement.
- **Verdict** is a deterministic weighted vote over all nodes: SUPPORTS=+1, NEI=0, REFUTES=−1, weighted by mean judge score. No LLM call.
- **Opening expert alternates**: A opens even rounds, B opens odd rounds.

## Standing rules

- **KG class** lives in `src/debate_kg/kg/graph.py` and uses NetworkX. Triples have stable UUIDs. Don't re-implement this elsewhere.
- **LLM access** goes through `src/debate_kg/models/`. Never call Ollama or vLLM HTTP directly from module code.
- **Prompts** are separate files under `src/debate_kg/debate/prompts/`. No prompts as Python string literals in the orchestrator.
- **Provenance is non-optional.** Every debate node must carry a list of KG triple IDs it cited. If a function loses provenance, that's a bug.
- **Determinism:** all randomness through `numpy.random.default_rng(seed)` or `torch.Generator(seed)`. Seed comes from Hydra config.
- **Paths** via `pathlib.Path`, never `os.path`.
- **Type hints** on every public function. Public = anything not prefixed with `_`.
- **Secrets** live in `.env` at project root, loaded via `python-dotenv` in `main.py`. Never commit `.env`. Maintain a `.env.example` (committed) listing required keys without values. Never paste tokens into prompts, configs, or code.
- **TODO format:** `# TODO(<module>): <thing>` or `# TODO(<module>, ref=SPEC §X): <thing>`. Bare `# TODO` is forbidden — they become invisible after week one.

## Logging

- Use Python's stdlib `logging`, not `print`. Configure once in `main.py`. Library code does `logger = logging.getLogger(__name__)` and never reconfigures.
- Levels: DEBUG for per-turn detail, INFO for round/module milestones, WARNING for recoverable issues, ERROR for failures. No INFO inside tight loops.
- Each Hydra run writes to `outputs/<timestamp>/` automatically. Don't fight this — debate transcripts, KG snapshots, and metrics go in there (see `SPEC.md` §Output artifacts).

## Things to avoid

- Don't add the GNN merger to v1. That's v2.
- Don't try to enforce full logical consistency in the merger. The spec acknowledges this is unsolved; minimal type/symmetry/ontology-light checks only.
- Don't introduce closed-source or paid model dependencies. Strict $0 budget.
- Don't hardcode dataset paths — use HuggingFace `datasets`.
- Don't write a "let me also build X" feature without asking. Stay in scope.
- Don't catch exceptions silently. If you catch, log and re-raise or handle visibly.

## Code style

- Black (line length 100) + ruff. Run before committing.
- Tests with pytest. Every module gets at least a smoke test that imports the module and exercises one happy path.
- Docstrings: short. One-line summary; expand only when behavior is non-obvious.

## Definition of done (per module)

A module is done for v1 when all four hold:

1. Public API matches what `SPEC.md` and dependent modules need.
2. `pytest tests/<module>/` passes, including at least one smoke test.
3. The end-to-end smoke run (`run=smoke`) exercises this module without a stub for it.
4. The human has eyeballed real output for ≥10 examples and they look sane.

Claude Code does not declare a module done. Only the human does, after step 4.

## Build, test, run

```bash
# install
uv sync                                  # or: pip install -e .

# test
pytest                                   # all
pytest tests/kg                          # one module
pytest -m "not network"                  # skip live Wikidata

# run
python -m debate_kg.main run=smoke       # tiny end-to-end on 1 claim
python -m debate_kg.main run=fever_100   # first 100 FEVER claims
```

## Working preferences

- Use plan mode for anything touching multiple modules or the round loop.
- For single-module changes, normal mode is fine — but show me the diff before applying.
- When stuck, read the relevant paper section listed in `SPEC.md` rather than guessing.
- Update this file when you discover a convention that prevented a class of mistakes. Keep it under ~200 lines.
