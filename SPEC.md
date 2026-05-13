# Multi-Agent Agreement and Decision-Making via Knowledge Graph Merging — v1 Spec

This document is the source of truth for what to build. Reading list and learning roadmap live in `READING.md`.

## Glossary

- **Claim** — input to the system; one FEVER claim text.
- **Triple** — `(subject, predicate, object)` in a knowledge graph, with a stable UUID for provenance.
- **Expert turn** — one LLM call by one expert, which produces one or more claim nodes.
- **Claim node** (`DebateNode`) — one factual assertion made by an expert, corresponding to one `[CLAIM]` block in the structured response. A single expert turn may produce N claim nodes.
- **Short claim ID** — a debate-wide sequential label (`c1`, `c2`, …) assigned to each claim node in order of production. Experts use these IDs to address prior claims.
- **Edge** — relation between two claim nodes; signed `+` (AGREE) or `−` (DISAGREE). Derived by parsing `[ADDRESSED:<ID>][AGREE|DISAGREE]` tags — no separate LLM classifier.
- **Round** — one full pass: retrieve → expert A turn → expert B turn → judge → consensus check → (merge if continuing). Opening expert alternates: A opens even rounds, B opens odd rounds.
- **Verdict** — final SUPPORTS / REFUTES / NOT ENOUGH INFO label produced by a weighted vote over all claim nodes when the loop terminates. No LLM call — fully deterministic.
- **Expert** — one of the two LLM agents grounded in its own KG. Both experts use the same base model in v1.
- **Judge** — one of the three scoring LLMs from different model families.

## Goal

Build an end-to-end pipeline for multi-agent agreement and decision-making between two LLM agents grounded in different knowledge graphs. Agents hold a cooperative dialogue; disagreement signals incomplete knowledge; the dialogue itself drives iterative knowledge graph merging until consensus or a round limit.

v1 uses a rule-based merger as a strong, interpretable baseline. v2 (later) replaces it with a learned GNN.

## Core framing

Existing multi-agent LLM debate research treats agents as adversarial debaters where one wins, or as truth-seekers around a known answer. Both assume disagreement is something to resolve and discard.

We reframe disagreement as a signal about the agents' underlying knowledge: when two grounded experts disagree, their knowledge is incomplete, and the dialogue is the mechanism by which that incompleteness gets fixed. Each agent is grounded in a knowledge graph; debate exposes gaps; the gaps drive iterative KG merging; agreement emerges as the KGs converge. The technical mechanism is operationalized via belief merging — score-weighted operations over triples, with provenance from debate nodes back to source triples.

## Pipeline (six modules, looped)

```
input claim
  ↓
[1 KG Construction] → [2 GraphRAG Retriever] → [3 Debate Orchestrator]
                                                       ↓
                                                  [4 Judge Panel]
                                                       ↓
                                                  [5 Consensus Check] ── stop ──→ verdict + fused KGs
                                                       │ continue
                                                       ↓
                                                  [6 KG Merger] ── feedback ──→ back to module 2
```

Each round: experts retrieve from their current KGs via GraphRAG, debate one turn at a time, judges score every utterance, the consensus check decides whether to stop, otherwise the merger updates both KGs based on judge scores. Loop until consensus or `max_rounds`.

### Module 1 — KG Construction
- Two **non-overlapping** subgraphs from Wikidata, scoped to topics relevant to FEVER claims.
- Shared **triple-based** schema with stable UUIDs for provenance.
- Offline preprocessing — not part of the round loop.
- Design lineage: REBEL, Pan et al. roadmap, EDC. See `READING.md`.

### Module 2 — GraphRAG Retriever
- Per-turn subgraph retrieval: entity-link the current debate context, k-hop expand, serialize into the prompt.
- Bridge between symbolic KG and LLM input.
- Design lineage: Edge et al. GraphRAG, G-Retriever, Think-on-Graph. See `READING.md`.

### Module 3 — Debate Orchestrator
- Cooperative turn-taking. Both experts use the **same** base model (Llama-3.1-8B via Ollama in dev, vLLM on cluster).
- **Structured response format**: each expert turn produces one or more `[CLAIM]` blocks. Every block ends with `[LABEL: SUPPORTS | REFUTES | NOT ENOUGH INFO]` (the expert's verdict on the FEVER claim) and optional `[ADDRESSED:<ID>][AGREE|DISAGREE]` tags referencing prior claim nodes by short ID.
- **One DebateNode per [CLAIM] block.** A single LLM call may produce N nodes. Short IDs (`c1`, `c2`, …) are globally sequential across all rounds.
- **Edges are parsed from tags** — no separate edge-classifier LLM call. `[ADDRESSED:c2][DISAGREE]` → edge with sign `−` from the new node to c2.
- **Alternating opener**: Expert A opens even-numbered rounds; Expert B opens odd-numbered rounds. The second expert sees the first expert's new claims before responding.
- **History**: all prior DebateNodes are passed as context each round so experts can address any previous claim by short ID.
- Provenance is non-optional — each DebateNode carries the Triple UUIDs cited via `[N]` inline references; the merger uses these to up-/down-weight specific triples.
- Design lineage: Du et al., TreeDebater, Khan et al. See `READING.md`.

### Module 4 — Judge Panel
- Three locally-hosted open-weight judges from **different model families** (Llama, Qwen, Mistral) to reduce correlated bias.
- Per-utterance rubric: **KG-groundedness** + **factuality** (2 dimensions, 0–100 each).
- Each judge also scores each expert overall (0–100) at end of debate.
- Aggregate via mean; report inter-judge agreement (Krippendorff's α).
- Design lineage: Zheng et al., Verga et al. See `READING.md`.

### Module 5 — Consensus Check
- Stop if **all debate edges are positive (no `−` edges) AND at least one edge exists**, OR after `max_rounds` (hard cap).
- Rounds that produce zero edges (no `[ADDRESSED]` tags used) do **not** satisfy the all-positive condition — the hard cap is what terminates in practice when experts make no cross-references.
- Pure logic, no learning.

### Module 6 — Rule-based Merger (v1)
- Operate over the union of both KGs. Each triple gets a decision: **keep / drop / resolve** in favor of the higher-scoring side.
- Score combines the expert's overall score with the per-utterance scores of debate nodes citing the triple. The dialogue drives the merge.
- Three minimal consistency checks after triple-level decisions: type, symmetry, ontology-light.
- Triples scoring below `merger_score_floor` are dropped, not resolved.
- **Acknowledged limitation**: no full logical/semantic consistency under edits. That is the belief revision problem — decades old, unsolved at scale. Documented as future work.
- Debate graph is discarded between rounds; only KGs persist.
- Design lineage: Konieczny & Pino-Pérez, Flouris et al. See `READING.md`.

## Stack

| Component | Choice | Role |
|---|---|---|
| Language | Python 3.11 | — |
| KG storage | NetworkX (MultiDiGraph) | with stable triple UUIDs |
| LLM serving | Ollama (debug) / vLLM (cluster) | local, $0 |
| Expert model | Llama-3.1-8B-Instruct | both experts, same weights |
| Judge models | Llama-3.1-8B + Qwen-2.5-7B + Mistral-7B-Instruct | different families |
| Embeddings | sentence-transformers | entity linking, retrieval ranking |
| Tensor / GNN | PyTorch + PyG | PyG plumbed in early though GNN is v2 |
| Data loading | HuggingFace `datasets` | FEVER |
| Wikidata | SPARQLWrapper or qwikidata | choose one |
| Validation | Pydantic | data structures crossing module boundaries |
| Config | Hydra | composable run configurations |
| Testing | pytest | unit + smoke tests |

Strict $0 budget. Everything runs locally on the university cluster. No paid APIs.

## Verdict mechanism

The final verdict is a **weighted vote** — no LLM call:
- Each DebateNode contributes its label value: SUPPORTS=+1, NOT ENOUGH INFO=0, REFUTES=−1.
- Each node is weighted by its mean judge score: `(groundedness + factuality) / 2`.
- Weighted average is thresholded: `> 0.5` → SUPPORTS, `< −0.5` → REFUTES, otherwise NOT ENOUGH INFO.
- If all judge weights are zero, falls back to unweighted majority vote.

## Output artifacts (per run)

Every run writes to `outputs/<timestamp>/`:

- `config.yaml` — resolved Hydra config (automatic).
- `claims_index.json` — list of all claims with ID, text, slug, and verdict.
- `debate_<id>_<slug>.json` — debate graph: claim nodes with provenance to KG triple IDs, signed edges.
- `kgs/<id>_<slug>/round_<n>/{a,b}.json` — both experts' KG state per round; lets KG drift be computed retroactively.
- `judge_scores_<id>_<slug>.json` — per-node scores per judge + overall expert scores.
- `metrics.json` — aggregated run-level metrics.
- `run.log` — stdlib logging output.

Artifact filenames use a `<numeric_id>_<slug>` prefix (e.g. `75193_beethoven-was-born-in`) for human readability. A run is fully described by `config.yaml` + a seed.

## Benchmark and metrics

**Benchmark**: FEVER (Thorne et al., NAACL 2018). 185K Wikipedia-derived claims, labels SUPPORTS / REFUTES / NEI, Wikidata-linkable.

**Four reported metrics**:

1. **Label accuracy / macro-F1** — main metric vs. ground truth.
2. **Convergence rate** — % of cases reaching all-positive consensus within `max_rounds`.
3. **Rounds to consensus** — average dialogue length.
4. **KG drift** — set-difference between consecutive rounds.

## Baselines

1. **Single-expert** (no debate, GraphRAG only) — does the dialogue help?
2. **Naive union merger** (no scoring) — does score-weighted merging help?
3. **Random merger** — sanity check on signal quality.

## Success criteria for v1

v1 has something to say if all four hold:

1. **End-to-end pipeline runs** on at least 200 FEVER claims without manual intervention.
2. **Beats the single-expert baseline** on label accuracy.
3. **Convergence is non-trivial** — ≥50% of cases reach all-positive consensus within `max_rounds`; average rounds-to-consensus meaningfully below the cap.
4. **Inter-judge agreement is meaningfully above chance** (Krippendorff's α or similar).

Failure of (2) or (3) is a result to report. Failure of (4) blocks reporting until fixed.

## Hyperparameters (v1 initial values)

| Param | Initial | Where | Notes |
|---|---|---|---|
| `max_rounds` | 5 | conf/run | hard consensus cap |
| `kg_split_overlap` | 0.0 | conf/data | strict disjoint; ablation values: 0.0, 0.1, 0.25 |
| `retriever_k_hops` | 2 | conf/run | per-turn subgraph radius |
| `retriever_max_triples` | 50 | conf/run | per-turn token budget |
| `expert_temperature` | 0.7 | conf/model | same for both experts |
| `judge_temperature` | 0.0 | conf/model | stable scoring |
| `judge_count` | 3 | conf/judges | one per family |
| `merger_score_floor` | 0.3 | conf/run | triples below this are dropped, not resolved |

These are starting points. `kg_split_overlap`, `judge_count`, `merger_score_floor` are ablation axes.

## Operating commitments

- Same expert model with different KGs (controls for model effect).
- Different judge model families (reduces correlated bias).
- Strict $0 budget, all local on the cluster, fully reproducible.

## Path to v2 (out of scope for v1)

- Replace rule-based merger with a self-supervised GNN (R-GCN or CompGCN on RotatE embeddings).
- Add neuro-symbolic consistency.
- Extend benchmarks: FEVER → SciFact → legal cases.

Anything in this section is **forbidden in v1**. Don't sketch it, don't scaffold it, don't add the imports.

## Non-goals for v1

- No GNN-based merger.
- No full logical consistency in merging.
- No domains beyond FEVER.
- No closed-source / paid model dependencies.

## Known risks

- **Debate quality is the biggest risk.** Vacuous, repetitive, or hallucinated debates kill the merger signal. Plan to iterate on prompts ≥10 times.
- **Judge reliability.** LLM judges are noisy. If three judges disagree wildly, the merger gets garbage signal. Compute inter-judge agreement before trusting any other number.
- **Wikidata coverage.** FEVER claims may reference entities with sparse Wikidata neighborhoods. Have a fallback (drop the claim, or expand to Wikipedia text → REBEL).
