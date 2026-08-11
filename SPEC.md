# MSc Thesis — Combined Project Spec

This is the single source of truth for all subprojects. Subdirectory `SPEC.md` files have been merged here.

---

# Part 1 — BreastMNIST: Multi-Agent Debate + KG Dataset

## Goal

Run a six-mode ablation study to isolate the contribution of each pipeline component — debate, graph-based verdict, domain KG, adversarial stances, adaptive escalation — on binary classification of breast ultrasound images (BENIGN / MALIGNANT) from the BreastMNIST dataset (MedMNIST v2).

Separately, build a full merged graph dataset (debate graph + KG) across all dataset splits for downstream GNN training.

## Dataset

- **BreastMNIST** from MedMNIST v2. Images resized to 224×224 and converted to RGB JPEG (base64-encoded for prompt injection).
- **Label convention**: `0 → MALIGNANT`, `1 → BENIGN` (confirmed from MedMNIST v2 source).
- **Ablation set**: `data/breast/ablation_indices.json` — 50 test-set indices generated once from `np.random.default_rng(42)`. All six ablation stages use exactly these indices. Never regenerate.
- **Full dataset splits** (for graph dataset construction):
  - Test: 50 ablation + 106 remaining = 156 total (`data/breast/remaining_indices.json`)
  - Val: 78 samples (`data/breast/val_indices.json`, indices 0–77)
  - Train: 546 samples (`data/breast/train_indices.json`, indices 0–545)

## Domain Knowledge Graph

Pre-built ACR BI-RADS ultrasound lexicon.

- `data/breast/knowledge_graph.json` — 370 triples, IDs `t_001`–`t_370`.
- `data/breast/definitions.json` — 100 entity definitions, IDs `d_001`–`d_100`.
- By default only the 370 triples are serialized into prompts (`include_definitions=False`, ~6.5K tokens). The 100 definitions are loaded but not shown to the model unless `include_definitions=True` (triples + definitions together would be ~25K tokens, too large for the expert context window).
- Experts cite KG nodes with `[CITED:t_NNN]` and definitions with `[CITED:d_NNN]`. Citations are parsed into `DebateNode.provenance`.

## KG Retrieval

Rather than injecting the full KG into every prompt, a **TF-IDF retriever** (`TFIDFRetriever`) selects a focused subset of triples per round:

- **Round 0 (seed)**: top-`kg_round0_k` triples (default 30) retrieved for a label-conditioned seed query derived from each expert's assigned stance.
- **Rounds 1+**: only triples previously cited (`[CITED:t_NNN]`) by any claim node in prior rounds (*focused KG*). Falls back to the seed KG if no citations exist.
- **Claim→triple edges in dataset**: top-`kg_retrieve_k` TF-IDF nearest triples per claim node (default k=3), computed fresh at dataset-build time (not stored in debate JSON).

Parameters in `conf/run/breast_adversarial.yaml`:

| Param | Value | Meaning |
|---|---|---|
| `kg_retrieve_k` | 3 | claim→triple edges per claim in graph dataset |
| `kg_round0_k` | 30 | seed triples injected in round 0 |

## Stack

| Component | Choice |
|---|---|
| Expert model | Qwen2.5-VL-7B-Instruct — **always via Ollama**; CPU in dev, GPU on cluster. No vLLM path for experts. |
| Context window | `num_ctx=8192` (KV-cache; 12 288 tokens causes GPU OOM on available hardware) |
| KG retrieval | `TFIDFRetriever` (scikit-learn TF-IDF, `ngram_range=(1,2)`, cosine similarity) |
| Judge model | MedGemma-4B-IT — single judge, two scoring dimensions (`skip_judge=true` for full dataset runs) |
| Judge backend | `ollama` (shares the expert's Ollama instance) or `vllm` (separate server) — set in `conf/judges/medgemma.yaml` |
| Input | 224×224 breast ultrasound image (base64 JPEG) + optional text |
| Labels | BENIGN / MALIGNANT |

## Six Ablation Modes

| Mode | Config | Debate | KG | Verdict mechanism |
|---|---|---|---|---|
| `single_agent` | `breast_single` | No | No | Parse `[LABEL][CONFIDENCE]` from single VLM call |
| `debate_opinion` | `breast_debate_opinion` | Free-text, multi-round | No | MedGemma `pick_winner` → winner's last label |
| `debate_graph` | `breast_debate_graph` | Structured `[CLAIM]` | No | Weighted node-score vote |
| `debate_kg` | `breast_debate_kg` | Structured `[CLAIM]` | Yes (TF-IDF) | Weighted node-score vote |
| `debate_kg_adversarial` | `breast_adversarial` | Structured `[CLAIM]` | Yes (TF-IDF) | Weighted node-score vote |
| `adaptive_adversarial` | `breast_adaptive` | Conditional | Yes (TF-IDF) | Confidence ≥ threshold → single-agent; else → adversarial weighted vote |

In `debate_kg_adversarial` and `adaptive_adversarial`: Expert A is locked to BENIGN, Expert B to MALIGNANT throughout all rounds.

## Verdict Mechanisms

**Weighted vote (modes 3–5)**:
```
score = Σ weight_i × vote_i
weight_i = (groundedness + factuality) / 200   # MedGemma judge scores, 0–100 each
vote_i   = +1 (MALIGNANT) | −1 (BENIGN)
score > 0 → MALIGNANT, score < 0 → BENIGN, tie → majority vote
```
`malignant_score` for AUC-ROC = `clip((score + 1) / 2, 0, 1)`.

When `skip_judge=true` (used for full dataset construction runs): all weights default to **0.5** (equal), reducing the weighted vote to a plain majority vote. This avoids MedGemma inference cost at the scale of the full dataset.

**Opinion verdict (mode 2)**: MedGemma given full transcript + image; outputs `WINNER: expert_a|expert_b`. Winner's most recent stated `VERDICT: BENIGN|MALIGNANT` is the system verdict. `malignant_score` = 1.0 or 0.0 (no confidence in this mode).

**Adaptive verdict (mode 6)**: Run single agent first. If `confidence >= confidence_threshold` (default 70), use single-agent label. Else escalate to full adversarial KG debate and use the weighted vote.

## Debate Format — Opinion Mode (mode 2)

Free-text — no `[CLAIM]` blocks. One `DebateNode` per expert turn. Each expert knows their identity (A or B). The system prompt enforces two distinct response shapes:

- **Opening turn**: name specific observable features — echogenicity, shape, margins, posterior acoustic features, vascularity, orientation, size, calcifications — then end with `VERDICT: BENIGN` or `VERDICT: MALIGNANT`.
- **Response turn**: first check whether there is a named image feature not yet mentioned by anyone.
  - If yes → `NEW FINDING: <one sentence>` + one AGREE/DISAGREE sentence re: the other expert's last point + `VERDICT: ...`
  - If no → `[FINISH]` + `VERDICT: ...`. `[FINISH]` means "nothing new to add" — it does not imply agreement; the verdict can still differ.

Max 3 sentences total before the verdict line.

## Debate Format — Structured Modes (modes 3–5)

Each expert turn produces 2–4 `[CLAIM]` blocks. One `DebateNode` per block. Short IDs (`c1`, `c2`, …) are globally sequential across all rounds. Example:
```
[CLAIM] <observation> [CITED:t_042] [LABEL: BENIGN]
[CLAIM] <response to prior claim> [ADDRESSED:c3][DISAGREE] [CITED:d_007] [LABEL: BENIGN]
```
**Mandatory addressing**: whenever debate history is non-empty, the expert is instructed that at least one `[CLAIM]` block must carry an `[ADDRESSED:cN][AGREE|DISAGREE]` tag. Prompt-enforced, not code-enforced.

## Consensus Check

- **Structured modes**: stop if all debate edges are positive (no `−` edges) AND at least one edge exists, OR after `max_rounds`. **Round 0 never stops early** — consensus can only be declared from `round_idx >= 1` onward.
- **Opinion mode**: stops early (from round ≥ 1) on either: (1) any expert writes `[FINISH]`, or (2) both experts state the same verdict for 2 consecutive rounds (`consecutive_agreement >= 2`).

## Judge — MedGemma-4B-IT

Single judge. `JudgeScore` fields repurposed:
- `groundedness` → **IMAGE_GROUNDING** (0–100): how well the claim cites observable image features.
- `factuality` → **MEDICAL_ACCURACY** (0–100): how ACR BI-RADS correct the claim is.

`score_utterance(node, kg, image_b64)` → `JudgeScore` (structured modes).
`pick_winner(all_nodes, image_b64)` → `WinnerJudgment` (opinion mode).

When `backend: ollama`, the judge's `num_ctx` is capped at 4096 (vs the expert's 8192) so its KV-cache doesn't crowd the expert model off the GPU.

Used for the 50-sample ablation. Disabled (`skip_judge=true`) for full dataset runs.

## GPU Memory Management

The expert model (always Ollama) and an Ollama-backend judge share one Ollama instance. `breast_main` unloads each model (`keep_alive=0`) before the other runs. `OllamaClient` retries connection errors, timeouts, and 5xx with backoff (~30 min total) and restarts Ollama via `$OLLAMA_RESTART_SCRIPT` (set in `.env`). On OOM (`"model requires more system memory"`): retries up to 72 times × 10 min = 12 h, waiting for RAM to free.

## Metrics

- **Accuracy** — primary metric.
- **Sensitivity** (recall for MALIGNANT) — clinically critical; false negatives are dangerous.
- **Specificity** (recall for BENIGN).
- **AUC-ROC** — computed from per-sample `malignant_score`.
- **Convergence rate** — % of samples reaching early stop within `max_rounds`.
- **Mean rounds** — average debate length.
- **Mean time per sample** — wall-clock seconds.

All metrics written to `outputs/<timestamp>/metrics.json`.

## Output Artifacts (per run)

- `metrics.json` — aggregated metrics + per-sample records.
- `debate_sample-<id>.json` — debate nodes, edges, verdict for each sample.
- `outputs/resume/<mode>_<num_samples>_<seed>.jsonl` — one JSON line per completed sample, appended after each. Rerunning skips already-completed IDs. Deleted on clean pipeline exit.

## Resume and Fault Tolerance

- Resume file tracks completed sample IDs; on restart the pipeline reads it, skips done samples, and resumes where it left off.
- Per-split `.split_{split}` marker files in each output dir prevent double-counting across splits with overlapping IDs (val 0–77 and train 0–545 share IDs).
- A watchdog shell loop wraps the monitor daemon to restart it if OOM-killed.
- `STALL_LIMIT=46800s` (13h) — longer than OOM retry patience (12h) so the daemon doesn't kill the pipeline mid-retry.

## Graph Dataset Construction

After all splits complete, `debate_kg.dataset.build_graph_dataset` builds a merged heterogeneous graph per sample:

| Element | Description |
|---|---|
| Claim nodes | One per `[CLAIM]` block: text, expert label, round, expert identity |
| Triple nodes | KG triples connected to at least one claim in this sample |
| Claim→claim edges | AGREE / DISAGREE, parsed from `[ADDRESSED:cN][AGREE\|DISAGREE]` tags |
| Claim→triple edges | Top-k TF-IDF nearest triples per claim (k=3), computed fresh at build time |
| Triple→triple edges | Shared-entity edges: two triples sharing a subject or object |
| Graph label | Gold label: BENIGN=0, MALIGNANT=1 |

**Output layout** (`data/breast/dataset_full/`):
- `graphs/sample_{id}.json` — one file per sample
- `index.json` — `{sample_id: {label, verdict, correct, num_claims, num_triples, …}}`
- `stats.json` — aggregate statistics

Run:
```bash
python -m debate_kg.dataset.build_graph_dataset \
    --debate-dir outputs/2026-07-18/16-12-46 \
    --debate-dir outputs/... \
    --out-dir data/breast/dataset_full \
    --k 3 --force
```

## Key Design Decisions

1. **TF-IDF KG retrieval, not full injection.** The context window constraint (`num_ctx=8192`) makes injecting all 370 triples impractical. Round 0 uses a label-conditioned seed query to retrieve the 30 most relevant triples; subsequent rounds use only triples already cited in the debate (focused KG).

2. **`skip_judge` for dataset runs.** MedGemma scoring adds ~5 min/sample. For 730 samples across all splits that's ~60h of judge inference. `skip_judge=true` uses equal-weight majority vote instead.

3. **Single judge for ablation.** MedGemma-4B-IT is medically fine-tuned — more appropriate than a general-purpose panel for this domain.

4. **Fixed ablation sample set.** `ablation_indices.json` ensures all six stages compare on identical inputs. Never regenerate.

5. **Claim→triple edges recomputed at build time.** Debate JSON stores `[CITED:t_NNN]` provenance from the prompt, but claim→triple edges in the graph dataset use fresh TF-IDF retrieval (top-3 per claim) for consistency.

---

# Part 2 — HINPool Benchmark Experiment

## Goal

Reproduce HINPool (AAAI 2026) — type-aware heterogeneous graph pooling for graph-level classification — on the TUDataset benchmarks HINPool reports on (MUTAG, PROTEINS, optionally ENZYMES), with a rigorous fixed-seed, 5-run, mean ± std protocol. Then the same model transfers to the BreastMNIST debate graphs (Part 1) later.

Build the **baseline first**, prove the harness, then add HINPool on top.

## Benchmark Datasets

| Dataset | Graphs | Avg nodes | Classes | Notes |
|---|---|---|---|---|
| MUTAG | 188 | ~18 | 2 | Smallest; use for fast iteration |
| PROTEINS | 1113 | ~39 | 2 | Headline dataset; closest in size to 546 BreastMNIST train graphs |
| ENZYMES | 600 | ~33 | 6 | Optional; multi-class stress test |

These are homogeneous by default. HINPool treats them as HINs using existing node labels (atom type / amino-acid secondary-structure) as the **node type**. We do the same so numbers are comparable to the paper.

## Rigor Protocol (non-negotiable)

1. Set ALL seeds at the start of each run: `random`, `numpy`, `torch`, `torch.cuda` via `set_seed(seed)`.
2. Run each config **5 times** with seeds `[0, 1, 2, 3, 4]`.
3. Report **mean ± std** of the test metric — never a single number.
4. **Fixed 80/10/10 split** per dataset, persisted to `splits/<dataset>.json`, reused across all 5 runs (only model init varies).
5. **Early stopping on validation metric**; report test metric at best-val epoch (not best test — that leaks).
6. Primary metric: **AUROC** (binary), to match HINPool. Accuracy as secondary.
7. Log every hyperparameter and seed to `results/<name>.csv`.

## Stack

- Python 3.10+, PyTorch + PyTorch Geometric
- `TUDataset` for data
- `RGCNConv` (with `num_bases`) as heterogeneous encoder inside HINPool
- scikit-learn for `roc_auc_score`

## Code Layout

```
hinpool-bench/
  src/
    seed.py      # set_seed()
    data.py      # load TUDataset, assign node types, fixed split
    model.py     # TAS, RA, THeGPLayer, HINPool, baselines
    train.py     # train/eval loop, early stopping
    run.py       # CLI: 5 seeds, writes results CSV
  results/       # CSV/JSON outputs
  splits/        # persisted split indices per dataset
```

## HINPool Architecture (three components)

### Type-Aware Selector (TAS)
- One small MLP **per node type** maps node embedding → scalar score.
- **Top-K within each type** (pool_ratio default 0.9).
- Gate kept embeddings by sigmoid score (differentiable).
- Slice node set + adjacency to kept nodes for next layer.

### Readout Aggregator (RA)
- Per-type mean → one vector per node type → fuse (concat, paper's best).
- Plus a mixed global mean over all nodes.
- Concatenate the two → this layer's graph embedding.

### Cross-Layer Fusion
- N THeGP layers (start N=3). Collect each layer's RA output; fuse across layers (concat or attention).
- **This is the highest-value component** — HINPool ablation: removing cross-layer fusion = −39% on MUTAG; removing TAS = −1..−5%.

## Build Order

1. Environment + seeding + data with fixed split.
2. Baseline RGCN + mean pool running end-to-end on MUTAG — **prove the harness before building HINPool**.
3. Add PROTEINS.
4. HINPool: TAS → RA → cross-layer fusion, one component at a time.
5. Comparison table vs paper's reported numbers (MUTAG ~100, PROTEINS ~89.8, ENZYMES ~92.8 AUROC).

## Known Pitfalls

- HINPool's gains come mostly from cross-layer fusion, not type-aware machinery — implement it carefully.
- TAS does node dropping (per-type top-K). For the eventual BreastMNIST debate graphs where claim nodes are sparse, keep an attention-pooling readout variant available.
- Use AUROC + weighted loss (`pos_weight` in `BCEWithLogitsLoss`) to handle class imbalance.
- Pooling ratio 0.9 over 3 layers is HINPool's reported sweet spot — start there.

## Definition of Done

- [ ] Baseline RGCN reproduces sane AUROC on MUTAG and PROTEINS (5 runs, mean ± std)
- [ ] HINPool implemented and run on MUTAG and PROTEINS (5 runs, mean ± std)
- [ ] Results table: baseline vs HINPool vs paper's reported numbers
- [ ] `results/` contains CSV with every seed + hyperparameter logged
- [ ] One-paragraph reproducibility statement
