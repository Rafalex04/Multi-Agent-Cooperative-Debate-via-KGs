# CLAUDE.md — HINPool Benchmark Experiment

Conventions and context for this repo. Read this first in every session.

## What this project is

A **standalone experiment**, separate from ThothGNN. Goal: reproduce HINPool
(type-aware heterogeneous graph pooling, AAAI 2026) on a public graph-level
classification benchmark, with rigorous protocol (fixed seeds, multiple runs,
mean ± std). Once it works here, the same model code transfers to the ThothGNN
debate graphs in a different repo.

This repo does **not** contain any ThothGNN / BreastMNIST / debate code. Keep it
clean and self-contained.

## The benchmark

Use the **TUDataset** graph-classification datasets that HINPool itself reports
on, because they are the only ones where we can directly compare to published
numbers:

- **MUTAG** (188 graphs, ~18 nodes, binary) — smallest, use for fast iteration
- **PROTEINS** (1113 graphs, ~39 nodes, binary) — closest in size to ThothGNN's
  546 graphs; this is the headline dataset
- **ENZYMES** (600 graphs, ~33 nodes, 6-class) — optional, multi-class stress test

These are homogeneous by default. HINPool treats them as HINs by using the
existing node labels (atom type / amino-acid secondary-structure) as the
**node type**. We do the same so our numbers are comparable to the paper.
Document this clearly — it is the paper's own protocol, not a shortcut.

NOTE: a true heterogeneous graph-classification benchmark with multiple typed
node sets per graph is rare. If we later want one, the candidates to investigate
are OGB molecular datasets or sampled KG subgraphs — but for *reproducing
HINPool's reported numbers*, the TUDatasets above are the correct choice. Do not
silently swap datasets.

## Rigor protocol (non-negotiable — this is the whole point)

The supervisor's complaint was lack of rigour, specifically inconsistent seeds.
Every experiment MUST:

1. Set ALL seeds at the start of each run: `random`, `numpy`, `torch`,
   `torch.cuda`. Use a helper `set_seed(seed)`.
2. Run each config **5 times** with seeds `[0, 1, 2, 3, 4]` (or `base+run`).
3. Report **mean ± std** of the test metric, never a single number.
4. Use a **fixed train/val/test split** per dataset (HINPool uses 80/10/10).
   Use the same split indices across the 5 runs; only the model init / seed
   varies. Persist the split to disk so it is reproducible.
5. **Early stopping on validation metric**, report the corresponding **test**
   metric (not the best test metric — that leaks).
6. Primary metric: **AUROC** (binary) to match HINPool. Accuracy as secondary.
7. Log every hyperparameter and seed to a results file (CSV or JSON).

If a result cannot be reproduced from the logged seed + config, it does not
count.

## Stack

- Python 3.10+
- PyTorch + PyTorch Geometric (PyG)
- `torch_geometric.datasets.TUDataset` for data
- `RGCNConv` (with `num_bases`) as the heterogeneous encoder inside HINPool
- scikit-learn for `roc_auc_score`

## Code layout

```
hinpool-bench/
  CLAUDE.md            # this file
  SPEC.md              # what to build, step by step
  requirements.txt
  src/
    seed.py            # set_seed()
    data.py            # load TUDataset, assign node types, fixed split
    model.py           # TAS, RA, THeGPLayer, HINPool, baselines
    train.py           # train/eval loop, early stopping
    run.py             # CLI: runs 5 seeds, writes results
  results/             # CSV/JSON outputs, one per experiment
  splits/              # persisted split indices per dataset
```

## Conventions

- Keep the model in `model.py` **encoder-agnostic** so the same `HINPool`
  class can later be imported by the ThothGNN repo with a different data loader.
- Every script takes `--seed`, `--dataset`, `--out`. No hardcoded paths.
- Prefer small, runnable increments: get MUTAG + a plain RGCN baseline working
  end-to-end BEFORE building the full HINPool. A baseline that runs beats a
  perfect model that doesn't.
- Always implement the **baseline first** (RGCN + mean pool), confirm the
  rigour harness works, then add HINPool's TAS / RA / cross-layer fusion on top.
- When in doubt about a HINPool detail, the source of truth is the paper
  (HINPool, AAAI 2026) and its repo: https://github.com/ntuidssplab/HINPool

## Known pitfalls (from prior analysis of the paper)

- HINPool's gains come mostly from **cross-layer fusion**, not the type-aware
  machinery (their ablation: removing cross-layer fusion = −39% on MUTAG;
  removing TAS = −1..−5%). Implement cross-layer fusion carefully — it matters
  most.
- TAS does **node dropping** (per-type top-K). With `num_bases` and 40+
  relations this is fine here, but for the eventual ThothGNN graphs the debate
  nodes are sparse — keep an attention-pooling variant available as an
  alternative readout for later.
- Pooling ratio 0.9 over 3 layers is HINPool's reported sweet spot. Start there.
- Class imbalance: use AUROC + weighted loss (`pos_weight` in
  `BCEWithLogitsLoss`) so the model can't win by predicting the majority class.

## Definition of done (for the Monday deliverable)

- [ ] Baseline RGCN reproduces a sane AUROC on MUTAG and PROTEINS (5 runs, mean ± std)
- [ ] HINPool implemented and run on MUTAG and PROTEINS (5 runs, mean ± std)
- [ ] Results table comparing baseline vs HINPool vs paper's reported numbers
- [ ] results/ contains the CSV/JSON with every seed + hyperparameter logged
- [ ] One-paragraph reproducibility statement (seeds, split, early-stopping rule)
