# Experiments

Self-contained experiments on BreastMNIST. Each directory holds its scripts,
its results, and a README explaining what question it answers.

| # | Experiment | Question |
|---|---|---|
| 01 | `01_zeroshot_vlm_classification` | Can a local vision LLM classify BreastMNIST directly, with no debate and no fine-tuning? |

## Why these exist

The debate pipeline in `breastMnist/` produces graphs whose claim text carries
almost no label signal (TF-IDF → logistic regression reaches AUC 0.5405 on the
test split; structural graph features reach 0.4933, i.e. chance). Before
investing further in the debate architecture, we need to know **where** the
signal is lost. Three candidates:

1. the VLM cannot read breast ultrasound at all,
2. the debate design destroys information the VLM does have,
3. the graph construction discards it.

Experiment 01 isolates (1): strip away the debate, the knowledge graph, and the
multi-round structure, and just ask the model the question. Whatever it scores
is the ceiling any debate built on that model can hope to reach.

## Hardware

All of this runs CPU-only — 20 cores, 31 GB RAM, no CUDA device
(`torch.cuda.is_available()` is `False`; Ollama logs `offloaded 0/29 layers to
GPU`). Inference costs below are measured on that machine, not estimated.

## Metric convention

BreastMNIST is imbalanced — the test split is 27% malignant — so raw accuracy
is misleading: always answering "benign" scores **0.7308**. Every experiment
reports **balanced accuracy** `(sensitivity + specificity) / 2` as the headline
number, where 0.5 is chance regardless of class ratio. AUC, MCC, sensitivity
and specificity are reported alongside; raw accuracy and the majority baseline
are included only for reference.

Label convention follows MedMNIST v2: `0 -> MALIGNANT`, `1 -> BENIGN`.
MALIGNANT is the positive class throughout.

## Reference points

| source | accuracy | AUC |
|---|---|---|
| published MedMNIST baselines (ResNet-18/50, AutoML) | ~0.86–0.90 | ~0.89–0.92 |
| majority-class baseline (test split) | 0.7308 | 0.500 |
| debate pipeline, claim text → logistic regression | 0.5962 | 0.5405 |
| debate pipeline, structural graph features | — | 0.4933 |
