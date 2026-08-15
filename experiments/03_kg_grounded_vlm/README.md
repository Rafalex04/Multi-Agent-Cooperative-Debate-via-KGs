# 03 — KG-grounded single-agent classification

Does giving a VLM retrieved knowledge-graph triples improve its classification
of BreastMNIST? Three steps per sample, no debate:

1. **Observe** — the VLM describes the image against the KG-derived descriptor
   schema. Stance-free: no diagnostic vocabulary, no stance-bearing triples.
2. **Retrieve** — that observation conditions KG retrieval (anchor match, 1-hop
   graph expansion, dense similarity), so triples vary per image.
3. **Classify** — the VLM sees the image again, now with the retrieved triples,
   and answers malignant yes/no. `P(malignant)` comes from first-token logprobs.

Compare against `../01_zeroshot_vlm_classification` — same model, same 156 test
images, same prompt style. The only difference is the KG.

## Result: the KG made it worse

| | MedGemma alone (exp 01) | MedGemma + KG (exp 03) |
|---|---|---|
| AUC | **0.6013** | **0.4734** (below chance) |
| predictions | 17 malignant / 139 benign | **0 malignant / 95 benign** |
| sensitivity | 0.4048 | 0.0000 |
| max `p_malignant` | 0.23 | 0.00134 |

Partial run, 95/156 samples. It collapsed to a constant classifier: every
prediction BENIGN, confidence pinned near 1e-05, and the ranking inverted so
malignant images score *lower* than benign ones.

## Why — retrieval quality, not quantity

Inspecting the triples actually delivered (mean 11 triples, 191 tokens per
sample), every one is pure taxonomy:

```
irregular is a shape descriptor
heterogeneous echo pattern is a echo pattern descriptor
oval is a shape descriptor
shape descriptor category descriptor mass finding
```

The model observed "irregular shape, heterogeneous echo pattern" — both
malignant features — and was handed 191 tokens explaining that irregular is a
kind of shape. A dictionary, not evidence. Four contributing faults:

1. **`max_stance_triples: 4` capped away the useful content.** Stance triples
   (`non_parallel_orientation suggests_malignancy true`) are the only
   diagnostically loaded ones. Logs show `0 stance` on nearly every sample. The
   cap exists because in the *debate* two stance triples held 8,774 citations
   and crowded everything out — correct there, wrong here.
2. **1-hop expansion leaks siblings.** Expanding from `irregular` reaches
   `shape_descriptor`, then every other shape value, so `oval` and `lobulated`
   appear in a prompt for an irregular mass.
3. **Text/image imbalance.** 191 KG tokens against 64 image tokens.
4. **Safety prior.** With no diagnostic content to counterweight it, the model
   defaults to the non-alarming answer.

## Suggested fixes (untested)

- Make stance triples a **floor**, not a ceiling — guarantee 6–8 per sample.
- Drop `is_a` taxonomy from the classification prompt; keep
  `suggests_malignancy`, `suggests_benign`, `ultrasound_appearance_includes`,
  `likelihood_of_malignancy`.
- Block graph expansion through category hub nodes.
- Re-test on ~15 samples (~50 min) before any full run. If `p_malignant` stays
  pinned near 1e-05, KG content is not the bottleneck — stop there.

## Reproduce

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ../../breastMnist/.venv/bin/python run_kg_grounded.py \
    --model medgemma:latest --split test
```

Requires `breastMnist/data/breast/schema.json` (generate with
`python -m debate_kg.kg.kg_schema`). Resumable — re-running skips samples
already in the `.jsonl`.

Cost on CPU: ~200 s/sample (observe ~120 s + classify ~78 s), so ~8.7 h for the
full 156. Two VLM calls per sample versus one in experiment 01.
