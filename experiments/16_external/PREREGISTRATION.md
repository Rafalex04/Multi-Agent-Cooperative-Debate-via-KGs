# Round 3 pre-registration — external validation and probe validity

**Written 2026-08-22, before any external score was computed.**
Frozen model file: `results/frozen_models.json`
SHA-256 `a08cdd8836a36c0613a488b12c577be5d53110ac25284e61902accdfbec443ec`

Ordering note, for honesty about what was fixed when: the E1 probe *inference*
on BUS-BRA was launched before this file was finished. That is measurement only
— it reads images and records `p(yes)` per finding, and depends on no label and
no model. Every model below was fitted on BreastMNIST train alone and hashed
before any BUS-BRA score, threshold or comparison was computed.

---

## 1. Frozen models

Fitted on BreastMNIST **train** only; hyperparameters by 5-fold CV on train only;
bAcc thresholds swept on train+val and then applied unchanged. No refitting on
external data in the primary analysis.

| id | description | fitted params | in-domain test AUC / bAcc |
|---|---|---|---|
| `kgsum` | `sum_f prior_f * z_f` — the KG's own sign vector on standardised probes | **0** | 0.7531 / 0.7475 |
| `logit` | free 32-weight logistic, loss+L2 by CV | 33 | 0.8133 / 0.7538 |
| `s5` | KG-signed Laplacian on the per-finding calibration layer, λ=0.01 registered | 65 | 0.8156 / 0.7669 |

Deviation recorded rather than hidden: the BreastMNIST-internal sweep aligned
samples on having *both* a debate and a probe vector. A model that must run where
no debate exists cannot inherit that intersection, so these three are fitted on
the probe-only alignment (546/78/156). The frozen object is the one reported
in-domain above and the one applied externally, so both numbers describe the
same thing.

## 2. Probe set

**Registered primary: P2 + P3** (direct/negative and verify/positive), 2 × 16 =
32 features, 2 calls per finding per image. These are the two strongest by
BreastMNIST CV (P2 0.8161, P3 0.8133, vs P1 0.8056, P4 0.8073), chosen on train
CV, never on test.

## 3. Data and framing

BUS-BRA (Zenodo 8231412, CC-BY-4.0): 1875 images, 1064 cases, 607 malignant /
1268 benign images and 342 / 722 cases, four scanners.

Framing is **registered before scoring**: a square window of side
`max(w,h) × 2.0` centred on the supplied lesion BBOX, shifted (never zero-padded)
to stay in frame, resized to 224. Rationale and the rejected alternatives are in
`prepare_busbra.py`. PAD=1.5 is a registered sensitivity arm.

Measured domain artifact, declared up front: radiologist calipers and burned-in
Portuguese annotation survive into ~45% of crops, marginally **more often in
benign** (46.8% vs 41.9%), so the artifact does not leak malignancy and if
anything cuts against the hypothesis under test.

## 4. Primary endpoints

- **E1** — external AUC/bAcc of `kgsum` and `logit` on BUS-BRA, per-image and
  per-case (mean score per case). **Per-case is primary.**
- **E3** — `s5` vs λ=0 vs 5 shuffled-KG draws, paired bootstrap (4000 resamples),
  decided at **P(better) ≥ 0.95**.
- **E2** — per-descriptor probe AUC against radiologist annotation on BrEaST,
  plus the perception/inference decomposition.
- **E4** — Spearman ρ between `kgsum` and radiologist BI-RADS; AUC for BI-RADS
  4/5 vs 2/3.

## 5. The comparison that frames the thesis

ResNet-18 trained on BreastMNIST (in-domain 0.9442), applied unchanged to
BUS-BRA. Registered hypothesis: **the zero-shot generalisation gap is smaller
than the supervised one.** If ResNet holds too, that is reported with equal
prominence.

## 6. Predictions, in writing

| quantity | predicted |
|---|---|
| probe external AUC (per-case) | 0.75 – 0.82 |
| ResNet-18 external AUC | 0.80 – 0.90 |
| S5 effect replicates at P≥0.95 | ~0.4 probability |
| `kgsum` external AUC | 0.68 – 0.78 (0-param, so most exposed to shift) |

**Kill criterion:** if `kgsum` falls below ~0.65 externally, the probes are
dataset-fragile and the generalisation claim dies. That is reported, not buried;
the BreastMNIST KG ablation (0.7128 vs 0.4740) stands regardless.

## 7. Deviations from the round-3 spec, and why

- **arXiv 2603.08921 ("MedCBR") located and read.** An earlier note in this file
  said the id did not resolve; that was wrong — the arXiv *API* returned no entry
  for it, but the paper exists and is now cited correctly. Harmanani et al.,
  "Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical
  Reasoning", 9 Mar 2026.

  **It is not a like-for-like comparator, and the write-up must say so.** MedCBR
  reports 94.2 ± 0.4 AUROC / 89.0 bAcc on BUS-BRA under *5-fold cross-validation
  with patient-level splits* — i.e. trained on BUS-BRA, with a radiologist's 15
  BI-RADS concept annotations as concept supervision. Its baselines (CLIP ViT-L/14
  93.5, CLIP-CBM 91.8, AdaCBM 87.9, CBM 84.8) are all likewise fitted in-domain.
  This round evaluates models that never see a BUS-BRA label at all. The two
  numbers answer different questions and will be tabulated as such: in-domain
  supervised ceiling versus zero-shot transfer. Registered before seeing our own
  external result, so it cannot be reframed afterwards.

  Their concept list overlaps ours substantially (shadowing, enhancement, halo,
  calcifications, skin thickening, circumscribed/spiculated/indistinct/angular/
  microlobulated margins, regular shape, hyper/hypo/heterogeneous/cystic echo).
  Their BUS-BRA concept annotations do not appear to be publicly released — the
  repo ships the concept *names* only — so E2 stays on BrEaST, which publishes
  descriptor ground truth.
- E5/E6 remain optional and are attempted only after E1–E4 are complete.

---

# Outcomes, scored against the predictions above

Filled in 2026-08-23 after the analysis ran. The frozen-model hash was unchanged
throughout: `a08cdd88…c443ec`.

## Primary endpoints

| model | params | in-domain test | BUS-BRA case | **gap** |
|---|---|---|---|---|
| `kgsum` | 0 | 0.7531 | 0.6773 | **+0.076** |
| `logit` | 33 | 0.8133 | 0.7356 | **+0.078** |
| `s5` | 65 | 0.8156 | **0.7393** | **+0.076** |
| ResNet-18 | 11.2M | 0.9478 | 0.6605 | **+0.287** |

n = 1846 images / 1058 cases fully probed in both registered arms.

**The registered hypothesis is confirmed.** The zero-shot generalisation gap is
+0.076; the supervised one is +0.287, 3.8× larger. The ordering *reverses* across
the domain boundary: ResNet leads by 0.13 AUC in domain and trails by 0.08
outside it. A model that never saw a BUS-BRA label beats one trained on
BreastMNIST, on BUS-BRA.

The ResNet retrain reproduced its published baseline first (0.9478 vs 0.9442), so
the collapse is a transfer result and not a training failure. The PAD=1.5
sensitivity arm agrees (ResNet case AUC 0.6394), as does BrEaST (0.6490).

## Predictions, scored honestly

| quantity | predicted | actual | verdict |
|---|---|---|---|
| probe external AUC (per-case) | 0.75 – 0.82 | 0.7393 | **missed low**, by 0.011 |
| ResNet-18 external AUC | 0.80 – 0.90 | 0.6605 | **missed badly**, by 0.139 |
| `kgsum` external AUC | 0.68 – 0.78 | 0.6773 | **missed low**, by 0.003 |
| S5 replicates at P≥0.95 | ~0.4 probability | not confirmed | as expected |

Three of four predictions were too optimistic, the supervised one grossly so. The
kill criterion (`kgsum` below 0.65) was **not** triggered — 0.6773.

## E3 — S5, and a distinction worth keeping

| arm | BUS-BRA case AUC | vs S5 | P(better) |
|---|---|---|---|
| KG-signed Laplacian, λ=0.01 | 0.7393 | — | — |
| λ = 0 (no penalty at all) | 0.7380 | +0.0013 | 0.702 |
| shuffled KG, λ=0.01, 5 draws | 0.7219 ± 0.0112 | +0.0174 | **0.992** |

**Not confirmed** under the registered bar, which required beating both. But the
two comparisons say different things and the difference is the interesting part:
against a *wrong* graph the real ontology wins at P=0.992 — the first
ontology-topology effect in this project to clear 0.95 — while against *no graph*
it wins nothing. The KG is not a useful prior here, but a false KG is an actively
harmful one. The 0.878 P(better) seen at n=156 did not become a real effect at
n=1846; it became a precise zero.

## E4 — agreement with radiologist BI-RADS (BUS-BRA)

| model | Spearman ρ | AUC(BI-RADS 4/5 vs 2/3) |
|---|---|---|
| `kgsum` | +0.161 | 0.598 |
| `logit` | +0.314 | 0.667 |
| `s5` | +0.312 | 0.668 |

## E2 — probe validity against radiologists (BrEaST, n=252, complete)

Per-descriptor perception AUC averages **0.5698**: 8 of 13 above 0.60, but **5 of
13 below chance**, and the failures are not noise — `oval_shape` 0.2745 and
`irregular_shape` 0.3352 are *systematically inverted*, which is a sign error
rather than blindness. The strongest are `spiculated_or_irregular_mass` 0.7769,
`posterior_shadowing` 0.7457, `hyperechoic_mass` 0.7417, `circumscribed_margin`
0.7375.

Decomposition through the identical 0-parameter KG-signed sum:

| | AUC |
|---|---|
| probes → label | 0.7020 |
| **radiologist descriptors → label** | **0.8515** |
| perception error (ceiling − probes) | 0.1495 |
| inference error (1.0 − ceiling) | 0.1485 |

The ontology's stance mapping, given perfect inputs and fitting nothing, reaches
0.8515. So the inference rule is sound and the remaining gap is split almost
exactly evenly between seeing and reasoning — and the perception half is
concentrated in a handful of invertible probes.
