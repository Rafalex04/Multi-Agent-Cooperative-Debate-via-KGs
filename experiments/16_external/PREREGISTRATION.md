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
