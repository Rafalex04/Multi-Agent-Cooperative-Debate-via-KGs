# Pre-registration — does the mask-treatment gain replicate on BUS-BRA?

Written 2026-08-27, before any BUS-BRA mask probe has been run.
**This file is not to be edited after it is committed.** Outcomes go in a
separate `OUTCOMES.md`.

## What is being tested

On BrEaST (252 images) treating the segmentation as a *mask* rather than only as
a source of a bounding box produced the largest perception gain measured in this
project:

| BrEaST, verification phrasing | baseline | maskdim | maskring |
|---|---|---|---|
| mean descriptor agreement | 0.6531 | 0.7025 | 0.7187 |
| malignancy, zero-parameter KG sum | 0.7153 | 0.7889 | 0.7652 |

252 images is small and BrEaST is a single-centre set. This run asks whether the
effect survives 1875 images / 1064 patients across four scanners.

## Arms

Three, framing held at the registered 2.0× lesion-box window for all three, so
the mask treatment is the only variable.

| arm | treatment |
|---|---|
| `bbase` | 2.0× box crop, no mask applied |
| `bdim` | pixels outside the lesion attenuated to 35% |
| `bring` | lesion boundary drawn as a 2px contour, every pixel kept |

Attenuated rather than zeroed because a hard black surround is a synthetic
feature present in no real ultrasound.

All three arms are generated **in this run**, on the same nodes in the same
window, rather than reusing the archived `busbra_p3` probes. Reusing them would
confound the mask treatment with whatever drifted between runs. The archived
arm is reported separately as a run-to-run stability check, never as the
baseline.

## Probe protocol

Verification phrasing only for all primary analysis. The negated arm is
established anti-predictive (0.4882 descriptor agreement — chance; 0.4452
external malignancy AUC) and is not run as a primary. Same 16 findings, same
wording, same first-token logprob readout `p_yes = P(yes)/(P(yes)+P(no))` over
top-20 logprobs, `num_predict=3`. Nothing else changes.

## Primary endpoint

Zero-parameter KG-signed sum. Per-finding probes standardised, multiplied by the
ontology's ±1 stance, summed. No weights are fitted anywhere.

- case-level AUC, aggregating image scores to the patient by mean
- 1064 patients, folds and aggregation at patient level throughout
- paired bootstrap over **patients**, 4000 resamples, each mask arm against
  `bbase`
- **decision threshold P(better) ≥ 0.95**, the same bar used for the
  KG-topology endpoint

Balanced accuracy reported beside AUC for every arm, with n and bootstrap CIs.
Per-finding AUC reported for all 16 findings in all three arms.

## Exclusions

All 1875 rows carry a mask under `BUSBRA/Masks/mask_*.png`. A 60-image alignment
check gave mean IoU 0.957 against the shipped BBOX (min 0.875), no shape
mismatches, no empty masks. Any image whose mask is missing, shape-mismatched or
empty is excluded **identically from all three arms** and logged. If exclusions
exceed 2% the primary is reported on the intersection and the exclusion list is
published.

## Prediction

Stated before running, so the result can be read against it.

The BrEaST effect should attenuate on a larger, four-scanner corpus but remain
positive. I expect **maskring +0.02 to +0.05** case AUC over `bbase` and
**maskdim +0.02 to +0.06**, landing both arms roughly in 0.77–0.80 against an
archived baseline of 0.7499. At 1064 patients a paired effect of +0.03 should
clear P ≥ 0.95 comfortably and +0.015 would be marginal, so I predict **maskring
clears the bar** and maskdim probably does.

The way this fails: BrEaST's masks were drawn by the same radiologists who wrote
its descriptors, so the descriptor-agreement half of the BrEaST result could be
partly circular. The malignancy half cannot be — BrEaST labels are biopsy-proven,
as are BUS-BRA's — which is why the primary here is malignancy and why a null
would be interpretable rather than confusing.

## Carried-over confirmations

**1. Halo group.** `echogenic_rind`, `echogenic_pseudocapsule`,
`thin_uniform_pseudocapsule` improved under maskring and not under maskdim on
BrEaST, as predicted in advance. **This is only partially testable here**:
BUS-BRA has no descriptor annotations, so the three can be compared on
per-finding probe→malignancy AUC but not on whether the probe sees the finding.
That is a weaker test and will be labelled as such. Predicted direction: the
three improve under `bring` and not under `bdim`.

**2. Negation lockstep.** On a subsample the negated arm is run under `bring`.
The mechanism — the model perceives correctly and then applies the negation
backwards — predicts that a clearer image moves the two arms in *opposite*
directions on the same findings. On BrEaST: `irregular_shape` verification
0.7490→0.8925 while negated 0.1902→0.0941. Predicted here: paired per-finding
movement of opposite sign, and the negated arm's overall zero-parameter AUC
falling **below** its 0.4452 baseline.

## Reporting constraints

- Both mask arms consume a radiologist segmentation at test time. This is
  annotation the pipeline does not produce. Every headline figure is labelled
  **mask-assisted**, and the fully-automatic operating point — 0.7499,
  verification arm through the KG-signed sum, no mask — is reported beside it as
  a separate number. The two are never merged into one "best result".
- Every external figure states its protocol. Frozen-on-BreastMNIST transfer and
  within-BUS-BRA cross-validation rank arms differently and are not
  interchangeable. This experiment fits nothing, so it is neither: it is a
  zero-parameter rule applied directly, and will be described that way.
- If the effect does not clear P ≥ 0.95 it is reported plainly as such and the
  BrEaST result is described as unreplicated at scale. No variant will be
  searched for that clears it.

## Scope

This is the final experiment of the project. No follow-up runs.
