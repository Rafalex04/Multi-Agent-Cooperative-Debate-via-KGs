# Pre-registration: two published baselines against ThothGNN v3

Written 2026-08-31, before any baseline run. Committed before data exists.
Nothing below is edited afterwards; deviations are recorded in an ERRATA section
appended at the end, dated.

Both baselines are RE-IMPLEMENTATIONS from the papers' descriptions. Neither
paper's released code is used. Every result is labelled "our implementation of
X" in every table.

- B1 - Catfish Agent (Wang et al., arXiv 2505.21503): multi-agent medical
  debate with an injected dissenting agent. No graph, no GNN.
- B2 - GraphGeo (Zheng et al., arXiv 2511.00908): debate as a heterogeneous
  agent graph, classified by a relation-specific GNN. No knowledge graph.

## Shared protocol - identical for both baselines and for ThothGNN v3

- Backbone `qwen3-vl:8b-instruct`, non-thinking, same pinned tag and
  quantisation as every other run in this project.
- Data: BreastMNIST official splits 546/78/156; BUS-BRA 1875 images / 1064
  patients with patient-level folds. Never re-split.
- Selection: all hyperparameters by 5-fold CV on the 546 BreastMNIST training
  graphs. ONE test evaluation per baseline, after the configuration is frozen
  and logged. No test-informed iteration.
- Seeds: 5 fixed seeds for anything stochastic; mean +- sd reported.
- Metrics: tie-corrected (midrank) AUC and balanced accuracy side by side, with
  bootstrap 95% CI. bAcc thresholds fitted on train+val and applied unchanged to
  test. Paired bootstrap, 4000 resamples, against ThothGNN v3 and against the
  probe-only readout.
- Both image conditions: fully-automatic (no segmentation) and mask-assisted
  (2.0x crop + maskring).
- Budget logged per image: VLM calls, tokens, wall-clock.

## Substitution, forced, recorded before running

The spec names InternVL3.5-8B as the preferred second lineage for B2's 6-agent
corpus. It is NOT OBTAINABLE on this cluster. Nine tags were attempted across
two routes: absent from the ollama library (`internvl3.5:8b`, `internvl3:8b`,
`internvl:8b`, `internvl2:8b`, `internvl3.5`, `OpenGVLab/InternVL3_5-8B`) and
failing on the HuggingFace route (host-redirect bug on `hf.co` -> `huggingface.co`;
`OpenGVLab/InternVL3_5-8B` reported "Repository is not GGUF or is not compatible
with llama.cpp"). A text-only GGUF would carry no vision projector and is
useless as a VLM agent.

Second lineage is therefore `gemma3:12b`, named acceptable in the spec.
`llama3.2-vision:11b` is separately excluded: this ollama build cannot load it
("unknown model architecture: 'mllama'").

Cost consequence, stated now: gemma3:12b runs ~5x slower per call than
qwen3-vl:8b-instruct (Gemma 3's SigLIP tower emits far more image tokens),
measured today at 5.8 s/probe against 0.78 s/probe. The 6-agent corpus is priced
accordingly and this is not a reason to alter the configuration later.

A partner screen run today on a BreastMNIST TRAIN subsample (n=250, never the
test set), through the zero-parameter KG-signed sum with verification phrasing,
gives the competence ordering used to justify this choice:

    minicpm-v:8b          0.7986
    qwen3-vl:8b-instruct  0.7682   (incumbent)
    gemma3:12b            0.7303
    medgemma:4b           0.5974   saturated: mean p 0.9958, sd 0.0011

## B1 - Catfish Agent

Four roles on the same backbone. Agent A and Agent B see the image and the 16
BI-RADS findings, independently shuffled, and report 2-4 findings with
per-finding MALIGNANT/BENIGN labels. The Catfish sees the TRANSCRIPT ONLY, never
the image - the information-asymmetry condition; without it the Catfish is just
a third observer, which is the redundancy that made our own debate channel
harmful. The Moderator sees the full transcript and produces the final claim set.

Rounds: 0 open; 1 mutual rebuttal on the opponent's 3 most recent claims, format
`claim_id | AGREE/DISAGREE | tag | label | reason`; Catfish round if triggered;
2 response to the Catfish; Moderator consolidates.

Complexity-aware trigger:

    fire IF agents A and B agree on overall stance in round 0   (silent agreement)
         OR mean per-finding probe confidence < tau_conf         (ambiguous case)

`tau_conf` by CV on the training split. The trigger rate is logged; a rate above
90% or below 10% means the gate is inert and WILL be reported as such.

Tone calibration, selected by CV and never by test, both reported:
collaborative ("identify the weakest-supported claims and explain what
additional evidence would be needed") vs adversarial ("argue against the
emerging consensus").

Readout: `mal_share` over the final claim set - IDENTICAL to ours, so only the
protocol varies. No learned head on B1; that would confound it with B2.

Ablations: `B1-full`, `B1-no-catfish` (reduces to our open-stance protocol),
`B1-catfish-sees-image` (asymmetry removed), `B1-always-on` (gate removed),
`B1-adversarial` (tone).

Controls, mandatory, neither previously in our ledger: self-consistency (one
agent, 5 independent samples, majority vote per claim) and independent ensemble
(A and B answer separately, `mal_share` averaged, no interaction). If neither B1
nor our own protocol beats both, debate adds nothing beyond sampling, and that
is the finding.

## B2 - GraphGeo

Six agents: 3x `qwen3-vl:8b-instruct` at different sampling seeds and finding
orders, 3x `gemma3:12b`. Agent count is stated in every table; it differs from
our 2-agent protocol and that confound is acknowledged, not hidden. A 6-agent
version of OUR protocol is run so at least one row controls for it.

Edges, per ordered agent pair:

    r_agree     if stance_i == stance_j
    r_conflict  if stance_i != stance_j
    r_transfer  if |conf_i - conf_j| > tau_transfer   (directed, high -> low)

`stance_i` is agent i's overall verdict, `conf_i` its first-token logprob for
that verdict. `tau_transfer` is the only new hyperparameter, by CV on train.
`r_transfer` may coexist with agree or conflict, as in the paper.

Node features `[image_representation ; agent_embedding]` where
image_representation is the P2+P3 probe vector (32-dim) - THE SAME visual input
ThothGNN v3 receives, so the comparison isolates the graph and not the
perception - and agent_embedding is learnable, dim 8, one per agent slot.

Encoder, following their formulation:

    m_agree_ij    = W_agree    . sigma(h_i (*) h_j)
    m_conflict_ij = W_conflict . sigma(h_i -  h_j)
    m_transfer_ij = W_transfer . h_j
    h_i^(l+1) = LayerNorm(ReLU(W_self h_i^(l) + SUM_r MEAN_j m_r_ij))

Two layers, hidden 32, dropout 0.5, DropEdge 0.2, weight decay 5e-4. Same
capacity budget as ThothGNN v3 so capacity is not the confound.

Readout: sum over agent nodes, then MLP to one logit. NO ANCHOR. GraphGeo has no
such term and running it unanchored is the point of the comparison.

Ablations: `B2-full`, `B2-no-transfer`, `B2-no-relation` (one shared weight
matrix), `B2-no-agent-emb`, `B2-mean-readout`, `B2-anchored`.

## Predicted effect sizes, stated now

Context: on BreastMNIST at 780 images the probe-only readout scores AUC 0.8034,
and adding our homogeneous debate channel COSTS 0.0329 (P(helps) = 0.000).
Replacing one agent with a different model family made it slightly worse still
(-0.0380) despite raising claim uniqueness from 0.397 to 0.587 and pushing the
agents to strongly opposed priors (p(claim=MALIGNANT) 0.70 vs 0.24). Five
mechanisms already tested - attack edges 0.5017, contestation +0.0001, argument
mass +0.0005, rounds degrading monotonically, KG topology failing at P=0.873 -
all say the debate substrate carries no label information on this task.

- **B1-full: null against B1-no-catfish**, |delta AUC| < 0.02, P < 0.95. The
  Catfish targets Silent Agreement; our agents already open in disagreement on
  39.7% of cases homogeneously and 47.8% heterogeneously, so the failure mode it
  fixes is largely not our failure mode.
- **B1 trigger rate: high**, >70%, because the silent-agreement clause fires on
  every case where the two agents open on the same stance (~60%). If so the
  complexity gate is close to inert and `B1-always-on` will match `B1-full`.
- **Collaborative beats adversarial**, consistent with our v4 stance-locked
  corpus collapsing to claim uniqueness 0.086.
- **B1 does NOT beat the independent-ensemble control.** This is the prediction
  most likely to embarrass the whole debate line, and it is stated before
  running.
- **B2-full lands in 0.60-0.66**, the band occupied by our NO-ANCHOR (0.6067)
  and best GraphSAGE (0.6501), well below probe-only 0.8034.
- **B2-anchored approaches ThothGNN v3** and the residual gap is small
  (< 0.02). Stated before running, as required: this would mean OUR CONTRIBUTION
  IS THE ANCHOR, not claim-level nodes and not the ontology. We expect that
  outcome and will report it as such.

## Reporting requirements

One table, all rows, with parameters, VLM calls per image, in-domain AUC/bAcc,
external AUC/bAcc, and paired-bootstrap P(better) against ThothGNN v3:
0-parameter KG-signed sum; probe-64 flat readout; ThothGNN v3; B1 Catfish;
B1-no-catfish; self-consistency N=5; independent ensemble; B2 GraphGeo;
B2-anchored; ResNet-18 supervised (ceiling, not comparable).

Both baselines are reported whatever they score, including if either beats us.
No baseline is tuned past its pre-registered configuration to make it lose.
