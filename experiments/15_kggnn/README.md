# 15 — KG finding graph, visual probes, and the 0.80 result

## Headline

**Test AUC 0.8137, balanced accuracy 0.7826.** Previous project best was
0.7646 / 0.7036. Paired bootstrap against it: +0.0491, 95% CI
[-0.0114, +0.1102], **P(better) = 0.945** — by some way the strongest
improvement measured in this project.

The method is a logistic model on 32 features: 16 BI-RADS findings × 2 question
phrasings. **It does not use a GNN.**

## What produced the gain

### 1. Re-measuring the probes with the good backbone

The probe vectors in `dataset_v2` are noise — per-finding test AUCs run 0.35 to
0.56. Re-measured with `qwen3-vl:8b-instruct`, reading p(yes) off the first-token
distribution (the trick that worked for BI-RADS), they carry real signal:

    posterior_shadowing      0.766      thin_uniform_pseudocapsule  0.266
    spiculated_or_irregular  0.766      circumscribed_margin        0.280
    irregular_shape          0.755      oval_shape                  0.323

The benign findings invert correctly, and the KG's stance signs match the
measured directions without being fitted to anything.

### 2. The negation arm — the single largest jump

A yes/no probe is exposed to acquiescence bias, which inflates every finding
equally and survives any per-finding weighting. Asking the same question in the
negative ("is the mass FREE of X?") and inverting gives a second view in which
the bias points the *other* way. Measured:

    mean p(present)   positive phrasing 0.346   negated 0.608
    correlation between the two phrasings: +0.100

Nearly independent views of the same quantity.

    probe features            TEST      bAcc     cv
    positive only            0.7909    0.7713   0.8022
    negated only             0.8097    0.7757   0.8114
    mean of both             0.8122    0.7625   0.8124
    both as 32 features      0.8137    0.7826   0.8142   <-- CV picks this
    KG-signed sum, 0 params  0.7834    0.7588

CV selects the winning arm, so this is not test-set selection.

## Where the knowledge graph helps, measured

1. **It defines the probe set.** The 16 findings and their visual descriptions
   are read from the graph; without it there is nothing to probe.
2. **Its stance priors work unsupervised.** `sum_f stance(f) * p(f)` reaches
   **0.7834 / 0.7588 with zero fitted parameters**.
3. **Its edges are real but do not pay.** Lesion-mediated adjacency connects 72
   of 120 finding pairs and the strongest links are clinically correct
   (echogenic pseudocapsule with echogenic rind, circumscribed with oval,
   anechoic with thin uniform capsule). It is genuine structure. It just does
   not improve accuracy.

## The GNN: built properly, and it does not win

`thoth_kg.py` is the Stage 4 architecture re-derived with the KG as the node and
edge set — sign-separated message passing over A+/A-, bipolar attention or sum
readout, anchored score with gamma init 0. The analytic backward pass is
verified against finite differences (`check_grad`, max error 1e-10, both
readouts).

    configuration                          TEST      bAcc
    flat probe-32 model (no GNN)          0.8137    0.7826
    GNN anchored on probe-32, signed KG   0.8135    0.7826   gamma -0.041
    GNN sum readout, no anchor, signed    0.7947    0.7332
    GNN sum readout, no anchor, shuffled  0.7969    0.7196
    GNN attention, no anchor, signed      0.7755    0.7299
    GNN attention, no anchor, shuffled    0.7728    0.7005
    GNN on KG-prior anchor, signed        0.7863    0.7599
    GNN on KG-prior anchor, shuffled      0.7865    0.7588

Three honest observations:

- Anchored on the strong readout, gamma goes to -0.04 and the model reports the
  anchor back. The graph adds nothing on top of a good score.
- Unanchored, the GNN reaches 0.79 but never the flat model's 0.8137.
- **The KG-vs-shuffled comparison does not replicate across readouts.** Signed KG
  beats shuffled with attention readout (+0.0027 AUC, +0.029 bAcc) and loses to
  it with sum readout (-0.0022). An effect that flips sign with an unrelated
  design choice is not an effect.

### The Laplacian attempt

Propagation mixes features and destroys per-finding discrimination, so the last
idea was to constrain the *weights* instead: penalise `w' L_signed w`, pulling
KG-linked same-stance findings toward similar weights. `kg_laplacian.py` sweeps
lambda under the same CV protocol as plain L2.

**Cross-validation sets lambda to 0.** The KG penalty is not merely unhelpful; the
model selection procedure rejects it in favour of plain L2, with the identical
0.8137 result.

## Verdict

The 0.80 target is met on AUC (0.8137) and missed on balanced accuracy (0.7826).
It was met by measuring the KG's findings better, not by a graph network. Four
independent routes now say a learned graph model over this debate adds nothing:
the Stage 4 ablation, the zero-evidence dominant relation, the interaction gate
with its positive control, and — here — propagation, weight regularisation, and
the KG-vs-shuffled control all coming out flat.

## Files

    run_probes.py     probe runner, positive and --negate phrasings
    features.py       per-finding node features: probe, debate, KG prior
    kg_graph.py       lesion-mediated finding adjacency
    thoth_kg.py       the GNN; analytic gradients + finite-difference check
    kg_gnn.py         SGC propagation ablation
    probe_gnn.py      sign-separated propagation on probe features
    final_model.py    full model, anchored and unanchored
    kgprior_gnn.py    GNN on the zero-parameter KG-prior anchor
    kg_laplacian.py   KG as a weight prior instead of a propagation path
