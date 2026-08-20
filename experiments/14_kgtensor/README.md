# 14 — KG-indexed evidence tensor, and where the score actually improved

Implements the ledger redesign. Order of work followed the plan; two of its
recommendations were falsified by measurement and are corrected here.

## 1. Survival weighting — the plan's guess was backwards

The plan proposed down-weighting attacked claims. Measured, that *hurts*.

    readout                    1 run TEST    4 run TEST
    mal_share (baseline)          0.7128        0.7506
    survival-weighted             0.6699        0.7368
    endorsement-weighted          0.7160        0.7444

The diagnostic says why. Incoming verdicts, P(gold = MAL | stance, incoming):

    stance   attacked   endorsed   none        spread (MAL - BEN)
    MAL        0.275      0.357    0.315
    BEN        0.257      0.168    0.202
    ------------------------------------------------------------
    attacked spread 0.018    endorsed spread 0.189

**Being disagreed with carries nothing (0.018). Being agreed with carries a lot
(0.189).** The plan had the informative direction inverted. But endorsed claims
are only ~8% of the corpus, so endorsement cannot carry a whole readout — it
belongs as a tensor column, which is where it went.

This confirms deleting DISAGREE edges by a second independent route.

## 2. The KG-indexed tensor

F x 6 per sample (F = 16 BI-RADS findings): net, presence, mass, endorsed,
disputed, open. Two bugs in the first implementation, both fixed:

- **Bilinear dead point.** `gamma * (w.x + b)` with both at zero has zero
  gradient in both, so the model returned the anchor with gamma = 0.000 and the
  ablations were meaningless. The scale is now folded into `w`.
- **KG prior as initialisation is untestable.** With an L2 penalty the objective
  is convex, so the optimum is unique and the starting point cannot survive to
  it. The prior is now the *centre* of the penalty, `||w - w_kg||`, which does
  change the solution and can be ablated.

### Does the KG index help? Yes — this is the clearest structural result.

With no anchor at all, so the score is the representation and nothing else:

    TENSOR + prior + aug        0.7348
    TENSOR + KG prior           0.7143
    TENSOR, no KG prior         0.7172
    COLLAPSED (index destroyed) 0.6345      <-- same columns, summed over findings
    ---------------------------------------------------------------
    ThothGNN NO-ANCHOR          0.6067      (pure debate topology)
    GraphSAGE                   0.6493

**Indexing claims by their BI-RADS finding is worth +0.10 AUC over the identical
columns pooled globally, and +0.128 over the entire debate topology.** That is
the KG participating structurally, in a role that was never previously tested.

The KG *stance prior* does not help (0.7143 with, 0.7172 without). Report the
index as the contribution; not the prior.

### But it does not improve the ensemble

Correlation with the evidence-weighted readout is 0.856 — largely redundant —
and train 0.8122 against test 0.7348 shows 96 parameters overfitting 546
samples. Adding it to the winning combination costs AUC (0.7646 -> 0.7613).

## 3. Where the score actually improved

The BI-RADS two-turn run (experiments/13_birads2) is a different mechanism —
logprobs over an assessment category, 2 calls per image, no claims or rounds —
and correlates only **0.435** with the debate readout.

    readout                        train     val    TEST    bAcc
    evidence-weighted (previous)  0.7490  0.7853  0.7556  0.6930
    birads alone                  0.7621  0.8363  0.7598  0.6817
    evidence + birads (equal)     0.7724  0.8505  0.7646  0.7036
    all four (equal)              0.7855  0.8521  0.7642  0.7061

**Best AUC 0.7646, best bAcc 0.7061 — both beat the previous project best.**

### Honesty about the size of that gain

Paired bootstrap, 4000 resamples of the 156-sample test split:

    evidence+birads vs evidence   +0.0090   95% CI [-0.0215, +0.0399]   P=0.705

**Not significant on test alone.** What supports it is that the improvement
replicates in the same direction on all three splits (train +0.023, val +0.065,
test +0.009) and has a mechanism — two weakly-correlated signals of comparable
strength, the same variance-reduction argument that made run-ensembling work.

Claim it as "the point estimate improves on every split, consistent with the
ensembling mechanism, but the test-set difference is inside noise."

## 4. Where the KG is, and where it is proven to help

1. **In the debate prompt** — 0.7128 with, 0.4740 without, on matched graph
   statistics. Ablation-proven, large.
2. **As the BI-RADS ladder** — `birads_scale.py` derives the 7 rungs, their
   bands and their midpoints from the graph's own `likelihood_of_malignancy`
   triples. The birads component does not exist without it.
3. **As the feature index** — +0.10 AUC over the same columns collapsed.

The GNN remains the one place the KG does not help.

## Files

    claims.py       loader; incoming verdicts and the finding index
    survival.py     step 1
    kg_tensor.py    the tensor, the model, the ablations
    combine.py      component combination
    bootstrap.py    paired significance test
