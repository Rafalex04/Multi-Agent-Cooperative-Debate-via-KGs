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

---

# Stage 4 "make the GNN work" plan — executed, with the gate result

## §0 The interaction gate — the answer is NO, and the test has power

Per-finding counts vs the same counts plus all 496 pairwise products, identical
5-fold CV and L2 sweep, on the 546 training graphs.

    corpus     base CV    interaction CV    delta
    1 run       0.5996        0.5754       -0.0242
    4 runs      0.7309        0.7171       -0.0138

Interaction is worse at **every** regularisation strength on both corpora.

Two fixes were needed before this meant anything. Raw counts cannot express
mal_share, which is a ratio, so the unnormalised base scored 0.5811 against
mal_share's own 0.6476 — handicapping the base would have rigged the gate toward
finding interactions. And at l2 >= 10 the optimiser diverged, printing AUC
0.0000 rows that were overflow rather than heavy regularisation.

**Positive control.** A null is only worth reporting if the protocol can detect a
real effect. Against a synthetic XOR label (pure interaction, zero main effect):

    counts CV 0.5112    interaction CV 0.7039    delta +0.1926

The protocol finds a genuine interaction at +0.19 and finds nothing in the real
labels. **The null is real, not a power failure.**

Conclusion: the label is an additive function of independent claim evidence. No
GNN over these graphs will beat the weighted count.

## §1 Ranking loss — no gain

Pairwise AUC surrogate against cross-entropy, same model, features, anchor and
sweep, changing only the loss.

    anchor      CE      rank     CE+aug   rank+aug
    evidence   0.7308  0.7310    0.7391    0.7320
    mal_share  0.7368  0.7316    0.7345    0.7203
    none       0.7143  0.7235    0.7348    0.7203

Mixed and inside noise. Note the failure mode: with augmentation the ranking loss
reaches HIGHER CV (0.7533 vs 0.7400) and LOWER test — it overfits the selection
criterion. The supervision argument (58,000 pairs from 546 graphs) is sound in
principle and simply does not bind here.

## §4.1 Cross-run reproducibility — the one gate the plan passed, and it still
did not survive

The §0 gate pools across runs, which destroys exactly the quantity §4.1 is about:
a pooled count cannot tell one run flagging a finding twice from two independent
runs flagging it once. So it was gated separately, with per-finding "fraction of
runs mentioning it" and "spread of its stance across runs".

    base CV 0.7309    +crossrun CV 0.7386    delta +0.0076   -> passes

But end-to-end on the anchor:

    anchor alone        TEST 0.7556   bAcc 0.6930
    anchor + crossrun   TEST 0.7536   bAcc 0.6930
    anchor + base       TEST 0.7609   bAcc 0.7049
    anchor + both       TEST 0.7561   bAcc 0.6911

**+0.0076 on CV became -0.0020 on test.** Consistent with this project's repeated
pattern of CV and val gains not replicating on 156 test samples.

## Verdict

The plan's §8 failure case is what happened, and it is the reportable result:

> debate structure carries no interaction information, and the correct readout
> for an LLM debate is a weighted count.

Four independent routes now say the same thing — the ablation (everything but the
anchor inside seed noise), the mechanism (the dominant relation carries zero
evidence), the interaction gate with a validated positive control, and the
ranking-loss result. Sections 2, 3 and the rest of 4 were not run because §0
conditions them on a gate that failed.

Best figures in the project remain from combining mechanisms, not architectures:

    evidence + BI-RADS (equal)   TEST 0.7646   bAcc 0.7036
    all four (equal)             TEST 0.7642   bAcc 0.7061
    anchor + per-finding base    TEST 0.7609   bAcc 0.7049
