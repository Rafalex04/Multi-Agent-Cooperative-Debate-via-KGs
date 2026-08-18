# ThothGNN Stage 4 — Architecture and Implementation Plan

Concrete specification of the GNN readout described earlier: similarity-based
edge magnitudes, dual support/refute embeddings per node, two class nodes with
pooling toward them, MLP scoring, argmax. Grounded in the specific papers that
solved each sub-problem.

---

## 0. Provenance of each design choice

| Component | Source | What is borrowed |
|-----------|--------|------------------|
| Class nodes, bipolar attachment | Gould & Toni 2025, Gradual AA-CBR (arXiv 2505.15742) | Target arguments `T = {(x_δi, c_i)}`, one per class; cases attack opposing targets and support agreeing ones; prediction is `argmax_c σ(target_c)` |
| Base score anchoring | Same, Definition 2 (MLP semantics) | `ψ^(i+1) = φ(φ^{-1}(τ(a)) + ρ^(i+1))` — the base score enters additively in pre-activation, before relational influence |
| Dual support/refute embeddings | SGCN (Derr et al. 2018), SiReN | Separate propagation over positive and negative edges, then attention fusion |
| Relation-typed heterogeneous convolution | ABAGCN/ABAGAT (arXiv 2511.08982) | Dependency graph with heterogeneous edge labels for support/derive/attack, residual heterogeneous conv layers |
| Learnable edge representations across rounds | GraphGeo (arXiv 2511.00908) | Node-level refinement plus edge-level learnable representations modelling pairwise argumentation state |
| Continuity edges | SMAGDi (arXiv 2511.05528) | Edges linking consecutive responses from the same agent, alongside cross-agent influence edges |
| Cross-layer fusion | HINPool (Hong et al., AAAI 2026) | Your own ablation note: cross-layer fusion drives most of the gain (−39% MUTAG, −63% ENZYMES when removed); TAS contributes −1 to −5% |
| Homophilic/heterophilic decoupling | DuoGNN (Mancini & Rekik, GRAIL 2024) | AGREE edges are homophilic, DISAGREE edges heterophilic; decouple the aggregation |

TAS-style top-K selection is deliberately excluded — HINPool's own ablation puts
its contribution at −1 to −5%, and it costs parameters this dataset cannot fund.

---

## 1. Graph specification

Built from the existing debate output. One graph per sample.

### 1.1 Nodes

**Claim nodes** (~19 per graph). Features, 12 dimensions:

| Feature | Dim | Notes |
|---------|-----|-------|
| stance one-hot (MALIGNANT, BENIGN) | 2 | from the per-claim label |
| round index, normalised to [0,1] | 1 | 0 = opening, 1 = final rebuttal |
| agent one-hot | 2 | |
| in-degree AGREE, in-degree DISAGREE | 2 | how much this claim was backed / disputed |
| out-degree AGREE, out-degree DISAGREE | 2 | how much this claim engaged |
| mean cosine over its CITES edges | 1 | how well grounded in the KG |
| n distinct triples cited | 1 | |
| is_repeated flag | 1 | repeated sentences are kept and flagged |

**Triple nodes** (one per triple cited anywhere in this graph). Features:

| Feature | Dim |
|---------|-----|
| stance sign (+1 malignant-suggesting, −1 benign-suggesting, 0 definitional) | 1 |
| relation-type embedding, learned | 8 |
| corpus-wide citation frequency, standardised | 1 |

Relation type gets a learned 8-dim embedding rather than a 30-dim one-hot; 30
one-hot columns on 546 samples is wasteful.

**Two class nodes**, `c ∈ {BENIGN, MALIGNANT}`, with learned embeddings `e_c`.

### 1.2 Edges

| Type | From → To | Sign | Magnitude |
|------|-----------|------|-----------|
| `AGREE` | claim → claim | + | learned (Section 2) |
| `DISAGREE` | claim → claim | − | learned (Section 2) |
| `CITES` | claim → triple | 0 | retrieval cosine, fixed |
| `CITED_BY` | triple → claim | 0 | retrieval cosine, fixed |
| `NEXT_TURN` | claim → claim | 0 | 1.0 |
| `SUPPORTS_CLASS` | claim → class node | + | attention (Section 4) |
| `ATTACKS_CLASS` | claim → class node | − | attention (Section 4) |

`NEXT_TURN` links consecutive claims by the same agent — SMAGDi's continuity
edge. One edge type, trivially ablatable.

`CITED_BY` exists so KG content flows into claim representations. Without it,
triple nodes are sinks and contribute nothing.

---

## 2. Edge magnitude recomputation

Claim→claim edges arrive as bare signs from the AGREE/DISAGREE verdicts, with no
magnitude. This is the gap your original design identified.

**Do not use text cosine similarity.** A claim and its direct refutation are
lexically near-identical — same anatomy, same descriptors, opposite conclusion —
so cosine would assign the strongest attacks the same magnitude as the strongest
supports.

**Use an edge-conditioned MLP** (GraphGeo's edge-level learnable representation):

```
m_ji = sigmoid( MLP_edge([ h_j^(l), h_i^(l), type_ji ]) )     in [0, 1]
```

recomputed at each layer, so magnitude reflects the current node states rather
than raw features.

**Hard invariant, assert in code:** sign comes from the debate and is never
altered by any learned quantity. `m_ji` scales magnitude only. If the debate said
DISAGREE, it stays an attack edge at every layer.

Ablation `EDGE-FIXED`: set all `m_ji = 1.0`. If the learned magnitudes don't beat
this, drop `MLP_edge` and its parameters.

---

## 3. Encoder: dual support/refute embeddings

Your design, made concrete. Per layer `l`, for each claim node `i`:

```
h_sup = MEAN_{j ∈ AGREE(i)}     m_ji · W_sup  h_j^(l)
h_con = MEAN_{j ∈ DISAGREE(i)}  m_ji · W_con  h_j^(l)
h_kg  = SUM_{t ∈ CITES(i)}      cos_it · W_kg h_t^(l)   / SUM cos_it
h_nxt = W_nxt h_{prev turn}^(l)
```

Separate weight matrices per relation is R-GCN/ABAGCN-style relation-specific
aggregation. Empty neighbourhoods produce a zero vector, not a NaN — guard this,
it will happen often for `AGREE` in round 0.

### 3.1 Fusion of the two embeddings

Two options, both from SiReN's attention integration of positive and negative
embeddings:

**FUSE-CONCAT (default):**
```
h_i^(l+1) = LayerNorm( ReLU( W_self h_i^(l) + W_f [h_sup ; h_con ; h_kg ; h_nxt] ) )
```

**FUSE-GATE (ablation):**
```
g = sigmoid( W_g [h_sup ; h_con] )
h_fused = g ⊙ h_sup + (1 − g) ⊙ h_con
h_i^(l+1) = LayerNorm( ReLU( W_self h_i^(l) + W_f [h_fused ; h_kg ; h_nxt] ) )
```

Gating lets the model decide per-node whether support or contestation dominates.
Report both; the gate costs `hidden²` parameters.

### 3.2 Depth and cross-layer fusion

**Two layers.** Debate graphs have small diameter and 546 training samples do not
fund depth.

**Cross-layer fusion is mandatory, not optional.** Your own HINPool notes put it
at the largest single contributor in that architecture's ablation. Concatenate
all layer outputs (jumping-knowledge style):

```
h_i^final = W_jk [ h_i^(0) ; h_i^(1) ; h_i^(2) ]
```

This is the cheapest high-value component in the design. Ablation `NO-XLAYER`
uses `h_i^(2)` only, and is the ablation most likely to show a large drop.

### 3.3 Regularisation

Hidden dim 32. Dropout 0.5 on node features. DropEdge p=0.2 on claim→claim edges
during training. Weight decay 5e-4. These are aggressive on purpose — 546 graphs.

---

## 4. Readout: bipolar class nodes with anchored base score

### 4.1 Attachment

Your original design attached both class nodes to the first claim. Two problems:
the two nodes then receive identical messages and their embeddings converge
regardless of initialisation, and every claim's influence must route through one
node, which is over-squashing plus a dependency on which agent spoke first.

**Bipolar attachment to all claims**, following Gould & Toni's target arguments.
For claim `i` with stance `s_i`:

```
sign(i, c) = +1  if s_i == c
             −1  if s_i != c
              0  if s_i is neutral
```

Now the class nodes receive systematically opposite messages, so they cannot
collapse into each other.

### 4.2 Attention weights

```
a_ic  = MLP_att( [ h_i^final ; e_c ] )                (scalar)
α_ic  = softmax over i of a_ic
z_c   = SUM_i  α_ic · sign(i,c) · h_i^final
```

Because `h_i^final` already encodes how much AGREE and DISAGREE each claim
received, a claim that was successfully disputed contributes less than its raw
stance implies. **This is the entire reason to use a GNN here rather than count
stances**, and Section 6's `UNIFORM-ATT` ablation isolates it.

### 4.3 Anchored base score

In Gradual AA-CBR's MLP semantics, an argument's base score `τ(a)` enters the
pre-activation additively, before relational influence. The class nodes get the
same treatment, with mal_share as their base score:

```
score_MAL − score_BEN  =  β · logit(mal_share)  +  γ · ( w^T z_MAL − w^T z_BEN )
```

`β` initialised to 1, `γ` initialised to 0.

This is not a safety hack — it is the base-score construction from the framework
this readout is derived from. It has three consequences:

- The model starts at exactly mal_share and can only move away if that reduces
  validation loss.
- `γ`, and the AUC it buys, is a direct scalar measurement of how much the graph
  structure holds beyond the stance count. That number is a result, whatever its
  value.
- The uniform-attention, `γ=0` configuration reproduces your existing 0.7128
  baseline inside the same code path, so the comparison is exact rather than
  across implementations.

Apply L2 specifically to `γ`, tuned by cross-validation, and clip
`|γ · (·)|` to 1.0 in logit space so the correction can reorder borderline cases
but not overturn confident ones.

### 4.4 Prediction

```
p_mal = sigmoid( score_MAL − score_BEN )
```

Use the continuous score for AUC. Threshold fitted on train+val only, applied
unchanged to test — same protocol as the existing mal_share results.

---

## 5. Ensemble graphs

You have two runs per sample with run-to-run mal_share correlation of 0.308.
Three ways to use them, ordered by expected value:

**E1 — score averaging.** Run the GNN on each run's graph, average `p_mal`.
Matches the existing ensembling protocol exactly, so the comparison is clean.

**E2 — merged graph.** Union the claims from both runs into one graph. Claims
from different runs get no claim→claim edges between them but share triple
nodes, so the KG becomes the bridge. Roughly 38 claim nodes per graph, which also
doubles the effective structure the GNN sees. This is the more interesting option
and the one that could plausibly beat E1.

**E3 — training-set augmentation.** Treat each run's graph as a separate training
example with the same label. Doubles training data from 546 to 1092, which is the
single largest lever available against the sample-size constraint. Evaluate on
E1-averaged predictions.

Run E3 as the default training regime; it costs nothing and directly addresses
the governing constraint.

---

## 6. Ablations

Each isolates one component. Report all, including failures.

| ID | Change | Question |
|----|--------|----------|
| `BASE` | γ = 0, frozen | Reproduces mal_share = 0.7128 inside this code path |
| `FULL` | everything on | The proposed model |
| `UNIFORM-ATT` | α_ic = 1/N fixed, γ free | Does learned attention beat uniform pooling? |
| `NO-ANCHOR` | β = 0, γ = 1, both frozen | The unanchored model — expected to reproduce the GraphSAGE-class failure |
| `NO-XLAYER` | h^(2) only | Cross-layer fusion contribution |
| `EDGE-FIXED` | m_ji = 1.0 | Learned edge magnitude contribution |
| `NO-SIGN` | AGREE and DISAGREE share one weight matrix | Does edge polarity matter? |
| `NO-KG-NODES` | triple nodes removed | Structural KG contribution, after the prompt-injection route |
| `NO-NEXT` | NEXT_TURN edges removed | Continuity-edge contribution |
| `FUSE-GATE` | gated fusion instead of concat | SiReN-style fusion |
| `BALANCE` | SGCN balance-theory propagation | Does "enemy of my enemy" transfer to debate? |
| `E2` | merged ensemble graph | Does cross-run structure help? |

`BASE` vs `FULL` is the headline. `UNIFORM-ATT` vs `FULL` is the mechanism claim.
`NO-ANCHOR` is what makes the anchoring argument empirical rather than asserted.

---

## 7. Training protocol

- Official splits, 546/78/156. Never re-split.
- **Validation is 78 graphs — too small for epoch selection.** Use 5-fold CV on
  the 546 training graphs to select epochs and hyperparameters, then retrain on
  the full training split and evaluate once on test per configuration.
- Class-weighted cross-entropy; the train split is roughly 2.7:1.
- Adam, lr 1e-3, weight decay 5e-4, batch 32 graphs, max 200 epochs, patience 30
  on mean CV AUC.
- 5 seeds. Report mean ± std. Expect high seed variance; if std exceeds 0.05 AUC,
  say so rather than quoting the mean alone.
- Report through the existing harness so bootstrap CIs and DeLong tests against
  mal_share are directly comparable.

---

## 8. Implementation

- PyTorch Geometric `HeteroData`; `HeteroConv` with a `GraphConv` per relation.
  Do not hand-roll message passing.
- Serialise the graph corpus once to an `InMemoryDataset`. Freeze it before
  training — any change to retrieval, prompts, or agents invalidates every graph
  and forces full regeneration plus retrain.
- Round-trip test: build → serialise → load → assert node counts, per-type edge
  counts, and feature tensor shapes.
- Assert the sign invariant (Section 2) at every forward pass in debug mode.
- Assert `BASE` reproduces the existing mal_share AUC to within floating-point
  tolerance. If it does not, the graph construction differs from the pipeline
  that produced 0.7128 and everything downstream is uninterpretable.
- Seeds fixed for torch, numpy, random; `torch.use_deterministic_algorithms(True)`.

---

## 9. Build order

1. Graph builder + serialisation + round-trip test.
2. `BASE` configuration. **Gate: must reproduce 0.7128.** Do not proceed until it
   does.
3. Encoder (Section 3) with `UNIFORM-ATT` readout. Tests whether message passing
   alone helps before attention is introduced.
4. Attention readout → `FULL`.
5. `E3` training augmentation.
6. Ablation sweep.
7. `E2` merged graphs, if time.

---

## 10. Interpretability output

`α_ic` is the explanation and comes free. Per prediction, emit the top-5 claims
by attention weight with their stance, cited triples, and AGREE/DISAGREE degrees.

Two measurements, not assertions:

**Faithfulness.** Remove the top-weighted claim, re-run, measure the shift in
`p_mal`. Compare against removing a random claim. If they are equal, the
attention weights are decoration.

**Diversity.** Distribution of which claims and triples receive high attention
across the test set. If the same two triples dominate every explanation, the
interpretability claim fails.
