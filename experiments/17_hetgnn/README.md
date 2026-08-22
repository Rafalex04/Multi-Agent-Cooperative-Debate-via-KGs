# Why every graph over the debate has failed — measured, not assumed

Eight routes in this project have now tried to make a network over the debate
structure beat a flat readout, and all eight came back null. This experiment
stops trying architectures and measures the assumption underneath all of them.

## The merged graph, at claim resolution

Every previous version aggregated the debate into nine scalars per finding node
before the network saw it. That destroys the only things a graph knows that a
table does not: who spoke, when, and whether the claim was rebutted. `hetgraph.py`
keeps the real structure — 19.5 claim nodes, 11.9 signed claim→claim edges and
19.5 claim→finding citation edges per sample, joined to the 16-node KG.

## The assumption

Propagation over an attack graph can only help if **being attacked is evidence of
being wrong**. That is a claim about the debate protocol, not a theorem.

## The measurement

Credibility by the h-categoriser of gradual argumentation semantics,
`r_j = (1 + δ·endorse_j) / (1 + Σ_i attack_ij · r_i)`, then asked whether it
predicts which claims are actually correct.

| quantity | value |
|---|---|
| claims pooled | 15,220 |
| claim-level stance correct | **50.5%** |
| attacks received, correct vs wrong claims | 0.533 vs 0.536 (**−0.003**) |
| AUC(credibility → claim is correct), k=1 / 2 / 3 | **0.5017 / 0.4953 / 0.4914** |
| edges joining *opposing* stances | **81.3%** (chance 44.7%) |

## What this says

The debate produces a genuinely well-formed argumentation graph — 81.3% of edges
cross a stance boundary against a 44.7% baseline, so the agents really are
rebutting opposing claims rather than talking past each other. But rebuttal
tracks **stance opposition, not validity**: an attacked claim is no likelier to be
wrong than an unattacked one, to three decimal places.

So credibility propagation is a null *by construction*, and it does not matter
which architecture is placed on top of it. Confirmed end-to-end: propagation is
demonstrably active — it moves 54% of claims off uniform weight, sd 0.25 — and
still leaves the score flat or slightly worse (debate verdict 0.6843 at k=0 vs
0.6816/0.6795/0.6790 at k=1/2/3).

## The part that is not null

Individual claims are coin-flips, yet the pooled debate score reaches 0.6843 and
argument mass read through the KG prior reaches 0.6983. The signal is in the
**distribution** of claims — how much argument a finding attracts, and which way
the mass leans — not in which claims survive rebuttal. That is precisely why
aggregation works and propagation cannot: the informative quantity is a marginal,
and the graph's topology is independent of it.

This is the honest boundary of the debate-graph idea, and it is a result rather
than a failure: the KG side of the merge carries signal (ablation 0.7128 vs
0.4740), the claim→finding citation edges carry signal, and the claim→claim
attack edges carry none.
