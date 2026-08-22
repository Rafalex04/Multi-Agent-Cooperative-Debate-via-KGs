# ThothKG v2 — the candidate sweep

Executed the build spec in its stated order, with its protocol and its kill
criteria. One candidate survives.

## Result

**Best: S5, the KG-signed calibration Laplacian on probe-64 — test AUC 0.8206,
bAcc 0.7769.** Previous standing result was 0.8137 / 0.7826.

The comparison that matters is the matched one: every arm below goes through the
identical `fit_calib` path and the ONLY thing that changes is the penalty matrix.

    arm                      cv        TEST      bAcc
    lambda = 0 (no KG)     0.8255     0.8139    0.7462
    KG signed              0.8288     0.8206    0.7769
    KG unsigned            0.8280     0.8200    0.7607
    shuffled KG (5 draws)  0.8298     0.8117    0.7486
                                     +-0.0063  +-0.0187

    KG signed vs lambda=0 : TEST +0.0067   bAcc +0.0307
    KG signed vs shuffled : TEST +0.0089 (+1.4 sd)  bAcc +0.0283 (+1.5 sd)
    bootstrap vs lambda=0 : +0.0067  95% CI [-0.0045, +0.0188]  P(better) 0.878

**This does not clear the spec's own freeze bar of P >= 0.95.** It is the
strongest KG-structural result the project has produced and it is still only
suggestive. Two honest caveats: CV actually prefers the shuffled arms (0.8298 vs
0.8288), so the test-set win is not something model selection would have picked;
and 1.4 sd is roughly p = 0.08 one-sided.

Why this one worked where the earlier Laplacian failed: the earlier test
penalised the READOUT weights, and mixing those destroys per-finding
discrimination, so CV set lambda to 0. Here the penalty is on the per-finding
CALIBRATION only (a scale and bias per finding). Features are never mixed;
KG-adjacent same-stance findings only borrow strength for estimating their own
bias and scale. CV selects lambda = 0.01, not 0.

## S0 — the new anchor

Four phrasings, a 2x2 of polarity (present / free-of) crossed with framing
(observe yourself / adjudicate another reader's report). Not four hand-written
prompts: both axes are properties of how a question is asked, so they transfer.

    P1 direct/positive    TEST 0.7899   bAcc 0.7801
    P2 direct/negative    TEST 0.8089   bAcc 0.7456
    P3 verify/positive    TEST 0.7822   bAcc 0.7356
    P4 verify/negative    TEST 0.8001   bAcc 0.6974

    probe-32 (P1+P2)      TEST 0.8083   bAcc 0.7813   cv 0.8203
    probe-48 (P1..P3)     TEST 0.8150   bAcc 0.7813   cv 0.8293
    probe-64 (all four)   TEST 0.8154   bAcc 0.7538   cv 0.8316   <- CV picks this

Pooling four phrasings is worth +0.007 AUC over two. Real but small, and it
costs bAcc; the third phrasing captures most of it.

## Candidate outcomes

| # | Candidate | Outcome |
|---|-----------|---------|
| S0 | More phrasings | **Kept.** +0.007 AUC, CV-confirmed |
| G0 | KG-masked probe interactions | Signal on probe-32 (CV +0.0067, test 0.8329) that **vanished** on probe-64 (test 0.8056, P(better) 0.330) |
| S1/S6 | Lesion-layer network | Correctness gates pass (gradient 1.3e-10; S1 vs S6 forward 2.2e-16). CV +0.013 over anchor, **test flat** (0.8113 vs 0.8114). gamma negative — benign-pattern matches pushing the score toward malignant, which is backwards |
| S1b | Ontology-masked conjunctions | **Killed by its own control.** KG mask vs 5 shuffled masks of identical size: cv +0.0064, test +0.0025 on probe-32; on probe-64 test −0.0048. The gain came from *having* 72 product features, not from *which* 72 |
| S2 | Debate-probe fusion at finding level | **Killed.** Adding debate + cross-run columns drops CV 0.8156 -> 0.8026 |
| S4 | Cross-run consensus graph | **Not built** — conditional on S2 |
| S3 | Uncertainty-gated debate expert | Diagnostic fired but with the sign **reversed**: the debate tracks the probe residual where probes are most CONFIDENT (+0.260) and not at all where they are least confident (−0.015). Built with a free-sign gate anyway; delta +0.0002. Killed |
| S5 | Calibration-layer Laplacian | **Survives.** The only arm where CV selects a non-zero KG penalty |
| S7 | Ranking loss | As a standalone: CV up (0.8203 vs 0.8156), test down (0.8083 vs 0.8114) — it overfits the selection criterion. But it *is* the CV-selected loss inside most winning configurations |

## The structural finding about this ontology

S1 was the highest-prior candidate and it exposed something worth reporting.
Building the lesion layer required a class per lesion, and the graph supplies it
for only 16 of 27 candidates — **15 benign, 1 malignant**.

The reason is systematic: the KG catalogues benign entities *by appearance*
(113 `ultrasound_appearance_includes` triples) and describes malignancy through
*stance* triples instead. `breast_cancer` has one appearance triple,
`hyperechoic_rarely`. The four findings with no lesion at all —
`architectural_distortion`, `spiculated`, `spiculated_or_irregular_mass`,
`clustered_microcysts` — are precisely the malignant-stance ones.

So a differential-diagnosis layer over this ontology is not a differential: it is
a one-sided benign-pattern detector. That is a property of the knowledge graph,
not a modelling choice, and it caps what any lesion-layer architecture can do
here.

A note on classification hygiene: a keyword scan gets this badly wrong.
`fibroadenoma mimics phyllodes_tumour` makes a benign lesion read as malignant.
Class must come only from relations that classify (`is_a`,
`typically_classified_as`, `likelihood_of_malignancy`,
`associated_with_malignancy_predictor`), never from `mimics` or
`differential_for`, which say what a lesion can be CONFUSED WITH.

## Data integrity

Two records were lost to interleaved concurrent appends during a shard handover
and were detected as unparseable JSON, not silently absorbed. The files were
repaired, the 26 affected samples re-run, and the loader now skips a corrupt
line rather than aborting an evaluation or coercing a bad record.

## Files

    run_probes.py     four factorial phrasings (--phrasing 1..4)
    kg_structure.py   findings / lesions / classes contract
    gates.py          G0 interaction gate, S7 ranking loss, S3 diagnostic
    s1_lesion.py      S1 lesion-layer net + S6 independent implementation gate
    s1b_masked.py     ontology-masked conjunctions + shuffled-mask control
    s2_gate.py        S2 permutation kill gate
    s3_s5.py          S3 gated expert, S5 calibration Laplacian
    s5_control.py     S5 decided under a matched code path
    final_s0.py       S0 pooling result and re-anchored sweep
