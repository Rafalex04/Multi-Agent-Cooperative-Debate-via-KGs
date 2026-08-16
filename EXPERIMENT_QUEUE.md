# Experiment queue

## Running now

**v3 KG-free debate** — `experiments/07_debate_v3/run_debate_v3.py`
Agents see only the image; the KG is attached afterwards by retrieval. Sharded
over gpu21–24, ~393/780 at 12:43, ETA ~13:10.

## Queued (auto-starts when v3 finishes; watcher in scratchpad/queue.sh)

**1. Link + train GNN on v3** — `link_kg.py` then `train_gnn_text.py`
Answers whether removing the KG constraint from the debate produces graphs a
GNN can learn from. The numbers to beat, all on the same 156-sample test split:

| model | AUC | bAcc@val-th |
|---|---|---|
| probe vector MLP, no graph | 0.6985 | 0.6305 |
| v2 hybrid (graph + probes) | 0.6772 | 0.6323 |
| v2 GraphSAGE | 0.6037 | 0.5654 |

**2. Two-run contrastive scoring** — `experiments/08_contrastive/run_contrastive.py`
Score each image against the malignant stance findings and against the benign
ones separately, then subtract. Rationale: every single-run prompt method
applied a constant offset to all images (over-trigger or over-suppress); the
offset is shared between the two contexts, the shift between them is not.
Two variants computed in one pass — `verdict` difference and `match` difference
(the latter reads as a likelihood ratio between the two rule sets).

## Not yet queued — depends on result of (2)

**3. Cross-assigned adversarial debate.**
If the contrast works, apply the same asymmetry to the debate: give the agent
arguing MALIGNANT only the **benign** triples, and the agent arguing BENIGN only
the **malignant** triples. Each side then has to engage with the evidence
against its position rather than cherry-picking support for it, which is the
failure that kept `mal_share` at ~0.49 in both v1 and v2 — the agents argued
their assigned role rather than the image.

Implementation note: this is a small change to `run_debate_v2.py` —
`evidence_table(ev, side)` already splits findings into supporting and
contradicting, so the cross-assignment is a matter of passing the opposite side
and dropping the supporting block.

**4. LLM-generated probe questions.**
The fully automatic probe derivation (`probe_auto.py`) scores 0.5750 against
0.6685 for the hand-written questions, and the diagnosis is question wording:
only 2 of 29 features have `ultrasound_appearance_includes` triples in the KG,
because those 113 triples describe lesion *types*, not *features*. So 27 probes
ask "Does the mass show spiculated margin?" instead of describing what that
looks like. Fix without reintroducing hand curation: one LLM call per feature to
expand its name into a visual description, cached to disk. 29 calls, one time,
transfers to any KG.

Note also that variance filtering does **not** substitute for the manual
exclusion list, contrary to what `probe_auto.py`'s docstring originally claimed.
Unobservable features (`evidence_of_interval_growth` std 0.409, `hard_elasticity`
0.393, `axillary_adenopathy` 0.386) produce *high*-variance hallucination, not
constants. Only label correlation separates them, which means supervision.
