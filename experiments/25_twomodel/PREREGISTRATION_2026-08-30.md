# Pre-registration: heterogeneous-debater replication of every debate architecture

Written 2026-08-30, before any debate in this experiment is generated.
Fixed in git before the first run. Nothing below is edited after data exists.

## Question

Every debate architecture in this project (v2, v3, v4, v5) was run with **one
model playing both agents**. All four runners take a single `--model` flag and
build a single client; in v5 the two agents differ only in the order their
findings are listed (`agent_offset` 0/1 in `shuffled()`).

Each architecture failed, and each failure was diagnosed as a *diversity*
failure:

| arch | diagnosed failure | same-model explanation |
|---|---|---|
| v2 | both agents cite the same saturated features | one model, one saliency prior |
| v3 | boilerplate collapse (743 of 6249 claims identical) | one model, one phrasing prior |
| v4 | benign advocate never once conceded in 1064 debates | a model cannot authentically disagree with itself |
| v5 | agents split on 44.4% of cases, worth 0.0000 AUC | disagreement is sampling noise, not evidence |

Every one of those diagnoses is consistent with the debate carrying no
information *because both debaters are the same function*. That confound has
never been removed. This experiment removes it.

## H1 (primary)

Replacing agent_2 with a **different model family** makes the debate channel
carry information it does not carry when both agents are the same model.

## Partner selection - decided BEFORE the debates, on data that is not the test set

Candidates already pulled and passing the logprob gate (real inference on a real
image; `/api/tags` is not a health check):
`gemma3:12b`, `medgemma:4b`, `minicpm-v:8b`, `llama3.2-vision:11b`.

Selection substrate: **BreastMNIST train, stratified subsample n=250.**
The 156-image test set and BUS-BRA are NOT touched during selection.

Selection statistic: zero-parameter KG-signed-sum AUC (`z_f` standardised per
finding, signed by the KG's own polarity; nothing fitted), verification phrasing
P3 only - the negated arm P2 is established anti-predictive (0.4452 external)
and is excluded here.

Inclusion bar: a partner must reach **AUC >= 0.60** on that subsample. A model
below that bar is not a debater, it is noise injection, and admitting one would
make a null result uninterpretable. If no candidate clears 0.60, H1 is recorded
as **untestable on available models** and no debates are run.

Partner = the highest-scoring candidate that clears the bar. Ties broken by
lower correlation of its per-finding probe vector with qwen3-vl's, since
decorrelation is the mechanism under test.

## Arms

Incumbent agent_1 is `qwen3-vl:8b-instruct` throughout, pinned to its own nodes
so neither model is ever swapped off a GPU.

For each architecture in {v2, v3, v4, v5}:

- **HOM** - `qwen3-vl` as both agents. Re-run, not reused, so both arms share
  code path, seeds, sharding and image pipeline.
- **HET** - `qwen3-vl` as agent_1, the selected partner as agent_2.

Both arms use identical rounds (3), temperature (0.8), claim caps, parser, KG,
and shuffling seeds. The ONLY difference is which weights answer agent_2.

## Primary endpoint

Case-level AUC of **probes + debate channel** minus case-level AUC of
**probes alone**, measured in HET, compared against the same quantity in HOM.

The comparison is a *difference of differences*: the question is not whether the
heterogeneous system scores higher (a better partner model could raise probe
quality on its own) but whether **the debate contributes more when the debaters
differ**. Probe quality is controlled by construction this way.

Test: paired bootstrap over patients/samples, 4000 resamples.
Decision bar: **P(HET debate-contribution > HOM debate-contribution) >= 0.95.**

## Secondary endpoints (reported regardless of the primary)

1. **Concession rate** - fraction of claims with `stance_verdict == AGREE` from
   each agent. v4's diagnosis was that this is 0 for one side. If heterogeneous
   agents concede, the confabulation diagnosis was about self-debate.
2. **Claim uniqueness** - v2 0.640, v3 0.181. Does diversity restore vocabulary?
3. **Disagreement informativeness** - v5 homogeneous: agents split on 44.4% of
   cases, and gating on the split moved AUC by 0.0001. Repeat under HET.
4. **Per-finding probe correlation** between the two models - the mechanism.
5. **Attack-edge AUC** - dead at 0.5017 homogeneous.

## Predictions, stated now

- Primary: **null**. P < 0.95. Five independent mechanisms already say the
  debate substrate carries nothing, and the one previous introduction of a
  genuinely different model (BiomedCLIP as a second scorer, exp 21) *hurt* -
  hybrid 0.7364 vs P3-alone 0.7499 zero-parameter, and P=0.016 against when
  fitted.
- Secondary 1 and 2: **positive**. Heterogeneous agents should concede more and
  repeat less. These are properties of the transcript and do not require the
  debate to be diagnostic.

Predicting a null on the primary and a positive on the secondaries is the
honest position: the transcripts should visibly improve while the label signal
does not. If that is what happens, it is evidence that debate quality and
diagnostic value are separable - which is a reportable thesis result either way.

## Reporting constraints

- Balanced accuracy reported alongside AUC for every arm.
- Midrank (tie-corrected) AUC throughout.
- Both arms reported whatever the outcome; no arm dropped after the fact.
- If the primary fails, it is reported as a failure, not re-cut until it passes.
