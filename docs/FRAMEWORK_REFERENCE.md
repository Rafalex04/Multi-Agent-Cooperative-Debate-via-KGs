# ThothGNN v3 — formal reference

Written against the code as it stands on 2026-09-08. Every equation below was
read off the implementation, and every number was either recomputed for this
document or quoted from the results file named beside it.

**Five documented figures do not survive that check.** They are corrected in
place and collected in §12. The largest is the parameter count of ordinal head A.

---

## 1. The forward pass in full

### 1.1 Node set and the feature vector `x_f`

The graph has one node per **finding** — an ontology term naming something
visible in the image. `F = 16` on breast (BI-RADS stance findings), `F = 16` on
retina (ICDR), `F = 55` on derma.

Each node carries **four channels**, `d = 4`, built in
[thothgnn3.py:86-100](experiments/17_hetgnn/thothgnn3.py#L86-L100):

```
x_f = [ p_f , m_f , n_f , π_f ]
```

| # | channel | definition | source |
|---|---|---|---|
| 0 | `p_f` | probe: P(finding *f* present) from the VLM's first-token distribution | perception |
| 1 | `m_f` | **argument mass**: how much of the debate cited *f* | debate |
| 2 | `n_f` | **net stance**: which way that argument leaned | debate |
| 3 | `π_f` | **ontology prior**: what the KG says *f* means | knowledge graph |

The two debate channels are formed by pushing claim-level quantities along the
citation edges, with `mask` zeroing the padding claims:

```
m_f = Σ_c  mask[n,c] · Acf[n,c,f]                      (argument mass)
n_f = Σ_c  mask[n,c] · stance[n,c] · Acf[n,c,f]        (net stance)
```

where `stance[n,c] = +1` if claim *c* argued MALIGNANT, `−1` if BENIGN. So `m_f`
counts citations and `n_f` is their signed sum; `m_f = 3, n_f = +1` means three
claims cited *f*, two malignant and one benign.

`π_f` is `+1` if the KG asserts the finding suggests malignancy, `−1` if benign
(breast); on retina it is the ICDR level recentred on the referable-DR boundary
(grade ≥ 2), so its sign means "pushes above / below referable"; on derma it is
the finding→class assignment.

**Probe misses.** If the probe returns `None` the channel defaults to `0.5`
(`np.full(..., 0.5)`, [thothgnn3.py:70](experiments/17_hetgnn/thothgnn3.py#L70)).
Measured miss rates are negligible — see §6.

**Standardisation.** `Z = (X − μ)/σ` with `μ, σ` computed over
`train.reshape(-1, d)` — i.e. **pooled over samples and nodes, train split only**
([run_thothgnn3.py:44-49](experiments/17_hetgnn/run_thothgnn3.py#L44-L49)). One
`μ, σ` per channel, shared by all `F` nodes. Zero-variance channels get `σ = 1`.

> Note the single published exception: the *0-parameter* arm standardises
> differently and not on train. See §6.3.

### 1.2 Message passing — one layer

```
M = X·W₀  +  A⁺·X·W_p  +  A⁻·X·W_n  +  b
H = max(M, 0)                                    (ReLU)
```

In index form, for node *f* and hidden unit *k*:

```
M[n,f,k] = Σ_d x[n,f,d]·W₀[d,k]
         + Σ_g A⁺[f,g] Σ_d x[n,g,d]·W_p[d,k]
         + Σ_g A⁻[f,g] Σ_d x[n,g,d]·W_n[d,k]
         + b[k]
```

Four things are load-bearing here:

- **`W₀` is the self-transform**, applied to the node's own features. `A⁺` and
  `A⁻` have zero diagonal, so `W₀` is the *only* path from `x_f` to `H_f`
  directly; it is also the only weight that is meaningful when `A = 0`.
- **The sign split is two separate weight matrices**, not one matrix over a
  signed adjacency. `A⁺` carries edges between findings the ontology gives the
  **same** stance; `A⁻` between **opposing** stances. Giving them independent
  weights lets the network learn that corroboration and contradiction should be
  read differently — a single signed matrix would force `W_(−) = −W_(+)`.
- **Aggregation is a SUM, not a mean.** Deliberate, and recorded in the module
  docstring: a mean cannot distinguish "one finding argued heavily" from "many
  findings argued lightly", and that distinction is the entire content of the
  citation channel.
- **`A` is not degree-normalised.** `kg_graph.normalised()` implements the GCN
  `D^-1/2 (A+I) D^-1/2` but **v3 does not call it**. The adjacency enters raw,
  scaled only so `max(A) = 1`. There is exactly one nonlinearity in the whole
  network — this ReLU — and one layer, so there is no propagation beyond 1 hop.

> **On the "2 layers" in the sourcebook.** Lines 241 and 837 describe **B2
> GraphGeo** (`LayerNorm(ReLU(·)); 2 layers, hidden 32, dropout 0.5, DropEdge
> 0.2`), not ThothGNN. There is no disagreement about v3: `forward()` computes
> `M` exactly once. **v3 is one layer.**

### 1.3 Readout and loss

**Binary (v3):**

```
s_n = Σ_f Σ_k H[n,f,k]·u[f,k] + c            u ∈ R^(F×k),  c scalar
p_n = σ(s_n)
L   = −(1/N) Σ_n [ y_n log p_n + (1−y_n) log(1−p_n) ]
      + (λ/2)(‖W₀‖² + ‖W_p‖² + ‖W_n‖² + ‖b‖²)
      + (λ/2)‖u − ū‖²
```

`u` is indexed by **finding identity** — one weight vector per node, so the
readout learns "how much does *spiculated margin* matter", which is exactly the
part that cannot transfer to another ontology (§2.3).

**The prior is a penalty centre, never an initialisation.** `ū[:,0] = π`, all
other columns zero. `u` is initialised `N(0, 0.3)` and *pulled toward* the prior
by the L2 term. The docstring gives the reason and it is a real one: under a
convex L2-penalised objective the optimum is unique, so an initialisation cannot
survive to it and the hypothesis would be untestable. `c` is **unregularised**.

**Debate generation constants.** Breast v5q ran at `--temperature 0.8`, passed
explicitly. **Retina and derma did not.** Neither `run_debate_retina.py` nor
`run_debate_derma.py` defines a `--temperature` flag; both call
`cl.call(prompt, b64, num_predict=360)` and so inherit `Client.call`'s default
**`temperature=0.4`**. Rounds *are* 3 everywhere (verified: all 1,600 retina
records carry `rounds_used=3`, single lineage `qwen3-vl:8b-instruct` for both
agents). `num_ctx=4096`, `num_predict=360` throughout.

| corpus | rounds | temperature | verified by |
|---|---:|---:|---|
| breast `debates_v5q` | 3 | **0.8** | `run_debate_v5.py:356` default, passed at :332 |
| retina `debates_r1` | 3 | **0.4** | no flag; `Client.call` default at `run_debate_v5.py:200` |
| derma `debates_d1` | 3 | **0.4** | same |

**Optimiser:** full-batch momentum SGD, `v ← 0.9v + 0.1∇`, `θ ← θ − 0.15v`, 800
epochs, no schedule, no early stopping, no minibatching. Identical in every head.

**Gradient checks** (finite differences, all shipped in-module):
v3 `1.198e-09`, ordinal `check_grad()`, nominal `5.510e-10`. All PASS at `<1e-6`.

### 1.4 Multi-class readouts

**Ordinal head A (CORAL)** — [ordinal.py](retinaMnist/src/retina_kg/ordinal.py):

```
s_n = Σ_f Σ_k H[n,f,k]·u[f,k]                       ← NO bias term
P(y_n > c) = σ(s_n − θ_c),      c = 0 … C−2
L = (1/N) Σ_n Σ_c BCE( σ(s_n − θ_c), 1[y_n > c] )
grade_n = Σ_c 1[ σ(s_n − θ_c) > 0.5 ]
```

**Nominal head (derma)** — [nominal.py](dermaMnist/src/derma_kg/nominal.py):

```
s[n,c] = Σ_f Σ_k H[n,f,k]·u[f,k,c] + c₀[c]          u ∈ R^(F×k×C)
p = softmax(s),  L = weighted cross-entropy
```

`class_weight="balanced"` reweights by inverse frequency — necessary here and
nowhere else, since `nv` is 67% of the derma training set.

---

## 2. Parameter accounting

### 2.1 The counts, recomputed

All at `F = 16`, `d = 4`, `k = 2`, `C = 5` (4 cut points) — the published retina
configuration. Counted by instantiating each `init()` and summing `.size`:

| tensor | shape | v3 | head A | head C |
|---|---|---:|---:|---:|
| `W₀` | (d,k) | 8 | 8 | 4 × 8 |
| `W_p` | (d,k) | 8 | 8 | 4 × 8 |
| `W_n` | (d,k) | 8 | 8 | 4 × 8 |
| `b` | (k,) | 2 | 2 | 4 × 2 |
| `u` | (F,k) | 32 | 32 | 4 × 32 |
| `c` | scalar | 1 | — | 4 × 1 |
| `θ` | (C−1,) | — | 4 | — |
| **total** | | **59** | **62** | **236** |

⚠️ **The reported head C numbers were not produced at k=2.**
`ordinal_full.json` records `selected/C/kdim = 4` (CV 0.87007), so the published
C row (refAUC 0.8959, QWK 0.6962, 39 rank violations) came from **4 × 117 = 468
parameters**, not 236:

```
per head, k=4:  3·(4×4) + 4 + 16×4 + 1  =  48 + 4 + 64 + 1  =  117
head C:         4 × 117                  =  468
```

236 is the k=2 count and corresponds to **no reported C result**. Head A is
unaffected — `selected/A/kdim = 2`, so 62 stands.
[SOURCEBOOK:839](docs/THESIS_SOURCEBOOK.md#L839) is self-contradictory on its own
line: *"236 params (4 independent trunks), l2 0.03, **kdim 4**"*.

### 2.2 ⚠️ Head A is 62, not 63

The docstring reads *"trunk 59 params + 4 = 63"*. That **double-counts v3's
scalar `c`**: 59 already includes it, and head A does not have it — θ replaces
it. The correct arithmetic is `58 + 4 = 62`.

`59` and `236` are both correct. `236 = 4 × 59` exactly, because head C's heads
are *unmodified* `thothgnn3` models, each keeping its own `c`.

### 2.3 The "65" and the 26/39 split — both wrong

This appears in three places
([THESIS_SOURCEBOOK.md:579](docs/THESIS_SOURCEBOOK.md#L579),
[retinaMnist/README.md:229](retinaMnist/README.md#L229),
[transfer.py:6](retinaMnist/src/retina_kg/transfer.py#L6)):

```
W0, Wp, Wn (d x k), b (k)   FEATURE CHANNEL    26 of 65 params  -> can cross
u (F x k), theta            FINDING IDENTITY   39 of 65 params  -> cannot
```

The **split is the right idea and `26` is right**; the totals are not:

| | documented | actual |
|---|---:|---:|
| feature channel `(W₀,W_p,W_n,b)` | 26 | **26** ✓ |
| finding identity `(u,θ)` | 39 | **36** (32 + 4) |
| total | 65 | **62** |

`TRUNK = ("W0","Wp","Wn","b")` in both
[transfer.py:48](retinaMnist/src/retina_kg/transfer.py#L48) and
[transfer_matrix.py:42](shared/transfer_matrix.py#L42), so **the code does the
right thing** — only the prose is wrong. Every transfer number stands.

The distinction is the substantive point: the trunk is indexed by *feature
channel*, and the four channels mean the same thing in all three ontologies, so
it can cross. `u` is indexed by *finding identity* — breast finding 0 is
`architectural_distortion`, retina finding 0 is `microaneurysm` — so it cannot.

### 2.4 Reconciling "no bias"

Head A's forward has **no readout bias**: `s = Σ H·u`, no `+c`. It still has `b`,
the `k`-vector added **inside the ReLU** before the nonlinearity. Two different
biases:

- `b` (pre-activation, 2 params) — **present in every head**;
- `c` (post-readout scalar) — **present in v3 and C, absent from A**.

A drops `c` because θ already carries `C−1` intercepts; keeping both would be
redundant, and — the actual reason — dropping it makes the `C = 2` case reduce
*exactly* to v3, with `c ≡ −θ₀`. `test_reduction.py` asserts this on all three
breast splits at `max|s_v3 − s_A| = 0.000e+00`.

---

## 3. The merge, formally

### 3.1 What a resolved claim is

Agents emit pipe-delimited lines, parsed by `parse_claims`
([run_debate_v5.py:230-275](experiments/10_debate_v5/run_debate_v5.py#L230-L275)):

```
[addressed_id] | [AGREE|DISAGREE] | <finding tag> | <LABEL> | <body text>
```

A line becomes a **resolved claim** only if it survives all of:

1. contains `|`;
2. after stripping the optional `c\d+` target and the optional verdict, has ≥ 3 fields;
3. body has **≥ 4 words** after trailing-keyword stripping;
4. within the per-agent cap of `max_claims = 4` per round.

The resolved record is
`{node_id, text, label, label_explicit, expert_id, round_idx, cited_features,
addressed_ids, stance_verdict, is_repeat}`. Note `label` may be `None` — the
regex is searched, not required — and `hetgraph` keeps such claims (they carry
structure) while `claims.load` drops them.

Findings are presented to each agent under **per-sample shuffled tags** `f1…fN`
(`shuffled()`), so tag→feature is a per-sample mapping and position carries no
information across samples.

### 3.2 Claim → node

Claims are truncated to `CMAX = 20` and zero-padded, with `mask[:n] = 1`. Each
claim node carries **11 features**
([hetgraph.py:44](experiments/17_hetgnn/hetgraph.py#L44)):

```
CLAIM_COLS = (stance, explicit, r0, r1, r2, repeat,
              agent, n_out, in_agree, in_disagree, cites)
```

Every one is a property of the **argument**, not the image — image evidence
enters only through the finding nodes. `in_agree` / `in_disagree` are *incoming*
counts, obtained by scanning who addressed whom; a claim's own `stance_verdict`
says what **it** thought of someone else, which is a different quantity, and
`claims.py`'s docstring records that only the latter had ever been measured
before.

### 3.3 Typed provenance edges

Three edge types, two of which are used by v3:

| edge | tensor | type | used by v3? |
|---|---|---|---|
| claim → claim | `Acc[c,c']` | signed: `+1` AGREE, `−1` DISAGREE, `0` otherwise | **no** |
| claim → finding | `Acf[c,f]` | binary citation | **yes** |
| finding → finding | `A⁺/A⁻` | ontology, sign-split | **yes** |

`Acc` is built but **deliberately never propagated over**. The measurement is in
[17_hetgnn/README.md](experiments/17_hetgnn/README.md): AUC(credibility → claim
is correct) = **0.5017 / 0.4953 / 0.4914** at k = 1/2/3, and attacks received
differ by −0.003 between correct and wrong claims. Propagation over an attack
graph can only help if being attacked is evidence of being wrong, and it is not.
The graph is well-formed — 81.3% of edges cross a stance boundary against a 44.7%
baseline — it just does not track validity.

### 3.4 Attachment

`Acf[c,f] = 1` iff claim *c* cited finding *f*, resolved through the shuffled tag
map. The merge is **structural, not feature-level**: `cited_features` is drawn
from the same `all_findings()` list the KG is indexed by, so the two graphs share
vertices rather than being two models averaged at the end. Measured on
`debates_v5q` (780 samples): 19.5 claims, 11.9 claim→claim edges, 19.5 citations
per sample, no sample with zero edges, and 15,219 of 15,220 claims cite exactly
one finding.

---

## 4. Adjacency constructions

### 4.1 Breast — lesion-mediated Adamic-Adar

Two prior attempts failed and are recorded in
[kg_graph.py](experiments/15_kggnn/kg_graph.py): **direct triples** give zero
edges (the 16 findings are leaf terms), and **1-hop neighbours** connect 94/120
pairs but only through `suggests_malignancy`/`suggests_benign`, which is stance,
not structure — message passing over it is pure label homophily.

The construction used takes lesions as mediators:

```
tok(s)      = { t : t ∈ split(s,"_"), t ∉ STOP, |t| > 3 }
desc(L)     = ⋃ tok(o)  over  L --ultrasound_appearance_includes--> o
                              o --combination_indicates--> L
f2l(i)      = { L : tok(name_i) ∩ desc(L) ≠ ∅ }
deg(L)      = |{ i : L ∈ f2l(i) }|

A[i,j] = Σ_{L ∈ f2l(i) ∩ f2l(j)}  1 / log(1 + 1 + deg(L))          i ≠ j
A      = A / max(A)
```

Note the denominator is `np.log1p(1 + deg)` = `log(2 + deg)`, not the textbook
Adamic-Adar `1/log(deg)` — chosen to stay finite at `deg = 1`.

Matching is **token-overlap, not exact entity match**, because the graph's
appearance vocabulary (`hyperechoic`, `well_circumscribed`) and its finding
vocabulary (`hyperechoic_mass`, `circumscribed_margin`) differ in surface form
while naming the same thing; exact matching gives zero edges.

**Result:** 72 of 120 pairs connected, from 113 appearance + 8 combination
triples. The strongest links are clinically correct (echogenic pseudocapsule ↔
echogenic rind; circumscribed ↔ oval; posterior shadowing ↔ microcalcifications).

**Sign split** ([hetgraph.py:96](experiments/17_hetgnn/hetgraph.py#L96)):

```
same = (π_i · π_j) > 0
A⁺ = A ⊙ same        A⁻ = A ⊙ ¬same
```

Recomputed for this document: **36 edges in `A⁺`, 36 in `A⁻`** (72 total ✓).

### 4.2 Retina — the ICDR mediator set

The breast docstring claims the construction transfers to "any ontology with
appearance triples". **It does not transfer to ICDR**: there are 18
`fundus_appearance_includes` triples and 5 are the placeholder
`no_stated_morphology`, so the direct port gives a near-empty graph. It is kept
as `--mediators appearance` *precisely because* the negative KG-topology result
on breast was measured under that construction, and changing it silently would
not be a replication.

The primary construction keeps the **identical formula** and swaps only what
counts as a mediator:

```
A[i,j] = Σ_{M ∈ med(i) ∩ med(j)}  1 / log(1 + 1 + deg(M))
```

| mediator group | relations |
|---|---|
| `criterion` | `requires_criterion` (+ the `criteria.json` `lesion`/`sibling_criteria` fields) |
| `taxonomy` | `is_a`, `subtype_of`, `component_of` |
| `location` | `located_in`, `often_located_in` |
| `association` | `associated_with`, `confused_with`, `differential_for` |

Mediators are **symmetric**: a shared parent, location or partner joins both
endpoints, so both `s` and `o` get `rel:other` added to their mediator set.
Criteria are special-cased because they are referenced by a *field*, not only by
a relation.

**`level` is deliberately NOT a default mediator** (`establishes_level`,
`minimum_level_for_presence`, `compatible_with_level`), because using severity
level to build the graph and then predicting severity would be circular.

**Results:** retina 21 edges / 120 pairs, 46 mediators, 0 isolated
(`ordinal_full.json`); derma 162 edges / 1485 pairs, 55 findings, 80 mediators,
11 isolated (`derma_nominal.json`).

The sign split is the same rule, with `prior` = ICDR level recentred on the
referable boundary (retina) or class-set overlap (derma).

---

## 5. "Pooled" — averaging *or* concatenation, depending on the arm

**Both, and it is not consistently documented.** In
[phrasing_arms.py](experiments/20_perception/phrasing_arms.py):

- **0-parameter arm → per-finding AVERAGING.**
  `np.mean([X["test"][:, s] for s in sel[lab]], 0)` and `E.mean(2)` — a 16-vector,
  each entry the mean of P2 and P3 for that finding.
- **Fitted-readout arm → CONCATENATION.**
  `cols = [i for s in sel[lab] for i in range(*s.indices(2*F))]` — 32 columns,
  reported as `n_params = 32`.

So the same word denotes different operations in two adjacent tables of the same
results file. Elsewhere:

| consumer | "pooled" means | dimensionality |
|---|---|---|
| `phrasing_arms` zero-param | average | 16 |
| `phrasing_arms` fitted | concatenate | 32 |
| **ThothGNN v3 node tensor** | **concatenate** (`PHRASE_COLS` → separate channels) | `d = 3 + \|PHRASE_COLS\|` |
| B2 GraphGeo node features | **concatenate** (`np.concatenate([p[i] for p in pr])`) | 32 |
| `sign_audit.descriptor_signs` | average (`.mean(2)`) | 16 |

**The published v3 uses neither pooling — it uses P3 alone** (`PHRASE_COLS = (1,)`,
`d = 4`), because P2 is anti-predictive. From `phrasing_arms.json`:

| arm | agrees with radiologist's descriptor | in-domain | BUS-BRA case | gap |
|---|---:|---:|---:|---:|
| P2 negated | **0.4882** (chance), 6/13 inverted | 0.5478 | 0.4452 | +0.1026 |
| **P3 verification** | 0.6531, 3/13 inverted | **0.7467** | **0.7499** | **−0.0032** |
| pooled (average) | — | 0.7270 | 0.6919 | +0.0351 |

Pooling costs **−0.0197 in-domain and −0.0580 externally** against P3 alone:
averaging a good channel against an anti-predictive one is exactly what it does.
The descriptor column is decisive because it **uses no malignancy label at all** —
it compares the probe to the radiologist's own annotation of the descriptor the
probe names — so P3 can be selected without touching the target.
`P(P3 > pooled)` = 0.936 fitted, 0.929 zero-parameter.

---

## 6. Probe implementation

### 6.1 Top-20 miss rate

The call requests `logprobs: True, top_logprobs: 20` with `temperature: 0,
num_ctx: 2048, num_predict: 3`. `p_yes` returns `None` when neither a yes- nor a
no-token appears in the top 20; the node then falls back to `0.5`.

**Measured across every probe file in the project** (counted for this document):

| corpus | records | probe calls | nulls | miss rate |
|---|---:|---:|---:|---:|
| breast P3 `probep3_*` | 780 | 12,480 | 3 | **0.024%** |
| breast P2 `probeneg_*` | 927 | 14,832 | 0 | **0.000%** |
| retina P3 | 1,600 | 25,600 | 2 | **0.008%** |
| derma P3 | 10,015 | 550,825 | 13 | **0.0024%** |

**18 nulls in 603,737 calls (0.003%).** The top-20 window is never the binding
constraint; a yes/no question at temperature 0 puts essentially all first-token
mass on the two surface forms.

The failure that *did* matter was different and is guarded separately: two nodes
returning HTTP 500 wrote **452 rows of all-null probes**, which look like data,
survive resume-by-index, and enter as rows of 0.5. `run_probes.py` now refuses to
write a record with `got == 0` and aborts the node after 5 consecutive ones.

### 6.2 Token surface forms

```python
t = c.get("token", "").strip().lower()
if   t.startswith("yes"): y += exp(logprob)
elif t.startswith("no"):  n += exp(logprob)
return y / (y + n)
```

So: whitespace-stripped, lower-cased, **prefix** match. This catches `Yes`,
` yes`, `yes.`, `YES`, `no`, `No`, `not` — and `y`/`n` alone are **not** counted.
The mass is **renormalised over the yes/no subset only**, so `p_yes` is a
conditional probability given the model answered the question at all; any mass on
other tokens is discarded rather than treated as evidence.

`invert` handles polarity: for P2/P4 a "yes" means the finding is *absent*, so
the stored value is `1 − p_yes`. **Every arm is stored as p(present)**, which is
what makes them poolable at all.

### 6.3 ⚠️ Are `μ_f, σ_f` train-only? For the fitted arms yes; for the headline 0-parameter number, **no**

Three different implementations, and they do not agree:

| implementation | normalisation | train-only? |
|---|---|---|
| `phrasing_arms.py` **fitted** arms | `mu, sd = Xb["train"].mean(0), .std(0)` | ✅ **yes** |
| ThothGNN v3 / all GNN arms | `standardise()` on `X["train"]` | ✅ **yes** |
| `shared/transfer_matrix.zero_param` | uses `Z` from `_std()`, train statistics | ✅ **yes** |
| `retina transfer.zero_param` | `Xraw`, **no standardisation at all** | n/a |
| `phrasing_arms.py` **0-parameter** | `zscore(X["test"])`, `zscore(esel[lab])` | ❌ **no — the evaluation split itself** |

`zscore(P) = (P − P.mean(0)) / P.std(0)` standardises over **whatever matrix it
is handed** ([sign_audit.py:57](experiments/20_perception/sign_audit.py#L57)). In
the zero-parameter arm it is handed the test matrix, and for the external arm the
BUS-BRA matrix.

**This is transductive.** It uses no labels, so it does not leak the target, but
the published `0.7467 / 0.6466` figures use per-feature centring computed on the
evaluation set. Two consequences worth stating in the thesis: the number is not
reproducible on a *single* image (it needs a batch to centre against), and it
quietly adapts to each external cohort's base rates, which flatters the transfer
gap. The GNN arms carry no such caveat.

---

## 7. Perception-layer variants attempted

| # | variant | what changed | results file |
|---|---|---|---|
| 1 | **P1** direct/positive | "Does *X* show *desc*?" | (template only; not run at scale) |
| 2 | **P2** direct/negative | "Is *X* FREE of *desc*?", inverted | `15_kggnn/results/probeneg_*.jsonl` |
| 3 | **P3** verification/positive | peer reports *desc*, "do you agree?" | `15_kggnn/results/probep3_*.jsonl` |
| 4 | **P4** verification/negative | peer reports *not desc*, inverted | (template only; not run at scale) |
| 5 | pooled P2+P3 | average / concatenate | `20_perception/results/phrasing_arms.json` |
| 6 | per-phrasing breakdown | per-finding AUC by arm | `20_perception/results/per_phrasing.json` |
| 7 | **crop pad 1.25 / 2.00 / 3.00** | ROI crop tightness | `20_perception/results/crop_sweep.json`, `pad125_*`, `pad30_*` |
| 8 | **maskdim** | non-lesion pixels dimmed | `20_perception/results/mask_sweep.json`, `maskdim_*` |
| 9 | **maskring** | lesion replaced by a ring | `mask_sweep.json`, `maskring_*` |
| 10 | **BiomedCLIP** | CLIP image-text score replaces the VLM | `21_biomedclip/results/compare_scorers.json`, `clip_*.npz` |
| 11 | **VLM/CLIP hybrid** | per-finding best-scorer selection | `21_biomedclip/results/hybrid.json` |
| 12 | **lesion probes** | ask the differential, not the finding | `18_lesion/results/lesion_*.jsonl`, `lesion_analysis.json` |
| 13 | sign audit / control | polarity fitted on BrEaST descriptors | `20_perception/results/sign_audit.json`, `sign_control.json` |

Pooled-malignancy AUC across the masking arms (`_mal`, n = 252):

```
baseline pad2    P2 0.5806   P3 0.7153
maskdim          P2 0.3914   P3 0.7889
maskring         P2 0.4854   P3 0.7652
```

P3 **improves** when the surround is suppressed (0.7153 → 0.7889) while P2 falls
further below chance — the two arms do not just differ in quality, they respond
to the same intervention in opposite directions.

---

## 8. The lesion channel

### 8.1 Query template

Same mechanism as the finding probes, one yes/no call, `p(yes)` off the first
token ([run_lesion_probes.py](experiments/18_lesion/run_lesion_probes.py)):

```
Look at the image and answer one question about the most likely diagnosis.
Is the lesion in this image {desc}?                        ← positive arm
Can you rule out that the lesion in this image is {desc}?  ← negative arm
```

`{desc}` is `_plain(entity)`: `simple_breast_cyst` → `a simple breast cyst`
(article chosen by first letter).

### 8.2 Derivation of the lesion set

From [lesions.py](experiments/18_lesion/lesions.py), over the 31 named lesions
carrying 113 `ultrasound_appearance_includes` edges:

1. drop appearance-phrase entities via `_NOT_A_DIAGNOSIS` (`hypoechoic|…`);
2. merge near-synonyms via `_ALIAS` (9 rules — probing both would double-weight
   the same question);
3. assign a malignancy prior by priority:
   **(a)** explicit `typically_classified_as birads_N` → ACR band midpoint
   `{1:0.0, 2:0.0, 3:0.01, 4:0.485, 5:0.975}`;
   **(b)** the asserted malignant set `{breast_cancer 0.975,
   metastatic_intramammary_lymph_node 0.975, breast_neoplasms 0.75}`;
   **(c)** `is_a benign_breast_lesion` → 0.0;
   otherwise **drop**.

Step (b) is asserted in code, not derived, and the docstring says why: those two
entities have no `is_a` and no BI-RADS edge — **that absence is itself the
one-sided-ontology finding**. The ontology catalogues benign entities by
appearance and describes malignancy through finding stances, so a lesion probe is
mostly a *benignity* detector and is expected to work by ruling out.

### 8.3 Readout

```
w₀ = π_L − mean(π_L)                 ← centred, so a 15/20-benign channel
s  = Σ_L p(lesion L) · w₀[L]           is not just a constant offset
```

Zero fitted parameters. Bipartite lesion↔finding incidence uses the *same*
token-overlap rule as the finding graph, keeping the two consistent.

### 8.4 Results and the +2.82 sd control

From `18_lesion/results/lesion_analysis.json` (test AUC):

| arm | test AUC | bAcc |
|---|---:|---:|
| KG differential, 0 params | 0.6949 | 0.6385 |
| lesion probes, fitted | 0.7845 | 0.7074 |
| findings only (P2+P3) | 0.8133 | 0.7538 |
| **findings + lesions** | **0.8417** | **0.7619** |

**The control.** The gain is `+0.0284`. Is that the ontology, or just 20 more
columns? The lesion block is **permuted across samples** (20 draws), preserving
its shape and marginals and destroying only the image↔lesion correspondence:

```
real gain        +0.028404
shuffled mean    −0.020760  ± 0.017456
real − shuffled  +0.049164   →   +2.82 sd
```

This is **the one place in the project where a KG-derived structure beats its own
permutation control at over 2 sd.** It is the *lesion* layer, not the
finding-finding topology, and it is worth being precise about that in the thesis:
C3 (KG edges are a null) and this result are not in conflict — they concern
different slices of the ontology.

### 8.5 BiomedCLIP deflation

From `21_biomedclip/results/hybrid.json`, zero-parameter KG-signed sum,
evaluated **case-level on the external BUS-BRA cohort** (the same evaluation as
the `ext_case` column in §5, not the in-domain one):

```
VLM P2 only          0.4452
VLM P3 only          0.7499     ← best
CLIP only            0.6902
hybrid per finding   0.7364     ← per-finding best-scorer selection
P(hybrid > P3)       0.084
```

Per-finding scorer selection **deflates P3 by −0.0135**, at P = 0.084. CLIP wins
only 2 of 13 findings (`clustered_microcysts`, `microcalcifications_in_hypoechoic_mass`)
and 2 findings are dropped entirely (`kept = 0.0`, leaving 14 columns). Selecting
per finding on limited data buys nothing and costs a little — the honest reading
is that the selection overfits the selection set.

---

## 9. TF-IDF retrieval

### 9.1 ⚠️ It is not in the framework

**`TFIDFRetriever` belongs to the Hydra/FEVER pipeline
([breastMnist/src/debate_kg/retriever/tfidf_retriever.py](breastMnist/src/debate_kg/retriever/tfidf_retriever.py)),
the same place as the MedGemma judge, and it is not used by any debate corpus
that produces a ThothGNN v3 number.** `run_debate_v5.py` imports no retriever.

`breastMnist/README.md:145` states it plainly for the v5 line: *"No retrieval
(the ~370-triple KG is injected wholesale)."* What v5q actually injects is
narrower still — `all_findings()` selects the **16 stance findings** out of 370
triples, and all 16 go to both agents, shuffled per sample. The docstring gives
the reason: *"with no assigned side there is no principled way to give one agent
a subset without reintroducing the bias v5 exists to remove."*

Finding selection (`build_probes`,
[run_kg_featureprobe.py:90](experiments/03_kg_grounded_vlm/run_kg_featureprobe.py#L90)):
keep subjects of a `_POLARITY` relation; drop `_NOT_VISIBLE` and `birads*`;
weight `×0.5` if `object == "possible"`; deduplicate by feature keeping the
largest `|weight|`; sort malignant-first then alphabetically.

### 9.2 The spec, for the FEVER pipeline

From [SPEC.md:36-46](SPEC.md#L36-L46):

- **Index:** `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` over
  `"{subject} {predicate} {object}"` with underscores → spaces, cosine similarity,
  built once at startup.
- **Round 0 (seed):** top-`kg_round0_k` = **30** triples for a *label-conditioned
  seed query* derived from each expert's assigned stance.
- **Rounds 1+ (focused KG):** **only triples already cited** (`[CITED:t_NNN]`) by
  any claim node in prior rounds; falls back to the seed KG if no citations exist.
- **Claim→triple edges:** top-`kg_retrieve_k` = **3** per claim, recomputed fresh
  at dataset-build time rather than read from the debate JSON.
- **Motivation:** `num_ctx = 8192` makes injecting all 370 triples impractical.

The focused-KG rule is a monotone narrowing: the KG available in round *r+1* is a
subset of what was cited in rounds ≤ *r*, so the debate can only ever lose
ontology access as it proceeds, never gain it.

---

## 10. The controls, formally

### 10.1 ⚠️ The shuffled-KG control is **not** degree-matched

Both implementations are the same operation:

```python
def shuffle_kg(Ap, An, rng):           # run_thothgnn3.py
def rewire_matched(A, rng):            # sparsity.py — "same weight multiset"
    iu = np.triu_indices(F, 1)
    w = A[iu].copy(); rng.shuffle(w)
    B = np.zeros_like(A); B[iu] = w
    return B + B.T
```

Verified numerically on the real breast operators:

| property | preserved? |
|---|---|
| weight multiset | ✅ exactly |
| edge count (36 / 36) | ✅ |
| total weight (18.556 / 20.985) | ✅ |
| symmetry, zero diagonal | ✅ |
| **degree sequence** | ❌ **max deviation 0.4992 (`A⁺`), 0.5165 (`A⁻`)** |

So the control is **density- and weight-matched with random placement**. It is a
perfectly good control — arguably the right one, since it holds the graph's
"amount of smoothing" fixed and varies only *which* findings are joined — but it
is **not** degree-matched.

`thothgnn3.py:34` ("the same degree sequence") is wrong.
`run_thothgnn3.py:31` ("same edge count and weight distribution") is right.
`THESIS_SOURCEBOOK.md` lines 526, 716, 740, 820 say "degree-matched rewiring" and
should say **weight-multiset-matched rewiring**. `A⁺` and `A⁻` are shuffled
**independently**, so a shuffled draw also breaks the correspondence between the
two sign classes.

### 10.2 The other graph controls

- **no-graph:** `A⁺ = A⁻ = 0`. Isolates message passing from capacity — the model
  keeps every parameter, so any difference is the *edges*, not the size.
- **unsigned:** one matrix over `A⁺ + A⁻`, ablating the sign split.
- **`random_adj(F, n_edges, seed)`** ([kg_gnn.py:64](experiments/15_kggnn/kg_gnn.py#L64)):
  matched density, edges drawn afresh rather than permuted.
- **`top_k` sparsification** (`sparsity.py`): keep the *k* heaviest edges.

### 10.3 Shape-matched noise (the debate control)

The debate analogue of the shuffled KG, in
[debate_cost.py](experiments/17_hetgnn/debate_cost.py). The debate block
(channels 2–3, standardised) is **row-permuted across samples**, 20 draws:

```
Zs = hstack([ Zp , Zd[perm] ])      Zp = probe block, Zd = debate block
```

This holds column count, per-column marginal distribution and covariance
*within* the block fixed, and destroys only the image↔debate correspondence — so
it asks whether the debate contributes anything **beyond its dimensionality**.

From `debate_cost.json`:

```
probes only        test 0.8133   (32 features)
debate only        test 0.7022   (32 features)
probes + debate    test 0.8095   (64 features)
real delta         −0.003759
shuffled delta     −0.008688 ± 0.010298
real − shuffled    +0.004929   →   +0.48 sd
```

The debate alone reaches 0.7022, well above chance — but adds **nothing** once
probes are present, and is inside noise of a permuted block. Hence C2.

### 10.4 The prior-shuffle control (new, 2026-09-08)

`shared/prior_shuffle.py` permutes `π` **across findings in both places it
appears** — the feature channel `x_f[3]` and the penalty centre `ū[:,0]` —
keeping vocabulary and graph intact and destroying only semantics. Results so far:

```
              real     shuffled (5 draws)   no prior    real − shuffled
breast  AUC   0.7542   0.7673 ± 0.0216      0.7640      −0.0132  (−0.61 sd)
retina  refAUC 0.8727  0.8814 ± 0.0027      0.8914      −0.0087  (−3.16 sd)
derma   macroR 0.5587  0.5582 ± 0.0040      0.5594      +0.0005  (+0.11 sd)
```

On retina the **real ontology prior is significantly worse than a permuted one**,
and dropping it entirely is better than both. Derma is a null in the other
direction (+0.11 sd, indistinguishable from zero). Neither dataset shows the
ontology's semantics helping.

---

## 11. The baselines

### 11.1 B1 — Catfish Agent (Wang et al., arXiv 2505.21503)

Four roles on one backbone; A and B see the image and the 16 findings
independently shuffled; the **Catfish sees the transcript only, never the image**
(the information-asymmetry condition); a Moderator consolidates.

**Generation vs gating.** The complexity-aware trigger is *not* applied at
generation. Every case gets a Catfish round and the trigger's **inputs** are
recorded per sample; the gate is applied at analysis time. Equivalent, lets
`τ_conf` be cross-validated without regenerating the corpus, and makes
`B1-always-on` fall out of the same data. Both tones share a rounds-0/1 prefix,
so tone costs one extra branch rather than a re-run.

**The gate** ([analyse_b1.py:51](experiments/26_baselines/analyse_b1.py#L51)):

```python
def fires(d, tau):
    if d["trigger"]["silent_agreement"]: return True        # unconditional
    c = d["trigger"]["mean_conf"]
    return c is not None and tau is not None and c < tau     # ambiguity term
```

`silent_agreement = (o1 == o2)` on the two openings. `mean_conf` is the probe
ambiguity term, `mean_f |2·p_f − 1|` — 0 when every probe sits at 0.5, 1 when all
are saturated. So the Catfish fires when the agents **agree silently** or when
the image is **ambiguous**.

**Readout** — vote share, no fitted parameters:

```
score_base(d)      = malignant share of base_claims labels
score_branch(d,t)  = malignant share of the moderator's consolidated labels
                     (falls back to base+catfish+response if the moderator is empty)
score = score_branch if fires(d,τ) else score_base
```

**Selection:** tone × `τ_conf` over `[None] + quantile(train mean_conf,
[.1,.25,.5,.75,.9])`, 5-fold CV on the 546 training graphs only, then one test
evaluation. Frozen (breast): `tone=collaborative`, `τ_conf=0.8580`,
`cv_auc=0.6432`, **`trigger_rate=0.9652`**, `threshold=0.55`.

⚠️ The trigger rate exceeds 0.9, which trips the code's own warning: *"the gate
fires on >90% of cases — it is close to inert."* B1-full (0.6499) and
B1-always-on (0.6543) are consequently near-identical, and this should be stated
whenever B1's gate is described as complexity-aware — **as implemented and
selected, it is very nearly always-on.** Also note B1-full **loses** to
B1-no-catfish (0.6844, Δ = −0.0346, P = 0.038): the injected dissenter hurts.

### 11.2 B2 — GraphGeo

`G = (V, E, R)`, `R = {agree, conflict, transfer}`, **one node per AGENT per
image** (6 agents) — not per claim. That granularity difference is the point of
the comparison: per-assertion ontology grounding is impossible in their design.

Edges come from **prediction geometry**, not parsed verdicts (GraphGeo thresholds
geodesic distance between predicted coordinates; the binary port is):

```
r_agree     stance_u == stance_v                    (both directions, v → u)
r_conflict  stance_u != stance_v
r_transfer  conf_v − conf_u > τ_transfer            (directed, high → low)
```

`r_transfer` may coexist with agree or conflict, as in the paper.

**Node features:**

```python
img = np.concatenate([p[i] for p in pr])   # 32-dim: P2 ‖ P3, CONCATENATED
x   = np.tile(img, (len(AGENTS), 1))       # (6, 32) — same view for every agent
```

So the image representation is **exactly the probe vector ThothGNN v3 receives**,
which is what makes the comparison isolate the *graph* rather than the
perception. Missing probes → 0.5. The agent embedding is learnable and lives in
the model, not in the graph builder. Samples are skipped if any of the 6 agents
has a null stance.

**Result:** B2 ties v3 at full data (0.9058 vs 0.9048 referable AUC) with ~170×
the parameters and 3× the agents — a confound that runs in **B2's favour** and
still does not separate.

---

## 12. Corrections this document makes

| # | claim as documented | where | correct value |
|---|---|---|---|
| 1 | head A has **63** params | `ordinal.py` docstring; SOURCEBOOK 254, 548, 804, 838 | **62** — `59+4` double-counts v3's scalar `c` |
| 2 | total **65** params | SOURCEBOOK 579-580; retina README 229-230; `transfer.py` 6-7 | **62** |
| 3 | finding-identity block is **39** params | same three places | **36** (`u` 32 + `θ` 4) |
| 4 | shuffled-KG has "the same **degree sequence**" | `thothgnn3.py` 34; SOURCEBOOK 526/716/740/820 "degree-matched" | weight-multiset- and density-matched; **degree deviates by up to 0.52** |
| 5 | 0-parameter KG-signed sum standardised per feature | implied throughout | `μ_f, σ_f` from the **evaluation split**, not train — transductive |
| 6 | head C = **236** params | SOURCEBOOK 256, 548, 839 | **468** — CV selected `kdim=4`; 236 is the k=2 count and matches no reported C result |
| 7 | probe-64 = "logistic on the **32**-column vector (2 arms × 16)", AUC **0.8104**, 32 perception calls | SOURCEBOOK 199, 381 | **64 columns / 4 arms (P1–P4) / 65 params / 64 calls**; test AUC **0.8154**. 0.8104 is the *shuffled-mask control mean* |
| 8 | "Debate: rounds 3, **temperature 0.8**" | SOURCEBOOK 835 | true of **breast only**; retina and derma ran at the `Client.call` default **0.4** |

### probe-64, in full

`final_s0.py:71` builds it as `p64 = np.hstack([B[t][s] for t in TAGS])` over
**all four** phrasing arms, and labels it `probe-64 (all four)`. From
`final_s0.json`:

```
P1 direct/positive     16 feats   test 0.7899
P2 direct/negative     16 feats   test 0.8089
P3 verify/positive     16 feats   test 0.7822
P4 verify/negative     16 feats   test 0.8001
probe-32 (P1+P2)       32 feats   test 0.8083
probe-48 (P1..P3)      48 feats   test 0.8150
probe-64 (all four)    64 feats   test 0.8154   ← the row
/shuffled/test                    0.8104        ← what the sourcebook quotes
```

So the sourcebook row is wrong on four counts: the arms are **P1–P4, not P2+P3**;
the vector is **64 columns, not 32**; perception cost is **64 calls, not 32**;
and `0.8104` is the **mean of the five shuffled-mask control draws** (0.8076,
0.8120, 0.8170, 0.8091, 0.8062 → 0.81038), not probe-64's score.

The parameter count is **65**: `fit_ce` returns `w ∈ R^64` plus a scalar `b`.
Neither 64 (hardcoded in `assemble_table.py`'s `STATIC` dict) nor the `n_params:
32` in `phrasing_arms.json` is right — the latter is a *different model* (the
P2+P3 fitted arm) and its `32` is `len(cols)`, a column count mislabelled as a
parameter count. That model has **33** parameters.

`TRUNK` is defined by name in the code, so every transfer number is unaffected.
1–4 and 6–8 are documentation-only. 5 is substantive and should be stated as a
caveat wherever `0.7467 / 0.6466` appears.

---

# Part II — chapter queries, answered

Verified 2026-09-09. Duplicates of §1–§12 are not repeated; §-references point there.

## 13. Method

**13.1 probe-64 parameter count** → §12 item 7. Short form: **65** (64 weights +
intercept). It is four arms (P1–P4), not P2+P3; the `n_params: 32` in
`phrasing_arms.json` is a different model, and 32 there is a column count.

**13.2 Head C `kdim`** → §2.1. `selected/C/kdim = 4` → **468 params**. 236 is k=2
and matches no reported C result. `sum(p.size)` per head at k=4 = 117.

**13.3 Which `CLAIM_COLS` does v3 read? Only `stance`.** In
[thothgnn3.py:92-93](experiments/17_hetgnn/thothgnn3.py#L92-L93):

```python
mass = np.einsum("nc,ncf->nf", g["mask"], g["Acf"])
net  = np.einsum("nc,ncf->nf", g["mask"] * g["Xc"][:, :, 0], g["Acf"])
```

`g["Xc"]` is indexed **only at column 0** (`stance`), plus `mask`. `CLAIM_COLS`
has **11** entries, so **10 are never read** by v3 — not nine: `explicit`, `r0`,
`r1`, `r2`, `repeat`, `agent`, `n_out`, `in_agree`, `in_disagree`, `cites`. They
are built, stored, and unused. Round index, agent identity and rebuttal counts
therefore reach the model **only** through which claims exist, never as features.

**13.4 Temperature and corpus directory** → §1.3. Retina/derma both **0.4**
(inherited default), breast 0.8. The derma corpus directory **is** `debates_d1`
(`dermaMnist/data/derma/debates_d1`), 10,015 records.

**13.5 CMAX truncation on `debates_v5q`: zero samples truncated.**

```
780 samples · mean 19.51 claims · max 20 · >20 claims: 0 (0.00%)
distribution: 15:1  16:3  17:9  18:46  19:244  20:477
```

`CMAX = 20` cannot bind, because the **generator** caps first:
`max_claims = 4` in round 0 and `3` in rounds 1–2
([run_debate_v5.py:340](experiments/10_debate_v5/run_debate_v5.py#L340)), so
2 agents × (4+3+3) = **20 is the arithmetic ceiling**. The honest statement is
not "CMAX truncates nothing" but "**the generation cap binds on 61.2% of
samples** (477/780 sit exactly at 20)". Claim budget, not padding, is the limit.

**13.6 BI-RADS adjacency 72/120 — confirmed, and the file never changed.**
Recomputed: **72 edges / 120 pairs**, 33 lesions, one isolated finding
(`clustered_microcysts`). `git log --follow` on `kg_graph.py` returns **a single
commit** (`490118bd`), so the version that produced every v3 number is the
version in the tree.

## 14. Background

**14.1 The `.bib` — there is none.** `find . -name "*.bib"` returns **zero
files**. None of the four topics (VLM calibration on medical images;
acquiescence/assent bias; option-order/position bias; prompt-phrasing
sensitivity) is a verified entry, because there is no bibliography to verify
against. **Recast §2.3 ¶1 as forward-references to your own measurements** —
which are strong enough to carry it: acquiescence is measured directly (mean
p(present) 0.346 negated vs 0.608 positive, correlation +0.100), and
phrasing sensitivity is the P2/P3 split in §5.

**14.2 Quantisation: `Q4_K_M`, GGUF.** From `/api/show` on gpu10 —
`family qwen3vl`, `parameter_size 8.8B`, `quantization_level Q4_K_M`,
`format gguf`. So ~4.5-bit k-quant, not fp16. Usable for both the §2.3 clause
and the sustainability declaration.

**14.3 ICDR: DR axis only, with one shared term.** 152 triples, 16 derived
findings. **15 triples mention DME/macular/edema**, and of those only
**`hard_exudate`** touches a derived finding. The other 15 findings are DR-axis
only, and `level` comes from the DR ladder. So §2.2's single-severity-axis
framing is right, with one qualification worth a clause: hard exudate is the
sole finding the ontology places on both axes.

**14.4 BI-RADS pack: ultrasound-only in its assertions.** 370 triples;
`ultrasound_appearance_includes` is the largest relation (113). **No MRI,
tomosynthesis or elastography anywhere.** Four triples mention mammography — but
**only inside a free-text `notes` field** (e.g. t_333 *"Listed for mammography
but morphology applies to US"*), never as subject, relation or object. `notes` is
not read by `build_probes` or `all_findings`, so nothing non-ultrasound reaches a
prompt or the graph. Safe to state as ultrasound-lexicon-only; the notes are
provenance.

## 15. Introduction

**15.1 No judge feeds ThothGNN v3.** The four channels are the whole feature set.
Zero judge references in `thothgnn3.py`, `hetgraph.py`, `claims.py`, `ordinal.py`,
`nominal.py`, `retina_features.py`, `derma_features.py`. Corpora carry no judge
field. **Contribution 1 must say claims are merged with their stance and argument
mass, not "scored".** A MedGemma judge exists only in the legacy Hydra pipeline
(`breastMnist/src/debate_kg/judges/`) with `skip_judge=true` for every dataset run.

A judge arm now exists as a **diagnostic**, not a component
(`shared/run_judge.py`, `shared/results/judge_score.json`): retina test n=400,
judge macro-recall 0.2694 vs aggregator 0.2509, paired bootstrap **+0.0186, 95%
CI [−0.0532, +0.0814]** — includes zero, and worse on QWK/MAE/accuracy. It
strengthens C2 rather than feeding the model.

**15.2 ResNet-18 is ImageNet-pretrained, fine-tuned end-to-end.**
`resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, `fc → Linear(512, 2)`,
grayscale replicated to 3 channels, ImageNet normalisation. **Not from scratch.**
So "far below where the curve flattens" is *not* supported — a pretrained
backbone on 546 images is a transfer-learning regime, not a data-starved
from-scratch one. Recommend dropping that clause.

**15.3 There is no ResNet learning curve.** `19_curves` covers our pipeline only;
`02_resnet_baseline` ran one training-set size. **Cut the clause** — the
parameter-count argument is weaker but true, and §16.4 gives you the honest
alternative (the transfer-gap law), which is a stronger argument anyway.

**15.4 Open weights: a design choice, and the sourcebook supports it.** One
backbone throughout: **`qwen3-vl:8b-instruct`, Q4_K_M**, served by Ollama on a
local GPU cluster at **$0 external cost**, no paid API. Partner models were
*screened* (`minicpm-v:8b` 0.7986, `gemma3:12b` 0.7303, `medgemma:4b` 0.5974 —
train n=250, bar ≥0.60) and `minicpm-v:8b` was used for the heterogeneous arm, so
the infrastructure is genuinely model-agnostic. **Write it as an objective**;
nothing in the record indicates a closed model was wanted and refused. The
single-backbone limitation is separate and stays.

**15.5 No citation for "long text diverts attention from the image".** I can
name no paper for it with confidence, and there is no `.bib` to check against.
Your Round-1 result (`03_kg_grounded_vlm`: KG-in-prompt **0.6013**, one arm
**0.4734**, below chance) is the evidence you actually hold. **Write it as a
mechanism your own results demonstrate, uncited, flagged for chapter 2.**

## 16. Conclusion / generalisation

**16.1 B1 and B2 *are* run on retina.** `retinaMnist/results/b1_retina.json`
(Sep 5) and `b2_retina.json` (Sep 6) both exist with full results; B2 has **five
seeds (0–4)** on three arms. **§6.9 is current; §5 Phase 6 "NOT RUN",
`01-outline.md` and `02-numbers-discipline.md` are stale.** The outline's
"handling the unfinished work" section and that future-work entry should go.

```
retina, head A, 5 seeds, test n=400
  B2-full          refAUC 0.9070 ± 0.0033   QWK 0.7071   acc 0.5435
  B2-anchored      refAUC 0.9052 ± 0.0033   QWK 0.7049   acc 0.5390
  B2-no-relation   refAUC 0.9057 ± 0.0017   QWK 0.7002   acc 0.5215
  ThothGNN v3 (A)  refAUC 0.9043 (seed 0) / 0.9053 ± 0.0029 (5 seeds)
```

**16.2 Nothing measures backbone capability against debate contribution.** The
screening figures are **single-agent classification on train n=250** — not debate
quality. The only paired evidence is HOM vs HET (`25_twomodel`, n=780):

```
HOM qwen×qwen     probes 0.8034  +debate 0.7639  contribution −0.0395
HET qwen×minicpm  probes 0.8034  +debate 0.7654  contribution −0.0380
DoD +0.0015   P(HET>HOM) = 0.5485   bar 0.95 → does not clear
```

Two backbone pairings, one contrast, null. **Mark your sentence as inference from
the mechanism results.**

**16.3 C1's ResNet number is 0.9478, not 0.9442.** Two different runs of the same
model on the same split (BreastMNIST test n=156):

| | AUC | source |
|---|---|---|
| 0.9442 | original baseline, 224px, seed 0 | `02_resnet_baseline/results/224px/summary.json` |
| **0.9478** | **retrain inside `16_external`** | `16_external/results/resnet_external.json` |

`resnet_external.json` records `"original_auc": 0.9442, "reproduced": true`. Use
**0.9478**, because it is the same run that produced the external 0.6605 — the
pair must come from one model. The master table's 0.9442 is the older run.

**16.4 ResNet training procedure — documented, and it weakens C1's fairness.**
`02_resnet_baseline/train_resnet.py`:

| item | value |
|---|---|
| initialisation | **ImageNet-pretrained** `IMAGENET1K_V1` |
| augmentation | **none** — load, replicate to 3 channels, ImageNet-normalise |
| optimiser | Adam, lr 1e-3, batch 128 |
| schedule | 100 epochs, lr ×0.1 at 50 and 75 |
| early stopping | **none**; best-val-AUC checkpoint restored |
| model selection | maximum validation AUC (best epoch 6 of 100) |
| seeds | **1 (seed 0)** at 224px — `config.seeds: [0]`, std 0.0 |
| class weighting | **off** |

⚠️ The module docstring says "5 seeds (0–4), mean ± std" but **the 224px run used
one seed**. And **no augmentation** on 546 images is a weak baseline. Both cut
against C1's fairness: the supervised comparator is under-tuned, so its in-domain
0.9478 is if anything a *floor*, and its single seed has no error bar. State this.

**16.5 §6.2's "KG pipeline (findings + lesions)" arm.** From
`18_lesion/analyse_lesion.py` + `external_lesion.py`:

- **Fitted**, not zero-parameter: logistic (`fit_ce`/`fit_rank`, CV-selected) on
  the concatenated block.
- **52 columns → 53 parameters**: 32 finding columns (**pooled P2+P3**, both arms
  concatenated) + 20 lesion probe columns.
- **Not ThothGNN v3.** No message passing, no graph, no debate channel. It is a
  flat readout over probes + lesion probes. C1 is a claim about **the perception
  layer plus the ontology's lesion differential**, not about the GNN.

**16.6 Phrasing arm for all four §6.2 rows: pooled P2+P3** (the fitted arm of
§5, `n_params = 32` columns). ResNet has no phrasing arm. ⚠️ Because it is
pooled, the **P2 contamination disclosure belongs in that table's caption**: P3
alone scores higher externally (0.7499) than the pooled arm (0.6919).

**16.7 The six gap-law blocks** (`22_transfer/results/gap_law.json`). All
in-domain BreastMNIST test **n = 156 (image level)**; all external BUS-BRA
**n_cases = 1064 (patient level)**, single run, no ensemble:

| block | params | in-domain | external | gap |
|---|---:|---:|---:|---:|
| BiomedCLIP contrastive | 16 | 0.8745 | 0.7027 | +0.1718 |
| hybrid, scorer per finding | 14 | 0.8241 | 0.7202 | +0.1039 |
| VLM both phrasings pooled | 32 | 0.8133 | 0.7325 | +0.0808 |
| VLM negated phrasing | 16 | 0.8089 | 0.7171 | +0.0918 |
| VLM verification phrasing | 16 | 0.7822 | 0.7456 | +0.0366 |
| VLM verification, KG rule | 0 | 0.7467 | 0.7499 | −0.0032 |

In-domain range **0.7467 – 0.8745** (0.128 wide).

**16.8 Slope 1.393 — SE and leave-one-out, computed here.**

```
slope 1.3931   SE 0.0772   95% CI [1.1787, 1.6075]   dof 4   p = 6e-5
intercept −1.0458   r = 0.9939   R² = 0.9879
largest residual +0.0107 (VLM negated); all |resid| ≤ 0.011
leave-one-out slopes: 1.3655 – 1.4304  (drop-one range ±0.037)
```

The fit is stable to dropping any single point, and no point is an outlier. It is
still **6 points on a 0.128-wide x-range**, so quote the CI and the LOO range
rather than the bare r, and describe it as a **within-family regularity across
perception variants**, not a law about model classes.

**16.9 No paired bootstrap exists between the KG pipeline and ResNet-18.** The
only bootstrap in `external_lesion.json` is `P_combined_better = 0.93125`
(findings+lesions vs findings-only). **Worth running** — it is the single
inferential statement C1 most needs, and everything required is on disk.

**16.10 ⚠️ External patient-level bAcc — the reversal does not survive it.**

| system | ext AUC (case) | **ext bAcc (case)** |
|---|---:|---:|
| ResNet-18 | 0.6605 | **0.5580** |
| KG, findings + lesions | **0.7485** | **0.5021** |
| KG, findings only | 0.7331 | 0.5048 |
| KG, lesions only | 0.6770 | 0.5014 |

**The KG pipeline is at chance (0.502) on thresholded external decisions, and
ResNet beats it there.** C1's ordering reversal holds for **ranking (AUC)** and
**not** for operating-point performance. This must appear wherever C1 does:
the claim is "ranks better out of domain", not "decides better out of domain".

**16.11 `n_ext` 1850 vs 1875 — the two systems used different image sets.**
ResNet was evaluated on **1875** images (`busbra_pad2`); the KG pipeline on
**1850** — 25 images lack lesion probes. Both aggregate to the **same 1064
patients**, and both use `busbra_pad2`, so the case-level comparison is close to
like-for-like but not exactly paired. Fix the 25 or state the discrepancy.

## 17. Cross-ontology transfer

**17.1 Trunk forward equations** → §1.2–§1.3.
**17.2 The d=4 channel definitions** → §1.1 (mass = masked citation count; net
stance = masked signed sum; prior = KG stance / recentred ICDR level / class map).

**17.3 ⚠️ The ceiling is confounded — your suspicion is correct.**

- `ceiling`: `ordinal.fit(target)` — **all** parameters trained jointly.
- `transfer`: `fit_readout_only(target, source_trunk)` — **only `u`, `theta`**;
  `TRUNK = (W0, Wp, Wn, b)` frozen.

Both use the **identical optimiser and budget** (800 epochs, lr 0.15, momentum
0.9, same l2, same prior centre), so the schedule is not the confound — but the
ceiling optimises **26 more free parameters**. `ceiling − transfer` therefore
mixes *trunk provenance* with *trunk trainability*, and **99.4% "retained" is not
a clean ratio**. The missing arm is: train trunk on target → freeze → refit
readout with `fit_readout_only`. It does not exist and would cost minutes.

**Lead with `transfer − random` instead**, which *is* clean — both arms use
`fit_readout_only` with a frozen trunk, differing only in where the trunk came
from: **+0.0587, +4.35 sd** (breast→retina).

**17.4 Both directions are head A, k=2. Head C transfer was not tested.**
`transfer.json/config`: `kdim 2`, `seed 0`, `l2 0.03`, `trunk_params
[W0,Wp,Wn,b]`, `readout_params [u,theta]`. Calls `ordinal.fit`/`ordinal.predict`
= head A throughout.

**17.5 Breast ceiling 0.7713: P3-only, single seed.** `transfer.py:66` sets
`thothgnn3.PHRASE_COLS = (1,)` — "P3 alone, per round 4". Seed 0 only.
Script `retinaMnist/src/retina_kg/transfer.py`; file
`retinaMnist/results/transfer.json` → `retina->breast/ceiling/auc = 0.7713`.

**17.6 The readout refit consumes the full target training split** — retina
**1080**, breast **546** labels. There is **no label-budget sweep**, and it would
be cheap: `fit_readout_only` is seconds per fit, so a 5-point × 3-seed curve is
minutes. Worth running — it converts C5 from "the trunk transfers" into "the
trunk transfers and saves *N* labels", which is the deployable claim.

**17.7 The random-trunk control refits the readout on every draw** — `Qr =
fit_readout_only(...)` inside the loop — **5 draws**, both directions. The sd
gap is not a procedural difference: breast→retina sd 0.0135 on 400 test images
(referable AUC), retina→breast sd 0.0561 on 156 (binary AUC). Smaller, noisier
target, wider spread. Hence retina→breast is only **+1.73 sd** and, as the
sourcebook notes, loses to the 0-parameter rule (0.7556 vs 0.7201).

## 18. Graph topology

**18.1 `retina KG − shuffled (paired)` measures referable-DR AUC**, paired
**by seed** across the 5 seeds (not by sample). `ordinal_seeds.json`:

```
A/kg_minus_shuffled  mean −0.00077  sd 0.00526
vals: −0.00030, +0.00654, −0.00311, +0.00220, −0.00917
```

**18.2 The three unnamed rows.** The "every topology variant matches its shuffled
control" list (SOURCEBOOK:226) pairs are:

| real / shuffled | architecture | source |
|---|---|---|
| 0.8137 / 0.8137 | KG-Laplacian readout (#3) | `15_kggnn/kg_laplacian.py` |
| 0.7913 / 0.7903 | Probe-GNN (#4) | `15_kggnn/probe_gnn.py` |
| **0.7863 / 0.7865** | **GNN on KG-prior anchor** | `15_kggnn/kgprior_gnn.py`, README:76-77 |
| 0.7602 / 0.7573 | ThothKG (#5) | `15_kggnn/thoth_kg.py` |

**18.3 So 0.7863/0.7865 is a sixth architecture**, `kgprior_gnn.py`, **not listed
in §4.4's table of five**. The "four independent implementations" are #3, #4, #5
and this sixth; **ThothGNN v3 and B2 are not among them**. Either add it to §4.4
as architecture 6 or say "four of the six variants".

**18.4 All eight numbers: BreastMNIST test, n = 156, single run (seed 0), not a
seed mean.** ⚠️ They are **pooled P2+P3** — these predate the Round-4 P3-only
decision — so **the P2 contamination disclosure has to appear in the caption**.

**18.5 B2 parameter counts, computed here:**

```
B2-full         breast/binary 10,737   retina/head-A 10,741
B2-no-relation  breast/binary  6,001   retina/head-A  6,005
```

Retina **B2-no-relation = 6,005**. (The +4 is head A's θ.)

## 19. Debate

**19.1 Both targeted-placement results have numbers.**

*Low-confidence gating* — `17_hetgnn/results/uncertainty_gate.json`, BUS-BRA
external, **n = 1875 images / 1064 cases**, case-level AUC:

```
probes only                     0.8055
debate at all 16 findings       0.7976   (−0.0079)
best gated arm (near5, k=8)     0.8024   (−0.0031 vs probes, +0.0048 vs ungated)
best gated arm (spread, k=8)    0.7998   (−0.0057 vs probes)
```

*Disagreement gating* — `17_hetgnn/results/debate_when_split.json`, **n = 1870**:

```
probes            0.8080
debate everywhere 0.8080
debate only where the agents split (44.4% of cases)  0.8079
claims saved 55.5%
```

**Every gated arm is still below probes-only.** Targeting recovers roughly half
the damage the debate does and never turns it positive. That is a publishable
sentence — keep the section.

**19.2 `+0.48 sd` shape-matched noise:** BreastMNIST **test, n = 156**, breast
`debates_v5q`, **20 permutation draws**, debate block = channels 2–3 standardised
on train. `real − shuffled = +0.0049, sd 0.0103`
(`17_hetgnn/results/debate_cost.json`).

**19.3 Attack-graph 0.5017:** **all 780 samples, all three splits pooled** —
15,220 claims. Not a test-split figure. Source `17_hetgnn/diag_credibility.py` →
`results/credibility.txt`; the table is in `17_hetgnn/README.md`.

**19.4 The 08-20 additivity result has numbers.** `14_kgtensor/gate_1run.json`:

```
real labels:  counts CV 0.5996   interaction CV 0.5754   delta −0.0242
positive control (synthetic XOR):
              counts CV 0.5112   interaction CV 0.7039   delta +0.1926
```

Not qualitative. The positive control is what makes it worth printing: the same
protocol detects a genuine interaction at +0.19 and finds **−0.02** in the real
labels, so the null is a null and not a power failure.

**19.5 There is no no-debate arm in the five-seed retina run.**
`ordinal_seeds.json` carries `refauc`, `qwk`, `acc`, `macroR`, `mae`,
`no_graph_refauc`, `shuffled_refauc`, `kg_minus_shuffled` — **no `no_debate`**.
The no-debate arm exists at **seed 0 only** (`ordinal_full.json`). Since the
five-seed run is what corrected the topology estimate, **the debate claim on
retina rests on one seed**; either say so or run it (minutes).

**19.6 Three B1 comparisons carry P values**, not one:

```
B1-full vs B1-no-catfish        Δ −0.0346   P 0.038
B1-full vs B1-always-on         Δ −0.0045   P 0.000
B1-collaborative vs adversarial Δ −0.0149   P 0.273
```

The B1-full vs self-consistency comparison lives in the master table. The
6-agent row has no P value against B1.

**19.7 Citation keys: I cannot verify either paper.** No `.bib`, no network
access, and I will not assert a title or author list from an arXiv id. All I can
confirm is what the code claims: `run_catfish.py` cites *Catfish Agent, Wang et
al., arXiv 2505.21503*; `b2_graph.py` cites *GraphGeo, arXiv 2511.00908*. Both
are **our re-implementations from the papers' descriptions; no released code was
used** — that phrasing is required wherever they appear. Verify both records
yourself before writing them.

## 20. Perception

**20.1 `P(better)` is never defined in the sourcebook** — it appears bare in the
tables at lines 417, 427, 450. The pieces exist: config table gives *"Bootstrap:
paired, 4000 resamples, on patients"*, and `16_external/PREREGISTRATION.md:61`
sets *"decided at P(better) ≥ 0.95"*. **Nothing writes the definition.** You will
need to draft the two sentences; the components are: fraction of 4000 paired
bootstrap resamples, resampled at patient level, in which method A's metric
exceeds method B's, against a pre-registered bar of 0.95.

**20.2 Mask pre-registered bar: `P(better) ≥ 0.95`** —
`23_masktest/PREREGISTRATION_2026-08-27.md:58`, *"the same bar used for the
KG-topology endpoint"*. Same 0.95. The document also records the prediction:
*"maskring clears the bar and maskdim probably does"*.

**20.3 0.8515:** the ontology's 0-parameter KG-signed sum applied to the
**radiologist's own BrEaST descriptor annotations** instead of probe outputs —
**BrEaST, n = 252**. Source `16_external/analyse_e2.py:109`, reported in
`16_external/PREREGISTRATION.md:191`. Its companion decomposition:

```
probes → label                 0.7020
radiologist descriptors → label 0.8515      ← the ceiling
perception error (ceiling − probes)  0.1495
inference error (1 − ceiling)        0.1485
```

**20.4 BiomedCLIP geometry −0.20 is CLIP minus the VLM-P3 arm, within the
geometry family:** `0.5788 − 0.7818 = −0.2031`, on **n = 2 findings**
(`irregular_shape`, `oval_shape`). Per-family (`compare_scorers.json/by_family`):

```
geometry     n=2  P3 0.7818  CLIP 0.5788   −0.2031
posterior    n=1  P3 0.7805  CLIP 0.6500   −0.1305
echogenicity n=4  P3 0.7279  CLIP 0.6271   −0.1008
margin       n=6  P3 0.5391  CLIP 0.4559   −0.0833
```

⚠️ Write the delta and the **n = 2**. "Worst on geometry" is true of the family
means but rests on two findings; the honest form is that CLIP trails P3 on all
four families and most on geometry.

**20.5 "Five of thirteen": the BrEaST-annotated subset.** 13 of the 16 findings
carry a family/descriptor annotation; the other 3 are unannotated. Split
**BrEaST, n = 252**, against the **BrEaST radiologist descriptor annotations**
(not the malignancy label) — which is what makes the selection label-free.
`sign_audit.py:20`; the families in `compare_scorers.py` sum to 2+1+6+4 = 13.

## 21. Experimental programme

**21.1 08-15 / 08-16.** The chronology rows carry no AUCs; the only figures in
`03_kg_grounded_vlm/README.md` are **0.6013** and **0.4734 (below chance)** for
two KG-in-prompt arms, on **`medgemma:4b`** — a backbone later *failed* by the
two-model screening bar (0.5974 < 0.60). ⚠️ **Your prompt-integration hypothesis
cannot be stated even as a suggestion from these**: the arms are confounded with
a backbone that cannot do the task. The comparable qwen3-vl numbers do not exist.
Either re-run one arm on qwen3-vl (cheap) or drop the claim.

**21.2 Stage labels: Phases 0–6, with Phases 3–6 *also* carrying Round numbers.**
`### Phase 0 … Phase 1 … Phase 2 …`, then `### Phase 3 — Round 3`,
`Phase 4 — Round 4`, `Phase 5 — Round 5`, `Phase 6 — Round 6`.
**Nothing is named "Round 1" or "Round 2".** Round *n* = Phase *n* for n ≥ 3.

**21.3 The Round 3 registration exists — it is just missing from §12.**
`experiments/16_external/PREREGISTRATION.md`, **added 2026-08-22** (git
`--diff-filter=A`), the first day of Phase 3 (22–24 Aug), so it genuinely
predates the results. It states the bar at line 61 (*"decided at P(better) ≥
0.95"*) and carries the Round-4 KG-topology endpoint at lines 202+.
**The row is not post-hoc.** Add the file to §12; no threats-chapter finding.
All four registrations, with git add-dates:

```
16_external/PREREGISTRATION.md               2026-08-22
23_masktest/PREREGISTRATION_2026-08-27.md    2026-08-27
25_twomodel/PREREGISTRATION_2026-08-30.md    2026-08-30
26_baselines/PREREGISTRATION_2026-08-31.md   2026-08-31
```

Every one is dated on or before the run it governs.

**21.4 The Catfish predictions — there are six, not four**
(`26_baselines/PREREGISTRATION_2026-08-31.md:150-168`), four on B1 and two on B2:

1. **B1-full null against B1-no-catfish**, |Δ| < 0.02, P < 0.95.
2. **B1 trigger rate high (>70%)**, gate close to inert, `B1-always-on` matches.
3. **Collaborative beats adversarial.**
4. **B1 does not beat the independent-ensemble control.**
5. B2-full lands in 0.60–0.66.
6. B2-anchored approaches v3, residual < 0.02.

Outcomes on the four B1 predictions: **(1) FAILED** — Δ = −0.0346 exceeds 0.02
and P = 0.038 is decisive, B1-full is *worse*, not null. **(2) HELD** — trigger
rate 0.9652, and B1-full 0.6499 vs B1-always-on 0.6543. **(3) FAILED** —
collaborative 0.6499 < adversarial 0.6648, P = 0.273. **(4) HELD** — B1-full
0.6499 vs self-consistency 0.7496. So **"three of four failed" is wrong: two of
four failed.** Check that sentence against this list.

**21.5 Yes — `01-outline.md` and `02-numbers-discipline.md` are stale** on B1/B2
retina (§16.1) and should be updated to the five-seed C3 retina result
(`kg_minus_shuffled −0.00077 ± 0.00526`, §18.1).

Note (added 2026-09-09, this reconstruction only): the original derma
prior-shuffle result was pending when this document was first written and is
now filled in at §10.4 as +0.0005 (+0.11 sd) — a null in the other direction
from retina.
