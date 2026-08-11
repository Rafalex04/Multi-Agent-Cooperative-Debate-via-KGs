# Reading list and learning roadmap

This file is for the human, not Claude Code. Architectural source of truth lives in `SPEC.md`.

Citations marked **✅** are confirmed venues; **⚠️** means likely correct but verify before citing in the paper.

## Recommended build order

1. **Weeks 1–4**: reading. Anchors: Du et al., Edge et al. (GraphRAG), Zheng et al., Pan et al., FEVER, plus Konieczny & Pino-Pérez foundational paper. Toy KG pipeline with REBEL on a few Wikipedia articles.
2. **Weeks 5–6**: Modules 1–3. Get one debate running end-to-end on one claim. Plumbing first, quality later.
3. **Weeks 7–8**: Modules 4–6. Run the full loop on 5–10 claims; inspect outputs by hand. Most of v1's intellectual effort lives here.
4. **Weeks 9–10**: Run on 100–200 FEVER claims. Compute the four metrics. Build baselines. v1 should beat them on accuracy; if it doesn't, debug.
5. **Weeks 11–12**: Ablations and writing. Method and experiments first; intro and related work last.
6. **Months 4–6**: v2 — learned GNN merger.

## Target venue

Lead venue family: ACL / EMNLP / NeurIPS (multi-agent agreement and decision-making framing). Plan B if results are marginal: KR / AAAI (belief-merging framing). Pick lead venue early and align related-work emphasis accordingly.

## Background — multi-agent debate and agreement

- ✅ Du, Li, Torralba, Tenenbaum, Mordatch. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* **ICML 2024.**
- ⚠️ Irving, Christiano, Amodei. *AI Safety via Debate.* arXiv 2018. Foundational framing of debate as alignment mechanism.
- ⚠️ Khan et al. *Debating with More Persuasive LLMs Leads to More Truthful Answers.* **ICML 2024** (verify).
- ⚠️ Liang et al. *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* EMNLP 2024 (verify).
- ⚠️ Chan et al. *ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate.* ICLR 2024 (verify).
- ⚠️ Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
- ⚠️ Wu et al. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* arXiv 2023 (later venue — verify).
- ⚠️ Park et al. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.
- ⚠️ TreeDebater (Wang et al.). *Strategic Planning and Rationalizing on Trees Make LLMs Better Debaters.* Recent ICLR — verify venue and year.
- Optional broader context: Minsky, *Society of Mind* (1986). Conceptual ancestor of cooperative multi-agent reasoning.

## Background — belief merging and knowledge integration

- ⚠️ Konieczny & Pino-Pérez. *On the Logic of Merging.* KR 1998 + successors in *Journal of Logic and Computation* and *Artificial Intelligence*.
- ⚠️ Flouris et al. *Ontology Change: Classification and Survey.* Survey on ontology repair.

## Module 1 — KG construction

- ✅ Huguet Cabot & Navigli. *REBEL: Relation Extraction By End-to-end Language generation.* **Findings of EMNLP 2021.**
- ⚠️ Zhang & Soh. *Extract-Define-Canonicalize (EDC).* EMNLP 2024 (verify track).
- ⚠️ Pan et al. *Unifying Large Language Models and Knowledge Graphs: A Roadmap.* IEEE TKDE.
- ⚠️ Hogan et al. *Knowledge Graphs.* Synthesis Lectures / open-access book, 2021. Reference book.
- ⚠️ Bordes et al. *Translating Embeddings for Modeling Multi-relational Data (TransE).* NeurIPS 2013. Foundational KG embedding.

## Module 2 — GraphRAG retriever

- ⚠️ Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research, 2024.
- ✅ He et al. *G-Retriever.* **NeurIPS 2024.**
- ✅ Sun et al. *Think-on-Graph.* **ICLR 2024.**
- ⚠️ Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering (DPR).* EMNLP 2020.
- ⚠️ Wu et al. (BLINK) or De Cao et al. (GENRE) — entity linking. Pick one and verify.

## Module 3 — Debate orchestrator

- See Background — multi-agent debate.
- Use TreeDebater's prompt appendix as a depth target for prompt engineering.

## Module 4 — Judge panel

- ✅ Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* **NeurIPS 2023.**
- ⚠️ Verga et al. *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models.* arXiv 2024 (verify).

## Module 5 — Consensus check

- See Konieczny & Pino-Pérez. Iterated merging convergence properties are the relevant theory.

## Module 6 — Rule-based merger

- See Konieczny & Pino-Pérez and Flouris et al.

## Benchmark

- ✅ Thorne et al. *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* **NAACL 2018.**
- ⚠️ Aly et al. *FEVEROUS.* NeurIPS 2021 datasets track. Harder version with structured evidence.
- ⚠️ Jiang et al. *HoVer.* EMNLP 2020. Multi-hop fact verification.
- ⚠️ Wadden et al. *SciFact.* EMNLP 2020. v2 target benchmark.

## v2 reading (for later)

- ⚠️ Schlichtkrull et al. *R-GCN.* ESWC 2018.
- ⚠️ Vashishth et al. *CompGCN.* ICLR 2020.
- ⚠️ Sun et al. *RotatE.* ICLR 2019.
- ⚠️ Recent graph foundation models surveys (verify titles).
