# Copolymer CQ Pipeline

An agentic, retrieval-augmented pipeline that automatically generates,
validates, and consolidates **competency questions (CQs)** from scientific
literature for copolymer knowledge graph and ontology development.

Applied to 14 expert-curated papers on supercritical-CO₂ polymer foaming,
the pipeline produced 682 CQs with a mean question faithfulness of **0.98**
and **zero off-paper questions**, consolidated into **120 domain-level CQs**
- a 77.6% reduction.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Running the pipeline](#running-the-pipeline)
5. [Evaluation metrics](#evaluation-metrics)
6. [Reproducing the results](#reproducing-the-results)
7. [Data](#data)
8. [Reproducibility](#reproducibility)
9. [License](#license)

---

## Pipeline overview

The pipeline operates in two phases.

### Phase A - Per-paper CQ generation (v2.1 LangGraph)

Each paper is processed independently through a LangGraph state machine
with the following stages:

| Stage | Agent / utility | What it does |
|---|---|---|
| **Chunk** | `paper_chunker.py` | Splits paper text into overlapping chunks (~3,500 chars, 200-char overlap) using a recursive character splitter that prefers natural boundaries (section → paragraph → sentence) |
| **Index** | `paper_indexer.py` | Embeds each chunk with BGE-small and stores in a per-paper FAISS index for later retrieval |
| **Iterative extraction** | `paper_iterative_extractor.py` | Reads chunks one at a time while maintaining a bounded **rolling knowledge state** - a structured, evidence-bearing summary of what the paper says so far. Each chunk's output is merged into four categories: *concepts*, *relations*, *properties*, *findings*. Every item carries a verbatim evidence sentence, surrounding context, and a calibrated confidence (high / medium / low). Category caps (80 / 60 / 60 / 40) keep the state compact regardless of paper length |
| **Retrieve** | `paper_retriever.py` | Queries the FAISS index with the top-K most-mentioned concepts to fetch the most relevant passages - these ground the generator |
| **Generate** | `paper_persona_cq_generator_v2.py` | For each of **seven domain personas** (Synthesis Chemist, Characterisation Scientist, Processing Engineer, Computational Scientist, Ontology Engineer, Student, PhD Researcher), generates persona-conditioned CQs from the knowledge state plus retrieved passages. A coverage-gap pass generates additional questions for any extracted concepts left untouched by the first round |
| **Validate** | `paper_persona_cq_validator_v2.py` | Two-layer check: (1) cheap rule checks (persona coverage, concept-usage hallucination guard, near-duplicate detection); (2) per-question LLM judge scoring answerability and specificity against semantically selected evidence |
| **Refine** | `paper_persona_cq_refiner_v2.py` | Failing/borderline questions are passed to a refiner that receives the judge's diagnosis and revises, keeps, or flags each question as unrepairable - nothing is silently dropped |
| **Evaluate** | `cq_evaluator.py` | Reference-free quality evaluation (see [Evaluation metrics](#evaluation-metrics)) |

**Output:** `data/papers/v2/cq_output/{paper_id}_cq_v2.json` - one file per paper.

<details>
<summary><strong>Validator pass threshold - why 0.75?</strong></summary>
<br>

The LLM judge scores each question on two independent dimensions, each on a three-point scale:

- **Answerability** - can the question be answered using only the retrieved evidence passages?
  - `1.0` the answer is explicitly stated in the passages
  - `0.5` the answer can be derived with a short inference
  - `0.0` the passages do not contain the answer
- **Specificity** - is the question concrete enough to be useful for ontology construction?
  - `1.0` names a specific material, method, property, or value
  - `0.5` somewhat specific but uses generic terms
  - `0.0` vague catch-all question with no anchor

The per-question score is the mean of the two dimensions. Because each dimension can only be 0.0, 0.5, or 1.0, only a small number of combined scores are possible:

| Answerability | Specificity | Score | Verdict |
|---|---|---|---|
| 1.0 | 1.0 | 1.00 | pass |
| 1.0 | 0.5 | **0.75** | pass |
| 0.5 | 1.0 | **0.75** | pass |
| 0.5 | 0.5 | 0.50 | borderline - sent to refiner |
| 1.0 | 0.0 | 0.50 | borderline - sent to refiner |
| 0.0 | 1.0 | 0.50 | borderline - sent to refiner |
| 0.5 | 0.0 | 0.25 | fail - sent to refiner |
| 0.0 | 0.5 | 0.25 | fail - sent to refiner |
| 0.0 | 0.0 | 0.00 | fail - sent to refiner |

**0.75 is the natural breakpoint**, not an arbitrary cut-off. It is the lowest score achievable when at least one dimension is fully satisfied. A question scoring 0.75 is perfect on one dimension and partial on the other - good enough to be useful without rewriting. A question scoring 0.50 has a real problem on at least one dimension and genuinely needs the refiner.

Questions scoring below 0.75 are sent to the refiner with the judge's exact diagnosis. The refiner can revise, keep, or flag each question as unrepairable. Nothing is silently dropped.

> Note: this threshold governs individual question survival during generation. It is separate from the corpus-level evaluation metrics (faithfulness, coverage, diversity, triplifiability) which measure the quality of the final question set after all questions have passed through the validator and refiner.

</details>

---

### Phase B - Cross-paper CQ narrowing

All per-paper CQ sets are pooled and narrowed in four sequential steps:

| Step | Agent | What it does |
|---|---|---|
| **Step 1 · Dedup** | `cq_dedup_agent.py` | Merges near-equivalent questions *within* each paper. Merges only within the same archetype (a causal question is never merged into a definitional one). Prefers the version at the higher Bloom level. Every removed question is logged with a pointer to the question that absorbed it |
| **Step 2 · Generalise** | `cq_generalizer_agent.py` | Groups functionally equivalent questions *across* papers and replaces paper-specific entity names with their domain role - for example, a question about a specific amorphous polymer is generalised to refer to "amorphous polymer" so it applies to the whole domain |
| **Step 3 · Validate (RAG)** | `cq_validator_agent.py` | Checks each generalised CQ against the existing FAISS indexes (no re-indexing). Retrieves the most relevant passages, generates a candidate answer, judges fidelity, and assigns a quality flag: `STRONG`, `PARTIAL`, `OVER_GENERALISED`, `PAPER_SPECIFIC`, or `UNDER_ANSWERED` |
| **Step 4 · Consolidate** | `cq_consolidation_agent.py` | Two-stage merge. **Stage 1** merges questions sharing the same triple signature (subject-class, predicate, object-class) within each archetype, replacing differing fillers with typed variables. **Stage 2** merges across archetypes wherever the same signature still appears. Every consolidated CQ carries a `triple_pattern`, `sparql_sketch`, typed `variables`, and `merged_from` provenance. A semantic audit flags any unaccounted or hallucinated identifiers |

**Output:** `data/papers/v2/consolidated/cross_paper_consolidated.json`

---

### Model assignments

Each agent role is assigned a specific open model family (configured in `config/models.json`):

| Role | Model family | Reasoning |
|---|---|---|
| Iterative extractor | Qwen | Strong instruction following for structured JSON extraction |
| CQ generator | Gemma | Creative, persona-conditioned generation |
| CQ validator / judge | DeepSeek | Deliberate reasoning (think mode) for quality judgement |
| CQ refiner | Qwen | Precise targeted rewriting |
| Dedup + generalise | DeepSeek | think mode for archetype-locked merge decisions |
| Consolidation | Llama | Large context for cross-archetype merge |
| Embeddings | BGE-small (BAAI/bge-small-en-v1.5) | Fast, local, no API dependency |

Using different model families for generation and judging reduces
self-preference effects in LLM-based evaluation.

---

## Repository structure

```
Agentic-CQ-Generation/
│
├── main_paper_persona_cq_v2.py   # Phase A entry point
├── main_cq_dedup.py              # Phase B Step 1
├── main_cq_generalize.py         # Phase B Step 2
├── main_cq_validate.py           # Phase B Step 3
├── main_cq_consolidate.py        # Phase B Step 4
├── aggregate_eval_v2.py          # Aggregate evaluation → corpus stats
│
├── agents/                       # One file per agent role
│   ├── paper_chunker.py
│   ├── paper_indexer.py
│   ├── paper_iterative_extractor.py
│   ├── paper_retriever.py
│   ├── paper_persona_cq_generator_v2.py
│   ├── paper_persona_cq_validator_v2.py
│   ├── paper_persona_cq_refiner_v2.py
│   ├── paper_metadata_extractor.py
│   ├── cq_dedup_agent.py
│   ├── cq_generalizer_agent.py
│   ├── cq_validator_agent.py
│   └── cq_consolidation_agent.py
│
├── workflows/                    # LangGraph flow definitions
│   ├── paper_persona_cq_v2_flow.py
│   ├── cq_dedup_flow.py
│   ├── cq_generalize_flow.py
│   ├── cq_validate_flow.py
│   └── cq_consolidate_flow.py
│
├── utils/
│   ├── llm_client.py             # Ollama client + LLMNetworkError
│   ├── rolling_state.py          # Bounded rolling state (merge + dedup + caps)
│   ├── state_compactor.py        # State compaction for prompt budgets
│   ├── embeddings.py             # BGE-small embedder wrapper
│   ├── vectorstore.py            # FAISS wrapper
│   ├── cq_evaluator.py           # Reference-free 4-family evaluation
│   ├── pdf_extractor.py          # PDF → plain text
│   ├── rich_pdf_extractor.py     # PDF stats (pages, figures, tables)
│   └── model_config.py           # Model name resolution from config/models.json
│
├── config/
│   ├── models.json               # Model name + endpoint per pipeline role
│   ├── domain_priors.json        # Static domain knowledge injected at extraction
│   ├── extraction_schema.json    # JSON schema for extracted state items
│   ├── paper_persona_cq_v2_checks.json  # Validation rule thresholds
│   ├── personas.json             # Seven domain persona definitions + Bloom levels
│   ├── cq_archetypes.json        # CQ archetype templates
│   └── cq_few_shots.json         # Few-shot examples for CQ generation
│
└── data/
    └── papers/
        ├── raw/                  # Input PDFs (not committed — see Data section)
        └── v2/
            ├── cq_output/        # Phase A output: per-paper CQ sets
            ├── dedup/            # Step 1 output: deduped CQs per paper
            ├── generalized/      # Step 2 output: cross-paper generalised CQs
            │   └── by_archetype/
            ├── cq_validation/    # Step 3 output: RAG validation results
            ├── consolidated/     # Step 4 output: consolidated CQs + schema
            ├── evaluation/       # Per-paper reference-free evaluation JSONs
            └── evaluation_summary.json  # Corpus-wide aggregated stats
```

---

## Installation

**Requirements:** Python 3.12,
[uv](https://github.com/astral-sh/uv) (recommended) or pip,
[Ollama](https://ollama.com) accessible at your HPC endpoint.

```bash
# 1. Clone
git clone https://github.com/fusion-jena/Agentic-CQ-Generation.git
cd Agentic-CQ-Generation

# 2. Install dependencies
uv sync
# or: pip install -r requirements.txt

# 3. Configure the Ollama endpoint
# Edit config/models.json → set "base_url" to your Ollama server, e.g.:
#   "base_url": "http://ollama.draco.uni-jena.de:11434"

# 4. Place input PDFs in data/papers/raw/
```

---

## Running the pipeline

Run the five stages in order. Each stage skips already-processed outputs
by default — add `--overwrite` to re-run.

```bash
# Phase A - per-paper CQ generation
python main_paper_persona_cq_v2.py

# Phase B - Step 1: intra-paper deduplication
python main_cq_dedup.py

# Phase B - Step 2: cross-paper generalisation
python main_cq_generalize.py

# Phase B - Step 3: RAG-based validation
python main_cq_validate.py

# Phase B - Step 4: consolidation + ontology schema
python main_cq_consolidate.py
```

To process a single paper in Phase A (useful for testing):

```bash
python main_paper_persona_cq_v2.py Jacobs2007
```

---

## Evaluation metrics

Phase A closes with a **reference-free evaluation** - the deliverable is the
set of questions, not their answers, so the primary metric is *question*
faithfulness rather than answer groundedness.

Four metric families form the composite score
(weights shown in parentheses):

### 1. Question Faithfulness (weight: 0.44) - primary metric

Measures whether the question text aligns semantically with content
actually present in the paper.

For each question, the maximum cosine similarity between the question
embedding and all retrieved passage embeddings is computed.
Questions scoring ≥ 0.70 are "well-faithful", ≥ 0.55 "weakly faithful",
< 0.55 "off-paper". The family score is:

```
score = (n_well_faithful × 1.0 + n_weak × 0.5) / n_questions
```

A score of 1.0 means every question is semantically grounded in the paper.
Off-paper questions (score < 0.55) are the primary hallucination signal.

### 2. Coverage (weight: 0.38)

Measures what fraction of the extracted knowledge state is referenced
by at least one competency question.

Coverage is computed across three sub-categories:

```
score = 0.5 × concept_coverage + 0.3 × property_coverage + 0.2 × relation_coverage
```

where `concept_coverage = |concepts referenced by ≥1 CQ| / |total concepts|`,
and similarly for properties and relations. A low coverage score means the
question set touches only part of what the pipeline extracted from the paper.

### 3. Diversity (weight: 0.12)

Measures how semantically distinct the questions are from one another.

```
score = 1 - mean_pairwise_cosine(all question embeddings)
```

A pair of questions with cosine ≥ 0.92 is counted as a near-duplicate.
Lower diversity indicates redundant questions that should be merged or removed.

### 4. Triplifiability (weight: 0.06)

A heuristic measuring whether each question asks about a
subject-predicate-object relationship between extracted entities,
making it directly usable for ontology construction.

A question scores 1.0 if both (a) its phrasing matches a triple pattern
(e.g. "how does X affect Y", "which X did Y use") AND (b) at least one
extracted concept is named. 0.5 if only one condition holds. 0.0 if neither.

```
score = mean over questions of {1.0, 0.5, 0.0}
```

### Composite score

```
overall = weighted_mean(coverage, faithfulness, diversity, triplifiability)
        = 0.38×coverage + 0.44×faithfulness + 0.12×diversity + 0.06×triplifiability
```

### Corpus results (14-paper run)

| Metric | Mean | SD | Range |
|---|---|---|---|
| Overall score | 0.69 | 0.03 | 0.62-0.76 |
| **Question faithfulness** | **0.98** | **0.01** | **0.96-1.00** |
| Coverage | 0.48 | 0.08 | 0.32-0.65 |
| Diversity | 0.29 | 0.02 | 0.25-0.32 |
| Triplifiability | 0.68 | 0.04 | 0.63-0.76 |

---

## Reproducing the results

```bash
# Print per-paper table + corpus means
python aggregate_eval_v2.py

# Also save to JSON
python aggregate_eval_v2.py --out data/papers/v2/evaluation_summary.json
```

The pre-computed `evaluation_summary.json` from our 14-paper run is
included at `data/papers/v2/evaluation_summary.json`.

**Narrowing funnel (corpus-wide):**

```
536 per-paper CQs  →  490 generalised CQs  →  120 consolidated domain CQs
                                                    (77.6% reduction, 11 LLM calls)
```

---

## Data

The 14 input papers cover supercritical-CO₂ foaming of amorphous polymers
and blends. The corpus was curated by a domain expert within the
[COIN project](https://www.coin-project.eu) (group C1).

**Raw PDFs are not committed** to this repository due to publisher copyright.
Obtain them via DOI through institutional library access.
All pipeline outputs are included and fully versioned.

| Paper ID | Reference |
|---|---|
| Bärwinkel2016 | Bärwinkel et al., *Adv. Eng. Mater.* (2016) |
| Costeux2013 | Costeux et al., *J. Cell. Plast.* (2013) |
| Costeux2014 | Costeux et al., *J. Cell. Plast.* (2014) |
| Di Maio2018 | Di Maio et al., *Prog. Polym. Sci.* (2018) |
| Dumon2012 | Dumon et al., *Macromol. Mater. Eng.* (2012) |
| Guo2015 | Guo et al., *Polymer* (2015) |
| Jacobs2007 | Jacobs et al., *Polymer* (2007) |
| Liu2019 | Liu et al., *Polymer* (2019) |
| Otsuka2008 | Otsuka et al., *Macromol. Mater. Eng.* (2008) |
| Reglero2011 | Reglero Ruiz et al., *Macromol. Mater. Eng.* (2011) |
| Ruiz2011 | Ruiz et al., *J. Supercrit. Fluids* (2011) |
| Shah2025 | Shah et al., *Polymer* (2025) |
| Shieh2002 | Shieh & Liu, *J. Supercrit. Fluids* (2002) |
| Wu2021 | Wu et al., *Polymer* (2021) |

---

## Reproducibility

The pipeline runs entirely on open models served via the university's
HPC infrastructure (Ollama on the Draco endpoint, University of Jena).
Exact model tags are specified in `config/models.json`.
No proprietary API access is required.

## License

Code: [Apache License 2.0](LICENSE)
Data outputs (CQs, evaluation): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Raw PDFs: not redistributed, obtain via publisher DOIs.