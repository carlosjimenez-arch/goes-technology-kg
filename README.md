# GOES Technology Knowledge Graph

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)](https://docs.astral.sh/uv/)

## Contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Curriculum proposals](#curriculum-proposals)
- [Evidence pipeline](#evidence-pipeline)
- [Architecture](#architecture)
- [Validation](#validation)
- [MLOps and DataOps](#mlops-and-dataops)
- [Status and limitations](#status-and-limitations)

## Overview

A reproducible foundation for developing a Technology curriculum for El Salvador, grades 2–6. Technology includes design, technical systems, computational thinking, digital citizenship, and data and AI literacy.

The repository implements source admission, evidence extraction, vector indexing, recorded model proposal/review/revision, graph snapshots and deterministic curriculum allocation. Outputs are research proposals; documentary alignment is not classroom validation. The distribution is `goes-tech-kg`; Python imports use `goes_tech_kg`.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
uv python install 3.13
uv sync --locked
uv run --locked goes-tech --help
make check
```

uv manages Python, the virtual environment, dependencies and package builds. `uv.lock` is versioned. No separate pip or Conda setup is required. Optional settings use `.env` through pydantic-settings; see [.env.example](.env.example).

Restore the pinned corpus and run an offline build:

```sh
uv run --locked goes-tech restore
uv run --locked goes-tech build
```

Restoration requires network access and verifies each document's SHA-256. Build reuses recorded embeddings and never silently fetches missing inputs. **Exit code 2 means curriculum-source coverage is incomplete**, even when index generation succeeds.

For explicit embedding generation, install the optional dependencies and provide the pinned local model:

```sh
uv sync --locked --extra embeddings
uv run --locked --extra embeddings goes-tech build \
  --model-path /absolute/path/to/model/snapshot \
  --generate-embeddings
```

The model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, revision `4328cf26390c98c5e3c738b4460a05b95f4911f5`. Its file hashes are recorded in [model.json](data/processed/embeddings/model.json). Loading is local-only.

## Curriculum proposals

The [revised proposal](data/processed/curriculum_releases/v2/curriculum_proposal.html) contains observable skills, teaching sequences, assessment tasks and criteria, accessibility, equipment, teacher preparation, exact source locators and remaining critique. Open the HTML locally or use the [CSV](data/processed/curriculum_releases/v2/curriculum_proposal.csv).

- [First graph and curriculum](data/processed/curriculum_releases/v1/): all original candidate units are retained; invalid evidence is quarantined from the graph, and an incomplete draft cannot receive a valid schedule.
- [Revised graph and curriculum](data/processed/curriculum_releases/v2/): the deliverable for this iteration, with human review still required.
- [Version comparison](data/processed/curriculum_releases/comparison.json), [semantic graph diff](data/processed/curriculum_releases/graph_diff.json) and [pedagogical review](evaluation/curriculum_release_review.yaml).
- [Manifest](data/processed/curriculum_releases/manifest.json): code, input, response and artifact hashes. [Progression links](data/processed/curriculum_releases/progression_links.json) are teaching hypotheses, not asserted necessary prerequisites.

The map declares 25 units: five strands in each grade. The revised graph contains 90 micro-skills and 144 edges. The proposed core requires 19–26.33 clock hours per grade; it is not a complete 160-hour annual course. The [recorded run summary](evaluation/curriculum_run_summary.json) accounts for 144 model responses, including failed and superseded attempts. The final model audit retains 28 major flags, which are uncalibrated review findings, not a quality score. Two unnecessary prerequisite hypotheses were removed through [explicit publication curation](config/curriculum_curation.json); other evidence and progression concerns remain visible.

The primary configuration is an **unverified 160-hour standalone capacity scenario**, with 40/80-hour sensitivity. Actual planned time is reported separately; unused capacity is not filled. `standalone`, `transversal` and `hybrid` allocation share the same deterministic ledger. Borrowing requires a host subject, capacity and proposed indicator. A feasible ledger does not certify a school's timetable, equipment or cross-subject readiness.

After restoring and building the pinned evidence, replay the workflow without credentials:

```sh
uv run --locked python scripts/generate_curriculum_proposals.py
uv run --locked python scripts/repair_curriculum_evidence.py
uv run --locked python scripts/apply_curriculum_editorial_review.py
uv run --locked python scripts/audit_curriculum_revision.py
uv run --locked python scripts/build_curriculum_release.py
```

Generation resumes verified completed records. Use a separate experiment directory for changed input plans. New model answers require the explicit `--online` option and authorized GCP credentials; a replay miss never silently calls a provider. The self-contained real-source fixture is exercised by `tests/integration/test_curriculum_release.py` without restoring the full corpus.

## Evidence pipeline

```text
discover → license gate → fetch → hash + manifest
         → parse → normalize → curriculum-unit chunks → embed → index
```

Discovery reads a reviewed [source catalogue](corpus/sources.yaml). Every source declares `provenance: external`; the gate rejects `generated` so pipeline output can never re-enter the corpus. An allowed publisher does not automatically grant admission: each resource needs recorded evidence of an open license or official publication. Rejections and acquisition failures retain their reasons. `goes-tech fetch` deliberately acquires a new snapshot and archives the previous manifest.

PDF parsing preserves page and unit boundaries. Each source declares its text extraction mode: layout keeps horizontal positions, plain follows the content stream and is used for the Salvadoran programme, whose three-column tables layout mode interleaved line by line. HTML parsing preserves complete activities and section/DOM locators. Reversible newline normalization maps evidence intervals back to extracted source text. Citations resolve through document hash, paragraph ID and character offsets. Unknown textless pages fail unless a digest-pinned manual review is recorded.

Long activities use token windows for embedding only; cited units remain complete. Search uses an exact float32 index with stable ordering and tie-breaking.

## Architecture

| Location | Responsibility |
| --- | --- |
| `src/goes_tech_kg/schemas/` | Pydantic contracts: graph and curriculum invariants, requests, outputs, plans, reports, golden records and scope |
| `src/goes_tech_kg/corpus/` | Acquisition, parsing, traceability, embeddings and indexing |
| `src/goes_tech_kg/graph/` | Semantic graph-version diffs |
| `src/goes_tech_kg/eval/` | Confidence calibration, prompt-experiment harness, golden metrics, agreement statistics, report tables |
| `evaluation/` | Golden records and protocol, judge calibration sample and key, baseline tables |
| `src/goes_tech_kg/llm/` | Vertex AI boundary: canonical requests, immutable replay records, replay-first gateway |
| `src/goes_tech_kg/prompts/` | Versioned Spanish prompts (decomposition variants, curricular judge); version derives from text |
| `src/goes_tech_kg/retrieval/` | Chunk store, context references and deterministic context assembly |
| `src/goes_tech_kg/curriculum/` | Proposal/review workflow, immutable-response verification, graph compilation, three-mode allocation, external snapshot checks and HTML/CSV publication |
| `corpus/`, `decisions/` | Reviewed sources and validated architecture decisions |
| `data/manifests/`, `data/processed/` | Acquisition records and reproducible derived artifacts |
| `tests/` | Contract, property, real-source, VCR and performance tests |

Skills are context-free archetypes; context enters during curricularization. A micro-skill's `confidence` is never authored: it is computed from judge verdicts weighted by their measured agreement with humans, citation support and revision count, then mapped through a calibration table fitted on unaided human labels (`eval/calibration.py`). Without calibrated judges or a table the value stays null, and `verify_confidence` fails any stored value that does not reproduce. `VOLATILE` tool operations live outside graph snapshots and cannot become prerequisites. Material downgrades must disclose lost practical performance and its instructional bridge.

Budgets support `standalone`, `transversal` and `hybrid` contracts with explicit capacities and host-subject accounting. The release config uses 160 hours only as a nonofficial scenario. Cross-subject dependencies include repository, snapshot hash, node ID and declared availability grade; a hash-pinned science export supports grade-conflict detection, but its labels remain provisional. Release external-knowledge requirements are unresolved until linked to reviewed sibling IDs.

Decisions [0010](decisions/0010-system-design.yaml) and [0011](decisions/0011-evaluation-design.yaml) propose the production framing and the evaluation design. Structure follows the sibling repositories. Context follows the available science sibling; Mathematics uses a different vocabulary and needs an explicit adapter. Decisions [0004](decisions/0004-contracts-and-compatibility.yaml) and [0005](decisions/0005-ingestion-and-replay.yaml) explain compatibility and ingestion trade-offs.

## Validation

```sh
make static-check   # Lock, Ruff, strict typing, offline tests and ≥75% coverage
make benchmark      # Real benchmarks; fail above 20% regression
make check          # Both gates
uv build            # Source distribution and wheel
```

Tests do not mock domain logic. Three attributed real excerpts cover PDF and HTML; VCR replays a real HTTP failure. Hypothesis checks canonical identities, graph ordering, text conservation and offsets. Tests also exercise volatile-node rejection, material tiers, budgets, grade conflicts and source/vector corruption.

Hosted CI is suspended by decision [0012](decisions/0012-ci-suspension.yaml); `make static-check` runs the portable checks locally. Performance thresholds are enforced by `make check` against a reviewed machine-specific [baseline](tests/benchmarks/baseline.json), not compared across unrelated hosted runners. The benchmark workloads are one real PDF page, a real HTML curriculum excerpt, a seeded 1,000 × 768-vector index, and graph compilation/allocation for a three-skill unit from real recorded evidence. Migration measurements are recorded in [desktop-migration-validation.json](research/desktop-migration-validation.json). Current counts are reported by `make static-check`.

| Local benchmark | Median |
| --- | ---: |
| PDF parsing | 13.665 ms |
| HTML parsing | 0.363 ms |
| Index construction | 0.194 ms |
| Evidence-backed graph compilation (three skills) | 0.285 ms |
| Complete-unit allocation | 0.020 ms |

## MLOps and DataOps

Version source manifests, reviewed configuration, decisions, processed locators, embedding replays and `uv.lock` together. Raw/interim corpus, environments, caches, checkpoints, credentials and local exports are ignored. The bounded golden excerpts and sanitized HTTP cassette are explicit testing exceptions. Official publication is an admission basis, not a redistribution license.

Build reports record source-tree, manifest, lock and model hashes, runtime versions and seed. Acquisition timestamps remain pinned. Offline replay uses recorded vectors; cross-platform numerical regeneration is not guaranteed. Earlier Python 3.12 reports are historical evidence, not current-runtime certification.

LLM calls happen only through `goes_tech_kg.llm`. A request is canonical JSON over prompt version, model, location, generation settings, output schema, replicate and the digest of the rendered user content; the rendered text is never stored because it embeds licensed corpus passages, only its locators are. Responses live under `data/processed/llm_responses/<sha256>.json`, credential-free and never overwritten. Replay never calls a model: `scripts/run_prompt_experiment.py <plan>` reproduces a report with zero acquisitions, and `--online` is the separately authorized acquisition mode (ADC, with the active gcloud account as fallback). Decision [0013](decisions/0013-llm-provider-and-prompt-selection.yaml) records the provider choice, the pre-declared selection criteria and the measured prompt/model tables; `scripts/summarize_prompt_experiment.py` prints them from a recorded report. Credentials belong in local `.env` files and must never appear in artifacts or logs.

### Model workflow and retrieval evaluation

Decision [0015](decisions/0015-curriculum-release-workflow.yaml) uses Gemini 2.5 Flash for proposals/revisions and Gemini 2.5 Pro for criticism. Exact citations, schema validity, graph constraints and accounting are deterministic gates. Narrow format corrections are logged; unsupported content requires a recorded repair. Reviewers cannot waive human validation, and model ratings never become calibrated confidence.

The earlier [prompt experiments](data/processed/prompt_experiments/) compared four models and three prompts. Source-authored golden labels and model judges disagreed on rankings. Repeated temperature-zero calls were byte-identical in only four of eight pairs, so reproducibility depends on response replay. See [decision 0013](decisions/0013-llm-provider-and-prompt-selection.yaml), [golden evaluation](evaluation/golden/) and [judge calibration protocol](evaluation/judges/) for the evidence and uncompleted human validation.

A [frozen retrieval comparison](evaluation/retrieval-v2/report.json) evaluated local multilingual MPNet against Vertex `gemini-embedding-001`: 14 queries, 45 candidate paragraphs, eight development and six held-out queries. Both achieved held-out MRR and recall@3 of 1.0; the predeclared improvement threshold was not met, so the local default remains. This small, provisionally labeled pool cannot establish full-corpus superiority. The initial [diagnostic run](evaluation/retrieval/) omitted indicator-based labels; the corrected v2 reports that change. Curriculum generation uses pinned source-diverse packets, so this experiment does not silently alter its evidence.

## Status and limitations

The recorded corpus contains 104 indexed units, eight foreign resources per target grade from five countries, and three international references counted separately. These counts describe resource coverage, not validated instructional quality or equivalence between national grades.

**Phase 3 remains incomplete.** The current Salvadoran Computer Science programme returned 404/403; the older programme cannot replace that vertical anchor. The UNESCO source is an overview, not the complete competency-framework PDF. OCR, authoritative sibling-grade integration, optional-node optimization and school deployment validation remain incomplete. The implemented allocator covers the complete declared map; it does not claim to optimize alternative learning paths.

See the [ingestion report](data/processed/ingestion/report.json), [research findings](research/findings.yaml) and [replay evidence](research/replay-validation.json) for provenance and unresolved gaps.
