# GOES Technology Knowledge Graph

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)](https://docs.astral.sh/uv/)

## Contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Evidence pipeline](#evidence-pipeline)
- [Architecture](#architecture)
- [Validation](#validation)
- [MLOps and DataOps](#mlops-and-dataops)
- [Status and limitations](#status-and-limitations)

## Overview

A reproducible foundation for developing a Technology curriculum for El Salvador, grades 2–6. Technology includes design, technical systems, computational thinking, digital citizenship, and data and AI literacy.

The repository implements typed knowledge contracts, semantic graph diffs, source admission, evidence extraction and vector indexing. Curriculum inference and scheduling are future work. The distribution is `goes-tech-kg`; Python imports use `goes_tech_kg`.

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
| `src/goes_tech_kg/schemas/` | Pydantic contracts and graph/curriculum invariants |
| `src/goes_tech_kg/corpus/` | Acquisition, parsing, traceability, embeddings and indexing |
| `src/goes_tech_kg/graph/` | Semantic graph-version diffs |
| `src/goes_tech_kg/eval/` | Confidence calibration, prompt-experiment harness, golden metrics, agreement statistics, report tables |
| `evaluation/` | Golden records and protocol, judge calibration sample and key, baseline tables |
| `src/goes_tech_kg/llm/` | Vertex AI boundary: canonical requests, immutable replay records, replay-first gateway |
| `src/goes_tech_kg/prompts/` | Versioned Spanish prompts (decomposition variants, curricular judge); version derives from text |
| `src/goes_tech_kg/retrieval/` | Deterministic context assembly from chunk locators |
| `src/goes_tech_kg/curriculum/` | Declared cross-subject grade-conflict checks |
| `src/goes_tech_kg/agents/` | License-gate pipeline node |
| `corpus/`, `decisions/` | Reviewed sources and validated architecture decisions |
| `data/manifests/`, `data/processed/` | Acquisition records and reproducible derived artifacts |
| `tests/` | Contract, property, real-source, VCR and performance tests |

Skills are context-free archetypes; context enters during curricularization. A micro-skill's `confidence` is never authored: it is computed from judge verdicts weighted by their measured agreement with humans, citation support and revision count, then mapped through a calibration table fitted on unaided human labels (`eval/calibration.py`). Without calibrated judges or a table the value stays null, and `verify_confidence` fails any stored value that does not reproduce. `VOLATILE` tool operations live outside graph snapshots and cannot become prerequisites. Material downgrades must disclose lost practical performance and its instructional bridge.

Budgets support `standalone`, `transversal` and `hybrid` contracts with explicit capacities and host-subject accounting. There is **no assumed 160-hour allocation**. Cross-subject dependencies include repository, snapshot hash, node ID and declared availability grade; authoritative sibling-export adapters remain necessary.

Decisions [0010](decisions/0010-system-design.yaml) and [0011](decisions/0011-evaluation-design.yaml) propose the production framing and the evaluation design. Structure follows the sibling repositories. Context follows the available science sibling; Mathematics uses a different vocabulary and needs an explicit adapter. Decisions [0004](decisions/0004-contracts-and-compatibility.yaml) and [0005](decisions/0005-ingestion-and-replay.yaml) explain compatibility and ingestion trade-offs.

## Validation

```sh
make static-check   # Lock, Ruff, strict typing, offline tests and ≥75% coverage
make benchmark      # Real benchmarks; fail above 20% regression
make check          # Both gates
uv build            # Source distribution and wheel
```

Tests do not mock domain logic. Three attributed real excerpts cover PDF and HTML; VCR replays a real HTTP failure. Hypothesis checks canonical identities, graph ordering, text conservation and offsets. Tests also exercise volatile-node rejection, material tiers, budgets, grade conflicts and source/vector corruption.

Hosted CI is suspended by decision [0012](decisions/0012-ci-suspension.yaml); `make static-check` runs the portable checks locally. Performance thresholds are enforced by `make check` against a reviewed machine-specific [baseline](tests/benchmarks/baseline.json), not compared across unrelated hosted runners. The benchmark workloads are one real PDF page, a real HTML curriculum excerpt, and a seeded 1,000 × 768-vector index. Migration measurements are recorded in [desktop-migration-validation.json](research/desktop-migration-validation.json). Current counts are reported by `make static-check`.

| Local benchmark | Median |
| --- | ---: |
| PDF parsing | 13.665 ms |
| HTML parsing | 0.363 ms |
| Index construction | 0.194 ms |

## MLOps and DataOps

Version source manifests, reviewed configuration, decisions, processed locators, embedding replays and `uv.lock` together. Raw/interim corpus, environments, caches, checkpoints, credentials and local exports are ignored. The bounded golden excerpts and sanitized HTTP cassette are explicit testing exceptions. Official publication is an admission basis, not a redistribution license.

Build reports record source-tree, manifest, lock and model hashes, runtime versions and seed. Acquisition timestamps remain pinned. Offline replay uses recorded vectors; cross-platform numerical regeneration is not guaranteed. Earlier Python 3.12 reports are historical evidence, not current-runtime certification.

LLM calls happen only through `goes_tech_kg.llm`. A request is canonical JSON over prompt version, model, location, generation settings, output schema, replicate and the digest of the rendered user content; the rendered text is never stored because it embeds licensed corpus passages, only its locators are. Responses live under `data/processed/llm_responses/<sha256>.json`, credential-free and never overwritten. Replay never calls a model: `scripts/run_prompt_experiment.py <plan>` reproduces a report with zero acquisitions, and `--online` is the separately authorized acquisition mode (ADC, with the active gcloud account as fallback). Decision [0013](decisions/0013-llm-provider-and-prompt-selection.yaml) records the provider choice, the pre-declared selection criteria and the measured prompt/model tables; `scripts/summarize_prompt_experiment.py` prints them from a recorded report. Credentials belong in local `.env` files and must never appear in artifacts or logs.

### Prompt and model selection (decision 0013)

Three decomposition prompt variants and four Vertex AI Gemini models were run on twelve cases (five official technology-axis units and paraphrases, three computational thinking, one digital citizenship, two malformed, one injection), every proposal judged by two independent models. Eligibility gates were declared before inspection: no unsupported quote on any valid case, correct refusals, no injected content, schema adherence at least 0.95. Judge scores rank eligible cells but do not certify quality; judge-human agreement is still pending.

| Eligible cell (v4, parse/1.1 chunks) | Judge mean | Worst cell | Quote exactness | Indicator coverage | Micro-skills per case |
| --- | ---: | ---: | ---: | ---: | ---: |
| guided-v3 × gemini-3.1-pro-preview | 0.964 | 0.78 | 1.00 | 0.84 | 3.6 |
| guided-v3 × gemini-2.5-flash | 0.906 | 0.60 | 0.96 | 0.78 | 5.9 |

Against the provisional golden set (`evaluation/golden/`, seven records authored from the sources, not yet human-reviewed), the picture changes: B0, the official programme, covers 57 percent of golden micro-skills and no prerequisite edge; B1 with `gemini-2.5-flash` recovers 91 percent of micro-skills and 79 percent of prerequisite edges at 88 percent precision; B1 with `gemini-3.1-pro-preview` recovers 93 percent of micro-skills but only 42 percent of edges. The judges ranked the two the other way round, which is why decision 0013 refuses to select on judge score alone. B2 and S are not built yet; `scripts/compare_baselines.py` regenerates the table from recorded responses. Decision 0014 fixes which programme indicators count as Technology (`config/technology_scope.yaml`), so coverage ceilings are explicit.

Four findings matter more than the ranking. Layout-mode PDF extraction interleaved the programme's table columns line by line, so faithful quotes were not verbatim substrings; switching the programme to plain extraction (parse/1.1) took strict quote exactness from about 0.6 to 1.0. Injecting programmatic quote verification into the judge prompt raised inter-judge verdict agreement from 0.56 to 0.85. Few-shot demonstrations helped two models and broke two others, so they are validated per model. Identical requests at temperature 0 and seed 0 returned byte-identical responses in only 4 of 8 pairs, so reproducibility rests on recorded responses, never on provider settings. Human calibration materials for the judges are under `evaluation/judges/`; judges gate nothing until the maintainer rates them. Reports replay offline from recorded responses; runs recorded before parse/1.1 are reproducible only at their recording commit.

## Status and limitations

The recorded corpus contains 104 indexed units, eight foreign resources per target grade from five countries, and three international references counted separately. These counts describe resource coverage, not validated instructional quality or equivalence between national grades.

**Phase 3 remains incomplete.** The current Salvadoran Computer Science programme returned 404/403; the older programme cannot replace that vertical anchor. The UNESCO source is an overview, not the complete competency-framework PDF. OCR, authoritative sibling-grade integration, curriculum generation and the solver are not implemented.

See the [ingestion report](data/processed/ingestion/report.json), [research findings](research/findings.yaml) and [replay evidence](research/replay-validation.json) for provenance and unresolved gaps.
