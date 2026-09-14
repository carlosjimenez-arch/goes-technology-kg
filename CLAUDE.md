# CLAUDE.md — working contract for goes-tech-kg

## 0. Scope and current state

The product is a deterministic, auditable pipeline that computes a Technology
curriculum for El Salvador, grades 2 through 6, from a high-level skill map.
Technology means design, technological systems, artefacts, computational thinking
and digital citizenship, not instrumental computer literacy.

Conversation with the maintainer is in Spanish, using voseo. Repository prose,
code, identifiers, docstrings, comments, logs, commits and tests are in English;
LLM prompt bodies are in Spanish. Source quotations are evidence data and retain
their original language; they are not an exception for Spanish code or comments.

The official repository is `goes-technology-kg` at
`~/Desktop/goes-technology-kg`, with working branch `feat/context`.
The distribution and import names remain `goes-tech-kg` and `goes_tech_kg` for
compatibility. Python 3.13 and uv are required. The maintainer authorized transfer,
README revision, dependency migration and Git delivery. Phase 3 remains incomplete
while the current Salvadoran Computer Science programme is unavailable. See README
and the machine-readable ingestion report for implemented behavior and limitations.

The instruction to write this file conflicts literally with the rule that
`README.md` is the only Markdown file. The explicit request authorizes this
single exception: `CLAUDE.md` is the agent contract; `README.md` is the only
product documentation Markdown file. No other Markdown files are allowed.

## 1. Four roles

Act simultaneously as:

- Technology-education curriculum designer: prioritize design processes,
  systems, materials, artefacts and their social and environmental implications.
- Computational-thinking and learning-progressions researcher: distinguish
  concepts, practices and perspectives; describe observable progressions and
  distinguish supported learning claims from hypotheses.
- Graph-theory researcher: protect prerequisite semantics, acyclicity,
  reachability, sequencing constraints and computational tractability.
- AI software engineer: deliver executable transformations, validated contracts,
  evidence, tests and reproducible artifacts. A chat-only result is not a result.

When these roles pull in different directions, explicitly name the role behind
each position, explain the tension, and present options to the maintainer.
Do not silently choose between pedagogy, evidence and graph feasibility.
Record any accepted trade-off in `decisions/` before implementing it.

## 2. Sibling inspection and compatibility

The following local sources were inspected for this contract:

- `../goes-math-kg/README.md`, `pyproject.toml`, `.gitignore`,
  `.github/workflows/ci.yml`, and `schemas/learning_graph_contract_v1.json`.
- `../goes-linguistics-kg/README.md`, `pyproject.toml`, `.gitignore`,
  `.github/workflows/ci.yml`, test layout, and
  `src/cgl-core/tests/test_edge.py`.
- `../goes-natural-science-kg/CLAUDE.md`.

`../goes-science-kg` does not exist. The available science sibling is
`goes-natural-science-kg`; its contract is a design reference, not proof of an
implemented science pipeline. These are targeted inspections, not a full audit.
Claims from older contracts about sibling defects must be rechecked against
actual files before being reported as current findings.

Inherit the src-layout, underscore Python package names, `uv`, `pyproject.toml`,
lockfile, Typer CLI, Pydantic contracts, pytest, absolute imports, Ruff with
100-character lines, Makefile entrypoints and provenance-backed graph claims.
Use distribution `goes-tech-kg`, package `goes_tech_kg`, CLI `goes-tech`.
Python 3.13 is the proposed baseline, matching the science contract.

### Explicit differences and corrections

| Observed sibling convention | Technology contract | Reason / proposed correction |
|---|---|---|
| Math uses `tests/unit/` and `tests/integration/`; linguistics has root tests plus workspace package suites | Keep unit/integration separation; add golden, cassettes and benchmarks | Preserve recognizable layout and satisfy the requested test types |
| Math and linguistics keep payloads in GCS and ignore broad data/schema paths | Commit processed data, graph artifacts, manifests and fixtures | User requires auditability from Git; do not copy broad ignore rules |
| Both ignore `CLAUDE.md`; science proposes keeping it | Keep this explicitly requested contract versioned | A checkout must contain its working rules |
| Both CI workflows use reduced gates; linguistics overrides coverage options and permits artifact-related skips | Full offline test gate with committed replay inputs and enforced coverage | Reduced gates do not establish reproducibility of the whole pipeline |
| Math config limits mypy to selected modules and permits complexity 44 | Propose strict typing throughout and complexity 10 from the start | Avoid importing legacy debt; finalize tooling in Phase 0 |
| Linguistics contains Spanish test names and a narrower Ruff rule set | English names in all code, scripts and tests | User's language rule; check filenames and identifiers as well as docstrings |
| Science places replay cache under ignored `.cache/` while requiring versioning | Separate disposable cache from committed LLM response records | Ignored files cannot be the sole reproduction source |
| Science fixes a 160-hour budget | Parameterized policy with three modes; default remains unverified | Technology's timetable must be established independently |
| Existing contracts scatter Pydantic models across `schemas.py` files | Canonical models in `src/goes_tech_kg/schemas/` | Follow the user's explicit `schemas/` requirement; avoid duplicate contracts |

These are declared proposals or user-required differences, not completed
migrations. Do not modify siblings in this task. Other framework, identifier or
curricular trade-offs require a decision with alternatives before code.

## 3. Hard invariants

### Invariant 1 — Language and documentation

Enforce the language rules in Section 0 across `src/`, `tests/` and `scripts/`,
including filenames. Prompts have English constant names and Spanish bodies.
Keep architecture decisions in `decisions/NNNN-<slug>.yaml`, validated by a
Pydantic decision model. Keep data contracts in the package's `schemas/` directory;
JSON Schema is generated from those models, never a parallel hand-written source.
MLOps and DataOps belong in README sections. The only Markdown files are the
requested `CLAUDE.md` exception and `README.md`.

### Invariant 2 — Meaningful tests

Never mock domain logic. Network and LLM boundaries may use only recordings of
real responses with VCR; invented response strings are not recordings. Scrub
credentials and irrelevant transport metadata while preserving response payloads.
Use committed golden datasets with SHA-256 manifests, Hypothesis properties and
curricular constraint tests. Small synthetic domain inputs are allowed; they must
exercise real domain functions rather than pretend to be real source evidence.

Test prerequisite acyclicity, unique identifiers, valid endpoints, prerequisite
closure of selected skills, nondecreasing grades and progression stages,
permutation-independent output, full input-map coverage and all budget modes.
Test infeasible inputs and malformed evidence, not just happy paths.
A test that passes with an empty function body does not establish correctness.
CI runs the complete suite offline without credentials. Missing replay records
or required golden data fail; they do not become skips or network downloads.

### Invariant 3 — Performance and state

Use `numpy`/`polars` for tabular transformations and `scipy.sparse` for adjacency
and reachability. Do not substitute dense all-pairs structures without measuring
and recording the trade-off. Use classes only for state with invariants to
protect, such as the graph store, solver or budget ledger; other transformations
are typed pure functions.

Use `pytest-benchmark` for graph, corpus and solver workloads. Commit baselines
and explicitly load the reference baseline in CI. A mean regression greater
than 20% fails the build; an absent or incomparable baseline also fails. Record
runner characteristics, workload size, seeds and dependency versions. Baseline
updates require review and must not silently absorb a regression.

### Invariant 4 — Git and secrets

Before future Git initialization, create `.gitignore` excluding `data/raw/`,
`data/interim/`, `.cache/`, `.venv/`, PDFs, LangGraph checkpoints, local databases,
`.env*` except `.env.example`, credentials, private keys and tooling caches.
Do not initialize Git during this documentation-only task.

Commit `data/processed/`, `data/graph/`, `data/manifests/`, replay records, golden
fixtures and benchmark baselines. Do not commit the source corpus, extracted full
source text, secrets or provider authorization headers through those exceptions.
Test both effective ignore behavior with `git check-ignore` and tracked/indexed
payloads so forced additions cannot bypass policy.

Use `.env` and `pydantic-settings`; no credentials in code or committed config.
Use one settings tree with prefix `GOES_TECH_KG_` and nested delimiter `__`.
Provider-standard names may be explicit aliases. Do not copy sibling `.env`
contents or expose values while inspecting configuration. A local `.env` is
allowed, but must remain ignored. `.env.example` contains names and comments
without secret values.

### Invariant 5 — Corpus and licenses

The committed corpus manifest is `data/manifests/corpus.jsonl`. Records include
resource ID, URL, license or official-publication basis, verification evidence,
SHA-256, byte count, acquisition date, status and rejection reason. Accepted
sources must have a verified digest before downstream use. For sources rejected
before acquisition, digest/bytes/acquisition date are null, never invented;
record the inspection date separately.

Admit only verifiably open-licensed sources or verified official publications
under the user's policy. Official status does not imply an open license: record
these bases separately. Unverifiable eligibility means `rejected` with a reason;
do not ingest that source. Phase 2 fixes a canonical license/domain policy.
Re-download accepted corpus files from the manifest and verify their digests;
changed or unavailable sources cause explicit errors, not replacement evidence.

Follow the math anti-circularity boundary: curricular expectations provide input
and alignment targets, not independent proof that generated skills are grounded.
Learning claims require source locators, evidence and provenance. Curriculum
presentation order alone never establishes a necessary prerequisite.

### Invariant 6 — Determinism and LLM replay

Same input + same commit + same seed must produce byte-identical artifacts.
Version the skill map, accepted configuration, budget policy, environment lock
and every response used by an LLM stage. An input includes its pinned source
manifest and recorded evidence; remote mutable state is not a hidden input.

Cache keys are SHA-256 over canonical serialization including `prompt_version`
and input, plus prompt text, model identifier, output schema version, generation
settings, reasoning budget and cache format version. Hash every effective
request field that can change the result. Temperature zero alone is not a
reproducibility guarantee.

Store immutable, credential-free response records under
`data/processed/llm_responses/<sha256>.json`, with request and response digests.
`.cache/llm/` is only a disposable acceleration layer rebuilt from these records.
Do not overwrite an existing request's accepted response. Acquisition of new
responses is a separately authorized online operation; replay never calls a model
on a cache miss. Ensure records do not smuggle corpus payloads or secrets into Git.

Canonicalize JSON encoding, key order, stream row order, newline convention,
numeric serialization and Unicode handling. Sort inputs before optimization;
resolve equally optimal solutions with a documented deterministic tie-break.
Pin solver options, supported runtime and thread counts. Do not claim that a
single-thread setting alone proves deterministic optimization.

Use SHA-256-derived persistent IDs, never Python's randomized `hash()`. Test
under different `PYTHONHASHSEED` values and node/edge permutations. Artifact clocks
and directory labels derive from pinned inputs or a committed epoch, never the
wall clock. Operational timestamps stay outside reproducible payloads.
Published builds require a clean checkout; exploratory builds record `git_dirty`
and a source-tree digest so uncommitted code cannot masquerade as the same commit.

## 4. Budget policy — verify in Phase 1

The requested default is 160 hours per grade, approximately four hours weekly
over 40 weeks. It is a requested scenario, not an accepted national timetable.
Do not assume 40 weeks, session duration, a standalone Technology subject or a
regional comparison is verified. No 160-hour constant belongs in domain code.

Phase 1 must inspect current official El Salvador documents and applicable
reference curricula. Record jurisdiction, grade, subject, effective year, weekly
load, teaching weeks, definition of an hour/session, URL, document digest and
page/table evidence. Distinguish clock hours from pedagogical periods. Report
missing or contradictory evidence rather than converting it into a default.
Reference candidates include Chilean Technology, regional Technology education
frameworks and Brazilian Computing; selection and applicability need evidence.

Deliver `decisions/0002-time-budget-policy.yaml` with the verified comparisons,
options, criteria, proposed default and rationale. Until the maintainer accepts
it, policy status is `pending_verification` and publication is blocked. Explicit
scenario runs may use the requested amount if labeled unverified.

All internal time values are nonnegative integer minutes, per grade:

- `standalone`: allocated minutes cannot exceed that grade's own budget.
- `transversal`: every minute is charged to a host subject and an identified unit
  or indicator. Respect total borrowed capacity and each host's capacity; never
  silently remove mandatory host content to create room.
- `hybrid`: separate own-time and borrowed-time allocations. Each component must
  satisfy its constraints; a minute cannot be counted in both.

`BudgetPolicy` contains mode, grade budgets, host capacities and verified session
and calendar parameters where relevant. Hash the policy into every run manifest.
The ledger records skill, grade, host subject, host unit/indicator, minutes,
justification and evidence ID. Its totals must equal solver allocations.
Shared activities require an explicit accounting rule to prevent double counting.
The solver supports all three modes, prerequisite closure and required coverage;
infeasibility yields a diagnostic and refusal, never an overrun or hidden omission.

## 5. Graph and artifact contracts

Preserve math's base entity names: `curriculum_skill`, `specific_skill`,
`micro_skill`, `indicator`, `representation`, `misconception`. Preserve edge names
`decomposes_to`, `prerequisite_of`, `assesses`, `requires`,
`uses_representation`, `targets_misconception`, `aligns_to`, `supported_by`.
Statuses remain `candidate`, `reviewed`, `published`, `rejected`.
Retain `source` and `target` endpoint names and source locators at interchange.

Proposed Technology extensions are `design_challenge`, `practice`, `concept`,
`host_subject`, and edges `anchored_in`, `applies_practice`, `expresses_concept`,
`hosted_by`. Phase 1 decides these in `0003-graph-vocabulary.yaml`, with an
explicit mapping to sibling contracts. Use a Technology schema family/version;
new enums are not silently compatible with math's existing enum-constrained v1.

Only the prerequisite projection must be a DAG; do not impose acyclicity on all
semantic edge kinds. Host subjects cannot be prerequisite endpoints. Distinguish
necessary dependencies, pedagogical recommendations and observed presentation
order. The exact Technology prerequisite vocabulary is a Phase 1 decision:
`computationally_required` alone may not cover design/material prerequisites.

Every micro-skill has an observable action and success criteria, grade in 2..6,
evidence and a progression reference/stage. Supported progression claims and
hypotheses are distinct. Within a progression, prerequisite direction must not
reduce stage or grade. Do not equate documentary alignment with demonstrated
student mastery. Phase 1 decides strand codes, cognitive fields and ID grammar;
prefer sibling `CS_`, `SS_G`, `MS_G`, `res_` and `e_` patterns, with one validated
scheme per entity and collision checks. Record deviations and migration mappings.

Use JSON for manifests/objects, JSONL for `nodes.jsonl`, `edges.jsonl` and
`budget_ledger.jsonl`, and CSV for human-facing tables. Parquet is permitted in
ignored interim storage. Each contract has `schema_version`; each artifact
manifest records Git SHA, dirty status, seed, runtime/settings/budget digests,
input paths and digests, replay-set digest, artifact hashes and byte counts.
Missing artifact, absent evidence and digest mismatch are distinct failures.

## 6. Target layout — specification only

```text
goes-tech-kg/
  CLAUDE.md  README.md  LICENSE  NOTICE
  pyproject.toml  uv.lock  .python-version  Makefile
  .gitignore  .env.example  .pre-commit-config.yaml
  .github/workflows/ci.yml
  config/goes.yml
  decisions/
  data/
    raw/                         # ignored corpus
    interim/                     # ignored transformations
    manifests/                   # corpus and input manifests
    processed/                   # versioned curriculum and LLM replay records
    graph/                       # versioned graph releases and manifests
  src/goes_tech_kg/
    schemas/                     # canonical Pydantic data and decision contracts
    settings.py
    corpus/
    graph/
    budget/
    solver/
    prompts/
    pipeline/
    cli.py
  scripts/                       # thin entrypoints; domain logic stays in src
  tests/
    unit/
    integration/
    golden/
    cassettes/
    benchmarks/
    conftest.py
```

Keep provider adapters at the boundary. Siblings provide precedents for Vertex
AI via `google-genai`, ADC and LangGraph orchestration. Pin models and prompt
versions in config/catalogs; use different proposer and validator models when
model validation is employed. Provider, solver and orchestration choices with
real trade-offs remain decisions, not implicit new dependencies in this turn.

## 7. Decisions, phases and approval

Decision YAML records have string ID (for example `"0001"`), title, status
(`proposed`, `pending`, `accepted`, `superseded`), date, supersedes, affected
invariants, roles, context, options with pros/cons, criteria, selected option,
rationale, consequences, source evidence and `enforced_by` test references.
Accepted entries require resolvable enforcement references and source evidence
when grounded in external documents. Proposed/pending entries may identify open
questions without inventing an accepted choice or nonexistent completed test.

Before each new phase, present scope, outputs, trade-offs and verification plan
in Spanish and wait for the maintainer's explicit OK. The user's later phase
specifications supersede the original draft numbering:

1. Research: structured findings and decisions 0001–0003.
2. Structure and contracts: no curricular inference or scheduling.
3. Ingestion: licensed sources, complete activities, provenance, embeddings and index.

Phases 2 and 3 are authorized. Later graph inference and solver phases require
new plans and approval. Decisions 0004 and 0005 record implementation trade-offs.
Accepted Phase 1 decisions retain `specified_not_implemented` enforcement entries
for future curriculum/solver research; do not present those entries as passing tests.

If a requirement is ambiguous or erroneous, explain before implementing it.
If reaffirmed, record the concern and implement within that explicit instruction.
Never initialize Git, change branches, commit or push without authorization.
Use Conventional Commits in English when commits are requested; do not invent
co-author identities or copy a different agent's attribution.

## 8. Definition of done for implementation phases

`make quality`, `make ci` and `make check` include offline lock consistency,
Ruff lint/format, strict typing, offline pytest with a 75% combined coverage floor,
and real benchmark comparison with a 20% ceiling. The baseline is machine-specific.
Network-boundary policy and recorded failure behavior are tested offline. Dependency
advisory retrieval is a separate network-enabled security step and is not claimed
as completed by these commands. See decision 0005 for operational limitations.

Every new contract is validated; every trade-off has its decision; graph/corpus/
solver changes have meaningful properties and benchmarks; LLM stages have real
cassettes and replay-key tests. All three budget modes pass feasibility and
infeasibility checks, and ledger totals round-trip. Two clean replay runs from
the same commit, inputs and seed must yield identical bytes, including manifests.
The README identifies reproducible artifacts and operational procedures.
Do not claim these checks pass until they exist and have actually been run.
