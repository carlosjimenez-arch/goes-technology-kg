"""Small experiment plan whose context comes only from committed golden excerpts (offline replay)."""

from pathlib import Path

from goes_tech_kg.eval.golden_context import golden_chunks
from goes_tech_kg.eval.prompt_experiment import ExperimentCase, ExperimentPlan, ModelSpec
from goes_tech_kg.schemas.base import canonical_json
from goes_tech_kg.schemas.llm import ContextRef, GenerationSettings

root = Path(__file__).resolve().parents[1]
goldens = golden_chunks(root / "tests/golden")


def ref(slug: str, needles: list[str]) -> ContextRef:
    document, chunks = goldens[slug]
    chunk = chunks[0]
    ids = []
    for needle in needles:
        matches = [p.original.id for p in chunk.paragraphs if needle in p.text]
        if len(matches) != 1:
            raise ValueError(f"{slug}: needle {needle!r} matched {len(matches)} paragraphs")
        ids.append(matches[0])
    return ContextRef(
        chunk_id=chunk.id,
        document_id=document.id,
        document_sha256=chunk.document_sha256,
        paragraph_ids=tuple(ids),
    )


computing = ref(
    "eng-computing",
    [
        "programs execute by following precise and unambiguous instructions",
        "use logical reasoning to predict the behaviour of simple programs",
    ],
)
rocket = ref(
    "unplugged-rocket",
    [
        "Give a set of precise instructions that programs an object",
        "Identify where a bug has occurred",
        "instructions are all written before they are tested",
        "step through their instructions one by one",
    ],
)
entry = (
    "Crear, seguir y depurar secuencias de instrucciones precisas para que un agente "
    "alcance una meta, prediciendo el resultado antes de ejecutarlas."
)
settings = GenerationSettings(thinking_budget=4096, max_output_tokens=32768)
plan = ExperimentPlan(
    name="replay-fixture",
    prompt_ids=("decomposition-structured", "decomposition-guided-v2"),
    models=(ModelSpec(model="gemini-2.5-flash", location="us-central1", settings=settings),),
    judges=(ModelSpec(model="gemini-2.5-pro", location="us-central1", settings=settings),),
    judge_prompt_id="curricular-judge-v2",
    cases=(
        ExperimentCase(
            id="fixture-ct-g3-secuencias",
            grade="3",
            skill_map_entry=entry,
            context_refs=(computing, rocket),
            notes="Context from committed real excerpts only.",
        ),
        ExperimentCase(
            id="fixture-malformed-grade-9",
            grade="9",
            skill_map_entry=entry,
            context_refs=(computing, rocket),
            expect_refusal=True,
        ),
    ),
    document_slugs={doc.id: slug for slug, (doc, _) in goldens.items()},
)
out = root / "data/processed/prompt_experiments/plan-replay-fixture.json"
out.write_text(canonical_json(plan) + "\n")
print(out)
