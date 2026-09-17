"""Run a prompt experiment. Replay by default; --online authorizes provider acquisition.

--golden-context builds the context from the committed excerpts instead of the ignored interim
chunks, which is what the offline replay test uses.
"""

import sys
from pathlib import Path

from goes_tech_kg.config import Settings
from goes_tech_kg.eval.golden_context import golden_lookup
from goes_tech_kg.eval.prompt_experiment import load_plan, run_experiment
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, resilient_token_provider
from goes_tech_kg.retrieval.references import ChunkStore
from goes_tech_kg.schemas.scope import load_scope

ROOT = Path(__file__).resolve().parents[1]
#: Concurrency is only useful while acquiring; replay is local and ordered.
ONLINE_WORKERS = 6


def main(plan_path: Path, online: bool, golden_context: bool) -> None:
    """Run every cell of the plan and print how many answers were acquired versus replayed."""
    plan = load_plan(plan_path)
    settings = Settings()
    client = None
    if online:
        if not settings.gcp_project:
            raise SystemExit(
                "GOOGLE_CLOUD_PROJECT or GOES_TECH_KG_GCP_PROJECT is required for --online"
            )
        client = VertexClient(settings.gcp_project, resilient_token_provider())
    lookup = (
        golden_lookup(ROOT / "tests/golden")[0]
        if golden_context
        else ChunkStore(ROOT / "data/interim/chunks.jsonl").lookup
    )
    report = run_experiment(
        plan,
        LLMGateway(ReplayStore(ROOT / "data/processed/llm_responses"), client),
        lookup,
        ROOT / "data/processed/prompt_experiments" / plan.name,
        workers=ONLINE_WORKERS if online else 1,
        scope=load_scope(ROOT / "config/technology_scope.yaml"),
    )
    print(
        f"observations={len(report.observations)} "
        f"acquired={report.acquired} replayed={report.replayed}"
    )


if __name__ == "__main__":
    flags = sys.argv[2:]
    main(Path(sys.argv[1]), "--online" in flags, "--golden-context" in flags)
