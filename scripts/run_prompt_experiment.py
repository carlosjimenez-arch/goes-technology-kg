"""Run a prompt experiment. Replay by default; --online authorizes provider acquisition."""

import sys
from pathlib import Path

from goes_tech_kg.config import Settings
from goes_tech_kg.eval.golden_context import golden_lookup
from goes_tech_kg.eval.prompt_experiment import load_plan, run_experiment
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, resilient_token_provider
from goes_tech_kg.retrieval.context import load_interim_chunks

root = Path(__file__).resolve().parents[1]
plan_path = Path(sys.argv[1])
online = "--online" in sys.argv[2:]
golden = "--golden-context" in sys.argv[2:]
plan = load_plan(plan_path)
settings = Settings()
client = None
if online:
    if not settings.gcp_project:
        raise SystemExit(
            "GOOGLE_CLOUD_PROJECT or GOES_TECH_KG_GCP_PROJECT is required for --online"
        )
    client = VertexClient(settings.gcp_project, resilient_token_provider())
gateway = LLMGateway(ReplayStore(root / "data/processed/llm_responses"), client)
report = run_experiment(
    plan,
    gateway,
    golden_lookup(root / "tests/golden")[0]
    if golden
    else load_interim_chunks(root / "data/interim/chunks.jsonl"),
    root / "data/processed/prompt_experiments" / plan.name,
    workers=6 if online else 1,
)
print(
    f"observations={len(report['observations'])} acquired={report['acquired']} "
    f"replayed={report['replayed']}"
)
