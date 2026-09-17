"""Acquire or replay the approved 25-unit proposal/critique/revision experiment."""

import argparse
import json
from pathlib import Path

from goes_tech_kg.config import Settings
from goes_tech_kg.curriculum.generate import generate_units
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, gcloud_token_provider
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.release import UnitPlan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--unit")
    args = parser.parse_args()
    plans = tuple(
        UnitPlan.model_validate(u)
        for u in json.loads(Path("config/curriculum_map.json").read_text())["units"]
    )
    if args.unit:
        plans = tuple(p for p in plans if p.id == args.unit)
        if not plans:
            raise ValueError("unknown unit")
    settings = Settings()
    if args.online and not settings.gcp_project:
        raise ValueError("GCP project required")
    client = (
        VertexClient(str(settings.gcp_project), gcloud_token_provider(), timeout_seconds=240)
        if args.online
        else None
    )
    gateway = LLMGateway(ReplayStore(Path("data/processed/llm_responses")), client)
    generate_units(
        plans, EvidenceIndex(Path("data")), gateway, Path("data/processed/curriculum_experiment")
    )
    print(f"acquired={gateway.acquired} replayed={gateway.replayed}")


if __name__ == "__main__":
    main()
