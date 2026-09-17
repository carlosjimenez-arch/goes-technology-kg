"""Explicit acquisition of real release-boundary responses for the offline integration fixture."""

import argparse
from pathlib import Path

from goes_tech_kg.config import Settings
from goes_tech_kg.curriculum.generate import generate_units
from goes_tech_kg.eval.release_fixture import release_fixture
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, gcloud_token_provider

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    evidence, plan = release_fixture(Path.cwd(), Path(".cache/release-fixture-data"))
    client = (
        VertexClient(str(Settings().gcp_project), gcloud_token_provider()) if args.online else None
    )
    gateway = LLMGateway(ReplayStore(Path("data/processed/llm_responses")), client)
    generate_units((plan,), evidence, gateway, Path("data/processed/release_fixture"), workers=1)
    print(f"acquired={gateway.acquired} replayed={gateway.replayed}")
