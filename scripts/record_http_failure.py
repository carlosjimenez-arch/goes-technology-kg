"""Explicitly record a real inaccessible Salvadoran source; never run in CI."""

from pathlib import Path

import httpx
import vcr

from goes_tech_kg.corpus.pipeline import read_manifest

record = next(
    r
    for r in read_manifest(Path("data/manifests/corpus.jsonl"))
    if r.document.slug == "sv-cs-current"
)
with vcr.use_cassette(
    "tests/cassettes/sv-cs-current.yaml",
    record_mode="once",
    before_record_response=lambda response: {
        **response,
        "headers": {k: v for k, v in response["headers"].items() if k.lower() != "set-cookie"},
    },
    filter_headers=["authorization", "cookie"],
    filter_query_parameters=["key", "token"],
):
    with httpx.Client(timeout=45) as client:
        response = client.get(str(record.document.url))
        if response.status_code not in {403, 404}:
            raise RuntimeError(
                "Source changed: review the acquisition and replace the obsolete failure test"
            )
        print(response.status_code)
