from pathlib import Path

import httpx
import vcr
import yaml

from goes_tech_kg.corpus.fetch import fetch
from goes_tech_kg.corpus.pipeline import read_manifest


def test_real_unavailable_source_never_becomes_accepted(tmp_path):
    record = next(
        r
        for r in read_manifest(Path("data/manifests/corpus.jsonl"))
        if r.document.slug == "sv-cs-current"
    )
    policy = yaml.safe_load(Path("corpus/allowlist.yaml").read_text())
    with vcr.use_cassette("tests/cassettes/sv-cs-current.yaml", record_mode="none") as cassette:
        with httpx.Client(timeout=45) as client:
            result = fetch(record.document, record.license_decision, policy, tmp_path, client)
        assert cassette.all_played
    assert result.status == "fetch_failed"
    assert result.sha256 is None and result.error
    assert not list(tmp_path.iterdir())
