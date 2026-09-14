from datetime import UTC, datetime

import pytest

from goes_tech_kg.corpus.license_gate import allowed_url, license_gate
from goes_tech_kg.schemas.corpus import Acquisition, SourceDocument

POLICY = {
    "publishers": [{"host": "www.csunplugged.org", "official": False, "path_prefixes": ["/en/"]}]
}


def document(**updates):
    data = dict(
        slug="rocket",
        title="Rocket activity",
        url="https://www.csunplugged.org/en/topics/kidbots/sending-a-rocket-to-mars/",
        publisher="University of Canterbury",
        country="NZ",
        role="foreign_curriculum",
        grades=[2, 3],
        alignment_rationale="Sequence and debugging",
        format="html",
        license="CC-BY-SA-4.0",
        admission_basis="open_license",
        license_evidence_url="https://www.csunplugged.org/en/",
        license_evidence="Publisher footer declares CC BY-SA 4.0",
        reviewed_at=datetime(2026, 9, 14, tzinfo=UTC),
    )
    data.update(updates)
    return SourceDocument(**data)


def test_allowlist_is_not_a_license():
    assert license_gate(document(), POLICY).status == "accepted"
    rejected = license_gate(document(license=None, admission_basis="unverified"), POLICY)
    assert rejected.status == "rejected" and "license" in rejected.reason.lower()
    assert license_gate(document(license_evidence=None), POLICY).status == "rejected"
    assert not allowed_url("https://www.csunplugged.org.evil.test/en/", POLICY)
    assert not allowed_url("http://www.csunplugged.org/en/", POLICY)


def test_missing_fetch_digest_cannot_be_accepted():
    d = document()
    gate = license_gate(d, POLICY)
    with pytest.raises(ValueError, match="complete acquisition"):
        Acquisition(document=d, license_decision=gate, status="accepted")
