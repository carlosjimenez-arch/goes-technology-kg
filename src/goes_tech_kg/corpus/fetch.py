"""Network boundary with bounded redirects, payload checks and hash-pinned replay."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from goes_tech_kg.corpus.license_gate import allowed_url, license_gate
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.corpus import Acquisition, LicenseDecision, SourceDocument

MAX_BYTES = 50_000_000


def fetch(
    document: SourceDocument,
    gate: LicenseDecision,
    policy: dict[str, Any],
    raw_root: Path,
    client: httpx.Client,
) -> Acquisition:
    if gate.document_id != document.id:
        raise ValueError("gate belongs to another document")
    if gate.status != "accepted":
        return Acquisition(
            document=document, license_decision=gate, status="rejected", error=gate.reason
        )
    url = str(document.url)
    try:
        for _ in range(6):
            if not allowed_url(url, policy):
                raise ValueError("redirect outside allowlist")
            with client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                response.raise_for_status()
                payload = bytearray()
                for part in response.iter_bytes():
                    payload.extend(part)
                    if len(payload) > MAX_BYTES:
                        raise ValueError("document exceeds size limit")
                body = bytes(payload)
                if document.format == "pdf" and not body.startswith(b"%PDF-"):
                    raise ValueError("PDF response does not have a PDF signature")
                if document.format == "html" and b"<" not in body[:4096]:
                    raise ValueError("HTML response has no markup")
                if not body:
                    raise ValueError("empty document")
                sha = byte_digest(body)
                raw_root.mkdir(parents=True, exist_ok=True)
                path = raw_root / sha
                if path.exists() and byte_digest(path.read_bytes()) != sha:
                    raise ValueError("corrupted content-addressed store")
                path.write_bytes(body)
                return Acquisition(
                    document=document,
                    license_decision=gate,
                    status="accepted",
                    sha256=sha,
                    byte_count=len(body),
                    acquired_at=datetime.now(UTC),
                    final_url=url,
                )
        raise ValueError("too many redirects")
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return Acquisition(
            document=document, license_decision=gate, status="fetch_failed", error=str(exc)
        )


def replay_bytes(record: Acquisition, raw_root: Path) -> bytes:
    if record.status != "accepted" or record.sha256 is None:
        raise ValueError("only fully accepted acquisitions can replay")
    data = (raw_root / record.sha256).read_bytes()
    if byte_digest(data) != record.sha256 or len(data) != record.byte_count:
        raise ValueError("source digest or byte count mismatch")
    return data


def restore(
    record: Acquisition, policy: dict[str, Any], raw_root: Path, client: httpx.Client
) -> None:
    """Re-download a pinned acquisition without replacing its manifest or accepting drift."""
    if record.status != "accepted" or record.sha256 is None:
        raise ValueError("only accepted acquisitions can be restored")
    gate = license_gate(record.document, policy)
    if gate.status != "accepted":
        raise ValueError(f"current license policy rejects restoration: {gate.reason}")
    if (raw_root / record.sha256).exists():
        replay_bytes(record, raw_root)
        return
    recovered = fetch(record.document, gate, policy, raw_root, client)
    if recovered.status != "accepted":
        raise ValueError(f"source restoration failed: {recovered.error}")
    if recovered.sha256 != record.sha256 or recovered.byte_count != record.byte_count:
        raise ValueError("remote source changed; pinned manifest was not modified")
