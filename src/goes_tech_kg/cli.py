"""Explicit acquisition and offline replay; imports never touch the network."""

from pathlib import Path

import httpx
import typer
import yaml

from goes_tech_kg.config import Settings
from goes_tech_kg.corpus.embed import Embedder
from goes_tech_kg.corpus.fetch import restore as restore_record
from goes_tech_kg.corpus.pipeline import acquire, compile_corpus, read_manifest
from goes_tech_kg.schemas.base import canonical_json

app = typer.Typer(no_args_is_help=True)


@app.command()
def restore(data_root: Path = Path("data"), policy: Path = Path("corpus/allowlist.yaml")) -> None:
    """Restore missing raw files at the hashes pinned in the manifest."""
    registry = yaml.safe_load(policy.read_text())
    with httpx.Client(timeout=45) as client:
        for record in read_manifest(data_root / "manifests/corpus.jsonl"):
            if record.status == "accepted":
                restore_record(record, registry, data_root / "raw/corpus", client)
    typer.echo("All accepted source hashes verified")


@app.command()
def fetch(
    catalogue: Path = Path("corpus/sources.yaml"),
    policy: Path = Path("corpus/allowlist.yaml"),
    data_root: Path = Path("data"),
) -> None:
    records = acquire(catalogue, policy, data_root)
    typer.echo(
        canonical_json(
            {
                "accepted": sum(r.status == "accepted" for r in records),
                "failed": sum(r.status != "accepted" for r in records),
            }
        )
    )


@app.command()
def build(
    data_root: Path = Path("data"),
    model_path: Path | None = None,
    generate_embeddings: bool = False,
    skip_embeddings: bool = False,
) -> None:
    embedder = (
        None
        if skip_embeddings
        else Embedder(
            model_path, data_root / "processed/embeddings", replay=not generate_embeddings
        )
    )
    report = compile_corpus(data_root, embedder, Settings().seed)
    typer.echo(canonical_json(report))
    if not report["coverage_complete"]:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
