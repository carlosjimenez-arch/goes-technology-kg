"""Baseline table of decision 0011: B0 (official programme) and B1 (single call) on the golden set.

B2 (single call plus one judge with a revision loop) and S (hierarchical plus judge panel) are
not built, and the manifest says so rather than leaving their absence to be inferred.
"""

import json
import sys
from pathlib import Path

import polars as pl

from goes_tech_kg.eval import golden_metrics
from goes_tech_kg.eval.prompt_experiment import load_outputs, load_report
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.experiment import ExperimentReport
from goes_tech_kg.schemas.golden import GoldenRecord, GoldenScore, load_golden

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "evaluation/golden"
RESPONSE_STORE = ROOT / "data/processed/llm_responses"
RESULTS_DIR = ROOT / "evaluation/baselines/results"
SYSTEMS_NOT_BUILT = {
    "B2": "single call plus one generic judge with a revision loop: no revision stage exists yet",
    "S": "hierarchical decomposition plus judge panel: not built",
}


def golden_digest(records: dict[str, GoldenRecord]) -> str:
    """Digest of the golden set the table was measured on, so rows cannot be mixed."""
    return byte_digest(
        "".join(
            (GOLDEN_DIR / f"{case_id}.yaml").read_text() for case_id in sorted(records)
        ).encode()
    )


def score_rows(report: ExperimentReport, records: dict[str, GoldenRecord]) -> list[GoldenScore]:
    """One row per system and case: the official programme, then every recorded B1 cell."""
    rows = [golden_metrics.official_baseline(g) for g in records.values()]
    outputs = load_outputs(report, RESPONSE_STORE)
    rows.extend(
        golden_metrics.score(output, records[case_id], f"B1-{prompt_id}", model)
        for (prompt_id, model, case_id), output in sorted(outputs.items())
        if case_id in records
    )
    return rows


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the per-case rows into one row per system, worst case kept visible."""
    return (
        frame.group_by(["system", "model"])
        .agg(
            pl.len().alias("cases"),
            pl.col("decomposition_recall").mean().alias("decomposition_recall"),
            pl.col("decomposition_recall").min().alias("worst_case_recall"),
            pl.col("decomposition_precision").mean().alias("decomposition_precision"),
            pl.col("edge_precision").mean().alias("edge_precision"),
            pl.col("edge_recall").mean().alias("edge_recall"),
            pl.col("edge_f1").mean().alias("edge_f1"),
            pl.col("cognitive_agreement").mean().alias("cognitive_agreement"),
            pl.col("tier_agreement").mean().alias("tier_agreement"),
        )
        .sort(["system", "model"])
    )


def main(report_path: Path) -> None:
    """Write the table under a directory named by the golden digest, and print it."""
    records = {g.case_id: g for g in load_golden(GOLDEN_DIR)}
    digest = golden_digest(records)
    report = load_report(report_path)
    rows = score_rows(report, records)
    frame = pl.DataFrame(
        [
            {
                # CSV has no nested type; the unmatched list travels as its JSON text.
                key: (json.dumps(value) if isinstance(value, list) else value)
                for key, value in row.model_dump(mode="json").items()
            }
            for row in rows
        ]
    )
    table = summarize(frame)
    out_dir = RESULTS_DIR / digest[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(out_dir / "cells.csv")
    table.write_csv(out_dir / "table.csv")
    (out_dir / "manifest.json").write_text(
        canonical_json(
            {
                "schema_version": "baseline-table/1.0",
                "golden_sha256": digest,
                "golden_annotation_methods": sorted(
                    {g.annotation_method for g in records.values()}
                ),
                "report": str(report_path),
                "report_plan_sha256": report.plan_sha256,
                "systems_not_built": SYSTEMS_NOT_BUILT,
            }
        )
        + "\n"
    )
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    print(table)
    print("per-case worst recall:")
    print(
        frame.filter(pl.col("system") != "B0-official-programme")
        .select("system", "model", "case_id", "decomposition_recall", "edge_f1", "unmatched_golden")
        .sort("decomposition_recall")
        .head(8)
    )
    print(out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
