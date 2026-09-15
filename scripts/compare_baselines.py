"""Baseline table of decision 0011: B0 (official programme) and B1 (single-call proposer)
against the golden set, from recorded responses only. B2 and S are reported as not built."""

import json
import sys
from pathlib import Path

import polars as pl

from goes_tech_kg.eval import golden_metrics
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.golden import load_golden
from goes_tech_kg.schemas.llm import ResponseRecord

root = Path(__file__).resolve().parents[1]
report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
golden = {g.case_id: g for g in load_golden(root / "evaluation/golden")}
golden_sha = byte_digest(
    "".join(
        (root / "evaluation/golden" / f"{cid}.yaml").read_text() for cid in sorted(golden)
    ).encode()
)
store = root / "data/processed/llm_responses"
rows: list[dict[str, object]] = []
for g in golden.values():
    rows.append(
        {"system": "B0-official-programme", "model": "none", **golden_metrics.official_baseline(g)}
    )
for row in report["observations"]:
    if row["case_id"] not in golden or not row.get("parse_ok") or row.get("status") != "ok":
        continue
    record = ResponseRecord.model_validate_json(
        (store / f"{row['request_sha256']}.json").read_bytes()
    )
    output = DecompositionOutput.model_validate_json(record.response_text)
    rows.append(
        {
            "system": f"B1-{row['prompt_id']}",
            "model": row["model"],
            **golden_metrics.score(output, golden[row["case_id"]]),
        }
    )
frame = pl.DataFrame(
    [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
)
summary = (
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
out_dir = root / "evaluation/baselines/results" / golden_sha[:12]
out_dir.mkdir(parents=True, exist_ok=True)
frame.write_csv(out_dir / "cells.csv")
summary.write_csv(out_dir / "table.csv")
(out_dir / "manifest.json").write_text(
    canonical_json(
        {
            "schema_version": "baseline-table/1.0",
            "golden_sha256": golden_sha,
            "golden_annotation_methods": sorted({g.annotation_method for g in golden.values()}),
            "report": str(report_path),
            "report_plan_sha256": report["plan_sha256"],
            "systems_not_built": {
                "B2": "single call plus one generic judge with a revision loop: no revision stage exists yet",
                "S": "hierarchical decomposition plus judge panel: not built",
            },
        }
    )
    + "\n"
)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)
print(summary)
print("per-case worst recall:")
print(
    frame.filter(pl.col("system") != "B0-official-programme")
    .select("system", "model", "case_id", "decomposition_recall", "edge_f1", "unmatched_golden")
    .sort("decomposition_recall")
    .head(8)
)
print(out_dir)
