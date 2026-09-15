"""Stratified sample of recorded proposals for human rating; judge verdicts are withheld.

Writes evaluation/judges/calibration-sample-<plan>.csv (one row per micro-skill) and a JSON
key linking rows to request digests. The rater fills verdict and score columns without seeing
any judge output; scripts/score_judge_agreement.py compares afterwards.
"""

import csv
import json
import random
import sys
from pathlib import Path

from goes_tech_kg.eval.prompt_experiment import load_plan
from goes_tech_kg.schemas.base import canonical_json
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.llm import ResponseRecord

root = Path(__file__).resolve().parents[1]
report_path = Path(sys.argv[1])
per_model = int(sys.argv[2]) if len(sys.argv) > 2 else 6
report = json.loads(report_path.read_text())
plan = load_plan(root / "data/processed/prompt_experiments" / f"plan-{report['plan']['name']}.json")
store = root / "data/processed/llm_responses"

# Deterministic sampling: seed from the report's plan digest so re-runs pick the same cells.
rng = random.Random(int(report["plan_sha256"][:8], 16))
cells = [
    row
    for row in report["observations"]
    if row.get("parse_ok") and row.get("status") == "ok" and row.get("judgements")
]
by_model: dict[str, list[dict[str, object]]] = {}
for row in cells:
    by_model.setdefault(str(row["model"]), []).append(row)
sample: list[dict[str, object]] = []
for _model, rows in sorted(by_model.items()):
    rows = sorted(rows, key=lambda r: str(r["request_sha256"]))
    rng.shuffle(rows)
    sample.extend(rows[:per_model])

out_dir = root / "evaluation/judges"
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / f"calibration-sample-{report['plan']['name']}.csv"
key_path = out_dir / f"calibration-key-{report['plan']['name']}.json"
key = []
with csv_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "item_id",
            "grade",
            "skill_map_entry",
            "micro_skill_statement",
            "strand",
            "cognitive_domain",
            "min_tier",
            "evidence_quote",
            "locator",
            "human_verdict(accept|revise|reject)",
            "human_score(0-1)",
            "human_issue_codes",
            "human_notes",
        ]
    )
    for index, row in enumerate(sample, 1):
        record = ResponseRecord.model_validate_json(
            (store / f"{row['request_sha256']}.json").read_bytes()
        )
        output = DecompositionOutput.model_validate_json(record.response_text)
        case = next(c for c in plan.cases if c.id == row["case_id"])
        for m_index, micro in enumerate(output.micro_skills, 1):
            item_id = f"{index:02d}-{m_index:02d}"
            writer.writerow(
                [
                    item_id,
                    case.grade,
                    case.skill_map_entry,
                    micro.statement,
                    micro.strand.value,
                    micro.cognitive_domain.value,
                    micro.min_tier.value,
                    micro.evidence_quotes[0].quote,
                    micro.evidence_quotes[0].locator_hint,
                    "",
                    "",
                    "",
                    "",
                ]
            )
        key.append(
            {
                "item_prefix": f"{index:02d}",
                "request_sha256": row["request_sha256"],
                "prompt_id": row["prompt_id"],
                "model": row["model"],
                "case_id": row["case_id"],
                "context_sha256": report["context_sha256"][case.id],
                "judgements": [
                    {k: j[k] for k in ("judge_model", "verdict", "score", "issue_codes") if k in j}
                    for j in row["judgements"]
                ],
            }
        )
key_path.write_text(
    canonical_json({"schema_version": "judge-calibration-key/1.0", "items": key}) + "\n"
)
print(csv_path, key_path, "cells:", len(sample))
