"""Print the decision 0013 tables for a recorded prompt-experiment report (offline)."""

import sys
from pathlib import Path

import polars as pl

from goes_tech_kg.eval import prompt_report as pr
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.llm import ResponseRecord

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_fmt_str_lengths(48)

report = pr.load_report(Path(sys.argv[1]))
frame = pr.observations_frame(report)
print("== eligibility (gates 1-3)")
print(pr.eligibility(frame))
print("== ranking (gates 4-7, valid non-refused cells)")
print(pr.ranking(frame))
print("== worst cells")
print(pr.worst_cells(frame))
print("== behavioural pairs")
print(pr.behavioural(frame, report))
print("== judge agreement")
print(pr.judge_agreement(report))

store = Path("data/processed/llm_responses")
outputs = {}
for row in report["observations"]:
    if row.get("parse_ok") and row.get("status") == "ok":
        record = ResponseRecord.model_validate_json(
            (store / f"{row['request_sha256']}.json").read_bytes()
        )
        outputs[(row["prompt_id"], row["model"], row["case_id"])] = (
            DecompositionOutput.model_validate_json(record.response_text)
        )
print("== paraphrase invariance")
print(pr.invariance(report, outputs))
