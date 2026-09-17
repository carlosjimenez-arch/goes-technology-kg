"""Print the decision 0013 tables for a recorded prompt-experiment report, entirely offline."""

import sys
from pathlib import Path

import polars as pl

from goes_tech_kg.eval import prompt_report
from goes_tech_kg.eval.prompt_experiment import load_outputs, load_report

ROOT = Path(__file__).resolve().parents[1]
RESPONSE_STORE = ROOT / "data/processed/llm_responses"


def main(report_path: Path) -> None:
    """Render every table of the decision, gates before ranking."""
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(24)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_fmt_str_lengths(48)
    report = load_report(report_path)
    frame = prompt_report.observations_frame(report)
    tables = {
        "eligibility (gates 1-3)": prompt_report.eligibility(frame),
        "ranking (gates 4-7, valid non-refused cells)": prompt_report.ranking(frame),
        "worst cells": prompt_report.worst_cells(frame),
        "behavioural pairs": prompt_report.behavioural(report),
        "judge agreement": prompt_report.judge_agreement(report),
        "paraphrase invariance": prompt_report.invariance(
            report, load_outputs(report, RESPONSE_STORE)
        ),
    }
    for title, table in tables.items():
        print(f"== {title}")
        print(table)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
