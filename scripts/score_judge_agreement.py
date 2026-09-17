"""Compare human ratings from a filled calibration CSV with the recorded judge verdicts.

A cell's human verdict is its strictest micro-skill verdict, which mirrors the judge rubric:
one fabricated quote or one wrong grade sinks the whole proposal.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from goes_tech_kg.eval import agreement

#: Ordinal positions of the shared verdict scale, strictest first.
VERDICT_ORDER = {"reject": 0, "revise": 1, "accept": 2}
HUMAN_COLUMN = "human_verdict(accept|revise|reject)"


def read_human_verdicts(csv_path: Path) -> dict[str, str]:
    """The strictest human verdict per sampled cell, keyed by the cell prefix of the item id."""
    by_cell: dict[str, list[str]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            verdict = row[HUMAN_COLUMN].strip().lower()
            if verdict:
                by_cell[row["item_id"].split("-")[0]].append(verdict)
    return {
        prefix: min(verdicts, key=lambda v: VERDICT_ORDER[v])
        for prefix, verdicts in by_cell.items()
    }


def paired_verdicts(
    human: dict[str, str], key: dict[str, dict[str, object]]
) -> dict[str, tuple[list[str], list[str]]]:
    """Human and judge verdicts for every cell both of them decided, per judge."""
    pairs: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for prefix, human_verdict in sorted(human.items()):
        for judgement in key[prefix]["judgements"]:  # type: ignore[index]
            if "verdict" in judgement:
                pairs[judgement["judge_model"]][0].append(human_verdict)
                pairs[judgement["judge_model"]][1].append(judgement["verdict"])
    return dict(pairs)


def main(csv_path: Path, key_path: Path) -> None:
    """Print each judge's agreement with the human rater and its activation status."""
    key = {item["item_prefix"]: item for item in json.loads(key_path.read_text())["items"]}
    human = read_human_verdicts(csv_path)
    if not human:
        raise SystemExit("no human verdicts filled in yet")
    for judge, (rated_by_human, rated_by_judge) in sorted(paired_verdicts(human, key).items()):
        kappa = agreement.cohen_kappa(rated_by_human, rated_by_judge)
        nominal = agreement.krippendorff_alpha([rated_by_human, rated_by_judge])
        ordinal = agreement.krippendorff_alpha(
            [
                [VERDICT_ORDER[v] for v in rated_by_human],
                [VERDICT_ORDER[v] for v in rated_by_judge],
            ],
            "ordinal",
        )
        print(
            f"{judge}: n={len(rated_by_human)} kappa={kappa:.3f} alpha_nominal={nominal:.3f} "
            f"alpha_ordinal={ordinal:.3f} status={agreement.activation(ordinal)}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
