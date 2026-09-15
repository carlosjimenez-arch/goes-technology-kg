"""Compare human ratings from a filled calibration CSV with recorded judge verdicts."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from goes_tech_kg.eval import agreement

csv_path, key_path = Path(sys.argv[1]), Path(sys.argv[2])
key = {item["item_prefix"]: item for item in json.loads(key_path.read_text())["items"]}
human_by_cell: dict[str, list[str]] = defaultdict(list)
with csv_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        verdict = row["human_verdict(accept|revise|reject)"].strip().lower()
        if verdict:
            human_by_cell[row["item_id"].split("-")[0]].append(verdict)
if not human_by_cell:
    raise SystemExit("no human verdicts filled in yet")
ORDER = {"reject": 0, "revise": 1, "accept": 2}
# A cell's human verdict is its strictest micro-skill verdict, mirroring the judge rubric.
human_cell = {
    prefix: min(verdicts, key=lambda v: ORDER[v]) for prefix, verdicts in human_by_cell.items()
}
judges: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
for prefix, human in sorted(human_cell.items()):
    for judgement in key[prefix]["judgements"]:
        if "verdict" in judgement:
            judges[judgement["judge_model"]][0].append(human)
            judges[judgement["judge_model"]][1].append(judgement["verdict"])
for judge, (human, machine) in sorted(judges.items()):
    kappa = agreement.cohen_kappa(human, machine)
    alpha = agreement.krippendorff_alpha([human, machine])
    ordinal = agreement.krippendorff_alpha(
        [[ORDER[v] for v in human], [ORDER[v] for v in machine]], "ordinal"
    )
    print(
        f"{judge}: n={len(human)} kappa={kappa:.3f} alpha_nominal={alpha:.3f} "
        f"alpha_ordinal={ordinal:.3f} status={agreement.activation(ordinal)}"
    )
