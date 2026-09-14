"""Fail a build when a measured median exceeds its pinned baseline by more than 20%."""

import json
import sys
from pathlib import Path


def regressions(baseline: dict, measured: dict) -> list[str]:
    reference = {row["name"]: row["stats"]["median"] for row in baseline["benchmarks"]}
    current = {row["name"]: row["stats"]["median"] for row in measured["benchmarks"]}
    if not reference or reference.keys() != current.keys():
        raise ValueError("benchmark set changed; review and establish an explicit baseline")
    return [name for name in sorted(reference) if current[name] > reference[name] * 1.2]


if __name__ == "__main__":
    baseline, measured = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
    failures = regressions(baseline, measured)
    if failures:
        raise SystemExit("Performance regression >20%: " + ", ".join(failures))
    print("All benchmark medians within 20% of baseline")
