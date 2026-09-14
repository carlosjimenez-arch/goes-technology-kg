import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "performance_gate", Path("scripts/check_benchmarks.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def result(median):
    return {"benchmarks": [{"name": "actual-parser-contract", "stats": {"median": median}}]}


def test_twenty_percent_threshold_and_missing_benchmarks():
    assert module.regressions(result(1), result(1.2)) == []
    assert module.regressions(result(1), result(1.200001)) == ["actual-parser-contract"]
    with pytest.raises(ValueError, match="benchmark set"):
        module.regressions(result(1), {"benchmarks": []})
