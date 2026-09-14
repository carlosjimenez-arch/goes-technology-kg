UV_CACHE_DIR := .cache/uv
export UV_CACHE_DIR

.PHONY: setup check quality ci static-check benchmark
setup:
	uv sync --locked
quality ci: check

check: static-check
	$(MAKE) benchmark

static-check:
	uv lock --check --offline
	uv run --locked ruff check src tests scripts
	uv run --locked ruff format --check src tests scripts
	uv run --locked mypy
	uv run --locked pytest tests/unit tests/integration --benchmark-disable --cov=goes_tech_kg --cov-fail-under=75
benchmark:
	mkdir -p reports
	uv run --locked pytest tests/benchmarks --benchmark-json=reports/benchmark.json --benchmark-min-rounds=30
	uv run --locked python scripts/check_benchmarks.py tests/benchmarks/baseline.json reports/benchmark.json
