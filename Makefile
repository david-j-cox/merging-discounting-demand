# Convenience targets for common workflows. None of this is required for
# correctness; running the underlying commands directly works fine. The
# Makefile is here mostly so that `make help` lists the things you might
# want to do without remembering uv / npm idioms.

.PHONY: help setup test test-fast test-slow lint format simulate notebooks experiment-test experiment-build clean reproduce all

help:
	@echo "Targets:"
	@echo "  setup            install Python deps via uv (run once)"
	@echo "  test             run all tests (Python + experiment); ~80s"
	@echo "  test-fast        Python tests minus @slow; ~10s"
	@echo "  test-slow        Python @slow tests (Bayesian sampling)"
	@echo "  lint             ruff check + ruff format check + mypy strict"
	@echo "  format           ruff --fix + ruff format (writes)"
	@echo "  simulate         run the Phase 3 simulation suite"
	@echo "  notebooks        execute every notebooks/*.ipynb in place"
	@echo "  experiment-test  vitest unit + e2e tests"
	@echo "  experiment-build vite production build"
	@echo "  reproduce        end-to-end check: lint + tests + simulate"
	@echo "  clean            remove caches and artifacts"

setup:
	uv sync --all-extras

test: test-fast test-slow experiment-test

test-fast:
	uv run pytest -q -m 'not slow'

test-slow:
	uv run pytest -q -m 'slow'

lint:
	uv run ruff check src/edu tests scripts
	uv run ruff format --check src/edu tests scripts
	uv run mypy src/edu

format:
	uv run ruff check --fix src/edu tests scripts
	uv run ruff format src/edu tests scripts

simulate:
	uv run python scripts/run_simulation.py

notebooks:
	uv run --extra notebook jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_demand_models.ipynb \
		notebooks/02_discounting_models.ipynb \
		notebooks/03_unified_derivation.ipynb \
		notebooks/04_simulation_recovery.ipynb \
		notebooks/05_pilot_analysis.ipynb

experiment-test:
	cd experiment && npm test

experiment-build:
	cd experiment && npm run build

# End-to-end reproducibility check. Run before tagging a release.
reproduce: lint test simulate
	@echo "All reproducibility checks passed."

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov
	rm -rf experiment/dist experiment/.vite
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
