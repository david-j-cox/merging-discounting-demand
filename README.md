# Effort–Demand Unification

A constrained two-commodity demand model that unifies behavioral economic
demand (Hursh & Silberberg, 2008; Koffarnus et al., 2015) with effort
discounting (Hartmann et al., 2013; Bialaszek et al., 2017; Klein-Flügge et
al., 2015). The empirical agenda tests three predictions: within-subject
parameter linkage between a hypothetical purchase task and effort discounting,
a functional-form crossover from exponential-like to capability-bounded
discounting across the effort range, and substitutability effects that shift
parameters in linked ways across both task types.

The project scaffolds: (a) the unified model in Python, (b) a simulation suite
that verifies identifiability and parameter recovery, (c) a novel "effort
purchase task" piloted via a jsPsych web experiment, and (d) the full
preregistered study.

See **[CLAUDE.md](CLAUDE.md)** for the full handoff document, build order, and
mathematical foundations. Derivations live in `docs/derivation.md` (created at
the start of Phase 2).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --all-extras --dev
uv run pre-commit install
uv run pytest
```

## Layout

```
src/edu/        # Python package: models, fitting, simulation, analysis
experiment/     # jsPsych 8 + TypeScript web experiment
notebooks/      # narrative analyses; import from src/edu, never define logic
scripts/        # reproducible CLIs for simulations and figure generation
tests/          # pytest suite mirroring src/edu structure
docs/           # theory, derivation, protocol, analysis plan
data/           # raw/ (read-only), processed/, simulated/
results/        # figures/, tables/
```
