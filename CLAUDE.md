# CLAUDE.md — Effort–Demand Unification Project

> Handoff document for Claude Code. Read this fully before taking any action. Build incrementally, commit often, and verify at each stage. Do not skip the simulation phase to get to the experiment phase. The simulation is what protects the experiment from being unfalsifiable.

---

## 1. Project context (read this first)

This project develops and tests a unification of two parallel literatures in the behavioral sciences:

- **Behavioral economic demand** (Hursh & Silberberg, 2008; Koffarnus et al., 2015) — characterizes consumption as a function of price using exponential demand curves, yielding parameters like Q₀, α, Pmax, Omax.
- **Effort discounting** (Mitchell, Hartmann, Klein-Flügge, Bialaszek, Westbrook, Treadway, Pessiglione) — characterizes how reward value decays as required effort increases, using hyperbolic, parabolic, power, or sigmoidal functions.

The two literatures measure overlapping phenomena with incompatible math. Hursh & Schwartz (2022, *Psych Addict Behav*) unified delay discounting with demand by treating time as the binding constraint. **This project does the analogous work for effort.**

The theoretical claim: effort discounting is a special case of constrained two-commodity demand where effort capacity is the binding constraint. The functional-form debate in the effort literature (hyperbolic vs. parabolic vs. power vs. sigmoidal) is, on this view, an artifact of different experiments probing different regions of the effort-budget constraint. Low-effort experiments see exponential-like decay; high-effort experiments see capability-bound truncation and infer concavity.

The empirical agenda has three predictions:

1. **Within-subject parameter linkage** — demand-curve α from a hypothetical purchase task should predict effort-discount parameters via a model-specified transformation.
2. **Functional-form crossover** — the same individual should show exponential-like effort discounting at low effort and concave/bounded discounting near capability.
3. **Substitutability effects** — manipulating substitutability of immediate vs. effortful alternatives should shift parameters in linked ways across both task types.

The repo this document scaffolds will: (a) implement the unified model, (b) run simulations to verify identifiability and parameter recovery, (c) pre-pilot a novel "effort purchase task," and (d) implement the full experiment as a runnable web-based study.

---

## 2. Repo structure

Create the following structure exactly. Do not improvise organization.

```
effort-demand-unification/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── docs/
│   ├── theory.md
│   ├── derivation.md
│   ├── experiment_protocol.md
│   └── analysis_plan.md
├── src/
│   └── edu/                          # package name: effort-demand unification
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── demand.py             # Hursh-Silberberg, Koffarnus exponentiated
│       │   ├── discounting.py        # hyperbolic, exponential, parabolic, power, sigmoidal
│       │   └── unified.py            # the unified constrained-demand model
│       ├── fitting/
│       │   ├── __init__.py
│       │   ├── nls.py                # nonlinear least squares wrappers
│       │   └── bayesian.py           # optional hierarchical Bayesian fitting
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── generate.py           # synthetic data generation
│       │   ├── recovery.py           # parameter recovery experiments
│       │   └── identifiability.py    # parameter trade-off checks
│       └── analysis/
│           ├── __init__.py
│           ├── individual.py         # per-subject fitting
│           └── group.py              # group/hierarchical analyses
├── experiment/
│   ├── README.md
│   ├── package.json
│   ├── tsconfig.json
│   ├── public/
│   ├── src/
│   │   ├── tasks/
│   │   │   ├── purchaseTask.ts       # Task 1: standard hypothetical purchase task
│   │   │   ├── effortDiscounting.ts  # Task 2: effort discounting across full range
│   │   │   ├── effortPurchaseTask.ts # Task 3: novel effort-denominated purchase task
│   │   │   └── calibration.ts        # MVC / N-back threshold calibration
│   │   ├── components/
│   │   ├── lib/
│   │   │   ├── titration.ts          # adjusting-amount procedure
│   │   │   └── randomization.ts      # task and substitutability arm randomization
│   │   └── main.ts
│   └── tests/
├── notebooks/
│   ├── 01_demand_models.ipynb
│   ├── 02_discounting_models.ipynb
│   ├── 03_unified_derivation.ipynb
│   ├── 04_simulation_recovery.ipynb
│   ├── 05_pilot_analysis.ipynb
│   └── 06_main_analysis.ipynb
├── data/
│   ├── raw/                          # never edited
│   ├── processed/
│   └── simulated/
├── results/
│   ├── figures/
│   └── tables/
├── tests/
│   ├── test_demand.py
│   ├── test_discounting.py
│   ├── test_unified.py
│   └── test_recovery.py
└── scripts/
    ├── run_simulation.py
    ├── fit_pilot.py
    └── make_figures.py
```

### Tech stack

- **Python 3.11+** for modeling, simulation, fitting, analysis
- Use `uv` for package management (fast, reproducible)
- Core deps: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `lmfit`, `numpyro` (or `pymc`), `arviz`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `jupyter`
- **TypeScript + jsPsych 8.x** for the web experiment. jsPsych is the standard for behavioral-economic web tasks and has plugins for the relevant choice procedures.
- Use `vite` for the experiment build, `vitest` for experiment unit tests
- **Data collection pipeline:** jsPsych client → DataPipe (https://pipe.jspsych.org) → OSF project storage. DataPipe is free, designed for jsPsych, and routes data to an OSF component you control. Each completed session writes one JSON file named by Prolific ID (or a generated UUID if running outside Prolific). Python analysis reads these JSON files directly via pandas. No self-hosted backend required.
- **Why DataPipe over alternatives:** zero infrastructure, IRB-friendly (OSF is a vetted academic data store), integrates with the pre-registration workflow, swappable later if the project grows. Alternatives considered and rejected for this stage: self-hosted backend (more flexibility but unnecessary maintenance burden), Firebase/Supabase (introduces browser-side API keys), Pavlovia (paid, no advantage at this scale).
- **DataPipe setup steps** (do this during Phase 4): create an OSF project for the study, create a DataPipe experiment linked to that OSF component, copy the experiment ID into the jsPsych code as a config constant, configure DataPipe to require a completion code from Prolific to prevent random POSTs from the open internet.
- **Local data flow during development:** jsPsych also writes a local JSON download as a backup. During pilot, configure DataPipe in "test mode" so submissions are flagged and don't pollute the production dataset. Switch to production mode only after the pre-registration is filed.
- **What gets stored:** raw trial-by-trial responses, calibration values, randomization assignments, timing data, and a participant ID. No PII beyond the Prolific ID. The OSF project is set to private during data collection and made public alongside the paper.

### Coding conventions

- All Python code passes `ruff check` and `mypy --strict` before commit
- All public functions have type hints and NumPy-style docstrings
- All model-fitting functions return a typed result dataclass, never a raw dict
- All random operations take an explicit `rng` argument (pass `np.random.default_rng(seed)`)
- No notebook-as-code: notebooks import from `src/edu/`, never define core logic
- Tests live in `tests/`, mirror the `src/` structure, use pytest fixtures

---

## 3. Mathematical foundations (implement these first)

### 3.1 Hursh–Silberberg exponential demand

```
log10(Q) = log10(Q0) + k * (exp(-alpha * Q0 * P) - 1)
```

where `Q` is consumption, `Q0` is demand at zero price, `P` is price, `k` is the span of the data in log10 units (typically fitted as a shared constant across subjects), `alpha` is the rate of decay in elasticity.

Derived quantities:
- Pmax (price at unit elasticity) — has a closed-form solution (Gilroy, Kaplan, Reed, Hantula & Hursh, 2019)
- Omax = Pmax × Q(Pmax)
- Essential value: EV ∝ 1 / (alpha * k^1.5) per Hursh's later work

### 3.2 Koffarnus exponentiated demand

```
Q = Q0 * 10^(k * (exp(-alpha * Q0 * P) - 1))
```

This is the form that handles zero consumption naturally. Implement both; default to Koffarnus for new fits.

### 3.3 Effort discount functions

Implement all of these with a shared interface so they're swappable in model comparison:

```
hyperbolic:    SV = A / (1 + l*E)
exponential:   SV = A * exp(-l*E)
parabolic:     SV = A - l*E^2                    (Hartmann et al. 2013)
power:         SV = A - l*E^s                    (Bialaszek et al. 2017)
sigmoidal:     SV = A / (1 + exp(s*(E - E0)))    (Klein-Flügge et al. 2015, simplified)
hyperboloid:   SV = A / (1 + l*E)^s              (Rachlin / Myerson-Green form)
```

Note `s` parameter conventions differ across forms; document each in the docstring.

### 3.4 The unified model

This is the contribution. Implement carefully; this is what the simulations test.

**Setup.** A decision-maker chooses between an immediate effortless commodity (subscript 1) and a delayed effortful commodity (subscript 2) with identical reward magnitude A. Effort budget B is the binding constraint. Demand for each commodity follows Hursh–Silberberg with shared Q0 (defended in derivation.md as a normalization).

**Naive form (Step 4 of derivation):**

```
SV(E) / A = 10^(k * (exp(-alpha * Q0 * E / A) - 1))
```

This is convex/asymptotic. It recovers Hursh-Silberberg-shaped effort discounting but does not recover the concave Bialaszek/Klein-Flügge form.

**Capability-bounded form (Step 5, Option B):**

```
Q2(E) = min[Q0 * 10^(k * (exp(-alpha * Q0 * E / A) - 1)),  B / E]
```

For low E, the first term binds (exponential demand). For high E, the second term binds (B/E hyperbolic truncation). Crossover occurs where the two terms equal each other; at that point the discount function bends sharply and reaches zero at finite E.

**Implementation requirements:**

- Vectorized across E using numpy; never loop
- Returns SV(E) and a flag indicating which regime (unconstrained / truncated) is active
- Provides analytic derivative dSV/dE for fitting
- Provides closed-form or numerically stable computation of the crossover effort E*

Also implement the Option A variant (nonlinear perceived-effort function `psi(E) = E^s` substituted into the demand equation) for model comparison. Document in derivation.md why Option B is preferred theoretically but both should be tested empirically.

### 3.5 Substitutability and the IF parameter

Hursh's interaction factor IF parameterizes substitutability between commodities. For the experimental manipulation:

```
Q2_effective(E) = Q2(E) * (1 - IF * Q1_consumed)
```

where IF = 1 is perfect substitutes, IF = 0 is independence. Implement IF as a free parameter that can be estimated from cross-price data or fixed by design (high vs low substitutability arms).

---

## 4. Build order

Do these in sequence. Do not jump ahead. Each phase ends with a commit and a passing test suite.

### Phase 0: Repo setup (1 session)

1. Initialize repo, create directory structure, set up `pyproject.toml` with `uv`
2. Create `.gitignore` (Python + Node + IDE + data/raw/* + .env)
3. Set up pre-commit with ruff and mypy
4. Stub all `__init__.py` files
5. Write minimal README with project description and one-line setup instructions
6. Commit: "initial repo scaffold"

### Phase 1: Demand and discount models (2–3 sessions)

1. Implement Hursh–Silberberg and Koffarnus models in `src/edu/models/demand.py`
2. Implement all six discount functions in `src/edu/models/discounting.py`
3. Each model class exposes: `value(params, x)`, `log_likelihood(params, x, y)`, `fit(x, y)`, derived quantities where applicable
4. Write `tests/test_demand.py` and `tests/test_discounting.py` — verify against published parameter values from Bialaszek et al. (2017, supplementary data is publicly available) and Hursh & Silberberg (2008) figure reproductions
5. Notebook `01_demand_models.ipynb` reproduces a Hursh-Silberberg demand curve from synthetic data
6. Notebook `02_discounting_models.ipynb` reproduces Bialaszek's model comparison on synthetic effort data
7. Commit: "demand and discounting models"

### Phase 2: Unified model (3–4 sessions)

This is the conceptually hard phase. Take it slow.

1. Implement the naive unified form in `src/edu/models/unified.py` first
2. Write `tests/test_unified.py` checking: SV(0) = A, SV asymptotes to A·10^(-k), monotonicity, vectorization
3. Implement the capability-bounded form (Option B). Test: smooth crossover, SV reaches 0 at finite E, parameter recovery on synthetic data
4. Implement the nonlinear-cost form (Option A) for comparison
5. Write `docs/derivation.md` documenting the math step by step (use the proof we worked out — Steps 1 through 5)
6. Notebook `03_unified_derivation.ipynb` walks through the derivation with executable code at each step, plotting the resulting curves under different parameter regimes
7. Verify: when the unified model is fit to data simulated from each candidate discount function (hyperbolic, parabolic, power), it recovers the source function's shape via different parameter regimes
8. Commit: "unified model with capability-bounded form"

### Phase 3: Simulation suite (3–4 sessions)

This phase determines whether the experiment is worth running. Be rigorous.

1. **Parameter recovery.** Generate synthetic data under known parameters across realistic ranges, fit the model back, recover parameters. Run with n = {30, 60, 120, 240} simulated subjects to determine sample-size requirements. Plot recovered vs. true parameters with 95% intervals. Required: bias < 5% and 95% CI width interpretable.
2. **Identifiability.** Profile likelihood for each parameter pair. Check whether α, B, and the cost-shape parameter trade off. If any pair shows a flat ridge in the likelihood surface, the model is not identifiable and the experiment as designed cannot test it. Document any non-identifiabilities and propose constraints (e.g., fixing k as in Hursh's work, or anchoring B to measured MVC).
3. **Cross-task linkage simulation.** Simulate the full three-task design (purchase task, effort discounting, effort purchase task) for n = 200 subjects with a realistic distribution of individual α values. Verify the framework's parameter-linkage prediction holds in simulated data and that it can be detected as a hypothesis test against the null of independent task parameters.
4. **Functional-form crossover simulation.** Generate effort-discounting data at narrow (10–60% capability) vs. wide (10–85% capability) effort ranges from the same underlying unified model. Fit single-form discount functions to each. Verify that narrow-range data fits exponential well and wide-range data requires the unified or power form. This validates the experimental design choice of wide effort range.
5. Notebook `04_simulation_recovery.ipynb` produces all simulation figures destined for the methods/supplement section of the paper.
6. Each simulation gets a CLI script in `scripts/` that reproduces it from a config file with a fixed seed.
7. Commit: "simulation suite"

**Stop point.** If any simulation reveals fundamental non-identifiability or insufficient power at reasonable sample sizes, escalate to the user before proceeding to Phase 4. Do not paper over this — it is the entire point of doing simulations first.

### Phase 4: Experiment build (4–5 sessions)

1. Set up jsPsych 8.x project in `experiment/` with TypeScript and Vite
2. Implement the calibration task in `tasks/calibration.ts`:
   - Physical: handgrip MVC via three maximal squeezes, take median
   - Cognitive: adaptive N-back to find threshold level (define threshold operationally)
3. Implement Task 1 (standard purchase task) in `tasks/purchaseTask.ts`:
   - Use a generic commodity (snack food) for broad applicability
   - Standard 17-price array (free, $0.01, $0.05, $0.13, ..., $1120)
   - Fit Hursh-Silberberg post hoc, not real-time
4. Implement Task 2 (effort discounting) in `tasks/effortDiscounting.ts`:
   - Six effort levels: 10, 25, 40, 55, 70, 85% of individual capability
   - Adjusting-amount titration procedure (Du, Green & Myerson 2002 algorithm)
   - Counterbalance order; randomize within block
5. Implement Task 3 (effort purchase task) in `tasks/effortPurchaseTask.ts` — **this is novel; budget extra time**:
   - "How many [calibrated effort units] would you perform for X [reward]?"
   - Price array in effort units, with reward fixed
   - Open-ended response with bounds checked at the calibrated capability
   - Pilot this task in isolation before building it into the full battery
6. Implement randomization (`lib/randomization.ts`):
   - Substitutability arm: between-subjects, 50/50
   - Task order: counterbalanced within-subjects (Latin square or random permutation)
7. Implement data export — jsPsych writes a local JSON backup AND POSTs to DataPipe via the `@jspsych-contrib/plugin-pipe` plugin. Configuration:
   - DataPipe experiment ID stored in `experiment/src/config.ts` (committed) or `.env` (not committed if you prefer)
   - Set up the OSF project before this step so the DataPipe ID exists
   - Test the full POST cycle with a dummy submission; verify the JSON lands in OSF
   - Implement a fallback: if DataPipe POST fails (network error), prompt the participant to download the JSON file manually so the data isn't lost
8. Write end-to-end test that runs the full experiment programmatically with simulated responses
9. Deploy a test build to a static host (Cloudflare Pages, Vercel, or GitHub Pages) for piloting
10. Commit: "experiment build"

### Phase 5: Pilot and analysis (2–3 sessions)

1. Recruit 20–30 pilot participants via Prolific. Pre-screen for English fluency and desktop-only.
2. Notebook `05_pilot_analysis.ipynb` — load pilot data, run sanity checks (completion rates, attention checks, calibration validity), fit the three tasks per subject, evaluate Task 3 data quality specifically
3. Decision point after pilot: is Task 3 yielding analyzable data? If not, redesign before main study. Document specific failure modes if they occur.
4. Pre-register the main study on OSF using the analysis plan from `docs/analysis_plan.md`
5. Commit: "pilot analysis"

### Phase 6: Main study (deferred until pilot validates)

1. Run main study (n ≈ 150–200) via Prolific with the validated experiment
2. Notebook `06_main_analysis.ipynb` — full analysis per the pre-registered plan
3. Generate publication figures via `scripts/make_figures.py`
4. Write up results

---

## 5. Analysis plan (write this into docs/analysis_plan.md and pre-register before main study)

### Primary hypotheses

**H1: Parameter linkage.** A unified model with a single individual-level α parameter shared between Task 1 and Task 3 will fit the joint data as well or better (ΔBIC ≥ 0 favoring shared) than a model with independent α parameters per task. Test: hierarchical Bayesian model with explicit prior on cross-task parameter sharing.

**H2: Functional-form crossover.** When effort-discounting data (Task 2) is split into low-effort (≤ 50% capability) and high-effort (> 50%) subsets, the best-fitting discount function differs across subsets in the predicted direction (low-effort → exponential favored; high-effort → power/bounded favored). Test: per-subject AIC/BIC comparison, sign test across subjects.

**H3: Substitutability.** Effort-discount steepness (k from the unified model fit) is greater in the low-substitutability arm than the high-substitutability arm. Test: between-subjects t-test on individual k estimates, with prior on effect size from Hursh's substitutability work.

### Secondary hypotheses

- Individual α from Task 1 correlates with k from Task 2 with the model-predicted transformation (not just a raw correlation)
- The capability-bound parameter B from the unified-model fit correlates with measured MVC (or N-back threshold)
- Functional-form crossover replicates across physical and cognitive effort modalities

### Inference framework

Hierarchical Bayesian throughout. Use NumPyro for speed. Specify priors in pre-registration. Report posterior distributions for all parameters; do not collapse to point estimates. Use posterior predictive checks for model adequacy.

### Stopping rules

If H1 fails, the unification claim is empirically falsified. The paper still publishes — that's a real finding. Do not p-hack.

---

## 6. What good looks like at each milestone

- **End of Phase 1:** I can fit Hursh-Silberberg to a published demand dataset and recover their reported parameters within rounding; same for Bialaszek's effort-discount data with the power function.
- **End of Phase 2:** I can plot the unified model under varying capability bounds and visually reproduce the empirical shapes of hyperbolic, exponential, and power discount functions as limiting cases.
- **End of Phase 3:** Parameter recovery plots show < 5% bias and reasonable CI widths at n = 150. Identifiability check shows no pathological parameter trade-offs given the planned design. If either fails, I report this and stop.
- **End of Phase 4:** End-to-end experiment runs in a browser, writes valid data, passes automated tests with simulated responses.
- **End of Phase 5:** Pilot data is analyzed, Task 3 produces analyzable output, pre-registration is filed.
- **End of Phase 6:** Main analysis is complete; figures are publication-quality; manuscript scaffold (already in `docs/`) is filled with results.

---

## 7. References (key papers — make sure these are accessible in the repo's `references/` folder if added later)

- Hursh & Silberberg (2008). Economic demand and essential value. *Psychological Review*, 115(1), 186–198.
- Hursh & Schwartz (2022). A general model of demand and discounting. *Psychology of Addictive Behaviors*, 37(1), 37–56. **The template paper for this project.**
- Koffarnus, Franck, Stein & Bickel (2015). A modified exponential behavioral economic demand model to better describe consumption data. *Experimental and Clinical Psychopharmacology*.
- Gilroy, Kaplan, Reed, Hantula & Hursh (2019). An exact solution for unit elasticity in the exponential model of operant demand. *Experimental and Clinical Psychopharmacology*, 27(6), 588–597.
- Hartmann, Hager, Tobler & Kaiser (2013). Parabolic discounting of monetary rewards by physical effort. *Behavioural Processes*, 100, 192–196.
- Bialaszek, Marcowski & Ostaszewski (2017). Physical and cognitive effort discounting across different reward magnitudes: Tests of discounting models. *PLoS ONE*, 12(7), e0182353. **Key empirical anchor.**
- Klein-Flügge, Kennerley, Friston & Bestmann (2015). Neural signatures of value comparison in human cingulate cortex during decisions requiring an effort-reward trade-off. *Journal of Neuroscience*; and the *PLoS Comp Biol* paper on dissociable effects of effort and delay.
- Treadway, Buckholtz, Schwartzman, Lambert & Zald (2009). Worth the 'EEfRT'? *PLoS ONE*, 4(8), e6598.
- Westbrook, Kester & Braver (2013). What is the subjective cost of cognitive effort? *PLoS ONE*.
- Le Bouc et al. on subtractive discounting in effort.
- Du, Green & Myerson (2002). Cross-cultural comparisons of discounting delayed and probabilistic rewards. *Psychological Record*. **Source of the titration algorithm.**
- Kaplan, Gilroy, Reed, Koffarnus & Hursh (2019). The R package `beezdemand`. *Perspectives on Behavior Science*, 42(1), 163–180. — port relevant fitting logic.

---

## 8. Operating principles for Claude Code

- **Do not skip the simulation phase.** It is what determines whether the experiment is worth running.
- **Verify against published data wherever possible.** Bialaszek's supplementary data and `beezdemand` example datasets are the targets.
- **Commit at every milestone.** Use conventional commits (`feat:`, `fix:`, `test:`, `docs:`).
- **Never modify `data/raw/`.** Treat it as read-only once data lands.
- **Stop and ask if a simulation reveals non-identifiability** rather than tweaking until something looks right.
- **Document every assumption in `docs/derivation.md`** as you implement it, especially the Q0,1 = Q0,2 normalization and the choice between Option A and Option B.
- **Tests before features.** When implementing a new model, write the test for known limiting cases first.
- **No magic numbers.** All seeds, thresholds, and array sizes are named constants in a `config.py` or YAML file.
- **The novel piece is Task 3 (effort purchase task).** When in doubt about prioritization, prioritize getting Task 3 right — it is the empirical hinge of the unification claim.

---

## 9. Likely failure modes and what to do

- **Identifiability failure between α and capability bound B.** Fix B from measured MVC; re-fit. Document this constraint as a methodological commitment in the paper.
- **Task 3 doesn't yield clean Hursh-Silberberg fits.** Try the Koffarnus form first. If that fails, the issue is likely participant comprehension of effort-as-price. Redesign instructions and re-pilot.
- **Effort-discount data doesn't show the predicted crossover.** Could mean (a) the framework is wrong, (b) the effort range was insufficient (check whether 85% MVC was actually reached), (c) the capability bound is more participant-specific than calibration captured. Investigate (b) and (c) before concluding (a).
- **jsPsych performance issues with the calibration task.** Handgrip integration requires either a real device (USB dynamometer with browser support is rare) or a proxy task (sustained key-pressing rate). Default to the proxy and document the deviation.

---

## 10. First action

When you start work on this repo, your first action is:

1. Read this entire document
2. Read `docs/derivation.md` if it exists (it won't on first run; create it from the math in Section 3 of this doc)
3. Confirm the build order in Section 4 with the user before starting Phase 0 if there are any ambiguities
4. Begin Phase 0

Do not begin coding until you have an explicit go-ahead for the first phase.
