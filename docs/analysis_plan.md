# Analysis plan

> Pre-registered analysis for the main study (Phase 6). This plan is the
> result of the Phase 3 simulation suite, which exposed identifiability and
> heterogeneity issues that the original CLAUDE.md §5 hypotheses did not
> anticipate. Updated hypotheses are *more conservative*, not weaker —
> the simulation suite earned them.

This plan locks down: what we will fit, how we will compare models, how
we will handle stratification, and what counts as supporting / refuting
each hypothesis. It will be filed on OSF before the main study begins
(CLAUDE.md §4 Phase 5 step 4).

---

## 1. Pre-processing

Each participant contributes:

* **Task 1** — purchase task (17-price array). Filter prices where
  consumption is suspect (e.g. `Q > Q0` by more than 2σ).
* **Task 2** — effort discounting (six effort levels at 10/25/40/55/70/85%
  of measured capability). Apply standard Johnson–Bickel attention checks
  per Du, Green & Myerson (2002).
* **Task 3** — effort purchase task (effort-denominated price array).
  Calibrate the price array per-subject during piloting.

Subjects who fail attention checks or who have insufficient response
variability are excluded *before* model fitting. Exclusion rules are
locked in this plan and will not be modified post-hoc.

---

## 2. Per-subject fitting

For each retained subject we fit:

| model | params estimated | params held | data source |
|---|---|---|---|
| Koffarnus single-task | `Q0`, `α` | `k = pop. median` | Task 1 alone |
| UnifiedCapabilityBounded joint | `Q0`, `α`, `k` | `B = MVC / N-back threshold` | Tasks 1 + 2 |
| Independent-α | `Q0_1`, `Q0_3`, `α_1`, `α_3` | `k = pop. median` | Tasks 1 + 3 |
| Shared-α | `Q0_1`, `Q0_3`, `α_shared` | `k = pop. median` | Tasks 1 + 3 |
| Power discount | `A`, `λ`, `s` | none | Task 2 alone |
| Exponential discount | `A`, `λ` | none | Task 2 alone |
| Hyperbolic discount | `A`, `λ` | none | Task 2 alone |
| Parabolic discount | `A`, `λ` | none | Task 2 alone |
| Hyperboloid discount | `A`, `λ`, `s` | none | Task 2 alone |
| Sigmoidal discount | `A`, `s`, `E0` | none | Task 2 alone |

`k` is **shared across participants** — Hursh's standard practice in the
demand-curve literature. In the NLS fits this means a single value (the
pilot-estimated population median) is held constant; in the hierarchical
Bayesian fit it means a single `k` parameter is sampled with a population
prior rather than a per-subject deviate. Phase 3 simulations confirmed
this is what the data prefer: fitting `k` free per subject produces an
α-k identifiability problem (α RMSE goes from ~14% with fixed k to ~57%
with free k).

`B` is anchored to each subject's measured capability (handgrip MVC for
physical effort, N-back threshold for cognitive effort), per CLAUDE.md
§9 fallback.

All fits are NLS in this plan. A hierarchical Bayesian re-analysis is a
secondary, exploratory step (CLAUDE.md §5 specifies NumPyro priors but
defers them to a later phase).

---

## 3. Hypotheses

### H1 — Parameter linkage (Tasks 1 ↔ 3)

**Claim.** A unified model with one `α` shared between Task 1 (purchase
task) and Task 3 (effort purchase task) fits the joint per-subject data
as well or better than a model with independent `α` per task.

**Test statistic.** Per-subject ΔBIC = BIC(independent) − BIC(shared).
Positive ΔBIC favours the shared model.

**Population-level summary.** Mean ΔBIC across subjects, plus the
proportion of subjects with ΔBIC > 0.

**Decision rule.**
* H1 **supported** if mean ΔBIC > 6 (decisive evidence by convention)
  AND ≥ 80% of subjects show ΔBIC > 0.
* H1 **refuted** if mean ΔBIC < 0 OR < 50% of subjects show ΔBIC > 0.
* Otherwise **inconclusive**: report as such.

Phase 3 simulations on synthetic data (where `α` is exactly shared by
construction) showed mean ΔBIC ≈ +85 and 96–98% favouring shared. The
test should easily detect the effect if it exists in real data; the
decision rule above is calibrated to allow for the strong signal but
tolerate ~15% of subjects looking different from the population.

### H2 — Functional-form crossover (Task 2 narrow vs wide)

> **Note (Phase 3).** The original CLAUDE.md §5 H2 ("narrow → exponential,
> wide → power") fails under between-subject heterogeneity in capability.
> The reformulated version below is conditional on each subject's
> crossover position and is what the simulation actually supports.

**Reformulated claim.** *Among subjects whose model-implied crossover
effort* `E*` *lies between* `0.3·B` *and* `0.85·B`, the AIC-best discount
function on Task 2 data shifts from convex shapes (Hyperbolic,
Exponential, Hyperboloid) to concave shapes (Parabolic, Power) when the
effort range is wide vs narrow.

**Stratification.** After the unified-model fit produces `α`, `Q0`, `k`
per subject, compute the implied `E*` and the ratio `E*/B`. Three strata:

* `E*/B < 0.3` — capability binds throughout. Hyperbolic-dominated;
  manipulation cannot affect winning form. **Excluded from H2 test.**
* `0.3 ≤ E*/B ≤ 0.85` — manipulation window. **Primary H2 test.**
* `E*/B > 0.85` — capability never binds. Exponential-dominated.
  **Excluded from H2 test.**

Pre-register the strata definitions and the expected stratum sizes from
the pilot (Phase 5). If the manipulation-window stratum has < 30 subjects,
H2 is underpowered and the result is reported as inconclusive rather than
treated as a refutation.

**Test statistic.** Within the manipulation-window stratum, the
proportion of subjects with a concave AIC-winner under wide vs narrow
sampling.

**Decision rule (within stratum).**
* H2 **supported** if `prop_concave(wide) − prop_concave(narrow) > 0.20`
  (an effect size larger than Phase 3 between-subject noise).
* H2 **refuted** if the difference is non-positive.
* Otherwise **inconclusive**.

### H3 — Substitutability (between-subjects arm)

**Claim.** Effort-discount steepness (`α` from the unified-model fit) is
greater in the low-substitutability arm than in the high-substitutability
arm.

**Test statistic.** Between-subjects t-test on individual `α` estimates,
with prior on the effect size from Hursh's substitutability work
(typically a 1.5–2× ratio).

**Decision rule.**
* H3 **supported** if the 95% credible interval on the arm difference
  excludes zero in the predicted direction.

---

## 4. Secondary analyses

* **Linkage with the predicted transformation** — H1's stronger form:
  not just that `α` is shared, but that it relates to Task 2 steepness
  through the model-specified mapping. This is a more stringent test
  than H1 itself.
* **`B` ↔ measured capability** — Phase 3 fixed `B` at its true value
  in simulations. In real data, `B` is anchored at MVC/N-back. We will
  compare the implied `B` from a model fit with `B` left free against
  the measured capability; correlation > 0 is supporting evidence for
  Option B over Option A (which has no `B` parameter at all).
* **Modality replication** — H2 should replicate across physical and
  cognitive effort modalities. If it works for one but not the other,
  that's a finding too.

---

## 5. Stopping rules

Per CLAUDE.md §5: **if H1 fails, the unification claim is empirically
falsified. The paper publishes the negative result.** Same applies if
the secondary linkage-with-transformation test fails — the unification's
stronger form is wrong, and we report it.

H2 reformulation by stratum was an explicit response to a Phase 3
finding; we will not relax it further post-hoc.

---

## 6. Inference framework

NLS for the primary analyses (decision rules above). Hierarchical
Bayesian re-analysis with NumPyro is a secondary exploratory step,
specified in detail when we get to Phase 6 (priors, posteriors, PPCs).
Pre-registration covers the NLS analyses; the Bayesian analysis will
be reported as exploratory.

---

## 7. What was learned in Phase 3 that this plan absorbs

1. Free-`k` fits suffer α-k identifiability problems. **`k` fixed.**
2. Joint-task fitting recovers `α` ~8x more precisely than single-task
   fitting. **The unified joint-task fit is the primary fit.**
3. H2 fails at the population level under reasonable priors because
   most subjects' crossover lies below the experimental range.
   **H2 reformulated with stratification by `E*/B`.**
4. H1's signal is very strong on synthetic data, but the test cannot
   distinguish "exactly shared α" from "approximately shared α".
   **The decision rule on H1 admits this and treats values > 6 as
   "consistent with H1" rather than "proves H1".**
