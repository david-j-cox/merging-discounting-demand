# Derivation of the unified effort–demand model

> Companion to `src/edu/models/unified.py`. The code mirrors the steps below.
> Anything that depends on a load-bearing assumption is flagged with **A1**,
> **A2**, etc. and revisited in §6 (assumptions and limitations).

This document derives the unification of behavioural-economic demand
(Hursh & Silberberg 2008; Koffarnus et al. 2015) with effort discounting
(Hartmann 2013; Bialaszek 2017; Klein-Flügge 2015) along the lines of
Hursh & Schwartz (2022) for delay discounting. The empirical claim is that
effort discounting is a special case of constrained two-commodity demand
where effort capacity is the binding constraint, and that the
functional-form debate in the effort literature is an artefact of different
experiments probing different regions of the constraint.

---

## Step 1 — Setup: effort as a price

A decision-maker chooses between two commodities of identical reward
magnitude `A`:

- **Commodity 1:** immediate, effortless. Always available at zero effort
  cost. Indexed `i = 1`.
- **Commodity 2:** delayed by an effort requirement `E`. Indexed `i = 2`.

We treat **effort as a price denominated in effort per unit reward**:

> **A1 (effort-as-price)** — The effective price of commodity 2 is
> `P_effort = E / A`. Mirrors the Hursh-Schwartz (2022) treatment of delay
> (time per unit reward).

Under A1, demand for each commodity follows the standard Hursh–Silberberg
(or Koffarnus) exponential form:

```
Q_i = Q_{0,i} · 10^( k · ( exp( −α_i · Q_{0,i} · P_i ) − 1 ) )
```

with `P_1 = 0` (effortless) and `P_2 = E / A` (effortful).

> **A2 (shared parameters)** — `α_1 = α_2 = α` and `k` is shared across
> commodities. Defended in §6: in the experimental design both commodities
> deliver the same reward type, so demand-curve shape is a property of the
> reward (not the access path).

> **A3 (Q0 normalisation)** — `Q_{0,1} = Q_{0,2} = Q_0`. Each commodity is
> the *same* reward made accessible by a different cost-of-acquisition; at
> zero effective price the participant has no preference between them, so
> the unconstrained consumption rates must coincide. This normalisation is
> what makes the unification one-parameter rather than two.

---

## Step 2 — Subjective value as relative demand

The subjective value of the effortful commodity is the rate at which a
participant trades it against the effortless commodity. Under
indifference-style adjusting-amount procedures the empirical readout is

```
SV(E) / A = Q_2(E) / Q_{0,2}
```

— the ratio of constrained to unconstrained demand for the effortful
option. This is the mapping from "consumption" units back to subjective
value, and follows directly from A2/A3: at `E = 0`, `Q_2 = Q_0` so
`SV = A`. As `E` grows, `Q_2` falls, dragging `SV` down with it.

> **A4 (SV-from-demand mapping)** — Subjective value equals the fraction of
> baseline demand that the effortful commodity retains under cost. This
> mirrors the standard adjusting-amount titration logic: a participant
> reports `SV` by adjusting an immediate-reward alternative until indifferent
> with the effortful option.

---

## Step 3 — Substitute the effort price

Substituting `P_2 = E / A` into the Hursh–Silberberg form for commodity 2:

```
Q_2(E) = Q_0 · 10^( k · ( exp( −α · Q_0 · E / A ) − 1 ) )
```

Dividing by `Q_0`:

```
Q_2(E) / Q_0 = 10^( k · ( exp( −α · Q_0 · E / A ) − 1 ) )
```

Combining with A4:

---

## Step 4 — Naive unified form

```
SV(E) / A  =  10^( k · ( exp( −α · Q_0 · E / A ) − 1 ) )
```

This is the **naive** unified discount function. Properties:

- `SV(0) = A` (at zero effort the reward is undiscounted).
- `SV(E)` is monotonically decreasing in `E`.
- `SV(E) → A · 10^(−k)` as `E → ∞`: the function asymptotes to a *non-zero*
  floor, never crossing zero.

The asymptote is the central problem. Empirically, both physical (Hartmann
2013) and cognitive (Westbrook 2013) effort produce SV functions that
**reach zero at finite effort** — participants refuse the effortful
commodity entirely once the effort exceeds some threshold. The naive form
cannot represent this. We need the capability bound.

---

## Step 5 — Capability-bounded form (Option B)

Introduce an effort budget `B` ∈ (0, ∞), interpreted as the participant's
effort capability (e.g. handgrip MVC, sustained N-back threshold). The
capability bound enters as a *physical-feasibility constraint* on the rate
at which the effortful commodity can be acquired:

```
Q_2(E) ≤ B / E
```

— at effort `E` per acquisition, the participant can perform at most `B/E`
acquisitions per unit time before exhausting capability.

Combined with the demand-curve solution from Step 3, the constrained demand
is the **minimum** of the two:

```
Q_2(E)  =  min[ Q_0 · 10^( k · ( exp( −α · Q_0 · E / A ) − 1 ) ),  B / E ]
```

Equivalently, since `SV(E)/A = Q_2(E)/Q_0`:

```
SV(E) / A  =  min[ 10^( k · ( exp( −α · Q_0 · E / A ) − 1 ) ),  B / (Q_0 · E) ]
```

### Regime structure

Two regimes, separated by a crossover effort `E*`:

- **Unconstrained regime** (`E < E*`) — the demand-curve term binds. SV
  follows the exponential-like decay of the naive form.
- **Truncated regime** (`E ≥ E*`) — the capability term binds. SV decays as
  `B/(Q_0·E)`, a hyperbolic in `E`, and reaches zero only as `E → ∞`.

The crossover `E*` is the first `E > 0` (scanning upward from zero) at
which the two terms are equal:

```
Q_0 · 10^( k · ( exp( −α · Q_0 · E* / A ) − 1 ) )  =  B / E*
```

This has no closed form. The implementation uses a one-dimensional root
finder (Brent's method on a logarithmic bracketing scan).

> **Note on multiple roots.** The capability term `B/E` falls monotonically
> in `E`, but the demand term flattens to a *non-zero* asymptote
> `Q_0 · 10^(-k)` (Step 4). If that asymptote sits above `B/E_max` for
> some large `E_max`, the two curves re-cross at very large `E` (capability
> re-binds after demand saturates). The implementation returns the *first*
> crossing as the experimentally relevant one — that is the effort a
> participant traverses as their effort requirement grows from zero. The
> second crossing, if it exists, is in the asymptotic regime far beyond any
> realistic effort and has no behavioural meaning.

> **Why "Option B"?** This formulation puts the capability bound on the
> *quantity* axis (consumption rate). Option A (§3.3 below) instead applies
> a nonlinear transformation to the effort axis. Option B is preferred
> theoretically because it gives the capability bound a direct
> behavioural-physical interpretation — `B` is something we measure
> (handgrip MVC, N-back threshold) rather than infer.

### What the capability bound buys us

Empirical effort-discount functions show three apparent regimes across the
literature:

1. **Low-effort experiments** (e.g. Treadway EEfRT, modest grip force)
   produce **exponential-like** decay. ✔ The Option B model in its
   unconstrained regime reproduces this.
2. **Mid-range experiments** with a fixed reward across a wide effort range
   produce **hyperbolic-like** decay. ✔ The crossover regime spends time
   tracking `B/(Q_0·E)`, which is a hyperbolic.
3. **High-effort experiments** approaching capability produce **abrupt
   refusal at finite E**. ✔ The truncated regime reaches arbitrarily small
   `SV` as `E` grows; with realistic noise floors, refusal occurs at finite
   `E`.

Phase 3's simulation suite tests this prediction quantitatively: simulate
data from Option B at varying `(α, B)` regimes and verify that single-form
discount functions (hyperbolic, exponential, parabolic, power) win the
AIC race in the regions of effort space CLAUDE.md predicts.

---

## §3.3 — Option A: nonlinear-cost form

For comparison we also implement an alternative formulation in which the
*subjective effort* `ψ(E) = E^s` differs from the objective effort, while
demand remains exactly Hursh–Silberberg in this transformed effort:

```
SV(E) / A  =  10^( k · ( exp( −α · Q_0 · E^s / A ) − 1 ) )
```

The exponent `s > 1` produces concave SV(E) (parabolic-like); `s < 1`
produces convex SV(E) (sigmoidal-like).

Both options are testable: they make different predictions about how `B`
correlates with measured capability (Option B says it should; Option A has
no `B` parameter at all). Phase 6 main-study analyses (CLAUDE.md §5) will
adjudicate.

---

## Step 6 — Substitutability and the IF parameter

Hursh's interaction factor `IF` parameterises substitutability between the
two commodities. When commodity 1 is consumed, it reduces the effective
demand for commodity 2 in proportion to how substitutable they are:

```
Q_2_effective(E) = Q_2(E) · ( 1 − IF · Q_1_consumed )
```

- `IF = 1` — perfect substitutes; consuming commodity 1 fully eliminates
  any demand for commodity 2.
- `IF = 0` — independence; consumption of commodity 1 has no bearing on
  commodity 2.
- `0 < IF < 1` — partial substitutes; the empirically interesting range.

`IF` enters the experimental design via the substitutability arm
(CLAUDE.md §4 Phase 4 step 6, between-subjects 50/50). High-substitutability
arms supply commodity 1 alongside the effortful option; low-substitutability
arms enforce a delay or category mismatch that prevents commodity 1 from
substituting for commodity 2.

The unified-model fit can either estimate `IF` from cross-price data
(when the experimental design includes shared reward consumption) or hold
it fixed by arm assignment. The implementation supports both.

---

## §6 — Assumptions and limitations

| ID  | Assumption                                          | Risk if wrong                                                        |
|-----|-----------------------------------------------------|----------------------------------------------------------------------|
| A1  | `P_effort = E / A` (effort per unit reward)         | Misspecifies the price scaling; would shift fitted α systematically.|
| A2  | Shared `α`, `k` across commodities                  | Unifying claim falls apart; need two-α model.                        |
| A3  | `Q_{0,1} = Q_{0,2} = Q_0`                           | Adds one nuisance parameter; testable post hoc by relaxing.          |
| A4  | `SV/A = Q_2/Q_0` (SV-from-demand mapping)           | Re-derivation needed; would change the linkage prediction.           |
| A5  | Effort capability `B` is well-defined and measured  | Identifiability suffers; can fix `B` from MVC/N-back per §9.         |

A1–A4 are the load-bearing assumptions of the unification. A5 is the
practical anchor that makes the model identifiable in the experimental
design.

The Phase 3 simulation suite (CLAUDE.md §4 step 2) checks identifiability
between `α`, `B`, and the cost-shape parameter `s` (Option A). If any pair
shows a flat ridge in the likelihood surface, A5 becomes mandatory — we
must fix `B` from measured capability rather than estimate it freely.

---

## References

See CLAUDE.md §7 for the primary references. The structure of this
derivation parallels Hursh & Schwartz (2022) for delay discounting; the
substitution at Step 1 (effort price = `E/A` rather than time price = `D/A`)
is the only conceptual change.
