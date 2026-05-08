/**
 * Adjusting-amount titration (Du, Green & Myerson 2002).
 *
 * Used to find each subject's indifference reward (subjective value, SV) at
 * a given effort level. Standard procedure:
 *
 *   - Start with the immediate reward equal to half the maximum.
 *   - On each step, ask: "Would you do <effortful option> for $X, or take
 *     $immediate now?"
 *   - If they pick the effortful option, the immediate amount was too low;
 *     raise it by half the previous adjustment.
 *   - If they pick the immediate option, the immediate amount was too high;
 *     lower it by half the previous adjustment.
 *   - After `TITRATION_STEPS` steps, the indifference value is the immediate
 *     amount that would have been offered on the (n+1)-th step.
 *
 * The algorithm is deterministic given the choice sequence; we expose a pure
 * function `nextOffer` plus a stateful `Titrator` so the orchestration code
 * can drive it from jsPsych one trial at a time.
 */

import { REWARD_MAX_USD, TITRATION_STEPS } from "../config";

/** A single titration step: what was offered, what the subject chose. */
export interface TitrationStep {
  /** Step index, 0-based. Total step count is `TITRATION_STEPS`. */
  step: number;
  /** Immediate-reward amount offered on this step (USD). */
  offer: number;
  /**
   * Subject's choice on this step.
   *
   *  - `"immediate"` — they chose the immediate (effortless) reward.
   *  - `"effortful"` — they chose the effortful option.
   */
  choice: "immediate" | "effortful";
}

export interface TitrationResult {
  /** All steps in order. */
  steps: TitrationStep[];
  /** Inferred indifference reward after the final step (USD). */
  indifference: number;
  /** Maximum reward used as the starting bracket. */
  rewardMax: number;
}

/**
 * Compute the offer for the next step from the history so far.
 *
 * If `history` is empty, return the starting offer (`rewardMax / 2`).
 *
 * @param history All steps completed so far, in order.
 * @param rewardMax The bracket maximum (typically `REWARD_MAX_USD`).
 */
export function nextOffer(history: TitrationStep[], rewardMax: number = REWARD_MAX_USD): number {
  if (history.length === 0) return rewardMax / 2;
  // Standard halving-step rule: each adjustment is half the previous.
  // Starting adjustment magnitude on step 1 is `rewardMax / 4`.
  const lastStep = history[history.length - 1] as TitrationStep;
  const adjustment = rewardMax / 2 ** (history.length + 1);
  return lastStep.choice === "effortful"
    ? lastStep.offer + adjustment
    : lastStep.offer - adjustment;
}

/**
 * Stateful driver. Call `nextOffer()` to get the next amount to display,
 * then `record(choice)` to commit the subject's response. Stops automatically
 * after `TITRATION_STEPS` steps.
 */
export class Titrator {
  private readonly history: TitrationStep[] = [];

  constructor(
    public readonly rewardMax: number = REWARD_MAX_USD,
    public readonly maxSteps: number = TITRATION_STEPS,
  ) {}

  /** True if no further steps should be presented. */
  isDone(): boolean {
    return this.history.length >= this.maxSteps;
  }

  /** Return the offer for the next (still-pending) step. */
  nextOffer(): number {
    if (this.isDone()) {
      throw new Error("Titration is complete; cannot request another offer.");
    }
    return nextOffer(this.history, this.rewardMax);
  }

  /** Record the subject's choice for the current step and advance. */
  record(choice: "immediate" | "effortful"): TitrationStep {
    if (this.isDone()) {
      throw new Error("Titration is complete; cannot record another choice.");
    }
    const offer = this.nextOffer();
    const step: TitrationStep = {
      step: this.history.length,
      offer,
      choice,
    };
    this.history.push(step);
    return step;
  }

  /** Final indifference value: the offer that would have been presented next. */
  result(): TitrationResult {
    if (!this.isDone()) {
      throw new Error("Titration is not complete; cannot compute the result yet.");
    }
    // The "would-be next offer" is the conventional indifference estimate.
    const indifference = nextOffer(this.history, this.rewardMax);
    return {
      steps: [...this.history],
      indifference,
      rewardMax: this.rewardMax,
    };
  }
}

/**
 * Convenience: run a titration to completion against an arbitrary chooser
 * function. Useful for E2E testing with simulated subjects.
 *
 * @param chooseFn A function that, given the current offer, returns the
 *   subject's choice. (Typically a smooth function of `offer / true_SV`.)
 */
export function simulate(
  chooseFn: (offer: number) => "immediate" | "effortful",
  rewardMax: number = REWARD_MAX_USD,
  maxSteps: number = TITRATION_STEPS,
): TitrationResult {
  const t = new Titrator(rewardMax, maxSteps);
  while (!t.isDone()) {
    const offer = t.nextOffer();
    t.record(chooseFn(offer));
  }
  return t.result();
}
