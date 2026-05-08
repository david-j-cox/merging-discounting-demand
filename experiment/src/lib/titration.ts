/**
 * Du, Green & Myerson (2002) adjusting-amount titration.
 *
 * Halving-step bisection: start at `rewardMax / 2`, adjust by half the
 * previous step toward (effortful chosen) or away from (immediate chosen)
 * the effortful option. After `TITRATION_STEPS` choices the indifference
 * estimate is the offer that would be presented next.
 */

import { REWARD_MAX_USD, TITRATION_STEPS } from "../config";

/** A single titration step: what was offered, what the subject chose. */
export interface TitrationStep {
  /** Step index, 0-based; total step count is `TITRATION_STEPS`. */
  step: number;
  /** Immediate-reward amount offered on this step (USD). */
  offer: number;
  /** `"immediate"` chose the effortless option; `"effortful"` chose the effortful one. */
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

/** Next offer from history; ``rewardMax / 2`` when history is empty. */
export function nextOffer(history: TitrationStep[], rewardMax: number = REWARD_MAX_USD): number {
  if (history.length === 0) return rewardMax / 2;
  const lastStep = history[history.length - 1] as TitrationStep;
  const adjustment = rewardMax / 2 ** (history.length + 1);
  return lastStep.choice === "effortful"
    ? lastStep.offer + adjustment
    : lastStep.offer - adjustment;
}

/** Stateful driver: ``nextOffer()`` to peek, ``record(choice)`` to advance. */
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

/** Run a titration to completion against a chooser; useful for E2E tests. */
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
