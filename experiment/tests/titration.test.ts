/**
 * Tests for `lib/titration.ts`. Verifies the Du-Green-Myerson 2002 algorithm:
 *
 *  1. The first offer is `rewardMax / 2`.
 *  2. Adjustments halve each step.
 *  3. With a true indifference value of `v`, the algorithm converges to
 *     within `rewardMax / 2^(n+1)` of `v` after `n` steps.
 *  4. Pure-`nextOffer` and stateful-`Titrator` agree.
 */

import { describe, expect, it } from "vitest";

import {
  Titrator,
  nextOffer,
  simulate,
  type TitrationStep,
} from "../src/lib/titration";
import { REWARD_MAX_USD, TITRATION_STEPS } from "../src/config";

describe("nextOffer", () => {
  it("returns rewardMax/2 for an empty history", () => {
    expect(nextOffer([], 10)).toBe(5);
  });

  it("raises the offer after an 'effortful' choice", () => {
    const history: TitrationStep[] = [{ step: 0, offer: 5, choice: "effortful" }];
    // After the first step the adjustment is rewardMax / 2^2 = 2.5.
    expect(nextOffer(history, 10)).toBe(7.5);
  });

  it("lowers the offer after an 'immediate' choice", () => {
    const history: TitrationStep[] = [{ step: 0, offer: 5, choice: "immediate" }];
    expect(nextOffer(history, 10)).toBe(2.5);
  });

  it("halves the adjustment magnitude each step", () => {
    let h: TitrationStep[] = [];
    let o = nextOffer(h, 10);
    expect(o).toBe(5); // start
    h = [...h, { step: 0, offer: o, choice: "effortful" }];
    o = nextOffer(h, 10);
    expect(o).toBe(7.5); // +2.5 = +10/4
    h = [...h, { step: 1, offer: o, choice: "immediate" }];
    o = nextOffer(h, 10);
    expect(o).toBe(6.25); // -1.25 = -10/8
  });
});

describe("Titrator", () => {
  it("requires exactly TITRATION_STEPS choices before producing a result", () => {
    const t = new Titrator(REWARD_MAX_USD, TITRATION_STEPS);
    for (let i = 0; i < TITRATION_STEPS - 1; i++) t.record("effortful");
    expect(() => t.result()).toThrow();
    t.record("immediate");
    expect(() => t.result()).not.toThrow();
  });

  it("rejects further choices after completion", () => {
    const t = new Titrator(10, 2);
    t.record("effortful");
    t.record("immediate");
    expect(t.isDone()).toBe(true);
    expect(() => t.record("effortful")).toThrow();
  });

  it("agrees with the pure nextOffer function", () => {
    const t = new Titrator(10, 4);
    const history: TitrationStep[] = [];
    while (!t.isDone()) {
      const offerFromTitrator = t.nextOffer();
      const offerFromPure = nextOffer(history, 10);
      expect(offerFromTitrator).toBe(offerFromPure);
      const choice = history.length % 2 === 0 ? "effortful" : "immediate";
      const step = t.record(choice);
      history.push(step);
    }
  });
});

describe("simulate", () => {
  /**
   * If a subject's true indifference is `v`, they choose `effortful` whenever
   * the offer is below `v` (the immediate option is worse than the effortful
   * one) and `immediate` whenever the offer is above `v`.
   */
  function chooserAtIndifference(v: number) {
    return (offer: number): "immediate" | "effortful" =>
      offer < v ? "effortful" : "immediate";
  }

  it("converges to within rewardMax / 2^(steps+1) of the true indifference", () => {
    const rewardMax = 10;
    const trueV = 3.7;
    for (const steps of [4, 5, 6, 8]) {
      const result = simulate(chooserAtIndifference(trueV), rewardMax, steps);
      const tolerance = rewardMax / 2 ** (steps + 1);
      // Halving-step convergence guarantees |estimate - true| <= tolerance
      expect(Math.abs(result.indifference - trueV)).toBeLessThanOrEqual(tolerance);
    }
  });

  it("recovers a wide range of indifference values", () => {
    const rewardMax = 10;
    for (const trueV of [0.5, 1.5, 4.0, 7.3, 9.5]) {
      const result = simulate(chooserAtIndifference(trueV), rewardMax, 6);
      expect(Math.abs(result.indifference - trueV)).toBeLessThan(0.5);
    }
  });

  it("records the right number of steps", () => {
    const result = simulate(() => "effortful", 10, 6);
    expect(result.steps).toHaveLength(6);
  });
});
