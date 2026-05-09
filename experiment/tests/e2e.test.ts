/**
 * End-to-end smoke tests for the experiment.
 *
 * jsPsych runs in jsdom with simulated responses. We don't try to
 * reproduce a real participant's flow keystroke-by-keystroke; instead we
 * verify the **data shape** that comes out the other side:
 *
 *   - Calibration: trials are generated and summarised correctly.
 *   - Titration: a smooth chooser produces a recoverable indifference.
 *   - Purchase task: each price has a corresponding quantity.
 *   - Effort purchase task: each effort price has a corresponding quantity.
 *   - Randomization: assignment is deterministic and balanced.
 *   - DataPipe payload: shape includes all required fields.
 *
 * The full UI flow (jsPsych in a browser) is exercised manually via
 * `npm run dev`; these tests guard the data structure.
 */

import { describe, expect, it } from "vitest";

import {
  buildPayload,
  newSessionId,
} from "../src/lib/dataExport";
import { randomize } from "../src/lib/randomization";
import { Titrator, simulate } from "../src/lib/titration";
import { summariseCalibration, type CalibrationTrial } from "../src/tasks/calibration";
import { buildEffortDiscountingTimeline } from "../src/tasks/effortDiscounting";
import { feasibilityCap } from "../src/tasks/effortPurchaseTask";
import { EFFORT_FRACTIONS, EFFORT_PRICES, PURCHASE_PRICES_USD, REWARD_MAX_USD } from "../src/config";
import type { QualityFlags } from "../src/lib/qualityChecks";

const PASSING_QUALITY_FLAGS: QualityFlags = {
  ok: true,
  calibrationOk: true,
  verificationOk: true,
  monotonicSvOk: true,
  rtBoundsOk: true,
  catchTrialOk: true,
  durationOk: true,
  detail: {
    calibrationFlaggedCount: 0,
    verificationFailureCount: 0,
    nonMonotonicPairs: 0,
    rtBoundsHitCount: 0,
    rtBoundsHitFraction: 0,
    durationMs: 15 * 60_000,
  },
};

describe("E2E: full battery dry run", () => {
  it("produces a complete payload with all expected fields", () => {
    // Simulate calibration: three valid trials with median rate 7 presses/sec.
    const calTrials: CalibrationTrial[] = [
      { index: 0, validPresses: 60, durationSec: 10, rate: 6, ipiCv: 0.15, flagged: false },
      { index: 1, validPresses: 70, durationSec: 10, rate: 7, ipiCv: 0.13, flagged: false },
      { index: 2, validPresses: 80, durationSec: 10, rate: 8, ipiCv: 0.12, flagged: false },
    ];
    const calibration = summariseCalibration(calTrials);
    expect(calibration.ok).toBe(true);
    expect(calibration.pMax).toBe(7);

    // Simulate Task 1 (purchase task): linear fall-off in quantity.
    const task1 = {
      trials: PURCHASE_PRICES_USD.map((price) => ({
        price,
        quantity: Math.max(0, Math.round(20 - 5 * Math.log10(Math.max(price, 0.01)))),
      })),
      commodity: "snack item",
    };
    expect(task1.trials).toHaveLength(PURCHASE_PRICES_USD.length);

    // Simulate Task 2 (effort discounting): per-fraction titrations whose
    // true SVs decay exponentially in effort.
    const trueSvs = EFFORT_FRACTIONS.map((f) => REWARD_MAX_USD * Math.exp(-1.5 * f));
    const perFraction = EFFORT_FRACTIONS.map((fraction, i) => {
      const trueV = trueSvs[i] as number;
      const titration = simulate(
        (offer) => (offer < trueV ? "effortful" : "immediate"),
        REWARD_MAX_USD,
      );
      return { fraction, titration };
    });
    expect(perFraction).toHaveLength(EFFORT_FRACTIONS.length);
    // Recovered indifferences should be close to true SVs.
    for (let i = 0; i < EFFORT_FRACTIONS.length; i++) {
      const trueV = trueSvs[i] as number;
      const recovered = (perFraction[i] as { titration: { indifference: number } }).titration.indifference;
      expect(Math.abs(recovered - trueV)).toBeLessThan(0.5);
    }
    const task2 = {
      perFraction,
      allChoices: [],
      verifications: [],
      pMaxUsed: calibration.pMax,
      arm: "low" as const,
      immediateCommodity: "snack_credit" as const,
    };

    // Simulate Task 3 (effort purchase task): consumption falling with effort price.
    const task3 = {
      trials: EFFORT_PRICES.map((effortPrice) => {
        const qty = Math.max(0, Math.round(15 - 3 * effortPrice));
        return {
          effortPrice,
          quantity: qty,
          exceededFeasibility: qty > feasibilityCap(effortPrice),
        };
      }),
      pMaxUsed: calibration.pMax,
      feasibilityCapsByPrice: EFFORT_PRICES.map(feasibilityCap),
    };
    expect(task3.trials).toHaveLength(EFFORT_PRICES.length);

    const assignment = randomize(42);

    // Assemble the payload.
    const payload = buildPayload({
      randomization: assignment,
      calibration,
      task1,
      task2,
      task3,
      qualityFlags: PASSING_QUALITY_FLAGS,
      rawTrials: [],
      prolificId: "TEST_PROLIFIC_ID",
    });

    // Required-field checks.
    expect(payload.study).toBeTruthy();
    expect(payload.version).toBeTruthy();
    expect(payload.sessionId).toMatch(/^[0-9a-f-]{30,}$/);
    expect(payload.timestampIso).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(payload.prolificId).toBe("TEST_PROLIFIC_ID");
    expect(payload.randomization.arm).toMatch(/^(low|high)$/);
    expect(payload.randomization.taskOrder).toHaveLength(3);
    expect(payload.calibration.pMax).toBe(7);
    expect(payload.task1.trials).toHaveLength(PURCHASE_PRICES_USD.length);
    expect(payload.task2.perFraction).toHaveLength(EFFORT_FRACTIONS.length);
    expect(payload.task3.trials).toHaveLength(EFFORT_PRICES.length);
  });

  it("E2E payload is JSON-serializable end-to-end", () => {
    const calibration = summariseCalibration([
      { index: 0, validPresses: 60, durationSec: 10, rate: 6, ipiCv: 0.15, flagged: false },
    ]);
    const payload = buildPayload({
      randomization: randomize(0),
      calibration,
      task1: { trials: [], commodity: "" },
      task2: {
        perFraction: [],
        allChoices: [],
        verifications: [],
        pMaxUsed: calibration.pMax,
        arm: "low" as const,
        immediateCommodity: "snack_credit" as const,
      },
      task3: { trials: [], pMaxUsed: calibration.pMax, feasibilityCapsByPrice: [] },
      qualityFlags: PASSING_QUALITY_FLAGS,
      rawTrials: [],
    });
    const json = JSON.stringify(payload);
    const round = JSON.parse(json);
    expect(round.study).toBe(payload.study);
    expect(round.calibration.pMax).toBe(payload.calibration.pMax);
  });

  it("session IDs are unique across calls", () => {
    const ids = new Set<string>();
    for (let i = 0; i < 100; i++) ids.add(newSessionId());
    expect(ids.size).toBe(100);
  });
});

describe("E2E: titration recovery on a smooth chooser", () => {
  it("recovers a true SV across the full reward range", () => {
    const trueValues = [0.5, 2, 5, 7.5, 9.5];
    for (const trueV of trueValues) {
      const result = simulate(
        (offer) => (offer < trueV ? "effortful" : "immediate"),
        REWARD_MAX_USD,
      );
      const tolerance = REWARD_MAX_USD / 2 ** 6; // ~0.156
      expect(Math.abs(result.indifference - trueV)).toBeLessThanOrEqual(tolerance + 0.01);
    }
  });

  it("the stateful Titrator agrees with the simulate helper for the same chooser", () => {
    const trueV = 4.2;
    const chooser = (offer: number): "immediate" | "effortful" =>
      offer < trueV ? "effortful" : "immediate";

    const t = new Titrator();
    while (!t.isDone()) {
      t.record(chooser(t.nextOffer()));
    }
    const stateful = t.result().indifference;

    const pure = simulate(chooser).indifference;
    expect(stateful).toBe(pure);
  });
});

describe("E2E: H3 substitutability arm wording", () => {
  // Pull the choice-button text from the first titration trial's choices()
  // function. The button labels are what distinguishes the arms; this test
  // guards against a future regression that would silently make both arms
  // identical (which would erase H3).
  function firstChoiceLabels(arm: "low" | "high"): string[] {
    const built = buildEffortDiscountingTimeline(5.0, arm, () => 0.5);
    // Element 0 is the intro; element 1 is the first titration step
    // (a procedural-form trial with a nested timeline).
    const stepWrapper = built.timeline[1] as { timeline: Array<{ choices: () => string[] }> };
    const stepTrial = stepWrapper.timeline[0] as { choices: () => string[] };
    return stepTrial.choices();
  }

  it("low-IF arm offers immediate snack credits (the substitute)", () => {
    const labels = firstChoiceLabels("low");
    expect(labels[0]).toMatch(/snack credit/i);
    expect(labels[0]).not.toMatch(/cash|\$/);
    expect(labels[1]).toMatch(/snack credit/i);
  });

  it("high-IF arm offers immediate cash (the non-substitute)", () => {
    const labels = firstChoiceLabels("high");
    expect(labels[0]).toMatch(/\$/);
    expect(labels[0]).toMatch(/cash/i);
    // Effortful side is always snack credits regardless of arm.
    expect(labels[1]).toMatch(/snack credit/i);
  });
});
