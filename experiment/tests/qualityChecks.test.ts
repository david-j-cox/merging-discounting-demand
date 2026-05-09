/**
 * Unit tests for the six subject-level data-quality checks. The
 * computeQualityFlags function takes a synthetic per-task payload and
 * returns boolean flags + numeric detail; we hand-build the inputs to
 * exercise each pass/fail path.
 */

import { describe, expect, it } from "vitest";

import {
  CATCH_TRIAL_EXPECTED_QUANTITY,
  DURATION_MAX_MS,
  DURATION_MIN_MS,
  RT_MAX_MS,
  RT_MIN_MS,
} from "../src/config";
import {
  computeQualityFlags,
  isRtOutsideBounds,
  type QualityFlags,
} from "../src/lib/qualityChecks";
import type { CalibrationResult } from "../src/tasks/calibration";
import type { EffortDiscountingResult } from "../src/tasks/effortDiscounting";
import type { PurchaseTaskResult } from "../src/tasks/purchaseTask";

// ---------- helpers to build "perfect-subject" inputs --------------

function goodCalibration(): CalibrationResult {
  return {
    pMax: 7,
    ok: true,
    trials: [
      { index: 0, validPresses: 60, durationSec: 10, rate: 6, ipiCv: 0.15, flagged: false },
      { index: 1, validPresses: 70, durationSec: 10, rate: 7, ipiCv: 0.13, flagged: false },
      { index: 2, validPresses: 80, durationSec: 10, rate: 8, ipiCv: 0.12, flagged: false },
    ],
  };
}

function goodTask1(): PurchaseTaskResult {
  return {
    commodity: "snack credit",
    trials: [
      { price: 0.01, quantity: 20 },
      { price: 1.0, quantity: 10 },
      // Catch trial passed: entered the expected number.
      { price: Number.NaN, quantity: CATCH_TRIAL_EXPECTED_QUANTITY, isCatch: true, failedCatch: false },
      { price: 100, quantity: 1 },
    ],
  };
}

function goodTask2(): EffortDiscountingResult {
  return {
    perFraction: [
      { fraction: 0.10, titration: { steps: [], indifference: 8.5, rewardMax: 10 } },
      { fraction: 0.40, titration: { steps: [], indifference: 5.0, rewardMax: 10 } },
      { fraction: 0.85, titration: { steps: [], indifference: 1.5, rewardMax: 10 } },
    ],
    allChoices: [
      { fraction: 0.10, targetRate: 0.7, step: 0, offer: 5, immediateCommodity: "snack_credit", choice: "effortful", rtMs: 1500, rtFlag: false },
      { fraction: 0.10, targetRate: 0.7, step: 1, offer: 7.5, immediateCommodity: "snack_credit", choice: "immediate", rtMs: 2000, rtFlag: false },
    ],
    verifications: [],
    pMaxUsed: 7,
    arm: "low",
    immediateCommodity: "snack_credit",
  };
}

const NORMAL_DURATION_MS = (DURATION_MIN_MS + DURATION_MAX_MS) / 2;

function compute(
  overrides: Partial<{
    calibration: CalibrationResult;
    task1: PurchaseTaskResult;
    task2: EffortDiscountingResult;
    durationMs: number;
  }> = {},
): QualityFlags {
  return computeQualityFlags({
    calibration: overrides.calibration ?? goodCalibration(),
    task1: overrides.task1 ?? goodTask1(),
    task2: overrides.task2 ?? goodTask2(),
    startedAtMs: 0,
    finishedAtMs: overrides.durationMs ?? NORMAL_DURATION_MS,
  });
}

// ---------- happy path ----------------------------------------------

describe("computeQualityFlags - all-passing baseline", () => {
  it("ok=true when every check passes", () => {
    const flags = compute();
    expect(flags.ok).toBe(true);
    expect(flags.calibrationOk).toBe(true);
    expect(flags.verificationOk).toBe(true);
    expect(flags.monotonicSvOk).toBe(true);
    expect(flags.rtBoundsOk).toBe(true);
    expect(flags.catchTrialOk).toBe(true);
    expect(flags.durationOk).toBe(true);
  });
});

// ---------- per-check failure paths ----------------------------------

describe("calibrationOk", () => {
  it("fails when fewer than 2 trials are unflagged", () => {
    const cal = goodCalibration();
    cal.trials[0]!.flagged = true;
    cal.trials[1]!.flagged = true;
    const flags = compute({ calibration: cal });
    expect(flags.calibrationOk).toBe(false);
    expect(flags.ok).toBe(false);
    expect(flags.detail.calibrationFlaggedCount).toBe(2);
  });

  it("passes when only 1 trial is flagged (2 of 3 valid)", () => {
    const cal = goodCalibration();
    cal.trials[0]!.flagged = true;
    expect(compute({ calibration: cal }).calibrationOk).toBe(true);
  });
});

describe("verificationOk", () => {
  it("fails when any verification trial failed", () => {
    const t2 = goodTask2();
    t2.verifications = [
      { fraction: 0.85, targetRate: 6, observedRate: 3, tolerance: 1.2, passed: false, nPresses: 90 },
    ];
    const flags = compute({ task2: t2 });
    expect(flags.verificationOk).toBe(false);
    expect(flags.detail.verificationFailureCount).toBe(1);
  });

  it("passes when no verification trials were run", () => {
    expect(compute().verificationOk).toBe(true);
  });
});

describe("monotonicSvOk", () => {
  it("fails when SV at higher effort exceeds SV at lower effort", () => {
    const t2 = goodTask2();
    // Make 0.40 indifference > 0.10 indifference (incoherent).
    t2.perFraction[1]!.titration.indifference = 9.5;
    const flags = compute({ task2: t2 });
    expect(flags.monotonicSvOk).toBe(false);
    expect(flags.detail.nonMonotonicPairs).toBeGreaterThan(0);
  });

  it("passes when SVs are strictly decreasing in effort", () => {
    expect(compute().monotonicSvOk).toBe(true);
  });
});

describe("rtBoundsOk", () => {
  it("fails when more than 25% of choice trials hit either bound", () => {
    const t2 = goodTask2();
    // Replace allChoices with 4 trials, 2 of which are RT-flagged (50%).
    t2.allChoices = [
      { fraction: 0.10, targetRate: 0.7, step: 0, offer: 5, immediateCommodity: "snack_credit", choice: "effortful", rtMs: 100, rtFlag: true },
      { fraction: 0.10, targetRate: 0.7, step: 1, offer: 5, immediateCommodity: "snack_credit", choice: "immediate", rtMs: 1500, rtFlag: false },
      { fraction: 0.10, targetRate: 0.7, step: 2, offer: 5, immediateCommodity: "snack_credit", choice: "effortful", rtMs: 90_000, rtFlag: true },
      { fraction: 0.10, targetRate: 0.7, step: 3, offer: 5, immediateCommodity: "snack_credit", choice: "immediate", rtMs: 2000, rtFlag: false },
    ];
    const flags = compute({ task2: t2 });
    expect(flags.rtBoundsOk).toBe(false);
    expect(flags.detail.rtBoundsHitCount).toBe(2);
    expect(flags.detail.rtBoundsHitFraction).toBe(0.5);
  });

  it("passes when no RT bounds are hit", () => {
    expect(compute().rtBoundsOk).toBe(true);
  });
});

describe("isRtOutsideBounds", () => {
  it("flags RTs below the lower bound", () => {
    expect(isRtOutsideBounds(RT_MIN_MS - 1)).toBe(true);
  });
  it("flags RTs above the upper bound", () => {
    expect(isRtOutsideBounds(RT_MAX_MS + 1)).toBe(true);
  });
  it("accepts RTs at the bounds inclusive", () => {
    expect(isRtOutsideBounds(RT_MIN_MS)).toBe(false);
    expect(isRtOutsideBounds(RT_MAX_MS)).toBe(false);
  });
});

describe("catchTrialOk", () => {
  it("fails when the catch trial entered the wrong quantity", () => {
    const t1 = goodTask1();
    const catchTrial = t1.trials.find((t) => t.isCatch);
    catchTrial!.quantity = 99;
    catchTrial!.failedCatch = true;
    expect(compute({ task1: t1 }).catchTrialOk).toBe(false);
  });

  it("fails when no catch trial was recorded at all", () => {
    const t1 = goodTask1();
    t1.trials = t1.trials.filter((t) => !t.isCatch);
    expect(compute({ task1: t1 }).catchTrialOk).toBe(false);
  });
});

describe("durationOk", () => {
  it("fails when session is too fast (< DURATION_MIN_MS)", () => {
    const flags = compute({ durationMs: DURATION_MIN_MS - 1 });
    expect(flags.durationOk).toBe(false);
  });

  it("fails when session is too slow (> DURATION_MAX_MS)", () => {
    const flags = compute({ durationMs: DURATION_MAX_MS + 1 });
    expect(flags.durationOk).toBe(false);
  });

  it("passes at the exact bounds", () => {
    expect(compute({ durationMs: DURATION_MIN_MS }).durationOk).toBe(true);
    expect(compute({ durationMs: DURATION_MAX_MS }).durationOk).toBe(true);
  });
});
