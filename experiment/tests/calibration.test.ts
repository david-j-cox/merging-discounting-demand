/**
 * Unit tests for the pure-function pieces of `tasks/calibration.ts`.
 * The full jsPsych timeline is exercised in the end-to-end test.
 */

import { describe, expect, it } from "vitest";

import { flagTrial, summariseCalibration, type CalibrationTrial } from "../src/tasks/calibration";
import { MAX_PLAUSIBLE_RATE } from "../src/config";

function makeTrial(overrides: Partial<CalibrationTrial> = {}): CalibrationTrial {
  return {
    index: 0,
    validPresses: 80,
    durationSec: 10,
    rate: 8,
    ipiCv: 0.15,
    flagged: false,
    ...overrides,
  };
}

describe("flagTrial", () => {
  it("does not flag a normal trial", () => {
    const t = flagTrial(makeTrial());
    expect(t.flagged).toBe(false);
    expect(t.flagReason).toBeUndefined();
  });

  it("flags rates above the physical-plausibility ceiling", () => {
    const t = flagTrial(makeTrial({ rate: MAX_PLAUSIBLE_RATE + 5 }));
    expect(t.flagged).toBe(true);
    expect(t.flagReason).toMatch(/ceiling/);
  });

  it("flags suspiciously regular inter-press intervals (auto-repeat / macro)", () => {
    const t = flagTrial(makeTrial({ ipiCv: 0.01 }));
    expect(t.flagged).toBe(true);
    expect(t.flagReason).toMatch(/macro/);
  });

  it("does not flag low-CV trials with too few presses to judge", () => {
    // 8 presses < 10-press threshold; CV check skipped.
    const t = flagTrial(makeTrial({ ipiCv: 0.01, validPresses: 8 }));
    expect(t.flagged).toBe(false);
  });
});

describe("summariseCalibration", () => {
  it("returns the median rate across non-flagged trials", () => {
    const trials = [
      makeTrial({ index: 0, rate: 6 }),
      makeTrial({ index: 1, rate: 7 }),
      makeTrial({ index: 2, rate: 9 }),
    ];
    const result = summariseCalibration(trials);
    expect(result.ok).toBe(true);
    expect(result.pMax).toBe(7);
  });

  it("excludes flagged trials from the median", () => {
    const trials = [
      makeTrial({ index: 0, rate: 6 }),
      makeTrial({ index: 1, rate: 99, flagged: true, flagReason: "macro" }),
      makeTrial({ index: 2, rate: 8 }),
    ];
    const result = summariseCalibration(trials);
    expect(result.ok).toBe(true);
    // median of {6, 8} = 7
    expect(result.pMax).toBe(7);
  });

  it("returns ok=false if every trial is flagged", () => {
    const trials = [
      makeTrial({ index: 0, rate: 99, flagged: true }),
      makeTrial({ index: 1, rate: 99, flagged: true }),
    ];
    const result = summariseCalibration(trials);
    expect(result.ok).toBe(false);
    expect(result.pMax).toBeNaN();
  });

  it("handles two-trial median (mean of the pair)", () => {
    const trials = [
      makeTrial({ index: 0, rate: 6 }),
      makeTrial({ index: 1, rate: 8 }),
    ];
    const result = summariseCalibration(trials);
    expect(result.pMax).toBe(7);
  });
});
