/**
 * Subject-level data-quality checks.
 *
 * Six checks run after the battery completes; results land in
 * ``QualityFlags`` and are attached to the session payload. Nothing
 * here triggers automatic exclusion — exclusion decisions are made at
 * analysis time per the pre-registered analysis plan. Per-trial flags
 * (set by individual task code) are aggregated here into subject-level
 * summaries.
 */

import {
  DURATION_MAX_MS,
  DURATION_MIN_MS,
  RT_BOUND_SUBJECT_FRACTION,
  RT_MAX_MS,
  RT_MIN_MS,
} from "../config";
import type { CalibrationResult } from "../tasks/calibration";
import type { EffortDiscountingResult } from "../tasks/effortDiscounting";
import type { PurchaseTaskResult } from "../tasks/purchaseTask";

export interface QualityFlags {
  /** True if the subject passes every quality check (convenience aggregate). */
  ok: boolean;
  /** True if at least 2 of N calibration trials produced valid (non-flagged) data. */
  calibrationOk: boolean;
  /** True if no verification trial they "agreed to" failed the rate-tolerance check. */
  verificationOk: boolean;
  /** True if the per-fraction SV trajectory is non-increasing in effort. */
  monotonicSvOk: boolean;
  /** True if fewer than RT_BOUND_SUBJECT_FRACTION of Task 2 choice trials hit either RT bound. */
  rtBoundsOk: boolean;
  /** True if the Task 1 catch trial produced exactly the expected quantity. */
  catchTrialOk: boolean;
  /** True if total session length is between DURATION_MIN_MS and DURATION_MAX_MS. */
  durationOk: boolean;
  /** Per-check numeric details for downstream analysis / methods reporting. */
  detail: {
    calibrationFlaggedCount: number;
    verificationFailureCount: number;
    nonMonotonicPairs: number;
    rtBoundsHitCount: number;
    rtBoundsHitFraction: number;
    durationMs: number;
  };
}

/**
 * Compute all six quality flags from the per-task results plus the
 * session start timestamp. ``startedAtMs`` is the wall-clock millis when
 * the participant began (e.g. after consent), used for duration bounds.
 */
export function computeQualityFlags(args: {
  calibration: CalibrationResult;
  task1: PurchaseTaskResult;
  task2: EffortDiscountingResult;
  startedAtMs: number;
  finishedAtMs: number;
}): QualityFlags {
  const { calibration, task1, task2 } = args;

  // 1. Calibration: at least 2 of N trials produced valid data.
  const calibrationFlaggedCount = calibration.trials.filter((t) => t.flagged).length;
  const validCount = calibration.trials.length - calibrationFlaggedCount;
  const calibrationOk = validCount >= 2;

  // 2. Verification trials: any failure flags the subject. ``task2.verifications``
  //    is populated by the verification trial's on_finish; usually 0-1 records.
  const verificationFailureCount = task2.verifications.filter((v) => !v.passed).length;
  const verificationOk = verificationFailureCount === 0;

  // 3. Non-monotonic SV (adapted Johnson-Bickel): for the sequence of
  //    indifference values across effort fractions (sorted ascending),
  //    SV(higher effort) must be <= SV(lower effort). Count violations.
  const fractions = [...task2.perFraction].sort((a, b) => a.fraction - b.fraction);
  let nonMonotonicPairs = 0;
  for (let i = 1; i < fractions.length; i++) {
    const prev = fractions[i - 1];
    const cur = fractions[i];
    if (!prev || !cur) continue;
    if (cur.titration.indifference > prev.titration.indifference + 1e-9) {
      nonMonotonicPairs++;
    }
  }
  const monotonicSvOk = nonMonotonicPairs === 0;

  // 4. RT bounds: per-trial flag was set in task2.allChoices when the
  //    subject's RT fell outside [RT_MIN_MS, RT_MAX_MS].
  const rtBoundsHitCount = task2.allChoices.filter((c) => c.rtFlag === true).length;
  const rtBoundsHitFraction =
    task2.allChoices.length > 0 ? rtBoundsHitCount / task2.allChoices.length : 0;
  const rtBoundsOk = rtBoundsHitFraction < RT_BOUND_SUBJECT_FRACTION;

  // 5. Catch trial: per-trial flag set on the catch trial in Task 1.
  const catchTrial = task1.trials.find((t) => t.isCatch === true);
  const catchTrialOk = catchTrial !== undefined && catchTrial.failedCatch === false;

  // 6. Duration bounds.
  const durationMs = args.finishedAtMs - args.startedAtMs;
  const durationOk = durationMs >= DURATION_MIN_MS && durationMs <= DURATION_MAX_MS;

  const ok =
    calibrationOk && verificationOk && monotonicSvOk && rtBoundsOk && catchTrialOk && durationOk;

  return {
    ok,
    calibrationOk,
    verificationOk,
    monotonicSvOk,
    rtBoundsOk,
    catchTrialOk,
    durationOk,
    detail: {
      calibrationFlaggedCount,
      verificationFailureCount,
      nonMonotonicPairs,
      rtBoundsHitCount,
      rtBoundsHitFraction,
      durationMs,
    },
  };
}

/** Bound a Task 2 RT against [RT_MIN_MS, RT_MAX_MS]. Used per-trial in the choice on_finish. */
export function isRtOutsideBounds(rtMs: number): boolean {
  return rtMs < RT_MIN_MS || rtMs > RT_MAX_MS;
}
