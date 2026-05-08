/**
 * Key-press calibration task (Phase 4 substitute for handgrip MVC).
 *
 * Three 10-second maximal-rate spacebar trials separated by 30-second rests.
 * Per-trial rate = (number of valid presses) / trial duration. Take the
 * median across trials as `p_max` (the analog of B in the unified model).
 *
 * Anti-cheating measures (CLAUDE.md §9 raised these as expected concerns):
 *
 *  1. **Auto-repeat detection.** Browser key-repeat fires `keydown` only;
 *     a real human press fires `keydown` + `keyup`. We require alternating
 *     events.
 *  2. **Rate ceiling.** Above ~15 presses/sec is physically implausible for
 *     a typical participant. Trials with rates above the ceiling get flagged
 *     and excluded from the median.
 *  3. **Inter-press-interval regularity.** Macros fire at near-perfect
 *     intervals; humans have ~5-15% jitter. We compute the coefficient of
 *     variation of inter-press intervals; values below a threshold flag
 *     the trial.
 *
 * The task returns a `CalibrationResult` describing the median rate and the
 * per-trial detail. Callers (e.g. `effortDiscounting.ts`) read `pMax` and
 * scale effort levels to it.
 */

import type { JsPsych } from "jspsych";
import jsPsychHtmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";
import jsPsychInstructions from "@jspsych/plugin-instructions";

import {
  CALIBRATION_REST_SECONDS,
  CALIBRATION_TRIAL_SECONDS,
  CALIBRATION_TRIALS,
  MAX_PLAUSIBLE_RATE,
} from "../config";

export interface CalibrationTrial {
  /** Index, 0-based. */
  index: number;
  /** Number of valid key-presses. Excludes auto-repeat events. */
  validPresses: number;
  /** Trial duration in seconds (nominally CALIBRATION_TRIAL_SECONDS). */
  durationSec: number;
  /** Rate in presses/sec. */
  rate: number;
  /** Coefficient of variation of inter-press intervals (jitter). */
  ipiCv: number;
  /** True if any anti-cheating heuristic flagged the trial. */
  flagged: boolean;
  /** Human-readable reason if flagged. */
  flagReason?: string;
}

export interface CalibrationResult {
  /** Median rate across non-flagged trials. */
  pMax: number;
  /** Whether at least one trial passed all anti-cheating checks. */
  ok: boolean;
  /** All per-trial detail. */
  trials: CalibrationTrial[];
}

/**
 * Compute the coefficient of variation of inter-press intervals.
 * Returns 0 if fewer than 2 intervals are available (i.e. fewer than 3 presses).
 */
function ipiCoefficientOfVariation(timestamps: number[]): number {
  if (timestamps.length < 3) return 0;
  const intervals: number[] = [];
  for (let i = 1; i < timestamps.length; i++) {
    intervals.push((timestamps[i] as number) - (timestamps[i - 1] as number));
  }
  const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  if (mean === 0) return 0;
  const variance =
    intervals.reduce((acc, x) => acc + (x - mean) ** 2, 0) / intervals.length;
  return Math.sqrt(variance) / mean;
}

/**
 * Apply the anti-cheating heuristics to a single trial's measurements.
 *
 * Returns the same trial with `flagged` / `flagReason` filled in.
 */
export function flagTrial(trial: CalibrationTrial): CalibrationTrial {
  if (trial.rate > MAX_PLAUSIBLE_RATE) {
    return {
      ...trial,
      flagged: true,
      flagReason: `rate ${trial.rate.toFixed(2)} > ${MAX_PLAUSIBLE_RATE} presses/sec ceiling`,
    };
  }
  // CV below ~5% suggests macro/auto-repeat; humans rarely fall below ~10%.
  if (trial.validPresses >= 10 && trial.ipiCv < 0.05) {
    return {
      ...trial,
      flagged: true,
      flagReason: `inter-press interval CV ${trial.ipiCv.toFixed(3)} below 0.05 (suggests macro)`,
    };
  }
  return { ...trial, flagged: false };
}

/**
 * Build a single max-rate trial that uses jsPsych's html-keyboard-response
 * plugin. We listen on the document for keydown/keyup pairs and only count
 * a press when both events have fired since the previous press, defeating
 * browser auto-repeat (which fires keydown only).
 */
function buildMaxRateTrial(jsPsych: JsPsych, index: number) {
  const trialDurationMs = CALIBRATION_TRIAL_SECONDS * 1000;

  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div style="text-align: center;">
        <h2>Trial ${index + 1} of ${CALIBRATION_TRIALS}</h2>
        <p>Press the SPACE bar as fast as you can.</p>
        <p id="cal-counter" style="font-size: 3rem; font-weight: bold;">0</p>
        <p id="cal-timer" class="countdown">${CALIBRATION_TRIAL_SECONDS}</p>
      </div>
    `,
    choices: "NO_KEYS" as const,
    trial_duration: trialDurationMs,
    on_load: () => {
      const counter = document.getElementById("cal-counter");
      const timer = document.getElementById("cal-timer");
      const presses: number[] = [];
      let awaitingKeyup = false;

      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code !== "Space") return;
        e.preventDefault();
        if (awaitingKeyup) return; // ignore auto-repeat keydowns
        awaitingKeyup = true;
        presses.push(performance.now());
        if (counter) counter.textContent = String(presses.length);
      };
      const onKeyUp = (e: KeyboardEvent) => {
        if (e.code !== "Space") return;
        awaitingKeyup = false;
      };

      document.addEventListener("keydown", onKeyDown);
      document.addEventListener("keyup", onKeyUp);

      // Countdown timer display, ticked once per second.
      const start = performance.now();
      const tick = (): void => {
        const elapsed = (performance.now() - start) / 1000;
        const remaining = Math.max(0, CALIBRATION_TRIAL_SECONDS - Math.floor(elapsed));
        if (timer) timer.textContent = String(remaining);
        if (remaining > 0) setTimeout(tick, 250);
      };
      tick();

      // Stash the presses on the trial's data so on_finish can read them.
      // jsPsych v8 attaches data via the on_finish callback; we expose the
      // array on a globally-scoped key that on_finish below reads back.
      (jsPsych as unknown as { _calPresses?: number[] })._calPresses = presses;
      (jsPsych as unknown as { _calCleanup?: () => void })._calCleanup = () => {
        document.removeEventListener("keydown", onKeyDown);
        document.removeEventListener("keyup", onKeyUp);
      };
    },
    on_finish: (data: Record<string, unknown>) => {
      const j = jsPsych as unknown as {
        _calPresses?: number[];
        _calCleanup?: () => void;
      };
      const presses = j._calPresses ?? [];
      j._calCleanup?.();
      j._calPresses = [];

      const validPresses = presses.length;
      const durationSec = CALIBRATION_TRIAL_SECONDS;
      const rate = validPresses / durationSec;
      const ipiCv = ipiCoefficientOfVariation(presses);
      const trial = flagTrial({
        index,
        validPresses,
        durationSec,
        rate,
        ipiCv,
        flagged: false,
      });
      data.calibration_trial = trial;
    },
  };
}

function buildRestTrial(index: number) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div style="text-align: center;">
        <h2>Rest</h2>
        <p>Take a ${CALIBRATION_REST_SECONDS}-second break before trial ${index + 2}.</p>
        <p id="rest-timer" class="countdown">${CALIBRATION_REST_SECONDS}</p>
      </div>
    `,
    choices: "NO_KEYS" as const,
    trial_duration: CALIBRATION_REST_SECONDS * 1000,
    on_load: () => {
      const timer = document.getElementById("rest-timer");
      const start = performance.now();
      const tick = (): void => {
        const elapsed = (performance.now() - start) / 1000;
        const remaining = Math.max(0, CALIBRATION_REST_SECONDS - Math.floor(elapsed));
        if (timer) timer.textContent = String(remaining);
        if (remaining > 0) setTimeout(tick, 250);
      };
      tick();
    },
  };
}

/**
 * Aggregate per-trial calibration data into a final `CalibrationResult`.
 *
 * Returns the median rate across non-flagged trials. If every trial is
 * flagged, returns `ok: false` with `pMax = NaN` so the caller can decide
 * whether to retry, exclude the participant, or fall back.
 */
export function summariseCalibration(trials: CalibrationTrial[]): CalibrationResult {
  const valid = trials.filter((t) => !t.flagged);
  if (valid.length === 0) {
    return { pMax: Number.NaN, ok: false, trials };
  }
  const rates = valid.map((t) => t.rate).sort((a, b) => a - b);
  const mid = Math.floor(rates.length / 2);
  const pMax =
    rates.length % 2 === 0
      ? ((rates[mid - 1] as number) + (rates[mid] as number)) / 2
      : (rates[mid] as number);
  return { pMax, ok: true, trials };
}

/**
 * Build the full calibration timeline. Returns the jsPsych timeline array
 * to splice into the master timeline.
 */
export function buildCalibrationTimeline(jsPsych: JsPsych): unknown[] {
  const intro = {
    type: jsPsychInstructions,
    pages: [
      `<h2>Calibration</h2>
       <p>You'll see ${CALIBRATION_TRIALS} short timed trials.</p>
       <p>In each trial, press the SPACE bar as <strong>fast as you can</strong>
          for ${CALIBRATION_TRIAL_SECONDS} seconds. There is a ${CALIBRATION_REST_SECONDS}-second rest between trials.</p>
       <p>This sets your personal effort scale for the rest of the study,
          so please give your real maximum effort.</p>`,
      `<p>Place your dominant index finger on the SPACE bar.</p>
       <p>When you press <strong>Next</strong>, the first trial starts immediately.</p>`,
    ],
    show_clickable_nav: true,
  };

  const timeline: unknown[] = [intro];
  for (let i = 0; i < CALIBRATION_TRIALS; i++) {
    timeline.push(buildMaxRateTrial(jsPsych, i));
    if (i < CALIBRATION_TRIALS - 1) timeline.push(buildRestTrial(i));
  }
  return timeline;
}
