/**
 * Key-press calibration task. Phase 4 substitute for handgrip MVC.
 *
 * Three 10-second max-rate spacebar trials with 30-second rests; the
 * median rate becomes `pMax`. Three anti-cheating measures:
 *
 * 1. **Auto-repeat detection.** Browser key-repeat fires only `keydown`,
 *    not `keyup`; we require alternating events.
 * 2. **Rate ceiling** (`MAX_PLAUSIBLE_RATE`). Implausibly fast trials
 *    are flagged and excluded.
 * 3. **Inter-press-interval CV.** Macros fire near-perfectly; humans
 *    have ~5-15% jitter. CV below 0.05 flags the trial.
 */

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

/** CV of inter-press intervals; 0 when fewer than 3 timestamps are available. */
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

/** Apply the three anti-cheating heuristics; returns the trial with `flagged` set. */
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
 * Build one max-rate trial. Each trial creates a fresh closure that owns
 * its own press timestamps and cleanup; on_load and on_finish reference
 * them directly through that closure (no global state needed).
 */
function buildMaxRateTrial(index: number) {
  // The fresh-per-trial state these handlers share via closure.
  const presses: number[] = [];
  let cleanup: () => void = () => {};

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
    trial_duration: CALIBRATION_TRIAL_SECONDS * 1000,
    on_load: () => {
      const counter = document.getElementById("cal-counter");
      const timer = document.getElementById("cal-timer");
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
      cleanup = () => {
        document.removeEventListener("keydown", onKeyDown);
        document.removeEventListener("keyup", onKeyUp);
      };

      const start = performance.now();
      const tick = (): void => {
        const elapsed = (performance.now() - start) / 1000;
        const remaining = Math.max(0, CALIBRATION_TRIAL_SECONDS - Math.floor(elapsed));
        if (timer) timer.textContent = String(remaining);
        if (remaining > 0) setTimeout(tick, 250);
      };
      tick();
    },
    on_finish: (data: Record<string, unknown>) => {
      cleanup();
      const validPresses = presses.length;
      const trial = flagTrial({
        index,
        validPresses,
        durationSec: CALIBRATION_TRIAL_SECONDS,
        rate: validPresses / CALIBRATION_TRIAL_SECONDS,
        ipiCv: ipiCoefficientOfVariation(presses),
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

/** Median rate across non-flagged trials; `ok=false`, `pMax=NaN` if all flagged. */
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

/** Full calibration timeline (intro + maximal trials interleaved with rests). */
export function buildCalibrationTimeline(): unknown[] {
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
    timeline.push(buildMaxRateTrial(i));
    if (i < CALIBRATION_TRIALS - 1) timeline.push(buildRestTrial(i));
  }
  return timeline;
}
