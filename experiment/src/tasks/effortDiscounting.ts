/**
 * Effort discounting task (Task 2).
 *
 * One adjusting-amount titration per effort fraction in
 * `EFFORT_FRACTIONS`. Effort level = `f * pMax` presses/sec sustained
 * for `SUSTAINED_TRIAL_SECONDS`. Mostly hypothetical; a sparse
 * `VERIFICATION_TRIAL_FRACTION` of trials require the real key-press
 * task as a compliance check.
 */

import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";
import jsPsychHtmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";
import jsPsychInstructions from "@jspsych/plugin-instructions";

import {
  EFFORT_FRACTIONS,
  REWARD_MAX_USD,
  SUSTAINED_TRIAL_SECONDS,
  RATE_TOLERANCE,
  VERIFICATION_TRIAL_FRACTION,
} from "../config";
import { Titrator, type TitrationResult } from "../lib/titration";

export interface EffortDiscountingTrialChoice {
  /** Effort fraction (10-85% of pMax). */
  fraction: number;
  /** Absolute target rate in presses/sec for the effortful option. */
  targetRate: number;
  /** Step within the titration (0-indexed). */
  step: number;
  /** Immediate-reward offer in USD. */
  offer: number;
  /** Subject's choice. */
  choice: "immediate" | "effortful";
  /** Reaction time in milliseconds. */
  rtMs: number;
}

export interface EffortDiscountingResult {
  /** One titration per effort fraction. */
  perFraction: { fraction: number; titration: TitrationResult }[];
  /** All choices, in trial order, for the data export. */
  allChoices: EffortDiscountingTrialChoice[];
  /** Subject's pMax used to scale effort levels. */
  pMaxUsed: number;
}

/** Pick `~nTotal*fraction` trial indices for verification, skipping index 0. */
function chooseVerificationIndices(
  nTotal: number,
  fraction: number,
  rng: () => number,
): Set<number> {
  const target = Math.max(1, Math.round(nTotal * fraction));
  const indices = new Set<number>();
  while (indices.size < target) {
    const idx = Math.floor(rng() * nTotal);
    if (idx === 0) continue; // skip the very first trial
    indices.add(idx);
  }
  return indices;
}

/** Verification trial: subject performs SUSTAINED_TRIAL_SECONDS at the target rate. */
function buildVerificationTrial(targetRate: number, fraction: number) {
  // Per-trial closure state (no global stash needed).
  const timestamps: number[] = [];
  let cleanup: () => void = () => {};

  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div style="text-align: center;">
        <h2>Verification trial</h2>
        <p>Press SPACE at about <strong>${targetRate.toFixed(1)} presses/sec</strong>
           for ${SUSTAINED_TRIAL_SECONDS} seconds.</p>
        <p id="ver-counter" style="font-size: 2.5rem; font-weight: bold;">0</p>
        <p id="ver-rate" style="font-size: 1.2rem;">0.0 presses/sec</p>
        <p id="ver-timer" class="countdown">${SUSTAINED_TRIAL_SECONDS}</p>
      </div>
    `,
    choices: "NO_KEYS" as const,
    trial_duration: SUSTAINED_TRIAL_SECONDS * 1000,
    on_load: () => {
      const counter = document.getElementById("ver-counter");
      const rateLabel = document.getElementById("ver-rate");
      const timer = document.getElementById("ver-timer");
      let awaitingKeyup = false;
      const start = performance.now();

      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code !== "Space") return;
        e.preventDefault();
        if (awaitingKeyup) return;
        awaitingKeyup = true;
        timestamps.push(performance.now());
        if (counter) counter.textContent = String(timestamps.length);
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

      const tick = (): void => {
        const elapsed = (performance.now() - start) / 1000;
        const remaining = Math.max(0, SUSTAINED_TRIAL_SECONDS - Math.floor(elapsed));
        const currentRate = elapsed > 0 ? timestamps.length / elapsed : 0;
        if (timer) timer.textContent = String(remaining);
        if (rateLabel) rateLabel.textContent = `${currentRate.toFixed(2)} presses/sec`;
        if (remaining > 0) setTimeout(tick, 250);
      };
      tick();
    },
    on_finish: (data: Record<string, unknown>) => {
      cleanup();
      const observedRate = timestamps.length / SUSTAINED_TRIAL_SECONDS;
      const tolerance = RATE_TOLERANCE * targetRate;
      data.verification = {
        fraction,
        targetRate,
        observedRate,
        passed: Math.abs(observedRate - targetRate) <= tolerance,
        tolerance,
        nPresses: timestamps.length,
      };
    },
  };
}

/**
 * Full effort-discounting timeline.
 *
 * Sequential per-fraction titrations: each step's stimulus is a function
 * of the prior choices (jsPsych procedural form). The Titrator instances
 * persist via the closure over `titrators`. Verification trials are
 * spliced in for the fractions selected by `chooseVerificationIndices`.
 */
export function buildEffortDiscountingTimeline(
  pMax: number,
  rng: () => number = Math.random,
): { timeline: unknown[]; readResult: () => EffortDiscountingResult } {
  const totalFractions = EFFORT_FRACTIONS.length;
  const verifyFractions = chooseVerificationIndices(
    totalFractions,
    VERIFICATION_TRIAL_FRACTION,
    rng,
  );

  const titrators = new Map<number, Titrator>();
  const allChoices: EffortDiscountingTrialChoice[] = [];

  const intro = {
    type: jsPsychInstructions,
    pages: [
      `<h2>Effort discounting</h2>
       <p>You'll choose between two options on each trial:</p>
       <ul>
         <li>Take a smaller amount of money <strong>now</strong>, with no effort.</li>
         <li>Earn $${REWARD_MAX_USD.toFixed(2)} by performing a key-press task at a target rate.</li>
       </ul>
       <p>Most trials are hypothetical &mdash; please choose as if the money and effort were real.
          A small number of trials will ask you to actually perform the task.</p>`,
      `<p>Your earlier calibration set your maximum rate at about
         <strong>${pMax.toFixed(1)} presses/sec</strong>.
         Effort levels in this task are scaled to that maximum.</p>
       <p>Click <strong>Next</strong> to begin.</p>`,
    ],
    show_clickable_nav: true,
  };

  const timeline: unknown[] = [intro];

  for (let fIdx = 0; fIdx < totalFractions; fIdx++) {
    const fraction = EFFORT_FRACTIONS[fIdx] as number;
    const targetRate = fraction * pMax;
    const titrator = new Titrator(REWARD_MAX_USD);
    titrators.set(fraction, titrator);

    for (let step = 0; step < titrator.maxSteps; step++) {
      timeline.push({
        timeline: [
          {
            type: jsPsychHtmlButtonResponse,
            stimulus: () => {
              return `
                <div style="text-align: center;">
                  <h2>Choose</h2>
                  <p style="font-size: 1.1em;">Effort level:
                     <strong>${(fraction * 100).toFixed(0)}%</strong>
                     of your maximum rate
                     (~${targetRate.toFixed(1)} presses/sec
                     for ${SUSTAINED_TRIAL_SECONDS} seconds).</p>
                  <p>Step ${step + 1} of ${titrator.maxSteps}.</p>
                </div>
              `;
            },
            choices: () => {
              const offer = titrator.nextOffer();
              return [
                `Take $${offer.toFixed(2)} now`,
                `Do the effort task for $${REWARD_MAX_USD.toFixed(2)}`,
              ];
            },
            on_finish: (data: Record<string, unknown>) => {
              const buttonIndex = data.response as number;
              const choice: "immediate" | "effortful" =
                buttonIndex === 0 ? "immediate" : "effortful";
              const rt = (data.rt as number | null) ?? 0;
              // `record()` reads the next offer internally and writes the
              // step into the titrator history, so we capture the recorded
              // step for our own log.
              const recordedStep = titrator.record(choice);
              const choiceRecord: EffortDiscountingTrialChoice = {
                fraction,
                targetRate,
                step,
                offer: recordedStep.offer,
                choice,
                rtMs: rt,
              };
              allChoices.push(choiceRecord);
              data.titration_choice = choiceRecord;
            },
          },
        ],
      });
    }

    if (verifyFractions.has(fIdx)) {
      timeline.push(buildVerificationTrial(targetRate, fraction));
    }
  }

  const readResult = (): EffortDiscountingResult => ({
    perFraction: EFFORT_FRACTIONS.map((fraction) => {
      const t = titrators.get(fraction);
      if (!t || !t.isDone()) {
        throw new Error(`Titration for fraction ${fraction} is not complete.`);
      }
      return { fraction, titration: t.result() };
    }),
    allChoices,
    pMaxUsed: pMax,
  });

  return { timeline, readResult };
}
