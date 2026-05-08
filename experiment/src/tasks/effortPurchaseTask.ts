/**
 * Task 3: Effort purchase task (NOVEL — the empirical hinge of the
 * unification per CLAUDE.md §8).
 *
 * Design parallel: the standard purchase task asks "how many would you buy
 * at price $X?" measuring consumption Q as a function of monetary price P.
 * The effort purchase task asks "how many would you do at effort cost X?"
 * measuring consumption Q as a function of *effort* price.
 *
 * Operationally: each "price" is an effort cost expressed in seconds of
 * sustained key-pressing at the subject's calibration rate `pMax`. A price
 * of 0.5 means "0.5 seconds of effort per unit reward acquired"; the
 * subject is asked how many of the maximum-reward unit they would acquire
 * at that effort cost. The dependent measure is `Q(P_effort)`, exactly
 * analogous to `Q(P_money)` in Task 1.
 *
 * Why this works for the unification: under the model in
 * `docs/derivation.md`, both Q-vs-P_money and Q-vs-P_effort follow a
 * Koffarnus exponential demand curve with a *shared α*. The H1 test is
 * whether the joint two-task fit prefers a shared α over independent αs.
 *
 * Bounds checking: at very high effort prices, the participant might
 * report an absurdly large number of units. We clamp the displayed
 * acceptable range against `pMax` and the participant's own time budget
 * (10 minutes total assumed), and warn the participant if they exceed
 * that, but accept their answer either way for analysis.
 */

import jsPsychInstructions from "@jspsych/plugin-instructions";
import jsPsychSurveyHtmlForm from "@jspsych/plugin-survey-html-form";

import { EFFORT_PRICES, REWARD_MAX_USD } from "../config";

export interface EffortPurchaseTrial {
  /** Effort price (effort-seconds per acquired reward unit). */
  effortPrice: number;
  /** Quantity reported by participant. */
  quantity: number;
  /** True if reported quantity exceeded the soft physical-feasibility cap. */
  exceededFeasibility: boolean;
}

export interface EffortPurchaseTaskResult {
  /** All trials in price order. */
  trials: EffortPurchaseTrial[];
  /** Subject's pMax used to anchor the effort scale. */
  pMaxUsed: number;
  /** Soft feasibility cap (in units) used per trial; mostly for analysis. */
  feasibilityCapsByPrice: number[];
}

/** Total session minutes assumed available for hypothetical acquisitions. */
const HYPOTHETICAL_BUDGET_MINUTES = 10;

/**
 * Soft feasibility cap on hypothetical units at a given effort price.
 *
 * "Hypothetical budget" of `B_min` minutes at the subject's rate `pMax`
 * gives a maximum number of units `(B_min * 60) / effortPrice`. We use
 * this only as a warning threshold; the participant's answer is recorded
 * regardless.
 */
function feasibilityCap(effortPrice: number): number {
  return Math.max(1, Math.floor((HYPOTHETICAL_BUDGET_MINUTES * 60) / effortPrice));
}

/**
 * Build the full effort-purchase-task timeline.
 */
export function buildEffortPurchaseTaskTimeline(pMax: number): {
  timeline: unknown[];
  readResult: () => EffortPurchaseTaskResult;
} {
  const trialRecords: EffortPurchaseTrial[] = [];

  const intro = {
    type: jsPsychInstructions,
    pages: [
      `<h2>Effort purchase task</h2>
       <p>Now imagine you have unlimited <strong>time</strong> for this task,
          but each acquired reward costs <em>effort</em> instead of money.</p>
       <p>You can earn $${REWARD_MAX_USD.toFixed(2)} per acquisition. Each acquisition
          costs you a fixed amount of sustained key-pressing at your maximum
          rate (about ${pMax.toFixed(1)} presses/sec).</p>
       <p>For each effort price below, please report how many acquisitions you
          would make.</p>`,
      `<p>Imagine you have about ${HYPOTHETICAL_BUDGET_MINUTES} minutes available for the task.
          That sets a soft cap on how many units are physically achievable; the system
          will warn you if you exceed it but will record whatever you enter.</p>
       <p>Click <strong>Next</strong> to begin.</p>`,
    ],
    show_clickable_nav: true,
  };

  const trials = EFFORT_PRICES.map((effortPrice, idx) => {
    const cap = feasibilityCap(effortPrice);
    return {
      type: jsPsychSurveyHtmlForm,
      preamble: `
        <h2>Effort price ${idx + 1} of ${EFFORT_PRICES.length}</h2>
        <p>Each acquisition costs <strong>${effortPrice.toFixed(2)} seconds</strong>
           of sustained key-pressing at your maximum rate.</p>
        <p>You earn <strong>$${REWARD_MAX_USD.toFixed(2)}</strong> per acquisition.</p>
        <p>Soft feasibility cap (${HYPOTHETICAL_BUDGET_MINUTES} minutes available):
           about ${cap} acquisitions.</p>
        <p>How many acquisitions would you make at this effort price?</p>
      `,
      html: `
        <p>
          <label for="ept-${idx}">Quantity:</label>
          <input id="ept-${idx}" name="quantity" type="number" min="0" step="1" required
                 style="width: 6em; font-size: 1.1em;" />
        </p>
      `,
      on_finish: (data: Record<string, unknown>) => {
        const response = data.response as { quantity?: string };
        const raw = response?.quantity ?? "0";
        const qty = Math.max(0, parseInt(raw, 10) || 0);
        const trial: EffortPurchaseTrial = {
          effortPrice,
          quantity: qty,
          exceededFeasibility: qty > cap,
        };
        trialRecords.push(trial);
        data.effort_purchase_trial = trial;
      },
    };
  });

  const timeline = [intro, ...trials];

  const readResult = (): EffortPurchaseTaskResult => ({
    trials: [...trialRecords],
    pMaxUsed: pMax,
    feasibilityCapsByPrice: EFFORT_PRICES.map(feasibilityCap),
  });

  return { timeline, readResult };
}

export { feasibilityCap };
