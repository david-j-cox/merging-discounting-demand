/**
 * Task 3: Effort purchase task — novel; the empirical hinge of the
 * unification (CLAUDE.md §8).
 *
 * Parallel to Task 1 with effort substituted for money: each "price"
 * is `effortPrice` seconds of sustained key-pressing at `pMax`, and
 * the participant reports how many max-reward units they would
 * acquire. Under the unified model both `Q(P_money)` and
 * `Q(P_effort)` follow Koffarnus demand with a *shared* alpha — that
 * shared-alpha hypothesis is the H1 test.
 *
 * Bounds: a soft cap from a `HYPOTHETICAL_BUDGET_MINUTES` time
 * budget triggers a warning if exceeded but does not reject the
 * answer.
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

/** Soft feasibility cap: max units fitting in `HYPOTHETICAL_BUDGET_MINUTES` minutes. */
function feasibilityCap(effortPrice: number): number {
  return Math.max(1, Math.floor((HYPOTHETICAL_BUDGET_MINUTES * 60) / effortPrice));
}

/** Full effort-purchase-task timeline plus a `readResult` reader. */
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
