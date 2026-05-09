/**
 * Task 1: standard hypothetical purchase task. For each price in
 * `PURCHASE_PRICES_USD`, ask how many *snack credits* the participant
 * would buy. Snack credits are the unified commodity throughout this
 * study (defined in the consent text); see Task 2 / Task 3 for the
 * other side of the H1 linkage. Demand-curve fitting happens post-hoc
 * in Python.
 */

import jsPsychInstructions from "@jspsych/plugin-instructions";
import jsPsychSurveyHtmlForm from "@jspsych/plugin-survey-html-form";

import {
  CATCH_TRIAL_EXPECTED_QUANTITY,
  CATCH_TRIAL_POSITION,
  PURCHASE_PRICES_USD,
} from "../config";

export interface PurchaseTaskTrial {
  /** Price in USD. */
  price: number;
  /** Quantity reported by participant. */
  quantity: number;
  /** True if this is the catch trial (a normal-looking trial that explicitly
   *  asks for a specific quantity). The Python loader skips these. */
  isCatch?: boolean;
  /** True if isCatch and the participant entered the wrong quantity. */
  failedCatch?: boolean;
}

export interface PurchaseTaskResult {
  /** All trials in price order. */
  trials: PurchaseTaskTrial[];
  /** Free-form description of the commodity used. */
  commodity: string;
}

const COMMODITY_LABEL = "snack credit";
const COMMODITY_DESCRIPTION =
  "redeemable for one small packaged snack of your choice (e.g. a candy " +
  "bar, a small bag of chips, or similar).";

/**
 * Build a normal purchase-task trial (one price, one "how many would
 * you buy" question). Pushes a PurchaseTaskTrial into ``records``.
 */
function buildPriceTrial(
  price: number,
  idxLabel: string,
  records: PurchaseTaskTrial[],
) {
  return {
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <h2>Price ${idxLabel}</h2>
      <p>Each ${COMMODITY_LABEL} costs <strong>$${price.toFixed(2)}</strong>.</p>
      <p>How many would you buy at this price?</p>
    `,
    html: `
      <p>
        <label for="qty">Quantity:</label>
        <input id="qty" name="quantity" type="number" min="0" step="1" required
               style="width: 6em; font-size: 1.1em;" />
      </p>
    `,
    on_finish: (data: Record<string, unknown>) => {
      const response = data.response as { quantity?: string };
      const raw = response?.quantity ?? "0";
      const qty = Math.max(0, parseInt(raw, 10) || 0);
      const trial: PurchaseTaskTrial = { price, quantity: qty };
      records.push(trial);
      data.purchase_trial = trial;
    },
  };
}

/**
 * Build the attention-check trial. Looks like a normal purchase trial
 * but asks the participant to enter a specific number. Sets
 * ``isCatch`` and ``failedCatch`` on the recorded trial so quality
 * checks can detect non-attentive subjects.
 */
function buildCatchTrial(idxLabel: string, records: PurchaseTaskTrial[]) {
  return {
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <h2>Attention check</h2>
      <p>To show you're reading these prompts carefully, please enter the
         number <strong>${CATCH_TRIAL_EXPECTED_QUANTITY}</strong> in the box below.</p>
      <p>(This is question ${idxLabel} and is used as an attention check.)</p>
    `,
    html: `
      <p>
        <label for="qty">Quantity:</label>
        <input id="qty" name="quantity" type="number" min="0" step="1" required
               style="width: 6em; font-size: 1.1em;" />
      </p>
    `,
    on_finish: (data: Record<string, unknown>) => {
      const response = data.response as { quantity?: string };
      const raw = response?.quantity ?? "0";
      const qty = Math.max(0, parseInt(raw, 10) || 0);
      const trial: PurchaseTaskTrial = {
        price: Number.NaN, // catch trial has no meaningful price
        quantity: qty,
        isCatch: true,
        failedCatch: qty !== CATCH_TRIAL_EXPECTED_QUANTITY,
      };
      records.push(trial);
      data.purchase_trial = trial;
    },
  };
}

/** Full purchase-task timeline plus a `readResult` reader. */
export function buildPurchaseTaskTimeline(): {
  timeline: unknown[];
  readResult: () => PurchaseTaskResult;
} {
  const trialRecords: PurchaseTaskTrial[] = [];

  // Total trial count includes the catch trial (one extra "question N of M").
  const totalTrials = PURCHASE_PRICES_USD.length + 1;

  const intro = {
    type: jsPsychInstructions,
    pages: [
      `<h2>Purchase task</h2>
       <p>Imagine you have unlimited income for this task.</p>
       <p>For a series of prices, you'll be asked how many <strong>${COMMODITY_LABEL}s</strong>
          you would purchase at each price.</p>
       <p>For this task, a <em>${COMMODITY_LABEL}</em> means ${COMMODITY_DESCRIPTION}</p>`,
      `<p>Please answer as if these were real prices and you were really buying.</p>
       <p>Click <strong>Next</strong> to begin.</p>`,
    ],
    show_clickable_nav: true,
  };

  // Build the trial sequence with the catch trial spliced in at the
  // configured position.
  const trials: unknown[] = [];
  let priceIdx = 0;
  for (let trialIdx = 0; trialIdx < totalTrials; trialIdx++) {
    const label = `${trialIdx + 1} of ${totalTrials}`;
    if (trialIdx === CATCH_TRIAL_POSITION) {
      trials.push(buildCatchTrial(label, trialRecords));
    } else {
      const price = PURCHASE_PRICES_USD[priceIdx] as number;
      trials.push(buildPriceTrial(price, label, trialRecords));
      priceIdx++;
    }
  }

  const timeline = [intro, ...trials];

  const readResult = (): PurchaseTaskResult => ({
    trials: [...trialRecords],
    commodity: COMMODITY_LABEL,
  });

  return { timeline, readResult };
}
