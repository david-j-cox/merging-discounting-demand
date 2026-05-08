/**
 * Task 1: standard hypothetical purchase task. For each price in
 * `PURCHASE_PRICES_USD`, ask how many of the (generic snack)
 * commodity the participant would buy. Demand-curve fitting happens
 * post-hoc in Python.
 */

import jsPsychInstructions from "@jspsych/plugin-instructions";
import jsPsychSurveyHtmlForm from "@jspsych/plugin-survey-html-form";

import { PURCHASE_PRICES_USD } from "../config";

export interface PurchaseTaskTrial {
  /** Price in USD. */
  price: number;
  /** Quantity reported by participant. */
  quantity: number;
}

export interface PurchaseTaskResult {
  /** All trials in price order. */
  trials: PurchaseTaskTrial[];
  /** Free-form description of the commodity used. */
  commodity: string;
}

const COMMODITY_LABEL = "snack item";
const COMMODITY_DESCRIPTION =
  "a single small packaged snack (e.g. a candy bar, a small bag of chips, or similar).";

/** Full purchase-task timeline plus a `readResult` reader. */
export function buildPurchaseTaskTimeline(): {
  timeline: unknown[];
  readResult: () => PurchaseTaskResult;
} {
  const trialRecords: PurchaseTaskTrial[] = [];

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

  // One trial per price, in ascending price order.
  const trials = PURCHASE_PRICES_USD.map((price, idx) => ({
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <h2>Price ${idx + 1} of ${PURCHASE_PRICES_USD.length}</h2>
      <p>Each ${COMMODITY_LABEL} costs <strong>$${price.toFixed(2)}</strong>.</p>
      <p>How many would you buy at this price?</p>
    `,
    html: `
      <p>
        <label for="qty-${idx}">Quantity:</label>
        <input id="qty-${idx}" name="quantity" type="number" min="0" step="1" required
               style="width: 6em; font-size: 1.1em;" />
      </p>
    `,
    on_finish: (data: Record<string, unknown>) => {
      const response = data.response as { quantity?: string };
      const raw = response?.quantity ?? "0";
      const qty = Math.max(0, parseInt(raw, 10) || 0);
      const trial: PurchaseTaskTrial = { price, quantity: qty };
      trialRecords.push(trial);
      data.purchase_trial = trial;
    },
  }));

  const timeline = [intro, ...trials];

  const readResult = (): PurchaseTaskResult => ({
    trials: [...trialRecords],
    commodity: COMMODITY_LABEL,
  });

  return { timeline, readResult };
}
