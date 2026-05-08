/**
 * Main entry point. Orchestrates the full battery:
 *
 *   1. Welcome / consent placeholder
 *   2. Calibration
 *   3. Three tasks in counterbalanced order (per randomization)
 *   4. Debrief / data export
 *
 * URL flags supported (development only — hidden from production
 * participants):
 *
 *   ?dev=calibration   skip into calibration only
 *   ?dev=task1         skip into Task 1 only (uses default pMax = 5)
 *   ?dev=task2         skip into Task 2 only (uses default pMax = 5)
 *   ?dev=task3         skip into Task 3 only (uses default pMax = 5)
 *   ?seed=N            override the randomization seed
 *
 * In production, the orchestrator runs the full battery.
 */

import { initJsPsych, type JsPsych } from "jspsych";
import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";

/** Run-time type alias: jsPsych v8 doesn't export TimelineArray, so we
 *  derive the parameter type from the `run` method itself. */
type RunTimeline = Parameters<JsPsych["run"]>[0];

import { STUDY_NAME, STUDY_VERSION } from "./config";
import { buildCalibrationTimeline, summariseCalibration, type CalibrationResult, type CalibrationTrial } from "./tasks/calibration";
import { buildEffortDiscountingTimeline, type EffortDiscountingResult } from "./tasks/effortDiscounting";
import { buildEffortPurchaseTaskTimeline, type EffortPurchaseTaskResult } from "./tasks/effortPurchaseTask";
import { buildPurchaseTaskTimeline, type PurchaseTaskResult } from "./tasks/purchaseTask";
import {
  buildDataPipeTrial,
  buildPayload,
  downloadPayloadAsJson,
  readProlificId,
} from "./lib/dataExport";
import { defaultSeed, mulberry32, randomize, type TaskName } from "./lib/randomization";

interface UrlFlags {
  dev: string | null;
  seed: number | null;
}

function readUrlFlags(): UrlFlags {
  if (typeof window === "undefined") {
    return { dev: null, seed: null };
  }
  const params = new URLSearchParams(window.location.search);
  const dev = params.get("dev");
  const seedParam = params.get("seed");
  const seed = seedParam ? parseInt(seedParam, 10) || null : null;
  return { dev, seed };
}

const DEFAULT_DEV_PMAX = 5.0;

async function run(): Promise<void> {
  const flags = readUrlFlags();
  const seed = flags.seed ?? defaultSeed();
  const assignment = randomize(seed);
  const prolificId = readProlificId();

  const jsPsych = initJsPsych({
    show_progress_bar: true,
    auto_update_progress_bar: false,
  });

  const welcome = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <h1>${STUDY_NAME} v${STUDY_VERSION}</h1>
      <p>Thank you for participating.</p>
      <p>This study runs in your browser and takes about 20 minutes.</p>
      <p>Please use a desktop or laptop with a physical keyboard. Mobile devices are not supported.</p>
    `,
    choices: ["Begin"],
  };

  // ---- Calibration ----
  const calTimeline = buildCalibrationTimeline(jsPsych);

  // The per-trial CalibrationTrial objects are written to data by
  // calibration.ts. We collect them after the timeline completes.
  const collectCalibration = (): CalibrationResult => {
    const trials: CalibrationTrial[] = [];
    const allData = jsPsych.data.get().values() as Array<Record<string, unknown>>;
    for (const row of allData) {
      const t = row.calibration_trial as CalibrationTrial | undefined;
      if (t) trials.push(t);
    }
    return summariseCalibration(trials);
  };

  // ---- Build the three tasks against the calibration result ----
  // The closures capture pMax which is set after calibration runs. We
  // therefore build task timelines lazily inside a call_function trial.

  const taskBuilders: Record<
    TaskName,
    (pMax: number) => { timeline: unknown[]; readResult: () => unknown }
  > = {
    task1_purchase: () => buildPurchaseTaskTimeline() as {
      timeline: unknown[];
      readResult: () => unknown;
    },
    task2_effort_discount: (pMax: number) => {
      const rng = mulberry32(seed + 1);
      return buildEffortDiscountingTimeline(jsPsych, pMax, rng) as {
        timeline: unknown[];
        readResult: () => unknown;
      };
    },
    task3_effort_purchase: (pMax: number) => buildEffortPurchaseTaskTimeline(pMax) as {
      timeline: unknown[];
      readResult: () => unknown;
    },
  };

  const taskResults: Partial<{
    task1_purchase: PurchaseTaskResult;
    task2_effort_discount: EffortDiscountingResult;
    task3_effort_purchase: EffortPurchaseTaskResult;
  }> = {};

  // ---- Dev shortcut ----
  if (flags.dev) {
    const devPMax = DEFAULT_DEV_PMAX;
    let devTimeline: unknown[] = [];
    if (flags.dev === "calibration") {
      devTimeline = calTimeline;
    } else if (flags.dev === "task1") {
      const built = taskBuilders.task1_purchase(devPMax);
      devTimeline = built.timeline;
    } else if (flags.dev === "task2") {
      const built = taskBuilders.task2_effort_discount(devPMax);
      devTimeline = built.timeline;
    } else if (flags.dev === "task3") {
      const built = taskBuilders.task3_effort_purchase(devPMax);
      devTimeline = built.timeline;
    }
    await jsPsych.run(devTimeline as RunTimeline);
    return;
  }

  // ---- Full battery ----
  const debrief = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <h2>You're done</h2>
      <p>Saving your data...</p>
    `,
    choices: ["Continue"],
    trial_duration: 2000,
  };

  // We assemble the full timeline up to the post-calibration split, then
  // append per-task timelines after pMax is known. The cleanest pattern in
  // jsPsych 8 is to pass a single concatenated array; we run the
  // calibration on its own first, then the rest.
  await jsPsych.run([welcome, ...calTimeline] as RunTimeline);
  const calResult = collectCalibration();
  if (!calResult.ok) {
    document.body.innerHTML = `
      <div style="padding: 2rem; max-width: 600px; margin: 4rem auto; font-family: sans-serif;">
        <h2>Calibration could not be completed</h2>
        <p>Every calibration trial was flagged. This usually means the SPACE bar
           was held down or pressed by software. Please refresh and try again
           with a manual finger press.</p>
      </div>
    `;
    return;
  }

  const restOfBattery: unknown[] = [];
  for (const taskName of assignment.taskOrder) {
    const built = taskBuilders[taskName](calResult.pMax);
    restOfBattery.push(...built.timeline);
    restOfBattery.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: "<p>Recording your responses...</p>",
      choices: ["Continue"],
      trial_duration: 500,
      on_finish: () => {
        const r = built.readResult();
        if (taskName === "task1_purchase") {
          taskResults.task1_purchase = r as PurchaseTaskResult;
        } else if (taskName === "task2_effort_discount") {
          taskResults.task2_effort_discount = r as EffortDiscountingResult;
        } else {
          taskResults.task3_effort_purchase = r as EffortPurchaseTaskResult;
        }
      },
    });
  }

  await jsPsych.run([...restOfBattery, debrief] as RunTimeline);

  // ---- Build payload and export ----
  if (!taskResults.task1_purchase || !taskResults.task2_effort_discount || !taskResults.task3_effort_purchase) {
    console.error("[main] One or more task results missing; offering local download only.");
  }

  const payload = buildPayload({
    randomization: assignment,
    calibration: calResult,
    task1: taskResults.task1_purchase ?? { trials: [], commodity: "" },
    task2: taskResults.task2_effort_discount ?? {
      perFraction: [],
      allChoices: [],
      pMaxUsed: calResult.pMax,
    },
    task3: taskResults.task3_effort_purchase ?? {
      trials: [],
      pMaxUsed: calResult.pMax,
      feasibilityCapsByPrice: [],
    },
    rawTrials: jsPsych.data.get().values(),
    prolificId,
  });

  const dataPipeTrial = buildDataPipeTrial(payload);
  if (dataPipeTrial) {
    try {
      await jsPsych.run([dataPipeTrial] as RunTimeline);
    } catch (err) {
      console.error("[main] DataPipe POST failed; falling back to local download.", err);
      downloadPayloadAsJson(payload);
    }
  } else {
    // Placeholder DataPipe ID -> always offer the local download.
    downloadPayloadAsJson(payload);
  }

  document.body.innerHTML = `
    <div style="padding: 2rem; max-width: 600px; margin: 4rem auto; font-family: sans-serif;">
      <h2>Thank you for participating</h2>
      <p>Your data has been recorded. You may close this window.</p>
    </div>
  `;
}

run().catch((err: unknown) => {
  console.error(err);
  document.body.innerHTML = `
    <div style="padding: 2rem; max-width: 600px; margin: 4rem auto; font-family: sans-serif; color: #b00;">
      <h2>Something went wrong</h2>
      <p>Please refresh the page to start again.</p>
      <pre>${String(err)}</pre>
    </div>
  `;
});
