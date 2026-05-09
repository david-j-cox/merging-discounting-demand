/**
 * Main entry point. Orchestrates the full battery:
 *
 *   1. Welcome
 *   2. Informed consent (decline -> polite exit, no data recorded)
 *   3. Calibration
 *   4. Three tasks in counterbalanced order (per randomization)
 *   5. Debrief / data export
 *
 * URL flags (development only):
 *
 *   ?dev=consent|calibration|task1|task2|task3   skip into one stage
 *   ?seed=N                                       override the seed
 */

import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";
import { initJsPsych, type JsPsych } from "jspsych";

import {
  buildDataPipeTrial,
  buildPayload,
  downloadPayloadAsJson,
  readProlificId,
} from "./lib/dataExport";
import { defaultSeed, mulberry32, randomize, type TaskName } from "./lib/randomization";
import {
  buildCalibrationTimeline,
  summariseCalibration,
  type CalibrationResult,
  type CalibrationTrial,
} from "./tasks/calibration";
import { CONSENT_TRIAL, readConsentAgreement } from "./tasks/consent";
import {
  buildEffortDiscountingTimeline,
  type EffortDiscountingResult,
} from "./tasks/effortDiscounting";
import {
  buildEffortPurchaseTaskTimeline,
  type EffortPurchaseTaskResult,
} from "./tasks/effortPurchaseTask";
import { buildPurchaseTaskTimeline, type PurchaseTaskResult } from "./tasks/purchaseTask";
import { STUDY_NAME, STUDY_VERSION } from "./config";

/** jsPsych v8 doesn't export ``TimelineArray``; derive it from ``run``. */
type RunTimeline = Parameters<JsPsych["run"]>[0];

/** Shape every task builder returns. The ``readResult`` reader runs after
 *  the timeline executes and returns the typed per-task result. */
interface TaskTimeline<R> {
  timeline: unknown[];
  readResult: () => R;
}

const DEFAULT_DEV_PMAX = 5.0;
const DEV_STAGES = new Set(["consent", "calibration", "task1", "task2", "task3"]);

interface UrlFlags {
  dev: string | null;
  seed: number | null;
}

function readUrlFlags(): UrlFlags {
  if (typeof window === "undefined") return { dev: null, seed: null };
  const params = new URLSearchParams(window.location.search);
  const dev = params.get("dev");
  const seedParam = params.get("seed");
  const seed = seedParam ? parseInt(seedParam, 10) || null : null;
  return { dev: dev && DEV_STAGES.has(dev) ? dev : null, seed };
}

/** Render a full-page message and clear the rest of the document. */
function showMessage(html: string, opts: { error?: boolean } = {}): void {
  const color = opts.error ? "color: #b00;" : "";
  document.body.innerHTML = `
    <div style="padding: 2rem; max-width: 600px; margin: 4rem auto; font-family: sans-serif; ${color}">
      ${html}
    </div>
  `;
}

const WELCOME_TRIAL = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <h1>${STUDY_NAME} v${STUDY_VERSION}</h1>
    <p>Thank you for participating.</p>
    <p>This study runs in your browser and takes about 20 minutes.</p>
    <p>Please use a desktop or laptop with a physical keyboard. Mobile devices are not supported.</p>
  `,
  choices: ["Begin"],
};

const DEBRIEF_TRIAL = {
  type: jsPsychHtmlButtonResponse,
  stimulus: "<h2>You're done</h2><p>Saving your data...</p>",
  choices: ["Continue"],
  trial_duration: 2000,
};

async function run(): Promise<void> {
  const flags = readUrlFlags();
  const seed = flags.seed ?? defaultSeed();
  const assignment = randomize(seed);
  const prolificId = readProlificId();

  const jsPsych = initJsPsych({
    show_progress_bar: true,
    auto_update_progress_bar: false,
  });

  const calTimeline = buildCalibrationTimeline();

  // After calibration runs, walk jsPsych's data store to collect per-trial
  // CalibrationTrial objects and reduce to the median rate.
  const collectCalibration = (): CalibrationResult => {
    const trials: CalibrationTrial[] = [];
    const allData = jsPsych.data.get().values() as Array<Record<string, unknown>>;
    for (const row of allData) {
      const t = row.calibration_trial as CalibrationTrial | undefined;
      if (t) trials.push(t);
    }
    return summariseCalibration(trials);
  };

  // Per-task builders. Each takes pMax and returns a TaskTimeline whose
  // readResult yields the task-specific typed result object. Task 2 also
  // takes the arm assignment, which drives the immediate-option commodity.
  const buildTask1 = (_pMax: number): TaskTimeline<PurchaseTaskResult> =>
    buildPurchaseTaskTimeline();
  const buildTask2 = (pMax: number): TaskTimeline<EffortDiscountingResult> =>
    buildEffortDiscountingTimeline(pMax, assignment.arm, mulberry32(seed + 1));
  const buildTask3 = (pMax: number): TaskTimeline<EffortPurchaseTaskResult> =>
    buildEffortPurchaseTaskTimeline(pMax);

  // ---- Dev shortcut: skip the welcome + battery and just run one stage. ----
  if (flags.dev) {
    const pMax = DEFAULT_DEV_PMAX;
    const stages: Record<string, () => unknown[]> = {
      consent: () => [CONSENT_TRIAL],
      calibration: () => calTimeline,
      task1: () => buildTask1(pMax).timeline,
      task2: () => buildTask2(pMax).timeline,
      task3: () => buildTask3(pMax).timeline,
    };
    await jsPsych.run((stages[flags.dev]?.() ?? []) as RunTimeline);
    return;
  }

  // ---- Welcome -> Consent ----
  await jsPsych.run([WELCOME_TRIAL, CONSENT_TRIAL] as RunTimeline);
  const allData = jsPsych.data.get().values() as Array<Record<string, unknown>>;
  if (!readConsentAgreement(allData)) {
    showMessage(
      `<h2>Thank you</h2>
       <p>You have declined to participate. No data has been recorded.
          You may close this window.</p>`,
    );
    return;
  }

  // ---- Calibration ----
  await jsPsych.run(calTimeline as RunTimeline);
  const calResult = collectCalibration();
  if (!calResult.ok) {
    showMessage(
      `<h2>Calibration could not be completed</h2>
       <p>Every calibration trial was flagged. This usually means the SPACE
          bar was held down or pressed by software. Please refresh and try
          again with a manual finger press.</p>`,
    );
    return;
  }

  // ---- Three tasks in randomized order ----
  const t1 = buildTask1(calResult.pMax);
  const t2 = buildTask2(calResult.pMax);
  const t3 = buildTask3(calResult.pMax);
  const builders: Record<TaskName, TaskTimeline<unknown>> = {
    task1_purchase: t1,
    task2_effort_discount: t2,
    task3_effort_purchase: t3,
  };

  const battery: unknown[] = [];
  for (const taskName of assignment.taskOrder) {
    battery.push(...builders[taskName].timeline);
  }
  await jsPsych.run([...battery, DEBRIEF_TRIAL] as RunTimeline);

  // ---- Build payload and export ----
  const payload = buildPayload({
    randomization: assignment,
    calibration: calResult,
    task1: t1.readResult(),
    task2: t2.readResult(),
    task3: t3.readResult(),
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
    // Placeholder DataPipe ID — always offer the local download.
    downloadPayloadAsJson(payload);
  }

  showMessage(
    "<h2>Thank you for participating</h2><p>Your data has been recorded. You may close this window.</p>",
  );
}

run().catch((err: unknown) => {
  console.error(err);
  showMessage(
    `<h2>Something went wrong</h2>
     <p>Please refresh the page to start again.</p>
     <pre>${String(err)}</pre>`,
    { error: true },
  );
});
