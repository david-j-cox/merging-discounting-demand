/**
 * Data export pipeline.
 *
 * Two channels:
 *
 *  1. **DataPipe** (https://pipe.jspsych.org). Posts the session JSON to a
 *     DataPipe experiment which forwards it to OSF. Configuration constants
 *     live in `config.ts`. The plugin handles network detail; we wrap it
 *     so that failures don't drop data.
 *  2. **Local download** as a JSON file. Always offered as a fallback if
 *     the DataPipe POST fails or the participant is running locally
 *     (e.g. during piloting). This guarantees no data loss even if the
 *     OSF integration is misconfigured.
 *
 * The export wraps the per-task results (calibration, Task 1, Task 2,
 * Task 3) plus randomization into a single JSON payload tagged with the
 * study version and a random session UUID.
 */

import jsPsychPipe from "@jspsych-contrib/plugin-pipe";

import {
  DATAPIPE_EXPERIMENT_ID,
  DATAPIPE_TEST_MODE,
  STUDY_NAME,
  STUDY_VERSION,
} from "../config";
import type { CalibrationResult } from "../tasks/calibration";
import type { EffortDiscountingResult } from "../tasks/effortDiscounting";
import type { EffortPurchaseTaskResult } from "../tasks/effortPurchaseTask";
import type { PurchaseTaskResult } from "../tasks/purchaseTask";
import type { RandomizationAssignment } from "./randomization";

export interface SessionPayload {
  study: string;
  version: string;
  sessionId: string;
  timestampIso: string;
  testMode: boolean;
  prolificId: string | null;
  randomization: RandomizationAssignment;
  calibration: CalibrationResult;
  task1: PurchaseTaskResult;
  task2: EffortDiscountingResult;
  task3: EffortPurchaseTaskResult;
  /** Free-form raw jsPsych trial data, retained for QA. */
  rawTrials: unknown;
}

/**
 * RFC4122-ish v4 UUID via crypto.randomUUID where available, with a
 * fallback that uses Math.random for very old browsers (negligible in
 * practice but cheap to support).
 */
export function newSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback: not cryptographically random, but unique enough for an ID.
  const hex = (n: number) => Math.floor(Math.random() * 16 ** n)
    .toString(16)
    .padStart(n, "0");
  return `${hex(8)}-${hex(4)}-${hex(4)}-${hex(4)}-${hex(12)}`;
}

/**
 * Build the payload object from per-task results.
 */
export function buildPayload(args: {
  randomization: RandomizationAssignment;
  calibration: CalibrationResult;
  task1: PurchaseTaskResult;
  task2: EffortDiscountingResult;
  task3: EffortPurchaseTaskResult;
  rawTrials: unknown;
  prolificId?: string | null;
}): SessionPayload {
  return {
    study: STUDY_NAME,
    version: STUDY_VERSION,
    sessionId: newSessionId(),
    timestampIso: new Date().toISOString(),
    testMode: DATAPIPE_TEST_MODE,
    prolificId: args.prolificId ?? null,
    randomization: args.randomization,
    calibration: args.calibration,
    task1: args.task1,
    task2: args.task2,
    task3: args.task3,
    rawTrials: args.rawTrials,
  };
}

/**
 * Build the jsPsych timeline trial that POSTs the payload to DataPipe.
 *
 * Returns a trial object compatible with jsPsych 8. If
 * `DATAPIPE_EXPERIMENT_ID` is the placeholder, the trial is skipped
 * (development mode).
 */
export function buildDataPipeTrial(payload: SessionPayload): unknown {
  if (DATAPIPE_EXPERIMENT_ID === "PLACEHOLDER_BEFORE_PILOT") {
    // No-op trial: print a warning to console so devs notice.
    console.warn(
      "[dataExport] DATAPIPE_EXPERIMENT_ID is a placeholder; not posting to DataPipe.",
    );
    return null;
  }
  const filename = `${payload.study}_${payload.sessionId}.json`;
  return {
    type: jsPsychPipe,
    action: "save",
    experiment_id: DATAPIPE_EXPERIMENT_ID,
    filename,
    data_string: JSON.stringify(payload),
  };
}

/**
 * Trigger a local JSON download in the browser. Fallback path if the
 * DataPipe POST fails or the experimenter is running offline.
 */
export function downloadPayloadAsJson(payload: SessionPayload): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${payload.study}_${payload.sessionId}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Read the Prolific ID from URL query parameters. Prolific routes
 * participants to the experiment with `?PROLIFIC_PID=...`.
 */
export function readProlificId(): string | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  return params.get("PROLIFIC_PID");
}
