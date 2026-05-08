/**
 * Data export pipeline. Two channels: DataPipe POST (forwards to OSF)
 * with a local-JSON-download fallback. The fallback runs whenever the
 * POST fails or the DataPipe ID is the placeholder, so no data is lost.
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

/** v4 UUID (or Math.random fallback for browsers without crypto.randomUUID). */
export function newSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const hex = (n: number) =>
    Math.floor(Math.random() * 16 ** n)
      .toString(16)
      .padStart(n, "0");
  return `${hex(8)}-${hex(4)}-${hex(4)}-${hex(4)}-${hex(12)}`;
}

/** Combine per-task results into the session JSON payload. */
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

/** jsPsych trial that POSTs to DataPipe; returns ``null`` when the ID is unset. */
export function buildDataPipeTrial(payload: SessionPayload): unknown {
  if (DATAPIPE_EXPERIMENT_ID === "PLACEHOLDER_BEFORE_PILOT") {
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

/** Trigger a local JSON download. Fallback when the DataPipe POST fails. */
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

/** Read ``?PROLIFIC_PID=...`` from the URL (Prolific's standard routing). */
export function readProlificId(): string | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  return params.get("PROLIFIC_PID");
}
