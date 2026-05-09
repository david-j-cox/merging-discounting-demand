/**
 * Experiment-wide configuration constants.
 *
 * Most values are pre-registered modeling choices that match the simulation
 * suite (`src/edu/simulation/generate.py`). Changing them after the
 * pre-registration is filed requires updating the OSF pre-registration too.
 */

export const STUDY_NAME = "edu-pilot";
export const STUDY_VERSION = "0.1.0";

/**
 * DataPipe experiment ID for the edu-pilot study (OSF project ``rys6a``).
 * Replace with ``"PLACEHOLDER_BEFORE_PILOT"`` to disable POSTs during
 * local development without an OSF target.
 */
export const DATAPIPE_EXPERIMENT_ID: string = "SfEZb5tfFFLa";

/**
 * Whether the DataPipe POST is flagged as test data (true) or production
 * (false). Keep ``true`` through pilot validation; flip to ``false`` only
 * for the main study after the pilot confirms Task 3 yields analysable
 * data.
 */
export const DATAPIPE_TEST_MODE = true;

/**
 * 17-price array for Task 1 (standard purchase task). Matches the array
 * in `src/edu/simulation/generate.py` `PRICE_ARRAY_17`.
 */
export const PURCHASE_PRICES_USD: readonly number[] = [
  0.01, 0.05, 0.13, 0.25, 0.5, 1, 2, 5, 13, 25, 50, 100, 200, 350, 500, 800, 1120,
];

/**
 * Effort fractions for Task 2 (effort discounting). Each value `f` means
 * "target rate = f * p_max for 30 seconds". Matches `EFFORT_FRACTIONS_DEFAULT`
 * in the simulation suite.
 */
export const EFFORT_FRACTIONS: readonly number[] = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85];

/** Effort-purchase-task price array; effort cost per unit reward acquired. */
export const EFFORT_PRICES: readonly number[] = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0];

/** Maximum reward magnitude (USD-equivalent) used in adjusting-amount titration. */
export const REWARD_MAX_USD = 10.0;

/** Adjusting-amount titration depth (Du, Green & Myerson 2002 standard = 6). */
export const TITRATION_STEPS = 6;

/** Sustained-rate trial duration in seconds, for verification trials. */
export const SUSTAINED_TRIAL_SECONDS = 30;

/** Calibration block: number of maximal trials and rest period. */
export const CALIBRATION_TRIALS = 3;
export const CALIBRATION_TRIAL_SECONDS = 10;
export const CALIBRATION_REST_SECONDS = 30;

/** Fraction of titration trials selected for verification (sparse real-effort). */
export const VERIFICATION_TRIAL_FRACTION = 0.10;

/**
 * Maximum physically-plausible key-press rate (presses/sec). Rates above
 * this are treated as auto-repeat / macro and rejected at calibration.
 */
export const MAX_PLAUSIBLE_RATE = 15.0;

/** Tolerance window for sustained-rate verification (fraction of target). */
export const RATE_TOLERANCE = 0.20;

// ---------------------------------------------------------------------------
// Quality-check thresholds
// ---------------------------------------------------------------------------

/** Per-trial RT lower bound for Task 2 choice trials (ms). */
export const RT_MIN_MS = 500;

/** Per-trial RT upper bound for Task 2 choice trials (ms). */
export const RT_MAX_MS = 60_000;

/**
 * Subject-level RT-bound flag fires when this fraction of Task 2 choice
 * trials hit either bound. ~25% of 36 trials = 9 trials.
 */
export const RT_BOUND_SUBJECT_FRACTION = 0.25;

/** Subject-level duration lower bound (ms). Sessions shorter than this
 *  are flagged as implausibly fast. ~8 minutes. */
export const DURATION_MIN_MS = 8 * 60_000;

/** Subject-level duration upper bound (ms). Sessions longer than this
 *  are flagged as walked-away-mid-task. ~60 minutes. */
export const DURATION_MAX_MS = 60 * 60_000;

/**
 * Catch trial in Task 1: a normal-looking purchase trial where the
 * preamble explicitly asks the participant to enter a specific number
 * (the value below). Failing to enter exactly this number flags the
 * subject's quality.
 */
export const CATCH_TRIAL_EXPECTED_QUANTITY = 7;

/**
 * Position (0-indexed) at which the catch trial is inserted into Task 1.
 * Hard-coded rather than randomized so the manipulation is reproducible
 * across runs and across subjects (the demand-curve fitting in Python
 * skips this trial by `price === null`).
 */
export const CATCH_TRIAL_POSITION = 9;
