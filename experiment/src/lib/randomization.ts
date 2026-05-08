/**
 * Randomization for the experimental battery.
 *
 * Two factors per CLAUDE.md §4 Phase 4 step 6:
 *
 *  - **Substitutability arm** (between-subjects, 50/50): low vs high.
 *    Determines whether the immediate-reward option in Task 2's titration
 *    is framed as a substitute commodity (high-IF arm) or an unrelated
 *    commodity (low-IF arm). Pre-registered as the H3 manipulation.
 *  - **Task order** (within-subjects): a Latin-square-style permutation
 *    of {Task 1, Task 2, Task 3} per subject so order effects average out.
 *    Six orderings exist for three tasks; the assignment cycles through
 *    them in a deterministic-by-seed manner.
 *
 * The randomizer is seedable so the unit tests and the E2E test can run
 * with reproducible assignments. In production we seed from
 * `Date.now() ^ navigator.userAgent.length` to get effectively-random
 * but auditable assignments.
 */

export type SubstitutabilityArm = "low" | "high";
export type TaskName = "task1_purchase" | "task2_effort_discount" | "task3_effort_purchase";

export interface RandomizationAssignment {
  /** Substitutability arm. */
  arm: SubstitutabilityArm;
  /** Order in which the three tasks will run. */
  taskOrder: TaskName[];
  /** Seed used to generate this assignment. */
  seed: number;
}

const TASKS: TaskName[] = [
  "task1_purchase",
  "task2_effort_discount",
  "task3_effort_purchase",
];

/** All 3! = 6 orderings of the three tasks. */
const TASK_PERMUTATIONS: TaskName[][] = (() => {
  const perms: TaskName[][] = [];
  function permute(arr: TaskName[], start: number): void {
    if (start === arr.length) {
      perms.push([...arr]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      [arr[start], arr[i]] = [arr[i] as TaskName, arr[start] as TaskName];
      permute(arr, start + 1);
      [arr[start], arr[i]] = [arr[i] as TaskName, arr[start] as TaskName];
    }
  }
  permute([...TASKS], 0);
  return perms;
})();

/**
 * Mulberry32: tiny seedable PRNG. Returns a function that yields uniform
 * doubles in [0, 1). Identical seed -> identical sequence; this is the
 * property the test suite relies on.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Assign a participant to an arm and a task order.
 *
 * `seed` is a 32-bit integer. The arm is `low` if the first random draw
 * is < 0.5 and `high` otherwise. The task order is the
 * `floor(rng() * 6)`-th permutation.
 */
export function randomize(seed: number): RandomizationAssignment {
  const rng = mulberry32(seed);
  const arm: SubstitutabilityArm = rng() < 0.5 ? "low" : "high";
  const orderIdx = Math.floor(rng() * TASK_PERMUTATIONS.length);
  const taskOrder = (TASK_PERMUTATIONS[orderIdx] ?? TASK_PERMUTATIONS[0]) as TaskName[];
  return { arm, taskOrder: [...taskOrder], seed };
}

/**
 * Generate a default seed from current time and a small entropy source.
 * Deterministic given the same `now` and `salt`; replace in tests by
 * passing a fixed seed to `randomize` directly.
 */
export function defaultSeed(now: number = Date.now(), salt: number = 0): number {
  return ((now ^ (salt * 2654435761)) >>> 0) | 0;
}

export const TASK_ORDER_PERMUTATIONS = TASK_PERMUTATIONS;
