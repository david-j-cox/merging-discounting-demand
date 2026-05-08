/**
 * Per-subject randomization. Two factors:
 *
 * - Substitutability arm (between-subjects 50/50, the H3 manipulation).
 * - Task order — uniform over the 6 permutations of the three tasks.
 *
 * Seedable so tests run reproducibly.
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

/** Mulberry32: tiny seedable PRNG yielding uniform doubles in [0, 1). */
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

/** Assign one participant from a 32-bit integer seed. */
export function randomize(seed: number): RandomizationAssignment {
  const rng = mulberry32(seed);
  const arm: SubstitutabilityArm = rng() < 0.5 ? "low" : "high";
  const orderIdx = Math.floor(rng() * TASK_PERMUTATIONS.length);
  const taskOrder = (TASK_PERMUTATIONS[orderIdx] ?? TASK_PERMUTATIONS[0]) as TaskName[];
  return { arm, taskOrder: [...taskOrder], seed };
}

/** Default seed from current time + a salt; deterministic given fixed inputs (for tests). */
export function defaultSeed(now: number = Date.now(), salt: number = 0): number {
  return ((now ^ (salt * 2654435761)) >>> 0) | 0;
}

export const TASK_ORDER_PERMUTATIONS = TASK_PERMUTATIONS;
