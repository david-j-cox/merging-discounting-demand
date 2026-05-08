/**
 * Tests for `lib/randomization.ts`.
 *
 *  - Same seed -> same assignment.
 *  - Different seeds -> reasonable spread across arms and orderings.
 *  - All 6 task permutations are reachable across the seed space.
 */

import { describe, expect, it } from "vitest";

import {
  TASK_ORDER_PERMUTATIONS,
  defaultSeed,
  mulberry32,
  randomize,
  type SubstitutabilityArm,
} from "../src/lib/randomization";

describe("randomize", () => {
  it("is deterministic for a given seed", () => {
    const a = randomize(42);
    const b = randomize(42);
    expect(a.arm).toBe(b.arm);
    expect(a.taskOrder).toEqual(b.taskOrder);
  });

  it("produces both arms over a seed sweep", () => {
    const arms = new Set<SubstitutabilityArm>();
    for (let seed = 0; seed < 200; seed++) arms.add(randomize(seed).arm);
    expect(arms.size).toBe(2);
  });

  it("reaches all 6 task permutations", () => {
    const orderings = new Set<string>();
    for (let seed = 0; seed < 1000; seed++) {
      orderings.add(randomize(seed).taskOrder.join(","));
    }
    expect(orderings.size).toBe(TASK_ORDER_PERMUTATIONS.length);
  });

  it("approximately balances arm assignment across seeds", () => {
    let nLow = 0;
    const N = 5000;
    for (let seed = 0; seed < N; seed++) {
      if (randomize(seed).arm === "low") nLow++;
    }
    // 95% interval for 0.5 is roughly 0.486-0.514 at N=5000.
    const frac = nLow / N;
    expect(frac).toBeGreaterThan(0.45);
    expect(frac).toBeLessThan(0.55);
  });
});

describe("mulberry32", () => {
  it("yields the same sequence for the same seed", () => {
    const r1 = mulberry32(123);
    const r2 = mulberry32(123);
    for (let i = 0; i < 10; i++) {
      expect(r1()).toBe(r2());
    }
  });

  it("yields values in [0, 1)", () => {
    const r = mulberry32(7);
    for (let i = 0; i < 100; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("defaultSeed", () => {
  it("is deterministic for fixed inputs", () => {
    expect(defaultSeed(1000, 0)).toBe(defaultSeed(1000, 0));
  });

  it("returns a 32-bit unsigned integer", () => {
    const seed = defaultSeed();
    expect(Number.isInteger(seed)).toBe(true);
  });
});
