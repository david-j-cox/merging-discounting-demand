// Vitest setup. Runs once per test process before any test file.
// jsPsych touches `window.performance` and `requestAnimationFrame`; jsdom
// provides reasonable defaults but we patch a couple of edge cases here.

if (typeof globalThis.requestAnimationFrame === "undefined") {
  globalThis.requestAnimationFrame = (cb: FrameRequestCallback): number =>
    setTimeout(() => cb(performance.now()), 16) as unknown as number;
  globalThis.cancelAnimationFrame = (id: number): void => {
    clearTimeout(id as unknown as ReturnType<typeof setTimeout>);
  };
}
