# Experiment build

jsPsych 8.x experiment for the effort-demand unification study. Companion
to the Python analysis package in `../src/edu/`.

## Setup

```bash
cd experiment
npm install
npm run dev      # vite dev server at http://localhost:5173
npm test         # vitest unit + end-to-end tests
npm run build    # production bundle in dist/
npm run lint     # tsc --noEmit (strict TypeScript check)
```

## Layout

```
src/
  config.ts                     constants matching the simulation suite
  main.ts                       full-battery orchestration (entry point)
  tasks/
    calibration.ts              key-press maximum-rate calibration
    purchaseTask.ts             Task 1: standard hypothetical purchase task
    effortDiscounting.ts        Task 2: effort discounting w/ titration
    effortPurchaseTask.ts       Task 3: novel effort purchase task
  lib/
    titration.ts                Du, Green & Myerson 2002 algorithm
    randomization.ts            arm + task-order randomization
  components/                   reusable jsPsych timeline fragments
public/                         static assets
tests/                          vitest tests, including an E2E with
                                programmatic responses
```

## Known issue: dev-only vulnerability

`npm audit` reports a moderate-severity issue in esbuild (transitive via
vite). The advisory affects only the **development server** — it lets a
malicious website send cross-origin requests to a running `npm run dev`
instance. The production build (`npm run build`) is static HTML+JS with
no server attached, so the deployed experiment is unaffected.

We accept the dev-server warning rather than upgrading to vite 8, which
would be a breaking change. Run `npm run dev` only on localhost; do not
expose the dev server to the public internet.

## Pre-registration coupling

Several constants in `config.ts` (price array, effort fractions, titration
depth, etc.) match the Phase 3 simulation suite in `src/edu/simulation/`.
Changing them after the OSF pre-registration is filed requires updating
the pre-registration. See `../docs/analysis_plan.md`.
