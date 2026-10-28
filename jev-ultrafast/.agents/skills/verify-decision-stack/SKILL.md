---
name: verify-decision-stack
description: Verify Azure/OpenAI finite-choice browser automation across jev-ultrafast and the browser-use DecisionAgent in this repository, including bounded state-space exploration, real Chromium actions, response validation and inspector controls. Use after changes to the covered files or before reviewing these integrations.
---

# Verify the decision browser stack

Run from the `jev-ultrafast/` directory of the browser-use repository:

```bash
cd jev-ultrafast
python3 verification/run.py
```

On Windows, run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-windows.ps1 -Mode verify` instead. This enables UTF-8 for Python and resolves an installed uv executable even when a Scoop shim is broken. For the authenticated local inspector, use `-Mode demo -Provider openai` or `-Provider azure`; credentials stay in the private per-user file described in README.md, outside the checkout.

`--browser-use` defaults to the repository root, the parent of `jev-ultrafast/`; pass it only to verify an out-of-tree checkout. The repository root and `jev-ultrafast/` each need their uv lockfile and the Decision API implementation. `uv` and Node.js must be installed. The runner uses Python 3.12, the locked Browser Use test dependencies, and an ephemeral editable install of this Jev checkout so its declared Browser Harness/HTTP2 dependencies are available. It does not change either lockfile or install dependencies globally.

The repository root does not track `uv.lock`; `jev-ultrafast/uv.lock` is tracked. On a fresh clone, first run `uv lock` and `uv sync --frozen` at the repository root, then `uv sync --frozen` in `jev-ultrafast/`. Keep the generated lock for later runs. A newly resolved lock can differ from the original machine; compare recorded lock hashes and runtime versions when reproducing a result.

Read `artifacts/verification/REPORT.md`, `report.json`, the per-stage logs and JUnit XML. Exit 0 means every stage and mapped feature passed and all four deliberately broken variants were detected. Skips, absent tests, setup errors, timeouts and known product defects are failures, never coverage. Do not remove a failing assertion to obtain a pass.

## What is checked

`verification/features.json` is the executable feature-to-test map. It covers the eight existing Decision operations, provider adapters, validation, current form values, native controls, stale observations, text helpers, budgets, lifecycle, history/recording and the local inspector. This is not the unrelated generative `browser_use.Agent` feature set.

The bounded properties operate on the real Python functions:

- Both choice validators: all 3,000 tuples from three choice identities and a ten-value domain for each of two probabilities and confidence. Invalid external values must not authorize execution.
- `jev_ultrafast.agent.Agent.command`: all 780 command strings of length 1–4 over predict, act, wrong-fingerprint act, stale and fresh. Under sequential callers, one prediction permits at most one executor attempt.
- `DecisionAgent.run`: all eight operations crossed with ten observed-state changes. An obsolete observation must stop before action or terminal acceptance.
- Native select mapping: all two-option tables from eight label/value prototypes and four disabled masks. Every offered candidate must select its advertised option index through the exact JavaScript extracted from the native handler. A second regression also exercises the full native Tools path.

Mutation checks compile scratch variants of the actual functions in isolated subprocesses: remove decision consumption, remove freshness comparison, or remove either validator's maximum-probability check. The corresponding baseline must pass and the mutated assertion must fail. No checkout is edited by these mutations. A mismatched source replacement is a drift error, not a detected mutant.

The Chromium checks use a fresh headless BrowserSession. The real Jev executor is connected to that owned browser by a CDP transport bridge; daemon discovery is replaced so the user's existing Chrome tabs are untouched. Decision and text HTTP responses are deterministic test fixtures. No cloud account, real API key or paid inference is used.

## Source synchronization

Last source review: browser-use `76e5847814d6ba5b83e5b9948ffb6648c2d76feb`; jev-ultrafast `ee28161b54378c1b5d7e97ed683c284cd734d43e`. Since 2026-10-28 jev-ultrafast lives in `jev-ultrafast/` of the browser-use repository, imported from `geekjapan/jev-ultrafast@f4d66bd`, so both `repository` entries in `report.json` record the same commit.

`verification/sources.json` lists covered files. `report.json` records their SHA-256 digests and checkout commits on each run. Review source changes against these commits before treating prior evidence as current. Advance the review commits only after checking that the assertions, extracted JavaScript, mutation needles and stated assumptions still describe the implementation. Use `/maintain-formal-verification-skill` for this maintenance workflow.

If a failure is in a product function, retain the counterexample and document it in `verification/FINDINGS.md`. If it is in test setup or modeling, repair the test and rerun its affected stage. A function returning DONE, a successful click, or a child process exiting is not an independent application outcome.

## Limits

This is bounded exhaustive testing, not an unbounded proof or a claim of correctness for every website. Callback internals outside each modeled boundary, concurrent direct library callers, browser/model races after the last check, arbitrary Unicode/options, provider reasoning quality, cloud browsers, CAPTCHA, authenticated sites and platform-specific behavior remain outside the stated bounds. Native handlers can make several CDP calls; the suite does not claim exactly-once DOM side effects.

For the resolved native select counterexample and its regression contract, read `verification/FINDINGS.md`. Do not mark a release verified while its regression still fails. Publication, merges, live API spending and changes to either upstream repository require their own authorization.
