# Verification findings

## Native select label normalization can select the wrong option

Status: resolved on 2026-10-24. The candidate mapper now uses the same trim character set as the native JavaScript matcher.

Affected source: `browser_use/decision.py` at the repository root, `DecisionAgent._option_text` / `_targets`; and `browser_use/browser/watchdogs/default_action_watchdog.py`, `on_SelectDropdownOptionEvent`.

Counterexample: a native select has option 0 with label `\ufeffB\ufeff` and no value attribute, followed by option 1 with label `B` and value `b`. The Python candidate mapper offers `b` for option 1. The native JavaScript trims the BOM characters from option 0's text and matches `b` against that earlier option. It reports success while selecting option 0. An earlier disabled option can also be matched in this configuration.

The problem is a mismatch between Python string normalization and JavaScript `trim()` during first-match selection. The bounded two-option exploration found four failing advertised targets. The exhaustive assertion remains unchanged and now passes. The full native regression verifies that the ambiguous second option is not advertised and that the remaining first option executes correctly.

Reproduce through `python3 verification/run.py` from `jev-ultrafast/`. The bounded check is `test_select_candidates_against_native_javascript`; the complete native Tools regression is `test_unicode_select_target_in_full_native_path`.

The existing first-match contract cannot uniquely address both colliding options. As with other label/value collisions, the ambiguous later option is excluded instead of risking a wrong-target action. No native selection API behavior was changed. The full suite passed 346 tests, all 26 feature groups, and all four mutation checks. The original Unicode cases remain in the bounded enumeration.
