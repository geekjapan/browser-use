# Jev Ultrafast

Read README.md before editing. Keep the loop small: page -> indexed elements -> operation + target -> execution.

This package lives in `jev-ultrafast/` of the browser-use repository as a separate uv project (Python >= 3.12, own `uv.lock`). Run the checks below from this directory. `python3 verification/run.py` verifies both Decision implementations and defaults to the repository root.

- The input is one natural-language goal. Do not add site-specific plans or hardcoded field values.
- The configured decision provider (Azure by default, or OpenAI Decisions) chooses an operation and operation-specific target heads in one request. Consume only the selected operation's target.
- Targets must map to observed elements and supported operations. Never let the model emit selectors or executable code.
- TYPE_TEXT invokes the text LLM. Cache a stale retry's value only while its entire helper input is identical.
- Never retry a browser mutation. Log execution before observing its result.
- Screenshots are optional; the model does not consume them. Keep demonstration footage at its original speed.
- Keep credentials server-side and .env ignored. Tests must not call paid APIs.
- Verify actual final outcomes independently. A DONE choice is not proof of success.
- Keep examples, README claims, raw evidence, and model-call counts consistent.
- Do not commit or push unless the user requests it.

Checks: uv run ruff check ., uv run pytest, node --check jev_ultrafast/static/app.js, node tests/check_demo_ui.cjs, uv build.
