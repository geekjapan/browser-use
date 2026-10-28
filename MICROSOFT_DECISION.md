# Use Azure Microsoft Decision 1 or OpenAI Decisions

`DecisionAgent` uses either backend to choose a finite browser operation and an observed target. It executes through Browser Use's native `BrowserSession` and `Tools`. The standard `Agent` continues to use generative models, with `ChatBrowserUse` recommended for general browser automation.

## Connect a backend

### Azure Microsoft Decision 1

Set the HTTPS resource root and credentials for an existing Azure deployment:

```bash
export AZURE_ENDPOINT=https://your-resource.services.ai.azure.com
export AZURE_DEPLOYMENT_NAME=decision-1
export AZURE_API_KEY=your-key
# Alternatively, set AZURE_ENTRA_TOKEN to a current Microsoft Entra bearer token.
```

The client sends `POST /providers/microsoft/v1/systemone`. An API key takes precedence when both credentials are set. It does not refresh Entra tokens, create cloud resources, or retry requests.

### OpenAI Decisions

Set your OpenAI credentials and optional model name:

```bash
export OPENAI_API_KEY=your-key
export OPENAI_DECISION_MODEL=gpt-6-luna
```

`OpenAIDecisionClient()` sends `POST https://api.openai.com/v1/decisions` with bearer authentication. It defaults to `gpt-6-luna`; pass `model=` or set `OPENAI_DECISION_MODEL` to override it. The client sends the observed state as JSON text in `input`, with named choice questions. It does not retry requests or follow redirects.

### Microsoft Foundry and Google Gemini

`AzureMicrosoftDecisionClient` is the Microsoft Foundry client: Foundry serves `Microsoft-Decision-1` on the same `/providers/microsoft/v1/systemone` route, with your deployment name as `model`. The Gemini API and Google AI Studio have no decision endpoint, so there is no Gemini decision client. For Google models, use the on-device ranker below.

In the example below, replace `AzureMicrosoftDecisionClient()` with `OpenAIDecisionClient()` and import `OpenAIDecisionClient` to use OpenAI. Both clients use the same agent and native browser guards. `MicrosoftDecisionAgent` remains an alias of `DecisionAgent` for existing imports.

## Run with an existing browser

Start your browser and navigate to the page before running the agent:

```python
import asyncio

from browser_use import (
	AzureMicrosoftDecisionClient,
	BrowserSession,
	ChatBrowserUse,
	DecisionAgent,
)

async def main():
	browser = BrowserSession()
	await browser.start()
	try:
		page = await browser.get_current_page()
		await page.goto('https://example.com/your-form')
		agent = DecisionAgent(
			task='Enter Ada in the name field and select Blue in the color list.',
			browser=browser,
			decision_model=AzureMicrosoftDecisionClient(),
			text_llm=ChatBrowserUse(),
		)
		result = await agent.run(max_steps=10)
		print(result.model_dump())
		# Check the actual fields or application outcome independently here.
	finally:
		await browser.stop()

asyncio.run(main())
```

Pass any configured `BaseChatModel` as `text_llm` to generate text for input fields. The helper runs only after the decision model chooses `input`. Its strict output schema is `{"text": "value"}`. Without a helper, the agent offers no input operation. The helper receives the selected index and observed field attributes, including its name and accessible label. Password, token, payment, and other recognizable secret fields are excluded. Do not use this agent for secret entry.

When you supply `browser`, you retain ownership. When you omit it, the agent creates a local session and stops that session when the run ends. This agent has no navigation operation, so supplying a browser on the target page is the useful starting point.

## Rank targets on device with an embedding model

On pages with more targets than the request can hold, the agent keeps the first observed targets, so a relevant field late on the page can be unavailable. Pass a `target_ranker` to order all observed targets by relevance to the task before that cut. The decision model still chooses the operation and target; the ranker only decides which targets it sees. The ranker runs only when some targets would be dropped.

`MediaPipeTargetRanker` runs MediaPipe Decision Maker locally with an EmbeddingGemma 2 model. Install the optional extra and download [`embeddinggemma-2-text-270m.litertlm`](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm) (165 MB, Apache-2.0):

```bash
uv add "browser-use[mediapipe]"
```

MediaPipe 1.2.0 publishes wheels for macOS arm64, Linux x86_64/aarch64 and Windows; there is no macOS x86_64 wheel.

```python
from browser_use import DecisionAgent, MediaPipeTargetRanker, OpenAIDecisionClient

ranker = MediaPipeTargetRanker('embeddinggemma-2-text-270m.litertlm')
try:
	agent = DecisionAgent(task, decision_model=OpenAIDecisionClient(), browser=browser, target_ranker=ranker)
	result = await agent.run(max_steps=10)
finally:
	ranker.close()
```

The ranker receives only the task text and the target descriptions; nothing leaves the machine. MediaPipe silently truncates context beyond `max_num_tokens` (default 2048), so a longer task raises an error instead. Pass `maker=` to reuse a `DecisionMaker` you configured; it remains yours to close, and `max_num_tokens` must match the value you built it with, or the guard cannot detect truncation. Target descriptions are not guarded; DecisionAgent already bounds them, and a truncated description only affects ranking. A ranking that omits a target or returns a non-finite score stops the run before prediction.

Measured on an Apple Silicon CPU with synthetic product pages: with about 270 targets, the intended target ranked within the top 20 in 12 of 12 tasks. With about 600 targets, the description budget holds about 88 targets; the intended target was offered to the decision model in 4 of 6 tasks with the ranker and 1 of 6 without it. The two misses were form fields crowded out by similar product buttons. The first call embeds every target description (about 14 s for 270 targets, 37 s for 600); repeated descriptions are cached (about 60 ms per call). The ranker receives at most 1000 targets, input fields and select options first, then clicks in observed order; later targets are offered only if budget remains after the ranked ones. The MediaPipe Python GPU delegate is limited to Ubuntu.

An embedding model is not used as the decision model. In the same measurements it chose the wrong target whenever the DOM was in its context, and it kept choosing `input` after the field already held the requested value.

## Verify completion independently

`result.status == 'model_done'` means the decision model chose DONE. It does not prove that the task succeeded. `result.verified` defaults to `None`.

To attach an independent check, pass an asynchronous callback that reads the application outcome:

```python
async def verify(browser):
	page = await browser.get_current_page()
	observed = await page.evaluate("() => document.querySelector('#name').value === 'Ada' && document.querySelector('#color').value === 'blue'")
	return observed == 'True'

result = await agent.run(max_steps=10, outcome_verifier=verify)
assert result.status == 'model_done' and result.verified is True
```

The callback runs only after DONE. A false result remains `verified=False` even when the model declares completion.

## Handle bounded execution

Both backends receive the current value of supported non-secret text fields and the current label of native select fields. Values longer than 160 characters are marked as truncated in target descriptions. This helps the model move on from completed inputs and selections; use an outcome verifier to check the actual result.

Each step sends one decision request containing an operation question and compatible target questions. The agent retains at most 250 targets in total, within a shared UTF-8 description budget. It retains click targets first, then input targets, then native select options, preserving observed order within each group. Other targets are unavailable for that step. Both backends require at least two candidates, so the agent resolves a sole compatible target locally. Only the selected operation's target answer can affect execution. A refusal for an unused target is ignored; a missing, refused, or invalid required answer stops the run before execution. Duplicate or unknown OpenAI answer names and duplicate probability values are rejected.

The agent checks the complete current URL, focus, DOM text, and observed node identities again after prediction and text generation. A changed observation returns `stale_state` without executing the selected action. Browser mutations are never retried. An action error stops the run with `action_error`; inspect the page before starting another run because the action may have partially completed.

A 429 response raises a rate-limit error immediately. Wait for your provider's quota to reset, then start a new run to observe the browser again. There is no automatic pacing. In particular, a 1-request-per-minute deployment cannot run a continuous multi-step loop without caller-managed pacing.

The request has a conservative 28 KB UTF-8 JSON limit for context32k. Target descriptions and visible DOM text are bounded to fit. The freshness check still uses the complete observation. A target beyond the candidate or description budget is unavailable until the visible page changes. A request whose metadata alone exceeds the limit raises an error before prediction. The agent offers only click, non-secret text input, native select options, one-page scrolling, one-second wait, DONE, and BLOCKED. Navigation, custom tools, file uploads, arbitrary JavaScript, extraction, screenshots, custom ARIA dropdown options, and history replay are unsupported. Dynamic pages can invalidate an observation while a model is running. Native select options whose label and value cannot identify the intended option under the native handler's first-match rule are excluded. The freshness check is conservative and cannot make observation and browser execution atomic.

Run the offline contract and behavior tests with:

```bash
uv run pytest tests/ci/test_microsoft_decision.py
```

The on-device ranker check needs the downloaded model, so it lives outside `tests/ci`:

```bash
MEDIAPIPE_DECISION_MODEL_PATH=embeddinggemma-2-text-270m.litertlm uv run --extra mediapipe pytest tests/scripts/test_mediapipe_ranker_live.py
```
