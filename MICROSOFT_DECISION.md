# Use Azure Microsoft Decision 1

`MicrosoftDecisionAgent` uses Decision 1 to choose a finite browser operation and an observed target. It executes through Browser Use's native `BrowserSession` and `Tools`. The standard `Agent` continues to use generative models, with `ChatBrowserUse` recommended for general browser automation.

## Connect your deployment

Set the HTTPS resource root and credentials for an existing Azure deployment:

```bash
export AZURE_ENDPOINT=https://your-resource.services.ai.azure.com
export AZURE_DEPLOYMENT_NAME=decision-1
export AZURE_API_KEY=your-key
# Alternatively, set AZURE_ENTRA_TOKEN to a current Microsoft Entra bearer token.
```

The client sends `POST /providers/microsoft/v1/systemone`. An API key takes precedence when both credentials are set. It does not refresh Entra tokens, create cloud resources, or retry requests.

## Run with an existing browser

Start your browser and navigate to the page before running the agent:

```python
import asyncio

from browser_use import (
	AzureMicrosoftDecisionClient,
	BrowserSession,
	ChatBrowserUse,
	MicrosoftDecisionAgent,
)

async def main():
	browser = BrowserSession()
	await browser.start()
	try:
		page = await browser.get_current_page()
		await page.goto('https://example.com/your-form')
		agent = MicrosoftDecisionAgent(
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

Pass any configured `BaseChatModel` as `text_llm` to generate text for input fields. The helper runs only after Decision 1 chooses `input`. Its strict output schema is `{"text": "value"}`. Without a helper, the agent offers no input operation. The helper receives the selected index and observed field attributes, including its name and accessible label. Password, token, payment, and other recognizable secret fields are excluded. Do not use this agent for secret entry.

When you supply `browser`, you retain ownership. When you omit it, the agent creates a local session and stops that session when the run ends. This agent has no navigation operation, so supplying a browser on the target page is the useful starting point.

## Verify completion independently

`result.status == 'model_done'` means Decision 1 chose DONE. It does not prove that the task succeeded. `result.verified` defaults to `None`.

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

Each step sends one Decision 1 request containing an operation question and compatible target questions. The agent retains at most 250 targets in total, within a shared UTF-8 description budget. It retains click targets first, then input targets, then native select options, preserving observed order within each group. Other targets are unavailable for that step. Azure requires at least two candidates, so the agent resolves a sole compatible target locally. Only the selected operation's target answer can affect execution.

The agent checks the complete current URL, focus, DOM text, and observed node identities again after prediction and text generation. A changed observation returns `stale_state` without executing the selected action. Browser mutations are never retried. An action error stops the run with `action_error`; inspect the page before starting another run because the action may have partially completed.

A 429 response raises a rate-limit error immediately. Wait for your deployment's quota to reset, then start a new run to observe the browser again. There is no automatic pacing. In particular, a 1-request-per-minute deployment cannot run a continuous multi-step loop without caller-managed pacing.

The request has a conservative 28 KB UTF-8 JSON limit for context32k. Target descriptions and visible DOM text are bounded to fit. The freshness check still uses the complete observation. A target beyond the candidate or description budget is unavailable until the visible page changes. A request whose metadata alone exceeds the limit raises an error before prediction. The agent offers only click, non-secret text input, native select options, one-page scrolling, one-second wait, DONE, and BLOCKED. Navigation, custom tools, file uploads, arbitrary JavaScript, extraction, screenshots, custom ARIA dropdown options, and history replay are unsupported. Dynamic pages can invalidate an observation while a model is running. Native select options whose label and value cannot identify the intended option under the native handler's first-match rule are excluded. The freshness check is conservative and cannot make observation and browser execution atomic.

Run the offline contract and behavior tests with:

```bash
uv run pytest tests/ci/test_microsoft_decision.py
```
