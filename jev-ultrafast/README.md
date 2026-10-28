# Decision browser agent

This fork of [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) uses
Microsoft Decision 1 in Azure Foundry (default) or OpenAI Decisions to select browser operations and targets.
A separate OpenAI-compatible LLM supplies text for `TYPE_TEXT`. No Jev API key or
TypeSafe requests are needed. The Python package and `jev` command keep their
upstream names. It lives in the `jev-ultrafast/` directory of the
[geekjapan/browser-use](https://github.com/geekjapan/browser-use) repository as a
separate uv project; the former standalone repository is no longer needed.

The browser runs through Browser Harness. This is separate from the standard
`browser-use` Agent, which expects a generative model.

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. Only supported operations and targets are offered.

```text
                      one decision request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements. A single compatible target is resolved locally; the inspector labels it as the only target rather than a model confidence. Native dropdown choices carry an observed element/option index.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome. The screenshot renderer adds labels afterward; it does not drive the browser.

## Try it

```bash
git clone --branch feat/openai-decisions-api https://github.com/geekjapan/browser-use.git
cd browser-use/jev-ultrafast
uv sync
cp .env.example .env
# Set AZURE_ENDPOINT, AZURE_API_KEY, and TEXT_MODEL_API_KEY.
uv run jev
```

Use your Azure resource root URL for `AZURE_ENDPOINT`, for example
`https://your-resource.cognitiveservices.azure.com`. Set `AZURE_DEPLOYMENT_NAME`
to your deployment name. Its default is `decision-1`. The client calls
`/providers/microsoft/v1/systemone` and authenticates with the `api-key` header.
Keep credentials in the ignored `.env` file or environment, never in source.

To use [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions), set:

```dotenv
DECISION_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
OPENAI_DECISION_MODEL=gpt-6-luna
```

Azure remains the default (`DECISION_PROVIDER=azure`); other provider values fail
before a request. OpenAI uses `POST https://api.openai.com/v1/decisions` with Bearer
authentication, a JSON-encoded state in `input`, and named choice questions.
Azure uses its existing `state` and question-map contract. Both select from the
same observed targets, resolve a sole compatible target locally, and validate
only the selected operation's target. The trace retains the actual request body,
raw provider answers, returned model, and usage. OpenAI and Azure decision keys
are never used for the independent text helper; `TEXT_MODEL_API_KEY` is still
required for `TYPE_TEXT`. The inspector displays the configured provider/model.
See the [OpenAI request schema](https://developers.openai.com/api/reference/resources/decisions/methods/create).

For Microsoft Entra ID, leave `AZURE_API_KEY` empty and supply a current token:

```bash
export AZURE_ENTRA_TOKEN="$(az account get-access-token \
  --resource https://cognitiveservices.azure.com --query accessToken -o tsv)"
uv run jev
```

Refresh the token when it expires. If both credentials are set, the API key wins.
See [Microsoft's deployment and authentication guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).

A deployment limited to one request per minute cannot run the agent continuously.
Configure enough request capacity before an automatic run. On HTTP 429, the client
stops without executing an action. Wait for quota to reset, then choose again;
page freshness is checked before the next prediction and execution.

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from jev_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** The decision provider consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **One browser call per snapshot.** Read visible controls, their names, values, and text atomically. Keep references to the actual DOM nodes.
- **Validate the selected target.** Clicks check the document, form values, target, and nearby context. Animation alone does not force another prediction. Resolve current geometry and reject covered controls before input.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms. These reads happen after execution is logged.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node. The executor rechecks page freshness and click occlusion. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](jev_ultrafast/agent.py) | The complete loop and text-helper handoff |
| [snapshot.js](jev_ultrafast/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](jev_ultrafast/browser.py) | Browser connection, current geometry, execution |
| [model.py](jev_ultrafast/model.py) | Dynamic operation/target heads and text generation |
| [questions.py](jev_ultrafast/questions.py) | Model instructions |
| [demo.py](jev_ultrafast/demo.py) | Local inspector |

## Upstream Jev evidence and limits

The measurements below were produced upstream with Jev. They do not
measure this fork or Microsoft Decision 1. No Azure speedup or browser-task
success rate is established by those results.

The upstream recording was a **7,073 ms** Google Flights run. Timing starts after initial page observation and includes model calls, generated text, browser work, stale decisions, and loading waits. A fresh independent check verifies the one-way setting, Zürich, London, September 20, 2026, and visible flight options.

In six alternating runs with identical models and settings, both versions passed **3/3**. Median task time went from **9.450 s → 7.092 s**, a **25% reduction**; median browser protocol calls went from **1,092 → 101**. This is three repeats of one task on one browser profile, not a general reliability benchmark.

The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter task in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice still requires independent outcome verification. The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling, and arbitrary keyboard widgets remain outside this MVP. Owned tabs share the existing Chrome profile.

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
node tests/check_demo_ui.cjs
uv build
```

Tests are offline. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls. Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.

### Verify both Decision implementations

From this directory; the enclosing repository root is used automatically:

```bash
python3 verification/run.py
```

The suite combines both existing regression suites, bounded enumeration of real Python code, isolated Chromium workflows, inspector checks, and deliberate broken-code variants. It does not call paid APIs or use the user's Chrome session. Results and the feature matrix are written to `artifacts/verification/REPORT.md` and `report.json`; any skipped, missing, failed, or undetected mutation check prevents success.

The scope is the existing eight-operation Decision integrations, not the entire generative Browser Use Agent. Current findings remain failing tests; see [verification/FINDINGS.md](verification/FINDINGS.md). Source synchronization and precise bounds are documented in the [project verification skill](.agents/skills/verify-decision-stack/SKILL.md).

### 別のPCへ検証スキルとテストを導入する

このリポジトリには、実行用の `verify-decision-stack` スキルとテストスイートを同梱しています。スキルを生成した `create-formal-verification-skill` を別途インストールする必要はありません。

1. Git、uv、Node.js、Google ChromeまたはChromiumを用意します。macOSとWindowsで実行確認済みです。Linuxでの動作は未確認です。
2. 以下のように **geekjapan/browser-use** を取得します。`jev-ultrafast` はこのリポジトリの `jev-ultrafast/` フォルダーに含まれています。Decision API対応は現時点でこの機能ブランチにあるため、`main` ではなく指定ブランチを使います。

   ```bash
   git clone --branch feat/openai-decisions-api https://github.com/geekjapan/browser-use.git
   cd browser-use
   uv python install 3.12
   uv lock
   uv sync --frozen
   uv sync --frozen --project jev-ultrafast
   ```

   リポジトリのルートは `uv.lock` をGit管理していないため（`jev-ultrafast/uv.lock` は管理対象）、初回だけ `uv lock` で生成します。解決される依存バージョンは導入時点に依存します。生成したロックを保持し、以降の実行は `--frozen` を使ってください。各検証結果にはロックのハッシュと実行時の主要バージョンが記録されます。

   Chrome/Chromiumがない場合は、Browser Useの既存インストールコマンドでChromiumを取得できます。

   ```bash
   uv run browser-use install
   ```

3. `jev-ultrafast` フォルダーで検証を実行します。クラウドアカウント、APIキー、`.env` は不要です。初回は依存パッケージのダウンロードにネット接続が必要です。

   ```bash
   cd jev-ultrafast
   uv run --no-project --python 3.12 python verification/run.py
   ```

   テスト専用のブラウザープロファイルを作成するため、既存のChromeプロファイルをコピーする必要はありません。結果は `artifacts/verification/REPORT.md` と `report.json` に出力されます。現在の構成では352件が成功し、終了コードは0になります。古いbrowser-useではドロップダウンの回帰テストが失敗するため、[修正記録](verification/FINDINGS.md)と使用中のコミットを確認してください。その他の失敗は個別のログで確認してください。

4. コーディングエージェントには、このリポジトリを作業フォルダーとして開き、`jev-ultrafast/.agents/skills/verify-decision-stack/SKILL.md` を読んで検証するよう指示します。スキル一覧から見つからない場合も、このファイルを直接指定できます。ファイル内の手順と `verification/` が一組なので、`SKILL.md` だけを別の場所へコピーしないでください。

`--browser-use` は既定でリポジトリのルートを指します。別の場所にあるチェックアウトを検証する場合だけ指定してください。更新時はスキルの「Source synchronization」にある確認済みコミットと差分を照合してから再実行してください。上流の `browser-use/*` リポジトリへのpushや変更は不要です。

**秘密情報を持ち込まないこと:** `.env`、`.env.*`（公開用の `.env.example` を除く）、APIキー、トークン、Cookie、ブラウザープロファイル、`.venv`、`artifacts/`、個人のエージェント設定を別PCへの共有物やGitコミットに含めないでください。この検証で使う認証文字列はテスト用のダミーです。実サービスへ接続する場合の資格情報は、各PCで別途安全に設定してください。

### Windowsでの起動

前述の準備を終えたら、`jev-ultrafast` フォルダーで以下を実行します。起動スクリプトはUTF-8を指定し、Scoopのショートカットが壊れている場合もインストール済みのuv本体を探します。OS全体の設定は変更しません。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-windows.ps1 -Mode verify
```

実APIを使う画面は、Git管理外の `%USERPROFILE%\.config\decision-stack\.env` に資格情報を設定してから起動します。このファイルは利用ユーザーとSYSTEMだけが読めるよう権限を制限してください。ファイルの内容をGitや実行ログへ出力しないでください。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-windows.ps1 -Mode demo -Provider openai
# または -Provider azure
```

サーバー起動後、`http://127.0.0.1:8766` を開きます。同時に起動する画面は1つにしてください。デモは一時プロファイルのChromeを使用し、通常のChromeプロファイルとは分離します。終了はCtrl+Cです。OpenAIの入力補助モデルを使用する場合は `TEXT_MODEL_BASE_URL=https://api.openai.com/v1`、`TEXT_MODEL_REASONING=none` と、利用可能な `TEXT_MODEL` を設定します。

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
