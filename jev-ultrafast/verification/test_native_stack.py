"""Real Chromium and both executors; only provider HTTP and daemon discovery are replaced."""

import ast
import asyncio
import importlib.util
import itertools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from browser_use import AzureMicrosoftDecisionClient, DecisionAgent, OpenAIDecisionClient
from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.decision import FieldText
from browser_use.llm.base import BaseChatModel
from browser_use.llm.views import ChatInvokeCompletion
from test_microsoft_decision import agent, node, option, state

from jev_ultrafast import agent as jev
from jev_ultrafast import browser as jev_browser
from jev_ultrafast import model

PAGE = """<!doctype html><title>Decision stack fixture</title>
<style>body{margin:24px}label{display:block}
input,select,button{margin:8px}#bottom{margin-top:1800px}</style>
<label>Name<input id="name" name="name"></label>
<label>Color<select id="color"><option value="red">Red</option><option value="cyan">Blue</option></select></label>
<label><input id="check" type="checkbox">Enabled</label>
<label><input id="radio" type="radio">Option A</label>
<button id="confirm"
onclick="window.clicks++;document.querySelector('#result').textContent='Confirmed'">Confirm</button>
<p id="result">Pending</p><p id="bottom">Bottom</p><script>window.clicks=0</script>"""


@pytest.fixture
def local_page():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest_asyncio.fixture
async def browser():
    session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True, enable_default_extensions=False, keep_alive=False, user_data_dir=None
        )
    )
    await session.start()
    try:
        if directory := os.environ.get("STACK_VERIFY_OUTPUT"):
            version = await session.cdp_client.send.Browser.getVersion()
            (Path(directory) / "chromium.json").write_text(json.dumps(version))
        yield session
    finally:
        await asyncio.wait_for(session.kill(), timeout=20)


@pytest_asyncio.fixture
async def harness(browser, monkeypatch):
    """Use the real Harness executor on this test's isolated CDP, never the user's daemon."""
    loop = asyncio.get_running_loop()

    def cdp(method, session_id=None, **params):
        domain, name = method.split(".", 1)
        send = getattr(getattr(browser.cdp_client.send, domain), name)
        kwargs = {"params": params}
        if session_id is not None:
            kwargs["session_id"] = session_id
        return asyncio.run_coroutine_threadsafe(send(**kwargs), loop).result(timeout=20)

    monkeypatch.setattr(jev_browser, "ensure_daemon", lambda: None)
    monkeypatch.setattr(jev_browser, "cdp", cdp)
    return cdp


async def evaluate(browser, expression):
    session = await browser.get_or_create_cdp_session()
    result = await session.cdp_client.send.Runtime.evaluate(
        params={"expression": expression, "returnByValue": True, "awaitPromise": True}, session_id=session.session_id
    )
    assert not result.get("exceptionDetails"), result
    return result["result"].get("value")


class ScriptedProvider:
    """Serve both real HTTP adapters the same deterministic operation/target sequence."""

    def __init__(self, provider, engine):
        self.provider, self.engine, self.calls = provider, engine, 0
        self.steps = [
            ("input", "Name"),
            ("select", "Blue"),
            ("click", "Enabled"),
            ("click", "Option A"),
            ("click", "Confirm"),
            ("scroll_down", None),
            ("scroll_up", None),
            ("wait", None),
            ("DONE", None),
        ]

    def __call__(self, request):
        body = json.loads(request.content)
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"text":"Ada"}'}}]})
        op, label = self.steps[self.calls]
        self.calls += 1
        if self.engine == "jev":
            op = {"input": "TYPE_TEXT"}.get(op, op.upper())
        if self.provider == "openai":
            questions = {q["name"]: {c["value"]: c["description"] for c in q["choices"]} for q in body["questions"]}
        else:
            questions = {name: q["criteria"] for name, q in body["questions"].items()}
        assert op in questions["operation"], (op, questions)
        answers = {}
        for name, candidates in questions.items():
            choice = op if name == "operation" else next(iter(candidates))
            if name == op.lower() + "_target":
                choices = [key for key, desc in candidates.items() if label in desc]
                assert choices, (label, candidates)
                choice = choices[-1]
            answer = {
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {key: float(key == choice) for key in candidates},
            }
            if self.provider == "openai":
                answer = {
                    **answer,
                    "name": name,
                    "probabilities": [{"value": key, "probability": p} for key, p in answer["probabilities"].items()],
                }
            answers[name] = answer
        return httpx.Response(
            200,
            json={"model": "offline", "answers": (list(answers.values()) if self.provider == "openai" else answers)},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize("engine", ["browser-use", "jev"])
async def test_both_engines_all_operations(browser, harness, local_page, monkeypatch, tmp_path, provider, engine):
    url = local_page
    server = ScriptedProvider(provider, engine)
    transport = httpx.MockTransport(server)
    expected = {"name": "Ada", "color": "cyan", "checked": True, "radio": True, "clicks": 1, "result": "Confirmed"}
    expression = """({name:document.querySelector('#name').value,
        color:document.querySelector('#color').value,checked:document.querySelector('#check').checked,
        radio:document.querySelector('#radio').checked,clicks:window.clicks,
        result:document.querySelector('#result').textContent})"""
    if engine == "browser-use":
        await browser.navigate_to(url)
        client = (
            OpenAIDecisionClient(api_key="offline", transport=transport)
            if provider == "openai"
            else AzureMicrosoftDecisionClient("https://fixture.invalid", api_key="offline", transport=transport)
        )
        helper = AsyncMock(spec=BaseChatModel)
        helper.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text="Ada"), usage=None)

        async def verify(_):
            return await evaluate(browser, expression) == expected

        runner = DecisionAgent("Complete the fixture", client, browser=browser, text_llm=helper)
        original_act = runner.tools.act
        scroll_positions, waits = [], []

        async def traced_act(action, session):
            started = time.monotonic()
            result = await original_act(action, session)
            fields = action.model_dump(exclude_unset=True)
            if "wait" in fields:
                waits.append(time.monotonic() - started)
            if "scroll" in fields:
                scroll_positions.append(
                    await evaluate(
                        browser,
                        """new Promise(resolve =>
                    requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(scrollY))))""",
                    )
                )
            return result

        runner.tools.act = traced_act
        result = await runner.run(9, outcome_verifier=verify)
        assert result.status == "model_done" and result.verified is True
        assert [s.operation for s in result.steps] == [s[0] for s in server.steps]
        helper.ainvoke.assert_awaited_once()
        actual = await evaluate(browser, expression)
        assert scroll_positions[0] > 0 and scroll_positions[1] < scroll_positions[0]
        assert len(waits) == 1 and waits[0] >= 0.9
    else:
        monkeypatch.setenv("DECISION_PROVIDER", provider)
        monkeypatch.setenv("AZURE_ENDPOINT", "https://fixture.invalid")
        monkeypatch.setenv("AZURE_API_KEY", "offline")
        monkeypatch.setenv("OPENAI_API_KEY", "offline")
        monkeypatch.setenv("TEXT_MODEL_API_KEY", "offline")
        monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://helper.invalid/v1")
        with httpx.Client(transport=transport) as client:
            monkeypatch.setattr(model, "CLIENT", client)
            runner = await asyncio.to_thread(jev.Agent, url, "Complete the fixture", record_dir=tmp_path)
            try:
                original_act = runner.browser.act
                waits = []

                def traced_act(action, page, text=None):
                    started = time.monotonic()
                    result = original_act(action, page, text=text)
                    if action["kind"] == "wait":
                        waits.append(time.monotonic() - started)
                    return result

                monkeypatch.setattr(runner.browser, "act", traced_act)
                snapshots = await asyncio.to_thread(lambda: list(runner.run()))
                assert snapshots[-1]["status"] == "done"
                actual = await asyncio.to_thread(runner.browser.evaluate, expression)
                assert len(runner.state["history"]) == 8
                assert len(runner.state["text_calls"]) == 1
                assert len(runner.state["decisions"]) == 9
                assert len(list(tmp_path.glob("*.jpg"))) >= 2
                assert runner.snapshot()["page"]["screenshot"]
                assert snapshots[5]["page"]["scroll"]["y"] > snapshots[4]["page"]["scroll"]["y"]
                assert snapshots[6]["page"]["scroll"]["y"] < snapshots[5]["page"]["scroll"]["y"]
                assert len(waits) == 1 and waits[0] >= 0.09
            finally:
                await asyncio.to_thread(runner.close)
                assert runner.browser.target is None
    assert actual == expected
    assert server.calls == 9


@pytest.mark.asyncio
async def test_existing_harness_guard_suite(harness):
    path = Path(__file__).resolve().parents[1] / "scripts/check_guards.py"
    spec = importlib.util.spec_from_file_location("native_guard_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    await asyncio.to_thread(module.main)


@pytest.mark.asyncio
async def test_select_candidates_against_native_javascript(browser):
    """Every offered candidate must select its advertised position in the actual native JS."""
    import browser_use.browser.watchdogs.default_action_watchdog as watchdog

    tree = ast.parse(Path(watchdog.__file__).read_text(encoding="utf-8"))
    scripts = [
        item.value.value
        for item in ast.walk(tree)
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "selection_script" for target in item.targets)
        and isinstance(item.value, ast.Constant)
    ]
    assert len(scripts) == 1
    prototypes = [
        ("A", "a"),
        ("B", "b"),
        ("A", "b"),
        (" B ", "a"),
        ("\u0085A\u0085", None),
        ("\ufeffB\ufeff", None),
        ("A", ""),
        ("A", None),
    ]
    cases = []
    runner, _, _ = agent()
    for pair in itertools.product(prototypes, repeat=2):
        for disabled in range(4):
            children = []
            for index, (label, value) in enumerate(pair):
                child = option(index + 2, label, value if value is not None else label)
                if value is None:
                    child.attributes.pop("value", None)
                if disabled & (1 << index):
                    child.attributes["disabled"] = ""
                children.append(child)
            targets = runner._targets(state({1: node(tag="select", children=children)})).get("select", {})
            for key, target in targets.items():
                cases.append(
                    {"options": pair, "disabled": disabled, "target": target.text, "expected": int(key.split(":")[1])}
                )
    payload = json.dumps(cases)
    expression = """(() => {
      const select=document.createElement('select'); document.body.append(select);
      const failures=[]; const cases=CASES; const choose=SELECTOR;
      for (const c of cases) {
        select.replaceChildren();
        c.options.forEach(([label,value],i)=>{
          const o=new Option(label); if(value!==null)o.value=value;
          o.disabled=!!(c.disabled&(1<<i)); select.add(o);
        });
        const result=choose.call(select,c.target);
        if(!result.success || select.selectedIndex!==c.expected)
          failures.push({...c,actual:select.selectedIndex,result});
      }
      select.remove();return {checked:cases.length,failures};
    })()""".replace("CASES", payload).replace("SELECTOR", "(" + scripts[0] + ")")
    result = await evaluate(browser, expression)
    assert result["checked"] == len(cases) and len(cases) > 100
    assert result["failures"] == [], json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_harness_roles_and_snapshot_budgets(harness):
    owned = await asyncio.to_thread(jev_browser.Browser, "about:blank")
    try:
        roles = [
            "button",
            "link",
            "checkbox",
            "radio",
            "switch",
            "tab",
            "menuitem",
            "menuitemradio",
            "option",
            "gridcell",
            "combobox",
            "textbox",
            "searchbox",
            "spinbutton",
        ]
        html = "<style>div,input{display:block;width:220px;height:20px}</style>"
        html += "".join(f'<div role="{role}" aria-label="Role {role}">Value</div>' for role in roles)
        html += (
            '<div contenteditable="true" aria-label="Editable">Original</div>'
            '<input type="password" aria-label="Secret" value="hidden-secret">'
            '<input disabled aria-label="Disabled"><input aria-label="Read only" readonly>'
            '<input style="display:none" aria-label="Hidden">'
            '<input inert aria-label="Inert"><input aria-hidden="true" aria-label="ARIA hidden">'
        )
        await asyncio.to_thread(owned.evaluate, "document.body.innerHTML=" + json.dumps(html))
        observed = await asyncio.to_thread(owned.observe, False)
        actions = observed["actions"]
        for role in roles:
            kinds = {a["kind"] for a in actions if a["label"] == "Role " + role or a["label"] == "Open Role " + role}
            assert kinds == ({"fill", "click"} if role in {"textbox", "searchbox", "spinbutton"} else {"click"}), role
        assert any(a["kind"] == "fill" and a["label"] == "Editable" for a in actions)
        assert not any(a["label"] in {"Secret", "Disabled", "Hidden", "Inert", "ARIA hidden"} for a in actions)
        assert {a["kind"] for a in actions if a["label"] == "Read only"} == {"click"}
        assert "screenshot" not in observed
        await asyncio.to_thread(
            owned.evaluate,
            """document.body.innerHTML='<p>'+ 'x'.repeat(9000)+'</p>';
            for(let i=0;i<300;i++){const b=document.createElement('button');b.textContent='Button '+i;
              b.style.cssText='position:absolute;top:20px;left:20px';document.body.append(b)}""",
        )
        observed = await asyncio.to_thread(owned.observe, False)
        assert len(observed["text"]) <= 6000
        assert observed["omitted_actions"] == 50
        assert len([a for a in observed["actions"] if "node" in a]) == 250
    finally:
        await asyncio.to_thread(owned.close)


@pytest.mark.asyncio
async def test_unicode_select_target_in_full_native_path(browser, local_page):
    await browser.navigate_to(local_page)
    html = (
        '<select id="unicode" aria-label="Unicode"><option>\ufeffB\ufeff</option><option value="b">B</option></select>'
    )
    await evaluate(browser, "document.body.innerHTML=" + json.dumps(html))
    observed = await browser.get_browser_state_summary(cached=False, include_screenshot=False)
    runner = DecisionAgent("Select the first B option", MockDecision(), browser=browser)
    candidates = runner._targets(observed).get("select", {})
    assert not any(key.endswith(":1") for key in candidates), "Ambiguous second option must not authorize input"
    chosen = next(value for key, value in candidates.items() if key.endswith(":0"))
    action = runner._action_model.model_validate({"select_dropdown": {"index": chosen.index, "text": chosen.text}})
    result = await runner.tools.act(action, browser)
    assert not result.error
    assert await evaluate(browser, "document.querySelector('#unicode').selectedIndex") == 0


class MockDecision:
    async def predict(self, _request):
        raise AssertionError("The native mapping check does not call a model")
