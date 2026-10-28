"""Bounded exhaustive checks on real Python functions, not a duplicate state model."""

import copy
import itertools
import time
from unittest.mock import AsyncMock, Mock

import pytest
from browser_use.decision import ChoiceAnswer, FieldText
from browser_use.llm.base import BaseChatModel
from browser_use.llm.views import ChatInvokeCompletion
from test_microsoft_decision import agent, node, state

from jev_ultrafast import agent as jev
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage


@pytest.mark.parametrize("engine", ["browser-use", "jev"])
def test_exhaustive_choice_domain(engine):
    """Enumerate every tuple in a finite invalid numeric/identity domain."""
    validate = ChoiceAnswer.validate_for if engine == "browser-use" else model.validate_choice
    domain = [-1, 0, 0.5, 1, 2, True, None, "0", float("nan"), float("inf")]
    checked = 0
    for selected, left, right, confidence in itertools.product(["a", "b", "foreign"], domain, domain, domain):
        answer = {
            "type": "choice",
            "choice": selected,
            "confidence": confidence,
            "probabilities": {"a": left, "b": right},
        }
        numeric = all(type(n) in (int, float) and 0 <= n <= 1 for n in (left, right, confidence))
        allowed = numeric and left + right == 1 and selected in {"a", "b"}
        if allowed:
            allowed = (left if selected == "a" else right) == max(left, right)
        if allowed:
            validate(answer, {"a": "A", "b": "B"})
        else:
            with pytest.raises(ValueError):
                validate(answer, {"a": "A", "b": "B"})
        checked += 1
    assert checked == 3000


def make_jev(monkeypatch):
    current = {
        "url": "https://fixture.invalid",
        "title": "Fixture",
        "text": "Fixture",
        "fingerprint": "page",
        "actions": [{"id": "click", "kind": "click", "node": 1, "label": "Confirm"}],
    }
    browser = Mock()
    browser.fresh.return_value = True
    browser.observe.return_value = current
    runner = jev.Agent.__new__(jev.Agent)
    runner.screenshots = False
    runner.pending_text = None
    runner.state = {
        "browser": browser,
        "page": current,
        "decision": None,
        "goal": "Confirm",
        "history": [],
        "decisions": [],
        "status": "ready",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    counter = [0]
    attempts = {}

    def choose(*_):
        counter[0] += 1
        return {
            "choice": "click",
            "operation": "CLICK",
            "target": "1",
            "confidence": 1,
            "probabilities": {"click": 1},
            "latency_ms": 0,
            "usage": {},
        }

    def act(*_args, **_kwargs):
        attempts[counter[0]] = attempts.get(counter[0], 0) + 1
        if not browser.fresh.return_value:
            raise StalePage("Changed before input")

    browser.act.side_effect = act
    monkeypatch.setattr(jev, "choose", choose)
    return runner, browser, attempts


def test_exhaustive_single_consumption(monkeypatch):
    """All 780 command strings of length 1..4: one prediction permits at most one attempt."""
    commands = ("predict", "act", "wrong_act", "stale", "fresh")
    checked = 0
    for length in range(1, 5):
        for sequence in itertools.product(commands, repeat=length):
            runner, browser, attempts = make_jev(monkeypatch)
            for command in sequence:
                if command in {"stale", "fresh"}:
                    browser.fresh.return_value = command == "fresh"
                    continue
                before = sum(attempts.values())
                try:
                    runner.command(
                        "act" if command == "wrong_act" else command,
                        {"fingerprint": "wrong" if command == "wrong_act" else "page"},
                    )
                except (ValueError, StalePage):
                    pass
                if command == "wrong_act":
                    assert sum(attempts.values()) == before, sequence
                assert all(count <= 1 for count in attempts.values()), sequence
            checked += 1
    assert checked == 780


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["click", "input", "select", "scroll_up", "scroll_down", "wait", "DONE", "BLOCKED"]
)
@pytest.mark.parametrize(
    "change", ["url", "title", "node", "value", "frame", "session", "visible", "scroll", "tabs", "error"]
)
async def test_browser_freshness_interleavings(operation, change):
    from browser_use.browser.views import TabInfo
    from test_microsoft_decision import option

    nodes = {1: node(tag="input" if operation == "input" else "button")}
    if operation == "select":
        nodes = {1: node(tag="select", children=[option(2, "Blue", "blue")])}
    helper = AsyncMock(spec=BaseChatModel)
    helper.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text="Ada"), usage=None)
    runner, browser, predictor = agent(operation, nodes, helper)
    initial = state(nodes)
    current = copy.deepcopy(initial)
    if change in {"url", "title"}:
        setattr(current, change, "changed")
    elif change in {"node", "value", "frame", "session", "visible"}:
        changed = current.dom_state.selector_map[1]
        if change == "node":
            changed.backend_node_id += 1
        elif change == "value":
            changed.attributes["value"] = "changed"
        elif change == "frame":
            changed.frame_id = "changed"
        elif change == "session":
            changed.session_id = "changed"
        else:
            changed.is_visible = False
    elif change == "scroll":
        current.pixels_above = 100
    elif change == "tabs":
        current.tabs = [TabInfo(target_id="changed", url="https://fixture.invalid", title="New")]
    else:
        current.state_error = "lost document"
    browser.get_browser_state_summary.side_effect = [initial, current]
    result = await runner.run(1)
    assert predictor.requests
    assert result.status == "stale_state"
    runner.tools.act.assert_not_awaited()


@pytest.mark.parametrize("callback", ["success", "stale", "error"])
def test_jev_replay_after_callback_failure(monkeypatch, callback):
    runner, browser, attempts = make_jev(monkeypatch)
    runner.command("predict")
    if callback != "success":
        browser.act.side_effect = StalePage("before input") if callback == "stale" else RuntimeError("partial input")
    try:
        runner.command("act", {"fingerprint": "page"})
    except (StalePage, RuntimeError):
        pass
    first_count = browser.act.call_count
    with pytest.raises(ValueError, match="Observe and choose"):
        runner.command("act", {"fingerprint": "page"})
    assert browser.act.call_count == first_count == 1


@pytest.mark.parametrize("limit", [1, 2, 60])
def test_jev_action_and_prediction_budgets(monkeypatch, limit):
    monkeypatch.setattr(jev, "MAX_STEPS", limit)
    runner, browser, _ = make_jev(monkeypatch)
    runner.state["history"] = [{}] * limit
    runner.command("predict")
    with pytest.raises(ValueError, match="action demo budget"):
        runner.command("act", {"fingerprint": "page"})
    browser.act.assert_not_called()
    runner.state["status"] = "ready"
    runner.state["decisions"] = [{}] * (limit * 2)
    with pytest.raises(ValueError, match="model-call budget"):
        runner.command("predict")


@pytest.mark.parametrize("terminal", ["DONE", "BLOCKED"])
@pytest.mark.parametrize("fresh", [True, False])
def test_jev_terminal_requires_fresh_state(monkeypatch, terminal, fresh):
    runner, browser, _ = make_jev(monkeypatch)
    runner.command("predict")
    runner.state["decision"]["choice"] = terminal
    browser.fresh.return_value = fresh
    if fresh:
        assert runner.command("act", {"fingerprint": "page"})["status"] == terminal.lower()
        with pytest.raises(ValueError, match="stopped"):
            runner.command("predict")
    else:
        with pytest.raises(StalePage):
            runner.command("act", {"fingerprint": "page"})
    browser.act.assert_not_called()


def test_jev_no_progress_stops_after_three_actions(monkeypatch):
    runner, _, _ = make_jev(monkeypatch)
    for _ in range(3):
        runner.command("tick")
    assert runner.state["status"] == "blocked"
    assert len(runner.state["history"]) == 3
    with pytest.raises(ValueError, match="stopped"):
        runner.command("predict")


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_browser_step_budget(limit):
    runner, _, predictor = agent("click")
    result = await runner.run(limit)
    assert result.status == "max_steps"
    assert len(predictor.requests) == len(result.steps) == runner.tools.act.await_count == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.2, None])
async def test_browser_invalid_budget_never_starts(limit):
    runner, browser, predictor = agent()
    with pytest.raises(ValueError, match="positive integer"):
        await runner.run(limit)
    browser.start.assert_not_awaited()
    assert predictor.requests == []
