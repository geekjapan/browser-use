"""OpenAI Decisions wire contracts; no network or browser required."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from jev_ultrafast import demo, model


def page():
    return {"url": "https://example.test", "title": "Choose", "text": "選ぶ", "actions": [
        {"id": "first", "kind": "click", "node": 1, "label": "戻る", "role": "button"},
        {"id": "next", "kind": "click", "node": 2, "label": "次へ", "role": "button"},
    ]}


def answer(name, choice, probabilities):
    return {"name": name, "type": "choice", "choice": choice, "confidence": 0.8,
            "probabilities": [{"value": k, "probability": v} for k, v in probabilities.items()]}


@pytest.fixture
def openai(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.delenv("OPENAI_DECISION_MODEL", raising=False)
    monkeypatch.setenv("AZURE_ENDPOINT", "invalid-unused-endpoint")
    monkeypatch.setenv("AZURE_API_KEY", "azure-unused-key")
    monkeypatch.setenv("AZURE_ENTRA_TOKEN", "azure-unused-token")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "text-unused-key")
    requests = []
    reply = {"status": 200, "body": {
        "model": "actual-model", "usage": {"input_tokens": 25, "output_tokens": 5},
        "answers": [answer("operation", "CLICK", {"CLICK": 0.9, "DONE": 0.05, "BLOCKED": 0.05}),
                    answer("click_target", "2", {"1": 0.1, "2": 0.9})],
    }}

    def handle(request):
        requests.append(request)
        return httpx.Response(reply["status"], content=json.dumps(reply["body"]))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(model, "CLIENT", client)
        yield requests, reply


def test_wire_contract_and_evidence(openai):
    requests, reply = openai
    raw = deepcopy(reply["body"])
    result = model.choose(page(), "次へ", [])
    assert result["choice"] == "next"
    assert result["model"] == "actual-model"
    assert result["usage"] == raw["usage"]
    assert result["raw_answers"] == raw["answers"]
    assert reply["body"] == raw
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer openai-test-key"
    assert "api-key" not in request.headers
    body = json.loads(request.content)
    assert result["request"] == body
    assert set(body) == {"model", "input", "questions"}
    assert body["model"] == "gpt-6-luna"
    assert json.loads(body["input"])["page"]["text"] == "選ぶ"
    assert [q["name"] for q in body["questions"]] == ["operation", "click_target"]
    for question in body["questions"]:
        assert set(question) == {"name", "type", "instructions", "choices"}
        assert question["type"] == "choice"
        assert isinstance(question["instructions"], str)
        assert all(set(c) == {"value", "description"} and all(isinstance(v, str) for v in c.values())
                   for c in question["choices"])
    assert json.loads(body["questions"][0]["instructions"])["goal"] == "次へ"
    assert "key" not in json.dumps(result)


def test_custom_model_and_singleton(openai, monkeypatch):
    requests, reply = openai
    monkeypatch.setenv("OPENAI_DECISION_MODEL", "custom-model")
    state = page()
    state["actions"] = state["actions"][:1]
    reply["body"]["answers"] = reply["body"]["answers"][:1]
    result = model.choose(state, "Back", [])
    assert result["choice"] == "first"
    assert result["target_probabilities"] == {"1": 1.0}
    body = json.loads(requests[0].content)
    assert body["model"] == "custom-model"
    assert len(body["questions"]) == 1


@pytest.mark.parametrize("selected", [False, True])
def test_refusal_is_validated_only_for_selected_target(openai, selected):
    _, reply = openai
    state = page()
    state["actions"] += [
        {"id": "city", "kind": "fill", "node": 3, "label": "City"},
        {"id": "date", "kind": "fill", "node": 4, "label": "Date"},
    ]
    reply["body"]["answers"][0] = answer("operation", "TYPE_TEXT" if selected else "CLICK",
                                                   {"CLICK": 0.0 if selected else 0.9,
                                                    "TYPE_TEXT": 0.9 if selected else 0.0,
                                                    "DONE": 0.05, "BLOCKED": 0.05})
    reply["body"]["answers"].append({"name": "type_text_target", "type": "refusal"})
    if selected:
        with pytest.raises(ValueError, match="Invalid decision response"):
            model.choose(state, "Next", [])
    else:
        assert model.choose(state, "Next", [])["choice"] == "next"


@pytest.mark.parametrize("mutation", ["duplicate_head", "unknown_head", "bad_head", "duplicate_probability",
                                      "boolean_id", "boolean_choice", "boolean_probability", "nan", "infinity",
                                      "negative", "missing_probability", "unknown_probability", "non_max",
                                      "bad_confidence", "refusal", "missing_operation", "not_list", "not_object"])
def test_invalid_answers_stop_before_execution(openai, mutation):
    _, reply = openai
    answers = reply["body"]["answers"]
    target = answers[1]
    if mutation == "duplicate_head":
        answers.append(deepcopy(target))
    elif mutation == "unknown_head":
        target["name"] = "invented"
    elif mutation == "bad_head":
        target["name"] = []
    elif mutation == "duplicate_probability":
        target["probabilities"].append(deepcopy(target["probabilities"][0]))
    elif mutation == "boolean_id":
        target["probabilities"][0]["value"] = True
    elif mutation == "boolean_choice":
        target["choice"] = True
    elif mutation in {"boolean_probability", "nan", "infinity", "negative"}:
        target["probabilities"][0]["probability"] = {
            "boolean_probability": True, "nan": float("nan"), "infinity": float("inf"), "negative": -0.1,
        }[mutation]
    elif mutation == "missing_probability":
        target["probabilities"].pop()
    elif mutation == "unknown_probability":
        target["probabilities"][0]["value"] = "3"
    elif mutation == "non_max":
        target["choice"] = "1"
    elif mutation == "bad_confidence":
        target["confidence"] = True
    elif mutation == "refusal":
        answers[0] = {"name": "operation", "type": "refusal"}
    elif mutation == "missing_operation":
        answers.pop(0)
    elif mutation == "not_list":
        reply["body"]["answers"] = {}
    else:
        answers[0] = []
    with pytest.raises(ValueError, match="Invalid decision response"):
        model.choose(page(), "Next", [])


@pytest.mark.parametrize("status", [401, 403, 429])
def test_provider_errors_are_sanitized_and_not_retried(openai, status):
    requests, reply = openai
    reply.update(status=status, body={"error": "private diagnostic"})
    with pytest.raises(RuntimeError) as error:
        model.choose(page(), "Next", [])
    assert "private" not in str(error.value)
    assert len(requests) == 1


def test_missing_key_never_falls_back(openai, monkeypatch):
    requests, _ = openai
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model.choose(page(), "Next", [])
    assert not requests


@pytest.mark.parametrize("provider", ["other", "", "OPENAI"])
def test_invalid_provider_never_requests(openai, monkeypatch, provider):
    requests, _ = openai
    monkeypatch.setenv("DECISION_PROVIDER", provider)
    with pytest.raises(ValueError, match="DECISION_PROVIDER"):
        model.choose(page(), "Next", [])
    assert not requests


@pytest.mark.parametrize("provider,env,default", [("azure", "AZURE_DEPLOYMENT_NAME", "decision-1"),
                                                  ("openai", "OPENAI_DECISION_MODEL", "gpt-6-luna")])
def test_demo_metadata(monkeypatch, provider, env, default):
    monkeypatch.setattr(demo, "AGENT", None)
    monkeypatch.setenv("DECISION_PROVIDER", provider)
    monkeypatch.delenv(env, raising=False)
    assert demo.response_state()["decision_provider"] == provider
    assert demo.response_state()["decision_model"] == default
    monkeypatch.setenv(env, "configured-model")
    assert demo.response_state()["decision_model"] == "configured-model"
    index = (Path(demo.ROOT) / "static" / "index.html").read_text()
    assert "Microsoft" not in index


def test_openai_selection_does_not_change_text_credentials(openai, monkeypatch):
    requests, reply = openai
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://text.example/v1")
    reply["body"] = {"choices": [{"message": {"content": '{"text":"東京"}'}}]}
    assert model.field_text({"goal": "Tokyo"})[0] == "東京"
    assert str(requests[0].url) == "https://text.example/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer text-unused-key"
    assert "api-key" not in requests[0].headers


@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize("operation,expected", [
    ("CLICK", "c2"), ("TYPE_TEXT", "t2"), ("SELECT", "s2"),
    ("WAIT", "wait"), ("SCROLL_UP", "scroll_up"), ("SCROLL_DOWN", "scroll_down"),
    ("DONE", "DONE"), ("BLOCKED", "BLOCKED"),
])
def test_provider_operation_parity(monkeypatch, provider, operation, expected):
    monkeypatch.setenv("DECISION_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://resource.test")
    monkeypatch.setenv("AZURE_API_KEY", "offline-key")
    state = page()
    state["actions"] = [
        {"id": "c1", "node": 1, "kind": "click", "label": "Back"},
        {"id": "c2", "node": 2, "kind": "click", "label": "Next"},
        {"id": "t1", "node": 3, "kind": "fill", "label": "First name"},
        {"id": "t2", "node": 4, "kind": "fill", "label": "Last name"},
        {"id": "s1", "node": 5, "kind": "select", "label": "Red", "value": "red"},
        {"id": "s2", "node": 5, "kind": "select", "label": "Blue", "value": "blue"},
        *({"id": kind, "kind": kind, "label": kind} for kind in ("wait", "scroll_up", "scroll_down")),
    ]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        questions = ({q["name"]: [c["value"] for c in q["choices"]] for q in body["questions"]}
                     if provider == "openai"
                     else {name: list(q["criteria"]) for name, q in body["questions"].items()})
        answers = []
        for name, candidates in questions.items():
            selected = operation if name == "operation" else candidates[-1]
            answers.append(answer(name, selected, {value: float(value == selected) for value in candidates}))
        payload = (answers if provider == "openai" else {
            a["name"]: {**a, "probabilities": {p["value"]: p["probability"] for p in a["probabilities"]}}
            for a in answers
        })
        return httpx.Response(200, json={"model": "test", "answers": payload})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(model, "CLIENT", client)
        result = model.choose(state, "Complete the task", [])
    assert result["operation"] == operation
    assert result["choice"] == expected
    assert len(requests) == 1
