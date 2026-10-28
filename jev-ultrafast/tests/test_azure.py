"""Azure transport and response contracts. All HTTP requests stay in-process."""

import json

import httpx
import pytest

from jev_ultrafast import model


@pytest.fixture
def azure(monkeypatch):
    for name in ("AZURE_API_KEY", "AZURE_ENTRA_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_ENDPOINT", "https://resource.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "my-decision")
    monkeypatch.setenv("AZURE_API_KEY", "azure-test-key")
    requests = []
    reply = {"status": 200, "body": {
        "model": "microsoft-decision-1",
        "answers": {
            "operation": {"type": "choice", "choice": "CLICK", "confidence": 0.8,
                          "probabilities": {"CLICK": 0.9, "DONE": 0.05, "BLOCKED": 0.05}},
            "click_target": {"type": "choice", "choice": "2", "confidence": 0.7,
                             "probabilities": {"1": 0.1, "2": 0.9}},
        },
    }}

    def handle(request):
        requests.append(request)
        return httpx.Response(reply["status"], content=json.dumps(reply["body"]))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(model, "CLIENT", client)
        yield requests, reply


def page():
    return {"url": "https://example.test", "title": "Choose", "text": "選んでください", "actions": [
        {"id": "first", "kind": "click", "node": 1, "label": "戻る", "role": "button", "value": ""},
        {"id": "next", "kind": "click", "node": 2, "label": "次へ", "role": "button", "value": ""},
    ]}


def test_azure_choice_uses_deployment_and_preserves_observed_target(azure):
    requests, _ = azure
    result = model.choose(page(), "次へ進む", [])
    assert result["choice"] == "next"
    assert result["confidence"] == 0.8 and result["target_confidence"] == 0.7
    assert result["probabilities"] == {"first": 0.1, "next": 0.9}
    assert result["model"] == "microsoft-decision-1"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://resource.cognitiveservices.azure.com/providers/microsoft/v1/systemone"
    assert request.headers["api-key"] == "azure-test-key"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "my-decision"
    assert set(body["questions"]) == {"operation", "click_target"}
    assert json.loads(body["questions"]["operation"]["instructions"])["goal"] == "次へ進む"
    assert json.loads(body["questions"]["click_target"]["criteria"]["2"])["element"] == "[2] 次へ"
    assert all(isinstance(value, str) for q in body["questions"].values() for value in q["criteria"].values())
    assert "azure-test-key" not in json.dumps(result)


def test_entra_token_authentication(azure, monkeypatch):
    requests, _ = azure
    monkeypatch.delenv("AZURE_API_KEY")
    monkeypatch.setenv("AZURE_ENTRA_TOKEN", "entra-test-token")
    model.choose(page(), "Next", [])
    assert requests[0].headers["authorization"] == "Bearer entra-test-token"
    assert "api-key" not in requests[0].headers


def test_only_available_target_is_resolved_without_a_target_question(azure):
    requests, reply = azure
    p = page()
    p["actions"] = p["actions"][:1]
    del reply["body"]["answers"]["click_target"]
    result = model.choose(p, "Go back", [])
    assert result["choice"] == "first" and result["target_probabilities"] == {"1": 1.0}
    assert set(json.loads(requests[0].content)["questions"]) == {"operation"}
    assert "click_target" not in result["raw_answers"]


def test_all_ambiguous_heads_share_one_request(azure):
    requests, reply = azure
    p = page()
    p["actions"] += [
        {"id": "city", "kind": "fill", "node": 3, "label": "City", "role": "textbox", "value": ""},
        {"id": "date", "kind": "fill", "node": 4, "label": "Date", "role": "textbox", "value": ""},
    ]
    reply["body"]["answers"]["operation"]["probabilities"] = {
        "CLICK": 0.8, "TYPE_TEXT": 0.1, "DONE": 0.05, "BLOCKED": 0.05,
    }
    result = model.choose(p, "Next", [])
    assert result["choice"] == "next" and len(requests) == 1
    assert set(json.loads(requests[0].content)["questions"]) == {"operation", "click_target", "type_text_target"}


@pytest.mark.parametrize("endpoint", ["http://resource.test", "https://user:password@resource.test",
                                      "https://resource.test/path", "https://resource.test?key=secret"])
def test_invalid_resource_endpoint_stops_before_sending_credentials(azure, monkeypatch, endpoint):
    requests, _ = azure
    monkeypatch.setenv("AZURE_ENDPOINT", endpoint)
    with pytest.raises(ValueError, match="AZURE_ENDPOINT"):
        model.choose(page(), "Next", [])
    assert not requests


def test_missing_azure_credentials_cannot_fall_back_to_jev(azure, monkeypatch):
    requests, _ = azure
    monkeypatch.delenv("AZURE_API_KEY")
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused")
    with pytest.raises(ValueError, match="AZURE_API_KEY.*AZURE_ENTRA_TOKEN"):
        model.choose(page(), "Next", [])
    assert not requests


@pytest.mark.parametrize("payload", [None, [], {}, {"answers": []}, {"answers": {}, "model": []}])
def test_invalid_azure_response_is_rejected(azure, payload):
    _, reply = azure
    reply["body"] = payload
    with pytest.raises(ValueError, match="Invalid decision response"):
        model.choose(page(), "Next", [])


def test_non_choice_answer_is_rejected(azure):
    _, reply = azure
    reply["body"]["answers"]["operation"]["type"] = "noul"
    with pytest.raises(ValueError, match="Invalid decision response"):
        model.choose(page(), "Next", [])


def test_malformed_json_does_not_disclose_the_provider_body(azure, monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="private provider diagnostic")
    )) as client:
        monkeypatch.setattr(model, "CLIENT", client)
        with pytest.raises(ValueError, match="^Invalid model response; no action executed.$"):
            model.choose(page(), "Next", [])


def test_rate_limit_stops_without_repeating_an_old_observation(azure):
    requests, reply = azure
    reply["status"] = 429
    with pytest.raises(RuntimeError, match="rate limit"):
        model.choose(page(), "Next", [])
    assert len(requests) == 1


def test_text_helper_keeps_bearer_authentication(azure, monkeypatch):
    requests, reply = azure
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "text-test-key")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://text.example/v1")
    reply["body"] = {"choices": [{"message": {"content": '{"text":"東京"}'}}]}
    assert model.field_text({"goal": "Search Tokyo"})[0] == "東京"
    assert requests[0].headers["authorization"] == "Bearer text-test-key"
    assert "api-key" not in requests[0].headers
