"""Loopback HTTP authorization, serialization, lifecycle and provider failure contracts."""

import http.client
import itertools
import json
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

import httpx
import pytest

from jev_ultrafast import demo, model


@pytest.fixture
def server(monkeypatch):
    instance = ThreadingHTTPServer(("127.0.0.1", 0), demo.Handler)
    monkeypatch.setattr(demo, "PORT", instance.server_port)
    monkeypatch.setattr(demo, "ORIGIN", f"http://127.0.0.1:{instance.server_port}")
    monkeypatch.setattr(demo, "TOKEN", "offline-token")
    monkeypatch.setattr(demo, "AGENT", None)
    monkeypatch.setattr(demo, "LOCK", threading.Lock())
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def request(server, method="POST", path="/api/predict", body=b"{}", **overrides):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    headers = {
        "Host": f"127.0.0.1:{server.server_port}",
        "X-Demo-Token": "offline-token",
        "Origin": f"http://127.0.0.1:{server.server_port}",
    }
    headers.update(overrides)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_all_host_origin_token_combinations(server, monkeypatch):
    command = Mock(return_value={"status": "ready"})
    monkeypatch.setattr(demo, "command", command)
    for host, origin, token in itertools.product([False, True], repeat=3):
        status, _, _ = request(
            server,
            **{
                "Host": f"127.0.0.1:{server.server_port}" if host else "attacker.invalid",
                "Origin": demo.ORIGIN if origin else "https://attacker.invalid",
                "X-Demo-Token": "offline-token" if token else "wrong",
            },
        )
        assert status == (200 if all((host, origin, token)) else 403)
    assert command.call_count == 1


def test_busy_lock_is_rejected_without_running_command(server, monkeypatch):
    command = Mock(return_value={})
    monkeypatch.setattr(demo, "command", command)
    with demo.LOCK:
        assert request(server)[0] == 409
    assert request(server)[0] == 200
    command.assert_called_once()


@pytest.mark.parametrize("body", [b"", b"{", b"x" * 8192])
def test_bad_body_releases_lock(server, monkeypatch, body):
    command = Mock(return_value={})
    monkeypatch.setattr(demo, "command", command)
    assert request(server, body=body)[0] == 400
    command.assert_not_called()
    assert request(server)[0] == 200


def test_internal_error_redacted_and_lock_released(server, monkeypatch):
    command = Mock(side_effect=Exception("private-provider-content"))
    monkeypatch.setattr(demo, "command", command)
    status, _, body = request(server)
    assert status == 500 and b"private-provider-content" not in body
    command.side_effect = None
    command.return_value = {}
    assert request(server)[0] == 200


@pytest.mark.parametrize(
    "path,mime",
    [("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css"), ("/fixture.html", "text/html")],
)
def test_static_response_headers_and_token(server, path, mime):
    status, headers, body = request(server, "GET", path)
    assert status == 200 and headers["Content-Type"].startswith(mime)
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    if path == "/":
        assert b"offline-token" in body and b"__TOKEN__" not in body
    assert request(server, "GET", path, Host="attacker.invalid")[0] == 403
    assert request(server, "GET", "/../pyproject.toml")[0] == 404


def test_reset_validation_and_old_browser_lifecycle(monkeypatch):
    previous = Mock()
    current = Mock()
    current.state = {}
    current.snapshot.return_value = {"status": "ready"}
    factory = Mock(return_value=current)
    monkeypatch.setattr(demo, "AGENT", previous)
    monkeypatch.setattr(demo, "Agent", factory)
    for body in ({"scenario": "unknown", "goal": "Goal"}, {"goal": ""}, {"goal": "x" * 2001}):
        with pytest.raises(ValueError):
            demo.command("reset", body)
    previous.close.assert_not_called()
    demo.command("reset", {"scenario": "research", "goal": "Find article"})
    previous.close.assert_called_once()
    assert factory.call_args.args == (demo.ORIGIN + "/fixture.html?scenario=research", "Find article")
    demo.close_browser()
    demo.close_browser()
    current.close.assert_called_once()


@pytest.mark.parametrize("status", [503, 529])
@pytest.mark.parametrize("recover", [False, True])
def test_retry_policy_exact_attempts(monkeypatch, status, recover):
    responses = [status, status, 200 if recover else status]
    calls = []
    sleeps = []

    def respond(request):
        calls.append(request)
        return httpx.Response(responses[len(calls) - 1], json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(model, "CLIENT", client)
        monkeypatch.setattr(model.time, "sleep", sleeps.append)
        if recover:
            assert model.post_json("https://fixture.invalid", {}, {}) == {"ok": True}
        else:
            with pytest.raises(RuntimeError):
                model.post_json("https://fixture.invalid", {}, {})
    assert len(calls) == 3 and sleeps == [0.5, 1.0]


@pytest.mark.parametrize("value", ["", " ", "x" * 2001, None, 3, {}, []])
def test_helper_output_boundary(monkeypatch, value):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "offline")
    monkeypatch.setattr(
        model, "post_json", lambda *_: {"choices": [{"message": {"content": json.dumps({"text": value})}}]}
    )
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({})


def test_helper_maximum_length(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "offline")
    monkeypatch.setattr(
        model, "post_json", lambda *_: {"choices": [{"message": {"content": json.dumps({"text": "x" * 2000})}}]}
    )
    assert len(model.field_text({})[0]) == 2000


def test_openai_text_helper_omits_unsupported_reasoning(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "offline")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TEXT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("TEXT_MODEL_REASONING", "none")
    request = Mock(return_value={"choices": [{"message": {"content": '{"text":"Ada"}'}}]})
    monkeypatch.setattr(model, "post_json", request)
    assert model.field_text({})[0] == "Ada"
    body = request.call_args.args[2]
    assert "reasoning" not in body and "reasoning_effort" not in body
