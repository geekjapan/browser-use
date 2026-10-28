"""Azure or OpenAI makes choices; a separate text model writes field values."""

import json
import math
import os
import time
from urllib.parse import urlsplit

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, headers, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers=headers)
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code == 429:
            raise RuntimeError("Model rate limit reached; wait for quota to reset, then observe and choose again.")
        if response.status_code in {529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        try:
            return response.json()
        except ValueError:
            raise ValueError("Invalid model response; no action executed.") from None
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError()
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict):
            raise ValueError()
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid decision response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def decision_settings():
    provider = os.environ.get("DECISION_PROVIDER", "azure").strip()
    if provider not in {"azure", "openai"}:
        raise ValueError("DECISION_PROVIDER must be azure or openai; no decision requested.")
    model = (os.environ.get("OPENAI_DECISION_MODEL", "gpt-6-luna") if provider == "openai"
             else os.environ.get("AZURE_DEPLOYMENT_NAME", "decision-1"))
    return provider, model


def openai_answers(result, questions):
    """Adapt named answers without changing the raw provider evidence."""
    answers = result.get("answers")
    if not isinstance(answers, list):
        raise ValueError("Invalid decision response; no action executed.")
    normalized = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid decision response; no action executed.")
        name = answer.get("name")
        if not isinstance(name, str) or name not in questions or name in normalized:
            raise ValueError("Invalid decision response; no action executed.")
        converted = dict(answer)
        if answer.get("type") == "choice":
            probabilities = answer.get("probabilities")
            if not isinstance(answer.get("choice"), str) or not isinstance(probabilities, list):
                raise ValueError("Invalid decision response; no action executed.")
            mapped = {}
            for probability in probabilities:
                if (not isinstance(probability, dict) or not isinstance(probability.get("value"), str)
                        or probability["value"] in mapped or "probability" not in probability):
                    raise ValueError("Invalid decision response; no action executed.")
                mapped[probability["value"]] = probability["probability"]
            converted["probabilities"] = mapped
        normalized[name] = converted
    return normalized


def decision_connection():
    provider, _ = decision_settings()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("Set OPENAI_API_KEY; no decision requested.")
        return "https://api.openai.com/v1/decisions", {"Authorization": f"Bearer {key}"}

    endpoint = os.environ.get("AZURE_ENDPOINT", "").strip()
    resource = urlsplit(endpoint)
    if (resource.scheme != "https" or not resource.hostname or resource.username or resource.password
            or resource.path not in {"", "/"} or resource.query or resource.fragment):
        raise ValueError("AZURE_ENDPOINT must be an HTTPS resource URL without credentials, a path, or a query.")
    key = os.environ.get("AZURE_API_KEY", "").strip()
    token = os.environ.get("AZURE_ENTRA_TOKEN", "").strip()
    if not key and not token:
        raise ValueError("Set AZURE_API_KEY or AZURE_ENTRA_TOKEN; no decision requested.")
    headers = {"api-key": key} if key else {"Authorization": f"Bearer {token}"}
    return endpoint.rstrip("/") + "/providers/microsoft/v1/systemone", headers


def choose(state, goal, history):
    provider, decision_model = decision_settings()
    url, headers = decision_connection()
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations,
                      "instructions": json.dumps({"goal": goal, "rules": NEXT_ACTION}, ensure_ascii=False)}
    }
    for operation, candidates in targets.items():
        # A sole compatible target needs no prediction.
        if len(candidates) == 1:
            continue
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: json.dumps({
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }, ensure_ascii=False)
                for index, a in candidates.items()
            },
            "instructions": json.dumps(
                {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]}, ensure_ascii=False
            ),
        }
    body = {
        "model": decision_model,
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    if provider == "openai":
        body = {
            "model": decision_model,
            "input": json.dumps(body["state"], ensure_ascii=False),
            "questions": [
                {"name": name, "type": "choice", "instructions": question["instructions"],
                 "choices": [{"value": value, "description": description}
                             for value, description in question["criteria"].items()]}
                for name, question in questions.items()
            ],
        }
    started = time.perf_counter()
    result = post_json(url, headers, body)
    if (not isinstance(result, dict)
            or not isinstance(result.get("model"), str) or not result["model"]):
        raise ValueError("Invalid decision response; no action executed.")
    answers = openai_answers(result, questions) if provider == "openai" else result.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Invalid decision response; no action executed.")
    operation_answer = validate_choice(answers.get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        candidates = targets[operation]
        if len(candidates) == 1:
            only = next(iter(candidates))
            target_answer = {"choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
        else:
            # Unused target heads cannot cause an action. Validate only the selected operation's head.
            target_answer = validate_choice(answers.get(operation.lower() + "_target", {}), candidates)
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {} if urlsplit(base).hostname == "api.openai.com" else {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
