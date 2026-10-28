"""Test-only, in-memory mutations of real production functions; never edits either checkout."""

import inspect
import os
import sys
import textwrap
from pathlib import Path


def pytest_configure(config):
    sys.path.insert(0, os.environ["BROWSER_USE_REPO"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from browser_use import decision

    from jev_ultrafast import agent, model

    mutations = {
        "consume": (
            agent,
            agent.Agent,
            "command",
            'state["decision"] = None\n        selected = decision["choice"]',
            'selected = decision["choice"]',
        ),
        "freshness": (
            decision,
            decision.DecisionAgent,
            "run",
            "if self._fingerprint(current) != fingerprint:",
            "if False:",
        ),
        "browser_choice": (
            decision,
            decision.ChoiceAnswer,
            "validate_for",
            "or p[answer.choice] < max(p.values()) - 1e-6",
            "",
        ),
        "jev_choice": (
            model,
            model,
            "validate_choice",
            'and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6',
            "",
        ),
    }
    name = os.environ["STACK_MUTATION"]
    module, owner, attribute, before, after = mutations[name]
    source = textwrap.dedent(inspect.getsource(getattr(owner, attribute)))
    assert source.count(before) == 1, f"Mutation {name} no longer matches; inspect source drift"
    namespace = {}
    exec(compile(source.replace(before, after), f"<stack-mutation:{name}>", "exec"), module.__dict__, namespace)
    setattr(owner, attribute, namespace[attribute])
    print(f"MUTATION_APPLIED:{name}")
