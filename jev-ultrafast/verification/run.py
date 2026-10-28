"""One offline entrypoint: regressions, bounded exploration, real DOM, UI and mutation sensitivity."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_cases(path):
    if not path.exists():
        return []
    result = []
    for case in ET.parse(path).iter("testcase"):
        status = "pass"
        for tag in ("skipped", "failure", "error"):
            if case.find(tag) is not None:
                status = tag
        result.append({"name": case.attrib["name"], "class": case.attrib.get("classname", ""), "status": status})
    return result


def feature_results(features, cases, stages=()):
    rows = []
    for feature in features:
        matched, missing = [], []
        for selector in feature.get("tests", []):
            selected = [case for case in cases if case["name"].split("[", 1)[0] == selector]
            matched.extend(selected)
            if not selected:
                missing.append(selector)
        status = "missing" if missing else "pass"
        if any(case["status"] != "pass" for case in matched):
            status = "fail"
        for name in feature.get("stages", []):
            stage = next((stage for stage in stages if stage["name"] == name), None)
            if stage is None:
                missing.append(name)
                status = "missing"
            elif stage["exit_code"] != 0:
                status = "fail"
        rows.append({**feature, "status": status, "cases": len(matched), "missing": missing})
    return rows


def all_checks_pass(cases, stages, sensitivity, features):
    normal = [stage for stage in stages if not stage["name"].startswith("mutation-")]
    return bool(cases and normal and sensitivity and features) and (
        all(case["status"] == "pass" for case in cases)
        and all(stage["exit_code"] == 0 for stage in normal)
        and all(row["detected"] for row in sensitivity)
        and all(row["status"] == "pass" for row in features)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-use", type=Path, default=ROOT.parent, help="repository root (default: parent of jev-ultrafast/)"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "verification")
    args = parser.parse_args()
    browser_use = args.browser_use.resolve()
    output = args.output.resolve()
    if not (browser_use / "browser_use/decision.py").is_file():
        parser.error("--browser-use must point to the repository root containing browser_use/decision.py")
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("API_KEY", "API_TOKEN", "ENTRA_TOKEN"))
    }
    environment.update(
        BROWSER_USE_REPO=str(browser_use),
        ANONYMIZED_TELEMETRY="false",
        PYTHONUTF8="1",
        PYTHONPATH=os.pathsep.join([str(ROOT), str(browser_use)]),
        STACK_VERIFY_OUTPUT=str(output),
    )
    stages = []

    def run(name, command, cwd=ROOT, extra_env=None, timeout=300):
        started = time.monotonic()
        print(f"Running {name}", flush=True)
        for argument in command:
            if argument.startswith("--junitxml="):
                Path(argument.split("=", 1)[1]).unlink(missing_ok=True)
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                env={**environment, **(extra_env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            code, text = process.returncode, process.stdout
        except subprocess.TimeoutExpired as error:
            code = 124
            text = str(error.stdout or "") + "\nVerification stage timed out; not a pass."
        (output / f"{name}.log").write_text(text, encoding="utf-8")
        row = {"name": name, "command": command, "exit_code": code, "seconds": round(time.monotonic() - started, 2)}
        stages.append(row)
        print(f"{name}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
        return row, text

    uv = ["uv", "run", "--frozen", "--python", "3.12", "--project", str(browser_use), "--with-editable", str(ROOT)]
    _, toolchain_text = run(
        "toolchain",
        uv
        + [
            "python",
            "-c",
            """
import json, platform
from importlib.metadata import version
print(json.dumps({'python': platform.python_version(), **{
    name: version(name) for name in ('pytest','pytest-asyncio','browser-harness','httpx','h2')}}))
""",
        ],
    )
    toolchain = json.loads(toolchain_text.strip().splitlines()[-1])
    run("node-version", ["node", "--version"])
    browser_suite = [
        str(browser_use / "tests/ci/test_microsoft_decision.py"),
        str(browser_use / "tests/ci/test_dom_live_input_value.py"),
    ]
    pytest_args = ["-q", "--asyncio-mode=auto", "--timeout=60", "--tb=short", "-o", "log_cli=false", "-o", "addopts="]
    run(
        "browser-tests",
        uv + ["python", "-m", "pytest", *browser_suite, *pytest_args, f"--junitxml={output / 'browser.xml'}"],
    )
    run(
        "jev-tests",
        ["uv", "run", "--frozen", "pytest", "tests", "-q", "--tb=short", f"--junitxml={output / 'jev.xml'}"],
        extra_env={"PYTHONPATH": str(ROOT)},
    )
    run(
        "stack-tests",
        uv
        + [
            "python",
            "-m",
            "pytest",
            "verification",
            "--rootdir",
            str(ROOT),
            *pytest_args,
            f"--junitxml={output / 'stack.xml'}",
        ],
    )
    cases = read_cases(output / "browser.xml") + read_cases(output / "jev.xml") + read_cases(output / "stack.xml")
    mutations = {
        "consume": "test_exhaustive_single_consumption",
        "freshness": "test_browser_freshness_interleavings[url-click]",
        "browser_choice": "test_exhaustive_choice_domain[browser-use]",
        "jev_choice": "test_exhaustive_choice_domain[jev]",
    }
    sensitivity = []
    for mutation, test in mutations.items():
        xml_path = output / f"mutation-{mutation}.xml"
        row, text = run(
            f"mutation-{mutation}",
            uv
            + [
                "python",
                "-m",
                "pytest",
                "-p",
                "verification.plugin_mutations",
                f"verification/test_state_space.py::{test}",
                *pytest_args,
                f"--junitxml={xml_path}",
            ],
            extra_env={"STACK_MUTATION": mutation},
        )
        baseline = [case for case in cases if case["name"] == test]
        mutated = read_cases(xml_path)
        detected = (
            baseline
            and all(case["status"] == "pass" for case in baseline)
            and row["exit_code"] == 1
            and f"MUTATION_APPLIED:{mutation}" in text
            and any(case["name"] == test and case["status"] == "failure" for case in mutated)
        )
        sensitivity.append({"mutation": mutation, "test": test, "detected": bool(detected)})
    run("ui", ["node", "verification/check_ui.cjs"])
    run("existing-ui", ["node", "tests/check_demo_ui.cjs"])
    run("syntax-ui", ["node", "--check", "jev_ultrafast/static/app.js"])
    run("syntax-snapshot", ["node", "--check", "jev_ultrafast/snapshot.js"])
    run("lint", ["uv", "run", "--frozen", "ruff", "check", "."])
    run("build-jev", ["uv", "build", "--out-dir", str(output / "dist-jev")])
    run("build-browser-use", ["uv", "build", "--out-dir", str(output / "dist-browser-use")], cwd=browser_use)
    run(
        "wheel-content",
        [
            sys.executable,
            "-c",
            """
import pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
for directory, required in (
    ('dist-jev', {'jev_ultrafast/snapshot.js', 'jev_ultrafast/static/app.js',
                  'jev_ultrafast/static/index.html', 'jev_ultrafast/agent.py'}),
    ('dist-browser-use', {'browser_use/decision.py', 'browser_use/__init__.py'})):
    wheels = list((root / directory).glob('*.whl'))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert required <= set(wheel.namelist()), required - set(wheel.namelist())
        if directory == 'dist-jev':
            entries = next(name for name in wheel.namelist() if name.endswith('entry_points.txt'))
            assert 'jev = jev_ultrafast.demo:main' in wheel.read(entries).decode()
print('PASS: runtime assets and CLI entrypoint are packaged')
""",
            str(output),
        ],
    )
    features = feature_results(
        json.loads((ROOT / "verification/features.json").read_text(encoding="utf-8")), cases, stages
    )
    covered = json.loads((ROOT / "verification/sources.json").read_text(encoding="utf-8"))
    sources = []
    for repository, files in covered.items():
        root = ROOT if repository == "jev-ultrafast" else browser_use
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        sources.append(
            {
                "repository": repository,
                "revision": revision,
                "files": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in files},
            }
        )
    passed = all_checks_pass(cases, stages, sensitivity, features)
    result = {
        "passed": passed,
        "cases": cases,
        "features": features,
        "mutations": sensitivity,
        "stages": stages,
        "sources": sources,
        "toolchain": toolchain,
        "chromium": json.loads((output / "chromium.json").read_text(encoding="utf-8"))
        if (output / "chromium.json").exists()
        else None,
        "verification_files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "verification").glob("*"))
            if path.is_file()
        },
        "bounds": {
            "choice_tuples_per_validator": 3000,
            "command_sequences": 780,
            "max_sequence_length": 4,
            "native_select_option_count": 2,
            "native_select_prototypes": 8,
            "native_select_disabled_masks": 4,
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Decision stack verification",
        "",
        f"Overall: {'PASS' if passed else 'FAIL'}",
        "",
        "No paid model calls. Mock HTTP responses drive real implementations and isolated Chromium.",
        "",
        "| Feature | Result | Test cases |",
        "|---|---|---|",
    ]
    lines.extend(f"| {row['id']}: {row['scope']} | {row['status']} | {row['cases']} |" for row in features)
    lines += ["", "## Mutation sensitivity", ""]
    lines.extend(f"- {row['mutation']}: {'detected' if row['detected'] else 'NOT DETECTED'}" for row in sensitivity)
    lines += [
        "",
        "## Limits",
        "",
        "This is bounded verification, not a proof for every site, input, or interleaving.",
        "Provider quality, external sites, cloud browsers, and real credentials are outside this offline suite.",
        "Browser use and Jev are separate executors. Harness CDP is routed to an isolated test browser.",
        "A failing assertion remains a failure; skipped, missing, and unexecuted checks never count as pass.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {len(cases)} test cases; report: {output / 'REPORT.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
