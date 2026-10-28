"""A skipped, absent or broken check must never become reported coverage."""

import importlib.util
from pathlib import Path


def runner_module():
    spec = importlib.util.spec_from_file_location("stack_runner", Path(__file__).with_name("run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_junit_pass_skip_failure_and_error(tmp_path):
    report = tmp_path / "results.xml"
    report.write_text("""<testsuites><testsuite>
        <testcase name="ok[a]" classname="module"/>
        <testcase name="skipped"><skipped/></testcase>
        <testcase name="broken"><failure/></testcase>
        <testcase name="error"><error/></testcase>
    </testsuite></testsuites>""")
    module = runner_module()
    cases = module.read_cases(report)
    assert [case["status"] for case in cases] == ["pass", "skipped", "failure", "error"]
    features = [{"id": name, "tests": [name]} for name in ("ok", "skipped", "broken", "error", "missing")]
    assert [row["status"] for row in module.feature_results(features, cases)] == [
        "pass",
        "fail",
        "fail",
        "fail",
        "missing",
    ]
    assert module.read_cases(tmp_path / "missing.xml") == []


def test_every_parameter_must_pass():
    module = runner_module()
    feature = [{"id": "parameterized", "tests": ["check"]}]
    cases = [{"name": "check[a]", "status": "pass"}, {"name": "check[b]", "status": "failure"}]
    assert module.feature_results(feature, cases)[0]["status"] == "fail"


def test_stage_must_have_completed_successfully():
    module = runner_module()
    feature = [{"id": "ui", "stages": ["ui"]}]
    assert module.feature_results(feature, [], [])[0]["status"] == "missing"
    assert module.feature_results(feature, [], [{"name": "ui", "exit_code": 1}])[0]["status"] == "fail"
    assert module.feature_results(feature, [], [{"name": "ui", "exit_code": 0}])[0]["status"] == "pass"


def test_unmapped_skip_also_blocks_overall_success():
    module = runner_module()
    stages = [{"name": "tests", "exit_code": 0}]
    sensitivity = [{"detected": True}]
    features = [{"status": "pass"}]
    passing = [{"name": "mapped", "status": "pass"}]
    assert module.all_checks_pass(passing, stages, sensitivity, features)
    for status in ("skipped", "failure", "error"):
        assert not module.all_checks_pass(
            passing + [{"name": "unmapped", "status": status}], stages, sensitivity, features
        )
