import json

import pytest

from judgepipe import cli


@pytest.mark.parametrize("status,expected", [("incomplete", 1), ("complete", 0)])
def test_exit_status_reflects_actual_completion(tmp_path, monkeypatch, status, expected):
    monkeypatch.setenv("JUDGE_MODEL", "claude-example")
    monkeypatch.setenv("GENERATOR_MODEL", "reference-written")
    monkeypatch.setenv("JUDGE_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(cli, "make_client", lambda config: object())
    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, cases):
            return {"run_status": status, "audit": {"prompt_tokens": 0, "completion_tokens": 0}}
    monkeypatch.setattr(cli, "Runner", FakeRunner)
    result = cli.main(["run", "--suite", "judgepipe/data/suite.json", "--out", str(tmp_path), "--allow-same-family"])
    assert result == expected
    assert json.loads((tmp_path / "report.json").read_text())["run_status"] == status
