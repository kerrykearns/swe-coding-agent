"""Tests for the M8 evaluation harness — eval/harness.py, run_eval.py, report.py.

Everything here is fully offline: agent runs are driven by the same scripted
LLM clients test_react_agent.py and test_baseline.py already use (imported
rather than re-implemented, exactly as test_run_on_issue.py imports
test_react_agent's FIXED_CALC — one definition of "what a scripted client
looks like" for this suite). No real Groq call is made anywhere in this file.

The fixture used here is a tiny, self-contained buggy repo (pkg/mod.py +
pkg/test_mod.py), deliberately decoupled from demo/sample_repos/calculator —
these tests exercise the harness's own logic (verification, resumability,
aggregation), not the calculator fixture's specific bugs, so they should not
need to change if that fixture's bug set ever does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI, RateLimitError
from pydantic import ValidationError

from agent.core import llm_client
from agent.core.react_agent import FINISH_TOOL, Turn
from eval import harness, report, run_eval
from eval.harness import EvalResult

from .test_baseline import _stub_client
from .test_react_agent import _call, _reply, _scripted_client

# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_fixture(tmp_path: Path) -> Path:
    """A minimal buggy repo: one function, one test, nothing else."""
    root = tmp_path / "tiny_fixture"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (pkg / "test_mod.py").write_text(
        "from pkg.mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def tiny_task(tiny_fixture: Path) -> dict:
    return {
        "id": "tiny-add",
        "title": "add() is broken",
        "description": "add(2, 3) returns -1 instead of 5.",
        "fixture_repo": str(tiny_fixture),
        "target_file": "pkg/mod.py",
        "verify_command": "pytest pkg/test_mod.py::test_add -v",
        "difficulty": "trivial",
    }


def _rate_limit_error() -> Exception:
    """A real openai.RateLimitError, built the same way test_react_agent.py's
    _tool_use_failed() builds a real BadRequestError — from an actual httpx
    response, so it carries whatever shape the installed SDK actually gives a
    429, not a guess at one.
    """
    payload = {
        "error": {
            "message": "Rate limit reached for this model.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, json=payload)
    client = OpenAI(api_key="not-a-real-key", base_url=llm_client.DEFAULT_BASE_URL)
    return client._make_status_error_from_response(response)


class _RateLimitedCompletions:
    """A client whose very first call raises a real rate-limit error."""

    def create(self, **kwargs):
        raise _rate_limit_error()


def _rate_limited_client() -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_RateLimitedCompletions()))


def _make_result(
    task_id: str = "t1",
    condition: str = "baseline",
    success: bool = True,
    turns: int = 1,
    total_tokens: int = 10,
    wall_clock: float = 0.1,
) -> EvalResult:
    return EvalResult(
        task_id=task_id,
        condition=condition,
        success=success,
        turns=turns,
        total_tokens=total_tokens,
        wall_clock_seconds=wall_clock,
        verify_output="ok",
        error=None,
    )


# --------------------------------------------------------------------------
# EvalResult construction / serialization
# --------------------------------------------------------------------------


def test_eval_result_round_trips_through_json():
    result = EvalResult(
        task_id="t1",
        condition="react",
        success=True,
        turns=3,
        total_tokens=500,
        wall_clock_seconds=12.5,
        verify_output="2 passed",
        error=None,
        full_trajectory=[
            Turn(turn_number=1, tool_name="write_file", tool_result={"ok": True}),
            Turn(turn_number=2, tool_name="run_tests", tool_result={"ok": True, "success": True}),
        ],
    )
    assert EvalResult.model_validate_json(result.model_dump_json()) == result


def test_eval_result_full_trajectory_defaults_to_none():
    result = EvalResult(
        task_id="t1",
        condition="baseline",
        success=True,
        turns=1,
        total_tokens=1,
        wall_clock_seconds=0.1,
        verify_output="",
    )
    assert result.full_trajectory is None


def test_eval_result_error_defaults_to_none():
    result = EvalResult(
        task_id="t1",
        condition="baseline",
        success=False,
        turns=1,
        total_tokens=1,
        wall_clock_seconds=0.1,
        verify_output="",
    )
    assert result.error is None


def test_eval_result_rejects_an_unknown_condition():
    with pytest.raises(ValidationError):
        EvalResult(
            task_id="t1",
            condition="not-a-real-condition",
            success=True,
            turns=1,
            total_tokens=1,
            wall_clock_seconds=0.1,
            verify_output="",
        )


# --------------------------------------------------------------------------
# _test_path_from_verify_command
# --------------------------------------------------------------------------


def test_test_path_strips_the_pytest_prefix_and_flags():
    assert (
        harness._test_path_from_verify_command(
            "pytest calculator/test_calc.py::test_add -v"
        )
        == "calculator/test_calc.py::test_add"
    )


def test_test_path_keeps_multiple_node_ids():
    assert (
        harness._test_path_from_verify_command(
            "pytest calculator/test_calc.py::test_divide "
            "calculator/test_calc.py::test_divide_by_zero -v"
        )
        == "calculator/test_calc.py::test_divide calculator/test_calc.py::test_divide_by_zero"
    )


def test_test_path_raises_when_nothing_is_left():
    with pytest.raises(ValueError, match="no test path"):
        harness._test_path_from_verify_command("pytest -v")


# --------------------------------------------------------------------------
# run_task — verified, not claimed
# --------------------------------------------------------------------------


def test_run_task_baseline_succeeds_when_the_fix_actually_verifies(tiny_task):
    client = _stub_client("```python\ndef add(a, b):\n    return a + b\n```")
    result = harness.run_task(tiny_task, "baseline", client=client)

    assert result.task_id == "tiny-add"
    assert result.condition == "baseline"
    assert result.success is True
    assert result.turns == 1
    assert result.total_tokens == 33  # from _stub_client's fixed usage numbers
    assert "1 passed" in result.verify_output
    assert result.error is None
    assert result.full_trajectory is None  # baseline has no turns to record


def test_run_task_baseline_fails_when_the_write_doesnt_fix_it(tiny_task):
    """The model's reply is well-formed and gets written — it just doesn't work."""
    client = _stub_client("```python\ndef add(a, b):\n    return a - b\n```")
    result = harness.run_task(tiny_task, "baseline", client=client)

    assert result.success is False
    assert "test_add" in result.verify_output


def test_run_task_react_ignores_an_unverified_claim_of_success(tiny_task):
    """The core 'verified, not claimed' case: the agent is confident and wrong.

    It writes a no-op (still-broken) fix, never calls run_tests, then claims
    success via finish(). Nothing about that claim may reach EvalResult.success
    — only the harness's own independent re-run of verify_command does.
    """
    client = _scripted_client(
        _reply(
            "Obvious fix, no need to check.",
            (
                _call(
                    "write_file",
                    "c1",
                    path="pkg/mod.py",
                    content="def add(a, b):\n    return a - b\n",  # still buggy
                ),
            ),
        ),
        _reply(
            "That is correct, I am confident.",
            (_call(FINISH_TOOL, "c2", success=True, summary="Fixed it."),),
        ),
    )

    result = harness.run_task(tiny_task, "react", client=client)

    assert result.success is False
    assert result.turns == 2
    assert "test_add" in result.verify_output
    # The claim is right there in the trajectory, available for inspection —
    # it just never reached `success`.
    assert result.full_trajectory is not None
    assert [turn.tool_name for turn in result.full_trajectory] == ["write_file", FINISH_TOOL]
    assert result.full_trajectory[-1].tool_args["success"] is True


def test_run_task_react_succeeds_when_the_fix_actually_verifies(tiny_task):
    client = _scripted_client(
        _reply(
            "Fixing.",
            (_call("write_file", "c1", path="pkg/mod.py", content="def add(a, b):\n    return a + b\n"),),
        ),
        _reply("Verify.", (_call("run_tests", "c2", test_path="."),)),
    )

    result = harness.run_task(tiny_task, "react", client=client)

    assert result.success is True
    assert result.turns == 2
    assert result.error is None
    assert result.full_trajectory is not None
    assert [turn.tool_name for turn in result.full_trajectory] == ["write_file", "run_tests"]


def test_run_task_never_mutates_the_tracked_fixture(tiny_fixture, tiny_task):
    original = (tiny_fixture / "pkg" / "mod.py").read_text(encoding="utf-8")
    client = _stub_client("```python\ndef add(a, b):\n    return a + b\n```")

    harness.run_task(tiny_task, "baseline", client=client)

    assert (tiny_fixture / "pkg" / "mod.py").read_text(encoding="utf-8") == original


def test_run_task_records_a_missing_fixture_as_a_failed_result_not_a_crash(tmp_path):
    task = {
        "id": "missing-fixture",
        "title": "x",
        "description": "y",
        "fixture_repo": str(tmp_path / "does-not-exist"),
        "target_file": "pkg/mod.py",
        "verify_command": "pytest pkg/test_mod.py::test_add -v",
        "difficulty": "trivial",
    }

    result = harness.run_task(
        task, "baseline", client=_stub_client("```python\nx = 1\n```")
    )

    assert result.task_id == "missing-fixture"
    assert result.success is False
    assert result.error is not None
    assert "does-not-exist" in result.error or "No such file" in result.error


def test_run_task_lets_a_rate_limit_error_propagate(tiny_task):
    """A 429 is not this task's failure — see harness.run_task's docstring."""
    with pytest.raises(RateLimitError):
        harness.run_task(tiny_task, "baseline", client=_rate_limited_client())


# --------------------------------------------------------------------------
# run_eval.run_matrix — resumability and rate-limit handling
# --------------------------------------------------------------------------


def test_existing_result_file_is_skipped_unless_forced(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "t1__baseline.json").write_text(
        _make_result().model_dump_json(), encoding="utf-8"
    )

    calls = []

    def fake_run_task(task, condition, **kwargs):
        calls.append((task["id"], condition))
        return _make_result(task_id=task["id"], condition=condition)

    summary = run_eval.run_matrix(
        [{"id": "t1"}], ["baseline"], results_dir, run_task_fn=fake_run_task
    )

    assert calls == []
    assert summary == {"completed": 0, "skipped": 1, "rate_limited": False}


def test_force_reruns_even_when_a_result_file_exists(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "t1__baseline.json").write_text(
        _make_result(total_tokens=1).model_dump_json(), encoding="utf-8"
    )

    calls = []

    def fake_run_task(task, condition, **kwargs):
        calls.append((task["id"], condition))
        return _make_result(task_id=task["id"], condition=condition, total_tokens=999)

    summary = run_eval.run_matrix(
        [{"id": "t1"}],
        ["baseline"],
        results_dir,
        run_task_fn=fake_run_task,
        force=True,
    )

    assert calls == [("t1", "baseline")]
    assert summary == {"completed": 1, "skipped": 0, "rate_limited": False}
    saved = EvalResult.model_validate_json(
        (results_dir / "t1__baseline.json").read_text(encoding="utf-8")
    )
    assert saved.total_tokens == 999  # really re-ran, not just re-saved the old one


def test_run_matrix_saves_one_file_per_task_condition_combination(tmp_path):
    results_dir = tmp_path / "results"
    tasks = [{"id": "t1"}, {"id": "t2"}]

    def fake_run_task(task, condition, **kwargs):
        return _make_result(task_id=task["id"], condition=condition)

    summary = run_eval.run_matrix(
        tasks, ["baseline", "react"], results_dir, run_task_fn=fake_run_task
    )

    assert summary == {"completed": 4, "skipped": 0, "rate_limited": False}
    assert sorted(p.name for p in results_dir.glob("*.json")) == [
        "t1__baseline.json",
        "t1__react.json",
        "t2__baseline.json",
        "t2__react.json",
    ]


def test_rate_limit_stops_the_matrix_and_preserves_progress(tmp_path):
    results_dir = tmp_path / "results"
    tasks = [{"id": "t1"}, {"id": "t2"}]
    call_count = 0

    def fake_run_task(task, condition, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise _rate_limit_error()
        return _make_result(task_id=task["id"], condition=condition)

    summary = run_eval.run_matrix(
        tasks, ["baseline"], results_dir, run_task_fn=fake_run_task
    )

    assert summary == {"completed": 1, "skipped": 0, "rate_limited": True}
    assert (results_dir / "t1__baseline.json").exists()
    assert not (results_dir / "t2__baseline.json").exists()


def test_rate_limit_on_the_very_first_call_attempts_only_one_combination(tmp_path):
    """A 429 on the first call must stop the matrix right there — not let a
    second (task, condition) combination start and fail the same way before
    halting. Regression test for a real run where run_react_agent swallowed
    the RateLimitError into a normal (recorded, not raised) failed turn, so
    run_matrix never saw it until the *next* combination hit the same
    exhausted quota through a path that did propagate it.
    """
    results_dir = tmp_path / "results"
    tasks = [{"id": "t1"}, {"id": "t2"}]
    call_count = 0

    def fake_run_task(task, condition, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _rate_limit_error()

    summary = run_eval.run_matrix(
        tasks, ["baseline"], results_dir, run_task_fn=fake_run_task
    )

    assert call_count == 1
    assert summary == {"completed": 0, "skipped": 0, "rate_limited": True}
    assert not (results_dir / "t1__baseline.json").exists()
    assert not (results_dir / "t2__baseline.json").exists()


def test_rate_limit_message_is_specific_and_actionable():
    message = run_eval._rate_limit_message(4)
    assert "4 runs" in message
    assert "tomorrow" in message


# --------------------------------------------------------------------------
# report.py — aggregation
# --------------------------------------------------------------------------


def test_compute_summary_success_rate_and_averages():
    results = [
        _make_result(task_id="a", condition="baseline", success=True, turns=1, total_tokens=100, wall_clock=1.0),
        _make_result(task_id="b", condition="baseline", success=False, turns=1, total_tokens=200, wall_clock=2.0),
        _make_result(task_id="a", condition="react", success=True, turns=3, total_tokens=1000, wall_clock=10.0),
        _make_result(task_id="b", condition="react", success=True, turns=5, total_tokens=2000, wall_clock=20.0),
    ]
    summary = report.compute_summary(results, {"a": "trivial", "b": "medium"})

    assert summary["baseline"]["n"] == 2
    assert summary["baseline"]["successes"] == 1
    assert summary["baseline"]["success_rate"] == 0.5
    assert summary["baseline"]["avg_total_tokens"] == 150
    assert summary["baseline"]["avg_wall_clock_seconds"] == 1.5

    assert summary["react"]["n"] == 2
    assert summary["react"]["successes"] == 2
    assert summary["react"]["success_rate"] == 1.0
    assert summary["react"]["avg_turns"] == 4.0
    assert summary["react"]["avg_total_tokens"] == 1500


def test_compute_summary_groups_by_difficulty_tier():
    results = [
        _make_result(task_id="a", condition="react", success=True),
        _make_result(task_id="b", condition="react", success=False),
        _make_result(task_id="c", condition="react", success=True),
    ]
    summary = report.compute_summary(results, {"a": "trivial", "b": "medium", "c": "medium"})

    by_difficulty = summary["react"]["by_difficulty"]
    assert by_difficulty["trivial"] == {"n": 1, "successes": 1, "success_rate": 1.0}
    assert by_difficulty["medium"] == {"n": 2, "successes": 1, "success_rate": 0.5}


def test_compute_summary_groups_unmapped_tasks_under_unknown():
    results = [_make_result(task_id="ghost", condition="react", success=True)]
    summary = report.compute_summary(results, {})  # no difficulty mapping at all
    assert summary["react"]["by_difficulty"]["unknown"] == {
        "n": 1,
        "successes": 1,
        "success_rate": 1.0,
    }


def test_load_results_reads_every_json_file_in_a_directory(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "a__baseline.json").write_text(
        _make_result(task_id="a").model_dump_json(), encoding="utf-8"
    )
    (results_dir / "b__react.json").write_text(
        _make_result(task_id="b", condition="react").model_dump_json(), encoding="utf-8"
    )

    loaded = report.load_results(results_dir)

    assert {r.task_id for r in loaded} == {"a", "b"}


def test_render_markdown_states_sample_size_modestly_and_never_overclaims():
    results = [_make_result(task_id="a", condition="baseline")]
    summary = report.compute_summary(results, {"a": "trivial"})
    markdown = report.render_markdown(summary)

    assert "1 for baseline" in markdown
    assert "directional findings" in markdown
    assert "n/a (single-shot)" in markdown  # baseline's avg-turns cell
    for overclaim in ("proves", "guarantees", "definitively", "statistically significant"):
        assert overclaim not in markdown.lower()


def test_render_markdown_reports_empty_results_without_crashing():
    markdown = report.render_markdown({})
    assert "No results were found" in markdown
