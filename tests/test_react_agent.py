"""Tests for the multi-turn ReAct agent.

Everything here runs offline except the one test marked ``integration``. The
offline tests drive the loop with a *scripted* LLM client — a stub that returns
pre-written tool calls in a fixed order — so the control flow (which tool ran,
when the loop stopped, and why) is fully deterministic and costs nothing. The
tools themselves are real: files are really written and pytest really runs, since
that is where the loop's observations come from.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, OpenAI

from agent.core import llm_client
from agent.core import react_agent as react_agent_module
from agent.core.prompts import load_prompt, placeholders, render_react_prompts
from agent.core.react_agent import (
    FINISH_TOOL,
    MAX_INITIAL_LISTING,
    MAX_LISTED_FILES,
    MAX_LISTED_MATCHES,
    MEMORY_SIMILARITY_THRESHOLD,
    TOOLS,
    StopReason,
    Trajectory,
    Turn,
    _initial_file_listing,
    _memory_reference,
    _render_observation,
    _safe_diff,
    _truncate,
    execute_tool,
    parse_tool_arguments,
    run_react_agent,
    tool_schemas,
)
from agent.tools import Workspace, read_file

FIXED_CALC = (
    '"""A deliberately broken calculator, used as an agent fixture.\n\n'
    "The bug in add() is intentional: test_calc.py fails against this file, which\n"
    "gives the agent (and the tool-layer tests) something real to detect and fix.\n"
    '"""\n\n\ndef add(a, b):\n    """Return the sum of a and b."""\n'
    "    return a + b\n\n\ndef subtract(a, b):\n"
    '    """Return a minus b."""\n    return a - b\n'
)


# --------------------------------------------------------------------------
# A scripted LLM client: no network, no non-determinism
# --------------------------------------------------------------------------


def _call(name: str, call_id: str = "call_1", **arguments) -> SimpleNamespace:
    """One tool call, shaped exactly as the OpenAI SDK returns it."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _raw_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    """A tool call whose arguments are not necessarily valid JSON."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _reply(
    content: str = "",
    tool_calls: tuple = (),
    tokens: int = 100,
    finish_reason: str = "tool_calls",
) -> SimpleNamespace:
    """One assistant response, shaped as the OpenAI SDK returns it."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, tool_calls=list(tool_calls) or None
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=tokens - 10, completion_tokens=10, total_tokens=tokens
        ),
    )


class _ScriptedCompletions:
    """Stands in for ``client.chat.completions``, replaying a fixed script."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(
                "the loop asked for more responses than the script provides — "
                f"it has already made {len(self.calls)} calls"
            )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _scripted_client(*script) -> SimpleNamespace:
    completions = _ScriptedCompletions(script)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _requests(client) -> list[dict]:
    return client.chat.completions.calls


def _unused_responses(client) -> int:
    return len(client.chat.completions.script)


# --------------------------------------------------------------------------
# The Trajectory model
# --------------------------------------------------------------------------


def test_trajectory_defaults_to_an_empty_failed_run():
    trajectory = Trajectory(task_description="fix add()")
    assert trajectory.turns == []
    assert trajectory.final_success is False
    assert trajectory.final_diff == ""
    assert trajectory.total_turns == 0
    assert trajectory.total_tokens == 0
    assert trajectory.stop_reason is StopReason.ERROR


def test_trajectory_round_trips_through_a_dict():
    original = Trajectory(
        task_description="fix add()",
        turns=[
            Turn(
                turn_number=1,
                reasoning="Let me read the file.",
                tool_name="read_file",
                tool_args={"path": "calculator/calc.py"},
                tool_result={"ok": True, "content": "def add(a, b): ..."},
                tokens_used=120,
            )
        ],
        final_success=True,
        final_diff="--- a/calc.py\n+++ b/calc.py\n",
        total_turns=1,
        total_tokens=120,
        stop_reason=StopReason.TESTS_PASSED,
    )

    restored = Trajectory.model_validate(original.model_dump())
    assert restored == original
    assert restored.turns[0].tool_args == {"path": "calculator/calc.py"}


def test_trajectory_serialises_its_stop_reason_as_a_plain_string():
    """The eval harness (M8) reads these back as JSON, not as Python enums."""
    trajectory = Trajectory(
        task_description="fix add()", stop_reason=StopReason.MAX_TURNS_REACHED
    )
    payload = json.loads(trajectory.model_dump_json())
    assert payload["stop_reason"] == "max_turns_reached"
    assert payload["task_description"] == "fix add()"


def test_turn_defaults_describe_a_turn_that_did_nothing():
    turn = Turn(turn_number=3)
    assert turn.reasoning == ""
    assert turn.tool_name is None
    assert turn.tool_args == {}
    assert turn.tool_result == {}
    assert turn.tokens_used == 0


# --------------------------------------------------------------------------
# Tool declarations and schemas
# --------------------------------------------------------------------------


def test_the_toolset_is_exactly_the_six_tools_plus_finish():
    assert set(TOOLS) == {
        "read_file",
        "write_file",
        "list_files",
        "search_text",
        "run_tests",
        "get_diff",
        FINISH_TOOL,
    }


def test_no_shell_tool_is_exposed_to_the_agent():
    """M3's scoping constraint: raw shell access waits for M5/M6.

    This test exists to fail loudly if `run_command` is ever wired into the
    toolset before the sandbox and the safety gate are in place.
    """
    assert "run_command" not in TOOLS
    serialised = json.dumps(tool_schemas()).lower()
    for forbidden in ("run_command", "shell", "subprocess", "bash"):
        assert forbidden not in serialised


def test_tool_schemas_are_in_openai_function_format():
    schemas = {schema["function"]["name"]: schema for schema in tool_schemas()}
    assert set(schemas) == set(TOOLS)

    for name, schema in schemas.items():
        assert schema["type"] == "function"
        assert schema["function"]["description"].strip(), f"{name} has no description"
        assert schema["function"]["parameters"]["type"] == "object"

    write = schemas["write_file"]["function"]["parameters"]
    assert set(write["required"]) == {"path", "content"}
    assert "REPLACES the whole file" in write["properties"]["content"]["description"]

    # Optional arguments carry defaults rather than being required.
    run_tests_params = schemas["run_tests"]["function"]["parameters"]
    assert run_tests_params["properties"]["test_path"]["default"] == "."
    assert not run_tests_params.get("required")

    # get_diff takes nothing at all.
    assert schemas["get_diff"]["function"]["parameters"].get("properties", {}) == {}

    assert set(schemas[FINISH_TOOL]["function"]["parameters"]["required"]) == {
        "success",
        "summary",
    }


# --------------------------------------------------------------------------
# parse_tool_arguments
# --------------------------------------------------------------------------


def test_parse_fills_in_schema_defaults():
    arguments, error = parse_tool_arguments("list_files", '{"pattern": "*.py"}')
    assert error is None
    assert arguments == {"subdir": ".", "pattern": "*.py"}


def test_parse_accepts_an_empty_argument_string_for_a_no_arg_tool():
    arguments, error = parse_tool_arguments("get_diff", "")
    assert (arguments, error) == ({}, None)


def test_parse_rejects_an_unknown_tool_name_by_listing_the_real_ones():
    arguments, error = parse_tool_arguments("run_command", '{"command": "rm -rf /"}')
    assert arguments == {}
    assert "no tool called 'run_command'" in error
    assert "run_tests" in error


def test_parse_reports_malformed_json():
    arguments, error = parse_tool_arguments("read_file", '{"path": "calc.py"')
    assert arguments == {}
    assert "not valid JSON" in error


def test_parse_reports_a_non_object_payload():
    arguments, error = parse_tool_arguments("read_file", '"calc.py"')
    assert arguments == {}
    assert "must be a JSON object" in error


def test_parse_reports_a_missing_required_argument():
    arguments, error = parse_tool_arguments("write_file", '{"path": "calc.py"}')
    assert arguments == {}
    assert "do not match its schema" in error
    assert "content" in error


# --------------------------------------------------------------------------
# execute_tool
# --------------------------------------------------------------------------


def test_execute_tool_reads_and_writes(calculator_ws: Workspace):
    read = execute_tool(calculator_ws, "read_file", {"path": "calculator/calc.py"})
    assert read["ok"] is True
    assert "return a - b" in read["content"]

    written = execute_tool(
        calculator_ws,
        "write_file",
        {"path": "calculator/calc.py", "content": FIXED_CALC},
    )
    assert written["ok"] is True
    assert written["path"] == "calculator/calc.py"
    assert written["chars_written"] == len(FIXED_CALC)
    assert read_file(calculator_ws, "calculator/calc.py") == FIXED_CALC

    diff = execute_tool(calculator_ws, "get_diff", {})
    assert diff["ok"] is True
    assert "+    return a + b" in diff["diff"]


def test_a_write_reports_its_line_count(calculator_ws: Workspace):
    result = execute_tool(
        calculator_ws,
        "write_file",
        {"path": "calculator/calc.py", "content": FIXED_CALC},
    )
    assert result["lines_written"] == len(FIXED_CALC.splitlines())
    assert result["suspect_escaped_newlines"] is False


def test_a_write_of_escape_mangled_content_is_flagged_back_to_the_agent(
    calculator_ws: Workspace,
):
    r"""Models double-escape ``content``; the agent is told, not silently corrected."""
    mangled = FIXED_CALC.replace("\n", "\\n")
    result = execute_tool(
        calculator_ws, "write_file", {"path": "calculator/calc.py", "content": mangled}
    )

    assert result["ok"] is True  # it really was written — we do not rewrite it
    assert result["suspect_escaped_newlines"] is True
    assert read_file(calculator_ws, "calculator/calc.py") == mangled


def test_real_code_mentioning_a_newline_escape_is_not_flagged(calculator_ws: Workspace):
    r"""``"\n"`` inside a string literal is ordinary Python, not a mangled file."""
    source = 'def render(rows):\n    """Join rows."""\n    return "\\n".join(rows) + "\\n"\n'
    result = execute_tool(
        calculator_ws, "write_file", {"path": "render.py", "content": source}
    )
    assert result["suspect_escaped_newlines"] is False


def test_execute_tool_lists_and_searches(calculator_ws: Workspace):
    listed = execute_tool(calculator_ws, "list_files", {"subdir": ".", "pattern": "*.py"})
    assert listed["ok"] is True
    assert "calculator/calc.py" in listed["files"]
    assert listed["count"] == len(listed["files"])

    found = execute_tool(
        calculator_ws,
        "search_text",
        {"query": "def add", "subdir": ".", "pattern": "*.py"},
    )
    assert found["ok"] is True
    assert found["matches"][0]["file"] == "calculator/calc.py"


def test_execute_tool_returns_a_missing_file_as_an_observation(calculator_ws: Workspace):
    result = execute_tool(calculator_ws, "read_file", {"path": "nope.py"})
    assert result["ok"] is False
    assert "FileNotFoundError" in result["error"]


def test_execute_tool_returns_a_containment_breach_as_an_observation(
    calculator_ws: Workspace,
):
    """Escaping the workspace is something the agent is told about, not a crash."""
    result = execute_tool(calculator_ws, "read_file", {"path": "../../etc/passwd"})
    assert result["ok"] is False
    assert "WorkspaceViolation" in result["error"]


def test_execute_tool_reports_a_declared_but_unwired_tool(calculator_ws: Workspace):
    """A tool added to the declarations and forgotten here must say so, not pass."""
    result = execute_tool(calculator_ws, "not_wired_up", {})
    assert result == {
        "ok": False,
        "error": "Tool 'not_wired_up' has no implementation.",
    }


# --------------------------------------------------------------------------
# Rendering observations — this is the text the agent actually reads
# --------------------------------------------------------------------------


def test_a_failed_observation_is_rendered_as_an_error():
    rendered = _render_observation("read_file", {"ok": False, "error": "nope"})
    assert rendered == "ERROR: nope"


def test_an_observation_that_failed_without_a_message_still_renders():
    assert _render_observation("read_file", {"ok": False}) == (
        "ERROR: the tool call failed."
    )


def test_an_empty_file_is_described_rather_than_rendered_as_nothing():
    assert _render_observation("read_file", {"ok": True, "content": ""}) == (
        "(the file is empty)"
    )


def test_empty_listings_and_searches_say_so():
    assert _render_observation("list_files", {"ok": True, "files": []}) == (
        "No files matched."
    )
    assert _render_observation("search_text", {"ok": True, "matches": []}) == (
        "No matches."
    )
    assert _render_observation("get_diff", {"ok": True, "diff": "   \n"}) == (
        "No changes have been made yet."
    )


def test_long_listings_are_capped_and_say_how_much_was_dropped():
    files = [f"src/module_{index}.py" for index in range(MAX_LISTED_FILES + 25)]
    rendered = _render_observation("list_files", {"ok": True, "files": files})
    assert rendered.count("\n") == MAX_LISTED_FILES
    assert "and 25 more files not shown" in rendered


def test_long_search_results_are_capped_and_say_how_much_was_dropped():
    matches = [
        {"file": "a.py", "line_number": index, "line_content": "x = 1"}
        for index in range(MAX_LISTED_MATCHES + 7)
    ]
    rendered = _render_observation("search_text", {"ok": True, "matches": matches})
    assert "a.py:0: x = 1" in rendered
    assert "and 7 more matches not shown" in rendered


def test_a_test_run_that_collected_nothing_is_called_out():
    rendered = _render_observation(
        "run_tests",
        {
            "ok": True,
            "success": False,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": 0,
            "timed_out": False,
            "raw_output": "no tests ran in 0.01s",
        },
    )
    assert "Tests FAILED" in rendered
    assert "No tests were collected" in rendered


def test_a_timed_out_test_run_is_called_out():
    rendered = _render_observation(
        "run_tests",
        {
            "ok": True,
            "success": False,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "total": 1,
            "timed_out": True,
            "raw_output": "",
        },
    )
    assert "timed out and was killed" in rendered


def test_only_the_tail_of_a_failing_test_run_is_shown():
    """pytest puts the summary last, and the whole log would crowd out the task."""
    output = "\n".join(f"log line {index}" for index in range(200)) + "\n1 failed in 1s"
    rendered = _render_observation(
        "run_tests",
        {
            "ok": True,
            "success": False,
            "passed": 0,
            "failed": 1,
            "errors": 0,
            "total": 1,
            "timed_out": False,
            "raw_output": output,
        },
    )
    assert "1 failed in 1s" in rendered
    assert "log line 199" in rendered
    assert "log line 0" not in rendered


def test_a_passing_test_run_does_not_replay_its_output():
    rendered = _render_observation(
        "run_tests",
        {
            "ok": True,
            "success": True,
            "passed": 2,
            "failed": 0,
            "errors": 0,
            "total": 2,
            "timed_out": False,
            "raw_output": "..                      [100%]\n2 passed in 0.05s",
        },
    )
    assert rendered == "Tests PASSED: 2 passed, 0 failed, 0 errors, 2 tests collected."


def test_truncate_leaves_short_text_alone_and_marks_long_text():
    assert _truncate("short", limit=10) == "short"
    trimmed = _truncate("x" * 30, limit=10)
    assert trimmed.startswith("x" * 10)
    assert "20 more characters were truncated" in trimmed


def test_an_over_long_observation_is_truncated_for_the_model_but_not_the_record(
    calculator_ws: Workspace,
):
    """The trajectory keeps everything; only the model's view is trimmed."""
    huge = "# padding\n" * 4000
    from agent.core.react_agent import MAX_OBSERVATION_CHARS

    client = _scripted_client(
        _reply("Writing something enormous.", (_call("write_file", "c1", path="big.py", content=huge),)),
        _reply("Reading it back.", (_call("read_file", "c2", path="big.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c3", success=False, summary="done"),)),
    )

    trajectory = run_react_agent(calculator_ws, "make a big file", client=client)

    assert trajectory.turns[1].tool_result["content"] == huge  # full record kept
    observation = _requests(client)[2]["messages"][-1]
    assert "characters were truncated" in observation["content"]
    assert len(observation["content"]) < len(huge)
    assert len(observation["content"]) <= MAX_OBSERVATION_CHARS + 200


# --------------------------------------------------------------------------
# Building the opening context
# --------------------------------------------------------------------------


def test_the_file_listing_reports_an_empty_repository(ws: Workspace):
    assert _initial_file_listing(ws) == "(the repository appears to be empty)"


def test_the_file_listing_is_capped(ws: Workspace):
    for index in range(MAX_INITIAL_LISTING + 3):
        (ws.root / f"file_{index:04d}.py").write_text("x = 1\n", encoding="utf-8")
    listing = _initial_file_listing(ws)
    assert "and 3 more files" in listing


def test_the_file_listing_degrades_instead_of_failing(ws: Workspace, monkeypatch):
    """Orientation is best-effort: a listing failure must not kill the run."""
    def explode(*args, **kwargs):
        raise OSError("the disk went away")

    monkeypatch.setattr("agent.core.react_agent.list_files", explode)
    assert "could not be produced" in _initial_file_listing(ws)
    assert "the disk went away" in _initial_file_listing(ws)


def test_the_final_diff_is_empty_rather_than_fatal_outside_a_git_repo(ws: Workspace):
    assert _safe_diff(ws) == ""


def test_a_run_in_a_non_git_workspace_still_produces_a_trajectory(ws: Workspace):
    """The agent's work is recorded even when there is no repository to diff."""
    (ws.root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    client = _scripted_client(
        _reply("Reading.", (_call("read_file", "c1", path="calc.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c2", success=False, summary="no repo"),)),
    )

    trajectory = run_react_agent(ws, "fix add()", client=client)

    assert trajectory.stop_reason is StopReason.AGENT_FINISHED
    assert trajectory.total_turns == 2
    assert trajectory.final_diff == ""


def test_a_run_without_a_client_needs_a_configured_key(calculator_ws: Workspace, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMConfigError):
        run_react_agent(calculator_ws, "fix add()")


# --------------------------------------------------------------------------
# The loop: stopping conditions
# --------------------------------------------------------------------------


def test_loop_reads_writes_tests_and_stops_the_moment_tests_pass(
    calculator_ws: Workspace,
):
    """The canonical run: read, fix, verify — and stop without spending a turn on finish."""
    client = _scripted_client(
        _reply(
            "I should read the file before changing it.",
            (_call("read_file", "c1", path="calculator/calc.py"),),
            tokens=100,
        ),
        _reply(
            "add() subtracts. Fixing the operator and nothing else.",
            (
                _call(
                    "write_file", "c2", path="calculator/calc.py", content=FIXED_CALC
                ),
            ),
            tokens=200,
        ),
        _reply(
            "Now verify the change.",
            (_call("run_tests", "c3", test_path="."),),
            tokens=300,
        ),
        # Scripted but must never be reached: the loop stops on a real pass.
        _reply("Done.", (_call(FINISH_TOOL, "c4", success=True, summary="fixed"),)),
    )

    trajectory = run_react_agent(
        calculator_ws, "fix add()", max_turns=8, client=client
    )

    assert [turn.tool_name for turn in trajectory.turns] == [
        "read_file",
        "write_file",
        "run_tests",
    ]
    assert trajectory.stop_reason is StopReason.TESTS_PASSED
    assert trajectory.final_success is True
    assert trajectory.total_turns == 3
    assert trajectory.total_tokens == 600
    assert trajectory.turns[0].reasoning.startswith("I should read the file")
    assert trajectory.turns[2].tool_result["passed"] == 2

    # The fix really landed, and the diff was really captured.
    assert read_file(calculator_ws, "calculator/calc.py") == FIXED_CALC
    assert "+    return a + b" in trajectory.final_diff

    # Three LLM calls, and the fourth scripted response was never asked for.
    assert len(_requests(client)) == 3
    assert _unused_responses(client) == 1


def test_loop_stops_when_the_agent_calls_finish(calculator_ws: Workspace):
    client = _scripted_client(
        _reply("Looking at the file.", (_call("read_file", "c1", path="calculator/calc.py"),)),
        _reply(
            "I cannot see how to fix this.",
            (
                _call(
                    FINISH_TOOL,
                    "c2",
                    success=False,
                    summary="Could not work out the intended behaviour.",
                ),
            ),
        ),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.stop_reason is StopReason.AGENT_FINISHED
    assert trajectory.total_turns == 2
    assert trajectory.turns[-1].tool_name == FINISH_TOOL
    assert trajectory.turns[-1].tool_args["success"] is False
    assert trajectory.final_success is False
    assert trajectory.final_diff == ""


def test_final_success_ignores_an_unverified_claim_of_success(calculator_ws: Workspace):
    """An agent that says it succeeded without running the tests has not succeeded.

    The claim is kept in the finishing turn, so M8 can measure how often the
    agent asserts a fix it never verified.
    """
    client = _scripted_client(
        _reply(
            "Obvious fix, no need to check.",
            (_call("write_file", "c1", path="calculator/calc.py", content=FIXED_CALC),),
        ),
        _reply(
            "That is correct, I am confident.",
            (_call(FINISH_TOOL, "c2", success=True, summary="Fixed the operator."),),
        ),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.stop_reason is StopReason.AGENT_FINISHED
    assert trajectory.turns[-1].tool_args["success"] is True  # the agent's claim
    assert trajectory.final_success is False  # what we actually verified
    assert "+    return a + b" in trajectory.final_diff


def test_a_write_does_not_trigger_an_automatic_test_run(calculator_ws: Workspace):
    """Verification is the agent's job — that is the behaviour M8 measures."""
    client = _scripted_client(
        _reply("Fixing.", (_call("write_file", "c1", path="calculator/calc.py", content=FIXED_CALC),)),
        _reply("Done.", (_call(FINISH_TOOL, "c2", success=True, summary="fixed"),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert [turn.tool_name for turn in trajectory.turns] == ["write_file", FINISH_TOOL]
    assert not any(turn.tool_name == "run_tests" for turn in trajectory.turns)


def test_the_loop_warns_the_agent_in_conversation_about_a_mangled_write(
    calculator_ws: Workspace,
):
    """The warning has to reach the model, not just the trajectory."""
    client = _scripted_client(
        _reply(
            "Writing the fix.",
            (
                _call(
                    "write_file",
                    "c1",
                    path="calculator/calc.py",
                    content=FIXED_CALC.replace("\n", "\\n"),
                ),
            ),
        ),
        _reply("Stopping.", (_call(FINISH_TOOL, "c2", success=False, summary="botched it"),)),
    )

    run_react_agent(calculator_ws, "fix add()", client=client)

    observation = _requests(client)[1]["messages"][-1]
    assert observation["role"] == "tool"
    assert "very likely corrupt" in observation["content"]


def test_loop_stops_at_max_turns_when_the_agent_never_finishes(calculator_ws: Workspace):
    client = _scripted_client(
        *[
            _reply("Reading again.", (_call("read_file", f"c{n}", path="calculator/calc.py"),))
            for n in range(1, 11)
        ]
    )

    trajectory = run_react_agent(
        calculator_ws, "fix add()", max_turns=3, client=client
    )

    assert trajectory.stop_reason is StopReason.MAX_TURNS_REACHED
    assert trajectory.total_turns == 3
    assert [turn.turn_number for turn in trajectory.turns] == [1, 2, 3]
    assert trajectory.final_success is False
    assert len(_requests(client)) == 3  # the budget is a budget on LLM calls


def test_a_failing_test_run_does_not_stop_the_loop(calculator_ws: Workspace):
    """Only a *passing* run auto-stops; a failing one is just an observation."""
    client = _scripted_client(
        _reply("Check the damage first.", (_call("run_tests", "c1", test_path="."),)),
        _reply("Now fix it.", (_call("write_file", "c2", path="calculator/calc.py", content=FIXED_CALC),)),
        _reply("Verify.", (_call("run_tests", "c3", test_path="."),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.turns[0].tool_result["success"] is False
    assert trajectory.turns[0].tool_result["failed"] == 1
    assert trajectory.stop_reason is StopReason.TESTS_PASSED
    assert trajectory.final_success is True
    assert trajectory.total_turns == 3


def test_max_turns_below_one_is_rejected(calculator_ws: Workspace):
    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        run_react_agent(calculator_ws, "fix add()", max_turns=0, client=_scripted_client())


# --------------------------------------------------------------------------
# The loop: recovering from bad responses
# --------------------------------------------------------------------------


def test_a_response_with_no_tool_call_is_recorded_and_retried(calculator_ws: Workspace):
    client = _scripted_client(
        _reply("The bug is that add() uses a minus sign.", (), finish_reason="stop"),
        _reply("Sorry — acting now.", (_call("write_file", "c2", path="calculator/calc.py", content=FIXED_CALC),)),
        _reply("Verify.", (_call("run_tests", "c3", test_path="."),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    bad_turn = trajectory.turns[0]
    assert bad_turn.tool_name is None
    assert bad_turn.tool_result["ok"] is False
    assert "no tool call" in bad_turn.tool_result["error"]
    assert bad_turn.reasoning == "The bug is that add() uses a minus sign."
    assert bad_turn.tokens_used == 100  # a wasted turn still costs tokens

    # The run recovered and went on to succeed.
    assert trajectory.stop_reason is StopReason.TESTS_PASSED
    assert trajectory.final_success is True
    assert trajectory.total_turns == 3

    # The model was told, in the conversation, that nothing happened.
    nudge = _requests(client)[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "no tool call" in nudge["content"]


def test_malformed_tool_arguments_become_an_observation_and_the_agent_retries(
    calculator_ws: Workspace,
):
    client = _scripted_client(
        _reply("Reading.", (_raw_call("read_file", '{"path": ', "c1"),)),
        _reply("Retrying with valid JSON.", (_call("read_file", "c2", path="calculator/calc.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c3", success=False, summary="ran out of ideas"),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.turns[0].tool_name == "read_file"
    assert trajectory.turns[0].tool_args == {}
    assert trajectory.turns[0].tool_result["ok"] is False
    assert "not valid JSON" in trajectory.turns[0].tool_result["error"]
    assert trajectory.turns[1].tool_result["ok"] is True
    assert trajectory.stop_reason is StopReason.AGENT_FINISHED

    # The failure was reported back through the tool channel, keyed to the call.
    observation = _requests(client)[1]["messages"][-1]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "c1"
    assert observation["content"].startswith("ERROR:")


def test_a_call_to_a_tool_that_does_not_exist_is_refused_as_an_observation(
    calculator_ws: Workspace,
):
    """An agent reaching for the shell is told no, and keeps working."""
    client = _scripted_client(
        _reply("I will just use the shell.", (_call("run_command", "c1", command="pytest"),)),
        _reply("Fine, using run_tests.", (_call("run_tests", "c2", test_path="."),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c3", success=False, summary="tests fail"),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.turns[0].tool_result["ok"] is False
    assert "no tool called 'run_command'" in trajectory.turns[0].tool_result["error"]
    assert trajectory.total_turns == 3
    assert trajectory.stop_reason is StopReason.AGENT_FINISHED


def test_a_failing_tool_call_is_an_observation_not_a_crash(calculator_ws: Workspace):
    client = _scripted_client(
        _reply("Reading the wrong path.", (_call("read_file", "c1", path="src/calc.py"),)),
        _reply("Listing instead.", (_call("list_files", "c2", subdir=".", pattern="*.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c3", success=False, summary="out of turns"),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.turns[0].tool_result["ok"] is False
    assert "FileNotFoundError" in trajectory.turns[0].tool_result["error"]
    assert trajectory.turns[1].tool_result["ok"] is True
    assert trajectory.stop_reason is StopReason.AGENT_FINISHED


def test_only_the_first_of_several_tool_calls_in_one_turn_is_executed(
    calculator_ws: Workspace,
):
    client = _scripted_client(
        _reply(
            "Doing everything at once.",
            (
                _call("read_file", "c1", path="calculator/calc.py"),
                _call("write_file", "c2", path="calculator/calc.py", content="broken"),
            ),
        ),
        _reply("One at a time, then.", (_call(FINISH_TOOL, "c3", success=False, summary="stopping"),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.turns[0].tool_name == "read_file"
    assert trajectory.turns[0].tool_result["extra_calls_ignored"] == 1
    # The dropped write never happened.
    assert "return a - b" in read_file(calculator_ws, "calculator/calc.py")

    # History stays wire-valid: one recorded tool_call, one matching answer.
    messages = _requests(client)[1]["messages"]
    assistant, observation = messages[-2], messages[-1]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1"]
    assert observation["tool_call_id"] == "c1"
    assert "only the first was executed" in observation["content"]


def _tool_use_failed(
    generation: str = '<function=read_file{"path": "calc.py"}',
) -> Exception:
    """A real Groq 400 for a garbled tool call, raised through the openai SDK.

    Groq validates tool calls server-side, so a model that garbles one does not
    produce an odd assistant message — the whole request fails. This is built by
    handing the SDK an actual 400 response so the exception carries the SDK's own
    ``body`` shape rather than our guess at it: that guess was wrong once already,
    and only a live run caught it. If a future SDK stops unwrapping the provider's
    ``{"error": ...}`` envelope, the tests below fail instead of the agent.
    """
    payload = {
        "error": {
            "message": "Failed to call a function. Please adjust your prompt.",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": generation,
        }
    }
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, json=payload)
    client = OpenAI(api_key="not-a-real-key", base_url=llm_client.DEFAULT_BASE_URL)
    return client._make_status_error_from_response(response)


def test_a_real_sdk_tool_use_failure_is_recognised():
    error = _tool_use_failed("<function=oops")
    assert isinstance(error, BadRequestError)
    assert llm_client.malformed_tool_call(error) == "<function=oops"


@pytest.mark.parametrize(
    "body",
    [
        None,
        "not a dict at all",
        {"error": "not a dict either"},
        {"error": {"code": "rate_limit_exceeded"}},
        {"code": "rate_limit_exceeded"},
        {"message": "you are out of quota"},
    ],
)
def test_malformed_tool_call_does_not_claim_unrelated_failures(body):
    error = RuntimeError("boom")
    error.body = body
    assert llm_client.malformed_tool_call(error) is None


@pytest.mark.parametrize(
    "body",
    [
        # What the SDK actually hands us: the envelope already unwrapped.
        {"code": "tool_use_failed", "failed_generation": "<function=oops"},
        # And the raw provider shape, in case that ever changes.
        {"error": {"code": "tool_use_failed", "failed_generation": "<function=oops"}},
    ],
)
def test_malformed_tool_call_accepts_either_body_shape(body):
    error = RuntimeError("boom")
    error.body = body
    assert llm_client.malformed_tool_call(error) == "<function=oops"


def test_malformed_tool_call_distinguishes_absent_text_from_a_different_error():
    """``""`` means "malformed, text unavailable" — callers must test ``is not None``."""
    error = RuntimeError("boom")
    error.body = {"code": "tool_use_failed"}
    assert llm_client.malformed_tool_call(error) == ""


def test_a_provider_rejected_tool_call_is_retried_not_fatal(calculator_ws: Workspace):
    """A model that garbles its tool call gets another turn, not a dead run."""
    client = _scripted_client(
        _tool_use_failed('I will search for it.\n<function=search_text{"query": "def add"}'),
        _reply("Retrying properly.", (_call("write_file", "c2", path="calculator/calc.py", content=FIXED_CALC),)),
        _reply("Verify.", (_call("run_tests", "c3", test_path="."),)),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    rejected = trajectory.turns[0]
    assert rejected.tool_name is None
    assert rejected.tool_result["ok"] is False
    assert "malformed" in rejected.tool_result["error"]
    assert "<function=search_text" in rejected.tool_result["failed_generation"]
    assert "I will search for it." in rejected.reasoning

    assert trajectory.stop_reason is StopReason.TESTS_PASSED
    assert trajectory.final_success is True
    assert trajectory.total_turns == 3

    # The model was told to retry, and was NOT fed its own broken syntax back.
    messages = _requests(client)[1]["messages"]
    assert messages[-1]["role"] == "user"
    assert "malformed" in messages[-1]["content"]
    assert not any("<function=" in message.get("content", "") for message in messages)


def test_a_failed_llm_call_ends_the_run_as_an_error(calculator_ws: Workspace):
    client = _scripted_client(
        _reply("Reading.", (_call("read_file", "c1", path="calculator/calc.py"),)),
        RuntimeError("connection reset by peer"),
    )

    trajectory = run_react_agent(calculator_ws, "fix add()", client=client)

    assert trajectory.stop_reason is StopReason.ERROR
    assert trajectory.total_turns == 2
    failed = trajectory.turns[-1]
    assert failed.tool_name is None
    assert "connection reset by peer" in failed.tool_result["error"]
    assert failed.tokens_used == 0
    assert trajectory.final_success is False


# --------------------------------------------------------------------------
# The loop: what the model is actually sent
# --------------------------------------------------------------------------


def test_the_first_request_carries_the_prompts_the_tools_and_the_file_listing(
    calculator_ws: Workspace,
):
    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="nothing to do"),))
    )

    run_react_agent(calculator_ws, "fix add() so it sums", client=client)

    request = _requests(client)[0]
    system, user = request["messages"]
    assert system["role"] == "system"
    assert "one tool action per turn" in system["content"].lower()
    assert user["role"] == "user"
    assert "fix add() so it sums" in user["content"]
    assert "calculator/calc.py" in user["content"]

    assert {schema["function"]["name"] for schema in request["tools"]} == set(TOOLS)
    assert request["temperature"] == 0.0


def test_observations_accumulate_in_the_conversation(calculator_ws: Workspace):
    """Turn N sees every earlier action and result — that is what makes it ReAct."""
    client = _scripted_client(
        _reply("Read.", (_call("read_file", "c1", path="calculator/calc.py"),)),
        _reply("Search.", (_call("search_text", "c2", query="def add", subdir=".", pattern="*.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c3", success=False, summary="done looking"),)),
    )

    run_react_agent(calculator_ws, "fix add()", client=client)

    third = _requests(client)[2]["messages"]
    roles = [message["role"] for message in third]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert "return a - b" in third[3]["content"]  # the file the agent read
    assert "calculator/calc.py:8" in third[5]["content"]  # the search result


def test_on_turn_is_called_live_as_each_turn_completes(calculator_ws: Workspace):
    client = _scripted_client(
        _reply("Read.", (_call("read_file", "c1", path="calculator/calc.py"),)),
        _reply("Stopping.", (_call(FINISH_TOOL, "c2", success=False, summary="done"),)),
    )

    seen: list[tuple[int, str | None]] = []
    trajectory = run_react_agent(
        calculator_ws,
        "fix add()",
        client=client,
        on_turn=lambda turn: seen.append((turn.turn_number, turn.tool_name)),
    )

    assert seen == [(1, "read_file"), (2, FINISH_TOOL)]
    assert len(seen) == trajectory.total_turns


# --------------------------------------------------------------------------
# CLI rendering
#
# These do not test how it looks; they test that every tool's result shape can
# actually be rendered. The summariser indexes result dicts by key, so a tool
# that changes its keys would otherwise blow up in the CLI and nowhere else.
# --------------------------------------------------------------------------


@pytest.fixture
def captured_console(monkeypatch):
    """Swap the module console for one writing to a buffer, and return the buffer."""
    from rich.console import Console

    buffer = io.StringIO()
    monkeypatch.setattr(
        "agent.core.react_agent.console", Console(file=buffer, width=120, no_color=True)
    )
    return buffer


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "tool_result", "expected"),
    [
        ("read_file", {"path": "calc.py"}, {"ok": True, "content": "x" * 42}, "42 chars"),
        (
            "write_file",
            {"path": "calc.py", "content": "x" * 300},
            {"ok": True, "path": "calc.py", "chars_written": 300, "lines_written": 1},
            "wrote 300 chars to calc.py",
        ),
        ("list_files", {"subdir": "."}, {"ok": True, "files": ["a.py"], "count": 1}, "1 files"),
        ("search_text", {"query": "def"}, {"ok": True, "matches": [], "count": 0}, "0 matches"),
        (
            "run_tests",
            {"test_path": "."},
            {
                "ok": True,
                "success": True,
                "passed": 2,
                "failed": 0,
                "errors": 0,
                "total": 2,
                "timed_out": False,
                "raw_output": "",
            },
            "tests passed",
        ),
        (
            "run_tests",
            {"test_path": "."},
            {
                "ok": True,
                "success": False,
                "passed": 1,
                "failed": 1,
                "errors": 0,
                "total": 2,
                "timed_out": False,
                "raw_output": "",
            },
            "tests failed",
        ),
        ("get_diff", {}, {"ok": True, "diff": "line one\nline two\n"}, "2 lines"),
        (
            FINISH_TOOL,
            {"success": True, "summary": "Changed the operator."},
            {"ok": True, "success": True, "summary": "Changed the operator."},
            "claims success",
        ),
        (
            FINISH_TOOL,
            {"success": False, "summary": "Out of ideas."},
            {"ok": False, "success": False, "summary": "Out of ideas."},
            "error",
        ),
        ("read_file", {"path": "gone.py"}, {"ok": False, "error": "FileNotFoundError"}, "error"),
    ],
)
def test_every_tool_result_shape_can_be_printed(
    captured_console, tool_name, tool_args, tool_result, expected
):
    from agent.core.react_agent import _print_turn

    _print_turn(
        Turn(
            turn_number=1,
            reasoning="Doing the thing.",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tokens_used=100,
        )
    )
    printed = captured_console.getvalue()
    assert "turn 1" in printed
    assert "Doing the thing." in printed
    assert expected in printed


def test_printing_a_turn_summarises_long_arguments_instead_of_dumping_them(
    captured_console,
):
    from agent.core.react_agent import _print_turn

    _print_turn(
        Turn(
            turn_number=2,
            tool_name="write_file",
            tool_args={"path": "calc.py", "content": "sentinel\n" * 100},
            tool_result={
                "ok": True,
                "path": "calc.py",
                "chars_written": 900,
                "lines_written": 100,
            },
        )
    )
    printed = captured_console.getvalue()
    assert "content=<900 chars>" in printed
    assert "sentinel" not in printed


def test_printing_a_turn_that_took_no_action(captured_console):
    from agent.core.react_agent import _print_turn

    _print_turn(
        Turn(turn_number=3, tool_result={"ok": False, "error": "no tool call was made"})
    )
    printed = captured_console.getvalue()
    assert "no action" in printed
    assert "no tool call was made" in printed


@pytest.mark.parametrize("show_diff", [True, False])
def test_the_final_panel_reports_the_outcome(captured_console, show_diff):
    from agent.core.react_agent import _print_final

    _print_final(
        Trajectory(
            task_description="fix add()",
            final_success=True,
            final_diff="-    return a - b\n+    return a + b\n",
            total_turns=4,
            total_tokens=9725,
            stop_reason=StopReason.TESTS_PASSED,
        ),
        show_diff=show_diff,
    )
    printed = captured_console.getvalue()
    assert "tests_passed" in printed
    assert "9,725" in printed
    assert ("return a + b" in printed) is show_diff


def test_the_final_panel_says_so_when_nothing_changed(captured_console):
    from agent.core.react_agent import _print_final

    _print_final(
        Trajectory(
            task_description="fix add()", stop_reason=StopReason.MAX_TURNS_REACHED
        )
    )
    printed = captured_console.getvalue()
    assert "no changes were made" in printed
    assert "max_turns_reached" in printed


# --------------------------------------------------------------------------
# M7 memory wiring (agent.memory) — optional, off by default
#
# find_similar/save_trajectory are monkeypatched here rather than exercised
# for real: their own ranking/persistence behaviour belongs to
# tests/test_trajectory_store.py, and a real call would need a real (or
# fakeredis) client plus the embedding model. These tests are only about
# run_react_agent's wiring: does it call the right thing, with the right
# arguments, only when memory_client is given.
# --------------------------------------------------------------------------


def _past_trajectory(
    task_description: str = "fix add() so it returns the sum",
    final_success: bool = True,
    summary: str = "changed the operator in add() from - to +",
) -> Trajectory:
    """A finished trajectory, as if loaded back out of Redis."""
    trajectory = Trajectory(
        task_description=task_description,
        final_success=final_success,
        stop_reason=StopReason.TESTS_PASSED,
        total_turns=2,
        total_tokens=250,
    )
    trajectory.turns = [
        Turn(turn_number=1, tool_name="write_file", tool_result={"ok": True}),
        Turn(
            turn_number=2,
            tool_name=FINISH_TOOL,
            tool_args={"success": final_success, "summary": summary},
            tool_result={"ok": True},
        ),
    ]
    return trajectory


def test_memory_client_none_never_touches_memory(calculator_ws, monkeypatch):
    """The default (no memory_client) must not call find_similar or save_trajectory."""
    calls: list[str] = []
    monkeypatch.setattr(
        react_agent_module, "find_similar", lambda *a, **k: calls.append("find") or []
    )
    monkeypatch.setattr(
        react_agent_module, "save_trajectory", lambda *a, **k: calls.append("save")
    )

    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="n/a"),))
    )
    run_react_agent(calculator_ws, "fix add()", client=client)

    assert calls == []


def test_a_similar_past_trajectory_above_threshold_is_injected(calculator_ws, monkeypatch):
    past = _past_trajectory()
    monkeypatch.setattr(
        react_agent_module, "find_similar", lambda *a, **k: [(past, 0.91)]
    )
    monkeypatch.setattr(react_agent_module, "save_trajectory", lambda *a, **k: None)

    seen_matches: list[tuple] = []
    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="n/a"),))
    )
    run_react_agent(
        calculator_ws,
        "fix add()",
        client=client,
        memory_client=object(),
        on_memory_match=lambda traj, sim: seen_matches.append((traj, sim)),
    )

    system, memory_system, user = _requests(client)[0]["messages"]
    assert system["role"] == "system"
    assert memory_system["role"] == "system"
    assert "similar past task was attempted before" in memory_system["content"]
    assert "0.91" in memory_system["content"]
    assert "changed the operator in add()" in memory_system["content"]
    assert user["role"] == "user"

    assert seen_matches == [(past, 0.91)]


def test_a_match_below_threshold_is_not_injected(calculator_ws, monkeypatch):
    below = MEMORY_SIMILARITY_THRESHOLD - 0.01
    monkeypatch.setattr(
        react_agent_module, "find_similar", lambda *a, **k: [(_past_trajectory(), below)]
    )
    monkeypatch.setattr(react_agent_module, "save_trajectory", lambda *a, **k: None)

    calls: list[tuple] = []
    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="n/a"),))
    )
    run_react_agent(
        calculator_ws,
        "fix add()",
        client=client,
        memory_client=object(),
        on_memory_match=lambda traj, sim: calls.append((traj, sim)),
    )

    assert len(_requests(client)[0]["messages"]) == 2  # system, user — no memory message
    assert calls == []


def test_no_stored_trajectories_injects_nothing(calculator_ws, monkeypatch):
    monkeypatch.setattr(react_agent_module, "find_similar", lambda *a, **k: [])
    monkeypatch.setattr(react_agent_module, "save_trajectory", lambda *a, **k: None)

    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="n/a"),))
    )
    run_react_agent(calculator_ws, "fix add()", client=client, memory_client=object())

    assert len(_requests(client)[0]["messages"]) == 2


def test_the_finished_trajectory_is_saved_when_memory_client_is_given(
    calculator_ws, monkeypatch
):
    monkeypatch.setattr(react_agent_module, "find_similar", lambda *a, **k: [])
    saved: list = []
    monkeypatch.setattr(
        react_agent_module,
        "save_trajectory",
        lambda client, trajectory, turn_scores=None: saved.append(
            (client, trajectory, turn_scores)
        ),
    )

    sentinel_client = object()
    client = _scripted_client(
        _reply("Stopping.", (_call(FINISH_TOOL, "c1", success=False, summary="n/a"),))
    )
    trajectory = run_react_agent(
        calculator_ws, "fix add()", client=client, memory_client=sentinel_client
    )

    assert len(saved) == 1
    saved_client, saved_trajectory, turn_scores = saved[0]
    assert saved_client is sentinel_client
    assert saved_trajectory is trajectory
    assert turn_scores is not None
    assert len(turn_scores) == trajectory.total_turns


def test_memory_reference_reports_outcome_and_the_past_runs_own_summary():
    text = _memory_reference(_past_trajectory(final_success=True), 0.83)
    assert "0.83" in text
    assert "succeeded" in text
    assert "changed the operator in add()" in text
    assert "background from a different run" in text


def test_memory_reference_says_so_when_the_past_run_never_finished():
    past = _past_trajectory()
    past.turns = [Turn(turn_number=1, tool_name="read_file", tool_result={"ok": True})]
    text = _memory_reference(past, 0.8)
    assert "What was done" not in text


# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------


def test_react_templates_render_completely():
    system_prompt, user_prompt = render_react_prompts(
        task_description="fix add() so it returns the sum",
        file_listing="calculator/calc.py\ncalculator/test_calc.py",
    )
    assert system_prompt.strip()
    assert "fix add() so it returns the sum" in user_prompt
    assert "calculator/test_calc.py" in user_prompt
    assert placeholders(user_prompt) == []


def test_react_user_template_declares_the_expected_placeholders():
    template = load_prompt("react_user_template.md")
    assert set(placeholders(template)) == {"task_description", "file_listing"}


def test_react_system_prompt_documents_every_tool_and_the_key_rules():
    system_prompt = load_prompt("react_system.md")
    for name in TOOLS:
        assert name in system_prompt, f"{name} is not described to the agent"
    lowered = system_prompt.lower()
    assert "smallest change" in lowered
    assert "do not refactor unrelated code" in lowered
    assert "re-run the tests after changing a file" in lowered


def test_react_system_prompt_declares_that_there_is_no_shell():
    """The scoping constraint has to reach the model, not just the code."""
    assert "cannot run arbitrary shell commands" in load_prompt("react_system.md")


# --------------------------------------------------------------------------
# integration: one real multi-turn run against Groq
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not llm_client.api_key_present(),
    reason="GROQ_API_KEY is not configured; skipping the live Groq run",
)
def test_react_agent_fixes_the_calculator_end_to_end(calculator_ws: Workspace, capsys):
    """A real multi-turn run against the buggy fixture.

    Unlike the baseline's integration test, this one *does* assert success: a
    one-operator bug with a test suite that names the expected behaviour is the
    floor of what an iterating agent should manage.
    """
    trajectory = run_react_agent(
        calculator_ws,
        "fix add() so it returns the correct sum instead of subtracting",
        max_turns=8,
    )

    with capsys.disabled():
        print(f"\n{'=' * 78}")
        print(f"TASK: {trajectory.task_description}")
        print("=" * 78)
        for turn in trajectory.turns:
            print(f"\n--- turn {turn.turn_number} " + "-" * 60)
            print(f"reasoning : {turn.reasoning or '(none)'}")
            print(f"tool      : {turn.tool_name or '(no tool call)'}")
            print(f"args      : {_readable_args(turn.tool_args)}")
            print(f"result    : {_readable_result(turn.tool_result)}")
            print(f"tokens    : {turn.tokens_used:,}")
        print(f"\n{'=' * 78}")
        print("FINAL DIFF")
        print("=" * 78)
        print(trajectory.final_diff or "(empty)")
        print("=" * 78)
        print(f"stop_reason  : {trajectory.stop_reason.value}")
        print(f"final_success: {trajectory.final_success}")
        print(f"total_turns  : {trajectory.total_turns}")
        print(f"total_tokens : {trajectory.total_tokens:,}")
        print("=" * 78)

    assert trajectory.total_turns >= 1
    assert trajectory.total_tokens > 0
    assert trajectory.stop_reason is not StopReason.ERROR

    # It really iterated: it used tools, and at least one was a real action.
    used = [turn.tool_name for turn in trajectory.turns]
    assert "write_file" in used, f"the agent never edited anything: {used}"
    assert "run_tests" in used, f"the agent never verified its work: {used}"

    # And it really worked.
    assert trajectory.final_success is True, (
        f"stopped as {trajectory.stop_reason.value} without passing tests"
    )
    assert "+    return a + b" in trajectory.final_diff
    assert "return a + b" in read_file(calculator_ws, "calculator/calc.py")


def _readable_args(arguments: dict) -> str:
    """Render args for the printed trajectory, summarising long file contents."""
    if not arguments:
        return "{}"
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 80:
            parts.append(f"{key}=<{len(value)} chars>")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _readable_result(result: dict) -> str:
    """Render one tool result for the printed trajectory, without dumping files."""
    if not result.get("ok", False):
        return f"ERROR {result.get('error', '')}"
    if "content" in result:
        return f"ok, {len(result['content'])} chars read"
    if "chars_written" in result:
        return f"ok, {result['chars_written']} chars written to {result['path']}"
    if "files" in result:
        return f"ok, {result['count']} files"
    if "matches" in result:
        return f"ok, {result['count']} matches"
    if "diff" in result:
        return f"ok, diff of {len(result['diff'].splitlines())} lines"
    if "passed" in result:
        return (
            f"tests {'PASSED' if result['success'] else 'FAILED'} — "
            f"{result['passed']} passed, {result['failed']} failed, "
            f"{result['errors']} errors"
        )
    if "summary" in result:
        return f"finish(success={result['success']}) — {result['summary']}"
    return str(result)
