"""The multi-turn ReAct agent: act, observe, adjust, repeat.

Where the baseline (:mod:`agent.core.baseline`) guesses once and stops, this loop
reasons its way to a fix: each turn the model sees everything that has happened
so far and chooses ONE tool action, we execute it, and the result goes back into
the conversation as an observation. It ends when the agent says it is done, when
the test suite actually passes, or when it runs out of turns.

SCOPING CONSTRAINT (M3)
=======================
The agent's toolset is deliberately limited to the six workspace-scoped tools
declared in :data:`TOOLS` — ``read_file``, ``write_file``, ``list_files``,
``search_text``, ``run_tests``, ``get_diff`` — plus the ``finish`` pseudo-tool.

``agent.tools.shell_exec.run_command`` is NOT exposed and must not be added
here. The agent cannot run arbitrary shell commands in this milestone; the only
thing it can execute is pytest, via ``run_tests``. Raw shell access, Docker
sandboxing (M5), and the safety/refusal confirmation gate (M6) are all out of
scope until those milestones land. Until then the *only* thing standing between
the agent and the host filesystem is :class:`~agent.tools.workspace.Workspace`
containment, which confines file reads and writes but cannot confine a child
process — so no tool that spawns one belongs in this list yet.

Run it directly against a repo::

    python -m agent.core.react_agent \\
        --repo demo/sample_repos/calculator \\
        --task "fix add() so it returns the correct sum instead of subtracting" \\
        --max-turns 8
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Callable, Optional

import typer
from agent_framework import FunctionTool
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..tools import (
    Workspace,
    get_diff,
    list_files,
    read_file,
    run_tests,
    search_text,
    write_file,
)
from . import llm_client
from .prompts import render_react_prompts

__all__ = [
    "FINISH_TOOL",
    "TOOLS",
    "StopReason",
    "Trajectory",
    "Turn",
    "execute_tool",
    "parse_tool_arguments",
    "run_react_agent",
    "tool_schemas",
]

#: Name of the pseudo-tool the agent calls to stop. It is a tool only so the
#: model can request it through the same structured channel as everything else;
#: nothing is executed against the workspace when it arrives.
FINISH_TOOL = "finish"

#: How much of a rendered observation the model is shown. Long pytest tracebacks
#: and large files would otherwise crowd out the rest of the conversation. The
#: *full* result is always kept in the trajectory — only the model's view is
#: trimmed, so the eval harness never loses data.
MAX_OBSERVATION_CHARS = 8000

#: Caps on how many list-shaped results are rendered into an observation.
MAX_LISTED_FILES = 200
MAX_LISTED_MATCHES = 50

#: Lines of pytest output shown to the agent. The tail is what matters: pytest's
#: summary line and the last failure land there.
MAX_TEST_OUTPUT_LINES = 40

#: Files listed in the opening user message.
MAX_INITIAL_LISTING = 200


# --------------------------------------------------------------------------
# Tool declarations
#
# Argument shapes are Pydantic models, and the tools are Microsoft Agent
# Framework `FunctionTool`s built declaration-only (`func=None`) — the framework
# supports exactly this case for "the implementation lives elsewhere". It has to
# live elsewhere here: every real tool takes a `Workspace` as its first
# argument, and the workspace is a per-run value the model must never see or
# name. So the framework owns the *declaration* (and, via
# `to_json_schema_spec()`, the wire format), while `execute_tool` below owns
# dispatch and injects the workspace.
# --------------------------------------------------------------------------


class ReadFileArgs(BaseModel):
    path: Annotated[
        str, Field(description="File to read, relative to the repository root.")
    ]


class WriteFileArgs(BaseModel):
    path: Annotated[
        str, Field(description="File to write, relative to the repository root.")
    ]
    content: Annotated[
        str,
        Field(
            description=(
                "The complete new contents of the file. This REPLACES the whole "
                "file, so include every line you want to keep, not just your "
                "change. Line breaks must be real newlines in the string; a "
                "literal backslash followed by 'n' is written to the file as "
                "those two characters and will corrupt the file."
            )
        ),
    ]


class ListFilesArgs(BaseModel):
    subdir: Annotated[
        str,
        Field(description="Directory to list, relative to the repository root."),
    ] = "."
    pattern: Annotated[
        str,
        Field(description='Glob matched against file names, e.g. "*.py".'),
    ] = "*"


class SearchTextArgs(BaseModel):
    query: Annotated[
        str,
        Field(description="Literal substring to search for (not a regular expression)."),
    ]
    subdir: Annotated[
        str, Field(description="Directory to search, relative to the repository root.")
    ] = "."
    pattern: Annotated[
        str, Field(description='Glob restricting which files are searched, e.g. "*.py".')
    ] = "*"


class RunTestsArgs(BaseModel):
    test_path: Annotated[
        str,
        Field(
            description=(
                'File or directory to test, relative to the repository root. "." '
                "runs the whole suite."
            )
        ),
    ] = "."


class GetDiffArgs(BaseModel):
    """No arguments: the diff is always the whole working tree."""


class FinishArgs(BaseModel):
    success: Annotated[
        bool,
        Field(
            description=(
                "Your own judgment: true if you believe the task is complete and "
                "the tests pass, false if you are stopping without fixing it."
            )
        ),
    ]
    summary: Annotated[
        str,
        Field(
            description=(
                "One or two sentences: what you changed and what the tests said."
            )
        ),
    ]


def _declare(name: str, description: str, args_model: type[BaseModel]) -> FunctionTool:
    """Declare a tool for the model without giving the framework an implementation."""
    return FunctionTool(
        name=name, description=description, func=None, input_model=args_model
    )


#: The agent's entire toolset, in the order it is advertised to the model. See
#: the SCOPING CONSTRAINT in the module docstring before adding anything.
TOOLS: dict[str, FunctionTool] = {
    tool.name: tool
    for tool in (
        _declare(
            "read_file",
            "Read a file and return its full text. Read a file before editing it.",
            ReadFileArgs,
        ),
        _declare(
            "write_file",
            "Replace a file's entire contents. There is no partial edit: send the "
            "whole file back, with only your fix changed.",
            WriteFileArgs,
        ),
        _declare(
            "list_files",
            "List the files under a directory, recursively, optionally filtered by "
            "a glob pattern.",
            ListFilesArgs,
        ),
        _declare(
            "search_text",
            "Find every line containing a literal substring, with its file and "
            "line number.",
            SearchTextArgs,
        ),
        _declare(
            "run_tests",
            "Run pytest and return the pass/fail/error counts and output. This is "
            "the only way to find out whether the code works.",
            RunTestsArgs,
        ),
        _declare(
            "get_diff",
            "Show every change made to the repository so far, as a unified diff.",
            GetDiffArgs,
        ),
        _declare(
            FINISH_TOOL,
            "Stop working and report the outcome. Call this when the task is done, "
            "or when you are stuck and more turns would not help.",
            FinishArgs,
        ),
    )
}


def tool_schemas() -> list[dict]:
    """Return every tool as an OpenAI-compatible function specification.

    ``FunctionTool.to_json_schema_spec()`` already emits the
    ``{"type": "function", "function": {...}}`` shape the OpenAI wire protocol —
    and therefore Groq — expects, so no hand-written schema is involved.
    """
    return [tool.to_json_schema_spec() for tool in TOOLS.values()]


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


class StopReason(str, Enum):
    """Why a run ended."""

    #: ``run_tests`` reported success, so the loop stopped without waiting for
    #: the agent to notice.
    TESTS_PASSED = "tests_passed"
    #: The agent called ``finish``.
    AGENT_FINISHED = "agent_finished"
    #: The turn budget ran out with the task unresolved.
    MAX_TURNS_REACHED = "max_turns_reached"
    #: The run itself broke — the LLM call failed. Tool failures are *not* this;
    #: they are observations the agent gets to react to.
    ERROR = "error"


class Turn(BaseModel):
    """One act–observe cycle: what the agent thought, did, and saw."""

    turn_number: int
    #: The model's text alongside its tool call. Often terse, sometimes empty —
    #: a model answering purely with a tool call is normal, not an error.
    reasoning: str = ""
    #: ``None`` when the model produced no usable tool call at all.
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    #: The observation, always including ``ok``. Stored in full, untruncated.
    tool_result: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0


class Trajectory(BaseModel):
    """The complete record of a run — M8's unit of measurement.

    ``final_success`` is deliberately *not* the agent's self-assessment. It is
    True only if a ``run_tests`` observation actually reported success during the
    run. An agent that calls ``finish(success=True)`` without ever having seen
    the tests pass produces ``final_success=False``, and the gap between the two
    is itself worth measuring — the agent's own claim is preserved in the
    finishing turn's ``tool_args``.
    """

    task_description: str
    turns: list[Turn] = Field(default_factory=list)
    final_success: bool = False
    final_diff: str = ""
    total_turns: int = 0
    total_tokens: int = 0
    stop_reason: StopReason = StopReason.ERROR


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------


def parse_tool_arguments(tool_name: str, raw_arguments: str) -> tuple[dict, Optional[str]]:
    """Validate a model-produced tool call into usable keyword arguments.

    Args:
        tool_name: The name the model asked for.
        raw_arguments: The raw JSON string it supplied.

    Returns:
        ``(arguments, error)``. On success ``error`` is ``None`` and
        ``arguments`` has defaults filled in from the tool's schema. On failure
        ``arguments`` is empty and ``error`` is a message written for the model
        to read and correct — an unknown tool name, malformed JSON, and
        arguments that do not match the schema are all recoverable mistakes,
        so none of them raise.
    """
    tool = TOOLS.get(tool_name)
    if tool is None:
        return {}, (
            f"There is no tool called {tool_name!r}. The tools you can call are: "
            f"{', '.join(TOOLS)}."
        )

    try:
        payload = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return {}, (
            f"The arguments for {tool_name!r} were not valid JSON ({exc}). "
            "Send the arguments again as a well-formed JSON object."
        )

    if not isinstance(payload, dict):
        return {}, (
            f"The arguments for {tool_name!r} must be a JSON object, but a "
            f"{type(payload).__name__} was sent."
        )

    model = tool.input_model
    if model is None:  # pragma: no cover - every tool here declares a model
        return payload, None

    try:
        return model(**payload).model_dump(), None
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()
        )
        return {}, (
            f"The arguments for {tool_name!r} do not match its schema: {problems}."
        )


def _looks_escape_mangled(content: str) -> bool:
    r"""True if ``content`` looks like a source file whose newlines got escaped.

    A recurring failure of tool-calling models is double-escaping the ``content``
    argument, so the JSON carries ``\\n`` and the file is written as one long line
    with a literal backslash and ``n`` wherever a line break belonged. It is not
    our place to silently rewrite what the agent asked to write, but it *is* worth
    telling the agent, because otherwise it sees "wrote 400 characters", assumes
    success, and spends the rest of its turns confused by a syntax error.

    The test is deliberately narrow — a substantial file containing the two-character
    sequence and essentially no real line breaks. A file with real newlines that also
    mentions ``\n`` inside a string literal (very common in Python) is not flagged.
    """
    return (
        len(content) > 80
        and "\\n" in content
        and content.count("\n") <= 1
    )


def execute_tool(workspace: Workspace, tool_name: str, arguments: dict) -> dict:
    """Run one tool against ``workspace`` and return the observation.

    Every failure is caught and returned as ``{"ok": False, "error": ...}``. That
    is the whole point: a missing file, a path that escapes the workspace, or a
    git repository that is not a git repository are all things the agent should
    see and adapt to, not things that should end the run. Only the LLM call
    itself is allowed to break a run.

    Returns:
        ``{"ok": True, ...}`` with tool-specific fields, or
        ``{"ok": False, "error": str}``.
    """
    try:
        if tool_name == "read_file":
            return {"ok": True, "content": read_file(workspace, arguments["path"])}

        if tool_name == "write_file":
            content = arguments["content"]
            write_file(workspace, arguments["path"], content)
            return {
                "ok": True,
                "path": workspace.relative(arguments["path"]),
                "chars_written": len(content),
                "lines_written": len(content.splitlines()),
                "suspect_escaped_newlines": _looks_escape_mangled(content),
            }

        if tool_name == "list_files":
            files = list_files(workspace, arguments["subdir"], arguments["pattern"])
            return {"ok": True, "files": files, "count": len(files)}

        if tool_name == "search_text":
            matches = search_text(
                workspace,
                arguments["query"],
                arguments["subdir"],
                arguments["pattern"],
            )
            return {"ok": True, "matches": matches, "count": len(matches)}

        if tool_name == "run_tests":
            return {"ok": True, **run_tests(workspace, arguments["test_path"])}

        if tool_name == "get_diff":
            return {"ok": True, "diff": get_diff(workspace)}

    except Exception as exc:  # noqa: BLE001 - a failed tool call is an observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Unreachable through the loop, which validates names first; kept so a new
    # declared-but-unwired tool fails loudly as an observation instead of silently.
    return {"ok": False, "error": f"Tool {tool_name!r} has no implementation."}


def _render_observation(tool_name: str, observation: dict) -> str:
    """Render an observation as the text the model sees for its tool call."""
    if not observation.get("ok", False):
        return f"ERROR: {observation.get('error', 'the tool call failed.')}"

    if tool_name == "read_file":
        return observation["content"] or "(the file is empty)"

    if tool_name == "write_file":
        report = (
            f"Wrote {observation['chars_written']} characters "
            f"({observation['lines_written']} lines) to {observation['path']}. "
            "The file's previous contents are gone. Nothing has been tested yet — "
            "run the tests to find out whether this worked."
        )
        if observation.get("suspect_escaped_newlines"):
            report += (
                "\n\nWARNING: the content you sent contains literal backslash-n "
                "sequences and almost no real line breaks, so the file is now one "
                "long line and is very likely corrupt. Write the file again, using "
                "real line breaks instead of the two characters backslash and n."
            )
        return report

    if tool_name == "list_files":
        files = observation["files"]
        if not files:
            return "No files matched."
        shown = files[:MAX_LISTED_FILES]
        text = "\n".join(shown)
        if len(files) > len(shown):
            text += f"\n... and {len(files) - len(shown)} more files not shown."
        return text

    if tool_name == "search_text":
        matches = observation["matches"]
        if not matches:
            return "No matches."
        shown = matches[:MAX_LISTED_MATCHES]
        text = "\n".join(
            f"{match['file']}:{match['line_number']}: {match['line_content']}"
            for match in shown
        )
        if len(matches) > len(shown):
            text += f"\n... and {len(matches) - len(shown)} more matches not shown."
        return text

    if tool_name == "run_tests":
        verdict = "PASSED" if observation["success"] else "FAILED"
        lines = [
            f"Tests {verdict}: {observation['passed']} passed, "
            f"{observation['failed']} failed, {observation['errors']} errors, "
            f"{observation['total']} tests collected."
        ]
        if observation["timed_out"]:
            lines.append("The test run timed out and was killed.")
        if observation["total"] == 0:
            lines.append(
                "No tests were collected, so nothing was verified. Check the "
                "test_path you passed."
            )
        output = (observation["raw_output"] or "").strip()
        if output and not observation["success"]:
            tail = output.splitlines()[-MAX_TEST_OUTPUT_LINES:]
            lines.append("\n".join(tail))
        return "\n".join(lines)

    if tool_name == "get_diff":
        return observation["diff"].strip() or "No changes have been made yet."

    if tool_name == FINISH_TOOL:
        # Rendered, then the loop stops — the model never reads this, but the
        # conversation must stay well-formed: the finishing tool_call needs an
        # answering tool message like any other.
        return "Finished."

    return json.dumps(observation, default=str)  # pragma: no cover - defensive


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Trim an over-long observation, saying so where the model can see it."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [{dropped} more characters were truncated]"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

_NO_TOOL_CALL_NUDGE = (
    "That turn produced no tool call, so nothing happened. You can only act by "
    "calling one of the tools. Call exactly one tool now — or call finish if you "
    "believe the task is done or you are stuck."
)

_MALFORMED_CALL_NUDGE = (
    "Your last turn was rejected because the tool call in it was malformed, so "
    "nothing happened. Do not write the call out as text. Use the tool-calling "
    "interface: exactly one tool, with its arguments as a well-formed JSON "
    "object. Try that action again now."
)


def _initial_file_listing(workspace: Workspace) -> str:
    """List the repository for the opening user message."""
    try:
        files = list_files(workspace)
    except Exception as exc:  # noqa: BLE001 - orientation is best-effort
        return f"(the file listing could not be produced: {type(exc).__name__}: {exc})"
    if not files:
        return "(the repository appears to be empty)"
    shown = files[:MAX_INITIAL_LISTING]
    listing = "\n".join(shown)
    if len(files) > len(shown):
        listing += f"\n... and {len(files) - len(shown)} more files"
    return listing


def _assistant_message(content: str, call: dict) -> dict:
    """Build the assistant history entry for the one tool call we executed.

    Only the executed call is recorded. The wire protocol requires every
    ``tool_call`` in an assistant message to be answered by a matching ``tool``
    message, so a model that requests three actions in one turn must not have
    all three written into the history when we deliberately run only the first.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
        ],
    }


def _safe_diff(workspace: Workspace) -> str:
    """Capture the final diff, tolerating a workspace that is not a git repo."""
    try:
        return get_diff(workspace)
    except Exception:  # noqa: BLE001 - a missing repo must not lose the trajectory
        return ""


def run_react_agent(
    workspace: Workspace,
    task_description: str,
    max_turns: int = 8,
    model: str | None = None,
    client=None,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Trajectory:
    """Work on ``task_description`` in ``workspace``, one tool action per turn.

    Each turn: send the whole conversation plus the tool schemas, take the
    model's first tool call, execute it, and append the result as an
    observation. A ``write_file`` does **not** trigger an automatic test run —
    the agent has to remember to verify its own work, which is both more
    realistic and the more interesting thing for M8 to measure.

    Args:
        workspace: The repo to work in. Should be a git repository, since the
            final diff is captured with ``git diff``; if it is not, the diff
            comes back empty rather than failing the run.
        task_description: Natural-language description of the task.
        max_turns: Hard ceiling on LLM calls. Every turn costs tokens, and an
            agent that has not converged in a handful of turns usually will not.
        model: Model override; defaults to ``GROQ_MODEL``.
        client: Pre-built client, mainly for tests. Built via
            :func:`~agent.core.llm_client.get_client` when omitted.
        on_turn: Called with each :class:`Turn` as it completes, for live
            progress reporting. Exceptions from it are not caught.

    Returns:
        A :class:`Trajectory`. The run stops when the agent calls ``finish``,
        when ``run_tests`` reports success (stopping there rather than spending
        a turn on the agent noticing), when ``max_turns`` is spent, or when the
        LLM call itself fails.

    Raises:
        LLMConfigError: If no client was passed and ``GROQ_API_KEY`` is unset.
            Everything that can go wrong *after* the first call is recorded in
            the trajectory instead of raised.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1, got {max_turns}")

    system_prompt, user_prompt = render_react_prompts(
        task_description=task_description,
        file_listing=_initial_file_listing(workspace),
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if client is None:
        client = llm_client.get_client()
    schemas = tool_schemas()

    trajectory = Trajectory(task_description=task_description)
    stop_reason = StopReason.MAX_TURNS_REACHED
    tests_have_passed = False

    def record(turn: Turn) -> None:
        trajectory.turns.append(turn)
        if on_turn is not None:
            on_turn(turn)

    for turn_number in range(1, max_turns + 1):
        try:
            reply = llm_client.chat_with_tools(
                client, messages, tools=schemas, model=model
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            attempted = llm_client.malformed_tool_call(exc)
            if attempted is None:
                # A real failure — auth, rate limit, network. The run is over.
                record(
                    Turn(
                        turn_number=turn_number,
                        tool_result={
                            "ok": False,
                            "error": f"The LLM call failed: {type(exc).__name__}: {exc}",
                        },
                    )
                )
                stop_reason = StopReason.ERROR
                break

            # The model garbled its tool call and the provider refused the whole
            # request. Same class of mistake as answering in prose, so it gets the
            # same treatment: record the wasted turn, say what went wrong, retry.
            # The malformed text is deliberately NOT echoed back into the history —
            # feeding a model its own broken syntax invites it to repeat it — but it
            # is kept in the trajectory, since "how often does the model fail to
            # form a tool call at all?" is a real property of the backend.
            messages.append({"role": "user", "content": _MALFORMED_CALL_NUDGE})
            record(
                Turn(
                    turn_number=turn_number,
                    reasoning=attempted.strip(),
                    tool_result={
                        "ok": False,
                        "error": (
                            "The provider rejected the turn: the model's tool call "
                            "was malformed, so no action was taken."
                        ),
                        "failed_generation": attempted,
                    },
                )
            )
            continue

        reasoning = reply["content"].strip()
        calls = reply["tool_calls"]
        tokens = reply["total_tokens"]

        if not calls:
            # Prose instead of an action. Recorded as a failed turn, nudged, and
            # retried next turn — one malformed response is not a dead run.
            messages.append({"role": "assistant", "content": reply["content"] or ""})
            messages.append({"role": "user", "content": _NO_TOOL_CALL_NUDGE})
            record(
                Turn(
                    turn_number=turn_number,
                    reasoning=reasoning,
                    tool_result={
                        "ok": False,
                        "error": (
                            "The model produced no tool call, so no action was "
                            "taken this turn."
                        ),
                    },
                    tokens_used=tokens,
                )
            )
            continue

        call = calls[0]
        messages.append(_assistant_message(reply["content"] or "", call))

        arguments, error = parse_tool_arguments(call["name"], call["arguments"])
        if error is not None:
            observation: dict = {"ok": False, "error": error}
        elif call["name"] == FINISH_TOOL:
            observation = {"ok": True, **arguments}
        else:
            observation = execute_tool(workspace, call["name"], arguments)

        if len(calls) > 1:
            observation["extra_calls_ignored"] = len(calls) - 1

        rendered = _truncate(_render_observation(call["name"], observation))
        if len(calls) > 1:
            rendered += (
                f"\n\n(You requested {len(calls)} tool calls in one turn; only the "
                "first was executed. Take exactly one action per turn.)"
            )
        messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": rendered}
        )

        record(
            Turn(
                turn_number=turn_number,
                reasoning=reasoning,
                tool_name=call["name"],
                tool_args=arguments,
                tool_result=observation,
                tokens_used=tokens,
            )
        )

        succeeded = observation.get("ok", False)
        if succeeded and call["name"] == "run_tests" and observation.get("success"):
            tests_have_passed = True
            stop_reason = StopReason.TESTS_PASSED
            break

        if succeeded and call["name"] == FINISH_TOOL:
            stop_reason = StopReason.AGENT_FINISHED
            break

    trajectory.stop_reason = stop_reason
    trajectory.final_success = tests_have_passed
    trajectory.final_diff = _safe_diff(workspace)
    trajectory.total_turns = len(trajectory.turns)
    trajectory.total_tokens = sum(turn.tokens_used for turn in trajectory.turns)
    return trajectory


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="Run the multi-turn ReAct agent against a repository.",
)
console = Console()

#: Per-tool colour, so a long run is skimmable.
_TOOL_STYLES = {
    "read_file": "cyan",
    "write_file": "yellow",
    "list_files": "cyan",
    "search_text": "cyan",
    "run_tests": "magenta",
    "get_diff": "blue",
    FINISH_TOOL: "bold green",
}


@app.command()
def main(
    repo: str = typer.Option(..., "--repo", help="Path to the target git repository."),
    task: str = typer.Option(..., "--task", help="Description of the task to do."),
    max_turns: int = typer.Option(
        8, "--max-turns", min=1, help="Maximum number of LLM turns."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model override (defaults to $GROQ_MODEL)."
    ),
    show_diff: bool = typer.Option(
        True, "--diff/--no-diff", help="Print the final diff."
    ),
) -> None:
    """Iterate — act, observe, adjust — until the tests pass or the turns run out."""
    try:
        workspace = Workspace(repo)
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[bold red]Bad --repo:[/bold red] {exc}")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            f"[bold]task[/bold]      {task}\n"
            f"[bold]repo[/bold]      {workspace.root}\n"
            f"[bold]model[/bold]     {model or llm_client.get_model()}\n"
            f"[bold]max turns[/bold] {max_turns}",
            title="ReAct agent",
            border_style="cyan",
        )
    )

    try:
        trajectory = run_react_agent(
            workspace,
            task,
            max_turns=max_turns,
            model=model,
            on_turn=_print_turn,
        )
    except llm_client.LLMConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2)

    _print_final(trajectory, show_diff=show_diff)
    raise typer.Exit(code=0 if trajectory.final_success else 1)


def _print_turn(turn: Turn) -> None:
    """Print one turn as it happens: reasoning, the action, and what came back."""
    console.rule(f"[bold]turn {turn.turn_number}[/bold]", style="grey50")

    if turn.reasoning:
        console.print(f"[italic grey70]{_clip(turn.reasoning, 400)}[/italic grey70]")

    if turn.tool_name is None:
        console.print(f"[red]no action[/red] — {turn.tool_result.get('error', '')}")
        return

    style = _TOOL_STYLES.get(turn.tool_name, "white")
    console.print(f"[{style}]{turn.tool_name}[/{style}]({_format_args(turn.tool_args)})")
    console.print(f"  {_summarise_result(turn.tool_name, turn.tool_result)}")


def _format_args(arguments: dict) -> str:
    """Render tool arguments compactly — file contents are summarised, not dumped."""
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 60:
            parts.append(f"{key}=<{len(value)} chars>")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _summarise_result(tool_name: str, result: dict) -> str:
    """One line describing what a tool call produced."""
    if not result.get("ok", False):
        return f"[red]error[/red] {_clip(result.get('error', ''), 200)}"

    if tool_name == "read_file":
        return f"[green]ok[/green] read {len(result['content'])} chars"
    if tool_name == "write_file":
        return f"[green]ok[/green] wrote {result['chars_written']} chars to {result['path']}"
    if tool_name == "list_files":
        return f"[green]ok[/green] {result['count']} files"
    if tool_name == "search_text":
        return f"[green]ok[/green] {result['count']} matches"
    if tool_name == "run_tests":
        verdict = (
            "[bold green]tests passed[/bold green]"
            if result["success"]
            else "[bold red]tests failed[/bold red]"
        )
        return (
            f"{verdict} — {result['passed']} passed, {result['failed']} failed, "
            f"{result['errors']} errors"
        )
    if tool_name == "get_diff":
        changed = len(result["diff"].strip().splitlines())
        return f"[green]ok[/green] diff is {changed} lines"
    if tool_name == FINISH_TOOL:
        claim = (
            "[bold green]claims success[/bold green]"
            if result.get("success")
            else "[bold yellow]gave up[/bold yellow]"
        )
        return f"{claim} — {_clip(result.get('summary', ''), 300)}"
    return "[green]ok[/green]"  # pragma: no cover - defensive


def _clip(text: str, limit: int) -> str:
    """Collapse text to one line, no longer than ``limit`` characters."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def _print_final(trajectory: Trajectory, show_diff: bool = True) -> None:
    """Render the final diff and the run summary."""
    if show_diff:
        if trajectory.final_diff.strip():
            console.print(
                Panel(
                    Syntax(
                        trajectory.final_diff,
                        "diff",
                        theme="ansi_dark",
                        word_wrap=True,
                    ),
                    title="final diff",
                    border_style="blue",
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]no changes were made[/yellow]",
                    title="final diff",
                    border_style="blue",
                )
            )

    table = Table(show_edge=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    verdict = (
        "[bold green]True[/bold green]"
        if trajectory.final_success
        else "[bold red]False[/bold red]"
    )
    table.add_row("success", verdict)
    table.add_row("stop reason", trajectory.stop_reason.value)
    table.add_row("turns", f"{trajectory.total_turns}")
    table.add_row("tokens", f"{trajectory.total_tokens:,}")
    console.print(
        Panel(
            table,
            title="summary",
            border_style="green" if trajectory.final_success else "red",
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
