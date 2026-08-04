"""The single-shot baseline agent: one LLM call, no iteration, no tools.

This is deliberately the weak agent. It reads the target file, asks the model
once for a corrected version, writes whatever came back, and runs the tests to
see what happened. It does not retry, does not loop, does not look at the test
output and try again — that is M3's multi-turn ReAct loop. Its whole purpose is
to be the honest floor that M8 measures the real agent against, so it must not
quietly borrow any of the real agent's machinery.

Run it directly against a repo::

    python -m agent.core.baseline \\
        --repo demo/sample_repos/calculator \\
        --task "fix add() so it returns the correct sum instead of subtracting" \\
        --file calculator/calc.py
"""

from __future__ import annotations

import re
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..tools import Workspace, get_diff, read_file, run_tests, write_file
from . import llm_client
from .prompts import render_baseline_prompts

__all__ = ["CodeBlockError", "extract_code_block", "run_baseline"]

#: A fenced block: ``` or ~~~ (three or more), an optional language tag, the
#: body, then the *same* fence again on its own line. Matching the opening run
#: exactly via a backreference means a longer fence can legally wrap a shorter
#: one, which is how markdown itself works.
_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[\w+.#-]*)[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"^[ \t]*(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

#: Language tags that mark a block as the Python file we asked for. Used only to
#: break a tie between several blocks, never to reject a lone one.
_PYTHON_TAGS = frozenset({"python", "py", "python3"})


class CodeBlockError(ValueError):
    """Raised when a model reply does not contain exactly one usable code block."""


def extract_code_block(response_text: str) -> str:
    """Pull the file content out of a markdown-fenced block in a model reply.

    Tolerates the shapes models actually produce: a bare block, a block with
    prose before and/or after it, a language tag or none, and ``~~~`` fences.
    Trailing whitespace is trimmed and the text is normalised to ``\\n`` line
    endings with exactly one trailing newline, so the write lands byte-stable
    and the resulting diff shows only real changes.

    When several blocks are present, exactly one Python-tagged block wins —
    that covers the common "here's the code, and here's how to run it" reply.
    Anything more ambiguous than that is an error rather than a guess, because
    guessing wrong means writing prose into a source file.

    Raises:
        CodeBlockError: If there is no code block, if every candidate block is
            empty, or if several blocks are equally plausible.
    """
    if not response_text or not response_text.strip():
        raise CodeBlockError(
            "The model returned an empty response, so there is no code to extract."
        )

    blocks = [
        (match.group("lang").lower(), match.group("body"))
        for match in _FENCE_RE.finditer(response_text)
    ]
    non_empty = [(lang, body) for lang, body in blocks if body.strip()]

    if not non_empty:
        detail = (
            "every code block in it was empty"
            if blocks
            else "no triple-backtick code block was found"
        )
        raise CodeBlockError(
            f"Could not extract code from the model response: {detail}. "
            "The response was expected to contain the corrected file wrapped in "
            "a single ```python ... ``` block. "
            f"Response began: {response_text.strip()[:200]!r}"
        )

    if len(non_empty) == 1:
        return _normalise(non_empty[0][1])

    python_blocks = [body for lang, body in non_empty if lang in _PYTHON_TAGS]
    if len(python_blocks) == 1:
        return _normalise(python_blocks[0])

    raise CodeBlockError(
        f"Ambiguous model response: found {len(non_empty)} non-empty code blocks "
        f"({len(python_blocks)} tagged as Python), so it is unclear which one is "
        "the corrected file. Exactly one code block was expected."
    )


def _normalise(body: str) -> str:
    """Normalise extracted code to LF endings with a single trailing newline."""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip() + "\n"


def run_baseline(
    workspace: Workspace,
    task_description: str,
    target_file: str,
    model: str | None = None,
    client=None,
    test_path: str = ".",
) -> dict:
    """Attempt ``task_description`` in ``target_file`` with a single LLM call.

    One shot: read the file, ask once, write the reply, record what happened.
    There is no retry and no self-correction, by design.

    Args:
        workspace: The repo to work in. Must be a git repository, since the
            diff is captured with ``git diff``.
        task_description: Natural-language description of the bug to fix.
        target_file: File to rewrite, relative to the workspace root.
        model: Model override; defaults to ``GROQ_MODEL``.
        client: Pre-built client, mainly for tests. Built via
            :func:`~agent.core.llm_client.get_client` when omitted.
        test_path: What to hand pytest, relative to the workspace root.

    Returns:
        A dict recording the attempt:

        * ``success`` — True only if the test suite passed afterwards.
        * ``diff`` — the working-tree diff, captured *before* the tests ran so
          it reflects the model's edit and nothing pytest left behind.
        * ``test_results`` — the full :func:`~agent.tools.run_tests` dict, or
          ``None`` if the run never got as far as testing.
        * ``llm_response`` — the raw reply, kept verbatim for later inspection.
        * ``token_usage`` — prompt/completion/total counts (M8 costs runs).
        * ``target_file`` / ``task_description`` — echoed back so a result is
          self-describing in the eval harness.
        * ``error`` — ``None`` on a completed run, otherwise why it stopped. A
          model that returns no usable code block is a baseline failure worth
          measuring, not a crash, so it is recorded here instead of raised.

    Raises:
        FileNotFoundError: If ``target_file`` does not exist.
        LLMConfigError: If no client was passed and ``GROQ_API_KEY`` is unset.
        openai.OpenAIError: If the API call itself fails.
    """
    relative_target = workspace.relative(target_file)
    original_content = read_file(workspace, target_file)

    system_prompt, user_prompt = render_baseline_prompts(
        task_description=task_description,
        file_path=relative_target,
        file_content=original_content,
    )

    if client is None:
        client = llm_client.get_client()
    completion = llm_client.chat_completion(
        client, system_prompt, user_prompt, model=model
    )

    result = {
        "success": False,
        "diff": "",
        "test_results": None,
        "llm_response": completion["content"],
        "token_usage": {
            "prompt_tokens": completion["prompt_tokens"],
            "completion_tokens": completion["completion_tokens"],
            "total_tokens": completion["total_tokens"],
        },
        "target_file": relative_target,
        "task_description": task_description,
        "error": None,
    }

    try:
        corrected = extract_code_block(completion["content"])
    except CodeBlockError as exc:
        # One shot means one shot: record the malformed reply and stop.
        result["error"] = str(exc)
        return result

    write_file(workspace, target_file, corrected)

    # Diff first, tests second: pytest writes __pycache__/.pytest_cache, and
    # the recorded diff should be the model's edit, nothing else.
    result["diff"] = get_diff(workspace)
    test_results = run_tests(workspace, test_path)
    result["test_results"] = test_results
    result["success"] = bool(test_results["success"])
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="Run the single-shot baseline agent against a repository.",
)
console = Console()


@app.command()
def main(
    repo: str = typer.Option(..., "--repo", help="Path to the target git repository."),
    task: str = typer.Option(..., "--task", help="Description of the bug to fix."),
    file: str = typer.Option(
        ..., "--file", help="File to fix, relative to the repo root."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model override (defaults to $GROQ_MODEL)."
    ),
    test_path: str = typer.Option(
        ".", "--test-path", help="What to run pytest against, relative to the repo."
    ),
) -> None:
    """One LLM call, one edit, one test run — then report what happened."""
    try:
        workspace = Workspace(repo)
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[bold red]Bad --repo:[/bold red] {exc}")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            f"[bold]task[/bold]  {task}\n"
            f"[bold]file[/bold]  {file}\n"
            f"[bold]repo[/bold]  {workspace.root}\n"
            f"[bold]model[/bold] {model or llm_client.get_model()}",
            title="single-shot baseline",
            border_style="cyan",
        )
    )

    try:
        result = run_baseline(
            workspace, task, file, model=model, test_path=test_path
        )
    except llm_client.LLMConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        console.print(f"[bold red]Could not run:[/bold red] {exc}")
        raise typer.Exit(code=2)

    _print_summary(result)
    raise typer.Exit(code=0 if result["success"] else 1)


def _print_summary(result: dict) -> None:
    """Render a run's diff, test outcome, and token usage."""
    if result["diff"]:
        console.print(
            Panel(
                Syntax(result["diff"], "diff", theme="ansi_dark", word_wrap=True),
                title="diff",
                border_style="blue",
            )
        )
    else:
        console.print(
            Panel("[yellow]no changes were made[/yellow]", title="diff", border_style="blue")
        )

    tests = result["test_results"]
    if tests is None:
        body = f"[red]tests never ran[/red] — {result['error']}"
    else:
        verdict = (
            "[bold green]PASSED[/bold green]"
            if result["success"]
            else "[bold red]FAILED[/bold red]"
        )
        body = (
            f"{verdict}   "
            f"{tests['passed']} passed, {tests['failed']} failed, "
            f"{tests['errors']} errors, {tests['total']} total"
        )
        if tests["timed_out"]:
            body += "\n[yellow]the test run timed out[/yellow]"
        if not result["success"] and tests["raw_output"]:
            tail = "\n".join(tests["raw_output"].strip().splitlines()[-15:])
            body += f"\n\n{tail}"
    console.print(
        Panel(
            body,
            title="tests",
            border_style="green" if result["success"] else "red",
        )
    )

    usage = result["token_usage"]
    table = Table(title="token usage", title_style="bold", show_edge=False)
    table.add_column("prompt", justify="right")
    table.add_column("completion", justify="right")
    table.add_column("total", justify="right")
    table.add_row(
        f"{usage['prompt_tokens']:,}",
        f"{usage['completion_tokens']:,}",
        f"{usage['total_tokens']:,}",
    )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
