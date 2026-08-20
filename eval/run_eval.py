"""CLI: run the full (task, condition) matrix and save results as it goes.

    python -m eval.run_eval --conditions baseline,react --tasks-dir eval/tasks

Every run's :class:`~eval.harness.EvalResult` is written to its own
``eval/results/{task_id}__{condition}.json`` file *immediately* after that
run completes, not batched at the end — a Groq 429 (or any other crash)
partway through the matrix loses at most the run in progress, never the
runs already done. A combination whose result file already exists is
skipped on the next invocation (``--force`` re-runs everything anyway), so
re-running the same command after a rate limit clears just picks up where
it left off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import typer
from openai import RateLimitError
from rich.console import Console

from . import harness
from .harness import EvalResult

__all__ = ["run_matrix"]

console = Console()

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _rate_limit_message(completed: int) -> str:
    return (
        f"Hit daily rate limit after {completed} runs. Progress saved. "
        "Re-run this same command tomorrow to continue."
    )


def run_matrix(
    tasks: list[dict],
    conditions: list[str],
    results_dir: Path,
    max_turns: int = 12,
    model: str | None = None,
    force: bool = False,
    run_task_fn: Callable[..., EvalResult] = harness.run_task,
    on_result: Optional[Callable[[dict, str, Optional[EvalResult], bool], None]] = None,
) -> dict:
    """Run every (task, condition) pair, saving each result as it lands.

    This is the whole CLI's logic, factored out of :func:`main` so it can be
    driven by tests without going through typer or a real LLM — ``run_task_fn``
    stands in for :func:`agent.harness.run_task` in exactly that spirit.

    Args:
        tasks: Task dicts, as returned by :func:`agent.harness.load_tasks`.
        conditions: Condition names to run each task under, e.g.
            ``["baseline", "react"]``.
        results_dir: Where ``{task_id}__{condition}.json`` files are written.
            Created if it does not exist.
        max_turns: Forwarded to ``run_task_fn`` for the ``"react"`` condition.
        model: Forwarded to ``run_task_fn``.
        force: Re-run a combination even if its result file already exists.
        run_task_fn: Injected for testing; defaults to the real
            :func:`agent.harness.run_task`.
        on_result: Called after each combination — ``(task, condition, result,
            skipped)``. ``result`` is ``None`` when ``skipped`` is True. Used
            by :func:`main` for live progress printing; tests can pass their
            own to assert on ordering without parsing console output.

    Returns:
        ``{"completed": int, "skipped": int, "rate_limited": bool}``. When
        ``rate_limited`` is True, the matrix stopped early — everything run
        before the rate limit hit is already saved to disk.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0

    for task in tasks:
        for condition in conditions:
            result_file = results_dir / f"{task['id']}__{condition}.json"
            if result_file.exists() and not force:
                skipped += 1
                if on_result is not None:
                    on_result(task, condition, None, True)
                continue

            try:
                result = run_task_fn(
                    task, condition, max_turns=max_turns, model=model
                )
            except RateLimitError:
                return {
                    "completed": completed,
                    "skipped": skipped,
                    "rate_limited": True,
                }

            result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            completed += 1
            if on_result is not None:
                on_result(task, condition, result, False)

    return {"completed": completed, "skipped": skipped, "rate_limited": False}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="Run the M8 evaluation matrix.")


def _print_progress(task: dict, condition: str, result: Optional[EvalResult], skipped: bool) -> None:
    """Live per-run line plus a running totals footer — the "rich progress
    summary" the CLI is required to show as it goes.
    """
    if skipped:
        console.print(f"[dim]skip[/dim]   {task['id']} / {condition} (result already on disk)")
        return

    verdict = "[bold green]PASS[/bold green]" if result.success else "[bold red]FAIL[/bold red]"
    console.print(
        f"{verdict}   {task['id']} / {condition}   "
        f"turns={result.turns} tokens={result.total_tokens:,} "
        f"time={result.wall_clock_seconds:.1f}s"
    )


@app.command()
def main(
    conditions: str = typer.Option(
        "baseline,react", "--conditions", help="Comma-separated conditions to run."
    ),
    tasks_dir: str = typer.Option(
        str(DEFAULT_TASKS_DIR), "--tasks-dir", help="Directory of task YAML files."
    ),
    results_dir: str = typer.Option(
        str(DEFAULT_RESULTS_DIR), "--results-dir", help="Where result JSON files are saved."
    ),
    max_turns: int = typer.Option(
        12, "--max-turns", min=1, help="Turn budget for the react condition."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model override (defaults to $GROQ_MODEL)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run every combination even if a result already exists."
    ),
) -> None:
    condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
    tasks = harness.load_tasks(tasks_dir)

    console.print(
        f"[bold]{len(tasks)}[/bold] tasks x [bold]{len(condition_list)}[/bold] "
        f"conditions ({', '.join(condition_list)}) = "
        f"[bold]{len(tasks) * len(condition_list)}[/bold] runs to consider\n"
    )

    summary = run_matrix(
        tasks,
        condition_list,
        Path(results_dir),
        max_turns=max_turns,
        model=model,
        force=force,
        on_result=_print_progress,
    )

    if summary["rate_limited"]:
        console.print(f"\n[bold red]{_rate_limit_message(summary['completed'])}[/bold red]")
        raise typer.Exit(code=1)

    console.print(
        f"\nDone: [bold]{summary['completed']}[/bold] run, "
        f"[bold]{summary['skipped']}[/bold] skipped."
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
