"""Aggregate eval/results/*.json into a report.

    python -m eval.report

Reads every result :func:`~eval.run_eval.run_matrix` has saved, groups by
condition, and reports overall success rate, success rate by difficulty
tier, average turns (react only — a single-shot baseline's average is
always 1 and says nothing), average tokens, and average wall-clock time.
Writes eval/results/report.md and prints the same table to the console.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .harness import EvalResult, load_tasks

__all__ = ["compute_summary", "load_results", "render_markdown"]

console = Console()

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks"

#: Canonical display order; only tiers actually present in the loaded tasks
#: are ever shown, so a smaller/larger task set is not padded with zeros.
_DIFFICULTY_ORDER = ["trivial", "medium", "hard"]


def load_results(results_dir: Path) -> list[EvalResult]:
    """Load every ``*.json`` file in ``results_dir`` as an :class:`EvalResult`."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(EvalResult.model_validate_json(path.read_text(encoding="utf-8")))
    return results


def difficulty_by_task_id(tasks_dir: Path) -> dict[str, str]:
    """Map ``task_id -> difficulty`` from the task YAMLs, for grouping results."""
    return {task["id"]: task["difficulty"] for task in load_tasks(tasks_dir)}


def _rate(successes: int, n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{successes / n:.1%} ({successes}/{n})"


def compute_summary(
    results: list[EvalResult], difficulty_by_task: dict[str, str]
) -> dict[str, dict]:
    """Compute per-condition aggregate stats.

    Returns a dict keyed by condition name, each value:

    * ``n`` — number of runs for this condition.
    * ``successes`` / ``success_rate`` — overall, as a count and a fraction.
    * ``by_difficulty`` — ``{tier: {"n", "successes", "success_rate"}}``, for
      every tier actually present among this condition's results. A
      ``task_id`` with no matching entry in ``difficulty_by_task`` is grouped
      under ``"unknown"`` rather than dropped, so a stale/missing task
      mapping is visible in the report instead of silently undercounting.
    * ``avg_turns`` / ``avg_total_tokens`` / ``avg_wall_clock_seconds`` —
      simple means over this condition's runs.
    """
    by_condition: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_condition[result.condition].append(result)

    summary: dict[str, dict] = {}
    for condition, condition_results in by_condition.items():
        n = len(condition_results)
        successes = sum(1 for r in condition_results if r.success)

        by_difficulty: dict[str, dict] = {}
        for result in condition_results:
            tier = difficulty_by_task.get(result.task_id, "unknown")
            bucket = by_difficulty.setdefault(tier, {"n": 0, "successes": 0})
            bucket["n"] += 1
            bucket["successes"] += int(result.success)
        for bucket in by_difficulty.values():
            bucket["success_rate"] = bucket["successes"] / bucket["n"]

        summary[condition] = {
            "n": n,
            "successes": successes,
            "success_rate": successes / n if n else 0.0,
            "by_difficulty": by_difficulty,
            "avg_turns": sum(r.turns for r in condition_results) / n if n else 0.0,
            "avg_total_tokens": sum(r.total_tokens for r in condition_results) / n if n else 0.0,
            "avg_wall_clock_seconds": (
                sum(r.wall_clock_seconds for r in condition_results) / n if n else 0.0
            ),
        }
    return summary


def _difficulty_tiers_present(summary: dict[str, dict]) -> list[str]:
    present = {tier for data in summary.values() for tier in data["by_difficulty"]}
    ordered = [tier for tier in _DIFFICULTY_ORDER if tier in present]
    ordered += sorted(present - set(_DIFFICULTY_ORDER))  # "unknown", or anything new
    return ordered


def _methodology_section(summary: dict[str, dict]) -> str:
    """A deliberately modest, auto-generated caveat about sample size.

    Never overclaims regardless of how the numbers turn out — this is
    boilerplate about the sample, not a summary of the results themselves.
    """
    if not summary:
        return (
            "## Methodology and limitations\n\n"
            "No results were found. This section will describe the sample "
            "size once `python -m eval.run_eval` has produced some.\n"
        )

    counts = ", ".join(f"{data['n']} for {condition}" for condition, data in sorted(summary.items()))
    return (
        "## Methodology and limitations\n\n"
        f"Results based on {counts} task run(s). A sample this small means "
        "individual results carry more weight than the percentages alone "
        "suggest — treat these as directional findings, not statistically "
        "robust claims.\n\n"
        "Every `success` value here comes from independently re-running the "
        "task's own `verify_command` against the agent's final workspace "
        "state, never from the agent's own claim of success (see "
        "`eval/harness.py`'s \"verified, not claimed\" principle).\n"
    )


def render_markdown(summary: dict[str, dict]) -> str:
    """Render the full report.md contents from a computed summary."""
    lines = ["# M8 Evaluation Report", "", "## Summary", ""]

    if not summary:
        lines.append("No results were found in eval/results/. Run `python -m eval.run_eval` first.")
        lines.append("")
    else:
        lines.append("| Condition | Runs | Success rate | Avg turns | Avg tokens | Avg wall-clock (s) |")
        lines.append("|---|---|---|---|---|---|")
        for condition in sorted(summary):
            data = summary[condition]
            avg_turns = f"{data['avg_turns']:.1f}" if condition == "react" else "n/a (single-shot)"
            lines.append(
                f"| {condition} | {data['n']} | {_rate(data['successes'], data['n'])} | "
                f"{avg_turns} | {data['avg_total_tokens']:.0f} | "
                f"{data['avg_wall_clock_seconds']:.1f} |"
            )
        lines.append("")

        tiers = _difficulty_tiers_present(summary)
        if tiers:
            lines.append("## Success rate by difficulty")
            lines.append("")
            lines.append("| Condition | " + " | ".join(tiers) + " |")
            lines.append("|---|" + "---|" * len(tiers))
            for condition in sorted(summary):
                by_difficulty = summary[condition]["by_difficulty"]
                cells = [
                    _rate(by_difficulty[tier]["successes"], by_difficulty[tier]["n"])
                    if tier in by_difficulty
                    else "n/a"
                    for tier in tiers
                ]
                lines.append(f"| {condition} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append(_methodology_section(summary))
    return "\n".join(lines)


def render_console_table(summary: dict[str, dict]) -> Table:
    """The same summary as :func:`render_markdown`'s top table, for the console."""
    table = Table(title="M8 evaluation summary")
    table.add_column("condition")
    table.add_column("runs", justify="right")
    table.add_column("success rate", justify="right")
    table.add_column("avg turns", justify="right")
    table.add_column("avg tokens", justify="right")
    table.add_column("avg wall-clock (s)", justify="right")

    for condition in sorted(summary):
        data = summary[condition]
        avg_turns = f"{data['avg_turns']:.1f}" if condition == "react" else "n/a"
        table.add_row(
            condition,
            str(data["n"]),
            _rate(data["successes"], data["n"]),
            avg_turns,
            f"{data['avg_total_tokens']:.0f}",
            f"{data['avg_wall_clock_seconds']:.1f}",
        )
    return table


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="Summarise eval/results/*.json into a report.")


@app.command()
def main(
    results_dir: str = typer.Option(str(DEFAULT_RESULTS_DIR), "--results-dir"),
    tasks_dir: str = typer.Option(str(DEFAULT_TASKS_DIR), "--tasks-dir"),
    output: Optional[str] = typer.Option(
        None, "--output", help="Defaults to <results-dir>/report.md."
    ),
) -> None:
    results = load_results(Path(results_dir))
    if not results:
        console.print(
            "[yellow]No results found — run `python -m eval.run_eval` first.[/yellow]"
        )
        raise typer.Exit(code=0)

    summary = compute_summary(results, difficulty_by_task_id(Path(tasks_dir)))
    console.print(render_console_table(summary))

    output_path = Path(output) if output else Path(results_dir) / "report.md"
    output_path.write_text(render_markdown(summary), encoding="utf-8")
    console.print(f"\nWrote {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
