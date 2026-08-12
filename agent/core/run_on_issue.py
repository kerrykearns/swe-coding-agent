"""End-to-end entrypoint: a GitHub issue in, a local commit out.

This is the wiring milestone. Nothing new is reasoned about here — the issue
comes from :mod:`agent.core.github_client`, the checkout from
:mod:`agent.core.repo_manager`, the task text from
:mod:`agent.core.task_from_issue`, and the actual work from M3's
:func:`agent.core.react_agent.run_react_agent`. This module's whole job is to
put them in order and report honestly on what happened.

Run it::

    python -m agent.core.run_on_issue \\
        --repo kerrykearns/agent-test-playground --issue 1 --max-turns 8

SCOPING CONSTRAINT (M4)
=======================
The flow stops at ``git commit``. There is no ``git push`` and no call to
:func:`agent.core.github_client.create_pull_request` — that function exists and
is tested, but wiring it in waits for M6's safety/confirmation gate, because
pushing a branch and opening a PR are outward-facing actions that a human should
approve before an agent takes them. Every run says so on the way out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..tools import SandboxError
from . import github_client, llm_client, repo_manager
from .react_agent import Trajectory, Turn, run_react_agent

# Reused rather than reimplemented: these render a Turn and a Trajectory exactly
# as `python -m agent.core.react_agent` does, and a second copy of that
# formatting would drift from it. They are private to react_agent only in the
# sense that they are not part of its library API.
from .react_agent import _print_final, _print_turn
from .task_from_issue import issue_to_task_description

__all__ = ["DEFAULT_WORKSPACES_DIR", "app", "main", "run_on_issue"]

#: Where clones go, relative to the project root. Gitignored (see .gitignore's
#: "Agent scratch space" section) because a run leaves a full checkout of
#: somebody else's repository behind, complete with its own .git directory.
DEFAULT_WORKSPACES_DIR = Path(__file__).resolve().parents[2] / "workspaces"

#: The closing note every run prints. Stated as a fact about what was and was
#: not done, because "committed" reads as "delivered" to anyone who has used a
#: PR-opening bot before.
NO_PUSH_NOTE = (
    "Changes committed locally on branch {branch}. Review before pushing — "
    "push/PR flow is not yet implemented (pending M6 safety gate)."
)


def _clone_dir_name(repo_full_name: str, issue_number: int) -> str:
    """Name the clone directory after the repo and issue it belongs to."""
    slug = repo_manager.sanitize_branch_name(repo_full_name)
    return f"{slug}-issue-{issue_number}"


def _available_dir(parent: Path, name: str) -> Path:
    """Return a path under ``parent`` that does not exist yet.

    A second run against the same issue must not fail, and must not silently
    reuse the first run's checkout either — a stale working tree would make the
    next agent's diff meaningless. So the name gets a counter instead.
    """
    candidate = parent / name
    attempt = 2
    while candidate.exists():
        candidate = parent / f"{name}-{attempt}"
        attempt += 1
    return candidate


def run_on_issue(
    repo_full_name: str,
    issue_number: int,
    max_turns: int = 8,
    model: Optional[str] = None,
    workspaces_dir: str | Path = DEFAULT_WORKSPACES_DIR,
    branch_prefix: str = "agent-fix",
    client=None,
    github=None,
    console: Optional[Console] = None,
    on_turn: Optional[Callable[[Turn], None]] = None,
    sandboxed: bool = False,
) -> dict:
    """Fetch an issue, clone the repo, branch, run the agent, commit if it worked.

    Args:
        repo_full_name: ``"owner/repo"``.
        issue_number: The issue to work on.
        max_turns: Turn budget handed to the ReAct loop.
        model: Model override; defaults to ``GROQ_MODEL``.
        workspaces_dir: Parent directory for the clone. Defaults to the
            gitignored ``workspaces/`` at the project root; tests pass a
            ``tmp_path``.
        branch_prefix: First segment of the branch name.
        client: Pre-built LLM client, passed through to the ReAct loop.
        github: Pre-built PyGithub client, mainly for tests.
        console: Where to report progress. ``None`` runs silently.
        on_turn: Per-turn callback for the ReAct loop.
        sandboxed: Run the agent's ``run_tests`` calls inside a Docker
            container with no network access rather than as local
            subprocesses. Defaults to False here, matching every other tool
            in :mod:`agent.tools` — this function's own default stays
            conservative and Docker-free so calling it directly (as the tests
            do) never requires Docker. The CLI below is the one that opts
            into ``sandboxed=True`` by default, since it always runs the
            agent against a real, arbitrary cloned repo.

    Returns:
        A dict describing the whole run:

        * ``repo``, ``issue``, ``task_description``, ``branch``
        * ``workspace`` — the :class:`~agent.tools.workspace.Workspace` of the
          clone — and ``workspace_path`` as a string
        * ``trajectory`` — the M3 :class:`~agent.core.react_agent.Trajectory`
        * ``commit`` — :func:`repo_manager.commit_changes`' result dict, or
          ``None`` when no commit was attempted because the agent did not
          succeed
        * ``committed`` — whether a commit actually landed

        Note that ``committed`` is *not* a second opinion on whether the fix is
        good. It follows ``trajectory.final_success``, which M3 defines as "a
        test run actually reported success" — never the agent's own claim.

    Raises:
        github_client.GitHubError: The issue could not be fetched.
        repo_manager.RepoManagerError: The clone or the branch failed.
        llm_client.LLMConfigError: No ``GROQ_API_KEY`` and no ``client``.
    """
    repo_full_name = github_client.normalise_repo_name(repo_full_name)

    def say(renderable) -> None:
        if console is not None:
            console.print(renderable)

    issue = github_client.get_issue(repo_full_name, issue_number, client=github)
    say(_issue_panel(repo_full_name, issue))

    parent = Path(workspaces_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    destination = _available_dir(parent, _clone_dir_name(repo_full_name, issue_number))

    # Printed text stays inside cp1252: a legacy Windows console is what rich
    # falls back to here, and it encodes its output with the OS codepage — a
    # decorative arrow in this line crashed the first live run with
    # UnicodeEncodeError before the clone had even started.
    say(f"[grey70]cloning[/grey70] {repo_full_name} -> {destination}")
    workspace = repo_manager.clone_repo(repo_full_name, destination)

    branch = repo_manager.branch_name_for_issue(issue_number, prefix=branch_prefix)
    repo_manager.create_branch(workspace, branch)
    say(f"[grey70]branch[/grey70]  {branch}")

    task_description = issue_to_task_description(issue)
    say(
        Panel(
            task_description,
            title="task description handed to the agent",
            border_style="grey50",
        )
    )

    trajectory = run_react_agent(
        workspace,
        task_description,
        max_turns=max_turns,
        model=model,
        client=client,
        on_turn=on_turn,
        sandboxed=sandboxed,
    )

    commit: Optional[dict] = None
    if trajectory.final_success:
        commit = repo_manager.commit_changes(
            workspace, f"Fix: {issue['title']} (closes #{issue['number']})"
        )
    else:
        say(
            "[yellow]no changes committed, agent did not reach success[/yellow] "
            f"(stopped as {trajectory.stop_reason.value})"
        )

    return {
        "repo": repo_full_name,
        "issue": issue,
        "task_description": task_description,
        "branch": branch,
        "workspace": workspace,
        "workspace_path": str(workspace.root),
        "trajectory": trajectory,
        "commit": commit,
        "committed": bool(commit and commit["success"]),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="Run the ReAct agent against a real GitHub issue (clone, branch, commit).",
)
console = Console()


@app.command()
def main(
    repo: str = typer.Option(
        ..., "--repo", help='Target repository as "owner/repo".'
    ),
    issue: int = typer.Option(..., "--issue", min=1, help="Issue number to work on."),
    max_turns: int = typer.Option(
        8, "--max-turns", min=1, help="Maximum number of LLM turns."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model override (defaults to $GROQ_MODEL)."
    ),
    workspaces_dir: Path = typer.Option(
        DEFAULT_WORKSPACES_DIR,
        "--workspaces-dir",
        help="Where to clone the repository (gitignored by default).",
    ),
    show_diff: bool = typer.Option(
        True, "--diff/--no-diff", help="Print the final diff."
    ),
    sandbox: bool = typer.Option(
        True,
        "--sandbox/--no-sandbox",
        help=(
            "Run tests inside a network-isolated Docker container instead of "
            "locally. On by default here, since this CLI runs the agent "
            "against a real cloned repository."
        ),
    ),
) -> None:
    """Fetch the issue, clone, branch, iterate, and commit locally if it worked."""
    console.print(
        Panel(
            f"[bold]repo[/bold]      {repo}\n"
            f"[bold]issue[/bold]     #{issue}\n"
            f"[bold]model[/bold]     {model or llm_client.get_model()}\n"
            f"[bold]max turns[/bold] {max_turns}\n"
            f"[bold]sandbox[/bold]   {sandbox}",
            title="agent on a GitHub issue",
            border_style="cyan",
        )
    )

    try:
        result = run_on_issue(
            repo,
            issue,
            max_turns=max_turns,
            model=model,
            workspaces_dir=workspaces_dir,
            console=console,
            on_turn=_print_turn,
            sandboxed=sandbox,
        )
    except github_client.GitHubError as exc:
        console.print(f"[bold red]GitHub error:[/bold red] {exc}")
        raise typer.Exit(code=2)
    except repo_manager.RepoManagerError as exc:
        console.print(f"[bold red]Git error:[/bold red] {exc}")
        raise typer.Exit(code=2)
    except llm_client.LLMConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2)
    except SandboxError as exc:
        console.print(f"[bold red]Sandbox error:[/bold red] {exc}")
        raise typer.Exit(code=2)

    _print_final(result["trajectory"], show_diff=show_diff)
    print_run_summary(result, console)
    raise typer.Exit(code=0 if result["trajectory"].final_success else 1)


def _issue_panel(repo_full_name: str, issue: dict) -> Panel:
    """Render the fetched issue as it came back from GitHub."""
    body = (issue.get("body") or "").strip() or "(no description)"
    return Panel(
        f"[bold]#{issue['number']}[/bold] {issue['title']}\n"
        f"[grey70]{issue['url']}  ·  state: {issue['state']}[/grey70]\n\n"
        f"{body}",
        title=f"issue fetched from {repo_full_name}",
        border_style="magenta",
    )


def print_run_summary(result: dict, out: Optional[Console] = None) -> None:
    """Print the final summary: issue, branch, trajectory, commit, and the note.

    Public so the integration test prints exactly what the CLI prints, instead
    of a second rendering of the same facts that could disagree with it.
    """
    out = out or console
    trajectory: Trajectory = result["trajectory"]
    commit: Optional[dict] = result["commit"]

    table = Table(show_edge=False, box=None)
    table.add_column(style="bold")
    table.add_column(overflow="fold")

    issue = result["issue"]
    table.add_row("repo", result["repo"])
    table.add_row("issue", f"#{issue['number']} {issue['title']}")
    table.add_row("issue url", issue["url"])
    table.add_row("workspace", result["workspace_path"])
    table.add_row("branch", result["branch"])
    table.add_row("", "")
    table.add_row(
        "agent success",
        "[bold green]True[/bold green]"
        if trajectory.final_success
        else "[bold red]False[/bold red]",
    )
    table.add_row("stop reason", trajectory.stop_reason.value)
    table.add_row("turns", f"{trajectory.total_turns}")
    table.add_row("tokens", f"{trajectory.total_tokens:,}")
    table.add_row("", "")

    if commit is None:
        table.add_row(
            "commit",
            "[yellow]not attempted — the agent did not reach success[/yellow]",
        )
    elif commit["success"]:
        table.add_row("commit", "[bold green]committed[/bold green]")
        table.add_row("commit hash", commit["commit_hash"] or "(unknown)")
    else:
        table.add_row("commit", "[bold red]not committed[/bold red]")
        table.add_row("reason", commit["reason"] or "(no reason given)")

    out.print(
        Panel(
            table,
            title="run summary",
            border_style="green" if result["committed"] else "yellow",
        )
    )
    out.print(
        Panel(
            NO_PUSH_NOTE.format(branch=result["branch"])
            if result["committed"]
            else (
                "Nothing was committed, so there is nothing to review. No push "
                "or PR was attempted — that flow is not yet implemented "
                "(pending M6 safety gate)."
            ),
            border_style="yellow",
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
