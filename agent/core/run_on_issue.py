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

PUSH / PULL REQUEST (M6)
=========================
The flow still always stops at ``git commit`` by itself. Pushing the branch
and opening a pull request now happen too, but ONLY when the caller supplies
a :class:`~agent.safety.confirmation_gate.ConfirmationGate` — the CLI below
always builds one, so a real run always gets the option, but calling
:func:`run_on_issue` as a library function (as every offline test in this
project does) still stops at the local commit exactly as it did in M4 unless
a gate is explicitly passed in. Neither :func:`agent.core.repo_manager.push_branch`
nor :func:`agent.core.github_client.create_pull_request` is ever called
except from behind ``gate.authorize(...)`` — see the M4 scoping decision in
planning.md for why that boundary matters, and the safety policy's
``confirm_push_branch``/``confirm_create_pull_request`` rules for why both
sit in the needs_confirmation tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..safety import ConfirmationGate
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

#: Printed whenever a commit landed but no pull request came out of this run —
#: whether because no gate was supplied (the library default, and every
#: offline test's path), or the gate denied/blocked the push. The specific
#: reason, if any, was already printed by _push_and_open_pr as it happened.
NO_PUSH_NOTE = (
    "Changes committed locally on branch {branch}. Review before pushing — "
    "push/PR was not completed this run."
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


def _pr_title(issue: dict) -> str:
    """Match the commit subject, so the PR title and the commit it carries agree."""
    return f"Fix: {issue['title']} (closes #{issue['number']})"


def _pr_body(issue: dict, trajectory: Trajectory) -> str:
    """Build the PR description from the agent's own closing summary, if it gave one."""
    summary = ""
    for turn in reversed(trajectory.turns):
        if turn.tool_name == "finish":
            summary = str(turn.tool_args.get("summary", "")).strip()
            break

    body = f"Automated fix for #{issue['number']}, opened by the ReAct agent.\n\n"
    if summary:
        body += f"{summary}\n\n"
    body += f"Closes #{issue['number']}."
    return body


def _push_and_open_pr(
    *,
    repo_full_name: str,
    branch: str,
    base_branch: str,
    issue: dict,
    trajectory: Trajectory,
    workspace,
    gate: ConfirmationGate,
    auto_push: bool,
    say: Callable[[Any], None],
) -> tuple[Optional[dict], Optional[dict]]:
    """Push the branch, then open a PR — each step individually gated.

    ``auto_push`` only changes which callback answers the gate for these two
    calls specifically: pre-supplying "yes" instead of prompting. The gate is
    still called either way, so a ``blocked`` classification (there is none
    for these tools today, but the policy is free to add one) would still
    refuse unconditionally regardless of ``auto_push``.

    Returns:
        ``(push_result, pr_result)``, either of which is ``None`` if that
        step was never reached (denied, blocked, or a prior step failed).
    """
    override = (lambda *_args: True) if auto_push else None

    push_auth = gate.authorize(
        "push_branch", {"repo": repo_full_name, "branch": branch}, override_callback=override
    )
    if not push_auth.allowed:
        say(f"[yellow]push skipped[/yellow] ({push_auth.tier}): {push_auth.reason}")
        return None, None

    push_result = repo_manager.push_branch(workspace, branch, repo_full_name)
    if not push_result["success"]:
        say(f"[bold red]push failed[/bold red]: {push_result['reason']}")
        return push_result, None
    say(f"[green]pushed[/green] {branch} -> origin")

    pr_auth = gate.authorize(
        "create_pull_request",
        {"repo": repo_full_name, "branch": branch, "base": base_branch},
        override_callback=override,
    )
    if not pr_auth.allowed:
        say(f"[yellow]pull request skipped[/yellow] ({pr_auth.tier}): {pr_auth.reason}")
        return push_result, None

    pull_request = github_client.create_pull_request(
        repo_full_name, branch, base_branch, _pr_title(issue), _pr_body(issue, trajectory)
    )
    say(f"[bold green]pull request opened[/bold green]: {pull_request['url']}")
    return push_result, pull_request


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
    confirmation_gate: Optional[ConfirmationGate] = None,
    auto_push: bool = False,
    pr_base_branch: str = "main",
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
        confirmation_gate: M6's safety layer, used here only for the push/PR
            step after a successful commit — NOT for the ReAct loop's own
            tool calls, which this function always runs ungated (see the
            comment above the ``run_react_agent`` call for why). ``None``
            (the default) keeps this function's behaviour identical to M4:
            the run stops at the local commit, and push/PR is never
            attempted — this is the path every offline test in this project
            takes. When a gate is given (the CLI below always gives one), a
            successful commit is followed by an attempt to push the branch
            and open a pull request, each individually authorized through
            ``gate.authorize(...)`` — a denial (or a ``blocked``
            classification) at either step stops there without raising.
        auto_push: Only meaningful with ``confirmation_gate`` set. Push and
            PR creation still go through the gate either way; this only
            changes which callback answers for those two calls —
            pre-supplying "yes" instead of asking. It does not skip the
            gate, and it does not affect anything else the gate is asked to
            authorize.
        pr_base_branch: The branch a pull request is opened against, when one
            is opened. GitHub does not report a repository's default branch
            through this project's issue-fetching path, so this is a plain
            parameter rather than an auto-detected value — worth revisiting
            if a target repo's default branch is not ``"main"``.

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
        * ``push`` — :func:`repo_manager.push_branch`'s result dict, or
          ``None`` when push was never attempted (no gate, denial, or block)
        * ``pull_request`` — :func:`github_client.create_pull_request`'s
          result dict, or ``None`` when no PR was opened

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

    # Deliberately ungated: the gate here exists for the push/PR step below,
    # not the agent's own turns. Pausing for a confirmation on every
    # write_file during a real, many-turn run against a real issue would be
    # a different (and much more tedious) product than "review the push".
    # A run wanting the loop itself gated can call run_react_agent directly
    # with its own ConfirmationGate — see agent/core/react_agent.py's CLI.
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
    push: Optional[dict] = None
    pull_request: Optional[dict] = None
    if trajectory.final_success:
        commit = repo_manager.commit_changes(
            workspace, f"Fix: {issue['title']} (closes #{issue['number']})"
        )
        if commit["success"] and confirmation_gate is not None:
            push, pull_request = _push_and_open_pr(
                repo_full_name=repo_full_name,
                branch=branch,
                base_branch=pr_base_branch,
                issue=issue,
                trajectory=trajectory,
                workspace=workspace,
                gate=confirmation_gate,
                auto_push=auto_push,
                say=say,
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
        "push": push,
        "pull_request": pull_request,
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
    confirm: bool = typer.Option(
        True,
        "--confirm/--no-confirm",
        help=(
            "Attach M6's safety gate to the push/PR step that follows a "
            "successful commit — on by default, since this runs against a "
            "real repository and can open a real pull request. The ReAct "
            "loop's own tool calls (write_file, etc.) are never gated here "
            "regardless of this flag; see the M6 decision log in "
            "planning.md for why. --no-confirm never attempts push/PR at "
            "all, matching every milestone before M6."
        ),
    ),
    auto_push: bool = typer.Option(
        False,
        "--auto-push",
        help=(
            "Only matters with --confirm (the default). The push and PR "
            "steps still go through the gate — this just pre-supplies "
            "'yes' for those two calls instead of prompting interactively."
        ),
    ),
    pr_base_branch: str = typer.Option(
        "main",
        "--pr-base",
        help="Branch a pull request is opened against, if one is opened.",
    ),
) -> None:
    """Fetch the issue, clone, branch, iterate, commit, and (if approved) push/PR."""
    console.print(
        Panel(
            f"[bold]repo[/bold]      {repo}\n"
            f"[bold]issue[/bold]     #{issue}\n"
            f"[bold]model[/bold]     {model or llm_client.get_model()}\n"
            f"[bold]max turns[/bold] {max_turns}\n"
            f"[bold]sandbox[/bold]   {sandbox}\n"
            f"[bold]confirm[/bold]   {confirm}\n"
            f"[bold]auto-push[/bold] {auto_push}",
            title="agent on a GitHub issue",
            border_style="cyan",
        )
    )

    gate = ConfirmationGate() if confirm else None

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
            confirmation_gate=gate,
            auto_push=auto_push,
            pr_base_branch=pr_base_branch,
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
    push: Optional[dict] = result["push"]
    pull_request: Optional[dict] = result["pull_request"]

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

    if result["committed"]:
        if push is None:
            table.add_row("push", "[yellow]not attempted[/yellow]")
        elif push["success"]:
            table.add_row("push", "[bold green]pushed[/bold green]")
        else:
            table.add_row("push", "[bold red]failed[/bold red]")
            table.add_row("push reason", push["reason"] or "(no reason given)")

        if pull_request is not None:
            table.add_row(
                "pull request", f"[bold green]opened[/bold green] {pull_request['url']}"
            )

    out.print(
        Panel(
            table,
            title="run summary",
            border_style="green" if result["committed"] else "yellow",
        )
    )

    if not result["committed"]:
        note = (
            "Nothing was committed, so there is nothing to review. No push "
            "or PR was attempted."
        )
    elif pull_request is not None:
        note = f"Pull request opened: {pull_request['url']}"
    elif push is not None and push["success"]:
        note = (
            f"Branch {result['branch']} was pushed to origin, but no pull "
            "request was opened. Review before opening one."
        )
    else:
        note = NO_PUSH_NOTE.format(branch=result["branch"])

    out.print(Panel(note, border_style="yellow"))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
