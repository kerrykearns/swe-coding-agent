"""Tests for the end-to-end issue → clone → branch → agent → commit flow.

The offline tests replace exactly two things: the GitHub API (a stub client, as
in ``test_github_client``) and the clone (a copy of the local calculator fixture,
turned into a real git repository). Everything downstream of those is the real
thing — a real branch, a real ReAct loop driven by the scripted client from
``test_react_agent``, real files written, real pytest, a real commit. That is
what makes them worth running: the flow's failure modes are ordering and
reporting failures, and a mock of git or of the loop would hide both.

The scripted client, its reply/tool-call builders, and the fixed calculator
source are imported from ``test_react_agent`` rather than copied, so there is one
definition of "what the OpenAI SDK's response shape is" in this suite.

One integration test does the whole thing for real: real GitHub, real clone,
real Groq, real commit.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

from agent.core import github_client, llm_client, repo_manager
from agent.core.react_agent import StopReason, Trajectory, _print_turn
from agent.core.run_on_issue import (
    DEFAULT_WORKSPACES_DIR,
    NO_PUSH_NOTE,
    print_run_summary,
    run_on_issue,
)
from agent.safety import ConfirmationGate
from agent.tools import Workspace, read_file

from .conftest import CALCULATOR_FIXTURE, git, init_repo
from .test_github_client import TEST_REPO, _StubClient, _StubIssue, _StubRepo
from .test_react_agent import FIXED_CALC, _call, _reply, _scripted_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The real issue #1, as the playground repository has it.
ISSUE_TITLE = "add() gives the wrong result"
ISSUE_BODY = "add(2, 3) returns -1 instead of 5."

_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")


# --------------------------------------------------------------------------
# Offline scaffolding
# --------------------------------------------------------------------------


@pytest.fixture
def stub_github() -> _StubClient:
    """A GitHub client that serves issue #1 of the playground, offline."""
    return _StubClient(
        _StubRepo(
            issues={
                1: _StubIssue(
                    1,
                    title=ISSUE_TITLE,
                    body=ISSUE_BODY,
                    html_url=f"https://github.com/{TEST_REPO}/issues/1",
                )
            }
        )
    )


class _FakeClone:
    """Stands in for ``clone_repo``: copies the local fixture and inits a repo.

    The result is indistinguishable from a real clone for everything downstream
    — a git repository on a ``main`` branch with one commit of buggy code, with
    the agent's own test artifacts excluded exactly as ``clone_repo`` does — but
    it needs no network. Records where it was asked to clone to, since *where*
    is part of the behaviour under test.
    """

    def __init__(self) -> None:
        self.destinations: list[Path] = []

    def __call__(self, repo_full_name: str, dest_dir, **kwargs) -> Workspace:
        destination = Path(dest_dir)
        self.destinations.append(destination)
        shutil.copytree(CALCULATOR_FIXTURE, destination, ignore=_COPY_IGNORE)
        init_repo(destination, "buggy calculator")
        workspace = Workspace(destination)
        repo_manager.exclude_agent_artifacts(workspace)
        return workspace


@pytest.fixture
def fake_clone(monkeypatch) -> _FakeClone:
    fake = _FakeClone()
    monkeypatch.setattr(repo_manager, "clone_repo", fake)
    return fake


def _fixing_client():
    """A scripted LLM client that writes the fix, then runs the tests."""
    return _scripted_client(
        _reply(
            "The operator is inverted; rewriting the file.",
            (_call("write_file", path="calculator/calc.py", content=FIXED_CALC),),
        ),
        _reply("Now verifying.", (_call("run_tests", test_path="."),), tokens=250),
    )


def _giving_up_client():
    """A scripted client that edits, then claims success without ever testing.

    M3's rule: ``final_success`` is only True if a test run actually reported
    success, so this run does not succeed however confidently it finishes.
    """
    return _scripted_client(
        _reply(
            "",
            (_call("write_file", path="calculator/calc.py", content=FIXED_CALC),),
        ),
        _reply(
            "",
            (_call("finish", success=True, summary="I am sure I fixed it."),),
        ),
    )


def _run(tmp_path: Path, stub_github, client, **kwargs) -> dict:
    return run_on_issue(
        TEST_REPO,
        1,
        max_turns=4,
        workspaces_dir=tmp_path / "workspaces",
        client=client,
        github=stub_github,
        **kwargs,
    )


def _subject(ws: Workspace) -> str:
    return git(ws.root, "log", "-1", "--pretty=%s").stdout.strip()


def _branch(ws: Workspace) -> str:
    return git(ws.root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_flow_fetches_branches_runs_and_commits(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())

    assert result["repo"] == TEST_REPO
    assert result["issue"]["number"] == 1
    assert result["issue"]["title"] == ISSUE_TITLE
    assert result["branch"] == "agent-fix/issue-1"
    assert result["trajectory"].final_success is True
    assert result["trajectory"].stop_reason is StopReason.TESTS_PASSED
    assert result["committed"] is True
    assert result["commit"]["success"] is True


def test_the_repository_name_is_normalised_before_anything_else(
    tmp_path, stub_github, fake_clone
):
    result = run_on_issue(
        f"  {TEST_REPO}  ",
        1,
        max_turns=4,
        workspaces_dir=tmp_path / "workspaces",
        client=_fixing_client(),
        github=stub_github,
    )
    assert result["repo"] == TEST_REPO
    assert stub_github.repos_asked == [TEST_REPO]


def test_the_branch_is_created_and_checked_out(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    assert _branch(result["workspace"]) == "agent-fix/issue-1"


def test_the_agent_works_on_the_branch_not_on_main(tmp_path, stub_github, fake_clone):
    """The commit must land on the agent's branch, leaving main untouched."""
    result = _run(tmp_path, stub_github, _fixing_client())
    workspace = result["workspace"]

    on_branch = git(workspace.root, "log", "--oneline", "agent-fix/issue-1").stdout
    on_main = git(workspace.root, "log", "--oneline", "main").stdout
    assert len(on_branch.splitlines()) == 2
    assert len(on_main.splitlines()) == 1


def test_the_commit_message_names_the_issue_and_closes_it(
    tmp_path, stub_github, fake_clone
):
    result = _run(tmp_path, stub_github, _fixing_client())
    assert _subject(result["workspace"]) == f"Fix: {ISSUE_TITLE} (closes #1)"


def test_the_fix_is_actually_in_the_commit(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    workspace = result["workspace"]

    assert "return a + b" in read_file(workspace, "calculator/calc.py")
    assert git(workspace.root, "status", "--porcelain").stdout.strip() == ""
    assert "calculator/calc.py" in git(
        workspace.root, "show", "--name-only", "--pretty=", "HEAD"
    ).stdout


def test_the_commit_contains_the_fix_and_nothing_else(tmp_path, stub_github, fake_clone):
    """The agent ran pytest in this checkout, so the caches are real, not staged
    by the test. Committing them was the first live run's actual behaviour."""
    result = _run(tmp_path, stub_github, _fixing_client())
    workspace = result["workspace"]

    assert (workspace.root / "calculator" / "__pycache__").exists(), (
        "pytest left no caches, so this test would pass without proving anything"
    )
    committed = git(
        workspace.root, "show", "--name-only", "--pretty=", "HEAD"
    ).stdout.splitlines()
    assert committed == ["calculator/calc.py"]


def test_the_task_description_is_the_formatted_issue(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    assert result["task_description"] == (
        f"GitHub Issue #1: {ISSUE_TITLE}\n\n{ISSUE_BODY}"
    )
    assert result["trajectory"].task_description == result["task_description"]


def test_the_trajectory_is_the_m3_trajectory(tmp_path, stub_github, fake_clone):
    """M8 measures these runs, so the record has to be the same object it knows."""
    result = _run(tmp_path, stub_github, _fixing_client())
    trajectory = result["trajectory"]
    assert isinstance(trajectory, Trajectory)
    assert [turn.tool_name for turn in trajectory.turns] == ["write_file", "run_tests"]
    assert trajectory.total_turns == 2
    assert trajectory.total_tokens > 0
    assert "+    return a + b" in trajectory.final_diff


def test_the_turn_callback_is_forwarded_to_the_loop(tmp_path, stub_github, fake_clone):
    seen = []
    _run(tmp_path, stub_github, _fixing_client(), on_turn=seen.append)
    assert [turn.tool_name for turn in seen] == ["write_file", "run_tests"]


# --------------------------------------------------------------------------
# The unsuccessful path
# --------------------------------------------------------------------------


def test_nothing_is_committed_when_the_agent_does_not_succeed(
    tmp_path, stub_github, fake_clone
):
    result = _run(tmp_path, stub_github, _giving_up_client())

    assert result["trajectory"].final_success is False
    assert result["trajectory"].stop_reason is StopReason.AGENT_FINISHED
    assert result["commit"] is None
    assert result["committed"] is False


def test_an_unsuccessful_run_leaves_its_work_uncommitted_for_review(
    tmp_path, stub_github, fake_clone
):
    """The edits stay on disk: the point is that *nothing was hidden*, not that
    the work was thrown away."""
    result = _run(tmp_path, stub_github, _giving_up_client())
    workspace = result["workspace"]

    assert len(git(workspace.root, "log", "--oneline").stdout.splitlines()) == 1
    assert git(workspace.root, "status", "--porcelain").stdout.strip() != ""
    assert "return a + b" in read_file(workspace, "calculator/calc.py")


def test_an_unsuccessful_run_says_so_where_the_user_can_see_it(
    tmp_path, stub_github, fake_clone
):
    buffer = io.StringIO()
    _run(
        tmp_path,
        stub_github,
        _giving_up_client(),
        console=Console(file=buffer, width=100, no_color=True),
    )
    printed = buffer.getvalue()
    assert "no changes committed" in printed
    assert "did not reach success" in printed
    assert "agent_finished" in printed


def test_a_commit_that_finds_nothing_to_commit_is_reported(
    tmp_path, stub_github, fake_clone, monkeypatch
):
    """The agent can verify a green suite without having changed anything.

    ``commit_changes`` calls that success=False with a reason; the flow must
    pass the reason through rather than claim a commit that never happened.
    """
    monkeypatch.setattr(
        repo_manager,
        "commit_changes",
        lambda ws, message: {
            "success": False,
            "commit_hash": None,
            "output": "$ git add -A (exit 0)",
            "reason": "Nothing to commit — the working tree is clean.",
        },
    )
    result = _run(tmp_path, stub_github, _fixing_client())

    assert result["trajectory"].final_success is True
    assert result["committed"] is False
    assert "Nothing to commit" in result["commit"]["reason"]


def test_a_github_failure_stops_before_anything_is_cloned(tmp_path, fake_clone):
    """No point cloning a repository whose issue could not be read."""
    client = _StubClient(_StubRepo(issues={}))
    with pytest.raises(github_client.IssueNotFoundError):
        run_on_issue(
            TEST_REPO,
            999,
            workspaces_dir=tmp_path / "workspaces",
            client=_fixing_client(),
            github=client,
        )
    assert fake_clone.destinations == []


# --------------------------------------------------------------------------
# Where the clone goes
# --------------------------------------------------------------------------


def test_the_clone_goes_under_the_workspaces_directory(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    workspaces = (tmp_path / "workspaces").resolve()
    assert Path(result["workspace_path"]).is_relative_to(workspaces)


def test_the_clone_directory_names_the_repo_and_the_issue(
    tmp_path, stub_github, fake_clone
):
    result = _run(tmp_path, stub_github, _fixing_client())
    owner, name = TEST_REPO.split("/")
    assert Path(result["workspace_path"]).name == f"{owner}-{name}-issue-1"


def test_the_workspaces_directory_is_created_if_missing(tmp_path, stub_github, fake_clone):
    workspaces = tmp_path / "does" / "not" / "exist"
    run_on_issue(
        TEST_REPO,
        1,
        max_turns=4,
        workspaces_dir=workspaces,
        client=_fixing_client(),
        github=stub_github,
    )
    assert workspaces.is_dir()


def test_a_second_run_does_not_reuse_the_first_clone(tmp_path, stub_github, fake_clone):
    """A stale working tree would make the second run's diff meaningless."""
    first = _run(tmp_path, stub_github, _fixing_client())
    second = _run(tmp_path, stub_github, _fixing_client())

    assert first["workspace_path"] != second["workspace_path"]
    assert Path(second["workspace_path"]).name.endswith("-2")
    assert Path(first["workspace_path"]).exists()


def test_the_default_workspaces_directory_is_gitignored():
    """Spec item 6, asserted rather than eyeballed.

    A run leaves a full checkout of somebody else's repository — including its
    own ``.git`` — under ``workspaces/``. ``git check-ignore`` is asked directly,
    so this passes only if the real rule in .gitignore covers the real path.
    """
    assert DEFAULT_WORKSPACES_DIR == PROJECT_ROOT / "workspaces"

    completed = subprocess.run(
        ["git", "check-ignore", "-v", "workspaces/owner-repo-issue-1/calc.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "workspaces/ is not gitignored — agent clones would be committable: "
        f"{completed.stderr.strip()}"
    )
    assert "workspaces/" in completed.stdout


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _rendered_summary(result: dict) -> str:
    buffer = io.StringIO()
    print_run_summary(result, Console(file=buffer, width=120, no_color=True))
    return buffer.getvalue()


def test_the_summary_reports_the_issue_the_branch_and_the_commit(
    tmp_path, stub_github, fake_clone
):
    result = _run(tmp_path, stub_github, _fixing_client())
    printed = _rendered_summary(result)

    assert TEST_REPO in printed
    assert "#1" in printed and ISSUE_TITLE in printed
    assert f"https://github.com/{TEST_REPO}/issues/1" in printed
    assert "agent-fix/issue-1" in printed
    assert "tests_passed" in printed
    assert "committed" in printed
    assert result["commit"]["commit_hash"][:8] in printed


def test_the_summary_reports_the_trajectory_numbers(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    printed = _rendered_summary(result)
    assert f"{result['trajectory'].total_turns}" in printed
    assert f"{result['trajectory'].total_tokens:,}" in printed


def test_the_summary_ends_with_the_no_push_note(tmp_path, stub_github, fake_clone):
    """The one thing a reader must not get wrong: this was not delivered.

    ``_run`` calls ``run_on_issue`` with no ``confirmation_gate`` — the
    library default — so push/PR is never attempted, exactly as in M4.
    """
    result = _run(tmp_path, stub_github, _fixing_client())
    printed = _rendered_summary(result)

    assert "Review before pushing" in printed
    assert "not completed this run" in printed
    assert result["push"] is None
    assert result["pull_request"] is None


def test_the_no_push_note_names_the_branch(tmp_path, stub_github, fake_clone):
    result = _run(tmp_path, stub_github, _fixing_client())
    assert NO_PUSH_NOTE.format(branch="agent-fix/issue-1").startswith(
        "Changes committed locally on branch agent-fix/issue-1."
    )
    assert "agent-fix/issue-1" in _rendered_summary(result)


def test_everything_printed_survives_a_legacy_windows_console(
    tmp_path, stub_github, fake_clone
):
    """Learned from the first live run, which died before the clone started.

    On a legacy Windows console rich encodes its output with the OS codepage
    (cp1252 here), not UTF-8, so one decorative arrow in a progress line raised
    UnicodeEncodeError and took the whole run with it. Writing through a
    strict cp1252 stream reproduces that exactly: any character the console
    cannot encode fails this test instead of a live run.
    """
    legacy = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", write_through=True
    )
    console = Console(file=legacy, width=100, no_color=True)

    result = _run(tmp_path, stub_github, _fixing_client(), console=console)
    print_run_summary(result, console)


def test_the_summary_does_not_claim_a_commit_that_did_not_happen(
    tmp_path, stub_github, fake_clone
):
    result = _run(tmp_path, stub_github, _giving_up_client())
    printed = _rendered_summary(result)

    assert "not attempted" in printed
    assert "did not reach success" in printed
    assert "nothing to review" in printed
    assert "Changes committed locally" not in printed


# --------------------------------------------------------------------------
# M6: push/PR is opt-in, and only ever runs behind the confirmation gate
# --------------------------------------------------------------------------


def test_no_gate_means_no_push_and_no_pull_request(
    tmp_path, stub_github, fake_clone, monkeypatch
):
    """The M4 behaviour survives M6 unchanged when no gate is supplied.

    ``_run`` (like every other test in this file) calls ``run_on_issue`` with
    no ``confirmation_gate`` — the library default. Every git command the
    flow issues is recorded and checked, and ``create_pull_request`` is
    replaced with a landmine. Asserted behaviourally because the source of
    these modules mentions both actions in the docstrings that explain why
    they are absent here.
    """
    recorded: list[str] = []
    real = repo_manager.run_command

    def record(workspace, command, timeout=30):
        recorded.append(command)
        return real(workspace, command, timeout=timeout)

    def explode(*args, **kwargs):
        raise AssertionError("create_pull_request was called with no gate supplied")

    monkeypatch.setattr(repo_manager, "run_command", record)
    monkeypatch.setattr(github_client, "create_pull_request", explode)

    result = _run(tmp_path, stub_github, _fixing_client())

    assert result["committed"] is True
    assert result["push"] is None
    assert result["pull_request"] is None
    assert recorded, "no git commands were recorded, so this proves nothing"
    assert not any("push" in command for command in recorded), recorded
    assert not any("remote add" in command for command in recorded), recorded


def test_a_gate_that_approves_pushes_and_opens_a_pull_request(
    tmp_path, stub_github, fake_clone, monkeypatch
):
    """With a gate that says yes, push and PR both happen, in order."""
    calls: list[str] = []

    def fake_push(ws, branch_name, repo_full_name, **kwargs):
        calls.append(f"push:{branch_name}")
        return {"success": True, "output": "pushed", "reason": None}

    def fake_create_pr(repo_full_name, branch, base, title, body, client=None):
        calls.append(f"pr:{branch}->{base}")
        return {
            "number": 7,
            "url": f"https://github.com/{repo_full_name}/pull/7",
            "title": title,
            "state": "open",
            "head": branch,
            "base": base,
        }

    monkeypatch.setattr(repo_manager, "push_branch", fake_push)
    monkeypatch.setattr(github_client, "create_pull_request", fake_create_pr)

    gate = ConfirmationGate(confirm_callback=lambda *_args: True)
    result = _run(tmp_path, stub_github, _fixing_client(), confirmation_gate=gate)

    assert calls == ["push:agent-fix/issue-1", "pr:agent-fix/issue-1->main"]
    assert result["push"]["success"] is True
    assert result["pull_request"]["number"] == 7
    assert result["pull_request"]["url"].endswith("/pull/7")

    # Both calls were classified needs_confirmation and actually asked.
    decisions = {entry.tool_name: entry.decision for entry in gate.audit_log}
    assert decisions["push_branch"] == "confirmed"
    assert decisions["create_pull_request"] == "confirmed"


def test_a_gate_that_denies_stops_before_pushing(
    tmp_path, stub_github, fake_clone, monkeypatch
):
    """A 'no' from the callback must actually stop the flow, not just log it."""

    def explode_push(*args, **kwargs):
        raise AssertionError("push_branch was called despite the gate denying it")

    def explode_pr(*args, **kwargs):
        raise AssertionError("create_pull_request was called despite the gate denying it")

    monkeypatch.setattr(repo_manager, "push_branch", explode_push)
    monkeypatch.setattr(github_client, "create_pull_request", explode_pr)

    gate = ConfirmationGate(confirm_callback=lambda *_args: False)
    result = _run(tmp_path, stub_github, _fixing_client(), confirmation_gate=gate)

    assert result["committed"] is True
    assert result["push"] is None
    assert result["pull_request"] is None
    assert gate.audit_log[-1].decision == "denied"


def test_auto_push_pre_supplies_yes_without_skipping_the_gate(
    tmp_path, stub_github, fake_clone, monkeypatch
):
    """``auto_push`` changes the answer, not whether the question is asked.

    The gate's own callback would deny everything; ``auto_push=True`` must
    still make the push/PR step succeed, by overriding just that callback —
    proving the gate itself was still consulted (and logged), not bypassed.
    """
    monkeypatch.setattr(
        repo_manager,
        "push_branch",
        lambda ws, branch_name, repo_full_name, **kw: {
            "success": True,
            "output": "pushed",
            "reason": None,
        },
    )
    monkeypatch.setattr(
        github_client,
        "create_pull_request",
        lambda repo_full_name, branch, base, title, body, client=None: {
            "number": 1,
            "url": f"https://github.com/{repo_full_name}/pull/1",
            "title": title,
            "state": "open",
            "head": branch,
            "base": base,
        },
    )

    gate = ConfirmationGate(confirm_callback=lambda *_args: False)
    result = _run(
        tmp_path, stub_github, _fixing_client(), confirmation_gate=gate, auto_push=True
    )

    assert result["push"]["success"] is True
    assert result["pull_request"]["number"] == 1
    # The gate really was asked — it just was not the deciding voice.
    decisions = [entry.decision for entry in gate.audit_log]
    assert "confirmed" in decisions
    assert "denied" not in decisions


# --------------------------------------------------------------------------
# integration: the real thing, end to end
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not github_client.token_present() or not llm_client.api_key_present(),
    reason="GITHUB_TOKEN and GROQ_API_KEY are both required for the live run",
)
def test_the_full_flow_against_the_real_playground_repo(tmp_path, capsys):
    """Real GitHub fetch, real clone, real ReAct loop via Groq, real local commit.

    Everything is printed: this test is as much a demonstration as an assertion.
    Nothing is pushed — the last thing it checks is that the clone's remote
    carries no credential and that the commit exists only locally.
    """
    with capsys.disabled():
        console = Console(width=100)
        console.rule("[bold]M4 live end-to-end run[/bold]")

        # 12 turns, not M3's 8. A live run against the real issue text reliably
        # loses its first two or three turns to Groq rejecting malformed tool
        # calls (see the decision log), and the first attempt at this test spent
        # five of eight turns that way and never reached a write. The budget is
        # sized for the backend's observed behaviour rather than the assertions
        # below being softened to match it.
        result = run_on_issue(
            TEST_REPO,
            1,
            max_turns=12,
            workspaces_dir=tmp_path / "workspaces",
            console=console,
            on_turn=_print_turn,
        )

        trajectory = result["trajectory"]
        print(f"\n{'=' * 78}\nFINAL DIFF\n{'=' * 78}")
        print(trajectory.final_diff or "(empty)")
        print(f"{'=' * 78}\nCOMMIT TRANSCRIPT\n{'=' * 78}")
        print((result["commit"] or {}).get("output") or "(no commit attempted)")
        print("=" * 78)
        print_run_summary(result, console)

    workspace = result["workspace"]
    trajectory = result["trajectory"]

    # The issue really came from GitHub.
    assert result["issue"]["number"] == 1
    assert result["issue"]["title"] == ISSUE_TITLE
    assert result["task_description"].startswith(f"GitHub Issue #1: {ISSUE_TITLE}")

    # The clone is a real checkout of the real repository, on the agent's branch.
    assert (workspace.root / ".git").exists()
    assert _branch(workspace) == "agent-fix/issue-1"

    # The agent really iterated, and really verified its own work.
    assert trajectory.stop_reason is not StopReason.ERROR
    used = [turn.tool_name for turn in trajectory.turns]
    assert "write_file" in used, f"the agent never edited anything: {used}"
    assert "run_tests" in used, f"the agent never verified its work: {used}"
    assert trajectory.final_success is True, (
        f"stopped as {trajectory.stop_reason.value} without passing tests"
    )
    assert "return a + b" in read_file(workspace, "calculator/calc.py")

    # And the work is committed locally, on the branch, with a clean tree.
    assert result["committed"] is True
    assert len(result["commit"]["commit_hash"]) == 40
    assert _subject(workspace) == f"Fix: {ISSUE_TITLE} (closes #1)"
    assert git(workspace.root, "status", "--porcelain").stdout.strip() == ""

    # Nothing was pushed, and no credential was left behind in .git/config.
    remote = git(workspace.root, "remote", "get-url", "origin").stdout.strip()
    assert remote == f"https://github.com/{TEST_REPO}.git"
    assert "x-access-token" not in remote
    ahead = git(
        workspace.root, "rev-list", "--count", "origin/HEAD..agent-fix/issue-1"
    ).stdout.strip()
    assert ahead == "1", "the local branch should be exactly one unpushed commit ahead"
