"""Tests for the GitHub client.

Everything here runs offline except the one test marked ``integration``. PyGithub
is never allowed to make a request: each test passes a stub client into the
``client=`` parameter, so the code under test walks its real code path
(``get_repo`` → ``get_issue`` → projection) against objects that answer in the
shapes PyGithub answers in.

The exceptions the stubs raise are *real* ``GithubException`` instances, built
with the status and payload GitHub actually sends. That is deliberate: the M3
decision log records a hand-made fake exception that encoded a wrong guess about
its own shape, passed happily, and let the agent break in a live run. A fake that
can be wrong about the thing being tested is worse than no test.
"""

from __future__ import annotations

import os

import pytest
from github import Github
from github.GithubException import (
    BadCredentialsException,
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)

from agent.core import github_client
from agent.core.github_client import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubError,
    IssueNotFoundError,
    RepoNotFoundError,
    create_pull_request,
    get_client,
    get_issue,
    get_token,
    list_open_issues,
    normalise_repo_name,
    token_present,
)

#: The real repository the integration test reads. Overridable so the test is
#: not welded to one developer's account.
TEST_REPO = os.getenv("GITHUB_TEST_REPO") or "kerrykearns/agent-test-playground"

#: A token-shaped string. Never a real token, and never sent anywhere.
FAKE_TOKEN = "github_pat_fake_000000000000000000"


# --------------------------------------------------------------------------
# Stubs: PyGithub's shapes, none of its network
# --------------------------------------------------------------------------


class _StubIssue:
    """The subset of ``github.Issue.Issue`` this module reads."""

    def __init__(
        self,
        number: int,
        title: str = "a title",
        body: str | None = "a body",
        state: str = "open",
        html_url: str | None = None,
        pull_request: object = None,
    ) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.state = state
        self.html_url = html_url or f"https://github.com/owner/repo/issues/{number}"
        # PyGithub exposes ``pull_request`` as None on a real issue and as an
        # object on a PR. GitHub's issues endpoint returns both.
        self.pull_request = pull_request


class _StubPull:
    """The subset of ``github.PullRequest.PullRequest`` this module reads."""

    def __init__(self, number: int = 42, title: str = "a PR", state: str = "open"):
        self.number = number
        self.title = title
        self.state = state
        self.html_url = f"https://github.com/owner/repo/pull/{number}"


class _StubRepo:
    """Answers the three repo calls, or raises whatever it was told to raise."""

    def __init__(
        self,
        issues: dict | None = None,
        open_issues: list | None = None,
        raises: Exception | None = None,
        pull: _StubPull | None = None,
    ) -> None:
        self.issues = issues or {}
        self.open_issues = open_issues or []
        self.raises = raises
        self.pull = pull or _StubPull()
        self.created_pulls: list[dict] = []
        self.issues_state_asked: list[str] = []

    def get_issue(self, number: int):
        if self.raises is not None:
            raise self.raises
        if number not in self.issues:
            raise UnknownObjectException(404, {"message": "Not Found"}, {})
        return self.issues[number]

    def get_issues(self, state: str = "open"):
        self.issues_state_asked.append(state)
        if self.raises is not None:
            raise self.raises
        # A real PaginatedList is lazy: iterating it is what makes the requests.
        return iter(self.open_issues)

    def create_pull(self, title: str, body: str, head: str, base: str):
        if self.raises is not None:
            raise self.raises
        self.created_pulls.append(
            {"title": title, "body": body, "head": head, "base": base}
        )
        return self.pull


class _StubClient:
    """Stands in for ``github.Github``, recording which repo was asked for."""

    def __init__(self, repo: _StubRepo | None = None, raises: Exception | None = None):
        self.repo = repo
        self.raises = raises
        self.repos_asked: list[str] = []

    def get_repo(self, full_name: str):
        self.repos_asked.append(full_name)
        if self.raises is not None:
            raise self.raises
        return self.repo


def _client_with_issue(**kwargs) -> _StubClient:
    """A client whose repo has exactly issue #1."""
    return _StubClient(_StubRepo(issues={1: _StubIssue(1, **kwargs)}))


# --------------------------------------------------------------------------
# Token handling
# --------------------------------------------------------------------------


def test_token_present_is_false_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert token_present() is False


def test_token_present_is_false_for_the_env_example_placeholder(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    assert token_present() is False


def test_token_present_is_false_for_whitespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert token_present() is False


def test_token_present_is_true_for_a_real_looking_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    assert token_present() is True


def test_get_token_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", f"  {FAKE_TOKEN}\n")
    assert get_token() == FAKE_TOKEN


def test_get_token_raises_an_actionable_error_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitHubAuthError) as caught:
        get_token()
    message = str(caught.value)
    assert "GITHUB_TOKEN is not set" in message
    # The message has to say what to do, not just what went wrong.
    assert ".env" in message and "github.com/settings/tokens" in message


def test_get_client_refuses_to_build_without_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitHubAuthError):
        get_client()


def test_get_client_builds_a_pygithub_client_when_configured(monkeypatch):
    """Construction is offline: PyGithub contacts nothing until a call is made."""
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    assert isinstance(get_client(), Github)


def test_every_error_is_catchable_as_one_class():
    """A caller that only wants to print the message needs one except clause."""
    for exception in (GitHubAuthError, RepoNotFoundError, IssueNotFoundError, GitHubAPIError):
        assert issubclass(exception, GitHubError)
    assert issubclass(GitHubError, RuntimeError)


# --------------------------------------------------------------------------
# Repository names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("owner/repo", "owner/repo"),
        ("  owner/repo  ", "owner/repo"),
        ("/owner/repo/", "owner/repo"),
        ("owner / repo", "owner/repo"),
    ],
)
def test_normalise_repo_name_accepts_and_trims_owner_slash_repo(given, expected):
    assert normalise_repo_name(given) == expected


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "repo", "owner/repo/extra", "owner/", "/repo", "owner//repo"],
)
def test_normalise_repo_name_rejects_anything_else(bad):
    with pytest.raises(RepoNotFoundError) as caught:
        normalise_repo_name(bad)
    assert "owner/repo" in str(caught.value)


def test_a_malformed_repo_name_costs_no_network_call():
    """The name is validated before the client is touched."""
    client = _StubClient(_StubRepo())
    with pytest.raises(RepoNotFoundError):
        get_issue("not-a-repo-name", 1, client=client)
    assert client.repos_asked == []


# --------------------------------------------------------------------------
# get_issue
# --------------------------------------------------------------------------


def test_get_issue_returns_exactly_the_five_documented_fields():
    client = _client_with_issue(
        title="add() gives the wrong result",
        body="add(2, 3) returns -1",
        state="open",
        html_url="https://github.com/owner/repo/issues/1",
    )
    issue = get_issue("owner/repo", 1, client=client)
    assert issue == {
        "number": 1,
        "title": "add() gives the wrong result",
        "body": "add(2, 3) returns -1",
        "url": "https://github.com/owner/repo/issues/1",
        "state": "open",
    }


def test_get_issue_asks_for_the_normalised_repository():
    client = _client_with_issue()
    get_issue("  owner/repo  ", 1, client=client)
    assert client.repos_asked == ["owner/repo"]


def test_get_issue_turns_a_null_body_into_an_empty_string():
    """GitHub sends ``null`` for an issue with no description; that is not an error."""
    client = _client_with_issue(body=None)
    assert get_issue("owner/repo", 1, client=client)["body"] == ""


def test_get_issue_reports_a_missing_repository_distinctly():
    client = _StubClient(raises=UnknownObjectException(404, {"message": "Not Found"}, {}))
    with pytest.raises(RepoNotFoundError) as caught:
        get_issue("owner/nope", 1, client=client)
    message = str(caught.value)
    assert "owner/nope" in message
    # The 404-for-private-repo trap is the whole reason this message is long.
    assert "404" in message and "private" in message


def test_get_issue_reports_a_missing_issue_distinctly():
    """The same 404 from a different call must not say "no such repository"."""
    client = _StubClient(_StubRepo(issues={1: _StubIssue(1)}))
    with pytest.raises(IssueNotFoundError) as caught:
        get_issue("owner/repo", 999, client=client)
    assert "has no issue #999" in str(caught.value)


def test_a_missing_issue_is_not_a_missing_repository():
    assert not issubclass(IssueNotFoundError, RepoNotFoundError)
    assert not issubclass(RepoNotFoundError, IssueNotFoundError)


def test_get_issue_reports_a_dead_token_as_an_auth_error():
    client = _StubClient(
        raises=BadCredentialsException(401, {"message": "Bad credentials"}, {})
    )
    with pytest.raises(GitHubAuthError) as caught:
        get_issue("owner/repo", 1, client=client)
    message = str(caught.value)
    assert "Bad credentials" in message
    assert "GITHUB_TOKEN" in message


def test_get_issue_reports_a_403_as_an_auth_error():
    """A 403 means "change the token", so it is an auth failure, not a generic one."""
    client = _StubClient(
        raises=GithubException(403, {"message": "Resource not accessible"}, {})
    )
    with pytest.raises(GitHubAuthError) as caught:
        get_issue("owner/repo", 1, client=client)
    assert "403" in str(caught.value)
    assert "scopes" in str(caught.value)


def test_get_issue_reports_a_rate_limit_as_an_api_error():
    client = _StubClient(
        raises=RateLimitExceededException(
            403, {"message": "API rate limit exceeded"}, {}
        )
    )
    with pytest.raises(GitHubAPIError) as caught:
        get_issue("owner/repo", 1, client=client)
    message = str(caught.value)
    assert "rate limit" in message.lower()
    assert "wait" in message


def test_rate_limit_is_not_swallowed_by_the_generic_clause():
    """RateLimitExceededException subclasses GithubException with status 403.

    Without specific-before-generic ordering it would be reported as an auth
    error telling the user to change a token that is working perfectly.
    """
    client = _StubClient(
        raises=RateLimitExceededException(403, {"message": "API rate limit exceeded"}, {})
    )
    with pytest.raises(GitHubAPIError):
        get_issue("owner/repo", 1, client=client)


def test_get_issue_reports_a_server_error_as_an_api_error():
    client = _StubClient(raises=GithubException(502, {"message": "Bad gateway"}, {}))
    with pytest.raises(GitHubAPIError) as caught:
        get_issue("owner/repo", 1, client=client)
    assert "502" in str(caught.value)
    assert "Bad gateway" in str(caught.value)


def test_no_raw_pygithub_exception_escapes():
    """Whatever PyGithub raises, the caller sees a GitHubError."""
    for raised in (
        UnknownObjectException(404, {"message": "Not Found"}, {}),
        BadCredentialsException(401, {"message": "Bad credentials"}, {}),
        RateLimitExceededException(403, {"message": "rate limited"}, {}),
        GithubException(500, {"message": "boom"}, {}),
        GithubException(422, {"message": "Validation Failed"}, {}),
    ):
        client = _StubClient(raises=raised)
        with pytest.raises(GitHubError):
            get_issue("owner/repo", 1, client=client)


def test_error_messages_flatten_githubs_errors_array():
    """GitHub's useful detail lives in ``errors``, not ``message``."""
    client = _StubClient(
        raises=GithubException(
            422,
            {
                "message": "Validation Failed",
                "errors": [{"message": "No commits between main and topic"}],
            },
            {},
        )
    )
    with pytest.raises(GitHubAPIError) as caught:
        get_issue("owner/repo", 1, client=client)
    assert "Validation Failed" in str(caught.value)
    assert "No commits between main and topic" in str(caught.value)


def test_error_messages_survive_a_non_dict_payload():
    """PyGithub hands back whatever the server sent, which is not always JSON."""
    client = _StubClient(raises=GithubException(500, "upstream exploded", {}))
    with pytest.raises(GitHubAPIError) as caught:
        get_issue("owner/repo", 1, client=client)
    assert "upstream exploded" in str(caught.value)


def test_error_messages_survive_an_empty_payload():
    client = _StubClient(raises=GithubException(500, None, {}))
    with pytest.raises(GitHubAPIError):
        get_issue("owner/repo", 1, client=client)


# --------------------------------------------------------------------------
# list_open_issues
# --------------------------------------------------------------------------


def test_list_open_issues_returns_issue_shaped_dicts():
    client = _StubClient(
        _StubRepo(
            open_issues=[
                _StubIssue(3, title="third", body="c"),
                _StubIssue(1, title="first", body="a"),
            ]
        )
    )
    issues = list_open_issues("owner/repo", client=client)
    assert [issue["number"] for issue in issues] == [3, 1]
    assert set(issues[0]) == {"number", "title", "body", "url", "state"}


def test_list_open_issues_asks_github_only_for_open_ones():
    client = _StubClient(_StubRepo(open_issues=[]))
    list_open_issues("owner/repo", client=client)
    assert client.repo.issues_state_asked == ["open"]


def test_list_open_issues_excludes_pull_requests():
    """GitHub models a PR as an issue; an agent must never be handed one."""
    client = _StubClient(
        _StubRepo(
            open_issues=[
                _StubIssue(1, title="a real issue"),
                _StubIssue(2, title="a pull request", pull_request=object()),
                _StubIssue(3, title="another real issue"),
            ]
        )
    )
    issues = list_open_issues("owner/repo", client=client)
    assert [issue["number"] for issue in issues] == [1, 3]
    assert all("pull request" not in issue["title"] for issue in issues)


def test_list_open_issues_returns_an_empty_list_for_a_quiet_repo():
    client = _StubClient(_StubRepo(open_issues=[]))
    assert list_open_issues("owner/repo", client=client) == []


def test_list_open_issues_reports_a_missing_repository():
    client = _StubClient(raises=UnknownObjectException(404, {"message": "Not Found"}, {}))
    with pytest.raises(RepoNotFoundError):
        list_open_issues("owner/nope", client=client)


def test_list_open_issues_translates_a_failure_during_pagination():
    """Iterating a PaginatedList is what issues the requests, so the loop must
    happen inside the translating block — not after it."""
    client = _StubClient(
        _StubRepo(raises=RateLimitExceededException(403, {"message": "rate"}, {}))
    )
    with pytest.raises(GitHubAPIError):
        list_open_issues("owner/repo", client=client)


# --------------------------------------------------------------------------
# create_pull_request — built and tested, deliberately not wired in (M4)
# --------------------------------------------------------------------------


def test_create_pull_request_returns_the_new_pull_requests_details():
    repo = _StubRepo(pull=_StubPull(number=7, title="Fix: add()", state="open"))
    client = _StubClient(repo)
    result = create_pull_request(
        "owner/repo",
        branch="agent-fix/issue-1",
        base="main",
        title="Fix: add()",
        body="closes #1",
        client=client,
    )
    assert result == {
        "number": 7,
        "url": "https://github.com/owner/repo/pull/7",
        "title": "Fix: add()",
        "state": "open",
        "head": "agent-fix/issue-1",
        "base": "main",
    }


def test_create_pull_request_sends_branch_as_head_and_base_as_base():
    """Swapping these opens a PR in the wrong direction, which is hard to notice."""
    repo = _StubRepo()
    create_pull_request(
        "owner/repo",
        branch="agent-fix/issue-1",
        base="main",
        title="t",
        body="b",
        client=_StubClient(repo),
    )
    assert repo.created_pulls == [
        {
            "title": "t",
            "body": "b",
            "head": "agent-fix/issue-1",
            "base": "main",
        }
    ]


def test_create_pull_request_reports_an_unpushed_branch_clearly():
    """The 422 a caller will actually hit, since M4 never pushes anything."""
    client = _StubClient(
        _StubRepo(
            raises=GithubException(
                422,
                {
                    "message": "Validation Failed",
                    "errors": [{"message": "Invalid head branch"}],
                },
                {},
            )
        )
    )
    with pytest.raises(GitHubAPIError) as caught:
        create_pull_request(
            "owner/repo",
            branch="agent-fix/issue-1",
            base="main",
            title="t",
            body="b",
            client=client,
        )
    message = str(caught.value)
    assert "422" in message
    assert "Invalid head branch" in message
    assert "agent-fix/issue-1" in message


def test_create_pull_request_reports_missing_write_access():
    """GitHub answers 404, not 403, when the token cannot write to the repo."""
    client = _StubClient(
        _StubRepo(raises=UnknownObjectException(404, {"message": "Not Found"}, {}))
    )
    with pytest.raises(RepoNotFoundError) as caught:
        create_pull_request(
            "owner/repo", branch="b", base="main", title="t", body="b", client=client
        )
    assert "write access" in str(caught.value)


def test_create_pull_request_is_not_called_anywhere_in_the_m4_flow():
    """M4's scoping constraint, asserted rather than trusted.

    ``create_pull_request`` is finished and covered above, but nothing in the
    milestone's flow may invoke it: a PR is an outward-facing action awaiting
    M6's confirmation gate. This mirrors M3's
    ``test_no_shell_tool_is_exposed_to_the_agent``, so the constraint cannot be
    undone by accident.
    """
    import inspect

    from agent.core import repo_manager, run_on_issue, task_from_issue

    for module in (repo_manager, run_on_issue, task_from_issue):
        source = inspect.getsource(module)
        assert "create_pull_request(" not in source, (
            f"{module.__name__} calls create_pull_request; opening a PR is M6's "
            "decision, not M4's"
        )


# --------------------------------------------------------------------------
# integration: one real read from the real repository
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not github_client.token_present(),
    reason="GITHUB_TOKEN is not configured; skipping the live GitHub read",
)
def test_get_issue_reads_the_real_playground_issue(capsys):
    """Fetch issue #1 of the real playground repo and check it is what is there.

    Asserted against the issue's actual text, not just its shape: a client that
    returns a well-formed dict of the wrong fields would pass a shape-only test.
    """
    issue = get_issue(TEST_REPO, 1)

    with capsys.disabled():
        print(f"\n{'=' * 78}")
        print(f"LIVE GitHub read: {TEST_REPO} issue #1")
        print("=" * 78)
        for key, value in issue.items():
            print(f"{key:8}: {value}" if key != "body" else f"{key}:\n{value}")
        print("=" * 78)

    assert issue["number"] == 1
    assert issue["title"] == "add() gives the wrong result"
    assert issue["state"] == "open"
    assert issue["url"] == f"https://github.com/{TEST_REPO}/issues/1"

    # The body is the part the agent actually reads, so check its substance.
    assert "add(2, 3)" in issue["body"]
    assert "Expected: 5" in issue["body"]
    assert "Actual: -1" in issue["body"]
