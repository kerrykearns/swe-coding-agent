"""Tests for turning a fetched issue into a task description.

Pure formatting, so every test here is offline and exact. The point of most of
them is *faithfulness*: whatever the reporter wrote has to survive into the
prompt unchanged, because M8 compares agents on the same input text and a
formatter that quietly edits it would make those numbers mean something else.
"""

from __future__ import annotations

import pytest

from agent.core.prompts import render_react_prompts
from agent.core.task_from_issue import NO_BODY_PLACEHOLDER, issue_to_task_description


def _issue(**overrides) -> dict:
    issue = {
        "number": 1,
        "title": "add() gives the wrong result",
        "body": "add(2, 3) returns -1 instead of 5.",
        "url": "https://github.com/owner/repo/issues/1",
        "state": "open",
    }
    issue.update(overrides)
    return issue


def test_the_format_is_heading_blank_line_body():
    assert issue_to_task_description(_issue()) == (
        "GitHub Issue #1: add() gives the wrong result\n\n"
        "add(2, 3) returns -1 instead of 5."
    )


def test_the_body_is_reproduced_verbatim():
    """No summarising, no rewrapping, no LLM. The reporter's words, as written."""
    body = (
        "Steps to reproduce:\n"
        "- Call add(2, 3)\n"
        "- Expected: 5\n"
        "- Actual: -1\n\n"
        "    indented code block\n"
        "|table|cell|\n"
    )
    task = issue_to_task_description(_issue(body=body))
    assert task.endswith(body.strip())
    assert "- Expected: 5" in task


def test_markdown_and_code_fences_survive():
    body = "```python\nassert add(2, 3) == 5\n```\n\nSee `calc.py`."
    assert body in issue_to_task_description(_issue(body=body))


def test_braces_in_the_body_survive():
    """A body full of braces reaches the prompt intact.

    Worth pinning at this seam: an issue body is very likely to contain code,
    and ``prompts.py`` fills templates with a single regex pass precisely so a
    value containing ``{something}`` is inserted rather than re-expanded. If
    either side of that contract changed, this is where it would show.
    """
    body = 'config = {"retries": 3}\nformat: "{file_content} {task_description}"'
    task = issue_to_task_description(_issue(body=body))
    assert body in task

    _, user_prompt = render_react_prompts(task_description=task, file_listing="calc.py")
    assert body in user_prompt


def test_carriage_returns_are_normalised():
    """GitHub returns web-form line endings; the prompt is assembled from LF text."""
    task = issue_to_task_description(_issue(body="line one\r\nline two\rline three"))
    assert "\r" not in task
    assert task.endswith("line one\nline two\nline three")


def test_surrounding_whitespace_is_stripped():
    task = issue_to_task_description(_issue(body="\n\n  real content  \n\n\n"))
    assert task.endswith("real content")


@pytest.mark.parametrize("empty", ["", "   ", "\n\n", None])
def test_an_issue_with_no_body_says_so(empty):
    """Better than an empty prompt tail: the agent is told to read the tests."""
    task = issue_to_task_description(_issue(body=empty))
    assert task == f"GitHub Issue #1: add() gives the wrong result\n\n{NO_BODY_PLACEHOLDER}"


def test_a_missing_body_key_is_treated_as_no_body():
    issue = _issue()
    del issue["body"]
    assert NO_BODY_PLACEHOLDER in issue_to_task_description(issue)


def test_the_title_is_trimmed_but_not_otherwise_touched():
    task = issue_to_task_description(_issue(title="  Fix: `add()` & friends  "))
    assert task.startswith("GitHub Issue #1: Fix: `add()` & friends\n\n")


def test_the_issue_number_appears_in_the_heading():
    """The agent should be able to say which issue it was working on."""
    assert issue_to_task_description(_issue(number=137)).startswith(
        "GitHub Issue #137:"
    )


def test_extra_keys_are_ignored():
    """The dict from github_client has url and state; neither belongs in the prompt."""
    task = issue_to_task_description(_issue())
    assert "https://github.com" not in task
    assert "open" not in task.splitlines()[0]


@pytest.mark.parametrize("missing", [{}, {"title": "t"}, {"number": None}])
def test_an_issue_without_a_number_is_rejected(missing):
    with pytest.raises(ValueError) as caught:
        issue_to_task_description(missing)
    assert "number" in str(caught.value)


@pytest.mark.parametrize("title", [None, "", "   "])
def test_an_issue_without_a_title_is_rejected(title):
    """Failing here is cheaper than a prompt whose heading reads "Issue #4: "."""
    with pytest.raises(ValueError) as caught:
        issue_to_task_description(_issue(title=title))
    assert "title" in str(caught.value)


def test_issue_number_zero_is_not_mistaken_for_missing():
    """``if not number`` would reject 0; GitHub never issues it, but the guard
    should still be about absence rather than falsiness."""
    assert issue_to_task_description(_issue(number=0)).startswith("GitHub Issue #0:")


def test_the_result_is_usable_as_a_react_task_description():
    """The one contract that matters: it renders into M3's prompt template."""
    task = issue_to_task_description(_issue())
    system_prompt, user_prompt = render_react_prompts(
        task_description=task, file_listing="calculator/calc.py"
    )
    assert task in user_prompt
    assert "GitHub Issue #1" in user_prompt
    assert system_prompt  # the system prompt is unaffected by the task text
