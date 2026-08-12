"""Turn a fetched GitHub issue into the task description the ReAct loop reads.

This is formatting, not interpretation. No LLM is involved and none should be:
the moment a model paraphrases the issue, the agent is working from a summary,
and every eval result in M8 becomes a measurement of two models instead of one.
Whatever the reporter wrote is what the agent gets — the only thing added is the
issue number and a heading, so the agent knows where the text came from.
"""

from __future__ import annotations

__all__ = ["NO_BODY_PLACEHOLDER", "issue_to_task_description"]

#: Stands in for an issue with a title and no description. Says the body was
#: empty rather than leaving the agent to wonder whether text went missing, and
#: points it at the one thing it can actually do about that: read the tests.
NO_BODY_PLACEHOLDER = (
    "(This issue has no description — the title above is all the reporter wrote. "
    "Work out what is wrong from the repository's tests and code.)"
)


def issue_to_task_description(issue: dict) -> str:
    """Format ``issue`` as a task description for :func:`react_agent.run_react_agent`.

    Args:
        issue: An issue dict from :mod:`agent.core.github_client` — at minimum
            ``number`` and ``title``; ``body`` is used when present.

    Returns:
        ``"GitHub Issue #{number}: {title}"``, a blank line, then the body
        verbatim (trailing whitespace stripped, CRLF normalised to LF since
        GitHub returns web-form line endings and the prompt is assembled from
        LF text). A missing or blank body becomes
        :data:`NO_BODY_PLACEHOLDER`.

    Raises:
        ValueError: If ``number`` or ``title`` is missing or blank. An issue
            without them is not something to guess about — it means the caller
            passed the wrong dict, and failing here is cheaper than sending a
            prompt whose heading reads "GitHub Issue #None".
    """
    number = issue.get("number")
    if number is None or str(number).strip() == "":
        raise ValueError(f"Issue dict has no usable 'number': {issue!r}")

    title = str(issue.get("title") or "").strip()
    if not title:
        raise ValueError(f"Issue #{number} has no usable 'title': {issue!r}")

    body = str(issue.get("body") or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    return f"GitHub Issue #{number}: {title}\n\n{body or NO_BODY_PLACEHOLDER}"
