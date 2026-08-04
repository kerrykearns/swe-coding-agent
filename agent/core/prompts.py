"""Loading and filling the prompt templates under ``configs/prompts``.

Prompts live on disk as plain markdown rather than as Python string literals so
they can be edited, diffed, and reviewed without touching code — a prompt
change is a prompt change, not a code change.

Filling is deliberately stricter than :meth:`str.format`. A prompt that reaches
the API with a literal ``{file_content}`` still in it burns tokens and returns
nonsense, so an unfilled placeholder (or a typo'd key that would silently fill
nothing) is an error here, before the request is made.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "PROMPTS_DIR",
    "PromptError",
    "load_prompt",
    "placeholders",
    "render_baseline_prompts",
    "render_prompt",
    "render_react_prompts",
    "render_template",
]

#: ``configs/prompts`` at the project root: agent/core/prompts.py -> ../../
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"

#: A placeholder is ``{identifier}`` — deliberately narrow, so braces in prose
#: or in embedded code (``{}``, ``{'a': 1}``) are not mistaken for one.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PromptError(ValueError):
    """Raised when a prompt cannot be loaded, or cannot be filled completely."""


def placeholders(template: str) -> list[str]:
    """Return the placeholder names in ``template``, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _PLACEHOLDER_RE.finditer(template):
        seen.setdefault(match.group(1), None)
    return list(seen)


def load_prompt(name: str, prompts_dir: Path | None = None) -> str:
    """Read a prompt template from ``configs/prompts``.

    Args:
        name: File name, e.g. ``"baseline_system.md"``.
        prompts_dir: Override the directory (used by tests).

    Raises:
        PromptError: If the file is missing or empty. Both are configuration
            mistakes, and both would otherwise surface as a baffling API reply.
    """
    directory = prompts_dir or PROMPTS_DIR
    path = directory / name
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(
            f"Prompt template not found: {path}. "
            f"Expected it under {directory}."
        ) from exc
    if not text.strip():
        raise PromptError(f"Prompt template {path} is empty.")
    return text


def render_template(template: str, **values: object) -> str:
    """Substitute every ``{placeholder}`` in ``template`` with ``values``.

    Substitution is a single pass, so a value containing something that looks
    like a placeholder (very likely — the values here are source code) is
    inserted verbatim and never re-expanded.

    Raises:
        PromptError: If a placeholder has no value, if a value was supplied for
            a placeholder the template does not contain (almost always a typo),
            or if any value is ``None``.
    """
    required = placeholders(template)

    missing = [name for name in required if name not in values]
    if missing:
        raise PromptError(
            "Prompt template has unfilled placeholder(s): "
            + ", ".join(f"{{{name}}}" for name in missing)
            + f". Provide a value for each of: {', '.join(required)}."
        )

    unexpected = sorted(set(values) - set(required))
    if unexpected:
        raise PromptError(
            "Value(s) supplied for placeholder(s) not present in the template: "
            + ", ".join(unexpected)
            + f". The template contains: {', '.join(required) or '(none)'}."
        )

    empty = sorted(name for name, value in values.items() if value is None)
    if empty:
        raise PromptError(
            "Placeholder value(s) are None: " + ", ".join(empty) + "."
        )

    return _PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), template)


def render_prompt(name: str, prompts_dir: Path | None = None, **values: object) -> str:
    """Load the template ``name`` and fill it — :func:`load_prompt` then
    :func:`render_template`."""
    return render_template(load_prompt(name, prompts_dir), **values)


def render_baseline_prompts(
    task_description: str,
    file_path: str,
    file_content: str,
    prompts_dir: Path | None = None,
) -> tuple[str, str]:
    """Build the single-shot baseline's ``(system_prompt, user_prompt)`` pair.

    The system template takes no placeholders and is loaded verbatim; the user
    template is filled with the task and the file being fixed.
    """
    system_prompt = load_prompt("baseline_system.md", prompts_dir)
    user_prompt = render_prompt(
        "baseline_user_template.md",
        prompts_dir=prompts_dir,
        task_description=task_description,
        file_path=file_path,
        file_content=file_content,
    )
    return system_prompt, user_prompt


def render_react_prompts(
    task_description: str,
    file_listing: str,
    prompts_dir: Path | None = None,
) -> tuple[str, str]:
    """Build the ReAct loop's opening ``(system_prompt, user_prompt)`` pair.

    Only the *first* user message is templated. Every later message in a ReAct
    run is a tool result, generated from real data at runtime, so it has no
    template to live in.

    The system template takes no placeholders and is loaded verbatim; the user
    template is filled with the task and a listing of the repository, which is
    the agent's only orientation before it starts calling tools.
    """
    system_prompt = load_prompt("react_system.md", prompts_dir)
    user_prompt = render_prompt(
        "react_user_template.md",
        prompts_dir=prompts_dir,
        task_description=task_description,
        file_listing=file_listing,
    )
    return system_prompt, user_prompt
