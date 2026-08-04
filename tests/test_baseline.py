"""Tests for the single-shot baseline agent.

Everything here runs offline except the one test marked ``integration``, which
makes a real Groq call and is skipped automatically when no API key is
configured. The offline tests cover the two places a single-shot run can go
wrong without the model being at fault: parsing its reply, and building the
prompt in the first place.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.core import llm_client
from agent.core.baseline import CodeBlockError, extract_code_block, run_baseline
from agent.core.prompts import (
    PromptError,
    load_prompt,
    placeholders,
    render_baseline_prompts,
    render_template,
)
from agent.tools import Workspace, read_file

FIXED_CALC = "def add(a, b):\n    return a + b\n"


# --------------------------------------------------------------------------
# extract_code_block
# --------------------------------------------------------------------------


def test_extract_clean_single_block():
    response = "```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code_block(response) == FIXED_CALC


def test_extract_ignores_prose_before_and_after():
    response = (
        "Sure! The bug is that `add` subtracts. Here is the corrected file:\n\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "Let me know if you would like me to explain further."
    )
    assert extract_code_block(response) == FIXED_CALC


def test_extract_accepts_untagged_fence():
    assert extract_code_block("```\ndef add(a, b):\n    return a + b\n```") == FIXED_CALC


def test_extract_accepts_tilde_fence():
    response = "~~~python\ndef add(a, b):\n    return a + b\n~~~"
    assert extract_code_block(response) == FIXED_CALC


def test_extract_normalises_crlf_and_trailing_whitespace():
    response = "```python\r\ndef add(a, b):\r\n    return a + b\r\n\r\n\r\n```"
    extracted = extract_code_block(response)
    assert "\r" not in extracted
    assert extracted == FIXED_CALC


def test_extract_raises_when_no_code_block():
    response = "The bug is on line 10: `add` uses `-` where it should use `+`."
    with pytest.raises(CodeBlockError, match="no triple-backtick code block"):
        extract_code_block(response)


def test_extract_raises_on_empty_response():
    with pytest.raises(CodeBlockError, match="empty response"):
        extract_code_block("   \n  ")


def test_extract_raises_on_multiple_ambiguous_blocks():
    response = (
        "Here is one option:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "Or, equivalently:\n\n```python\ndef add(a, b):\n    return sum((a, b))\n```\n"
    )
    with pytest.raises(CodeBlockError, match="Ambiguous model response"):
        extract_code_block(response)


def test_extract_picks_the_single_python_block_among_others():
    """The "here's the code, here's how to run it" reply is not ambiguous."""
    response = (
        "```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "Verify with:\n\n```bash\npytest -q\n```\n"
    )
    assert extract_code_block(response) == FIXED_CALC


def test_extract_raises_when_every_block_is_empty():
    with pytest.raises(CodeBlockError, match="every code block in it was empty"):
        extract_code_block("```python\n\n```")


# --------------------------------------------------------------------------
# prompt loading and filling
# --------------------------------------------------------------------------


def test_render_template_fills_every_placeholder():
    template = "Fix {what} in {where}.\n\n{body}\n"
    filled = render_template(
        template, what="add()", where="calc.py", body="return a - b"
    )
    assert filled == "Fix add() in calc.py.\n\nreturn a - b\n"


def test_render_template_raises_on_missing_placeholder():
    with pytest.raises(PromptError, match=r"unfilled placeholder\(s\): \{file_content\}"):
        render_template("{task_description}\n{file_content}", task_description="fix it")


def test_render_template_raises_on_unknown_key():
    """A typo'd key would otherwise leave a real placeholder silently unfilled."""
    with pytest.raises(PromptError, match="not present in the template"):
        render_template("{task}", task="fix it", tsak="fix it")


def test_render_template_raises_on_none_value():
    with pytest.raises(PromptError, match="are None"):
        render_template("{task}", task=None)


def test_render_template_does_not_re_expand_inserted_values():
    """Values are source code; a `{x}` inside one is content, not a placeholder."""
    filled = render_template("{code}", code="d = {'k': 1}\nf'{d}'")
    assert filled == "d = {'k': 1}\nf'{d}'"


def test_placeholders_are_returned_in_first_seen_order():
    assert placeholders("{b} {a} {b} {c}") == ["b", "a", "c"]


def test_load_prompt_raises_for_a_missing_file(tmp_path: Path):
    with pytest.raises(PromptError, match="Prompt template not found"):
        load_prompt("nope.md", prompts_dir=tmp_path)


def test_load_prompt_raises_for_an_empty_file(tmp_path: Path):
    (tmp_path / "blank.md").write_text("\n\n", encoding="utf-8")
    with pytest.raises(PromptError, match="is empty"):
        load_prompt("blank.md", prompts_dir=tmp_path)


def test_shipped_baseline_templates_render_completely():
    """The real templates in configs/prompts must fill with exactly our three values."""
    system_prompt, user_prompt = render_baseline_prompts(
        task_description="fix add() so it returns the sum",
        file_path="calculator/calc.py",
        file_content="def add(a, b):\n    return a - b\n",
    )
    assert system_prompt.strip()
    assert "fix add() so it returns the sum" in user_prompt
    assert "calculator/calc.py" in user_prompt
    assert "return a - b" in user_prompt
    # Nothing of the form {name} may survive into the outgoing prompt.
    assert placeholders(user_prompt) == []


def test_baseline_user_template_declares_the_expected_placeholders():
    template = load_prompt("baseline_user_template.md")
    assert set(placeholders(template)) == {
        "task_description",
        "file_path",
        "file_content",
    }


# --------------------------------------------------------------------------
# run_baseline, driven by a stub client (no API calls)
# --------------------------------------------------------------------------


class _StubCompletions:
    """Stands in for ``client.chat.completions``, recording what it was sent."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=22, total_tokens=33
            ),
        )


def _stub_client(content: str) -> SimpleNamespace:
    completions = _StubCompletions(content)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_run_baseline_writes_the_fix_and_reports_passing_tests(calculator_ws: Workspace):
    client = _stub_client(
        "```python\ndef add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n\n\n"
        "def subtract(a, b):\n"
        '    """Return a minus b."""\n'
        "    return a - b\n```"
    )
    result = run_baseline(
        calculator_ws, "fix add()", "calculator/calc.py", client=client
    )

    assert result["success"] is True
    assert result["error"] is None
    assert "-    return a - b" in result["diff"]
    assert "+    return a + b" in result["diff"]
    assert result["test_results"]["passed"] == 2
    assert result["token_usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 22,
        "total_tokens": 33,
    }
    assert result["target_file"] == "calculator/calc.py"
    assert "return a + b" in read_file(calculator_ws, "calculator/calc.py")

    # One call, and the file's current content really was sent to the model.
    sent = client.chat.completions.calls
    assert len(sent) == 1
    assert "return a - b" in sent[0]["messages"][1]["content"]


def test_run_baseline_records_an_unusable_reply_without_touching_the_file(
    calculator_ws: Workspace,
):
    """A model that answers in prose is a baseline failure, not an exception."""
    result = run_baseline(
        calculator_ws,
        "fix add()",
        "calculator/calc.py",
        client=_stub_client("You should change the minus sign to a plus sign."),
    )

    assert result["success"] is False
    assert "no triple-backtick code block" in result["error"]
    assert result["diff"] == ""
    assert result["test_results"] is None
    assert result["token_usage"]["total_tokens"] == 33
    assert "return a - b" in read_file(calculator_ws, "calculator/calc.py")


def test_run_baseline_reports_a_still_broken_fix(calculator_ws: Workspace):
    """A wrong-but-parseable edit is written, diffed, and honestly marked failed."""
    result = run_baseline(
        calculator_ws,
        "fix add()",
        "calculator/calc.py",
        client=_stub_client(
            "```python\ndef add(a, b):\n    return a * b\n\n\n"
            "def subtract(a, b):\n    return a - b\n```"
        ),
    )

    assert result["success"] is False
    assert result["error"] is None
    assert "+    return a * b" in result["diff"]
    assert result["test_results"]["failed"] == 1
    assert result["test_results"]["passed"] == 1


def test_run_baseline_rejects_a_target_outside_the_workspace(calculator_ws: Workspace):
    from agent.tools import WorkspaceViolation

    with pytest.raises(WorkspaceViolation):
        run_baseline(
            calculator_ws, "fix it", "../escape.py", client=_stub_client("```\nx\n```")
        )


def test_run_baseline_raises_for_a_missing_target(calculator_ws: Workspace):
    with pytest.raises(FileNotFoundError):
        run_baseline(
            calculator_ws, "fix it", "nope.py", client=_stub_client("```\nx\n```")
        )


# --------------------------------------------------------------------------
# llm_client configuration
# --------------------------------------------------------------------------


def test_get_client_raises_a_helpful_error_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMConfigError, match=r"\.env"):
        llm_client.get_client()


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    assert llm_client.api_key_present() is False


def test_get_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert llm_client.get_model() == llm_client.DEFAULT_MODEL
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")
    assert llm_client.get_model() == "some-other-model"


# --------------------------------------------------------------------------
# integration: one real Groq call
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not llm_client.api_key_present(),
    reason="GROQ_API_KEY is not configured; skipping the live Groq call",
)
def test_baseline_fixes_the_calculator_end_to_end(calculator_ws: Workspace, capsys):
    """One real call against the buggy fixture.

    Asserts the pipeline ran end to end and produced a real diff. It does *not*
    assert ``success is True``: the point of the baseline is that it sometimes
    fails, and M8 measures how often.
    """
    result = run_baseline(
        calculator_ws,
        "fix add() so it returns the correct sum instead of subtracting",
        "calculator/calc.py",
    )

    with capsys.disabled():
        print("\n--- llm_response ---")
        print(result["llm_response"])
        print("--- diff ---")
        print(result["diff"] or "(empty)")
        print("--- error ---")
        print(result["error"] or "(none)")
        print("--- tests ---")
        print(result["test_results"])
        print("--- tokens ---")
        print(result["token_usage"])
        print(f"--- success: {result['success']} ---")

    assert result["llm_response"].strip(), "the model returned nothing"
    assert result["token_usage"]["total_tokens"] > 0
    assert result["error"] is None, f"reply was unusable: {result['error']}"

    # The model edited the file, and the edit was captured before tests ran.
    assert isinstance(result["diff"], str)
    assert result["diff"].strip(), "no diff was produced"
    assert "calculator/calc.py" in result["diff"]

    # The test-suite outcome was captured, whatever it turned out to be.
    assert result["test_results"] is not None
    assert result["test_results"]["total"] == 2
    assert isinstance(result["success"], bool)
    assert result["success"] == result["test_results"]["success"]
