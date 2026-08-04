"""Groq chat completions, reached through an OpenAI-compatible client.

Groq speaks the OpenAI wire protocol, so the official ``openai`` SDK works
against it with nothing but a ``base_url`` override. Keeping that override in
one place means the rest of the agent never has to know which provider is
behind the client object.

Configuration is read from the environment (``.env`` is loaded once, at import
time). Nothing here validates that configuration until :func:`get_client` is
actually called, so importing this module can never fail on a missing key —
that matters for the test suite, which imports it to unit-test prompt handling
without ever talking to an API.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMConfigError",
    "api_key_present",
    "chat_completion",
    "chat_with_tools",
    "get_client",
    "get_model",
    "malformed_tool_call",
]

# Load `.env` once, on import, so every entrypoint (CLI, tests, eval harness)
# sees the same configuration without each having to remember to call this.
load_dotenv()

#: Groq's OpenAI-compatible endpoint, used when GROQ_BASE_URL is unset.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

#: Default model, matching the decision logged in planning.md.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

#: Values that mean "the developer copied .env.example but never filled it in".
#: Treated as a missing key so the error message points at the real problem.
_PLACEHOLDER_KEYS = frozenset(
    {"your_groq_api_key_here", "gsk_xxxxxxxxxxxxxxxxxxxxxxxx", "changeme"}
)

#: Provider error codes meaning "the model tried to call a tool and garbled the
#: syntax". Groq parses tool calls server-side, so a malformed call comes back as
#: an HTTP 400 rather than as assistant text — see :func:`malformed_tool_call`.
_MALFORMED_TOOL_CALL_CODES = frozenset({"tool_use_failed"})

_MISSING_KEY_MESSAGE = (
    "GROQ_API_KEY is not set (or is still the .env.example placeholder).\n"
    "Fix: copy .env.example to .env and set GROQ_API_KEY to a real key from "
    "https://console.groq.com/keys — the .env file lives in the project root "
    "and is gitignored."
)


class LLMConfigError(RuntimeError):
    """Raised when the LLM provider is not configured usably."""


def _env(name: str) -> str:
    """Return the stripped value of environment variable ``name``, or ``""``."""
    return (os.getenv(name) or "").strip()


def api_key_present() -> bool:
    """True if a usable (non-placeholder) ``GROQ_API_KEY`` is in the environment.

    Exists so callers — notably the integration tests — can decide to skip
    rather than provoke :class:`LLMConfigError`.
    """
    key = _env("GROQ_API_KEY")
    return bool(key) and key.lower() not in _PLACEHOLDER_KEYS


def get_model() -> str:
    """Return the configured model name, falling back to :data:`DEFAULT_MODEL`."""
    return _env("GROQ_MODEL") or DEFAULT_MODEL


def get_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at Groq.

    Reads ``GROQ_API_KEY`` (required) and ``GROQ_BASE_URL`` (optional) from the
    environment. The key is validated here, at call time, not at import time.

    Raises:
        LLMConfigError: If ``GROQ_API_KEY`` is missing or still the placeholder
            value from ``.env.example``.
    """
    if not api_key_present():
        raise LLMConfigError(_MISSING_KEY_MESSAGE)

    return OpenAI(
        api_key=_env("GROQ_API_KEY"),
        base_url=_env("GROQ_BASE_URL") or DEFAULT_BASE_URL,
    )


def chat_completion(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Make one non-streaming chat completion call and return content + usage.

    Args:
        client: A client from :func:`get_client`.
        system_prompt: Text for the ``system`` message.
        user_prompt: Text for the ``user`` message.
        model: Model name; defaults to :func:`get_model`.
        temperature: Defaults to 0 so the baseline is as reproducible as a
            hosted model allows — the eval harness (M8) compares runs.

    Returns:
        A dict with ``content`` (the assistant's reply, ``""`` if the model
        returned none) and ``prompt_tokens`` / ``completion_tokens`` /
        ``total_tokens``. Token counts come from the provider's ``usage``
        block and are ``0`` if it omitted one; they are recorded on every call
        because M8 measures token cost, and usage is not recoverable later.

    Raises:
        openai.OpenAIError: Propagated unchanged — network failures, auth
            rejections, and rate limits are the caller's to handle or record.
    """
    response = client.chat.completions.create(
        model=model or get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=False,
    )

    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    return {
        "content": content,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def malformed_tool_call(exc: Exception) -> str | None:
    """Classify ``exc`` as "the model garbled its tool call", or not.

    Groq validates tool calls on its own side, so a model that emits a
    syntactically broken call does not come back as an odd-looking assistant
    message — the request fails outright with HTTP 400 and
    ``code: "tool_use_failed"``. That is a *retryable mistake by the model*,
    not a broken run, and telling the two apart needs provider-specific
    knowledge, which is why it lives in this module rather than in the agent.

    Args:
        exc: An exception raised by :func:`chat_with_tools`.

    Returns:
        ``None`` if this is a real failure (auth, rate limit, network, bad
        request of any other kind) that the caller should treat as fatal.
        Otherwise the text the model *tried* to emit, taken from the provider's
        ``failed_generation`` field — possibly ``""`` if the provider did not
        include it. Callers must therefore test ``is not None``, not truthiness.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None

    # The ``openai`` SDK unwraps the provider's ``{"error": {...}}`` envelope
    # before building the exception, so ``body`` is normally the inner dict
    # already. Both shapes are accepted rather than relying on that.
    inner = body.get("error")
    error = inner if isinstance(inner, dict) else body

    if str(error.get("code") or "") not in _MALFORMED_TOOL_CALL_CODES:
        return None
    return str(error.get("failed_generation") or "")


def chat_with_tools(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Make one tool-enabled chat completion call over a full message history.

    The multi-turn counterpart to :func:`chat_completion`: it takes the whole
    conversation rather than a system/user pair, and it advertises tools so the
    model can answer with a structured tool call instead of prose.

    Args:
        client: A client from :func:`get_client`.
        messages: The conversation so far, in OpenAI wire format — including
            ``assistant`` messages carrying ``tool_calls`` and the ``tool``
            messages that answer them.
        tools: Tool specifications, each ``{"type": "function", "function":
            {...}}``. Omitted from the request entirely when empty, since some
            providers reject an explicit null.
        model: Model name; defaults to :func:`get_model`.
        temperature: Defaults to 0, as for :func:`chat_completion`.

    Returns:
        A dict with:

        * ``content`` — the assistant's text, ``""`` if it emitted none. A model
          answering with a tool call often says nothing, which is not an error.
        * ``tool_calls`` — a list of ``{"id", "name", "arguments"}`` dicts, empty
          if the model made none. ``arguments`` is the raw JSON *string* the
          model produced: it is deliberately not parsed here, because a model
          that emits malformed JSON is something the caller must observe and
          recover from, not something this layer should raise over.
        * ``finish_reason`` — the provider's stop reason, or ``""``.
        * ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` — usage,
          ``0`` if the provider omitted a usage block.

    Raises:
        openai.OpenAIError: Propagated unchanged, as in :func:`chat_completion`.
    """
    request: dict = {
        "model": model or get_model(),
        "messages": list(messages),
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        request["tools"] = list(tools)

    response = client.chat.completions.create(**request)

    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)

    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        tool_calls.append(
            {
                "id": getattr(call, "id", "") or "",
                "name": getattr(function, "name", "") or "",
                "arguments": getattr(function, "arguments", "") or "",
            }
        )

    usage = getattr(response, "usage", None)
    return {
        "content": getattr(message, "content", "") or "",
        "tool_calls": tool_calls,
        "finish_reason": getattr(choice, "finish_reason", "") or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
