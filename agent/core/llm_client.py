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
    "get_client",
    "get_model",
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
