"""Deterministic, rule-based risk classification for agent tool calls (M6).

Every call the agent (or the code driving it) wants to make — read a file,
overwrite one, run a shell command, push a branch, open a pull request — is
classified into one of three tiers by matching it against
:data:`configs/safety_policy.yaml`:

* ``safe`` — runs without asking anyone.
* ``needs_confirmation`` — a human must say yes first (see
  :mod:`agent.safety.confirmation_gate`).
* ``blocked`` — refused unconditionally; no confirmation can override it.

SCOPE LIMITATION
================
This classifier is a **good-faith safety net for an agent working on a
legitimate task**, not a security boundary against a malicious user trying to
break out of it. Classification is pure pattern matching against the tool
name and its arguments — deterministic, no LLM call, auditable by reading
this module and the policy file it loads. It is deliberately *not* hardened
against deliberately obfuscated adversarial input (e.g. base64-encoded
commands, string concatenation tricks, or a command built specifically to
dodge the regexes below). Closing that gap would require sandboxing and
allowlisting at the OS level, which is what M5's Docker sandbox is for — this
module's job is to catch ordinary, unintentional danger ("the agent decided
to run ``git reset --hard``"), not to survive an adversary crafting input
against it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel

__all__ = ["POLICY_PATH", "RiskAssessment", "classify", "load_policy"]

#: Default location of the policy file, resolved relative to this module so
#: it works regardless of the process's current working directory.
POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety_policy.yaml"

#: Reason/matched_rule used when no rule in any tier matches. Unrecognised
#: actions are exactly what this layer exists to catch, so the fail-safe
#: default is to ask a human rather than assume the action is harmless.
_DEFAULT_RULE_ID = "default_fallback_unmatched_tool"
_DEFAULT_REASON = (
    "No safety policy rule matched this tool call, so it defaults to "
    "requiring confirmation rather than being assumed safe."
)


class RiskAssessment(BaseModel):
    """The outcome of classifying one tool call."""

    tier: Literal["safe", "needs_confirmation", "blocked"]
    #: Human-readable explanation — the matched rule's rationale, or the
    #: fail-safe message when nothing matched.
    reason: str
    #: The policy rule's ``id``, or :data:`_DEFAULT_RULE_ID` when nothing
    #: matched. Useful for the audit log and for tests that pin coverage of
    #: every rule in the policy file.
    matched_rule: str


def load_policy(path: Path = POLICY_PATH) -> dict[str, list[dict]]:
    """Load and lightly validate the safety policy YAML.

    Returns:
        ``{"safe": [...], "needs_confirmation": [...], "blocked": [...]}``,
        each a list of rule dicts as written in the file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain the three expected tiers.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = {"safe", "needs_confirmation", "blocked"} - raw.keys()
    if missing:
        raise ValueError(
            f"Safety policy {path} is missing required tier(s): {sorted(missing)}"
        )
    return raw


# Loaded once at import time — the policy is read-only configuration, not
# per-run state, and re-parsing YAML on every classify() call would be pure
# waste on an agent loop that classifies every single turn.
_POLICY: dict[str, list[dict]] = load_policy()


def _matches_when(when: Optional[dict], tool_args: dict[str, Any]) -> bool:
    """True if every condition in ``when`` holds against ``tool_args``.

    ``when: None`` (the rule has no condition) always matches. Each key is a
    small, fixed vocabulary rather than a general expression language —
    see the header comment in ``configs/safety_policy.yaml`` for the
    supported keys.
    """
    if not when:
        return True

    for key, expected in when.items():
        if key == "branch_name_in":
            actual = str(tool_args.get("branch_name", "")).strip().lower()
            if actual not in {str(v).lower() for v in expected}:
                return False
        else:
            if tool_args.get(key) != expected:
                return False
    return True


def _blocked_match(rule: dict, tool_args: dict[str, Any]) -> bool:
    """True if a blocked rule's pattern matches the relevant argument.

    Matching is case-insensitive and tolerant of extra whitespace (the regex
    patterns themselves use ``\\s+``/``\\s*`` where whitespace is expected) —
    robust against ordinary variation, not against deliberate obfuscation
    (see the module docstring's scope limitation).
    """
    field = rule.get("field", "command")
    value = str(tool_args.get(field, "") or "")
    return re.search(rule["pattern"], value, re.IGNORECASE) is not None


def classify(tool_name: str, tool_args: dict[str, Any]) -> RiskAssessment:
    """Classify one tool call into safe / needs_confirmation / blocked.

    Rule-based and deterministic: the same ``(tool_name, tool_args)`` always
    produces the same result, and nothing here calls an LLM. Rules are
    checked in a fixed order — ``blocked`` first and unconditionally, then
    ``needs_confirmation``, then ``safe`` — so a blocked pattern always wins
    even if the same call would otherwise match a safe or needs_confirmation
    rule (e.g. ``rm -rf`` run with ``sandboxed=True`` is still blocked).

    Args:
        tool_name: The tool being invoked, e.g. ``"write_file"``,
            ``"run_command"``, ``"push_branch"``.
        tool_args: Its arguments. Callers may include synthetic keys beyond
            what the model itself supplied — e.g. the caller adds
            ``sandboxed`` to a ``run_tests`` call's args before classifying,
            since that is run-level configuration the model never sees.

    Returns:
        A :class:`RiskAssessment`. A tool name covered by no rule in any
        tier returns ``needs_confirmation`` with a reason saying so — see
        the module docstring's fail-safe default.
    """
    policy = _POLICY

    for rule in policy["blocked"]:
        if tool_name in rule["tools"] and _blocked_match(rule, tool_args):
            return RiskAssessment(
                tier="blocked", reason=rule["rationale"].strip(), matched_rule=rule["id"]
            )

    for rule in policy["needs_confirmation"]:
        if tool_name in rule["tools"] and _matches_when(rule.get("when"), tool_args):
            return RiskAssessment(
                tier="needs_confirmation",
                reason=rule["rationale"].strip(),
                matched_rule=rule["id"],
            )

    for rule in policy["safe"]:
        if tool_name in rule["tools"] and _matches_when(rule.get("when"), tool_args):
            return RiskAssessment(
                tier="safe", reason=rule["rationale"].strip(), matched_rule=rule["id"]
            )

    return RiskAssessment(
        tier="needs_confirmation", reason=_DEFAULT_REASON, matched_rule=_DEFAULT_RULE_ID
    )
