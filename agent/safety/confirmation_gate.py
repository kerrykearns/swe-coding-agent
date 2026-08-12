"""The human-in-the-loop gate every risky agent action must pass through (M6).

:class:`ConfirmationGate` is the only thing in this codebase allowed to say
yes to a ``needs_confirmation`` action, and the only thing that can say no to
a ``blocked`` one unconditionally. Nothing else — not ``react_agent``, not
``run_on_issue`` — should call ``push_branch`` or ``create_pull_request``
directly; they call ``gate.authorize(...)`` first and act only if it comes
back allowed. See the wiring in :mod:`agent.core.react_agent` and
:mod:`agent.core.run_on_issue`.

The gate does three things, in order, for every call:

1. Classify it with :func:`agent.safety.risk_classifier.classify` (pure,
   deterministic — see that module's scope-limitation docstring).
2. Decide: ``safe`` passes with no prompt; ``blocked`` refuses with no
   prompt and no way to override; ``needs_confirmation`` asks — via a
   pluggable callback, not necessarily a human at a keyboard.
3. Record the decision in :attr:`ConfirmationGate.audit_log`, so "how often
   did the agent attempt something that needed confirmation" is a number
   M8's eval can read off directly, not something that has to be
   reconstructed from logs after the fact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from . import risk_classifier
from .risk_classifier import RiskAssessment

__all__ = [
    "AuditEntry",
    "AuthorizationResult",
    "ConfirmationCallback",
    "ConfirmationGate",
    "cli_confirm",
]

#: A confirmation callback receives the tool name, its arguments, and the
#: classifier's reason for asking, and returns True (proceed) or False
#: (deny). It is called ONLY for needs_confirmation actions — never for
#: safe (no need) or blocked (no callback can override it).
ConfirmationCallback = Callable[[str, dict[str, Any], str], bool]


class AuthorizationResult(BaseModel):
    """What :meth:`ConfirmationGate.authorize` returns."""

    allowed: bool
    tier: str
    reason: str


class AuditEntry(BaseModel):
    """One line of the gate's audit trail."""

    timestamp: str
    tool_name: str
    tool_args: dict[str, Any]
    tier: str
    #: What actually happened: "safe" (auto-allowed), "confirmed" (human/
    #: callback said yes), "denied" (callback said no), or "blocked"
    #: (refused unconditionally, no callback ever asked).
    decision: str
    reason: str
    matched_rule: str


def cli_confirm(tool_name: str, tool_args: dict[str, Any], reason: str) -> bool:
    """Default confirmation callback: ask a human at the terminal, via Rich.

    Shown a clear description of what is about to happen and why it needs
    approval, then a plain y/n prompt. This is the only callback in this
    module that can block waiting for input — every other callback used in
    this codebase (tests, ``--auto-push``, M8's eval harness) is synchronous
    and returns immediately, precisely so nothing but an interactive CLI run
    ever waits on a human.
    """
    console = Console()
    console.print(
        Panel(
            f"[bold]tool[/bold]   {tool_name}\n"
            f"[bold]args[/bold]   {json.dumps(tool_args, default=str)}\n"
            f"[bold]why[/bold]    {reason}",
            title="[yellow]confirmation required[/yellow]",
            border_style="yellow",
        )
    )
    return Confirm.ask(f"Allow {tool_name}?", default=False)


class ConfirmationGate:
    """Classifies, gates, and logs every risky action the agent attempts.

    Args:
        confirm_callback: Called for ``needs_confirmation`` actions to get a
            yes/no answer. Defaults to :func:`cli_confirm`, an interactive
            terminal prompt. Inject something else to run unattended — an
            eval harness (M8) passes a callback that answers instantly from
            a script, so a batch of runs never blocks on a human who is not
            there.
        log_path: Optional file to append each :class:`AuditEntry` to as one
            JSON line. The in-memory :attr:`audit_log` is always kept
            regardless of whether this is set.
    """

    def __init__(
        self,
        confirm_callback: Optional[ConfirmationCallback] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._confirm_callback = confirm_callback or cli_confirm
        self._log_path = Path(log_path) if log_path else None
        self.audit_log: list[AuditEntry] = []

    def authorize(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        override_callback: Optional[ConfirmationCallback] = None,
    ) -> AuthorizationResult:
        """Classify ``tool_name(tool_args)`` and decide whether it may run.

        Args:
            tool_name: The action being requested, e.g. ``"write_file"``.
            tool_args: Its arguments (may include synthetic keys the caller
                added purely for classification — see
                :func:`agent.safety.risk_classifier.classify`).
            override_callback: Use this callback instead of the gate's own
                for this one call, if the action needs confirmation. Lets a
                single gate instance (and its one audit log) serve calls
                that should be asked for real and calls that should be
                pre-approved — e.g. ``run_on_issue``'s ``--auto-push`` answers
                yes automatically for the push/PR step without changing how
                the rest of the run is gated. Never consulted for a
                ``blocked`` action; see the class docstring.

        Returns:
            An :class:`AuthorizationResult`. Every call — safe, confirmed,
            denied, or blocked — is also appended to :attr:`audit_log`
            before this returns.
        """
        assessment = risk_classifier.classify(tool_name, tool_args)

        if assessment.tier == "blocked":
            # Unconditional: no callback is even looked at, let alone called.
            # This is the one branch nothing in this class may change.
            result = AuthorizationResult(
                allowed=False, tier=assessment.tier, reason=assessment.reason
            )
            self._log(tool_name, tool_args, assessment, decision="blocked")
            return result

        if assessment.tier == "safe":
            result = AuthorizationResult(
                allowed=True, tier=assessment.tier, reason=assessment.reason
            )
            self._log(tool_name, tool_args, assessment, decision="safe")
            return result

        # needs_confirmation
        callback = override_callback or self._confirm_callback
        approved = bool(callback(tool_name, tool_args, assessment.reason))
        result = AuthorizationResult(
            allowed=approved, tier=assessment.tier, reason=assessment.reason
        )
        self._log(
            tool_name,
            tool_args,
            assessment,
            decision="confirmed" if approved else "denied",
        )
        return result

    def _log(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        assessment: RiskAssessment,
        decision: str,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            tool_args=tool_args,
            tier=assessment.tier,
            decision=decision,
            reason=assessment.reason,
            matched_rule=assessment.matched_rule,
        )
        self.audit_log.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
