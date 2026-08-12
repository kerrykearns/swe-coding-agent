"""Risk classification and human-confirmation gating for agent actions (M6).

::

    from agent.safety import ConfirmationGate

    gate = ConfirmationGate()
    result = gate.authorize("write_file", {"path": "calc.py", "content": "..."})
    if result.allowed:
        ...
"""

from .confirmation_gate import (
    AuditEntry,
    AuthorizationResult,
    ConfirmationCallback,
    ConfirmationGate,
    cli_confirm,
)
from .risk_classifier import RiskAssessment, classify, load_policy

__all__ = [
    "AuditEntry",
    "AuthorizationResult",
    "ConfirmationCallback",
    "ConfirmationGate",
    "RiskAssessment",
    "classify",
    "cli_confirm",
    "load_policy",
]
