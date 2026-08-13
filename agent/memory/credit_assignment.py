"""Heuristic credit assignment over a completed trajectory (M7).

This is NOT a trained reward model and it is not real reinforcement-learning
credit assignment (no value function, no return, no discounting). It is a
simple, deterministic, rule-based heuristic over one already-finished
:class:`~agent.core.react_agent.Trajectory`, meant to give the memory layer
(:mod:`agent.memory.trajectory_store`) something slightly more useful than
"this run eventually succeeded or it didn't" when a future run retrieves it.

THE SCORING RULE
=================
``run_tests`` observations are the only ground truth the agent ever gets
about whether its code works, so they are used as checkpoints. Between two
consecutive successful ``run_tests`` checkpoints, the turns that happened
*between* them are exactly the actions whose effect the second checkpoint
verified:

* **+1** for every turn strictly between checkpoint *i* and checkpoint *j*,
  if *j* reported fewer failures+errors than *i* (an improvement the next
  test run confirmed).
* **-1** for the same turns, if *j* reported more failures+errors than *i*
  (a regression).
* **0** for the same turns, if the failure count did not change (flat).
* **0** for a ``run_tests`` turn itself — it is read-only, it changed
  nothing, so it cannot itself have caused an improvement or a regression.
* **0** for any turn before the first checkpoint or after the last one — its
  effect was never checked by a subsequent test run, so there is nothing to
  score it against. This includes every turn in a trajectory that never ran
  the tests at all.

A checkpoint only counts if the ``run_tests`` observation itself succeeded
(``tool_result["ok"]`` is True) and reports numeric ``failed``/``errors``
counts; a tool-level failure (e.g. a bad ``test_path``) is not a checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core.react_agent import Trajectory

__all__ = ["TurnScore", "score_trajectory"]


class TurnScore(BaseModel):
    """The heuristic credit assigned to one turn of a trajectory."""

    turn_number: int
    score: int  # -1, 0, or +1 — see the module docstring for the rule.
    reason: str


def _checkpoints(trajectory: "Trajectory") -> list[tuple[int, int]]:
    """Return ``(turn_index, failures)`` for every valid run_tests checkpoint.

    ``turn_index`` is the position in ``trajectory.turns`` (0-based), not the
    turn's own ``turn_number``, so it can index directly into the list.
    ``failures`` is ``failed + errors`` from that observation.
    """
    checkpoints = []
    for index, turn in enumerate(trajectory.turns):
        if turn.tool_name != "run_tests":
            continue
        result = turn.tool_result
        if not result.get("ok", False):
            continue
        if "failed" not in result or "errors" not in result:
            continue
        checkpoints.append((index, result["failed"] + result["errors"]))
    return checkpoints


def score_trajectory(trajectory: "Trajectory") -> list[TurnScore]:
    """Score every turn in ``trajectory`` using the rule in the module docstring.

    Returns one :class:`TurnScore` per turn, in turn order. The trajectory
    itself is not mutated — scores are a derived view, not stored on the
    :class:`~agent.core.react_agent.Trajectory`/``Turn`` models themselves.
    """
    turns = trajectory.turns
    scores = [0] * len(turns)
    reasons = ["not yet checked by a subsequent test run"] * len(turns)

    # Every run_tests call is read-only, whether or not it produced a usable
    # checkpoint (a bad test_path still runs no code) — so it never itself gets
    # credited or blamed for a result change, and it is excluded below even when
    # it falls between two valid checkpoints.
    read_only_indices = {
        index for index, turn in enumerate(turns) if turn.tool_name == "run_tests"
    }
    for index in read_only_indices:
        reasons[index] = "read-only: a test run does not itself change code"

    checkpoints = _checkpoints(trajectory)
    for (prev_index, prev_failures), (next_index, next_failures) in zip(
        checkpoints, checkpoints[1:]
    ):
        if next_failures < prev_failures:
            value, reason = 1, (
                f"preceded an improvement ({prev_failures} -> {next_failures} failing)"
            )
        elif next_failures > prev_failures:
            value, reason = -1, (
                f"preceded a regression ({prev_failures} -> {next_failures} failing)"
            )
        else:
            value, reason = 0, (
                f"no change in test outcome ({prev_failures} failing, still)"
            )
        for index in range(prev_index + 1, next_index):
            if index in read_only_indices:
                continue
            scores[index] = value
            reasons[index] = reason

    return [
        TurnScore(turn_number=turn.turn_number, score=scores[i], reason=reasons[i])
        for i, turn in enumerate(turns)
    ]
