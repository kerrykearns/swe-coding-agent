"""Tests for agent.memory.credit_assignment.

All hand-constructed: a Trajectory with a known sequence of run_tests
checkpoints, asserting the exact score pattern the module docstring's rule
predicts. No LLM, no Redis, no embeddings — this is pure logic over data
that already exists.
"""

from __future__ import annotations

from agent.core.react_agent import FINISH_TOOL, StopReason, Trajectory, Turn
from agent.memory.credit_assignment import TurnScore, score_trajectory


def _turn(turn_number: int, tool_name: str, tool_result: dict) -> Turn:
    return Turn(turn_number=turn_number, tool_name=tool_name, tool_result=tool_result)


def _run_tests_turn(turn_number: int, failed: int, errors: int = 0, ok: bool = True) -> Turn:
    result = {"ok": ok}
    if ok:
        result.update({"failed": failed, "errors": errors, "passed": 1, "total": failed + errors + 1})
    return _turn(turn_number, "run_tests", result)


def _scores(trajectory: Trajectory) -> list[int]:
    return [ts.score for ts in score_trajectory(trajectory)]


def test_a_turn_between_an_improving_pair_of_checkpoints_scores_positive():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _turn(1, "write_file", {"ok": True}),
            _run_tests_turn(2, failed=2),  # checkpoint A: 2 failing
            _turn(3, "write_file", {"ok": True}),  # this fix is what improved things
            _run_tests_turn(4, failed=0),  # checkpoint B: 0 failing -> improvement
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 0, 1, 0]


def test_a_turn_between_a_regressing_pair_of_checkpoints_scores_negative():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=0),  # checkpoint A: all passing
            _turn(2, "write_file", {"ok": True}),  # this edit broke something
            _run_tests_turn(3, failed=3),  # checkpoint B: 3 failing -> regression
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, -1, 0]


def test_a_turn_between_a_flat_pair_of_checkpoints_scores_zero():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=2),
            _turn(2, "read_file", {"ok": True, "content": "..."}),  # did not touch the bug
            _run_tests_turn(3, failed=2),  # unchanged
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 0, 0]


def test_turns_before_the_first_checkpoint_are_unverified_and_score_zero():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _turn(1, "read_file", {"ok": True, "content": "..."}),
            _turn(2, "search_text", {"ok": True, "matches": []}),
            _run_tests_turn(3, failed=1),
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 0, 0]


def test_turns_after_the_last_checkpoint_are_unverified_and_score_zero():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=0),
            _turn(
                2,
                FINISH_TOOL,
                {"ok": True, "success": True, "summary": "done"},
            ),
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 0]


def test_a_trajectory_that_never_ran_tests_scores_every_turn_zero():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _turn(1, "read_file", {"ok": True, "content": "..."}),
            _turn(2, "write_file", {"ok": True}),
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 0]


def test_a_failed_run_tests_call_is_not_a_checkpoint():
    """A run_tests observation with ok=False (e.g. a bad test_path) carries no
    failure count, so it must not be treated as a checkpoint — the two real
    checkpoints on either side of it are still compared to each other."""
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=2),
            _run_tests_turn(2, failed=0, ok=False),  # e.g. bad test_path; no counts
            _turn(3, "write_file", {"ok": True}),
            _run_tests_turn(4, failed=0),
        ],
    )

    scores = _scores(trajectory)
    # turn 2's failed run_tests call is still read-only (0), even though it wasn't
    # a valid checkpoint; turn 3 (write_file) sits between the two REAL checkpoints
    # (turn 1: 2 failing, turn 4: 0 failing) -> improvement.
    assert scores == [0, 0, 1, 0]


def test_multiple_consecutive_checkpoint_pairs_are_each_scored_independently():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=3),
            _turn(2, "write_file", {"ok": True}),  # improves things
            _run_tests_turn(3, failed=1),
            _turn(4, "write_file", {"ok": True}),  # breaks something else
            _run_tests_turn(5, failed=2),
            _turn(6, "write_file", {"ok": True}),  # finishes the job
            _run_tests_turn(7, failed=0),
        ],
    )

    scores = _scores(trajectory)
    assert scores == [0, 1, 0, -1, 0, 1, 0]


def test_score_trajectory_returns_one_turnscore_per_turn_in_order():
    trajectory = Trajectory(
        task_description="fix add()",
        turns=[
            _run_tests_turn(1, failed=1),
            _turn(2, "write_file", {"ok": True}),
            _run_tests_turn(3, failed=0),
        ],
    )

    scores = score_trajectory(trajectory)
    assert [s.turn_number for s in scores] == [1, 2, 3]
    assert all(isinstance(s, TurnScore) for s in scores)
    assert scores[1].score == 1
    assert "improvement" in scores[1].reason
    assert "read-only" in scores[0].reason
