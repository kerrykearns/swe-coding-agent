"""Tests for agent.memory.trajectory_store.

Two tiers, in the same spirit as tests/test_sandbox.py:

* Unit tests (the default) run against ``fakeredis.FakeRedis()`` — no real
  Redis needed. They still call the real embedding model (there is nothing
  meaningful to fake about "does cosine similarity actually rank these
  correctly" — that is the behaviour under test), so the first run on a
  machine downloads all-MiniLM-L6-v2 once; see agent/memory/embeddings.py and
  tests/test_embeddings.py for that.
* Real integration tests (``@pytest.mark.integration``) run against an actual
  Redis instance, skipped via a ping-based fixture — not an env var — exactly
  like M5's ``docker_client`` fixture in test_sandbox.py. Each test deletes
  only the keys it created, never a blanket scan-delete, since a real Redis
  reachable in this environment might not be exclusively this project's.
"""

from __future__ import annotations

import os

import fakeredis
import pytest
import redis as redis_module

from agent.core.react_agent import StopReason, Trajectory, Turn
from agent.memory.credit_assignment import score_trajectory
from agent.memory.trajectory_store import find_similar, load_trajectory, save_trajectory


def _trajectory(task_description: str, final_success: bool = True) -> Trajectory:
    trajectory = Trajectory(
        task_description=task_description,
        final_success=final_success,
        stop_reason=StopReason.TESTS_PASSED,
        total_turns=1,
        total_tokens=42,
    )
    trajectory.turns = [
        Turn(turn_number=1, tool_name="run_tests", tool_result={"ok": True, "failed": 0, "errors": 0})
    ]
    return trajectory


# --------------------------------------------------------------------------
# unit tests — fakeredis, real embeddings
# --------------------------------------------------------------------------


@pytest.fixture
def fake_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def test_save_trajectory_returns_a_key_matching_the_documented_pattern(fake_client):
    key = save_trajectory(fake_client, _trajectory("fix add()"))
    parts = key.split(":")
    assert parts[0] == "trajectory"
    assert parts[1].isdigit()  # unix timestamp
    assert len(parts[2]) == 8  # short uuid


def test_save_and_load_round_trips_a_trajectory(fake_client):
    original = _trajectory("fix add() so it returns the correct sum")
    key = save_trajectory(fake_client, original)

    loaded = load_trajectory(fake_client, key)

    assert loaded == original


def test_save_trajectory_persists_turn_scores_alongside_the_trajectory(fake_client):
    import json

    trajectory = _trajectory("fix add()")
    turn_scores = score_trajectory(trajectory)

    key = save_trajectory(fake_client, trajectory, turn_scores=turn_scores)

    stored = json.loads(fake_client.get(key))
    assert stored["turn_scores"] == [ts.model_dump() for ts in turn_scores]


def test_save_trajectory_without_turn_scores_omits_the_field(fake_client):
    import json

    key = save_trajectory(fake_client, _trajectory("fix add()"))

    stored = json.loads(fake_client.get(key))
    assert "turn_scores" not in stored


def test_load_trajectory_raises_keyerror_for_a_missing_key(fake_client):
    with pytest.raises(KeyError):
        load_trajectory(fake_client, "trajectory:does-not-exist")


def test_find_similar_returns_empty_when_nothing_is_stored(fake_client):
    assert find_similar(fake_client, "fix add()", k=3) == []


def test_find_similar_ranks_stored_trajectories_by_relevance(fake_client):
    """Three stored tasks, clearly ordered by how related they are to the query."""
    near_duplicate = _trajectory(
        "Fix add() so it returns the correct sum instead of subtracting."
    )
    same_domain = _trajectory(
        "Fix the multiply() function, which returns zero for every input."
    )
    unrelated = _trajectory(
        "Write a Kubernetes deployment manifest for the payments microservice, "
        "including a horizontal pod autoscaler."
    )
    for trajectory in (unrelated, same_domain, near_duplicate):  # stored out of rank order
        save_trajectory(fake_client, trajectory)

    results = find_similar(
        fake_client,
        "add() is broken — it subtracts instead of adding. Please fix it.",
        k=3,
    )

    assert [trajectory.task_description for trajectory, _ in results] == [
        near_duplicate.task_description,
        same_domain.task_description,
        unrelated.task_description,
    ]
    similarities = [similarity for _, similarity in results]
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] > similarities[-1]


def test_find_similar_respects_k(fake_client):
    for i in range(5):
        save_trajectory(fake_client, _trajectory(f"fix bug number {i} in add()"))

    results = find_similar(fake_client, "fix add()", k=2)

    assert len(results) == 2


# --------------------------------------------------------------------------
# real integration tests — require a reachable Redis
# (start one locally with: docker run --rm -p 6379:6379 redis:latest)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_redis_client():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    client = redis_module.Redis(host=host, port=port, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any reason Redis is unreachable
        pytest.skip(f"Redis is not reachable: {exc}")
    yield client
    client.close()


@pytest.mark.integration
def test_real_redis_save_and_load_round_trip(real_redis_client):
    trajectory = _trajectory("integration test: fix add()")
    key = save_trajectory(real_redis_client, trajectory)
    try:
        loaded = load_trajectory(real_redis_client, key)
        assert loaded == trajectory
    finally:
        real_redis_client.delete(key)


@pytest.mark.integration
def test_real_redis_find_similar_end_to_end(real_redis_client):
    """Real save, real scan, real ranking — against whatever else is in this Redis.

    Uses a generous k and filters the results down to just the two
    trajectories this test created, rather than asserting on absolute
    positions: a real, possibly shared Redis instance may hold other
    trajectories (including ones this same demo/test suite saved earlier),
    and the property this test actually cares about is the *relative* order
    between a close match and a far one, not whether nothing else outranks
    either of them.
    """
    close = _trajectory("Fix add() so it returns the correct sum instead of subtracting.")
    far = _trajectory(
        "Write a Kubernetes deployment manifest for the payments microservice."
    )
    keys = [save_trajectory(real_redis_client, t) for t in (close, far)]
    try:
        results = find_similar(
            real_redis_client, "add() subtracts instead of adding, please fix", k=50
        )
        ours = [
            (trajectory, similarity)
            for trajectory, similarity in results
            if trajectory.task_description in (close.task_description, far.task_description)
        ]
        assert {t.task_description for t, _ in ours} == {
            close.task_description,
            far.task_description,
        }
        close_similarity = next(s for t, s in ours if t.task_description == close.task_description)
        far_similarity = next(s for t, s in ours if t.task_description == far.task_description)
        assert close_similarity > far_similarity
    finally:
        for key in keys:
            real_redis_client.delete(key)
