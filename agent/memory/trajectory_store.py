"""Storing and retrieving trajectories in Redis, by task-description similarity (M7).

SCALE, AND WHY find_similar() IS A LINEAR SCAN
================================================
:func:`find_similar` embeds the query once, then reads every stored
trajectory and scores it against the query with cosine similarity — no
vector index, no approximate nearest-neighbour structure, no vector database.
That is a deliberate choice, not an oversight: this project stores one
trajectory per agent run, and even an ambitious amount of usage is dozens to
low hundreds of runs, not millions. A linear scan over a few hundred
384-dimensional vectors is well under a millisecond of real work; a vector DB
would add an operational dependency, a second thing to keep running alongside
Redis, and code to maintain an index that at this scale buys nothing a
Python loop does not already do just as well. Revisit only if trajectory
volume becomes the bottleneck it currently is nowhere close to being.

Every trajectory is stored under a key of the form
``trajectory:{unix_timestamp}:{short_uuid}``, as one JSON blob holding the
trajectory itself plus the embedding of its ``task_description`` (computed
once, at save time, so ``find_similar`` never re-embeds stored data).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Optional

import redis

from .credit_assignment import TurnScore
from .embeddings import cosine_similarity, embed_text

if TYPE_CHECKING:
    from ..core.react_agent import Trajectory

__all__ = ["find_similar", "load_trajectory", "save_trajectory"]

#: Redis key pattern every trajectory is stored under. See the module
#: docstring for the reasoning behind scanning this pattern linearly.
_KEY_PATTERN = "trajectory:*"


def _make_key() -> str:
    return f"trajectory:{int(time.time())}:{uuid.uuid4().hex[:8]}"


def save_trajectory(
    redis_client: redis.Redis,
    trajectory: "Trajectory",
    turn_scores: Optional[list[TurnScore]] = None,
) -> str:
    """Store ``trajectory`` (plus its task_description embedding) in Redis.

    Args:
        redis_client: A connected client — see
            :func:`agent.memory.redis_client.get_redis_client`. Any object
            with Redis's ``set``/``get``/``scan_iter`` methods works, which is
            what lets tests pass a :class:`fakeredis.FakeRedis` instead.
        trajectory: The completed run to store.
        turn_scores: The output of
            :func:`agent.memory.credit_assignment.score_trajectory`, stored
            alongside the trajectory for later inspection. Optional — a
            trajectory can be saved without having been scored.

    Returns:
        The generated key the trajectory was stored under.
    """
    key = _make_key()
    payload = {
        "trajectory": trajectory.model_dump(mode="json"),
        "embedding": embed_text(trajectory.task_description),
    }
    if turn_scores is not None:
        payload["turn_scores"] = [score.model_dump() for score in turn_scores]
    redis_client.set(key, json.dumps(payload))
    return key


def load_trajectory(redis_client: redis.Redis, key: str) -> "Trajectory":
    """Load and reconstruct the :class:`Trajectory` stored under ``key``.

    Raises:
        KeyError: If nothing is stored under ``key``.
    """
    from ..core.react_agent import Trajectory

    raw = redis_client.get(key)
    if raw is None:
        raise KeyError(f"No trajectory stored under key {key!r}.")
    payload = json.loads(raw)
    return Trajectory.model_validate(payload["trajectory"])


def find_similar(
    redis_client: redis.Redis, task_description: str, k: int = 3
) -> list[tuple["Trajectory", float]]:
    """Return the ``k`` stored trajectories most similar to ``task_description``.

    Embeds ``task_description`` once, then scans every trajectory stored
    under :data:`_KEY_PATTERN`, scoring each against the query with cosine
    similarity over the embedding saved at write time. See the module
    docstring for why this is a linear scan rather than a vector index.

    Returns:
        ``(trajectory, similarity)`` pairs, highest similarity first, at most
        ``k`` of them. Empty if nothing is stored yet.
    """
    from ..core.react_agent import Trajectory

    query_embedding = embed_text(task_description)

    scored: list[tuple["Trajectory", float]] = []
    for key in redis_client.scan_iter(match=_KEY_PATTERN):
        raw = redis_client.get(key)
        if raw is None:  # pragma: no cover - deleted between scan and get
            continue
        payload = json.loads(raw)
        similarity = cosine_similarity(query_embedding, payload["embedding"])
        trajectory = Trajectory.model_validate(payload["trajectory"])
        scored.append((trajectory, similarity))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
