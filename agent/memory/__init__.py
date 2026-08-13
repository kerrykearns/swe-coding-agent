"""Trajectory memory: Redis storage, similarity retrieval, credit assignment (M7).

::

    from agent.memory import get_redis_client, find_similar, save_trajectory

    client = get_redis_client()  # or get_redis_client(fakeredis.FakeRedis()) in tests
    key = save_trajectory(client, trajectory)
    similar = find_similar(client, "fix the add() function", k=3)
"""

from .credit_assignment import TurnScore, score_trajectory
from .embeddings import EmbeddingModelError, cosine_similarity, embed_text
from .redis_client import get_redis_client
from .trajectory_store import find_similar, load_trajectory, save_trajectory

__all__ = [
    "EmbeddingModelError",
    "TurnScore",
    "cosine_similarity",
    "embed_text",
    "find_similar",
    "get_redis_client",
    "load_trajectory",
    "save_trajectory",
    "score_trajectory",
]
