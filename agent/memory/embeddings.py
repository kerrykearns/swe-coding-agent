"""Local text embeddings for trajectory similarity search (M7).

Uses sentence-transformers' ``all-MiniLM-L6-v2`` — small (~80MB), free, no API
key, and fast enough on CPU that a synchronous call per trajectory is fine at
this project's scale. The model is loaded once, into a module-level
singleton, and reused for every call: reloading a multi-hundred-megabyte model
per embedding would dominate the run time of anything that calls
:func:`embed_text` more than once, which every caller here does.

FIRST-USE MODEL DOWNLOAD
=========================
``SentenceTransformer(...)`` downloads its weights from Hugging Face Hub the
first time this process (or this machine) asks for this model; after that
they are cached locally and no network call happens again. To keep a flaky or
absent network from hanging a test run forever instead of failing loudly,
``HF_HUB_DOWNLOAD_TIMEOUT`` is set (if the environment has not already set it)
before the download is attempted, and any failure is re-raised as
:class:`EmbeddingModelError` with a message that says what happened and how
to pre-warm the cache while online.
"""

from __future__ import annotations

import math
import os
from typing import Optional

__all__ = ["EMBEDDING_MODEL_NAME", "EmbeddingModelError", "cosine_similarity", "embed_text"]

#: sentence-transformers model name. Loaded lazily — see _get_model().
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# huggingface_hub's own default is already finite, but pin one explicitly so
# behaviour does not depend on whatever version happens to be installed.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

#: The loaded model, created on first use by _get_model(). Never reassigned
#: after that, so every call after the first reuses the same weights.
_model = None


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


def _get_model():
    """Return the module-level SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001 - reported, not left as a raw traceback
            raise EmbeddingModelError(
                f"Could not load the sentence-transformers model "
                f"{EMBEDDING_MODEL_NAME!r}. The first use on a machine downloads "
                "it from Hugging Face Hub, so this usually means no network "
                "connectivity right now. Pre-download it while online with: "
                f'python -c "from sentence_transformers import SentenceTransformer; '
                f'SentenceTransformer({EMBEDDING_MODEL_NAME!r})" '
                f"— after that it is cached locally and no network is needed. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
    return _model


def embed_text(text: str) -> list[float]:
    """Embed ``text`` into a fixed-length vector using all-MiniLM-L6-v2."""
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, in [-1.0, 1.0].

    Returns 0.0 for an empty or all-zero vector rather than raising — a
    zero-length embedding is a caller bug worth a neutral score, not a crash
    in the middle of a similarity ranking.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
