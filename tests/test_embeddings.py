"""Tests for agent.memory.embeddings.

cosine_similarity is pure math and needs nothing external. embed_text loads a
real all-MiniLM-L6-v2 model via sentence-transformers — there is no meaningful
way to fake "does this model produce sane embeddings", so these tests call it
for real. The first call on a machine with no local cache downloads the model
from Hugging Face Hub; see the module docstring in agent/memory/embeddings.py
for the timeout/error handling around that. Once cached (in
``~/.cache/huggingface``), no network is needed again, and the module-level
singleton means this file only pays the model *load* cost once per test run,
not once per test.
"""

from __future__ import annotations

import math

import pytest

from agent.memory.embeddings import cosine_similarity, embed_text


# --------------------------------------------------------------------------
# cosine_similarity — pure math, no model involved
# --------------------------------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    vector = [1.0, 2.0, 3.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_is_scale_invariant():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]  # same direction, different magnitude
    assert cosine_similarity(a, b) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "a,b",
    [
        ([], [1.0, 2.0]),
        ([1.0, 2.0], []),
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),  # mismatched length
    ],
)
def test_cosine_similarity_degenerate_inputs_return_zero_not_a_crash(a, b):
    assert cosine_similarity(a, b) == 0.0


# --------------------------------------------------------------------------
# embed_text — real model, loaded once
# --------------------------------------------------------------------------


def test_embed_text_returns_a_fixed_length_vector_of_floats():
    vector = embed_text("fix the add() function")
    assert len(vector) == 384  # all-MiniLM-L6-v2's known output dimension
    assert all(isinstance(x, float) for x in vector)
    assert all(math.isfinite(x) for x in vector)


def test_embed_text_is_deterministic_for_the_same_input():
    text = "fix the add() function so it returns the correct sum"
    assert embed_text(text) == embed_text(text)


def test_embed_text_puts_similar_sentences_closer_than_unrelated_ones():
    base = "Fix the add() function so it returns the correct sum."
    paraphrase = "The add() function is broken; make it return the right sum."
    unrelated = "Configure the CI pipeline to cache pip dependencies."

    close = cosine_similarity(embed_text(base), embed_text(paraphrase))
    far = cosine_similarity(embed_text(base), embed_text(unrelated))

    assert close > far


def test_embed_text_reuses_the_same_model_across_calls():
    """The module-level singleton must not reload the model per call."""
    from agent.memory import embeddings as embeddings_module

    embed_text("warm the singleton up")  # ensure it is already loaded
    model_before = embeddings_module._model
    assert model_before is not None

    embed_text("a second, unrelated call")

    assert embeddings_module._model is model_before
