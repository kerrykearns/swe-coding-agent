"""A deliberately broken calculator, used as an agent fixture.

The bug in add() is intentional: test_calc.py fails against this file, which
gives the agent (and the tool-layer tests) something real to detect and fix.
"""


def add(a, b):
    """Return the sum of a and b."""
    return a - b


def subtract(a, b):
    """Return a minus b."""
    return a - b
