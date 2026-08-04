# calculator (fixture repo)

A tiny, deliberately broken repo used as a test fixture for the agent's tool
layer and, later, for the evaluation harness.

- `calculator/calc.py` — `add()` returns `a - b` instead of `a + b`.
- `calculator/test_calc.py` — `test_add()` fails because of that bug;
  `test_subtract()` passes.

Expected state: **1 failed, 1 passed**.

This directory is its own git repository, committed in the buggy state, so
`git diff` / `git apply` have something realistic to work against. Its `.git`
directory is ignored by the parent project.
