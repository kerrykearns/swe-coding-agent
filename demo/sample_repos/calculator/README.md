# calculator (fixture repo)

A tiny, deliberately broken repo used as a test fixture for the agent's tool
layer and for M8's evaluation harness.

- `calculator/calc.py` — 11 functions. Ten each carry one self-contained,
  deliberate bug (`divide()`, `average()`, and `factorial()` each carry two:
  a wrong result plus an unsafe edge case); `subtract()` is the only one
  left correct.
- `calculator/test_calc.py` — 14 tests, independently detecting each bug
  (`divide()`, `average()`, and `factorial()` each get two tests, one per
  problem).

Expected state: **13 failed, 1 passed** (14 tests total).

Whatever uses this fixture copies it into a fresh tmp directory and
`git init`s that copy itself — see `tests/conftest.py`'s `calculator_ws`
fixture and `eval/harness.py`'s `run_task`. This directory is not a git
repository itself, so `git diff` / `git apply` only ever run against a
throwaway copy, never against the tracked fixture.
