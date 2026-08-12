# swe-coding-agent

A multi-turn, ReAct-style coding agent. Give it a GitHub issue or a plain-English
task and it plans the work, reads and edits files, runs tests and shell commands
in a Docker sandbox, and submits a patch. Built on Python and the Microsoft Agent
Framework, with Redis-backed trajectory memory and an evaluation harness that
scores the full agent against a single-shot baseline on success rate, average
turns, and token cost.

**Status: Work in Progress** — M1–M4 complete, currently on M5 (Docker
sandboxing). See [planning.md](planning.md) for the milestone plan.

## Setup

Install dependencies into a virtualenv:

```
pip install -r requirements.txt
```

### Sandboxed execution (optional, one-time)

Running tests inside a Docker sandbox (`--sandbox` on `run_on_issue`, on by
default there; `--sandbox` on the standalone `react_agent` CLI, off by
default) requires Docker Desktop (or another Docker daemon) running, and the
sandbox image built once:

```
python -c "from agent.tools import build_sandbox_image; build_sandbox_image()"
```

This builds `Dockerfile.sandbox` into the `swe-agent-sandbox:latest` image.
It only needs to run once — `build_sandbox_image()` checks whether the image
already exists and does nothing if it does, so it is safe to call again after
a change but is never run automatically as part of the test suite (a cold
build is slow enough that every `pytest` invocation doing it would make the
suite unusable).

To rebuild after editing `Dockerfile.sandbox`, remove the old image first
(`docker rmi swe-agent-sandbox:latest`) or build under a new tag.
