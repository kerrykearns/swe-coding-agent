# SWE Coding Agent — Project Plan

## Goal
Multi-turn ReAct-style coding agent: accepts a GitHub issue or natural-language
task, plans steps, edits files, runs tests/shell commands, and submits a patch.
Measured against a single-shot baseline on a curated set of real issues.

## Milestones
- [ ] M0 — Repo & environment scaffolding
- [x] M1 — Tool layer (file ops, shell exec, patch/diff, test runner) — no LLM yet
- [ ] M2 — Core ReAct loop, single-shot baseline agent
- [ ] M3 — Multi-turn ReAct planning loop with stopping conditions
- [ ] M4 — GitHub integration (fetch issue, clone, branch, patch/PR)
- [ ] M5 — Docker sandboxing for all execution
- [ ] M6 — Safety/refusal layer (risk classification + human confirmation)
- [ ] M7 — Trajectory memory (Redis) + credit-assignment scoring
- [ ] M8 — Evaluation harness (baseline vs full agent: success rate, avg turns,
      token cost) on curated local dummy repos with known failing tests
- [ ] M9 — Live demo (CLI/web) + real GitHub issue showcase + README polish

## Design decisions log
(append here as we make choices, with date and rationale)

### LLM Backend — Groq
Date: 2026-08-04
Decision: Using Groq (https://groq.com) as the LLM provider, accessed via
an OpenAI-compatible client (base_url override) since it offers a free
tier and Microsoft Agent Framework's native chat clients are OpenAI-shaped.
Default model: llama-3.3-70b-versatile. Rationale: free tier removes cost
friction during heavy agent-loop development/testing (many iterations =
many tokens), fast inference (Groq's LPU hardware) keeps dev loop fast.
Tradeoff logged: smaller/open-weight model may reason less reliably than
GPT-4-class models on complex multi-step coding tasks — revisit if eval
results in M8 show this as a bottleneck.

### Workspace containment as the single filesystem gatekeeper
Date: 2026-08-04
Decision: Every tool in `agent/tools/` takes a `Workspace` as its first
argument and resolves paths through `Workspace.resolve()`, which raises
`WorkspaceViolation` if the fully-resolved path lands outside the root.
Containment is checked after `Path.resolve()` (symlinks collapsed, `..`
applied), not by string-matching for `".."` — so `a/../../etc/passwd` and a
symlink/junction pointing out of the workspace are both rejected. Rationale:
one chokepoint to audit instead of per-tool path checks, and M6's safety layer
can then reason about a single boundary.
Tradeoff logged: `run_command` pins only the child process's *cwd*; it cannot
confine what the process itself touches. Filesystem confinement for shell
commands is M5's Docker sandbox, not this layer's.

### pytest invoked as `{sys.executable} -m pytest`
Date: 2026-08-04
Decision: `run_tests` shells out to `"{sys.executable}" -m pytest ...` rather
than a bare `pytest`. Rationale: the target workspace is an arbitrary checkout
with no activated virtualenv, so a bare `pytest` resolves against whatever is
on `PATH` (or nothing at all). Pinning to the current interpreter keeps the
agent's test feedback deterministic.

## Current status
M1 complete — tool layer built and tested (106 tests passing, 95% coverage on
`agent.tools`). Next: M2, the core ReAct loop and single-shot baseline agent.
