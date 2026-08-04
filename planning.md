# SWE Coding Agent — Project Plan

## Goal
Multi-turn ReAct-style coding agent: accepts a GitHub issue or natural-language
task, plans steps, edits files, runs tests/shell commands, and submits a patch.
Measured against a single-shot baseline on a curated set of real issues.

## Milestones
- [ ] M0 — Repo & environment scaffolding
- [x] M1 — Tool layer (file ops, shell exec, patch/diff, test runner) — no LLM yet
- [x] M2 — Core ReAct loop, single-shot baseline agent
- [x] M3 — Multi-turn ReAct planning loop with stopping conditions
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

### Prompts live on disk, not in Python string literals
Date: 2026-08-04
Decision: Baseline prompts are markdown files under `configs/prompts/`, loaded
and filled by `agent/core/prompts.py`. Filling is stricter than `str.format`:
a template placeholder with no value, *or* a supplied value whose placeholder is
not in the template (i.e. a typo), raises `PromptError` before the request is
made. Substitution is a single regex pass, so a value that itself contains
`{something}` — very likely, since the values are source code — is inserted
verbatim and never re-expanded.
Rationale: a prompt edit should be reviewable as a prompt edit, and a prompt
that reaches the API with a literal `{file_content}` in it burns tokens and
returns nonsense. Fail locally instead.

### The baseline records failure rather than raising it
Date: 2026-08-04
Decision: `run_baseline` returns an `error` field alongside `success`. A model
reply with no usable code block sets `error`, leaves the file untouched, and
returns — it does not raise, and it does not re-ask.
Rationale: "the single-shot agent produced an unparseable answer" is exactly the
kind of failure M8 needs to count. Raising would make the baseline look broken
instead of weak, which is the whole comparison. Retrying would make it not a
baseline.

### The ReAct agent gets six tools and no shell (M3 scoping)
Date: 2026-08-04
Decision: the loop exposes `read_file`, `write_file`, `list_files`,
`search_text`, `run_tests`, `get_diff`, and a `finish` pseudo-tool — and
deliberately does NOT expose `shell_exec.run_command`, even though the tool layer
has had it since M1.
Rationale: `Workspace` containment confines file reads and writes, but it cannot
confine a child process — it only pins cwd (see the containment decision above).
So until M5's Docker sandbox and M6's confirmation gate exist, the only execution
the agent gets is pytest via `run_tests`. A test asserts the absence
(`test_no_shell_tool_is_exposed_to_the_agent`) so this cannot be undone by
accident, and the system prompt tells the model there is no shell rather than
letting it discover that by failing.

### Tools are declared with Agent Framework, dispatched by us
Date: 2026-08-04
Decision: each tool is an `agent_framework.FunctionTool` built *declaration-only*
(`func=None`, `input_model=<pydantic model>`), and the wire schema comes from
`FunctionTool.to_json_schema_spec()`, which already emits the OpenAI-shaped
`{"type": "function", "function": {...}}` that Groq expects. Execution does not
go through the framework: `react_agent.execute_tool` dispatches by name and
injects the `Workspace`.
Rationale: every tool in `agent/tools` takes a `Workspace` as its first argument,
and the workspace is a per-run value the model must never see or be able to name.
Auto-invocation would have to bind it into the schema. Declaration-only is the
framework's own documented answer for "the implementation lives elsewhere", so we
keep its schema generation and argument validation while retaining the one thing
this milestone is about: control of the loop.
Tradeoff logged: we do not get the framework's automatic function-invocation
layer, so retries, approval modes, and invocation limits are ours to write if we
ever want them. That is the right trade while the loop itself is the deliverable.

### `final_success` is verified, never claimed
Date: 2026-08-04
Decision: `Trajectory.final_success` is True only if a `run_tests` observation
actually reported success during the run. An agent that calls
`finish(success=True)` without having seen the tests pass produces
`final_success=False`; its claim is preserved in the finishing turn's `tool_args`.
Rationale: this matches the baseline's `success` ("the test suite passed
afterwards"), so M8 compares like with like. It also makes "how often does the
agent assert a fix it never verified?" a measurable quantity rather than a
silently trusted one.

### A write never triggers a test run, but a passing test run ends the loop
Date: 2026-08-04
Decision: `write_file` does not auto-run the tests. `run_tests` reporting
`success=True` stops the loop immediately, without waiting for the agent to call
`finish`.
Rationale: the asymmetry is deliberate. Verifying its own work is a behaviour we
want to *measure*, not one we want to paper over — "did the agent remember to run
the tests?" is a real eval signal, and auto-running would destroy it. Once the
tests genuinely pass there is nothing left to measure, so spending a turn on the
agent noticing is just tokens.

### A malformed tool call is a turn, not a run failure
Date: 2026-08-04
Decision: three different kinds of model malformation are recorded as failed
turns and retried, and only a genuine API failure sets
`stop_reason="error"`: (a) a reply with no tool call, (b) arguments that are not
valid JSON or do not match the schema, and (c) Groq rejecting the request
outright with `code: "tool_use_failed"`.
Rationale: (c) is the one that had to be learned from a live run. Groq parses
tool calls server-side, so a model that garbles the syntax does not produce an
odd-looking message — the whole HTTP request fails with a 400, which looked like
a fatal error and killed the first live run on turn 1. Telling that apart from a
real 400 needs provider-specific knowledge, so it lives in
`llm_client.malformed_tool_call`, and its `body` shape is pinned by a test that
builds a real `openai` exception rather than a hand-made fake — the hand-made
fake encoded a wrong guess about that shape and passed while the agent broke.
The malformed text is recorded in the trajectory but deliberately *not* echoed
back into the conversation: feeding a model its own broken syntax invites it to
repeat it.

### Escape-mangled writes are reported, not silently repaired
Date: 2026-08-04
Decision: when `write_file` content looks like a source file whose newlines were
double-escaped (>80 chars, contains a literal backslash-n, essentially no real
line breaks), the observation says so loudly. The content is still written
exactly as asked.
Rationale: also learned from a live run — llama-3.3-70b sent a whole file with
`\n` as two characters, saw "wrote 359 characters", assumed success, and burned
all 8 turns confused by the wreckage. Silently un-escaping would be worse than
the bug: `"\n"` inside a string literal is ordinary Python, so a repair heuristic
would corrupt correct files. Reporting it keeps the tool honest and gives the
agent something it can act on. The narrow flag plus a schema/prompt warning was
enough for the live run to succeed in 4 turns.
Tradeoff logged: whole-file `write_file` is itself the root cause here, and even
on success the model rewrites incidental whitespace it was told not to touch. A
patch- or range-based edit tool would fix both, and is worth revisiting if M8
shows collateral edits hurting.

## Current status
M1 complete — tool layer built and tested (106 tests passing, 95% coverage on
`agent.tools`).

M2 complete — the single-shot baseline agent is built and tested
(`agent/core/llm_client.py`, `prompts.py`, `baseline.py`; 28 new tests, 139
passing offline plus 1 live-API integration test). The baseline is deliberately
one LLM call with no iteration and no tool use during reasoning: it is the floor
M8 measures the real agent against. Verified end to end against the calculator
fixture: one call, a one-line diff, 2 tests passing, 660 tokens.

M3 complete — the multi-turn ReAct loop (`agent/core/react_agent.py`,
`configs/prompts/react_system.md`, `react_user_template.md`). 86 new tests, 226
passing offline, plus 1 live-API integration test; 97% coverage on
`react_agent` (the uncovered remainder is the typer entrypoint body). The loop
drives itself entirely through scripted stub clients offline, so every stopping
condition and every malformation path is deterministic and free to test.

Verified end to end against the calculator fixture, twice (pytest and the CLI):
4 turns — `search_text` → `read_file` → `write_file` → `run_tests` — stopping on
`tests_passed` with `final_success=True` and ~9.7k tokens. Against the baseline's
660 tokens for the same fixture that is ~15x the cost for a task the baseline can
already do in one shot; the interesting comparison is the one M8 will make on
tasks where one shot is not enough.

Both live failures that shaped this milestone are logged as decisions above and
are worth remembering as backend properties, not one-off bugs: Groq rejects
malformed tool calls with an HTTP 400 rather than returning them as text, and
llama-3.3-70b will double-escape newlines in a whole-file write. Next: M4,
GitHub integration.
