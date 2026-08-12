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
- [x] M4 — GitHub integration (fetch issue, clone, branch, commit locally).
      PR creation is written and tested but deliberately not wired in — it waits
      for M6's confirmation gate. Nothing in M4 pushes.
- [ ] M5 — Docker sandboxing for all execution
- [ ] M6 — Safety/refusal layer (risk classification + human confirmation)
- [ ] M7 — Trajectory memory (Redis) + credit-assignment scoring
- [ ] M8 — Evaluation harness (baseline vs full agent: success rate, avg turns,
      token cost) on curated local dummy repos with known failing tests
- [ ] M9 — Live demo (CLI/web) + real GitHub issue showcase + README polish

## Design decisions log
(append here as we make choices, with date and rationale)

### LLM Backend — Groq

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

Decision: `run_tests` shells out to `"{sys.executable}" -m pytest ...` rather
than a bare `pytest`. Rationale: the target workspace is an arbitrary checkout
with no activated virtualenv, so a bare `pytest` resolves against whatever is
on `PATH` (or nothing at all). Pinning to the current interpreter keeps the
agent's test feedback deterministic.

### Prompts live on disk, not in Python string literals

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

Decision: `run_baseline` returns an `error` field alongside `success`. A model
reply with no usable code block sets `error`, leaves the file untouched, and
returns — it does not raise, and it does not re-ask.
Rationale: "the single-shot agent produced an unparseable answer" is exactly the
kind of failure M8 needs to count. Raising would make the baseline look broken
instead of weak, which is the whole comparison. Retrying would make it not a
baseline.

### The ReAct agent gets six tools and no shell (M3 scoping)

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

Decision: `Trajectory.final_success` is True only if a `run_tests` observation
actually reported success during the run. An agent that calls
`finish(success=True)` without having seen the tests pass produces
`final_success=False`; its claim is preserved in the finishing turn's `tool_args`.
Rationale: this matches the baseline's `success` ("the test suite passed
afterwards"), so M8 compares like with like. It also makes "how often does the
agent assert a fix it never verified?" a measurable quantity rather than a
silently trusted one.

### A write never triggers a test run, but a passing test run ends the loop

Decision: `write_file` does not auto-run the tests. `run_tests` reporting
`success=True` stops the loop immediately, without waiting for the agent to call
`finish`.
Rationale: the asymmetry is deliberate. Verifying its own work is a behaviour we
want to *measure*, not one we want to paper over — "did the agent remember to run
the tests?" is a real eval signal, and auto-running would destroy it. Once the
tests genuinely pass there is nothing left to measure, so spending a turn on the
agent noticing is just tokens.

### A malformed tool call is a turn, not a run failure
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

### M4 stops at a local commit; the PR function exists but is not wired in

Decision: the GitHub milestone fetches an issue, clones, branches, runs the M3
loop, and commits — locally. `github_client.create_pull_request` is written and
fully tested, and nothing calls it. There is no `git push` anywhere in
`repo_manager` or `run_on_issue`, and every run ends by printing that the branch
is local and unreviewed.
Rationale: a local commit is undone with `git reset`; a pushed branch is public
and an opened PR notifies people. Those are exactly the actions M6's
confirmation gate exists to sit in front of, so wiring them now would mean
building the gate afterwards, around code that already bypassed it. Building the
function anyway keeps the GitHub surface finished and covered in one place
instead of half-done in two milestones.
Two tests enforce the scoping rather than trusting it, in the spirit of M3's
`test_no_shell_tool_is_exposed_to_the_agent`: one asserts no module in the flow
contains a `create_pull_request(` call, and one records every git command an
assembled run issues and asserts none of them mentions `push`. The second is
deliberately behavioural rather than a source grep — the docstrings that explain
the constraint all contain the words "git push", so grepping for it would fail
on the explanation of why it is absent.

### GitHub failures are four distinct exceptions, never a PyGithub traceback

Decision: every call in `github_client` runs inside a translating context
manager that maps PyGithub's exceptions onto `GitHubAuthError`,
`RepoNotFoundError`, `IssueNotFoundError`, or `GitHubAPIError` — all subclasses
of one `GitHubError` — each carrying what was being looked up and what to do
about it.
Rationale: PyGithub reports a missing repo, a missing issue, a dead token, and a
rate limit as the *same class* with a status code buried in a payload dict, so an
un-translated failure reaches the user as `GithubException(404, {'message': 'Not
Found'})` — which does not say whether the repo or the issue was wrong. Two
things had to be learned rather than assumed:
* GitHub answers **404, not 403**, for a private repository (or a repo the token
  cannot write to), so "not found" genuinely means "missing *or* invisible" and
  the message has to say both. A user who trusts the literal text goes looking
  for a typo when the real fix is the token's repository access.
* `RateLimitExceededException` and `BadCredentialsException` both subclass
  `GithubException`, and the rate-limit one arrives as status **403**. Ordering
  the `except` clauses generic-first, or branching on status before class, would
  report a rate limit as "your token lacks permission" — telling the user to
  replace a token that works perfectly. The clauses are specific-first and a test
  pins that, since the bug it prevents is invisible in normal operation.
The exception messages are asserted on in tests, not just their types: the whole
point of the layer is the text.

### The token never lands in `.git/config`, and never in a log

Decision: `clone_repo` clones from
`https://x-access-token:<token>@github.com/<repo>.git`, then immediately resets
the remote to the token-free URL. Every string the module reports — error
messages included — passes through a redactor that replaces the token with
`***`.
Rationale: git *persists* the URL it was cloned from, so cloning with a
credential in it writes a live PAT in plaintext into the checkout, where it
outlives the run. Resetting the remote afterwards costs one local command. The
redaction is not belt-and-braces either: git echoes the remote URL in its own
error messages ("fatal: unable to access 'https://x-access-token:<token>@...'"),
so an un-redacted clone failure prints the PAT to the terminal and into any
captured log. A test asserts the token is absent from a simulated failure and
that `***` is present, because this is the kind of thing that is only ever
noticed after it has already leaked.
Tradeoff logged: the token is still on a shell command line during the clone,
where another user on the machine could see it in a process listing. Fixing that
properly means a credential helper or `http.extraheader`, which is worth doing
when M5's Docker sandbox changes how commands are spawned anyway.

### `commit_changes` reports; "nothing to commit" is not a failure

Decision: `commit_changes` returns `{success, commit_hash, output, reason}` and
does not raise for anything git says. An empty working tree returns
`success=False` with a reason that says it is not an error.
Rationale: an agent that finished without editing a file is an ordinary,
countable outcome — M8 needs it as a category, not as a traceback. The same
applies to a machine with no git identity configured: the caller's job is to tell
the user what happened, and it can do that with a dict. The one thing that *does*
raise is a blank commit message, because that is a bug in the caller rather than
a state of the repository.
The message is written to a temp file and passed as `git commit -F <file>`, never
`-m "<message>"`. The message is built from an issue title, i.e. from a stranger,
and `run_command` uses `shell=True` — so `-m` would interpolate somebody else's
text into a shell command line. A test commits the title
`Fix: "add()" & echo pwned > pwned.txt (closes #1)` and asserts both that the
subject is stored verbatim and that no `pwned.txt` exists.

### Branch names are sanitized by construction, and verified by git

Decision: `sanitize_branch_name` collapses every run of characters outside
`[a-z0-9]` to a single hyphen, and truncates at a word boundary. Titles with
nothing usable in them become `issue`.
Rationale: a blunt allowlist rules out every character git forbids in a ref
(space, `~^:?*[`, `\`, `..`, a trailing `.lock`, a leading `-`) without
enumerating the rules, which is the part that would rot. Since the rule is an
approximation, the tests do not check it against our own reading of the ref
format — they hand fourteen deliberately hostile titles to
`git check-ref-format --branch` and let git be the authority.

### The issue is formatted, not interpreted

Decision: `issue_to_task_description` is string formatting —
`"GitHub Issue #{n}: {title}\n\n{body}"`, with CRLF normalised and an explicit
placeholder for an empty body. No LLM is involved.
Rationale: the moment a model paraphrases the issue, the agent is working from a
summary, and every M8 number measures two models instead of one. The reporter's
words reach the prompt unchanged, and a test pins that a body full of braces
survives `prompts.py`'s template filling — the two modules' contract meets here.

### The agent does not commit its own tools' output

Decision: `clone_repo` appends `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, and
`.coverage` to the clone's `.git/info/exclude`. Not to the repository's
`.gitignore`.
Rationale: learned from the first live M4 run, which committed three `.pyc` files
alongside a one-line fix. `run_tests` makes pytest write bytecode caches *inside
the checkout*, and `commit_changes` stages with `git add -A`, so the agent's
verification step quietly became part of its patch — which in M6 would be part of
a pull request somebody has to read. These artifacts exist only because we ran
pytest there, so the opinion is ours and belongs in `.git/info/exclude`, which git
honours like a `.gitignore` but never commits and never shows a reviewer. Editing
the target repo's own `.gitignore` would put an unrequested change into the very
diff being reviewed. Already-tracked files are unaffected, so a repository that
deliberately versions a `.pyc` keeps it.
The offline flow test asserts the commit contains exactly `calculator/calc.py`,
and first asserts the caches really are on disk — otherwise it would pass by
proving nothing.

### Each run gets its own clone directory

Decision: `run_on_issue` clones into
`workspaces/<owner>-<repo>-issue-<n>`, appending `-2`, `-3`, … when that path
already exists, and refuses to clone into a non-empty directory.
Rationale: a second run against the same issue must neither fail nor silently
inherit the first run's edits — a stale working tree makes the new run's diff and
its `final_success` meaningless. `workspaces/` is gitignored, and a test asks
`git check-ignore` directly rather than trusting that, since what lands there is
a full checkout of somebody else's repository including its own `.git`.

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

M4 complete — GitHub integration (`agent/core/github_client.py`,
`repo_manager.py`, `task_from_issue.py`, `run_on_issue.py`). 191 new tests, 417
passing offline, plus 2 live integration tests (a real issue read, and the whole
flow end to end). The milestone stops at a local commit by design; see the
scoping decision above.

Verified end to end twice against the real `agent-test-playground` issue #1 —
once through the CLI and once through the integration test — with a real GitHub
fetch, a real clone, a real Groq-driven ReAct loop, and a real local commit
(`c698147` and `27dd54f`, both on `agent-fix/issue-1`, 9 turns and ~20.5k tokens
each, stopping on `tests_passed`).

Three things were learned from those live runs and are worth carrying into M8:

* **The turn budget has to be bigger for a real issue than for a hand-written
  task.** Every live run lost its first two to five turns to Groq's
  malformed-tool-call rejection — the M3 backend property, but noticeably worse
  here. The first attempt spent five of eight turns that way and never reached a
  write. M3's fixture task ("fix add() so it returns the correct sum") produced
  none of this; the issue's own prose ("I'm getting incorrect results…") produces
  it every time. 12 turns is now the live default.
* **The agent was committing its own test caches** (three `.pyc` files in a
  one-line fix). Found by reading the first successful run's commit transcript
  rather than by a test, which is exactly why the transcript is printed. Fixed
  and logged as a decision above; both successful runs above predate the fix, so
  it was re-verified against a fresh real clone with the LLM taken out of the
  loop (commit `b4078c1`, containing exactly `calculator/calc.py`).
* **The Groq free tier is 100,000 tokens per day**, and one live run on this
  repo costs ~20k. That is about five live runs a day — the fourth run of this
  session died on a 429 with 98,292 used, and the eval harness will want to
  compare a baseline and a full agent across a *set* of tasks. M8 needs either a
  paid tier or a run budget planned around this; it cannot be discovered while
  the harness is running.

Next: M5, Docker sandboxing.
