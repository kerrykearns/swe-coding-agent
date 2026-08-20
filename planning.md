# SWE Coding Agent — Project Plan

**Project complete** — M0–M9 all shipped: a safety-gated, sandboxed, multi-turn coding agent that beats a single-shot baseline on hard multi-defect tasks (100% vs 75%) at ~18x token cost, with GitHub issue-to-PR automation demonstrated end to end.

## Goal
Multi-turn ReAct-style coding agent: accepts a GitHub issue or natural-language
task, plans steps, edits files, runs tests/shell commands, and submits a patch.
Measured against a single-shot baseline on a curated set of real issues.

## Milestones
- [x] M0 — Repo & environment scaffolding
- [x] M1 — Tool layer (file ops, shell exec, patch/diff, test runner) — no LLM yet
- [x] M2 — Core ReAct loop, single-shot baseline agent
- [x] M3 — Multi-turn ReAct planning loop with stopping conditions
- [x] M4 — GitHub integration (fetch issue, clone, branch, commit locally).
      PR creation is written and tested but deliberately not wired in — it waits
      for M6's confirmation gate. Nothing in M4 pushes.
- [x] M5 — Docker sandboxing for all execution
- [x] M6 — Safety/refusal layer (risk classification + human confirmation)
- [x] M7 — Trajectory memory (Redis) + credit-assignment scoring
- [x] M8 — Evaluation harness (baseline vs full agent: success rate, avg turns,
      token cost) on curated local dummy repos with known failing tests
- [x] M9 — Live demo (CLI/web) + real GitHub issue showcase + README polish

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

### Sandboxing wraps `run_command`/`run_tests`; it does not add a new tool

Decision: `agent/tools/sandbox.py` adds `SandboxContainer` and
`build_sandbox_image`, and `shell_exec.run_command` / `test_runner.run_tests`
each grow a `sandboxed: bool = False` parameter plus an optional
already-open `sandbox` instance to run in. Both default to `False` and the
local-subprocess path is untouched, so M1–M4's calls to these functions, and
their tests, need no changes. A sandboxed `run_react_agent` opens exactly one
`SandboxContainer` for the whole run — not one per turn — and threads it
through `execute_tool` into every `run_tests` call, since starting a
container per turn would make a sandboxed run far slower than an
unsandboxed one for no isolation benefit (the workspace mount and
`network_mode="none"` are properties of the container, set once at start).
Rationale: M3 already scoped the agent's own toolset to
`read_file`/`write_file`/`list_files`/`search_text`/`run_tests`/`get_diff` —
no raw shell — specifically because nothing existed yet to confine a child
process. M5 is that confinement, not a reason to widen the toolset; the
agent still only ever executes pytest, now inside a network-isolated
container instead of on the host.
Tradeoff logged: `SandboxContainer.run()`'s timeout handling kills the exec'd
PID directly (found via `exec_inspect`, killed with `kill -9` from a second
exec) rather than tearing down the container, so one hung command doesn't
cost the whole container and force a restart mid-run. If that alone doesn't
unblock the reading thread within a grace period, the container is killed as
a last resort — correct for a run about to be torn down anyway, but it means
that `SandboxContainer` can't run anything else afterwards. Not observed in
testing; logged because it is the one path that is hard to exercise
deterministically without a command that resists `SIGKILL`.

### The CLI default and the library default for `sandboxed` disagree, deliberately

Decision: `run_on_issue()` itself defaults to `sandboxed=False`, matching
every other tool-layer default — but `run_on_issue.py`'s CLI defaults its
`--sandbox/--no-sandbox` flag to `True`. `react_agent.py`'s standalone CLI
defaults `--sandbox` to `False`, matching its own library default.
Rationale: `run_on_issue()` is called directly, without Docker mocked, by
every M4 test — the first pass at this milestone gave the library function
itself `sandboxed=True` by default, which broke all 25 of them the moment
they ran without a reachable sandbox image. The policy call ("real cloned
repos should sandbox by default") belongs at the CLI layer, which can afford
an opinion about *why* it's being invoked; the library function underneath
has no such context and should stay conservative so calling it — from tests,
from `run_react_agent`, from anywhere — never silently starts requiring
Docker.

### Classification is config, not code — and the fail-safe default is to ask

Decision: `configs/safety_policy.yaml` is the single source of truth for
which tier a tool call sits in; `agent/safety/risk_classifier.py` only
loads and matches it — no tier boundaries are hard-coded in Python, and no
LLM call is involved (deterministic, same input always gives the same
tier). A tool name covered by no rule in any tier classifies as
`needs_confirmation` rather than `safe`.
Rationale: the whole point of an auditable policy is that a human reviewing
`safety_policy.yaml` can see every rule and its one-line rationale without
reading classifier code. The fail-safe default matters more than it looks:
it means a new tool added to the agent's toolset without a matching policy
rule defaults to *asking*, not to running freely — the gap is caught by a
confirmation prompt, not by a silent bypass. `risk_classifier`'s module
docstring is explicit that this is a good-faith safety net for an agent
working on a legitimate task, not a security boundary hardened against
deliberately obfuscated adversarial input — that threat model belongs to
M5's sandbox (process/network isolation), not to regex matching on a
command string.

### The gate is threaded selectively, not blanket-wired everywhere the policy mentions

Decision: `agent/safety/confirmation_gate.py`'s `ConfirmationGate` is wired
into exactly two call sites: `react_agent.execute_tool` (every tool the
agent itself calls: `read_file`, `write_file`, `list_files`, `search_text`,
`run_tests`, `get_diff`) and `run_on_issue`'s new push/PR step
(`push_branch`, `create_pull_request`). The safety policy also classifies
`commit_changes` and `create_branch` (protected names) as
`needs_confirmation` — both are fully covered by `test_risk_classifier.py`
— but neither call site actually calls `gate.authorize()` in this
milestone: `run_on_issue` still commits (and creates its own working
branch) the same way M4 did, ungated. `run_on_issue` also does NOT thread
its gate into the ReAct loop it runs internally — the loop stays ungated
there, and only the post-commit push/PR step is gated.
Rationale: two things had to be balanced. First, `run_on_issue` runs
against a *real* GitHub issue for potentially a dozen turns; gating every
`write_file` there would mean a human blocked at the keyboard for the
entire run, which is a materially different product from "review before it
becomes irreversible" — the local commit is exactly as undoable as it was
in M4 (`git reset`), so it did not need the same treatment as an outbound
push. Second, this was learned the hard way: an early version threaded one
gate through both the loop and the push step, and a test built to prove
`auto_push=True` doesn't skip the gate — using a callback that denies
everything, then checking `auto_push` still gets a yes — instead found the
denial blocking `write_file` too, so the agent never got to commit at all
and the whole scenario under test never ran. A caller that wants the loop
itself gated can still call `react_agent.run_react_agent` directly with its
own `ConfirmationGate` (its CLI does exactly this, on by default) — the gap
is a deliberate scoping choice for `run_on_issue` specifically, not a
limitation of the gate itself. `commit_changes`/`create_branch` being
classified but unenforced is logged here rather than left as a silent
inconsistency, in the same spirit as M4's "the PR function exists and is
tested but nothing calls it".

### `push_branch` never touches the stored remote

Decision: `agent/core/repo_manager.push_branch` builds a one-off
authenticated URL (`https://x-access-token:<token>@github.com/...`) and
passes it directly to `git push`, exactly as `clone_repo` does for the
clone itself — it never runs `git remote set-url` or otherwise writes the
token into `.git/config`. Every string it returns has the token redacted
through the same `_redact` helper `clone_repo` and `commit_changes` already
use.
Rationale: `clone_repo` already resets the remote to a token-free URL
immediately after cloning specifically so a credential never sits in a
checkout the agent produced (see the M4 decision on this above) — a naive
`push_branch` that pushed to `origin` would either fail (no credential
configured) or require re-adding one to the stored remote, undoing that
work. Passing the URL directly to the one `git push` call keeps the token
in memory only, for exactly as long as that one command runs.

### Credit assignment is a heuristic over checkpoints, not a reward model

Decision: `agent/memory/credit_assignment.score_trajectory` scores each turn
using only the trajectory's own `run_tests` observations as checkpoints —
+1/-1/0 for the turns between two consecutive valid checkpoints, depending on
whether the failure count (`failed + errors`) went down, up, or stayed flat;
0 for every `run_tests` turn itself (read-only, regardless of whether that
particular call produced a valid checkpoint); 0 for every turn before the
first checkpoint or after the last one (never checked by a later test run).
The module docstring says explicitly that this is not RL and not a trained
reward model.
Rationale: the whole point is a signal that is auditable by reading the code,
not one that requires trusting a model. A live scoring pass against the M7
demo run's own trajectory (write → run_tests failing → three more writes →
never re-ran tests, `max_turns_reached`) is what caught a design gap the hand-
written tests alone had not: an early version only marked *valid* checkpoints
as read-only, so a `run_tests` call that failed for tool reasons (bad
`test_path`, not a real checkpoint) got scored as if it were an ordinary
code-changing action sitting between two real checkpoints — nonsensical,
since it never touched any file. Fixed so every turn whose `tool_name` is
`"run_tests"` is excluded from the between-checkpoints scoring pass, valid
checkpoint or not.

### The memory reference is an extra system message, not a template placeholder

Decision: when `run_react_agent`'s optional `memory_client` finds a match
above `MEMORY_SIMILARITY_THRESHOLD` (0.75), the summary
(`agent.core.react_agent._memory_reference`) is appended as its own
`{"role": "system", ...}` message, inserted between the loaded system prompt
and the first user message — `configs/prompts/react_system.md` itself takes
no new placeholder. Existing calls with `memory_client=None` (the default)
build exactly two messages, byte-for-byte the M1-M6 shape; a test
(`test_the_first_request_carries_the_prompts_the_tools_and_the_file_listing`)
already pins that count and needed no change.
Rationale: prompt templates are supposed to be reviewable as prompt edits
(see the M2 decision on prompts living on disk) — a retrieved memory is
per-run data, not a template change, so it does not belong baked into the
`.md` file. 0.75 itself is picked, not tuned: all-MiniLM-L6-v2 puts
paraphrases of the same task comfortably above 0.85-0.9 and unrelated tasks
under 0.5 in informal checks, so 0.75 favours precision (skip a weak match
entirely) over recall, since the model has no way to tell a strong match from
a noisy one once either is sitting in its context. The live demo below
reproduced a paraphrase landing at 0.76 — just above the line — which is
about as direct a validation of the threshold as a single run can give
without a proper eval set; M8 is where that gets checked properly.
Only the past run's own `finish()` summary is included, never full turns —
turns are token-expensive (file contents, raw pytest tracebacks) and the
model is told explicitly that this is "background from a different run" it
must still verify itself, not a shortcut to skip verification.

### `find_similar` is a linear scan on purpose, not a missing vector index

Decision: `agent/memory/trajectory_store.find_similar` embeds the query once
and then reads every key under `trajectory:*` with `scan_iter`, scoring each
against the query with cosine similarity computed in plain Python — no
vector database, no approximate-nearest-neighbour index.
Rationale: this project stores one trajectory per run. Even heavy use is
dozens to low hundreds of stored trajectories, and a linear scan over a few
hundred 384-dimensional vectors costs well under a millisecond of real work.
A vector DB would be a second service to keep running alongside Redis and
code to maintain an index that, at this scale, buys nothing a Python loop
does not already do. The reasoning is repeated as a comment in the module
itself so it survives independent of this log.

### Memory storage never creates an import cycle with the ReAct loop

Decision: `agent/memory/credit_assignment.py` and
`agent/memory/trajectory_store.py` only reference
`agent.core.react_agent.Trajectory` under `TYPE_CHECKING` at module scope;
the one place a real `Trajectory` class object is needed at runtime
(`load_trajectory`/`find_similar` reconstructing one from stored JSON) is a
deferred `from ..core.react_agent import Trajectory` inside the function
body. `agent/core/react_agent.py` itself imports `agent.memory` normally, at
module scope — the same pattern M6 used for `agent.safety.ConfirmationGate`.
Rationale: `agent.memory` never needs `agent.core.react_agent` to be
importable in order to import successfully, so there is nothing to cycle
against regardless of which module imports which first. That made the
`memory_client=None` default genuinely free: no lazy-import gymnastics were
needed in `react_agent.py` to keep M1-M6 callers working, because importing
`agent.memory` only costs `redis` and `pydantic` — already hard dependencies
— never `sentence-transformers`, which stays behind a lazy import inside
`agent/memory/embeddings._get_model()` and is only ever touched if
`embed_text` is actually called.

### A Windows/Docker Desktop `localhost` quirk, logged so it isn't relearned

Decision: none needed in the code — `agent/memory/redis_client.py` still
defaults to `REDIS_HOST=localhost`, matching `.env.example`. Logged here
because it cost real debugging time and belongs with M3's Groq/newline
findings as "a backend property of this environment, not a bug in our code".
On this Windows machine, `redis-py` connecting to Docker Desktop's published
Redis port via the hostname `localhost` completed the TCP handshake (`Test-
NetConnection` reported success) but then had the connection reset
mid-read (`WinError 10053/10054`) on every attempt, in both the Bash and
PowerShell tool — while a raw socket PING and `redis-py` both worked
immediately against `127.0.0.1`. This points at IPv6 (`::1`) loopback
forwarding through Docker Desktop's WSL2 network proxy being the flaky path,
with `localhost` resolving there first. `REDIS_HOST=127.0.0.1` is the
practical workaround for local dev on a machine with this quirk; nothing in
the product needed to change, since a real deployment target is unlikely to
share this specific Windows/WSL2/Docker Desktop combination.

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

M5 complete — Docker sandboxing (`Dockerfile.sandbox`, `agent/tools/sandbox.py`).
28 new tests: 19 mocked-docker unit tests covering `SandboxContainer`'s
lifecycle (start, run, timeout-kill, cleanup-on-exception) and
`build_sandbox_image`'s build-once behaviour, plus 6 dispatch tests in
`test_shell_exec.py`/`test_test_runner.py` for the new `sandboxed`/`sandbox`
parameters — all offline, no Docker required. 436 tests passing offline out
of 443 collected (7 marked `integration`: 4 carried over from M2–M4's live
LLM/GitHub checks, 3 new ones below). Then those 3 real integration tests
against an actual Docker daemon and the built `swe-agent-sandbox:latest`
image, all passing in ~21s: a real
command run and cleaned up (`docker` shows no leftover container), a real
network call from inside the sandbox failing fast rather than hanging
(`network_mode="none"` genuinely holds), and a real `sleep 60` killed within
its 5s timeout rather than running to completion (the exec'd PID is killed
directly, not the whole container — see the decision above).

`run_react_agent` and `execute_tool` both gained a `sandboxed`/`sandbox`
parameter that threads one `SandboxContainer` through every `run_tests` call
in a run; `shell_exec.run_command` and `test_runner.run_tests` both gained
matching `sandboxed`/`sandbox` parameters with `sandboxed=False` as the
default everywhere in the tool layer, so the M1–M4 offline suite runs
unmodified. `run_on_issue.py`'s CLI defaults `--sandbox` to `True`
(real cloned repos); `react_agent.py`'s standalone CLI defaults it to
`False` (local fixture iteration). One thing had to be caught and fixed
before it reached the offline suite: giving `run_on_issue()` itself (not
just its CLI) a default of `sandboxed=True` broke all 25 of its existing
tests, which call it directly without a reachable Docker daemon — logged as
a decision above, since the same mistake is easy to reintroduce in M6 or M7
if a library function's default is used to express a policy that belongs at
the CLI layer instead.

M6 complete — the safety/refusal layer (`configs/safety_policy.yaml`,
`agent/safety/risk_classifier.py`, `agent/safety/confirmation_gate.py`), plus
`agent/core/repo_manager.push_branch` and the gated push/PR wiring in
`run_on_issue.py`. 69 new tests (`test_risk_classifier.py`,
`test_confirmation_gate.py`), plus new/updated coverage in
`test_repo_manager.py`, `test_run_on_issue.py`, and `test_github_client.py`
for `push_branch` and the M6 push/PR flow — 520 tests passing offline
(`pytest tests/ -m "not integration"`), 0 failing.

`react_agent.execute_tool` now takes an optional `confirmation_gate`; when
given, every tool call is classified and a `needs_confirmation`/`blocked`
denial comes back as a normal `{"ok": False, ...}` observation the agent
reads and can react to, never an exception. `run_react_agent`'s own default
stays `confirmation_gate=None` (fully ungated, byte-for-byte the M1–M5
behaviour) — the CLI (`python -m agent.core.react_agent`) is the one that
opts into a real, interactive `ConfirmationGate` by default
(`--no-confirm` opts back out), mirroring the sandboxing
default-disagreement decision above. `run_on_issue` gained
`confirmation_gate` (used only for its new push/PR step — see the decision
above on why the loop itself stays ungated there), `auto_push` (still goes
through `gate.authorize()`, just pre-supplies "yes" for that one step —
never a bypass), and `pr_base_branch`. Its CLI always builds a gate and
exposes `--auto-push`.

Verified end to end against the calculator fixture via
`python -m agent.core.react_agent`: a task requiring `write_file` produced a
real terminal confirmation prompt; answering `y` let the write proceed and
the run continued normally; re-run with the same task, answering `n` at the
prompt produced a denied `write_file` observation, and the agent's next turn
visibly reacted to the denial instead of the run crashing.

Next: M7, trajectory memory (Redis) and credit-assignment scoring.

M7 complete — trajectory memory and credit-assignment scoring
(`agent/memory/redis_client.py`, `embeddings.py`, `trajectory_store.py`,
`credit_assignment.py`), plus optional `memory_client`/`on_memory_match`
wiring in `react_agent.run_react_agent` and a `--memory/--no-memory` CLI flag.
37 new offline tests (557 passing offline, up from 520, 0 failing) plus 2 new
`@pytest.mark.integration` tests against a real Redis, for 9 integration
tests project-wide. `fakeredis` backs every offline trajectory_store test;
`sentence-transformers`' all-MiniLM-L6-v2 is used for real everywhere
embeddings are exercised, since there is nothing meaningful to fake about
"does cosine similarity rank these correctly" — see the design decisions
above for the model-load and Windows/Docker-Desktop quirks that surfaced
while getting that model loading reliably in this environment.

`memory_client=None` remains the default everywhere (library function and
CLI both — unlike sandboxing/confirmation, there is no default-disagreement
decision to make here, since a CLI cannot reasonably default to *on* a piece
of optional infrastructure, Redis, that it cannot assume is running); every
M1-M6 test needed no changes, and the CLI's own memory flag defaults off for
the same reason `--sandbox` does.

Verified end to end against the calculator fixture, run twice through
`python -m agent.core.react_agent --memory`: the first run (task: "fix add()
so it returns the correct sum instead of subtracting") found nothing in an
empty Redis, fixed the bug in 4 turns, and saved its scored trajectory. The
second run, on a fresh independent copy of the same buggy repo with a
paraphrased task ("the add() function in the calculator is broken, it
subtracts instead of adding, please fix it"), printed a "memory match
found" panel showing the first run's task, outcome, and turn count at
similarity 0.76 — above the 0.75 threshold — genuinely injected into the
conversation before turn 1, not merely asserted in a test. The second run
itself then spent two turns on Groq's malformed-tool-call rejection (M3's
already-logged backend property) and ran out of its 8-turn budget without
calling `run_tests`, landing on `max_turns_reached` — a real live-model
outcome, left as-is rather than tuned away, since a memory demo is not the
place to paper over how noisy a live run already is.

M8 complete — the evaluation harness (`eval/harness.py`, `eval/run_eval.py`,
`eval/report.py`) and a 10-task, three-tier task set (`eval/tasks/*.yaml`):
trivial (3), medium (3), hard (4) — hard tasks each seed two independent
defects in one function, rather than one bug alone. 584 tests passing (up
from 557). Every `EvalResult.success` is decided the way the module
docstring insists on: an independent re-run of the task's own
`verify_command` against the final workspace, never the baseline's own
`success` or the ReAct loop's `final_success` — the "verified, not claimed"
principle applied one layer up.

The full 10-task x 2-condition matrix, run against Groq and saved
incrementally to `eval/results/*.json`, aggregates (via `eval/report.py`)
to: baseline 90% (9/10), react 100% (10/10) overall. By difficulty, both
conditions tie at 100% on trivial and medium, and diverge on hard —
baseline 75% (3/4), react 100% (4/4) — at a cost of ~18.3x the tokens and
~36x the wall-clock time per task for react. The hard tier is where the
comparison earns its design: every hard task hides two independent
defects, and every react win over baseline on that tier came from
catching a secondary defect (e.g. missing input validation) that
baseline's one-shot fix left unaddressed while still fixing the primary
bug. Full numbers in `eval/results/report.md`.

Two things surfaced while getting this matrix to actually complete:

* **Groq deprecated `llama-3.3-70b-versatile` mid-eval.** The model that
  had been the project's default since the "LLM Backend — Groq" decision
  logged above stopped being served partway through the M8 run. Because
  the model is read from `GROQ_MODEL`/`--model` at call time rather than hard-coded,
  the fix was pointing the environment variable at `openai/gpt-oss-120b`
  and re-running — no code changed.
* **A rate-limit propagation bug, found by the matrix itself.**
  `run_react_agent`'s per-turn loop caught every LLM-call exception,
  including `openai.RateLimitError`, in one generic `except Exception` and
  recorded it as an ordinary failed turn — correct for a malformed tool
  call, wrong for a 429: every subsequent call in the run, and every
  subsequent task in the matrix, was going to hit the same exhausted
  quota, so recording it as one task's failure meant
  `eval.run_eval.run_matrix` would burn through the rest of the matrix
  rediscovering the same rate limit one task at a time instead of
  stopping cleanly. Fixed with a `RateLimitError`-specific `except`
  clause, ordered before the catch-all, that re-raises so the error
  reaches `eval.harness.run_task` (which deliberately does not swallow
  it either — see its docstring) and then `run_matrix`, which now stops
  the whole run and reports how many combinations completed before the
  limit hit.

A separate, later finding — a live diff-syntax-corruption failure during
the M9 demo run, where the model wrote diff-hunk markers into a source
file's content instead of plain Python — is recorded as its own M9
finding, not folded in here: it is a different code path (malformed
`write_file` content, not an LLM-call exception) and a different cause
(non-deterministic model output, not a provider-side account change).

Next: M9, the live demo and README polish.
