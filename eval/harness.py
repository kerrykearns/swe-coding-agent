"""Run one (task, condition) combination and score it independently.

This is the unit M8 is built on: given a task description and a condition —
``"baseline"`` (M2's single-shot agent) or ``"react"`` (M3-M6's multi-turn
loop) — :func:`run_task` copies the task's fixture into a fresh throwaway
workspace, runs the chosen agent against it, and reports what happened as an
:class:`EvalResult`.

VERIFIED, NOT CLAIMED
======================
``EvalResult.success`` is set from exactly one source: an independent re-run
of the task's own ``verify_command`` against the final workspace state, via
:func:`agent.tools.test_runner.run_tests`. Neither the baseline's own
``success`` field nor the ReAct loop's ``Trajectory.final_success`` is ever
copied into it — both are already "verified, not claimed" in their own right
(see the design decisions in planning.md), but re-deriving the verdict here,
from scratch, against the *harness's* own idea of what "done" means for this
task, means a bug in either agent's internal bookkeeping can never leak into
the eval numbers. This is the same principle M3 used for
``Trajectory.final_success``, applied one layer up.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import yaml
from openai import RateLimitError
from pydantic import BaseModel

from agent.core.baseline import run_baseline
from agent.core.react_agent import StopReason, Turn, run_react_agent
from agent.tools import Workspace, run_tests

__all__ = ["EvalResult", "load_tasks", "run_task"]

#: The project root, so a task's ``fixture_repo`` (relative in every shipped
#: eval/tasks/*.yaml) resolves the same way regardless of the caller's cwd.
#: An absolute ``fixture_repo`` (as tests pass, pointing at a tmp_path fixture)
#: is unaffected: pathlib's ``/`` discards the left side when the right side
#: is already absolute.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Never copy a fixture's own git history or test caches into the throwaway
#: workspace — the workspace gets a fresh, single-commit repo of its own
#: (see _init_git_repo), and stale caches from a previous local pytest run
#: must not leak into a run that is supposed to start from a clean slate.
_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")


class EvalResult(BaseModel):
    """The outcome of one (task, condition) run — M8's unit of measurement.

    Mirrors the fields planning.md's M8 entry names as the headline
    comparison: success rate, turns, and token cost, per condition.
    """

    task_id: str
    condition: Literal["baseline", "react"]
    success: bool
    turns: int
    total_tokens: int
    wall_clock_seconds: float
    verify_output: str
    error: Optional[str] = None
    #: The react loop's own turn-by-turn record — reasoning, tool calls, and
    #: observations — for post-hoc debugging a surprising result (e.g. an
    #: unexpectedly high turn count) without re-spending tokens to reproduce
    #: it. Always None for "baseline" (a single LLM call has no turns worth
    #: recording) and never read by any scoring/success logic in this module.
    full_trajectory: Optional[list[Turn]] = None


def load_tasks(tasks_dir: str | Path) -> list[dict]:
    """Load every ``*.yaml`` file in ``tasks_dir`` as a task dict.

    Returns them sorted by filename, so the run order (and therefore the
    order results are printed and saved in) is deterministic across runs.
    """
    directory = Path(tasks_dir)
    tasks = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            task = yaml.safe_load(handle)
        tasks.append(task)
    return tasks


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True, text=True)


def _init_git_repo(root: Path) -> None:
    """Turn a freshly-copied fixture into its own single-commit git repo.

    Both agents' diff-capture (``get_diff``) requires a real git repository,
    and the fixture directories under demo/sample_repos are themselves
    ordinary files with no ``.git`` of their own (tests copy them and init a
    repo the same way; see tests/conftest.py's ``init_repo``, which this
    mirrors so the eval harness doesn't depend on the tests package).
    """
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "user.name", "Eval Harness")
    _git(root, "config", "user.email", "eval-harness@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture: initial buggy state")


def _test_path_from_verify_command(verify_command: str) -> str:
    """Pull the pytest node-id(s) out of a task's ``verify_command``.

    A task's ``verify_command`` is written for a human to read and run
    verbatim (e.g. ``"pytest calculator/test_calc.py::test_add -v"``), but
    the harness re-runs it through :func:`agent.tools.test_runner.run_tests`
    rather than shelling the string out directly, so its structured pass/fail
    counts (and its pinned-interpreter, sandboxable invocation) are reused
    instead of re-implemented. This strips the leading ``pytest``/``py.test``
    and any CLI flags, leaving just the node-id(s) — possibly more than one,
    space-separated, which ``run_tests`` passes straight through to pytest.
    """
    tokens = verify_command.split()
    if tokens and tokens[0] in ("pytest", "py.test"):
        tokens = tokens[1:]
    node_ids = [token for token in tokens if not token.startswith("-")]
    if not node_ids:
        raise ValueError(f"verify_command has no test path to run: {verify_command!r}")
    return " ".join(node_ids)


def run_task(
    task: dict,
    condition: Literal["baseline", "react"],
    max_turns: int = 12,
    model: str | None = None,
    client=None,
) -> EvalResult:
    """Run ``task`` under ``condition`` and independently verify the outcome.

    Args:
        task: A task dict, shaped like one of eval/tasks/*.yaml (``id``,
            ``title``, ``description``, ``fixture_repo``, ``target_file``,
            ``verify_command``, ``difficulty``).
        condition: ``"baseline"`` runs M2's single LLM call;  ``"react"``
            runs M3-M6's multi-turn loop.
        max_turns: Forwarded to :func:`run_react_agent`; ignored for
            ``"baseline"``, which is always exactly one call.
        model: Model override, forwarded to whichever agent runs.
        client: Pre-built LLM client, mainly for tests — see the module
            docstring of :mod:`agent.core.llm_client`. Built from
            ``GROQ_API_KEY`` when omitted.

    Returns:
        An :class:`EvalResult`. ``success`` is decided purely by re-running
        ``task["verify_command"]`` against the final workspace state — see
        the module docstring. A crash anywhere in the agent (a bad edit that
        breaks the workspace, an LLM config error, a malformed task dict) is
        caught and returned as a failed result with ``error`` populated,
        rather than raised, so one bad task never loses the rest of an eval
        run's results. For ``"react"``, ``full_trajectory`` carries the
        loop's own turn-by-turn record for later inspection; it is ``None``
        for ``"baseline"`` and plays no part in deciding ``success``.

    Raises:
        openai.RateLimitError: Propagated rather than swallowed. Every other
            agent-side exception becomes a recorded failure (see above), but
            a provider rate limit is not a property of this one task — it
            means every subsequent call will fail too, so the caller (
            :mod:`eval.run_eval`) needs to see it as itself, stop the whole
            eval run, and tell the user to come back tomorrow, rather than
            burning through the rest of the task matrix re-discovering the
            same 429 one task at a time.
    """
    task_id = task.get("id", "<unknown>")
    start = time.monotonic()
    turns = 0
    total_tokens = 0
    agent_error: Optional[str] = None
    full_trajectory: Optional[list[Turn]] = None

    tmp_dir = tempfile.mkdtemp(prefix=f"eval-{task_id}-{condition}-")
    try:
        workspace_root = Path(tmp_dir) / "workspace"
        fixture_repo = (PROJECT_ROOT / task["fixture_repo"]).resolve()
        shutil.copytree(fixture_repo, workspace_root, ignore=_COPY_IGNORE)
        _init_git_repo(workspace_root)
        workspace = Workspace(workspace_root)

        task_description = f"{task['title']}\n\n{task['description']}"

        if condition == "baseline":
            result = run_baseline(
                workspace,
                task_description,
                task["target_file"],
                model=model,
                client=client,
            )
            turns = 1  # single-shot, by definition
            total_tokens = result["token_usage"]["total_tokens"]
            agent_error = result["error"]

        elif condition == "react":
            trajectory = run_react_agent(
                workspace,
                task_description,
                max_turns=max_turns,
                model=model,
                client=client,
            )
            turns = trajectory.total_turns
            total_tokens = trajectory.total_tokens
            full_trajectory = trajectory.turns
            if trajectory.stop_reason is StopReason.ERROR and trajectory.turns:
                agent_error = trajectory.turns[-1].tool_result.get("error")

        else:
            raise ValueError(
                f"Unknown condition {condition!r}; expected 'baseline' or 'react'"
            )

        # ---- verified, not claimed: see the module docstring -------------
        test_path = _test_path_from_verify_command(task["verify_command"])
        verification = run_tests(workspace, test_path)

        return EvalResult(
            task_id=task_id,
            condition=condition,
            success=bool(verification["success"]),
            turns=turns,
            total_tokens=total_tokens,
            wall_clock_seconds=time.monotonic() - start,
            verify_output=verification["raw_output"],
            error=agent_error,
            full_trajectory=full_trajectory,
        )

    except RateLimitError:
        raise  # not this task's failure — see the Raises section above

    except Exception as exc:  # noqa: BLE001 - a crashed run is a recorded failure
        return EvalResult(
            task_id=task_id,
            condition=condition,
            success=False,
            turns=turns,
            total_tokens=total_tokens,
            wall_clock_seconds=time.monotonic() - start,
            verify_output="",
            error=f"{type(exc).__name__}: {exc}",
            full_trajectory=full_trajectory,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
