"""The agent's "hands": pure-Python tools for touching a code workspace.

Nothing in this package knows about LLMs. Every tool takes a
:class:`~agent.tools.workspace.Workspace` as its first argument and routes all
filesystem access through it, so the agent can only reach inside the directory
it was given.

Typical use::

    from agent.tools import Workspace, read_file, run_tests

    ws = Workspace("/path/to/repo")
    source = read_file(ws, "calculator/calc.py")
    results = run_tests(ws)
"""

from .diff_tools import apply_patch, get_diff, get_status
from .file_ops import list_files, read_file, search_text, write_file
from .sandbox import SandboxContainer, SandboxError, build_sandbox_image
from .shell_exec import run_command
from .test_runner import run_tests
from .workspace import Workspace, WorkspaceViolation

__all__ = [
    # workspace containment
    "Workspace",
    "WorkspaceViolation",
    # file operations
    "read_file",
    "write_file",
    "list_files",
    "search_text",
    # shell execution — see the SAFETY NOTE in shell_exec.run_command
    "run_command",
    # test execution
    "run_tests",
    # git diff / patching
    "get_diff",
    "get_status",
    "apply_patch",
    # Docker sandboxing — see agent.tools.sandbox
    "SandboxContainer",
    "SandboxError",
    "build_sandbox_image",
]
