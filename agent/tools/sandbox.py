"""Docker-based isolation for shell execution and test runs.

Wraps :mod:`docker`'s SDK so :mod:`shell_exec` and :mod:`test_runner` can run a
command inside a disposable, network-isolated container instead of a local
subprocess, without changing their public API. Nothing here runs by default —
:func:`build_sandbox_image` is a one-time setup step, and
:class:`SandboxContainer` is only used when a caller opts in with
``sandboxed=True``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

from .workspace import Workspace

__all__ = [
    "DEFAULT_IMAGE_TAG",
    "SandboxContainer",
    "SandboxError",
    "build_sandbox_image",
]

#: Tag used everywhere a caller doesn't specify one.
DEFAULT_IMAGE_TAG = "swe-agent-sandbox:latest"

#: Dockerfile.sandbox lives at the repo root; this file is agent/tools/sandbox.py.
_DEFAULT_DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile.sandbox"

#: Where the workspace is mounted inside the container.
MOUNT_POINT = "/workspace"

DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = 1.0

#: Grace period given to a thread whose exec's process was just killed, so its
#: blocking read on the exec stream has a moment to unblock before we give up
#: on it (it is a daemon thread either way, so giving up just means we stop
#: waiting -- it does not leak past process exit).
_KILL_GRACE_SECONDS = 5


class SandboxError(Exception):
    """Docker is unreachable, or an image/container operation failed."""


def _connect() -> docker.DockerClient:
    """Return a live Docker client, or raise :class:`SandboxError`.

    ``docker.from_env()`` alone does not fail when the daemon is unreachable —
    it builds a client lazily. ``ping()`` is what actually talks to the socket.
    """
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        raise SandboxError(
            "Could not reach Docker. Docker Desktop does not appear to be "
            "running. Start it and try again."
        ) from exc
    return client


def build_sandbox_image(
    tag: str = DEFAULT_IMAGE_TAG,
    dockerfile: str | Path = _DEFAULT_DOCKERFILE,
) -> None:
    """Build the sandbox image from ``dockerfile`` if ``tag`` doesn't exist yet.

    A one-time setup step, not something run before every test or every agent
    turn — building on a cold cache is slow, so this checks for the image
    first and does nothing if it is already there.

    Args:
        tag: Image tag to build and check for.
        dockerfile: Path to the Dockerfile. Defaults to ``Dockerfile.sandbox``
            at the repo root.

    Raises:
        SandboxError: Docker is unreachable, or the build itself fails.
    """
    client = _connect()
    try:
        client.images.get(tag)
        return
    except ImageNotFound:
        pass

    dockerfile_path = Path(dockerfile)
    try:
        client.images.build(
            path=str(dockerfile_path.parent),
            dockerfile=dockerfile_path.name,
            tag=tag,
            rm=True,
        )
    except (APIError, DockerException) as exc:
        raise SandboxError(f"Failed to build sandbox image {tag!r}: {exc}") from exc


class SandboxContainer:
    """One disposable, network-isolated container mounting a workspace.

    Used as a context manager::

        with SandboxContainer(workspace) as sandbox:
            result = sandbox.run("pytest -q")

    On enter, starts a container from the sandbox image with the workspace
    mounted read-write at ``/workspace``, ``network_mode="none"``, a memory
    limit, and a CPU limit. On exit -- including on an exception inside the
    ``with`` block -- the container is always stopped and removed.

    Args:
        workspace: Mounted read-write at ``/workspace`` inside the container.
        image: Sandbox image tag; must already be built (see
            :func:`build_sandbox_image`).
        memory_limit: Docker memory limit string, e.g. ``"512m"``.
        cpu_limit: Number of CPUs, e.g. ``1.0``.

    Raises:
        SandboxError: Docker is unreachable, the image is missing, or the
            container fails to start.
    """

    def __init__(
        self,
        workspace: Workspace,
        image: str = DEFAULT_IMAGE_TAG,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
    ) -> None:
        self.workspace = workspace
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._client: Optional[docker.DockerClient] = None
        self._container = None

    def __enter__(self) -> "SandboxContainer":
        self._client = _connect()
        try:
            self._container = self._client.containers.run(
                self.image,
                command="sleep infinity",
                volumes={
                    str(self.workspace.root): {"bind": MOUNT_POINT, "mode": "rw"}
                },
                working_dir=MOUNT_POINT,
                network_mode="none",
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1_000_000_000),
                detach=True,
                tty=False,
            )
        except NotFound as exc:
            raise SandboxError(
                f"Sandbox image {self.image!r} was not found. Build it first "
                "with agent.tools.sandbox.build_sandbox_image()."
            ) from exc
        except (APIError, DockerException) as exc:
            raise SandboxError(f"Failed to start sandbox container: {exc}") from exc
        return self

    def run(self, command: str, timeout: int = 30) -> dict:
        """Run ``command`` inside the container via ``/bin/sh -c``.

        Args:
            command: Shell command line, run inside the container's
                ``/bin/sh`` with cwd at ``/workspace``.
            timeout: Seconds to wait before killing the exec'd process.

        Returns:
            ``{"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}``
            -- the same shape as :func:`agent.tools.shell_exec.run_command`.
            On timeout, ``timed_out`` is True and ``exit_code`` is -1.

        Raises:
            SandboxError: Called outside a ``with`` block, or the exec call
                itself fails (as opposed to the executed command failing,
                which is reported in the returned dict, not raised).
        """
        if self._container is None:
            raise SandboxError("SandboxContainer.run() called outside a 'with' block.")

        api = self._client.api
        exec_id = api.exec_create(
            self._container.id,
            ["/bin/sh", "-c", command],
            stdout=True,
            stderr=True,
        )["Id"]

        result: dict = {}

        def _target() -> None:
            try:
                stdout, stderr = api.exec_start(exec_id, demux=True)
                result["stdout"] = stdout.decode("utf-8", errors="replace") if stdout else ""
                result["stderr"] = stderr.decode("utf-8", errors="replace") if stderr else ""
            except Exception as exc:  # noqa: BLE001 - surfaced via `result`
                result["thread_error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            return self._handle_timeout(thread, exec_id, timeout, result)

        if "thread_error" in result:
            raise SandboxError(f"Failed to exec command in sandbox: {result['thread_error']}")

        exit_code = api.exec_inspect(exec_id).get("ExitCode")
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": exit_code if exit_code is not None else -1,
            "timed_out": False,
        }

    def _handle_timeout(
        self, thread: threading.Thread, exec_id: str, timeout: int, result: dict
    ) -> dict:
        """Kill the exec'd process and report a timeout, actually stopping it.

        ``exec_run``/``exec_start`` have no native cancel, so the only way to
        stop the underlying process is to kill it from the outside: look up
        its PID via ``exec_inspect`` and send it a ``kill -9`` from a second,
        short-lived exec. If that alone doesn't unblock the reading thread
        within a grace period, the whole container is killed as a last
        resort -- correct behaviour for a run that will be torn down anyway,
        even though it means this ``SandboxContainer`` can't run anything
        else afterwards.
        """
        api = self._client.api
        try:
            pid = api.exec_inspect(exec_id).get("Pid")
        except (APIError, DockerException):
            pid = None

        if pid:
            try:
                self._container.exec_run(["kill", "-9", str(pid)])
            except (APIError, DockerException):
                pass

        thread.join(_KILL_GRACE_SECONDS)

        if thread.is_alive():
            try:
                self._container.kill()
            except (APIError, DockerException):
                pass
            thread.join(_KILL_GRACE_SECONDS)

        stderr = result.get("stderr", "")
        stderr += f"\nCommand timed out after {timeout} seconds and was killed."
        return {
            "stdout": result.get("stdout", ""),
            "stderr": stderr,
            "exit_code": -1,
            "timed_out": True,
        }

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._container is None:
            return
        try:
            self._container.remove(force=True)
        except (APIError, DockerException, NotFound):
            pass
        self._container = None
