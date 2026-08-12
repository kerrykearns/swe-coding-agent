"""Tests for agent.tools.sandbox.

Two tiers, clearly separated:

* Unit tests (the default) mock the docker SDK entirely — ``docker.from_env``,
  ``containers.run``, ``exec_create``/``exec_start``/``exec_inspect`` — so they
  verify ``SandboxContainer``'s lifecycle logic (start, run, cleanup,
  cleanup-on-exception) and ``build_sandbox_image``'s build-once behaviour
  without needing Docker running at all.
* Real integration tests (``@pytest.mark.integration``) start actual
  containers against a real Docker daemon. They are skipped — via a fixture
  that calls ``docker.from_env().ping()``, not an env var — whenever Docker is
  not reachable, and otherwise prove the two properties that actually matter
  for M5's safety claim: no network access, and a hanging command is really
  killed rather than left to run out its full timeout.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import docker
import pytest

from agent.tools import SandboxError, Workspace, build_sandbox_image
from agent.tools import sandbox as sandbox_module
from agent.tools.sandbox import DEFAULT_IMAGE_TAG, MOUNT_POINT, SandboxContainer

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------
# unit tests — mocked docker SDK, no real Docker needed
# --------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace", create=True)


def _fake_client() -> MagicMock:
    """A MagicMock shaped like a real docker.DockerClient for the low-level
    exec_create/exec_start/exec_inspect calls SandboxContainer.run() makes.
    """
    client = MagicMock()
    client.ping.return_value = True

    container = MagicMock()
    container.id = "fake-container-id"
    client.containers.run.return_value = container

    client.api.exec_create.return_value = {"Id": "fake-exec-id"}
    client.api.exec_start.return_value = (b"hello\n", b"")
    client.api.exec_inspect.return_value = {"ExitCode": 0, "Pid": 4242}

    return client


@pytest.fixture
def fake_docker(monkeypatch) -> MagicMock:
    client = _fake_client()
    monkeypatch.setattr(sandbox_module.docker, "from_env", lambda: client)
    return client


# -- _connect() / Docker unreachable ---------------------------------------


def test_connect_raises_sandbox_error_when_docker_unreachable(monkeypatch):
    def _boom():
        raise docker.errors.DockerException("no daemon socket")

    monkeypatch.setattr(sandbox_module.docker, "from_env", _boom)

    with pytest.raises(SandboxError, match="Docker Desktop"):
        sandbox_module._connect()


def test_connect_raises_sandbox_error_when_ping_fails(monkeypatch):
    client = MagicMock()
    client.ping.side_effect = docker.errors.DockerException("daemon not responding")
    monkeypatch.setattr(sandbox_module.docker, "from_env", lambda: client)

    with pytest.raises(SandboxError):
        sandbox_module._connect()


# -- build_sandbox_image ----------------------------------------------------


def test_build_sandbox_image_skips_build_when_image_already_exists(fake_docker):
    fake_docker.images.get.return_value = MagicMock()  # exists, no exception

    build_sandbox_image()

    fake_docker.images.get.assert_called_once_with(DEFAULT_IMAGE_TAG)
    fake_docker.images.build.assert_not_called()


def test_build_sandbox_image_builds_when_image_missing(fake_docker, tmp_path):
    fake_docker.images.get.side_effect = docker.errors.ImageNotFound("no such image")
    dockerfile = tmp_path / "Dockerfile.sandbox"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    build_sandbox_image(tag="some-tag:latest", dockerfile=dockerfile)

    fake_docker.images.build.assert_called_once_with(
        path=str(tmp_path), dockerfile="Dockerfile.sandbox", tag="some-tag:latest", rm=True
    )


def test_build_sandbox_image_raises_sandbox_error_on_build_failure(fake_docker, tmp_path):
    fake_docker.images.get.side_effect = docker.errors.ImageNotFound("no such image")
    fake_docker.images.build.side_effect = docker.errors.APIError("build failed")
    dockerfile = tmp_path / "Dockerfile.sandbox"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    with pytest.raises(SandboxError, match="Failed to build"):
        build_sandbox_image(dockerfile=dockerfile)


def test_build_sandbox_image_raises_when_docker_unreachable(monkeypatch):
    monkeypatch.setattr(
        sandbox_module.docker,
        "from_env",
        lambda: (_ for _ in ()).throw(docker.errors.DockerException("no daemon")),
    )

    with pytest.raises(SandboxError):
        build_sandbox_image()


# -- SandboxContainer.__enter__ ---------------------------------------------


def test_enter_starts_container_with_expected_isolation_settings(fake_docker, ws):
    with SandboxContainer(ws) as sandbox:
        assert sandbox._container is fake_docker.containers.run.return_value

    _, kwargs = fake_docker.containers.run.call_args
    assert kwargs["network_mode"] == "none"
    assert kwargs["volumes"] == {str(ws.root): {"bind": MOUNT_POINT, "mode": "rw"}}
    assert kwargs["working_dir"] == MOUNT_POINT
    assert kwargs["mem_limit"] == "512m"
    assert kwargs["nano_cpus"] == 1_000_000_000
    assert kwargs["detach"] is True


def test_enter_raises_sandbox_error_when_image_missing(fake_docker, ws):
    fake_docker.containers.run.side_effect = docker.errors.NotFound("no such image")

    with pytest.raises(SandboxError, match="build_sandbox_image"):
        with SandboxContainer(ws):
            pass


def test_enter_raises_sandbox_error_when_container_start_fails(fake_docker, ws):
    fake_docker.containers.run.side_effect = docker.errors.APIError("daemon refused")

    with pytest.raises(SandboxError, match="Failed to start"):
        with SandboxContainer(ws):
            pass


# -- SandboxContainer.run() -------------------------------------------------


def test_run_outside_with_block_raises(ws):
    sandbox = SandboxContainer(ws)
    with pytest.raises(SandboxError, match="'with' block"):
        sandbox.run("echo hi")


def test_run_returns_shell_exec_shaped_dict(fake_docker, ws):
    with SandboxContainer(ws) as sandbox:
        result = sandbox.run("echo hello")

    assert set(result) == {"stdout", "stderr", "exit_code", "timed_out"}
    assert result["stdout"] == "hello\n"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False


def test_run_execs_via_sh_c_with_the_given_command(fake_docker, ws):
    with SandboxContainer(ws) as sandbox:
        sandbox.run("pytest -q")

    args, kwargs = fake_docker.api.exec_create.call_args
    assert args[0] == "fake-container-id"
    assert args[1] == ["/bin/sh", "-c", "pytest -q"]


def test_run_reports_nonzero_exit_code(fake_docker, ws):
    fake_docker.api.exec_inspect.return_value = {"ExitCode": 3, "Pid": 1}

    with SandboxContainer(ws) as sandbox:
        result = sandbox.run("exit 3")

    assert result["exit_code"] == 3
    assert result["timed_out"] is False


# -- SandboxContainer.run() timeout handling --------------------------------


def test_run_timeout_kills_the_exec_pid_and_reports_timed_out(fake_docker, ws, monkeypatch):
    monkeypatch.setattr(sandbox_module, "_KILL_GRACE_SECONDS", 2)
    fake_docker.api.exec_inspect.return_value = {"ExitCode": None, "Pid": 9999}

    def _slow_exec_start(*args, **kwargs):
        time.sleep(0.2)
        return (b"", b"")

    fake_docker.api.exec_start.side_effect = _slow_exec_start

    with SandboxContainer(ws) as sandbox:
        result = sandbox.run("sleep 60", timeout=0.05)
        exec_run_mock = sandbox._container.exec_run

    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert "timed out" in result["stderr"]
    exec_run_mock.assert_called_once_with(["kill", "-9", "9999"])


def test_run_timeout_kills_whole_container_if_pid_kill_does_not_unblock(
    fake_docker, ws, monkeypatch
):
    monkeypatch.setattr(sandbox_module, "_KILL_GRACE_SECONDS", 0.05)
    fake_docker.api.exec_inspect.return_value = {"ExitCode": None, "Pid": 1}

    def _never_returns(*args, **kwargs):
        time.sleep(5)
        return (b"", b"")  # pragma: no cover - never reached within the test

    fake_docker.api.exec_start.side_effect = _never_returns

    with SandboxContainer(ws) as sandbox:
        result = sandbox.run("sleep 60", timeout=0.01)
        assert sandbox._container.kill.called

    assert result["timed_out"] is True
    assert result["exit_code"] == -1


# -- SandboxContainer cleanup ------------------------------------------------


def test_cleanup_removes_container_on_normal_exit(fake_docker, ws):
    with SandboxContainer(ws) as sandbox:
        container = sandbox._container

    container.remove.assert_called_once_with(force=True)


def test_cleanup_removes_container_on_exception(fake_docker, ws):
    container_holder = {}
    with pytest.raises(ValueError):
        with SandboxContainer(ws) as sandbox:
            container_holder["container"] = sandbox._container
            raise ValueError("boom")

    container_holder["container"].remove.assert_called_once_with(force=True)


def test_cleanup_tolerates_container_already_gone(fake_docker, ws):
    with SandboxContainer(ws) as sandbox:
        sandbox._container.remove.side_effect = docker.errors.NotFound("already gone")
    # __exit__ must not raise even though remove() failed.


def test_cleanup_is_a_noop_if_container_never_started(ws):
    sandbox = SandboxContainer(ws)
    sandbox._cleanup()  # nothing to clean up; must not raise


# --------------------------------------------------------------------------
# real integration tests — require a reachable Docker daemon and the sandbox
# image built (`docker build -t swe-agent-sandbox:latest -f Dockerfile.sandbox .`)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docker_client():
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any reason Docker is unreachable
        pytest.skip(f"Docker is not reachable: {exc}")
    yield client
    client.close()


@pytest.mark.integration
def test_real_sandbox_runs_a_command_and_cleans_up_after_itself(docker_client, tmp_path):
    ws = Workspace(tmp_path / "workspace", create=True)

    with SandboxContainer(ws) as sandbox:
        container_id = sandbox._container.id
        result = sandbox.run("echo hello-from-the-sandbox")

    assert "hello-from-the-sandbox" in result["stdout"]
    assert result["exit_code"] == 0
    assert result["timed_out"] is False

    with pytest.raises(docker.errors.NotFound):
        docker_client.containers.get(container_id)


@pytest.mark.integration
def test_real_sandbox_has_no_network_access(tmp_path):
    ws = Workspace(tmp_path / "workspace", create=True)

    with SandboxContainer(ws) as sandbox:
        result = sandbox.run(
            "python -c \"import urllib.request as u; "
            "u.urlopen('https://google.com', timeout=3)\"",
            timeout=15,
        )

    assert result["exit_code"] != 0
    assert result["timed_out"] is False, (
        "a real network failure (no route / DNS) should fail fast, not hang "
        "until the timeout -- a timeout here would suggest network access "
        "actually exists and the request just never got a response"
    )


@pytest.mark.integration
def test_real_sandbox_kills_a_hanging_command_within_the_timeout(tmp_path):
    ws = Workspace(tmp_path / "workspace", create=True)

    with SandboxContainer(ws) as sandbox:
        start = time.monotonic()
        result = sandbox.run("sleep 60", timeout=5)
        elapsed = time.monotonic() - start

    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert elapsed < 30, (
        f"took {elapsed:.1f}s to come back; a 5s timeout should kill the "
        "hanging process well before the full 60s sleep completes"
    )
