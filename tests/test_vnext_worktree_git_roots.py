from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.vnext_app_server import VNextAppServerAdapter, _runtime_workspace_roots
from vnext.vnext_runtime_types import RuntimePosture


class Idle:
    def __init__(self) -> None:
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def recv(self) -> str | None:
        self._closed.wait()
        return None

    def send(self, payload: str) -> None:
        del payload

    def close(self) -> None:
        self._closed.set()


class WorktreeGitRootsTests(unittest.TestCase):
    """A vNext worktree child must be able to commit its own work.

    Found 2026-09-22: every worktree child reported "index.lock: Operation not
    permitted" and the coordinator committed for it. The checkout's git
    metadata lives in the parent repository, outside the one root the thread
    was given.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("git") is None:
            raise unittest.SkipTest("git is required")

    @staticmethod
    def _git(cwd: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _worktree(self, branch: str = "vnext/s1/agent-1") -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        holder = tempfile.TemporaryDirectory()
        root = Path(holder.name)
        repo = root / "repo"
        repo.mkdir()
        self._git(repo, "init")
        (repo / "README").write_text("seed\n", encoding="utf-8")
        self._git(repo, "add", "README")
        self._git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "seed")
        worktree = repo / ".vnext" / "s1" / "agent-1"
        worktree.parent.mkdir(parents=True)
        self._git(repo, "worktree", "add", "-b", branch, str(worktree))
        return holder, repo, worktree

    def test_vnext_worktree_includes_git_metadata_roots(self) -> None:
        holder, repo, worktree = self._worktree()
        self.addCleanup(holder.cleanup)
        gitdir = Path((worktree / ".git").read_text(encoding="utf-8").split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (worktree / gitdir).resolve()
        common = Path((gitdir / "commondir").read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = (gitdir / common).resolve()
        expected = {
            worktree.resolve(),
            gitdir.resolve(),
            (common / "objects").resolve(),
            (common / "refs" / "heads" / "vnext" / "s1").resolve(),
            (common / "logs" / "refs" / "heads" / "vnext" / "s1").resolve(),
        }
        self.assertEqual(expected, {Path(value).resolve() for value in _runtime_workspace_roots(worktree)})

    def test_non_vnext_worktree_gets_only_cwd(self) -> None:
        holder, _repo, worktree = self._worktree("feature/x")
        self.addCleanup(holder.cleanup)
        self.assertEqual([str(worktree.resolve())], _runtime_workspace_roots(worktree))

    def test_plain_repo_and_directory_get_only_cwd(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        plain = Path(holder.name) / "plain"
        plain.mkdir()
        self.assertEqual([str(plain.resolve())], _runtime_workspace_roots(plain))
        repo = Path(holder.name) / "repo"
        repo.mkdir()
        self._git(repo, "init")
        self.assertEqual([str(repo.resolve())], _runtime_workspace_roots(repo))

    def test_all_runtime_requests_use_extra_roots(self) -> None:
        holder, _repo, worktree = self._worktree()
        self.addCleanup(holder.cleanup)
        idle = Idle()
        adapter = VNextAppServerAdapter(
            codex_executable=Path(shutil.which("sh") or os.devnull), codex_home=holder.name, workspace=worktree,
            transport=idle,
        )
        self.addCleanup(adapter.close)
        requests: list[tuple[str, dict]] = []

        def request(method: str, params: dict, *, timeout: float = 30) -> dict:
            del timeout
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {"thread": {"id": "thread-1"}}

        adapter.request = request  # type: ignore[method-assign]
        # On Windows the first turn asks the runtime about its sandbox first;
        # that request is not under test here.
        adapter.ensure_windows_sandbox_ready = lambda **_kwargs: None  # type: ignore[method-assign]
        with patch.object(adapter, "_attest_default_environment", side_effect=lambda result, timeout: dict(result)):
            posture = RuntimePosture(
                workspace_writes=True, network="restricted", approvals_requested=True,
                reviewer="reviewer", environment_ready=True,
            )
            adapter.start_thread(
                model="model", developer_instructions="instructions", tools=[],
                requested_posture=posture, workspace=worktree,
            )
            adapter.start_turn(thread_id="thread-1", prompt="hello", model="model", effort="high", workspace=worktree)
            adapter.resume_thread(thread_id="thread-1", model="model", workspace=worktree)
        gitdir = Path((worktree / ".git").read_text(encoding="utf-8").split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (worktree / gitdir).resolve()
        common = Path((gitdir / "commondir").read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = (gitdir / common).resolve()
        roots = [
            str(worktree.resolve()), str(gitdir), str((common / "objects").resolve()),
            str((common / "refs" / "heads" / "vnext" / "s1").resolve()),
            str((common / "logs" / "refs" / "heads" / "vnext" / "s1").resolve()),
        ]
        # thread/start now asks hooks/list first, for the clock hook's trust
        # hash.  That call carries cwds rather than workspace roots, so it is
        # named here and then held out of the roots assertion.
        self.assertEqual(
            ["hooks/list", "thread/start", "turn/start", "thread/resume"],
            [method for method, _ in requests],
        )
        for method, params in requests:
            if method == "hooks/list":
                self.assertEqual([str(worktree.resolve())], params["cwds"])
                continue
            self.assertEqual(roots, params["runtimeWorkspaceRoots"])


if __name__ == "__main__":
    unittest.main()
