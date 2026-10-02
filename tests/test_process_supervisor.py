from __future__ import annotations

import contextlib
import ctypes
import errno
import importlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vnext import process_supervisor
from vnext.process_supervisor import OwnedProcess, _PosixGroup

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def process_is_running(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        kernel32.CloseHandle(handle)


class OwnedProcessTests(unittest.TestCase):
    def test_cleanup_kills_owned_descendants_but_not_unrelated_process(self):
        parent_code = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        owned = OwnedProcess.start(
            [sys.executable, "-u", "-c", parent_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertIsNotNone(owned.process.stdout)
            child_pid = int(owned.process.stdout.readline().strip())

            receipt = owned.close(grace_seconds=0.01)
            deadline = time.monotonic() + 2
            while process_is_running(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)

            self.assertEqual("clean", receipt.outcome)
            self.assertEqual(0, receipt.residual_count)
            self.assertFalse(process_is_running(child_pid))
            self.assertIsNone(unrelated.poll())
            self.assertIs(receipt, owned.close())
        finally:
            if owned.process.poll() is None:
                owned.close(grace_seconds=0)
            for stream in (owned.process.stdout, owned.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            if unrelated.poll() is None:
                unrelated.terminate()
                try:
                    unrelated.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    unrelated.kill()
                    unrelated.wait(timeout=2)


@unittest.skipIf(os.name == "nt", "the process group is a POSIX object")
class ARefusedSignalToAnEmptyGroupTests(unittest.TestCase):
    """Darwin answers a kill aimed at an emptied group with EPERM.

    It decides whether the caller may signal the group before it decides
    whether the group still has members, so an emptied group refuses where
    Linux reports no such process. The kill caught only the Linux answer, the
    refusal reached the shutdown receipt as an error, and every session on this
    Mac ended with the verdict "residual" although nothing had survived. Three
    tests failed on that one line.
    """

    def test_a_refused_kill_does_not_escape_as_a_shutdown_error(self):
        group = _PosixGroup(os.getpid())
        with patch.object(sys, "platform", "darwin"):
            with patch.object(_PosixGroup, "active_processes", return_value=0):
                with patch.object(os, "killpg", side_effect=PermissionError(1, "Operation not permitted")):
                    # close() turns anything raised here into a receipt error, and
                    # an error is what makes the verdict "residual".
                    group.terminate()

    def test_the_second_kill_may_be_refused_too(self):
        """A group that empties between the two signals refuses the second one."""

        group = _PosixGroup(os.getpid())
        # The wait loop reads the count first and the escalation reads it after:
        # nothing is running, then one straggler earns the second signal.
        answers = iter([0, 1])
        with patch.object(sys, "platform", "darwin"):
            with patch.object(_PosixGroup, "active_processes", side_effect=lambda: next(answers, 0)):
                with patch.object(os, "killpg") as killpg:
                    killpg.side_effect = [None, PermissionError(1, "Operation not permitted")]
                    group.terminate()
        self.assertEqual(2, killpg.call_count)

    def test_a_refused_probe_still_stands_the_watchers_down_on_darwin(self):
        """terminate() stops at the refusal, so close_handle must disarm on it too.

        A watcher runs as the same user as its owner, so it can stop nothing
        that refuses the owner. Left armed, it would only ever hit the id once
        the kernel hands it to somebody else. The count stays cautious.
        """

        group = _PosixGroup(os.getpid())
        read_end, write_end = os.pipe()
        group._posts = [process_supervisor._WatcherPost(process=MagicMock(), link=write_end)]
        try:
            with patch.object(sys, "platform", "darwin"):
                with patch.object(os, "killpg", side_effect=PermissionError(1, "Operation not permitted")):
                    self.assertEqual(1, group.active_processes())
                    group.close_handle()
            self.assertEqual(process_supervisor._WATCHER_DISARM, os.read(read_end, 16))
        finally:
            os.close(read_end)

    def test_linux_permission_failure_reaches_the_shutdown_receipt(self):
        group = _PosixGroup(os.getpid())
        with patch.object(sys, "platform", "linux"):
            with patch.object(os, "killpg", side_effect=PermissionError(1, "Operation not permitted")):
                with self.assertRaises(PermissionError):
                    group.terminate()

    def test_a_refusal_is_never_read_as_a_clean_shutdown_on_its_own(self):
        """The survivors are counted after the kill, so the verdict is earned.

        Catching the refusal would be wrong if it also silenced a group that
        really did survive. It cannot: close() asks the count again and reports
        whatever is still there.
        """

        owned = OwnedProcess.start(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with patch.object(_PosixGroup, "active_processes", return_value=3):
                with patch.object(os, "killpg", side_effect=PermissionError(1, "nope")):
                    receipt = owned.close(grace_seconds=0.01)
            self.assertEqual(3, receipt.residual_count)
            self.assertEqual("residual", receipt.outcome)
        finally:
            owned.process.kill()
            owned.process.wait(timeout=5)
            for stream in (owned.process.stdout, owned.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


CHILD_SOURCE = """
import subprocess, sys, time

grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
print(grandchild.pid, flush=True)
time.sleep(300)
"""

PARENT_SOURCE = """
import subprocess, sys, time

from vnext.process_supervisor import OwnedProcess

owned = OwnedProcess.start(
    [sys.executable, "-u", sys.argv[1]],
    stdout=subprocess.PIPE,
    text=True,
)
print(owned.process.pid, owned.process.stdout.readline().strip(), flush=True)
time.sleep(300)
"""


@unittest.skipIf(os.name == "nt", "the Windows job already kills the tree when its handle closes")
class AnOwnerKilledWithoutWarningTests(unittest.TestCase):
    """A SIGKILL leaves the owner no chance to run close().

    Before the watcher, the worker and everything it had started kept running
    after Claude Code or the vNext server was killed or crashed, because
    start_new_session puts the group beyond the reach of anything the dying
    process no longer gets to do.
    """

    def test_the_owned_group_is_gone_five_seconds_after_the_owner_is_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            child_script = Path(directory) / "child.py"
            parent_script = Path(directory) / "parent.py"
            child_script.write_text(CHILD_SOURCE, encoding="utf-8")
            parent_script.write_text(PARENT_SOURCE, encoding="utf-8")

            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(REPOSITORY_ROOT), environment.get("PYTHONPATH", ""))
            )
            parent = subprocess.Popen(
                [sys.executable, "-u", str(parent_script), str(child_script)],
                stdout=subprocess.PIPE,
                text=True,
                env=environment,
            )
            child_pid = grandchild_pid = None
            try:
                self.assertIsNotNone(parent.stdout)
                child_pid, grandchild_pid = (int(part) for part in parent.stdout.readline().split())
                self.assertTrue(process_is_running(child_pid))
                self.assertTrue(process_is_running(grandchild_pid))

                os.kill(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)

                group = _PosixGroup(child_pid)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and (
                    group.active_processes() or process_is_running(grandchild_pid)
                ):
                    time.sleep(0.05)

                self.assertEqual(0, group.active_processes())
                self.assertFalse(process_is_running(grandchild_pid))
                with self.assertRaises(OSError):
                    os.killpg(child_pid, 0)
            finally:
                if parent.stdout is not None and not parent.stdout.closed:
                    parent.stdout.close()
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                for pid in (grandchild_pid, child_pid):
                    if pid is None:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass

    def test_an_orderly_shutdown_leaves_no_watcher_behind(self):
        owned = OwnedProcess.start(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        watchers = [post.process for post in owned._tree._posts]
        try:
            self.assertEqual(process_supervisor._WATCHERS_PER_GROUP, len(watchers))
            for watcher in watchers:
                self.assertIsNone(watcher.poll())
            receipt = owned.close(grace_seconds=0.01)
            self.assertEqual("clean", receipt.outcome)
            self.assertEqual(0, receipt.residual_count)
            for watcher in watchers:
                self.assertIsNotNone(watcher.poll())
            self.assertEqual([], owned._tree._posts)
        finally:
            if owned.process.poll() is None:
                owned.close(grace_seconds=0)
            for stream in (owned.process.stdout, owned.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class AWorkerIsNeverLeftUnwatchedTests(unittest.TestCase):
    """The watcher can fail to spawn after the worker is already running.

    The fork that starts the worker can succeed and the next one hit the
    per-user process limit, so a caller could be handed a worker that nothing
    watches: it would outlive a SIGKILL of its owner, which is the whole fault
    the watcher exists to stop. start() fails closed instead.
    """

    def test_a_watcher_that_cannot_spawn_takes_the_worker_with_it(self):
        real_popen = subprocess.Popen
        started: list[subprocess.Popen] = []

        def refuse_the_watcher(command, **kwargs):
            if len(command) >= 3 and command[2] == process_supervisor._PARENT_DEATH_WATCHER:
                raise OSError(errno.EAGAIN, "injected process limit")
            worker = real_popen(command, **kwargs)
            started.append(worker)
            return worker

        try:
            with patch.object(subprocess, "Popen", refuse_the_watcher):
                with self.assertRaises(OSError):
                    OwnedProcess.start(
                        [sys.executable, "-c", "import time; time.sleep(300)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            self.assertEqual(1, len(started))
            worker = started[0]
            self.assertIsNotNone(worker.poll())
            self.assertFalse(process_is_running(worker.pid))
        finally:
            for worker in started:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except OSError:
                    pass
                if worker.poll() is None:
                    worker.wait(timeout=5)


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class ARefusedStandbyWatcherSaysSoTests(unittest.TestCase):
    """The standby watcher can be refused while the first one is armed fine.

    The fork that refuses it is the machine at its process limit, and a worker
    with one watcher is watched as well as it was before the standby existed, so
    start() goes on succeeding. What the group loses is the cover over the
    window where a watcher is replaced, and that used to be visible only by
    reading the group's private list of posts. The owner now writes one line to
    stderr naming the worker and carrying the error.
    """

    def test_a_refused_standby_leaves_start_working_and_one_line_on_stderr(self):
        real_spawn = _PosixGroup._spawn_watcher

        def refuse_only_the_spare(group):
            if threading.current_thread().name.startswith("parent-death-arm-"):
                raise OSError(errno.EAGAIN, "injected process limit")
            return real_spawn(group)

        reported = io.StringIO()
        owned = None
        try:
            with patch.object(_PosixGroup, "_spawn_watcher", refuse_only_the_spare):
                with contextlib.redirect_stderr(reported):
                    owned = OwnedProcess.start(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            self.assertIsNone(owned.process.poll(), "the worker still starts")
            watchers = [post.process for post in owned._tree._posts]
            self.assertEqual(1, len(watchers), "the group lives with its first watcher")
            self.assertIsNone(watchers[0].poll())

            said = reported.getvalue().splitlines()
            self.assertEqual(1, len(said), f"exactly one line was written: {said}")
            line = said[0]
            self.assertIn(str(owned.process.pid), line, "the line names the worker pid")
            self.assertIn("one watcher", line)
            self.assertIn("injected process limit", line, "the line carries the error")

            receipt = owned.close(grace_seconds=0.01)
            self.assertEqual("clean", receipt.outcome)
            self.assertEqual(0, receipt.residual_count)
            self.assertIsNotNone(watchers[0].poll(), "the one watcher stood down")
            self.assertEqual([], owned._tree._posts)
        finally:
            if owned is not None and owned.process.poll() is None:
                owned.close(grace_seconds=0)


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class AnOrderlyCloseSignalsNothingTests(unittest.TestCase):
    """A closed pipe alone cannot tell a crash from a shutdown.

    close() empties the group and then releases the watcher, so the watcher woke
    up on end of file and aimed a SIGTERM at a process group id whose leader had
    already been reaped. The kernel is free to hand that number to somebody
    else's group, and the watcher would have signalled it. So the owner writes a
    disarm byte before it closes the pipe, and end of file with no byte in front
    of it is the only thing read as the death of the owner.
    """

    def test_an_orderly_close_leaves_the_watcher_with_nothing_to_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "signals.log"
            traced_watcher = (
                "import os\n"
                "_real_killpg = os.killpg\n"
                "def _traced_killpg(group, number):\n"
                f"    with open({str(trace)!r}, 'a') as _log:\n"
                '        _log.write(f"{group} {number}\\n")\n'
                "    return _real_killpg(group, number)\n"
                "os.killpg = _traced_killpg\n"
            ) + process_supervisor._PARENT_DEATH_WATCHER

            with patch.object(process_supervisor, "_PARENT_DEATH_WATCHER", traced_watcher):
                owned = OwnedProcess.start(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            watchers = [post.process for post in owned._tree._posts]
            try:
                self.assertEqual(process_supervisor._WATCHERS_PER_GROUP, len(watchers))
                receipt = owned.close(grace_seconds=0.01)
                self.assertEqual("clean", receipt.outcome)
                self.assertEqual(0, receipt.residual_count)
                for watcher in watchers:
                    self.assertEqual(0, watcher.poll(), "every watcher stood down")
                self.assertFalse(
                    trace.exists(),
                    f"the watcher signalled after an orderly close: {trace.read_text() if trace.exists() else ''}",
                )
            finally:
                if owned.process.poll() is None:
                    owned.close(grace_seconds=0)




@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class AWatcherThatDiesBeforeItArmsTests(unittest.TestCase):
    """A watcher can spawn and then die on its way to its first read.

    The Popen error above is the only watcher failure start() used to catch, so a
    watcher whose interpreter exited at once left a worker nobody watched: a
    probe that forced exit 77 saw start() return and the worker outlive a SIGKILL
    of its owner. The watcher now writes one byte the moment it is in position
    and start() waits for it, so an early exit closes the ready pipe and the
    arming fails with the worker cleaned up behind it.
    """

    def test_a_watcher_that_exits_at_once_takes_the_worker_with_it(self):
        early_exit = "import sys\nsys.exit(77)\n"
        real_popen = subprocess.Popen
        workers: list[subprocess.Popen] = []

        def remember_the_worker(command, **kwargs):
            process = real_popen(command, **kwargs)
            if len(command) < 3 or command[2] != early_exit:
                workers.append(process)
            return process

        try:
            with patch.object(process_supervisor, "_PARENT_DEATH_WATCHER", early_exit):
                with patch.object(subprocess, "Popen", remember_the_worker):
                    with self.assertRaises(process_supervisor.ProcessSupervisionError):
                        OwnedProcess.start(
                            [sys.executable, "-c", "import time; time.sleep(300)"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            self.assertEqual(1, len(workers))
            worker = workers[0]
            self.assertIsNotNone(worker.poll())
            self.assertFalse(process_is_running(worker.pid))
        finally:
            for worker in workers:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except OSError:
                    pass
                if worker.poll() is None:
                    worker.wait(timeout=5)


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class AWatcherThatDiesAfterItArmsTests(unittest.TestCase):
    """The ready byte proves the watcher was alive once, and nothing after that.

    A probe injected a watcher that wrote the ready byte and then exited 78
    before its first read of the owner link. start() returned, and the worker
    ran with nobody left to kill it if the owner were killed. The owner now
    keeps a thread on its watcher for the whole life of the group: one
    replacement is armed, and when that one cannot hold either the group is
    terminated, the same ending as a watcher that never armed at all.
    """

    def _watcher_that_arms_and_exits(self) -> str:
        return (
            "import os, sys\n"
            "ready = int(sys.argv[3])\n"
            f"os.write(ready, {process_supervisor._WATCHER_READY!r})\n"
            "os.close(ready)\n"
            "sys.exit(78)\n"
        )

    def test_a_watcher_that_dies_after_it_reported_ready_still_stops_the_worker(self):
        program = self._watcher_that_arms_and_exits()
        real_popen = subprocess.Popen
        workers: list[subprocess.Popen] = []
        watchers: list[subprocess.Popen] = []

        def sort_the_children(command, **kwargs):
            process = real_popen(command, **kwargs)
            if len(command) > 2 and command[2] == program:
                watchers.append(process)
            else:
                workers.append(process)
            return process

        owned = None
        try:
            with patch.object(process_supervisor, "_PARENT_DEATH_WATCHER", program):
                with patch.object(subprocess, "Popen", sort_the_children):
                    owned = OwnedProcess.start(
                        [sys.executable, "-c", "import time; time.sleep(300)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self.assertEqual(1, len(workers))
                    worker = workers[0]
                    deadline = time.monotonic() + 15.0
                    while worker.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertIsNotNone(
                        worker.poll(),
                        "the worker outlived a watcher that died after arming",
                    )
                    self.assertLess(worker.poll(), 0, "the worker was signalled")
                    self.assertEqual(
                        process_supervisor._WATCHERS_PER_GROUP + 1,
                        len(watchers),
                        "one replacement is armed on top of the pair before the "
                        "group is stopped",
                    )
        finally:
            if owned is not None:
                owned.close(grace_seconds=0)
            for process in workers + watchers:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    process.wait(timeout=5)


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class TheReplacementWatcherTakesOverTests(unittest.TestCase):
    """A replacement is only worth arming if it really watches.

    The test above ends in a terminated group, which a replacement that did
    nothing at all would also produce. This one lets the second watcher be the
    real program, then closes the owner's end of the link with no stand-down
    byte -- what the kernel does when the owner is killed -- and the group has
    to go down at the replacement's hand.
    """

    def _one_flaky_watcher_then_the_real_one(self, marker: Path) -> str:
        return (
            "import os, sys\n"
            f"marker = {str(marker)!r}\n"
            # Both watchers of the pair start at the same moment, so the claim on
            # the marker has to be the atomic one or they could both be flaky.
            "try:\n"
            "    os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))\n"
            "except FileExistsError:\n"
            "    pass\n"
            "else:\n"
            "    ready = int(sys.argv[3])\n"
            f"    os.write(ready, {process_supervisor._WATCHER_READY!r})\n"
            "    os.close(ready)\n"
            "    sys.exit(78)\n"
        ) + process_supervisor._PARENT_DEATH_WATCHER

    def test_the_replacement_kills_the_group_when_the_owner_link_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            program = self._one_flaky_watcher_then_the_real_one(
                Path(directory) / "first-watcher-ran"
            )
            real_popen = subprocess.Popen
            workers: list[subprocess.Popen] = []
            watchers: list[subprocess.Popen] = []

            def sort_the_children(command, **kwargs):
                process = real_popen(command, **kwargs)
                if len(command) > 2 and command[2] == program:
                    watchers.append(process)
                else:
                    workers.append(process)
                return process

            owned = None
            try:
                with patch.object(process_supervisor, "_PARENT_DEATH_WATCHER", program):
                    with patch.object(subprocess, "Popen", sort_the_children):
                        owned = OwnedProcess.start(
                            [sys.executable, "-c", "import time; time.sleep(300)"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        wanted = process_supervisor._WATCHERS_PER_GROUP + 1
                        deadline = time.monotonic() + 15.0
                        while len(watchers) < wanted and time.monotonic() < deadline:
                            time.sleep(0.05)
                        self.assertEqual(
                            wanted, len(watchers), "no replacement was armed"
                        )
                        worker = workers[0]
                        self.assertIsNone(
                            worker.poll(),
                            "the group was stopped instead of watched again",
                        )
                        replacement = watchers[-1]
                        self.assertIsNone(replacement.poll(), "the replacement is on duty")

                        # What the kernel does to this descriptor when the owner
                        # is killed: closed, with no stand-down byte in front of
                        # it.  The replacement has to read that as a death, and
                        # only its own link is closed here so the kill can have
                        # come from nowhere else.
                        post = next(
                            post
                            for post in owned._tree._posts
                            if post.process is replacement
                        )
                        os.close(post.link)
                        post.link = None
                        deadline = time.monotonic() + 15.0
                        while worker.poll() is None and time.monotonic() < deadline:
                            time.sleep(0.05)
                        self.assertIsNotNone(
                            worker.poll(),
                            "the replacement watcher never stopped the group",
                        )
                        self.assertLess(worker.poll(), 0, "the worker was signalled")
            finally:
                if owned is not None:
                    owned.close(grace_seconds=0)
                for process in workers + watchers:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    if process.poll() is None:
                        try:
                            process.kill()
                        except OSError:
                            pass
                        process.wait(timeout=5)


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class TheOwnerCanDieWhileAWatcherIsBeingReplacedTests(unittest.TestCase):
    """Replacing a watcher used to leave a moment with nobody on duty.

    A verifier killed the owner on entry to the replacement spawn, 2.3 ms after
    the first watcher died, and the worker lived: the dead watcher's link was
    already released and the replacement did not exist yet. So the group is
    given two watchers from the start. One death still leaves a watcher holding
    a link, the replacement is armed behind that cover, and only the death of
    both leaves the group unwatched -- which ends in a terminated group.

    The owner has to be killed with SIGKILL for the kernel to close the link the
    way a crash does, so the owner is a subprocess of this test.
    """

    def _one_flaky_watcher_then_the_real_one(self, marker: Path) -> str:
        return (
            "import os, sys\n"
            f"marker = {str(marker)!r}\n"
            "try:\n"
            "    os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))\n"
            "except FileExistsError:\n"
            "    pass\n"
            "else:\n"
            "    ready = int(sys.argv[3])\n"
            f"    os.write(ready, {process_supervisor._WATCHER_READY!r})\n"
            "    os.close(ready)\n"
            "    sys.exit(78)\n"
        ) + process_supervisor._PARENT_DEATH_WATCHER

    def _owner_program(self, marker: Path, held: Path, worker_pid_file: Path) -> str:
        """An owner that holds its replacement spawn open and waits to be killed."""

        return (
            "import os, sys, threading, time\n"
            "from vnext import process_supervisor as ps\n"
            f"ps._PARENT_DEATH_WATCHER = {self._one_flaky_watcher_then_the_real_one(marker)!r}\n"
            "real_spawn = ps._PosixGroup._spawn_watcher\n"
            "def held_spawn(self):\n"
            # Only a spawn asked for by a sentinel thread is a replacement, so
            # this holds the replacement open however many watchers the arming
            # itself started.
            '    if threading.current_thread().name.startswith("parent-death-watch"):\n'
            f"        open({str(held)!r}, 'w').write('held')\n"
            "        while True:\n"
            "            time.sleep(10)\n"
            "    return real_spawn(self)\n"
            "ps._PosixGroup._spawn_watcher = held_spawn\n"
            "owned = ps.OwnedProcess.start(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "    stdin=-3, stdout=-3, stderr=-3,\n"
            ")\n"
            f"open({str(worker_pid_file)!r}, 'w').write(str(owned.process.pid))\n"
            "time.sleep(300)\n"
        )

    def test_the_owner_dies_while_the_replacement_is_held_and_the_worker_still_goes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held = root / "replacement-held"
            worker_pid_file = root / "worker.pid"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            owner = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    self._owner_program(root / "first-watcher-ran", held, worker_pid_file),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            worker_pid = None
            try:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    if held.exists() and worker_pid_file.read_text().strip():
                        break
                    if owner.poll() is not None:
                        self.fail(f"the owner exited early: {owner.stderr.read()}")
                    time.sleep(0.01)
                self.assertTrue(held.exists(), "the replacement spawn was never reached")
                worker_pid = int(worker_pid_file.read_text())
                self.assertTrue(process_is_running(worker_pid))

                # The first watcher is dead and its replacement is held open:
                # exactly the moment the verifier killed the owner in.
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=10)
                deadline = time.monotonic() + 15.0
                while process_is_running(worker_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(
                    process_is_running(worker_pid),
                    "the worker outlived its owner because the group was unwatched "
                    "while a watcher was being replaced",
                )
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)
                if owner.stderr is not None and not owner.stderr.closed:
                    owner.stderr.close()
                if worker_pid is not None:
                    for number in (signal.SIGKILL,):
                        try:
                            os.killpg(worker_pid, number)
                        except OSError:
                            pass


@unittest.skipIf(os.name == "nt", "the watcher and the process group are POSIX objects")
class AWatcherNeverSignalsAGroupThatIsGoneTests(unittest.TestCase):
    """The owner can die between the empty-group check and the disarm write.

    close_handle asks whether the group still has members and only then writes
    the disarm byte. An owner killed in between leaves the watcher reading end of
    file while holding a process group id whose leader has already been reaped,
    so the number is free for the kernel to hand to somebody else. The watcher
    now asks the same question itself before it signals.
    """

    def _watcher_program(self, trace: Path) -> str:
        return (
            "import os\n"
            "_real_killpg = os.killpg\n"
            "def _traced_killpg(group, number):\n"
            f"    with open({str(trace)!r}, 'a') as _log:\n"
            '        _log.write(f"{group} {number}\\n")\n'
            "    return _real_killpg(group, number)\n"
            "os.killpg = _traced_killpg\n"
        ) + process_supervisor._PARENT_DEATH_WATCHER

    def test_the_watcher_signals_nothing_when_its_group_id_is_free_again(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "signals.log"
            # A session leader started and reaped: exactly what an orderly close
            # leaves behind, and the id it leaves is available for reuse.
            leader = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.exit(0)"],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            leader.wait(timeout=10)
            group_id = leader.pid

            read_end, write_end = os.pipe()
            ready_read, ready_write = os.pipe()
            watcher = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    self._watcher_program(trace),
                    str(read_end),
                    str(group_id),
                    str(ready_write),
                ],
                pass_fds=(read_end, ready_write),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            os.close(read_end)
            os.close(ready_write)
            try:
                # The owner dies without writing the disarm byte.
                os.close(write_end)
                watcher.wait(timeout=10)
                signalled = trace.read_text(encoding="utf-8") if trace.exists() else ""
                numbers = [
                    int(line.split()[1]) for line in signalled.splitlines() if line.strip()
                ]
                self.assertEqual(0, watcher.returncode)
                self.assertEqual(
                    [],
                    [number for number in numbers if number != 0],
                    f"the watcher signalled a process group id that was free again: {signalled}",
                )
            finally:
                os.close(ready_read)
                if watcher.poll() is None:
                    watcher.kill()
                    watcher.wait(timeout=5)


@unittest.skipIf(os.name == "nt", "this simulates Windows for a run that is not on it")
class TheWindowsJobStillKillsOnCloseTests(unittest.TestCase):
    """The Windows path keeps its own answer to the same problem.

    The POSIX watcher must not become the reason nobody notices the job losing
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, which is the flag that makes a crash of
    the owner close the handle and kill the tree.
    """

    def test_a_simulated_windows_job_sets_kill_on_job_close(self):
        kernel32 = MagicMock()
        kernel32.CreateJobObjectW.return_value = 0x1234
        recorded: list[tuple[int, int]] = []

        def set_information(handle, info_class, pointer, size):
            recorded.append((info_class, pointer._obj.BasicLimitInformation.LimitFlags))
            return 1

        kernel32.SetInformationJobObject.side_effect = set_information
        # ctypes carries WinDLL on Windows alone, so the stand-in has to be created.
        with patch.object(os, "name", "nt"), patch.object(
            ctypes, "WinDLL", create=True, return_value=kernel32
        ):
            module = importlib.reload(importlib.import_module("vnext.process_supervisor"))
            try:
                job = module._WindowsJob()
                self.assertEqual(
                    [(module.JobObjectExtendedLimitInformation, 0x00002000)],
                    recorded,
                )
                self.assertEqual(
                    0x00002000,
                    module.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                )
                self.assertEqual(0x1234, job.handle)
            finally:
                importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
