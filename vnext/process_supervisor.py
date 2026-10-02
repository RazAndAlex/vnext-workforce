from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


class ProcessSupervisionError(RuntimeError):
    pass


CREATE_SUSPENDED = 0x00000004


@dataclass(frozen=True)
class ProcessCleanup:
    outcome: str
    residual_count: int
    root_exit_code: int | None
    errors: tuple[str, ...] = ()


class OwnedProcess:
    """A subprocess whose process tree is owned by this controller.

    What ownership reaches differs by platform, and neither platform reaches
    everything:

    * On POSIX the unit is the worker's process group. The worker starts a
      session of its own, so every child and grandchild it forks inherits that
      group and a single killpg stops all of them. A descendant that calls
      setsid() or setpgid() of its own accord leaves the group and outlives the
      kill; a runtime verification saw exactly that with a double fork. Nothing
      short of a cgroup or a Darwin process-lifecycle API can hold it, and this
      class does not try.
    * On Windows the unit is a job object carrying
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. A descendant cannot leave the job
      unless it is created with a breakaway flag, and closing the handle,
      including by crashing, kills what is inside.
    """

    def __init__(self, process: subprocess.Popen[str], tree: _WindowsJob | _PosixGroup):
        self.process = process
        self._tree = tree
        self._cleanup: ProcessCleanup | None = None

    @classmethod
    def start(cls, command: list[str], **kwargs: Any) -> "OwnedProcess":
        if os.name == "nt":
            tree: _WindowsJob | _PosixGroup = _WindowsJob()
            creationflags = (
                int(kwargs.pop("creationflags", 0))
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | CREATE_SUSPENDED
            )
            try:
                process = subprocess.Popen(command, creationflags=creationflags, **kwargs)
            except BaseException:
                tree.close_handle()
                raise
            try:
                tree.assign(process)
                tree.resume(process)
            except BaseException:
                process.kill()
                process.wait(timeout=5)
                tree.close_handle()
                raise
            return cls(process, tree)

        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        group = _PosixGroup(process.pid)
        # The watchers are armed after Popen returns, because the process group
        # id is the child's pid and does not exist before then.  An owner killed
        # inside that window - measured at roughly 1 ms on an Apple-silicon Mac
        # and demonstrated by a probe that held the window open on purpose -
        # still leaks the group.  Closing it needs the Windows trick of starting
        # the child suspended, which POSIX has no plain equivalent for, so it
        # stays a known residual.
        try:
            group.arm_parent_death_watch()
        except BaseException:
            # Fail closed: a worker nobody watches is the fault the watcher
            # exists to stop, so the caller gets an error rather than a worker
            # that will outlive it.
            group.terminate()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise
        return cls(process, group)

    def close(self, *, grace_seconds: float = 1.0) -> ProcessCleanup:
        if self._cleanup is not None:
            return self._cleanup

        errors: list[str] = []
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError as exc:
                errors.append(f"stdin close failed: {exc}")
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=max(0.0, grace_seconds))
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None or self._tree.active_processes() > 0:
            try:
                self._tree.terminate()
            except OSError as exc:
                errors.append(f"tree termination failed: {exc}")
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append("root process did not exit after tree termination")
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"root kill failed: {exc}")

        deadline = time.monotonic() + 2.0
        residual_count = self._tree.active_processes()
        while residual_count and time.monotonic() < deadline:
            time.sleep(0.05)
            residual_count = self._tree.active_processes()
        self._tree.close_handle()
        outcome = "clean" if residual_count == 0 and not errors else "residual"
        self._cleanup = ProcessCleanup(
            outcome=outcome,
            residual_count=residual_count,
            root_exit_code=self.process.poll(),
            errors=tuple(errors),
        )
        return self._cleanup


# Windows kills an owned tree by itself: the job carries
# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so a crash that closes the job handle
# takes the tree with it.  POSIX has nothing equivalent that survives a SIGKILL
# of the owner.  Linux has PR_SET_PDEATHSIG, Darwin has no counterpart, and
# either way the signal would reach the worker alone and leave its own children
# behind.  So every owned group gets watcher processes, each holding the read end
# of a pipe whose write end lives in this process and nowhere else.  While this
# process is alive the read blocks.  When it dies, however it dies, the kernel
# closes the write ends, the reads return end of file, and the watchers kill the
# group.  Two of them are armed, for the reason given above _WATCHERS_PER_GROUP.  The read end has to stay open for the whole life of the watcher; the
# same watch written with kqueue was silently discarded in stow#4 by closing the
# descriptor right after registering it.
# End of file on its own cannot tell a crash from a shutdown, and the difference
# matters: after an orderly close the group leader has been reaped, so the number
# the watcher holds names a group that no longer exists and the kernel is free to
# give it to somebody else.  So the owner writes one byte before it closes the
# pipe.  A byte means "stand down"; end of file with no byte in front of it is
# the death of the owner.  The byte is delivered ahead of the end of file the
# same close produces, so the watcher cannot miss it.
#
# A Popen that returns has forked and nothing more, so the watcher can still die
# on its way to that first read: a probe that made the watcher exit 77 at once
# saw start() return happily and the worker outlive a SIGKILL of its owner.  So
# the watcher writes one byte of its own the moment it is in position, and
# arming is not finished until the owner has read it.  The pipe carrying that
# byte belongs to the watcher alone, so the byte failing to arrive - because the
# watcher exited, or because it never reached the line - closes the pipe and the
# owner reads end of file instead of waiting out the bound.
_WATCHER_DISARM = b"q"
_WATCHER_READY = b"r"
_WATCHER_READY_TIMEOUT = 5.0

_PARENT_DEATH_WATCHER = f"""
import os, signal, sys, time

link = int(sys.argv[1])
group = int(sys.argv[2])
ready = int(sys.argv[3])

os.write(ready, {_WATCHER_READY!r})
os.close(ready)

while True:
    try:
        message = os.read(link, 4096)
    except InterruptedError:
        continue
    except OSError:
        break
    if not message:
        break
    if {_WATCHER_DISARM!r} in message:
        sys.exit(0)

# The group id is the worker's pid, so the number is free for reuse from the
# moment the owner reaps the worker.  An owner killed between the empty-group
# check in close_handle and the disarm write leaves this watcher holding such a
# number, so ask whether anything still answers to it before signalling.  A
# group that is wholly gone answers ESRCH here, measured twice on Apple-silicon
# macOS 15 with a reaped session leader signalled from an unrelated session:
# killpg(pgid, 0), kill(leader, 0) and killpg(pgid, SIGTERM) all gave ESRCH.
# What stays open is the gap between this question and the answer below, which
# no signal can close from outside the kernel.
try:
    os.killpg(group, 0)
except ProcessLookupError:
    sys.exit(0)
except OSError:
    pass

# Darwin refuses a signal to an emptied group with EPERM where Linux reports no
# such process, and both answers mean the same thing here: there is nothing left
# to stop.
for number in (signal.SIGTERM, signal.SIGKILL):
    try:
        os.killpg(group, number)
    except OSError:
        break
    if number is signal.SIGTERM:
        time.sleep(0.4)
"""


# One watcher can die, and replacing it takes a fork and a ready byte: 25 ms on
# an Apple-silicon Mac.  A verifier killed an owner 2.3 ms into that gap and the
# worker lived, because the dead watcher's link was already released and the
# replacement did not exist yet.  Nothing spawned after the death can close that
# gap, so the cover has to be standing before it opens: every group is armed with
# two watchers, and the replacement for one of them is armed while the other is
# still holding its link.  The remaining window needs both watchers to die, and
# the second death ends in a terminated group.
#
# Measured on this Mac before it was built: a watcher is 14.0 MiB of ps RSS of
# which 2.7 MiB is memory the machine did not already have spent on the
# interpreter, and two watchers armed together reach ready in 21 ms against 25 ms
# for one on its own.  So the second watcher costs a worker about 2.7 MiB and no
# measurable start time.
_WATCHERS_PER_GROUP = 2


@dataclass
class _WatcherPost:
    """One watcher and the owner's end of the pipe it is reading.

    Each watcher gets a pipe of its own rather than a shared one, so the death
    of a watcher releases nothing the others are using and a stand-down byte
    reaches every watcher that is still on duty.
    """

    process: subprocess.Popen[bytes]
    link: int | None
    sentinel: threading.Thread | None = None


class _PosixGroup:
    def __init__(self, pid: int):
        self.pid = pid
        # The ready byte proves a watcher was alive once and says nothing about
        # the rest of the worker's life.  A probe injected a watcher that wrote
        # the byte and exited 78 before its first read of the owner link:
        # start() returned happily and the worker ran with nobody watching it.
        # So the owner keeps a thread on each watcher for as long as the group
        # lives.  These fields are those threads' state and are read and written
        # under _guard alone.
        self._guard = threading.Lock()
        self._posts: list[_WatcherPost] = []
        self._replacement_spent = False
        self._released = False

    def arm_parent_death_watch(self) -> None:
        """Hand each of the group's watchers a pipe that reports this death.

        The first one is what start() fails closed over.  How many there are
        is _WATCHERS_PER_GROUP, and setting it to 1 gives back the single
        watcher this group had before the standby existed.
        """

        spares: list[_WatcherPost] = []
        # Every watcher is armed at once, so the ones after the first cost the
        # worker almost nothing of its start time.
        helpers = [
            threading.Thread(
                target=self._arm_a_spare_post,
                args=(spares,),
                name=f"parent-death-arm-{self.pid}",
                daemon=True,
            )
            for _ in range(_WATCHERS_PER_GROUP - 1)
        ]
        for helper in helpers:
            helper.start()
        armed: _WatcherPost | None = None
        try:
            armed = _WatcherPost(*self._spawn_watcher())
        finally:
            # Joined whichever way the first watcher went, so a spare can never
            # be left running with nobody holding a reference to it.
            for helper in helpers:
                helper.join()
            if armed is None:
                for post in spares:
                    self._retire_an_unused_post(post)
        posts = [armed, *spares]
        with self._guard:
            self._posts = posts
            for post in posts:
                post.sentinel = self._watch_the_watcher(post)

    def _arm_a_spare_post(self, into: list[_WatcherPost]) -> None:
        """Arm the standby watcher, and let the group live with one if it cannot.

        The first watcher is what start() fails closed over.  This one is worth
        having and not worth failing a worker for: the fork that refuses it is
        the machine at its process limit, and a group with a single watcher is
        watched exactly as well as it was before the standby existed.  What it
        loses is the cover over the replacement window.
        """

        try:
            watcher, link = self._spawn_watcher()
        except Exception as refusal:
            # KeyboardInterrupt and SystemExit are left to travel, and the line
            # goes to stderr because the MCP protocol owns stdout.
            print(
                f"worker pid {self.pid} runs with one watcher: the standby could not be"
                f" armed, so the worker loses cover while that watcher is being"
                f" replaced: {refusal}",
                file=sys.stderr,
            )
            return
        # The thread that reads this list joins this one first, which is the
        # barrier that makes the append visible to it.
        into.append(_WatcherPost(watcher, link))

    def _retire_an_unused_post(self, post: _WatcherPost) -> None:
        """Take down a watcher that never went on duty.

        It is blocked on a read that only the death of this process ends, so it
        is killed rather than waited out, and its link is dropped after the kill
        so no watcher reads the closed pipe as that death.
        """

        post.process.kill()
        try:
            post.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._drop_the_link(post)

    def _watch_the_watcher(self, post: _WatcherPost) -> threading.Thread:
        """Start the one thread that answers the death of this watcher."""

        sentinel = threading.Thread(
            target=self._replace_the_watcher_or_stop_the_group,
            args=(post,),
            name=f"parent-death-watch-{self.pid}",
            daemon=True,
        )
        sentinel.start()
        return sentinel

    def _replace_the_watcher_or_stop_the_group(self, post: _WatcherPost) -> None:
        """Answer a watcher that died while the group still had work in it.

        One replacement is tried, and exactly one: a watcher that cannot stay
        alive is not made likelier to by a third attempt, and an owner that
        respawns forever would spend the machine on it.  The group is terminated
        when this death empties the roster - the same fail-closed ending as a
        watcher that never armed at all, because a worker nobody watches is the
        fault the watcher exists to stop.  While another watcher is still
        holding a link the group is watched, so a spent replacement budget is
        not by itself a reason to stop the worker.
        """

        try:
            post.process.wait()
        except BaseException:
            return
        with self._guard:
            if self._released or post not in self._posts:
                # The owner closed this group down, or this post was already
                # taken off the roster.  Either way this death is expected.
                return
            self._posts.remove(post)
            # The pipe's only reader has gone, so nothing can read this end
            # again and holding it would leak a descriptor per watcher death.
            self._drop_the_link(post)
            if self.active_processes() == 0:
                # Nothing is left to watch over.
                return
            if not self._replacement_spent:
                self._replacement_spent = True
                try:
                    replacement = _WatcherPost(*self._spawn_watcher())
                except BaseException:
                    pass
                else:
                    self._posts.append(replacement)
                    replacement.sentinel = self._watch_the_watcher(replacement)
            if self._posts:
                return
            self.terminate()

    def _spawn_watcher(self) -> tuple[subprocess.Popen[bytes], int]:
        """Start one armed watcher and hand back it and the owner's link end."""

        read_end, write_end = os.pipe()
        ready_read, ready_write = os.pipe()
        try:
            # The watcher gets two descriptors and nothing else: the read end
            # of the death pipe and the write end of the ready pipe.  DEVNULL on
            # all three standard streams keeps it clear of the worker's JSON-RPC
            # pipes and stops it holding this process's own stdout open.  A new
            # session keeps it out of both process groups, so neither a killpg of
            # the worker's group nor one aimed at this process's group can take it
            # down early.
            watcher = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _PARENT_DEATH_WATCHER,
                    str(read_end),
                    str(self.pid),
                    str(ready_write),
                ],
                pass_fds=(read_end, ready_write),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError):
            # The fork that started the worker can succeed and this one hit the
            # per-user process limit, so this is reachable with the worker
            # already running.  A runtime probe injected EAGAIN here and the
            # worker went on to survive a SIGKILL of its owner.  start() turns
            # the raise into a terminated group.
            for descriptor in (read_end, write_end, ready_read, ready_write):
                os.close(descriptor)
            raise
        os.close(read_end)
        os.close(ready_write)
        try:
            self._await_watcher_ready(ready_read, watcher)
        except BaseException:
            # Same fail-closed reasoning as the Popen error above: a watcher
            # that never armed is a worker nobody watches.  The watcher is
            # stopped here so start() has only the worker left to clean up.
            os.close(write_end)
            watcher.kill()
            try:
                watcher.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            os.close(ready_read)
        return watcher, write_end

    @staticmethod
    def _await_watcher_ready(
        ready_read: int, watcher: subprocess.Popen[bytes]
    ) -> None:
        """Block until the watcher says it is armed, or fail the arming."""

        deadline = time.monotonic() + _WATCHER_READY_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessSupervisionError(
                    "the parent-death watcher did not report itself armed within "
                    f"{_WATCHER_READY_TIMEOUT:g}s"
                )
            try:
                readable, _, _ = select.select([ready_read], [], [], remaining)
            except InterruptedError:
                continue
            if not readable:
                continue
            try:
                answer = os.read(ready_read, len(_WATCHER_READY))
            except InterruptedError:
                continue
            except OSError as exc:
                raise ProcessSupervisionError(
                    f"the parent-death watcher's ready pipe could not be read: {exc}"
                ) from exc
            if answer == _WATCHER_READY:
                return
            # End of file: the write end lived in the watcher alone, so the
            # kernel closed it when the watcher died.
            raise ProcessSupervisionError(
                "the parent-death watcher exited before it was armed "
                f"(exit {watcher.poll()})"
            )

    # Darwin decides whether the caller may signal the group before it decides
    # whether the group still has members, so a group whose processes have all
    # exited answers EPERM where Linux answers ESRCH.  On Darwin both mean the
    # same thing to a kill: there is nothing left to stop.  Neither is evidence
    # that something survived, because active_processes below is what counts
    # the survivors and it is asked again after every attempt.  On Linux EPERM
    # is a real refusal, so it is left to reach the receipt as "tree
    # termination failed".  Read at call time so a test can name the platform.
    @staticmethod
    def _already_gone() -> tuple[type[OSError], ...]:
        if sys.platform == "darwin":
            return (ProcessLookupError, PermissionError)
        return (ProcessLookupError,)

    def terminate(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except self._already_gone():
            return
        deadline = time.monotonic() + 1.0
        while self.active_processes() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.active_processes():
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except self._already_gone():
                pass

    def active_processes(self) -> int:
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return 0
        except PermissionError:
            return 1
        return 1

    def _nothing_a_watcher_can_stop(self) -> bool:
        """Read the group the way terminate() does, for the disarm decision.

        A watcher runs as the same user as its owner.  Whatever answer made
        terminate() stop is an answer the watcher would get too, so an armed
        watcher could only ever reach the id once it names somebody else's
        group.  active_processes keeps counting a refusal as a survivor, so the
        receipt never reads a refusal as a clean shutdown.
        """

        try:
            os.killpg(self.pid, 0)
        except self._already_gone():
            return True
        except PermissionError:
            return False
        return False

    def close_handle(self) -> None:
        """Release every watcher.  close() has already emptied the group."""

        with self._guard:
            self._released = True
            posts, self._posts = self._posts, []
            # Asked once for the whole roster: every watcher is holding the same
            # group id, so the answer cannot differ between them.
            nothing_left = self._nothing_a_watcher_can_stop()
            for post in posts:
                self._release_the_link(post, disarm=nothing_left)
        for post in posts:
            self._stand_the_post_down(post, disarm=False)

    def _release_the_link(self, post: _WatcherPost, *, disarm: bool) -> None:
        """Say stand down when there is nothing left, then drop the pipe."""

        if post.link is None:
            return
        if disarm:
            # Nothing is left to stop, so the watcher is told to stand down
            # rather than left to signal a group id that is now free for reuse.
            # A group that did survive keeps its watchers armed: the id still
            # names the group they were given, and killing it is the job.
            try:
                os.write(post.link, _WATCHER_DISARM)
            except OSError:
                pass
        self._drop_the_link(post)

    @staticmethod
    def _drop_the_link(post: _WatcherPost) -> None:
        if post.link is None:
            return
        try:
            os.close(post.link)
        except OSError:
            pass
        post.link = None

    def _stand_the_post_down(self, post: _WatcherPost, *, disarm: bool) -> None:
        """Wait out one watcher, and kill it if waiting does not end it."""

        if disarm:
            self._release_the_link(post, disarm=True)
        if post.sentinel is not None:
            # The sentinel thread is the one waiting on this process, so asking
            # for the exit status here as well would sit out the whole bound
            # against a wait already in flight.  Joining it reaps the watcher
            # through that thread instead.
            post.sentinel.join(timeout=2)
            if post.sentinel.is_alive():
                post.process.kill()
                post.sentinel.join(timeout=2)
            return
        try:
            post.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            post.process.kill()
            try:
                post.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectBasicAccountingInformation = 1
    JobObjectExtendedLimitInformation = 9

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    )
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.Thread32First.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    _kernel32.Thread32First.restype = wintypes.BOOL
    _kernel32.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    _kernel32.Thread32Next.restype = wintypes.BOOL
    _kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenThread.restype = wintypes.HANDLE
    _kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
    _kernel32.ResumeThread.restype = wintypes.DWORD

    TH32CS_SNAPTHREAD = 0x00000004
    THREAD_SUSPEND_RESUME = 0x0002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]


class _WindowsJob:
    def __init__(self):
        if os.name != "nt":
            raise ProcessSupervisionError("Windows process jobs are unavailable")
        self.handle = _kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            self.handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close_handle()
            raise error

    def assign(self, process: subprocess.Popen[str]) -> None:
        process_handle = wintypes.HANDLE(int(process._handle))
        if not _kernel32.AssignProcessToJobObject(self.handle, process_handle):
            raise ProcessSupervisionError(
                f"could not assign app-server to an owned Windows job: {ctypes.WinError(ctypes.get_last_error())}"
            )

    def resume(self, process: subprocess.Popen[str]) -> None:
        snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        resumed = False
        try:
            entry = THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            found = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if int(entry.th32OwnerProcessID) == process.pid:
                    thread_handle = _kernel32.OpenThread(
                        THREAD_SUSPEND_RESUME,
                        False,
                        entry.th32ThreadID,
                    )
                    if thread_handle:
                        try:
                            previous_count = _kernel32.ResumeThread(thread_handle)
                            if previous_count != 0xFFFFFFFF:
                                resumed = True
                        finally:
                            _kernel32.CloseHandle(thread_handle)
                found = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            _kernel32.CloseHandle(snapshot)
        if not resumed:
            raise ProcessSupervisionError(
                "could not resume the app-server after assigning its Windows job"
            )

    def terminate(self) -> None:
        if self.handle and not _kernel32.TerminateJobObject(self.handle, 1):
            error_code = ctypes.get_last_error()
            if error_code:
                raise ctypes.WinError(error_code)

    def active_processes(self) -> int:
        if not self.handle:
            return 0
        accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        if not _kernel32.QueryInformationJobObject(
            self.handle,
            JobObjectBasicAccountingInformation,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            return 1
        return int(accounting.ActiveProcesses)

    def close_handle(self) -> None:
        if self.handle:
            _kernel32.CloseHandle(self.handle)
            self.handle = None
