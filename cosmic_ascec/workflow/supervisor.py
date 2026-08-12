"""Foreground supervisor for detachable protocol runs.

A protocol run spawns its heavy work as child processes — annealing replicas
and QM jobs — and arms every one of them with ``PR_SET_PDEATHSIG`` so the
kernel kills the whole tree the instant the owning ascec dies (see
:func:`cosmic_ascec.workflow.job_registry._pdeathsig_preexec`).

That makes "detach by relaunching the command and exiting this process"
destructive: at Ctrl+D the foreground process exits, the kernel SIGKILLs every
in-flight replica/QM child, and the relaunched process finds no completed
artifacts, so it restarts the stage from zero. On screen the per-replica step
counter jumps *backwards* (``r1:32/80`` before detaching, ``r1:0/80`` climbing
again after) and hours of QM work can be lost.

The fix is structural: the process the shell waits on must never be the one
that owns the children. At startup — before any child is spawned — the run
forks a worker into its own session with a pty for a terminal, and the
original process stays in the foreground purely as a relay, copying
pty → terminal and keyboard → pty. Detaching then only exits the relay; the
worker and its whole process tree keep running untouched, still writing to the
same log and progress file that ``ascec status`` reads.

POSIX only. On Windows (no ``fork``, no pty, and ``PR_SET_PDEATHSIG`` is a
Linux no-op anyway) :func:`fork_detachable_worker` returns ``False`` and the
caller keeps its previous behaviour.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, NoReturn, Optional

# Keys the relay treats as "detach": Ctrl+D (EOT) and Ctrl+Z (SUSP). Ctrl+Z is
# included deliberately — the relay runs in raw mode, so the shell never sees
# the suspend, and a stopped relay could not drain the pty, which would
# eventually block the worker on a full terminal buffer. "Put this run in the
# background" is what the keystroke means, and detaching is how we honour it.
_DETACH_KEYS = (b"\x04", b"\x1a")

_READ_CHUNK = 65536


def is_supported() -> bool:
    """True when this platform can host a forked, pty-backed worker."""
    return sys.platform != "win32" and hasattr(os, "fork")


def _copy_winsize(src_fd: int, dst_fd: int) -> None:
    """Mirror the terminal window size from ``src_fd`` onto ``dst_fd``."""
    try:
        import fcntl
        import struct
        import termios

        packed = fcntl.ioctl(src_fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        fcntl.ioctl(dst_fd, termios.TIOCSWINSZ, packed)
    except Exception:
        pass


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``, tolerating partial writes."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            return
        view = view[written:]


def _signal_worker(worker_pid: int, sig: int) -> None:
    """Signal the worker's process group (it is a session leader after setsid).

    The group matters: a bare ``kill`` would reach ascec but not the replica or
    QM child it is currently waiting on, whereas today's foreground Ctrl+C
    reaches the whole foreground group.
    """
    try:
        os.killpg(worker_pid, sig)
    except OSError:
        try:
            os.kill(worker_pid, sig)
        except OSError:
            pass


def fork_detachable_worker(
    on_supervisor: Optional[Callable[[int], None]] = None,
    on_detach: Optional[Callable[[int], None]] = None,
    detach_signal: Optional[int] = None,
) -> bool:
    """Fork the run into a detachable worker process.

    Returns ``True`` in the worker, which should carry on with the run exactly
    as before; everything it spawns from here on is untouched by a later
    detach. Returns ``False`` when supervision is not possible (no fork, no
    pty, or stdin/stdout is not a terminal) — the caller then keeps its
    previous, unsupervised behaviour.

    In the supervisor this function never returns: it calls ``on_supervisor``
    (a hook for releasing resources the worker now owns, e.g. the log handle),
    relays the worker's terminal I/O until the worker exits or the user
    detaches, and exits the process — with the worker's exit status when the
    run finished, or 0 after calling ``on_detach`` when the user detached.

    Args:
        on_supervisor: called in the supervisor with the worker PID, right
            after the fork and before any relaying.
        on_detach: called in the supervisor with the worker PID when the user
            detaches, before the supervisor exits.
        detach_signal: signal delivered to the worker (and only the worker —
            never the group, whose members are QM jobs with lethal defaults)
            just before the supervisor exits, telling it the terminal is gone
            and it should write to its log from now on. Defaults to SIGUSR1.
            The worker must already handle it: its default action is fatal, so
            the handler has to be installed *before* this call.
    """
    if not is_supported():
        return False

    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError, OSError):
        return False

    try:
        import pty
    except ImportError:
        return False

    # Flush before forking: buffered bytes still in this process would
    # otherwise be written out twice, once by each side of the fork.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.flush()
        except Exception:
            pass

    try:
        master_fd, slave_fd = pty.openpty()
    except OSError:
        return False

    try:
        _copy_winsize(sys.stdout.fileno(), slave_fd)
    except (AttributeError, ValueError, OSError):
        pass

    try:
        pid = os.fork()
    except OSError:
        for _fd in (master_fd, slave_fd):
            try:
                os.close(_fd)
            except OSError:
                pass
        return False

    if pid == 0:
        # ── Worker ──────────────────────────────────────────────────────────
        # Own session, so the supervisor's exit (or the terminal closing) can
        # neither signal nor orphan-kill this side. The pty slave stands in for
        # the terminal, which keeps isatty() true for the progress panel; it is
        # deliberately *not* made a controlling terminal, so no SIGHUP arrives
        # when the supervisor closes the master.
        try:
            os.setsid()
        except OSError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            for _target in (0, 1, 2):
                os.dup2(slave_fd, _target)
            if slave_fd > 2:
                os.close(slave_fd)
        except OSError:
            pass
        return True

    # ── Supervisor ──────────────────────────────────────────────────────────
    try:
        os.close(slave_fd)
    except OSError:
        pass

    if callable(on_supervisor):
        try:
            on_supervisor(pid)
        except Exception:
            pass

    if detach_signal is None:
        import signal as _signal_mod

        detach_signal = getattr(_signal_mod, "SIGUSR1", 0)

    _relay(master_fd, pid, on_detach, detach_signal)


def _relay(
    master_fd: int,
    worker_pid: int,
    on_detach: Optional[Callable[[int], None]],
    detach_signal: int = 0,
) -> NoReturn:
    """Copy pty ↔ terminal until the worker exits or the user detaches."""
    import select
    import signal
    import termios
    import tty

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()

    # Raw mode so Ctrl+D / Ctrl+Z / Ctrl+C arrive as bytes we can act on rather
    # than being interpreted against *this* process, which owns nothing.
    old_attrs = None
    try:
        old_attrs = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)
    except Exception:
        old_attrs = None

    def _forward_winch(_signum, _frame):
        _copy_winsize(stdout_fd, master_fd)

    try:
        signal.signal(signal.SIGWINCH, _forward_winch)
    except (AttributeError, ValueError, OSError):
        pass

    detached = False
    try:
        while True:
            try:
                readable, _, _ = select.select([master_fd, stdin_fd], [], [], 0.5)
            except InterruptedError:
                continue
            except OSError:
                break

            if master_fd in readable:
                try:
                    data = os.read(master_fd, _READ_CHUNK)
                except OSError:
                    data = b""
                if not data:
                    break  # pty closed: the worker has exited
                try:
                    _write_all(stdout_fd, data)
                except OSError:
                    # Terminal went away (SSH dropped, window closed). Leave the
                    # worker running and stop relaying.
                    detached = True
                    break

            if stdin_fd in readable:
                try:
                    keys = os.read(stdin_fd, 1024)
                except OSError:
                    keys = b""
                if not keys or any(k in keys for k in _DETACH_KEYS):
                    detached = True
                    break
                if b"\x03" in keys:
                    # Ctrl+C keeps its old meaning: abort the run, whole group.
                    _signal_worker(worker_pid, signal.SIGINT)
                    keys = keys.replace(b"\x03", b"")
                    if not keys:
                        continue
                try:
                    _write_all(master_fd, keys)
                except OSError:
                    pass
    finally:
        if old_attrs is not None:
            try:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    if detached:
        # Tell the worker the terminal is gone *before* closing the master, so
        # it can switch to log-only output instead of taking an EIO on its next
        # write. Worker only: the process group also holds QM children.
        if detach_signal:
            try:
                os.kill(worker_pid, detach_signal)
            except OSError:
                pass
        if callable(on_detach):
            try:
                on_detach(worker_pid)
            except Exception:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        os._exit(0)

    try:
        os.close(master_fd)
    except OSError:
        pass

    status = 0
    try:
        _, status = os.waitpid(worker_pid, 0)
    except OSError:
        status = 0

    try:
        code = os.waitstatus_to_exitcode(status)
    except (AttributeError, ValueError):
        code = 0
    if code < 0:  # killed by a signal
        code = 128 + (-code)
    os._exit(code & 0xFF)
