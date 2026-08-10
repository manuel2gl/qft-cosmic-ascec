"""Job registry — v04's machine-wide SQLite ``jobs.db``.

R6 reverts **D-034** (the JSON-registry rewrite) for the second of v04's two
persistence mechanisms. v04 records every running / finished ``ascec`` job in a
SQLite database at ``~/.local/state/ascec/jobs.db`` and uses it to:

* refuse to start a *second* live run of the same input file
  (``_atomic_claim_ascec_job``'s ``BEGIN IMMEDIATE`` duplicate guard);
* reap stale rows whose PID has died and SIGKILL their orphaned QM children
  (``_cleanup_stale_jobs`` → ``_kill_orphaned_job_processes``);
* drive the ``ascec status`` viewer (``_get_recent_jobs``).

This module is a **verbatim port** of ``ascec-v04.py`` lines 18352-18675
(**D-039**: faithful decomposition — the registry's concurrency semantics are
load-bearing for the workflow's duplicate-run protection, so nothing here is
redesigned). The interactive ``show_ascec_status`` viewer (v04 18676+) is the
CLI surface and is re-ported in R7.

The first persistence mechanism — the per-protocol ``protocol_*.pkl`` resume
cache — is :mod:`cosmic_ascec.workflow.protocol_cache`.
"""

from __future__ import annotations

import os
import sqlite3 as _sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "_ascec_state_dir",
    "_ascec_db_path",
    "_init_ascec_db",
    "_is_pid_alive",
    "_pdeathsig_preexec",
    "_register_ascec_job",
    "_update_ascec_job",
    "_set_job_holding",
    "_remove_progress_artifacts",
    "_adopt_ascec_job",
    "_atomic_claim_ascec_job",
    "_kill_orphaned_job_processes",
    "_cleanup_stale_jobs",
    "_get_recent_jobs",
    "_LIVE_STATUSES",
]

# Statuses that represent a job which is still "live" — it occupies a slot and
# is shown in the top (active) section of ``ascec status`` rather than history.
# ``holding`` is a queued run blocking behind another job's PID (see
# ``_set_job_holding`` and the wait loop in ``execute_workflow_stages``).
_LIVE_STATUSES = ("running", "holding")


def _ascec_state_dir() -> Path:
    """Return the per-user state directory, creating it if needed.

    ``~/.local/state`` is an XDG convention with no meaning on Windows, where
    the equivalent is ``%LOCALAPPDATA%``.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        d = (Path(base) if base else Path.home() / "AppData" / "Local") / "ascec"
    else:
        d = Path.home() / ".local" / "state" / "ascec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ascec_db_path() -> Path:
    return _ascec_state_dir() / "jobs.db"


def _init_ascec_db(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id            INTEGER PRIMARY KEY,
        pid           INTEGER,
        input_file    TEXT,
        working_dir   TEXT,
        cache_file    TEXT,
        log_file      TEXT,
        progress_file TEXT,
        status        TEXT,
        started_at    TEXT,
        updated_at    TEXT
    )""")
    # Idempotent migration: ``after_pid`` records the PID a queued ('holding')
    # job is waiting on, so ``ascec status`` can show what it blocks behind.
    # DBs created before the queue feature lack the column.
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "after_pid" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN after_pid INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.commit()


def _is_pid_alive(pid: int) -> bool:
    """Check whether a PID is still running.

    POSIX uses the classic ``kill(pid, 0)`` existence probe. Windows must NOT:
    CPython's ``os.kill`` only treats ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` as
    signals and maps *every other* value onto ``TerminateProcess(handle, sig)``.
    So ``os.kill(pid, 0)`` there does not ask whether the process is alive — it
    kills it, with exit code 0, which downstream looks like a clean successful
    run. That silently truncated the job being probed by the duplicate-run guard
    in ``_atomic_claim_ascec_job`` and by the ``after <PID>`` queue wait loop.
    On Windows we open a handle and poll it instead, which is read-only.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            WAIT_TIMEOUT = 0x00000102
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                # No handle: either the PID is gone or it belongs to another
                # user. Treat both as "not ours to wait on".
                return False
            try:
                # A live process never signals its handle, so it times out.
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pdeathsig_preexec() -> None:
    """preexec_fn for subprocess: ask the kernel to SIGKILL this child when the
    parent ascec dies. Linux-only; silently no-op elsewhere. Combined with the
    SIGTERM handler, this makes orphan QM/replica processes impossible — even
    SIGKILL of the main ascec immediately kills its children via the kernel.
    """
    try:
        if sys.platform != "linux":
            return
        import ctypes
        PR_SET_PDEATHSIG = 1
        SIGKILL = 9
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)
        # Defensive: if the parent already died between fork() and now, exit
        # immediately rather than running orphaned.
        if os.getppid() == 1:
            os._exit(0)
    except Exception:
        pass


# Value to pass as subprocess ``preexec_fn=``. Must be ``None`` on non-POSIX:
# Windows' subprocess rejects ANY non-None ``preexec_fn`` with
# ``ValueError: preexec_fn is not supported on Windows platforms`` at Popen
# construction — before the callable ever runs — so passing the function
# directly crashes every spawn on Windows even though it is a Linux-only no-op.
_PDEATHSIG_PREEXEC = _pdeathsig_preexec if sys.platform == "linux" else None


def _register_ascec_job(input_file: str, working_dir: str, cache_file: str,
                        log_file: str, progress_file: str) -> int:
    """Insert a new running-job entry; returns the new row ID (0 on failure)."""
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            "INSERT INTO jobs (pid,input_file,working_dir,cache_file,"
            "log_file,progress_file,status,started_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (os.getpid(), input_file, working_dir, cache_file,
             log_file, progress_file, 'running', now, now),
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return job_id
    except Exception:
        return 0


def _update_ascec_job(job_id: int, status: str) -> None:
    """Update a job's status in the registry."""
    if not job_id:
        return
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
            (status, time.strftime('%Y-%m-%d %H:%M:%S'), job_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _set_job_holding(job_id: int, after_pid: int) -> None:
    """Mark a job as queued ('holding') behind another running job's PID.

    The wait loop in ``execute_workflow_stages`` calls this on the placeholder
    row claimed by ``_atomic_claim_ascec_job`` before blocking, so the job
    appears under the live section of ``ascec status`` as HOLDING until
    ``after_pid`` exits, at which point the normal ``_adopt_ascec_job`` flow
    flips it back to 'running'.
    """
    if not job_id:
        return
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        conn.execute(
            "UPDATE jobs SET status='holding', after_pid=?, updated_at=? WHERE id=?",
            (int(after_pid), time.strftime('%Y-%m-%d %H:%M:%S'), job_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _remove_progress_artifacts(progress_file: str) -> None:
    """Best-effort removal of progress JSON and temp file."""
    if not progress_file:
        return
    for p in (progress_file, progress_file + ".tmp"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def _adopt_ascec_job(job_id: int, pid: int, log_file: str = "", progress_file: str = "",
                     cache_file: Optional[str] = None) -> bool:
    """Rebind an existing job row to a new PID (used for detach handoff and
    early-claim finalization). When cache_file is provided, it is also stored.
    """
    if not job_id or pid <= 0:
        return False
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        if cache_file is None:
            conn.execute(
                "UPDATE jobs SET pid=?, log_file=?, progress_file=?, status='running', updated_at=? WHERE id=?",
                (pid, log_file, progress_file, now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET pid=?, log_file=?, progress_file=?, cache_file=?, status='running', updated_at=? WHERE id=?",
                (pid, log_file, progress_file, cache_file, now, job_id),
            )
        conn.commit()
        changed = conn.total_changes > 0
        conn.close()
        return changed
    except Exception:
        return False


def _atomic_claim_ascec_job(input_file: str, working_dir: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Atomically check for a live duplicate and claim a job slot in one
    SQLite transaction. Returns (job_id, conflict_info).

    If conflict_info is non-None, NO slot was claimed — another live ascec is
    already registered for this input. Otherwise job_id is the new row id with
    placeholder cache/log/progress fields that the caller will fill in via
    _adopt_ascec_job once those paths are known.

    BEGIN IMMEDIATE serializes against any concurrent ascec running this same
    function, closing the time-of-check / time-of-use race that the previous
    two-step (read-then-insert-much-later) guard had.
    """
    abs_input = os.path.abspath(input_file)
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _cleanup_stale_jobs(conn)
            row = conn.execute(
                "SELECT id, pid, started_at FROM jobs "
                "WHERE status IN ('running','holding') AND input_file=? "
                "ORDER BY id ASC LIMIT 1",
                (abs_input,),
            ).fetchone()
            if row:
                existing_id, existing_pid, started_at = row
                if _is_pid_alive(existing_pid):
                    conn.rollback()
                    conn.close()
                    return 0, {
                        'id': existing_id,
                        'pid': existing_pid,
                        'started_at': started_at,
                    }
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                "INSERT INTO jobs (pid,input_file,working_dir,cache_file,"
                "log_file,progress_file,status,started_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (os.getpid(), abs_input, working_dir, "", "", "", 'running', now, now),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
            conn.close()
            return int(new_id), None
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            return 0, None
    except Exception:
        return 0, None


def _kill_orphaned_job_processes(working_dir: str, input_file: str) -> None:
    """Kill QM and child ascec processes still running in a dead job's working directory.

    Called when _cleanup_stale_jobs detects a job whose main PID has died, to prevent
    orphaned subprocesses from interfering with a new run of the same input file.
    Only operates on Linux (requires /proc, os.getpgid and os.killpg — none of
    which exist on Windows).
    """
    if sys.platform != "linux":
        return
    if not working_dir:
        return
    try:
        import signal as _sig
        try:
            wd_real = os.path.realpath(working_dir)
        except Exception:
            wd_real = working_dir

        input_base = os.path.splitext(os.path.basename(input_file or ""))[0].lower()
        qm_terms = (
            'orca', 'xtb', 'crest', 'g16', 'gaussian', 'qchem', 'nwchem',
            'psi4', 'cp2k', 'mopac', 'molpro', 'turbomole',
        )

        self_pid = os.getpid()
        to_kill: List[int] = []

        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == self_pid:
                continue
            try:
                cwd = os.path.realpath(os.readlink(f'/proc/{pid}/cwd'))
            except Exception:
                continue
            if not (cwd == wd_real or cwd.startswith(wd_real + os.sep)):
                continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as cf:
                    raw = cf.read().replace(b'\x00', b' ').strip().lower()
                cmd = raw.decode('utf-8', errors='ignore')
            except Exception:
                cmd = ""
            if any(t in cmd for t in qm_terms):
                to_kill.append(pid)
                continue
            if 'ascec' in cmd:
                to_kill.append(pid)
                continue
            if input_base and input_base in cmd:
                to_kill.append(pid)
                continue
            try:
                exe = os.path.basename(os.readlink(f'/proc/{pid}/exe')).lower()
            except Exception:
                exe = ""
            if any(t in exe for t in qm_terms):
                to_kill.append(pid)

        killed_pgids: set = set()
        for pid in set(to_kill):
            try:
                pgid = os.getpgid(pid)
                if pgid > 0 and pgid != os.getpgrp() and pgid not in killed_pgids:
                    killed_pgids.add(pgid)
                    try:
                        os.killpg(pgid, _sig.SIGKILL)
                    except OSError:
                        pass
            except OSError:
                pass
            try:
                os.kill(pid, _sig.SIGKILL)
            except OSError:
                pass
    except Exception:
        pass


def _cleanup_stale_jobs(conn) -> None:
    """Mark 'running' jobs whose PID no longer exists as 'crashed' and kill orphaned children."""
    rows = conn.execute(
        "SELECT id, pid, progress_file, working_dir, input_file FROM jobs "
        "WHERE status IN ('running','holding')"
    ).fetchall()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    for job_id, pid, progress_file, working_dir, input_file in rows:
        if not _is_pid_alive(pid):
            conn.execute(
                "UPDATE jobs SET status='crashed', updated_at=? WHERE id=?", (now, job_id)
            )
            _remove_progress_artifacts(progress_file or "")
            _kill_orphaned_job_processes(working_dir or "", input_file or "")
    conn.commit()


def _get_recent_jobs() -> list:
    """Return all running jobs plus recent history from the last 7 days."""
    try:
        conn = _sqlite3.connect(str(_ascec_db_path()), timeout=10.0)
        _init_ascec_db(conn)
        _cleanup_stale_jobs(conn)
        cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

        running_rows = conn.execute(
            "SELECT id,pid,input_file,working_dir,log_file,progress_file,"
            "status,started_at,updated_at,after_pid FROM jobs "
            "WHERE status IN ('running','holding') ORDER BY id ASC"
        ).fetchall()

        # Keep a bounded history list for readability.
        history_rows = conn.execute(
            "SELECT id,pid,input_file,working_dir,log_file,progress_file,"
            "status,started_at,updated_at,after_pid FROM jobs "
            "WHERE status NOT IN ('running','holding') AND updated_at >= ? "
            "ORDER BY id DESC LIMIT 10",
            (cutoff,)
        ).fetchall()

        rows = running_rows + history_rows
        conn.close()
        cols = ['id', 'pid', 'input_file', 'working_dir', 'log_file',
                'progress_file', 'status', 'started_at', 'updated_at', 'after_pid']
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []
