"""QM call orchestrator — drives one adapter through one energy evaluation.

The annealing loop never talks to a QM program directly. It calls
:func:`make_energy_function` once at the start of a run; that returns a tiny
``Cluster -> float`` closure that the MC loop drives. Each invocation of the
closure performs exactly the steps below — which are identical regardless of
the engine — and the adapter is the only thing that changes between xtb,
gaussian, orca, and any new engine you add.

What one QM call does:

1. Bump the per-run QM call counter; the new value is the ``call_id``.
2. Ask the adapter to write ``qm_input_<call_id>.<ext>`` in the run dir.
3. Launch the program with ``cwd`` set to the run dir, streaming stdout +
   stderr into ``qm_output_<call_id>.<ext>``. Engines that read input from
   stdin (Gaussian) get it piped; engines that take it as an argument
   (xtb, ORCA) get the path on the command line.
4. Decide success: any termination marker must appear *and* the energy regex
   must match (ORCA falls back through extra patterns). ORCA's exit code is
   ignored — it routinely exits non-zero on otherwise-successful runs.
5. Preserve the last input/output as ``anneal.<ext>`` so a crashed annealing
   run leaves something investigable on disk; delete the numbered scratch.
6. Return ``(energy, status)`` — ``status`` is ``1`` on success, ``0`` on any
   failure. The closure converts a ``0`` into a :class:`QMError`, which the
   annealing engine treats as a rejected-but-counted trial.

A QM call is deterministic; this module never touches the run RNG.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from cosmic_ascec.exceptions import QMError
from cosmic_ascec.file_formats.asc_schema import QMSpec
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.monte_carlo import EnergyFn
from cosmic_ascec.quantum_chemistry.base import (
    QuantumChemistryAdapter,
    parse_energy_from_text,
)

# Cap BLAS/OpenMP thread fan-out for QM subprocesses. xtb's `--parallel 1` only
# controls xtb's own thread pool; the underlying BLAS still uses all cores
# unless these env vars are set. Mono sets them at runtime in
# optimize_qm_execution_environment (ascec-v04.py:3383-3384); the modular
# refactor missed that, so each xtb opt was spawning ~8 BLAS threads and the
# whole opt stage ran 100× slower than expected. setdefault preserves any
# explicit override the user may have set in their shell.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# --------------------------------------------------------------------------- #
# Subprocess hygiene                                                          #
# --------------------------------------------------------------------------- #


def _pdeathsig_preexec() -> None:  # pragma: no cover - exercised only out-of-process
    """Ask the kernel to SIGKILL this QM child when the parent ascec dies.

    Linux-only (uses ``PR_SET_PDEATHSIG``); silently a no-op elsewhere. Keeps
    QM children from outliving a killed parent ascec — without this, a
    Ctrl-C on the parent could leave orphaned xtb/orca processes burning CPU
    forever, especially on a shared cluster node. The post-fork getppid()
    check exits immediately if ascec is itself already gone.
    """
    try:
        if sys.platform != "linux":
            return
        import ctypes

        PR_SET_PDEATHSIG = 1
        SIGKILL = 9
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)
        if os.getppid() == 1:
            os._exit(0)
    except Exception:  # noqa: BLE001 - best-effort hardening, never fatal
        pass


# Value to pass as subprocess ``preexec_fn=``. Must be ``None`` on non-POSIX:
# Windows' subprocess rejects ANY non-None ``preexec_fn`` with
# ``ValueError: preexec_fn is not supported on Windows platforms`` at Popen
# construction — before the callable ever runs — so passing the function
# directly crashes every spawn on Windows even though it is a Linux-only no-op.
_PDEATHSIG_PREEXEC = _pdeathsig_preexec if sys.platform == "linux" else None


# --------------------------------------------------------------------------- #
# Preserved-file state — v04's per-run state.qm_attempt_debug bookkeeping      #
# --------------------------------------------------------------------------- #


class _PreserveState:
    """Per-run state for the QM call counter and ``anneal.<ext>`` preservation.

    A run can issue thousands of QM calls; this small object holds the two
    pieces of state that need to live for the whole run — the call counter
    (so each input file gets a fresh numeric suffix) and a single flag that
    says "the first successful call has already been frozen into
    ``anneal.<ext>``", which prevents later failures from clobbering it.
    """

    __slots__ = ("call_count", "first_success_preserved")

    def __init__(self) -> None:
        self.call_count = 0
        self.first_success_preserved = False


def _preserve_last_qm_files(
    adapter: QuantumChemistryAdapter,
    run_dir: Path,
    call_id: int,
    logger: Optional[logging.Logger],
) -> None:
    """Copy this call's numbered QM files to the ``anneal.<ext>`` keep-files.

    ``qm_input_<id>``/``qm_output_<id>`` (and Gaussian's ``qm_chk_<id>.chk``)
    are copied to ``anneal.inp``/``anneal.out``/``anneal.chk``. Copy errors
    are warnings only — losing the keep-file copy is annoying but never a
    reason to abort an annealing run.
    """
    src_input = run_dir / f"qm_input_{call_id}{adapter.input_ext}"
    src_output = run_dir / f"qm_output_{call_id}{adapter.output_ext}"
    dst_input = run_dir / f"anneal{adapter.input_ext}"
    dst_output = run_dir / f"anneal{adapter.output_ext}"
    try:
        if src_input.exists():
            shutil.copyfile(src_input, dst_input)
        if src_output.exists():
            shutil.copyfile(src_output, dst_output)
        if adapter.uses_checkpoint:
            src_chk = run_dir / f"qm_chk_{call_id}.chk"
            if src_chk.exists():
                shutil.copyfile(src_chk, run_dir / "anneal.chk")
    except OSError as exc:  # v04: "Could not preserve QM files"
        if logger is not None:
            logger.warning("could not preserve anneal.* files: %s", exc)


def _preserve_last_qm_files_debug(
    adapter: QuantumChemistryAdapter,
    run_dir: Path,
    call_id: int,
    status: int,
    state: _PreserveState,
    logger: Optional[logging.Logger],
) -> None:
    """Preserve only until the first success — then freeze ``anneal.<ext>``.

    Every failed call during the initial-QM retry phase overwrites the keep
    files (so a debugger has the most recent failure to inspect). The first
    successful call freezes them — after that, the keep files document the
    geometry/output that actually opened the annealing run, untouched by any
    later QM failure.
    """
    if state.first_success_preserved:
        return
    _preserve_last_qm_files(adapter, run_dir, call_id, logger)
    if status == 1:
        state.first_success_preserved = True


# --------------------------------------------------------------------------- #
# Single QM call                                                              #
# --------------------------------------------------------------------------- #


def calculate_energy(
    adapter: QuantumChemistryAdapter,
    cluster: Cluster,
    spec: QMSpec,
    run_dir: Path,
    *,
    call_id: int,
    preserve_state: _PreserveState,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, int]:
    """Run one external QM call and return ``(energy, status)``.

    ``status`` is ``1`` on a normal termination *with* a parseable energy,
    ``0`` on any failure (input-write error, missing output, no termination
    marker, no energy, or any exception during the subprocess). On a ``0``
    return the energy is ``0.0`` — callers must check the status first.
    """
    run_dir = Path(run_dir)
    input_name = f"qm_input_{call_id}{adapter.input_ext}"
    output_name = f"qm_output_{call_id}{adapter.output_ext}"
    input_path = run_dir / input_name
    output_path = run_dir / output_name
    chk_path = run_dir / f"qm_chk_{call_id}.chk" if adapter.uses_checkpoint else None

    # v04 keeps the input write in a try with no finally: an input-write or
    # command-build failure returns (0.0, 0) immediately, before any cleanup.
    try:
        adapter.write_input(cluster, spec, input_path, call_id=call_id)
        command = adapter.build_command(input_name, spec)
    except (OSError, ValueError) as exc:
        if logger is not None:
            logger.error("QM input/command build failed for call %d: %s", call_id, exc)
        return 0.0, 0

    energy = 0.0
    status = 0
    temp_files = [input_path, output_path]
    if chk_path is not None:
        temp_files.append(chk_path)

    try:
        # ---- run the program (v04 3236-3265) ----------------------------- #
        with open(output_path, "w") as out_fh:
            if adapter.uses_stdin_redirect:
                # Gaussian: v04 ran ``g09 < input > output`` under a shell;
                # v04 redirects the handles directly — same effect, no shell.
                with open(input_path, "r") as in_fh:
                    process = subprocess.run(
                        command,
                        cwd=str(run_dir),
                        stdin=in_fh,
                        stdout=out_fh,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        preexec_fn=_PDEATHSIG_PREEXEC,
                    )
            else:
                process = subprocess.run(
                    command,
                    cwd=str(run_dir),
                    stdout=out_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    preexec_fn=_PDEATHSIG_PREEXEC,
                )

        # ---- decide whether to trust the output (v04 3267-3336) ---------- #
        should_check_output = True
        if process.returncode != 0:
            if adapter.nonzero_exit_is_failure:
                if logger is not None:
                    logger.debug(
                        "%s exited %d for call %d",
                        adapter.name, process.returncode, call_id,
                    )
                status = 0
                should_check_output = False
            # else (ORCA): exit code ignored, parse the output anyway.

        if should_check_output:
            if not output_path.exists():
                status = 0
            else:
                content = output_path.read_text(errors="ignore")
                terminated = any(t in content for t in adapter.termination_strings)
                if not terminated:
                    if logger is not None:
                        logger.debug(
                            "%s did not terminate normally for call %d",
                            adapter.name, call_id,
                        )
                    status = 0
                else:
                    parsed = parse_energy_from_text(
                        content,
                        adapter.energy_pattern,
                        adapter.energy_fallback_patterns,
                    )
                    if parsed is not None:
                        energy = parsed
                        status = 1
                    else:
                        if logger is not None:
                            logger.debug(
                                "no energy found in %s output for call %d",
                                adapter.name, call_id,
                            )
                        status = 0
    except Exception as exc:  # noqa: BLE001 - v04: any exception => status 0
        if logger is not None:
            logger.error("QM calculation failed for call %d: %s", call_id, exc)
        status = 0
    finally:
        # v04 preserves the last files BEFORE cleaning the numbered scratch.
        _preserve_last_qm_files_debug(
            adapter, run_dir, call_id, status, preserve_state, logger
        )
        for pattern in adapter.scratch_patterns(call_id):
            for hit in glob.glob(str(run_dir / pattern)):
                hit_path = Path(hit)
                if hit_path.is_file() and hit_path not in temp_files:
                    temp_files.append(hit_path)
        for fpath in temp_files:
            if fpath.exists():
                try:
                    fpath.unlink()
                except OSError as exc:
                    if logger is not None:
                        logger.warning("could not clean %s: %s", fpath.name, exc)

    return energy, status


# --------------------------------------------------------------------------- #
# Bridge to the Monte Carlo loop                                              #
# --------------------------------------------------------------------------- #


def make_energy_function(
    adapter: QuantumChemistryAdapter,
    spec: QMSpec,
    run_dir: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> EnergyFn:
    """Return the ``Cluster -> float`` closure the annealing engine drives.

    The closure owns the per-run QM call counter (v04 ``state.qm_call_count``)
    and the ``anneal.<ext>`` preservation flag, calls :func:`calculate_energy`
    once per evaluation, and converts v04's ``status == 0`` into a
    :class:`~cosmic_ascec.exceptions.QMError` — the engine reads that as v04's
    ``jo_status == 0`` rejected-but-counted cycle (ascec-v04.py 20681-20683).
    """
    run_dir = Path(run_dir)
    state = _PreserveState()

    def _energy(cluster: Cluster) -> float:
        state.call_count += 1
        call_id = state.call_count
        energy, status = calculate_energy(
            adapter,
            cluster,
            spec,
            run_dir,
            call_id=call_id,
            preserve_state=state,
            logger=logger,
        )
        if status == 0:
            raise QMError(
                f"{adapter.name}: QM call {call_id} failed "
                f"(no normal termination or no parseable energy)"
            )
        return energy

    return _energy


__all__ = ["calculate_energy", "make_energy_function"]
