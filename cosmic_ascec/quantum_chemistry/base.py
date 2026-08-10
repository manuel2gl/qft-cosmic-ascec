"""Abstract contract every QM engine adapter must implement.

Every annealing cycle eventually boils down to one question — "what is the
energy of this geometry?" — and the answer comes from one of several QM
programs. This module defines the small surface each adapter must expose so
the rest of cosmic_ascec never has to special-case which program is in use.

The contract has two faces, used by two different parts of the codebase:

* **Annealing (Monte Carlo loop).** Only needs an energy. The MC loop calls
  the adapter's :meth:`energy_function` to get back a tiny
  ``Cluster -> float`` closure; under the hood that closure writes a QM input
  file, launches the program, parses the energy, and preserves the last input
  / output pair as ``anneal.<ext>`` so a failed run leaves something useful
  on disk. All the bookkeeping lives in :mod:`.runner`.

* **Clustering.** Needs the *rich* per-structure property surface — not just
  energy, but HOMO/LUMO, dipole, rotational constants, Gibbs free energy,
  imaginary-frequency counts, etc. That arrives via :meth:`parse_result`
  returning a :class:`QMResult` whose ``extras`` mapping is consumed by the
  feature pipeline in ``clustering/features/``.

**To add a new QM engine** (e.g. PySCF, NWChem, a remote service):
subclass :class:`QuantumChemistryAdapter`, set the class-level metadata
(``name``, ``input_ext``, ``output_ext``, ``energy_pattern``,
``termination_strings``), implement :meth:`write_input` and
:meth:`build_command`, and decorate the class with ``@register_adapter("foo")``
from :mod:`.registry`. That's it — the orchestrator and CLI pick the new
adapter up automatically.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Optional, Sequence

from cosmic_ascec.file_formats.asc_schema import QMSpec
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.monte_carlo import EnergyFn


# --------------------------------------------------------------------------- #
# Result container — the clustering parse surface (R5 territory, unchanged)    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QMResult:
    """The rich parse of one QM output file — consumed by the clustering stage.

    ``energy`` is the total electronic energy in Hartree, ``converged`` whether
    the program reported a normal termination, and ``extras`` is a free-form
    mapping holding every other property the output yielded (HOMO/LUMO, dipole,
    rotational constants, Gibbs free energy, imaginary-frequency count, …).

    The annealing path never builds a ``QMResult`` — it only needs the energy.
    ``QMResult`` is produced by :meth:`QuantumChemistryAdapter.parse_result` and
    fed to the clustering feature pipeline in ``clustering/features/``.
    """

    energy: float
    converged: bool
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze the mapping so callers holding a reference cannot mutate it.
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))


# --------------------------------------------------------------------------- #
# Energy parsing — shared between the orchestrator and parse_energy            #
# --------------------------------------------------------------------------- #


def parse_energy_from_text(
    content: str,
    energy_pattern: str,
    fallback_patterns: Sequence[str] = (),
) -> Optional[float]:
    """Return the first energy match in ``content``, or ``None`` when nothing matches.

    The primary ``energy_pattern`` is tried first with ``re.search`` — the
    **first** match wins, not the last. Only when it misses are the
    ``fallback_patterns`` walked in order (currently used by ORCA, which
    prints different energy lines depending on the method).
    """
    match = re.search(energy_pattern, content)
    if match is not None:
        return float(match.group(1))
    for pattern in fallback_patterns:
        match = re.search(pattern, content)
        if match is not None:
            return float(match.group(1))
    return None


# --------------------------------------------------------------------------- #
# Abstract adapter                                                            #
# --------------------------------------------------------------------------- #


class QuantumChemistryAdapter(ABC):
    """Adapter implemented once per QM package (xtb, gaussian, orca, ...).

    A concrete adapter declares the program metadata as class attributes
    (extensions, energy regex, termination markers) and implements two
    methods — :meth:`write_input` and :meth:`build_command` — that produce
    the program's input file and the argv to launch it. The orchestrator in
    :mod:`.runner` handles everything else generically: call counter, file
    naming, subprocess launch, exit-code handling, energy parsing, scratch
    cleanup, and preservation of the last-run files as ``anneal.<ext>``.

    Subclass this, decorate with ``@register_adapter("foo")``, and the
    annealing loop can use it. No other file needs to change.
    """

    # ------------------------------------------------------------------ #
    # Program metadata — set these as class attributes on the subclass    #
    # ------------------------------------------------------------------ #

    name: ClassVar[str]
    """Registry lookup key — e.g. ``"xtb"``, ``"orca"``, ``"gaussian"``."""

    input_ext: ClassVar[str]
    """Input-file extension, including the dot (e.g. ``".inp"``, ``".com"``)."""

    output_ext: ClassVar[str]
    """Output-file extension, including the dot (e.g. ``".out"``, ``".log"``)."""

    energy_pattern: ClassVar[str]
    """Regex matched against the output file — group 1 must capture the total
    electronic energy (Hartree)."""

    termination_strings: ClassVar[tuple[str, ...]]
    """Normal-termination markers. The run succeeded if *any* string in the
    tuple appears in the output (some programs print one of several phrases
    depending on the run type)."""

    energy_fallback_patterns: ClassVar[tuple[str, ...]] = ()
    """Extra energy regexes tried in order when :attr:`energy_pattern` misses.
    Empty for most packages; ORCA uses this because different methods print
    different energy lines."""

    uses_checkpoint: ClassVar[bool] = False
    """Whether the package writes a checkpoint file alongside its output.
    Gaussian does (``.chk``); the orchestrator preserves it as ``anneal.chk``
    so a failed restart still has the last wavefunction available."""

    uses_stdin_redirect: ClassVar[bool] = False
    """Whether the program reads its input from stdin rather than an argument.
    Gaussian does (``g09 < input > output``); xtb and ORCA take the input
    file as an argument."""

    nonzero_exit_is_failure: ClassVar[bool] = True
    """Whether a non-zero exit code means the run failed. ORCA routinely exits
    non-zero on otherwise-successful runs, so ORCA's adapter sets this to
    ``False`` and relies on termination strings alone."""

    # ------------------------------------------------------------------ #
    # Package-specific surface — implement these on your subclass          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def write_input(
        self,
        cluster: Cluster,
        spec: QMSpec,
        input_path: Path,
        *,
        call_id: int,
    ) -> None:
        """Write the QM input file at ``input_path``.

        ``input_path`` is named ``qm_input_<call_id>.<input_ext>``. ``call_id``
        is the per-run QM call counter — Gaussian needs it for the ``%chk``
        line so each call gets its own checkpoint file; xTB embeds it in the
        comment line of the .xyz so output files can be traced back to the
        triggering call.
        """

    @abstractmethod
    def build_command(self, input_name: str, spec: QMSpec) -> list[str]:
        """Return the argv list that runs the program on ``input_name``.

        ``input_name`` is the bare filename — the program runs with ``cwd``
        set to the run directory, so absolute paths are unnecessary. When
        :attr:`uses_stdin_redirect` is set the orchestrator feeds the input
        file on stdin, so ``input_name`` need not appear in the returned list.
        """

    def scratch_patterns(self, call_id: int) -> tuple[str, ...]:
        """Return extra glob patterns of run-dir scratch files to delete.

        Cleaned alongside the numbered ``qm_input``/``qm_output`` files in
        the orchestrator's ``finally`` block. ORCA scatters auxiliary files
        (``*.gbw``, ``*.densities``, ``*.cis``, ...); xTB drops a handful of
        small scratch files. Override if your engine litters the run dir.
        """
        return ()

    # ------------------------------------------------------------------ #
    # Clustering parse surface — used by the cosmic stage                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def parse_result(self, output_path: Path) -> QMResult:
        """Return the rich :class:`QMResult` for the clustering feature pipeline.

        Populate ``extras`` with whatever the output gives you (HOMO/LUMO,
        dipole, rotational constants, Gibbs free energy, imaginary frequency
        count, ...). The feature spec in ``clustering/features/`` decides
        which keys to consume.
        """

    @abstractmethod
    def detect_convergence(self, output_path: Path) -> bool:
        """Return whether the program reported a normal termination."""

    def parse_energy(self, output_path: Path) -> float:
        """Return the total electronic energy (Hartree) from ``output_path``.

        Concrete helper shared by every adapter — applies :attr:`energy_pattern`
        first, then :attr:`energy_fallback_patterns` in order. Raises
        :class:`~cosmic_ascec.exceptions.QMError` if nothing matches.
        """
        content = Path(output_path).read_text(errors="ignore", encoding='utf-8')
        energy = parse_energy_from_text(
            content, self.energy_pattern, self.energy_fallback_patterns
        )
        if energy is None:
            from cosmic_ascec.exceptions import QMError

            raise QMError(
                f"{self.name}: no energy ({self.energy_pattern!r}) in {output_path}"
            )
        return energy

    # ------------------------------------------------------------------ #
    # Bridge to the Monte Carlo loop                                      #
    # ------------------------------------------------------------------ #

    def energy_function(
        self,
        spec: QMSpec,
        run_dir: Path,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> EnergyFn:
        """Return a ``Cluster -> float`` closure used by the annealing engine.

        The closure owns the per-run QM call counter and the ``anneal.<ext>``
        preservation state. It drives one QM evaluation per Monte Carlo
        trial; on failure it raises :class:`~cosmic_ascec.exceptions.QMError`,
        which the annealing engine treats as a rejected-but-counted cycle
        (the QM call is still spent — the schedule advances).
        """
        from cosmic_ascec.quantum_chemistry.runner import make_energy_function

        return make_energy_function(self, spec, run_dir, logger=logger)


__all__ = ["QMResult", "QuantumChemistryAdapter", "parse_energy_from_text"]
