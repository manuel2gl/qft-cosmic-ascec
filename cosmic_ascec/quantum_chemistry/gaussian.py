"""Gaussian adapter — ``g09`` / ``g16`` single-point energies.

* **Input:** ``.gjf`` (Link-0 ``%chk=qm_chk_<id>.chk``, optional ``%nproc``,
  then ``# method basis``, title, charge/mult, Cartesian block, blank line).
* **Output:** ``.log``.
* **Energy parse:** ``SCF Done: E(...) = ... A.U.`` — group 1 is the energy.
* **Termination marker:** ``Normal termination``.
* **Invocation:** Gaussian reads its input from stdin, so the adapter sets
  ``uses_stdin_redirect = True`` and the orchestrator pipes the .gjf in and
  the .log out — equivalent to ``g09 < input > output`` but without a shell.
* **Checkpoint:** ``uses_checkpoint = True`` so the orchestrator copies the
  last ``.chk`` to ``anneal.chk`` for restart inspection.

The ``parse_result`` property surface uses cclib to extract HOMO/LUMO, dipole,
rotational constants, etc. for the clustering pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cosmic_ascec.elements import Z_TO_SYMBOL
from cosmic_ascec.exceptions import QMError
from cosmic_ascec.file_formats.asc_schema import QMSpec
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.quantum_chemistry.base import QMResult, QuantumChemistryAdapter
from cosmic_ascec.quantum_chemistry.registry import register_adapter

# cclib is an optional dependency: the adapter degrades to text parsing if it
# is missing, so a CI box without cclib can still exercise the energy path.
try:  # pragma: no cover - import guard
    from cclib.io import ccread as _ccread
except ImportError:  # pragma: no cover - exercised only without cclib
    _ccread = None


# v04 ``qm_program_details[1]['energy_regex']`` (ascec-v04.py line 2969) — used
# verbatim for both the annealing-path scan and the clustering parse.
_ENERGY_PATTERN = r"SCF Done:\s+E\(.+\)\s*=\s*([-\d.]+)\s+A\.U\."
_SCF_DONE_RE = re.compile(_ENERGY_PATTERN)
_TERMINATION_STRING = "Normal termination"
# cosmic-v01 ``.log`` Gibbs scan (cosmic-v01.py line 1324).
_GIBBS_RE = re.compile(
    r"Sum of electronic and thermal Free Energies=\s*([-+]?\d+\.\d+)"
)
# cclib reports rotational constants in GHz; cosmic-v01 line 1436 normalises.
_GHZ_TO_INVERSE_CM = 29.9792458


@register_adapter("gaussian", "g09", "g16", "g03")
class GaussianAdapter(QuantumChemistryAdapter):
    """Adapter for the Gaussian electronic-structure package."""

    name = "gaussian"
    input_ext = ".gjf"
    output_ext = ".log"
    energy_pattern = _ENERGY_PATTERN
    termination_strings = (_TERMINATION_STRING,)
    uses_checkpoint = True
    uses_stdin_redirect = True

    # ------------------------------------------------------------------ #
    # Annealing surface — v04 calculate_energy Gaussian branch            #
    # ------------------------------------------------------------------ #

    def write_input(
        self,
        cluster: Cluster,
        spec: QMSpec,
        input_path: Path,
        *,
        call_id: int,
    ) -> None:
        """Write the ``.gjf`` input file, byte-for-byte v04 (lines 3118-3137)."""
        parts = [f"%chk=qm_chk_{call_id}.chk\n"]
        # v04 line 3122: ``if state.qm_nproc:`` — emitted whenever nprocs is set.
        if spec.qm_nprocs:
            parts.append(f"%nproc={spec.qm_nprocs}\n")
        if spec.basis_set:
            parts.append(f"# {spec.method} {spec.basis_set}\n\n")
        else:
            parts.append(f"# {spec.method}\n\n")
        parts.append("ASCEC QM Calculation\n\n")
        parts.append(f"{spec.charge} {spec.multiplicity}\n")
        for z, (x, y, zc) in zip(cluster.atomic_numbers, cluster.coords):
            symbol = Z_TO_SYMBOL.get(z, "X")
            parts.append(f"{symbol} {x:.6f} {y:.6f} {zc:.6f}\n")
        # v04 writes one blank line after the geometry, then (skipping the
        # absent additional-keywords block) one more — the molecule terminator.
        parts.append("\n")
        parts.append("\n")
        input_path.write_text("".join(parts), encoding='utf-8')

    def build_command(self, input_name: str, spec: QMSpec) -> list[str]:  # noqa: ARG002
        """Build the Gaussian argv — input on stdin, report on stdout (v04 3195)."""
        return [spec.alias or "g09"]

    # ------------------------------------------------------------------ #
    # Clustering parse surface (R5 territory) — unchanged by R4           #
    # ------------------------------------------------------------------ #

    def parse_result(self, output_path: Path) -> QMResult:
        """Full property block: text for the mandatory fields, cclib for extras."""
        path = Path(output_path)
        if not path.exists():
            raise QMError(f"gaussian: output file {output_path} was not produced")
        content = path.read_text(errors="ignore", encoding='utf-8')
        converged = _TERMINATION_STRING in content

        energy_matches = _SCF_DONE_RE.findall(content)
        if not energy_matches:
            raise QMError(
                f"gaussian: no 'SCF Done' energy found in {output_path} "
                f"(converged={converged})"
            )
        energy = float(energy_matches[-1])

        extras: Dict[str, Any] = {"final_electronic_energy": energy}
        if (gibbs := _GIBBS_RE.findall(content)):
            extras["gibbs_free_energy"] = float(gibbs[-1])

        _enrich_extras_with_cclib(output_path, extras)
        return QMResult(energy=energy, converged=converged, extras=extras)

    def detect_convergence(self, output_path: Path) -> bool:
        """Return True iff the log carries Gaussian's normal-termination banner."""
        path = Path(output_path)
        if not path.exists():
            return False
        return _TERMINATION_STRING in path.read_text(errors="ignore", encoding='utf-8')


# --------------------------------------------------------------------------- #
# cclib enrichment — shared by the Gaussian and ORCA cclib paths (R5)          #
# --------------------------------------------------------------------------- #


def _enrich_extras_with_cclib(output_path: Path, extras: Dict[str, Any]) -> None:
    """Best-effort: fill ``extras`` from a cclib parse of ``output_path``.

    Mirrors cosmic-v01 ``extract_properties_with_cclib`` (cosmic-v01.py
    1156-1533). Every step is wrapped: a log cclib cannot parse simply leaves
    ``extras`` as the text scan produced it. This is part of the clustering
    parse surface — R5 audits it.
    """
    if _ccread is None:
        return
    try:
        raw = _ccread(str(output_path))
    except Exception:  # noqa: BLE001 - cclib raises a broad family of errors
        return
    if raw is None:
        return
    data = raw
    if hasattr(raw, "data") and isinstance(raw.data, list) and raw.data:
        data = raw.data[-1]

    metadata = getattr(data, "metadata", {}) or {}
    methods = metadata.get("methods") or []
    if methods:
        extras.setdefault("method", methods[0])
    if metadata.get("basis_set"):
        extras.setdefault("basis_set", metadata["basis_set"])

    charge = getattr(data, "charge", None)
    if charge is not None:
        extras.setdefault("charge", int(charge))
    mult = getattr(data, "mult", None)
    if mult is not None:
        extras.setdefault("multiplicity", int(mult))

    try:
        homos = getattr(data, "homos", None)
        moenergies = getattr(data, "moenergies", None)
        if homos is not None and moenergies is not None and len(homos) and len(moenergies):
            row = moenergies[0]
            idx = int(homos[0])
            if 0 <= idx and idx + 1 < len(row):
                homo = float(row[idx])
                lumo = float(row[idx + 1])
                extras.setdefault("homo_energy", homo)
                extras.setdefault("lumo_energy", lumo)
                extras.setdefault("homo_lumo_gap", lumo - homo)
    except (TypeError, IndexError, ValueError):
        pass

    try:
        rotconsts = getattr(data, "rotconsts", None)
        if rotconsts is not None and len(rotconsts):
            last = np.asarray(rotconsts[-1], dtype=float)
            if last.ndim == 1 and last.size == 3:
                extras.setdefault(
                    "rotational_constants", last / _GHZ_TO_INVERSE_CM
                )
    except (TypeError, IndexError, ValueError):
        pass

    try:
        vibfreqs = getattr(data, "vibfreqs", None)
        if vibfreqs is not None and len(vibfreqs):
            freqs = [float(f) for f in vibfreqs]
            extras.setdefault(
                "num_imaginary_freqs", sum(1 for f in freqs if f < 0)
            )
            reals = [f for f in freqs if f > 0]
            if reals:
                extras.setdefault("first_vib_freq", min(reals))
                extras.setdefault("last_vib_freq", max(reals))
    except (TypeError, ValueError):
        pass


__all__ = ["GaussianAdapter"]
