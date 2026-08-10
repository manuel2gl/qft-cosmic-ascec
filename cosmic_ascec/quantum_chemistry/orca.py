"""ORCA adapter — works against ORCA 4.x/5.x and (via text fallback) 6.x.

* **Input:** ``.inp`` — ``! method basis`` keyword line, optional ``%pal``
  block when nprocs > 1, then ``* xyz charge mult`` and the Cartesian block.
* **Output:** ``.out``.
* **Energy parse:** ``FINAL SINGLE POINT ENERGY`` (colon optional, covers
  ORCA 4/5/6), with four fallback patterns for methods that print differently.
* **Termination markers:** ``ORCA TERMINATED NORMALLY`` (plus two
  alternatives that appear under certain run types).
* **Exit code is ignored:** ORCA routinely returns non-zero *on success*, so
  ``nonzero_exit_is_failure = False`` and the adapter relies on termination
  strings + energy regex to decide success.

For ORCA 6.1+ with the ``opi`` package installed, the registry's
:func:`resolve_orca_adapter` routes to the richer ``orca_opi`` adapter; this
adapter's :meth:`parse_result` *also* self-delegates to OPI when it detects
a 6.1+ output, so consumers that hardcode ``"orca"`` still get good
properties on newer outputs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from cosmic_ascec.elements import Z_TO_SYMBOL
from cosmic_ascec.exceptions import QMError
from cosmic_ascec.file_formats.asc_schema import QMSpec
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.quantum_chemistry.base import QMResult, QuantumChemistryAdapter
from cosmic_ascec.quantum_chemistry.gaussian import _enrich_extras_with_cclib
from cosmic_ascec.quantum_chemistry.registry import (
    detect_orca_version,
    opi_importable,
    register_adapter,
)

# v04 ``qm_program_details[2]['energy_regex']`` (ascec-v04.py line 2977).
_FINAL_SP_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s*:?\s*([-+]?\d+\.\d+)"
)
_TERMINATION_STRING = "ORCA TERMINATED NORMALLY"
# v04 ORCA fallback energy patterns (ascec-v04.py lines 3321-3325).
_ENERGY_FALLBACKS = (
    r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)",
    r"Total Energy\s*:\s*([-+]?\d+\.\d+)\s*Eh",
    r"E\(SCF\)\s*=\s*([-+]?\d+\.\d+)\s*Eh",
    r"TOTAL SCF ENERGY\s*=\s*([-+]?\d+\.\d+)\s*Eh?",
)
# cosmic-v01 ORCA ``.out`` text fallbacks (cosmic-v01.py lines 1334, 1398, 1447).
_DIPOLE_RE = re.compile(
    r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
_HOMO_LUMO_GAP_RE = re.compile(
    r"HOMO-LUMO gap\s+(-?\d+\.?\d*)\s+eV"
)
_ROT_CONST_RE = re.compile(
    r"Rotational constants in cm-1:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
_GIBBS_RE = re.compile(r"Final Gibbs free energy\s+\.*\s*([-+]?\d+\.\d+)")

# An ``.asc`` alias that names a *different* QM package must not be used to
# locate ORCA (v04 ``detect_orca_executable``, lines 7772).
_NON_ORCA_ALIASES = {"xtb", "g16", "g09", "g03", "gaussian"}


@register_adapter("orca")
class ORCAAdapter(QuantumChemistryAdapter):
    """Adapter for the ORCA electronic-structure package (4.x / 5.x cclib path)."""

    name = "orca"
    input_ext = ".inp"
    output_ext = ".out"
    energy_pattern = r"FINAL SINGLE POINT ENERGY\s*:?\s*([-+]?\d+\.\d+)"
    termination_strings = (
        _TERMINATION_STRING,
        "****ORCA TERMINATED NORMALLY****",
        "OPTIMIZATION RUN DONE",
    )
    energy_fallback_patterns = _ENERGY_FALLBACKS
    # ORCA 5/6 exit non-zero even on a successful run (v04 line 3268).
    nonzero_exit_is_failure = False

    # ------------------------------------------------------------------ #
    # Annealing surface — v04 calculate_energy ORCA branch                #
    # ------------------------------------------------------------------ #

    def write_input(
        self,
        cluster: Cluster,
        spec: QMSpec,
        input_path: Path,
        *,
        call_id: int,  # noqa: ARG002 - ORCA's input carries no call id
    ) -> None:
        """Write the ORCA ``.inp`` file (v04 lines 3138-3162)."""
        keyword_line = f"! {spec.method}"
        if spec.basis_set:
            keyword_line += f" {spec.basis_set}"
        lines = [keyword_line]
        # v04 line 3155: ``if state.qm_nproc:`` — emitted whenever nprocs is set,
        # including a single process.
        if spec.qm_nprocs:
            lines.append(f"%pal nprocs {spec.qm_nprocs} end")
        lines.append(f"* xyz {spec.charge} {spec.multiplicity}")
        for z, (x, y, zc) in zip(cluster.atomic_numbers, cluster.coords):
            symbol = Z_TO_SYMBOL.get(z, "X")
            lines.append(f"{symbol} {x:.6f} {y:.6f} {zc:.6f}")
        lines.append("*")
        input_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    def build_command(self, input_name: str, spec: QMSpec) -> list[str]:
        """Build the ORCA argv — full exe path for ``%pal`` runs (v04 3182-3198)."""
        return [_resolve_orca_exe(spec.alias), input_name]

    def scratch_patterns(self, call_id: int) -> tuple[str, ...]:
        """ORCA auxiliary files v04 deletes after each call (lines 3346-3356)."""
        return (f"qm_input_{call_id}*", f".qm_input_{call_id}*")

    # ------------------------------------------------------------------ #
    # Clustering parse surface (R5 territory) — unchanged by R4           #
    # ------------------------------------------------------------------ #

    def parse_result(self, output_path: Path) -> QMResult:
        """Full property block, version-aware.

        ORCA >= 6.1 with ``opi`` installed delegates to the OPI adapter;
        otherwise it parses with cclib plus the text fallbacks.
        """
        version = detect_orca_version(output_path=output_path)
        if version is not None and version >= (6, 1) and opi_importable():
            from cosmic_ascec.quantum_chemistry.orca_opi import ORCAOPIAdapter

            return ORCAOPIAdapter().parse_result(output_path)
        return self._parse_result_text(output_path)

    def detect_convergence(self, output_path: Path) -> bool:
        """Return True iff the output carries ORCA's normal-termination banner."""
        path = Path(output_path)
        if not path.exists():
            return False
        return _TERMINATION_STRING in path.read_text(errors="ignore", encoding='utf-8')

    def _parse_result_text(self, output_path: Path) -> QMResult:
        """Parse ``energy`` / ``converged`` from text, enrich ``extras`` via cclib."""
        path = Path(output_path)
        if not path.exists():
            raise QMError(f"orca: output file {output_path} was not produced")
        content = path.read_text(errors="ignore", encoding='utf-8')
        converged = _TERMINATION_STRING in content

        energy_matches = _FINAL_SP_ENERGY_RE.findall(content)
        if not energy_matches:
            raise QMError(
                f"orca: no FINAL SINGLE POINT ENERGY found in {output_path} "
                f"(converged={converged})"
            )
        energy = float(energy_matches[-1])

        extras: Dict[str, Any] = {"final_electronic_energy": energy}
        _scan_orca_text_extras(content, extras)
        _enrich_extras_with_cclib(output_path, extras)
        return QMResult(energy=energy, converged=converged, extras=extras)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _resolve_orca_exe(alias: Optional[str]) -> str:
    """Resolve the ORCA executable, preferring an absolute path.

    ORCA needs its full path for parallel (``%pal``) runs (v04 lines 3184-3193).
    An alias that names a different QM package is ignored (v04
    ``detect_orca_executable``, lines 7772-7777).
    """
    # Either separator: on Windows an alias naming a real path uses backslashes,
    # and testing only for '/' would misclassify it as a bare command name.
    if alias and ("/" in alias or "\\" in alias):
        return alias
    candidates = []
    if alias and alias.lower() not in _NON_ORCA_ALIASES:
        candidates.append(alias)
    candidates.append("orca")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return alias or "orca"


def _scan_orca_text_extras(content: str, extras: Dict[str, Any]) -> None:
    """Fill ``extras`` from ORCA ``.out`` text patterns (cosmic-v01 fallbacks)."""
    if (gibbs := _GIBBS_RE.findall(content)):
        extras["gibbs_free_energy"] = float(gibbs[-1])
    if (gaps := _HOMO_LUMO_GAP_RE.findall(content)):
        extras["homo_lumo_gap"] = float(gaps[-1])
    if (dipoles := _DIPOLE_RE.findall(content)):
        dx, dy, dz = (float(v) for v in dipoles[-1])
        extras["dipole_moment"] = float(np.linalg.norm([dx, dy, dz]))
    if (rot := _ROT_CONST_RE.findall(content)):
        extras["rotational_constants"] = np.array(
            [float(v) for v in rot[-1]], dtype=float
        )


__all__ = ["ORCAAdapter"]
