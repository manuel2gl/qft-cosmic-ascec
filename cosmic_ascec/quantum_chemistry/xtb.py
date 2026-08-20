"""xTB adapter — Grimme's tight-binding code (default backend).

xTB is fast (~milliseconds per single-point) and accurate enough for the
annealing search, which is why it is the default annealing backend. The
input file is a plain XYZ; all other knobs (method, charge, multiplicity,
parallelism) are passed on the command line — there is no ``%`` block or
keyword section to render.

* **Input:** plain ``.xyz`` file. Comment line tags the call with
  ``"ASCEC annealing SP, call <call_id>"`` so a stray file in the run dir
  can be traced to a specific QM call.
* **Command:** ``xtb <input.xyz> --gfn 2 [--chrg N] [--uhf N] [--parallel N]``.
  The GFN flavour is taken from the ``.asc`` Line 10 method string
  (``GFN-FF``, ``GFN0-xTB``, ``GFN1-xTB``, or default ``GFN2-xTB``).
* **Energy parse:** the last ``TOTAL ENERGY ... Eh`` line.
* **Termination marker:** ``"normal termination of xtb"``.
* **Scratch:** ``xtbrestart`` / ``xtbtopo`` are deleted between calls so a
  long annealing run does not slowly accumulate junk.

The clustering pipeline also pulls HOMO/LUMO, dipole, and rotational
constants out of the xTB output — those regexes live alongside the
adapter below.

* **Constraints:** :func:`write_xcontrol_constraints` renders an xcontrol
  ``$constrain`` block that holds polar X–H bonds, for the charged-metal case
  where GFN2-xTB otherwise deprotonates the solvent. xtb has no ``--constrain``
  command-line flag; constraints reach it only via ``--input <file>``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from cosmic_ascec.elements import Z_TO_SYMBOL
from cosmic_ascec.exceptions import QMError
from cosmic_ascec.geometry.fragments import polar_xh_bonds
from cosmic_ascec.file_formats.asc_schema import QMSpec
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.quantum_chemistry.base import QMResult, QuantumChemistryAdapter
from cosmic_ascec.quantum_chemistry.registry import register_adapter


DEFAULT_CONSTRAIN_FORCE_CONSTANT: float = 0.5
"""Force constant (au) for the generated ``$constrain`` block.

xtb's own default is 0.05, which is far too weak to hold a bond: with a bond
biased toward a 1.300 Å reference, 0.05 lets it relax to 1.050 Å — the
constraint is effectively ignored. Measured against xtb 6.7.0, the same bond
lands at 1.216 Å (FC 0.25), 1.257 (0.5), 1.278 (1.0) and 1.296 (5.0). 0.5 holds
the bond while still letting the rest of the structure relax; raise it toward
1.0 if a proton still migrates.
"""

XCONTROL_EXTENSION: str = ".xcontrol"
"""Suffix for the per-structure detailed-input file, written next to the deck."""

_TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh")
_TERMINATION_STRING = "normal termination of xtb"

# Property-parser regexes — used only by parse_result() for clustering.
# The annealing path needs only the TOTAL ENERGY line above.
_HOMO_LUMO_GAP_RE = re.compile(
    r"HOMO-?LUMO\s+(?:GAP|gap)\s*[:=]?\s*([-+]?\d+\.\d+)", re.IGNORECASE
)
_HOMO_RE = re.compile(r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+\(HOMO\)")
_LUMO_RE = re.compile(r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+\(LUMO\)")
_DIPOLE_FULL_RE = re.compile(
    r"^\s*full:\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$",
    re.MULTILINE,
)
_ROT_CONSTANTS_RE = re.compile(
    r"rotational constants/cm.*?:\s*([-+]?\d+\.\d+E[+-]?\d+)\s+"
    r"([-+]?\d+\.\d+E[+-]?\d+)\s+([-+]?\d+\.\d+E[+-]?\d+)",
    re.IGNORECASE,
)
_CHARGE_RE = re.compile(r"net\s+charge\s+([-]?\d+)")
_UNPAIRED_RE = re.compile(r"unpaired\s+electrons\s+(\d+)")
_GFN_FF_RE = re.compile(r"G\s+F\s+N\s*-?\s*F\s+F")
_GFN0_BANNER_RE = re.compile(r"G\s+F\s+N\s+0")
_GFN1_BANNER_RE = re.compile(r"G\s+F\s+N\s+1\s")


@register_adapter("xtb")
class XTBAdapter(QuantumChemistryAdapter):
    """Adapter for Grimme's xTB tight-binding code."""

    name = "xtb"
    input_ext = ".xyz"
    output_ext = ".out"
    energy_pattern = r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh"
    termination_strings = (_TERMINATION_STRING,)

    # ------------------------------------------------------------------ #
    # Annealing surface — what the MC loop drives                         #
    # ------------------------------------------------------------------ #

    def write_input(
        self,
        cluster: Cluster,
        spec: QMSpec,  # noqa: ARG002 - xTB takes method/charge/spin on the CLI
        input_path: Path,
        *,
        call_id: int,
    ) -> None:
        """Write a plain XYZ input — atom count, comment, lines of ``Sym x y z``."""
        coords = cluster.coords
        natom = coords.shape[0]
        lines = [f"{natom}", f"ASCEC annealing SP, call {call_id}"]
        for z, (x, y, zc) in zip(cluster.atomic_numbers, coords):
            symbol = Z_TO_SYMBOL.get(z, "X")
            lines.append(f"{symbol} {x:.6f} {y:.6f} {zc:.6f}")
        input_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    def build_command(self, input_name: str, spec: QMSpec) -> list[str]:
        """Build ``xtb <input> --gfn N [--chrg C] [--uhf 2S] [--parallel N]``."""
        exe = spec.alias or "xtb"
        command = [exe, input_name, *_gfn_flags(spec.method)]
        # Charge/spin flags are only emitted when non-default, so the typical
        # neutral singlet command stays short.
        if spec.charge != 0:
            command += ["--chrg", str(spec.charge)]
        if spec.multiplicity > 1:
            # xTB takes the number of unpaired electrons (2S), not the
            # multiplicity (2S+1) — translate.
            command += ["--uhf", str(spec.multiplicity - 1)]
        if spec.qm_nprocs and spec.qm_nprocs > 0:
            command += ["--parallel", str(spec.qm_nprocs)]
        return command

    def scratch_patterns(self, call_id: int) -> tuple[str, ...]:  # noqa: ARG002
        """xtb drops these two files in cwd; clean them between calls."""
        return ("xtbrestart", "xtbtopo")

    # ------------------------------------------------------------------ #
    # Clustering parse surface — extra properties pulled from the output  #
    # ------------------------------------------------------------------ #

    def parse_result(self, output_path: Path) -> QMResult:
        """Parse method, energy, HOMO/LUMO, dipole, rotational constants."""
        content = Path(output_path).read_text(errors="ignore", encoding='utf-8')
        converged = _TERMINATION_STRING in content

        extras: Dict[str, Any] = {"method": _detect_method(content)}

        if (m := _CHARGE_RE.search(content)) is not None:
            extras["charge"] = int(m.group(1))
        if (m := _UNPAIRED_RE.search(content)) is not None:
            extras["multiplicity"] = int(m.group(1)) + 1

        energy_matches = _TOTAL_ENERGY_RE.findall(content)
        energy = float(energy_matches[-1]) if energy_matches else float("nan")
        extras["final_electronic_energy"] = energy if energy_matches else None

        if (g := _HOMO_LUMO_GAP_RE.findall(content)):
            extras["homo_lumo_gap"] = float(g[-1])
        if (homo := _HOMO_RE.findall(content)):
            extras["homo_energy"] = float(homo[-1][0])
        if (lumo := _LUMO_RE.findall(content)):
            extras["lumo_energy"] = float(lumo[-1][0])
        if (dipoles := _DIPOLE_FULL_RE.findall(content)):
            extras["dipole_moment"] = float(dipoles[-1][3])
        if (rc := _ROT_CONSTANTS_RE.search(content)) is not None:
            extras["rotational_constants"] = np.array(
                [float(rc.group(1)), float(rc.group(2)), float(rc.group(3))]
            )

        if not energy_matches:
            raise QMError(
                f"xtb: no TOTAL ENERGY found in {output_path} "
                f"(converged={converged})"
            )

        return QMResult(energy=energy, converged=converged, extras=extras)

    def detect_convergence(self, output_path: Path) -> bool:
        """Return True iff the log contains xTB's normal-termination banner."""
        path = Path(output_path)
        if not path.exists():
            return False
        return _TERMINATION_STRING in path.read_text(errors="ignore", encoding='utf-8')


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _gfn_flags(method: Optional[str]) -> list[str]:
    """Translate the method string from the .asc input into an xtb GFN flag."""
    upper = (method or "GFN2-xTB").upper()
    if "GFN-FF" in upper or "GFNFF" in upper:
        return ["--gfnff"]
    if "GFN0" in upper:
        return ["--gfn", "0"]
    if "GFN1" in upper:
        return ["--gfn", "1"]
    return ["--gfn", "2"]


def _detect_method(content: str) -> str:
    """Identify the GFN method from xtb's spaced banner (``G F N 2 - x T B``)."""
    if _GFN_FF_RE.search(content):
        return "GFN-FF"
    if _GFN0_BANNER_RE.search(content):
        return "GFN0-xTB"
    if _GFN1_BANNER_RE.search(content):
        return "GFN1-xTB"
    return "GFN2-xTB"


# --------------------------------------------------------------------------- #
# xcontrol constraint generation                                              #
# --------------------------------------------------------------------------- #


def build_xcontrol_constraints(
    coords,
    atomic_numbers: Sequence[int],
    *,
    force_constant: float = DEFAULT_CONSTRAIN_FORCE_CONSTANT,
) -> Optional[str]:
    """Render a ``$constrain`` block holding every polar X–H bond.

    Uses ``distance: i,j,auto`` — one biased pair per bond — and deliberately
    *not* ``atoms: <list>``. The ``atoms`` form looks like the obvious way to
    keep a fragment together, but repeated ``atoms:`` lines merge into a single
    list and the bias then covers every internal coordinate among the union,
    intermolecular distances included. Measured on two waters 5.52 Å apart:
    unconstrained they close to 2.84 Å, but under ``atoms: 1-3`` + ``atoms: 4-6``
    they stay pinned at 5.52 — the whole cluster freezes and the optimisation
    does nothing. The pairwise ``distance`` form leaves the same pair free to
    close to 2.84 while the O–H bonds hold.

    ``auto`` means "hold at the value this structure already has", so no
    reference geometry is needed.

    Args:
        coords: ``(N, 3)`` Cartesian coordinates in Ångström.
        atomic_numbers: ``N`` atomic numbers, aligned with ``coords``.
        force_constant: Bias strength in au; see
            :data:`DEFAULT_CONSTRAIN_FORCE_CONSTANT`.

    Returns:
        The block text, or ``None`` when the structure has no polar X–H bond —
        an empty ``$constrain`` block makes xtb warn, so there is nothing to
        write in that case.
    """
    pairs = polar_xh_bonds(coords, atomic_numbers)
    if not pairs:
        return None

    lines = ["$constrain", f"  force constant={force_constant}"]
    for heavy, hydrogen in pairs:
        # xtb atom indices are 1-based.
        label_heavy = f"{Z_TO_SYMBOL.get(int(atomic_numbers[heavy]), 'X')}{heavy + 1}"
        label_h = f"{Z_TO_SYMBOL.get(int(atomic_numbers[hydrogen]), 'X')}{hydrogen + 1}"
        lines.append(
            f"  distance: {heavy + 1},{hydrogen + 1},auto"
            f"     # {label_heavy}-{label_h}"
        )
    lines.append("$end")
    return "\n".join(lines) + "\n"


def write_xcontrol_constraints(
    coords,
    atomic_numbers: Sequence[int],
    path: Path,
    *,
    force_constant: float = DEFAULT_CONSTRAIN_FORCE_CONSTANT,
) -> int:
    """Write the ``$constrain`` block to ``path``; return the constraint count.

    Writes nothing and returns ``0`` when the structure has no polar X–H bond.
    Callers gate ``--input`` on the file existing, so a zero return needs no
    further bookkeeping.
    """
    block = build_xcontrol_constraints(
        coords, atomic_numbers, force_constant=force_constant
    )
    if block is None:
        return 0
    Path(path).write_text(block, encoding="utf-8")
    return block.count("distance:")


__all__ = [
    "DEFAULT_CONSTRAIN_FORCE_CONSTANT",
    "XCONTROL_EXTENSION",
    "XTBAdapter",
    "build_xcontrol_constraints",
    "write_xcontrol_constraints",
]
