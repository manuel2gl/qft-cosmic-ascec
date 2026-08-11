"""Typed configuration objects produced by the ``.asc`` parser.

The ``.asc`` format is the long-standing contract between this package and
the ``index.html`` input generator — the file layout is fixed; we may
redesign the internals but never the file. Each dataclass below mirrors
one section of a ``.asc`` file (box, schedule, cycles, moves, QM,
molecules, protocol).

The QM program index → name mapping is fixed:

    1 → Gaussian, 2 → ORCA, 3 → xTB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
# Enums                                                                       #
# --------------------------------------------------------------------------- #


class SimulationMode(IntEnum):
    """Line 1, first integer (v04: ``state.random_generate_config``)."""

    RANDOM = 0
    ANNEALING = 1


class QuenchingRoute(IntEnum):
    """Line 3 (v04: ``state.quenching_routine``)."""

    LINEAR = 1
    GEOMETRIC = 2


class QMProgram(IntEnum):
    """Line 9, first integer (v04: ``state.ia``)."""

    GAUSSIAN = 1
    ORCA = 2
    XTB = 3


# --------------------------------------------------------------------------- #
# Component dataclasses                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoxSpec:
    """Simulation cube. Line 2 (v04: ``state.cube_length`` / ``state.xbox``)."""

    cube_length_angstrom: float


@dataclass(frozen=True)
class LinearSchedule:
    """Linear quenching. Line 4 (v04: ``linear_temp_init``, ``linear_temp_decrement``,
    ``linear_num_steps``)."""

    initial_temperature: float
    delta_temperature: float
    num_steps: int


@dataclass(frozen=True)
class GeometricSchedule:
    """Geometric quenching. Line 5 (v04: ``geom_temp_init``, ``geom_temp_factor``,
    ``geom_num_steps``).

    ``factor`` is the multiplicative cooling factor in the v04 sense — already
    converted from the percentage decrease the user typed. v04 turns ``5.0`` on
    disk into ``0.95`` here (lines 1497-1502).
    """

    initial_temperature: float
    factor: float
    num_steps: int


@dataclass(frozen=True)
class ScheduleSpec:
    """Line 3-5 together. ``route`` selects which of the two schedules is used.

    Both schedule blocks are always populated (v04 unconditionally reads both
    lines) so downstream code can switch between them without re-parsing.
    """

    route: QuenchingRoute
    linear: LinearSchedule
    geometric: GeometricSchedule


@dataclass(frozen=True)
class CycleSpec:
    """Line 6 (v04: ``max_cycle``, ``max_cycle_floor``)."""

    max_cycles_per_temperature: int
    floor_value: Optional[int] = None  # absent → v04 default (10) applied later


@dataclass(frozen=True)
class MoveSpec:
    """Lines 7-8. Caps for the Monte Carlo moves.

    Line 7 (v04: ``max_displacement_a``, ``max_rotation_angle_rad``).
    Line 8 (v04: ``conformational_move_prob``, ``max_dihedral_angle_rad``).
    The conformational probability lives in ``[0, 1]`` here, not in percent.
    The dihedral cap is stored in radians (v04 converts on read).
    """

    max_displacement_angstrom: float
    max_rotation_radian: float
    conformational_probability: float
    max_dihedral_radian: float


@dataclass(frozen=True)
class QMSpec:
    """Lines 9-12. QM program selection plus charge/multiplicity."""

    program: QMProgram                # Line 9 col 1: 1/2/3
    alias: str                        # Line 9 col 2: e.g. "g09", "orca", "xtb"
    method: str                       # Line 10 col 1: "PM3", "GFN2-xTB", ...
    basis_set: Optional[str] = None   # Line 10 col 2 (optional)
    qm_nprocs: int = 1                # Line 11 col 1
    ascec_nprocs: Optional[int] = None  # Line 11 col 2 (optional → auto-decide)
    charge: int = 0                   # Line 12 col 1
    multiplicity: int = 1             # Line 12 col 2


# A single atom row inside a molecule definition: (Z, x, y, z) in Å.
AtomRow = Tuple[int, float, float, float]


@dataclass(frozen=True)
class MoleculeSpec:
    """One ``*``-delimited block after line 13.

    Captures the explicit atom count (v04 cross-checks it), the human-facing
    label (``water1``, ``Glycolaldehyde``, …), and the coordinates exactly as
    they appear in the file.

    ``frozen`` marks a block written between ``*$`` delimiters on *both* sides.
    Such a fragment is placed at exactly the coordinates in the file and is
    never translated or rotated during annealing — see
    :func:`cosmic_ascec.geometry.placement.initialize_cluster` and
    :func:`cosmic_ascec.monte_carlo.moves.propose_unified_move`. v04 has no
    equivalent, so an input without ``*$`` behaves exactly as before.
    """

    label: str
    atoms: Tuple[AtomRow, ...]
    frozen: bool = False

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class ProtocolSpec:
    """Optional ``# Protocol`` block (v04: ``extract_protocol_from_input``).

    ``raw_text`` is the joined-and-whitespace-collapsed line v04 produces; we
    keep it verbatim so the existing workflow parser can consume it later. The
    list form is the same protocol split on commas, surfaced for readability.
    """

    raw_text: str
    steps: Tuple[str, ...]


@dataclass(frozen=True)
class EmbeddedTemplate:
    """One ``#orca/#gaussian/#xtb <label>`` block at the bottom of the file.

    ``extension`` mirrors v04 (`.inp` / `.com` / `.xtb`) so the workflow stage
    can drop it on disk under the right filename.
    """

    program: QMProgram
    label: str
    content: str
    extension: str


# --------------------------------------------------------------------------- #
# Root container                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AscConfig:
    """Everything a ``.asc`` file expresses, fully typed.

    Field order tracks the file: 1) the 13 header lines, 2) molecules, 3) the
    optional protocol, 4) the optional embedded QM templates. ``source_path``
    is the file the config came from (or ``None`` for stdin / synthetic input).
    """

    mode: SimulationMode                          # Line 1 col 1
    num_configurations: int                       # Line 1 col 2
    box: BoxSpec                                  # Line 2
    schedule: ScheduleSpec                        # Lines 3-5
    cycles: CycleSpec                             # Line 6
    moves: MoveSpec                               # Lines 7-8
    qm: QMSpec                                    # Lines 9-12
    molecules: Tuple[MoleculeSpec, ...]           # blocks after line 13
    protocol: Optional[ProtocolSpec] = None
    embedded_templates: Tuple[EmbeddedTemplate, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    @property
    def num_molecules(self) -> int:
        return len(self.molecules)

    @property
    def total_atoms(self) -> int:
        return sum(m.num_atoms for m in self.molecules)


__all__ = [
    "AscConfig",
    "AtomRow",
    "BoxSpec",
    "CycleSpec",
    "EmbeddedTemplate",
    "GeometricSchedule",
    "LinearSchedule",
    "MoleculeSpec",
    "MoveSpec",
    "ProtocolSpec",
    "QMProgram",
    "QMSpec",
    "QuenchingRoute",
    "ScheduleSpec",
    "SimulationMode",
]
