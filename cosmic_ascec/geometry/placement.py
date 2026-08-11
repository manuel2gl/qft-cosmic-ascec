"""Initial cluster placement — drop molecules into the cube without overlap.

Annealing starts from a randomly-placed initial cluster. For each molecule:
draw a translation (uniform inside the cube), draw three Euler angles
(uniform), build the rotation matrix, apply rotation + translation to the
molecule's template coordinates, and check that the placed atoms do not
overlap any already-placed molecule. Retry up to ``max_attempts`` times per
molecule on overlap; raise :class:`GeometryError` if the cube is just too
small to fit the cluster.

Implementation notes worth knowing if you touch this:

* Translation is *one* ``rng.uniform(low, high, size=3)`` draw — not three
  scalar draws — so the RNG stream order stays stable across refactors and
  reseeded runs reproduce historical trajectories.
* Euler angles are three independent scalar draws, in the order
  ``alpha`` → ``beta`` → ``gamma``.
* Rotation matrices use ``np.cos`` / ``np.sin`` (numpy, *not* ``math``);
  numpy's vectorised transcendentals are not guaranteed to agree with libm
  in the last ULP, so mixing them would silently change bit-exact results.
* The overlap rule uses *covalent* radii with a 0.5 Å default for unknown
  elements — same rule the conformational-move overlap gate enforces, so a
  placement that passes here will pass move-time checks too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cosmic_ascec.elements.data import COVALENT_RADII
from cosmic_ascec.exceptions import GeometryError
from cosmic_ascec.geometry.molecule import Cluster, Molecule
from cosmic_ascec.geometry.overlap import DEFAULT_INTERMOLECULAR_OVERLAP_SCALE

# Fallback radius used when an atomic number is not in COVALENT_RADII.
_PLACEMENT_RADIUS_DEFAULT = 0.5
# Distances below this are treated as a self-comparison, not a real overlap.
_NUMERICAL_ZERO_DISTANCE = 1e-4


@dataclass(frozen=True)
class PlacementSettings:
    """Tuning knobs for :func:`initialize_cluster`. v04 defaults are the
    constructor defaults — most callers should not need to override them.

    ``max_attempts``       — v04 ``state.max_overlap_placement_attempts`` (line 528: 100_000).
    ``range_increase_step``— v04 ``RANGE_INCREASE_STEP`` (line 1237: 5_000).
    ``range_increase_factor``  v04 ``RANGE_INCREASE_FACTOR`` (line 1242: 1.1).
    ``max_range_factor``       v04 ``MAX_RANGE_FACTOR`` (line 1244: 2.0).
    ``overlap_scale``          v04 ``state.overlap_scale_factor`` (line 71: 0.7).
    """

    max_attempts: int = 100_000
    range_increase_step: int = 5_000
    range_increase_factor: float = 1.1
    max_range_factor: float = 2.0
    overlap_scale: float = DEFAULT_INTERMOLECULAR_OVERLAP_SCALE


@dataclass(frozen=True)
class PlacementOutcome:
    """Per-molecule outcome reported alongside the placed :class:`Cluster`."""

    molecule_index: int
    label: str
    attempts: int
    succeeded: bool
    final_translation_scale: float
    final_rotation_scale: float


@dataclass(frozen=True)
class PlacementResult:
    cluster: Cluster
    outcomes: tuple


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _euler_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZYX Euler-angle rotation matrix — verbatim from v04 (lines 1282-1300).

    Built with ``np.cos``/``np.sin`` (not ``math``) so the matrix is
    byte-identical to v04's; ``Rz @ Ry @ Rx`` uses the same ``@`` operator v04
    uses, so the assembled matrix matches bit-for-bit.
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)],
    ])

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])

    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def _placement_overlap_found(
    proposed_mol_atoms: list,
    placed_atoms_data: list,
    overlap_scale: float,
) -> bool:
    """v04 ``config_molecules`` inter-molecular overlap loop (lines 1310-1331).

    ``proposed_mol_atoms`` is a list of ``(Z, abs_coords)`` for the molecule
    being placed; ``placed_atoms_data`` is a list of ``(Z, x, y, z)`` for every
    atom already committed. Radii come from the covalent ``COVALENT_RADII``
    table (v04 ``state.r_atom``) with a ``0.5`` default — v04's placement loop
    does not use the monatomic/VDW distinction.
    """
    for prop_atom_num, prop_coords in proposed_mol_atoms:
        for placed_atom_data in placed_atoms_data:
            placed_atom_num = placed_atom_data[0]
            placed_coords = np.array(placed_atom_data[1:])

            distance = np.linalg.norm(prop_coords - placed_coords)

            radius1 = COVALENT_RADII.get(prop_atom_num, _PLACEMENT_RADIUS_DEFAULT)
            radius2 = COVALENT_RADII.get(placed_atom_num, _PLACEMENT_RADIUS_DEFAULT)

            min_distance_allowed = (radius1 + radius2) * overlap_scale

            if distance < min_distance_allowed and distance > _NUMERICAL_ZERO_DISTANCE:
                return True
    return False


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def initialize_cluster(
    molecules: Sequence[Molecule],
    *,
    box_length: float,
    rng: np.random.RandomState,
    settings: Optional[PlacementSettings] = None,
    logger=None,
) -> PlacementResult:
    """Generate an initial multi-molecule cluster — verbatim port of v04's
    ``config_molecules`` (ascec-v04.py lines 1214-1346).

    Args:
        molecules: One entry per *instance* to be placed, in the order v04
            iterates ``state.molecules_to_add`` (the ``.asc`` file order). A
            system of two identical waters → two :class:`Molecule` references.
        box_length: Cube edge in Angstroms (v04 ``state.xbox`` / ``cube_length``).
        rng: Explicit ``numpy.random.RandomState`` — never optional, never
            global (D-038).
        settings: Tuning knobs (see :class:`PlacementSettings`).
        logger: Optional ``logging.Logger``. If supplied, mirrors v04's
            verbose-warning behaviour when a molecule cannot be placed
            (``_print_verbose``, v04 line 1334).

    Returns:
        :class:`PlacementResult` with the final :class:`Cluster` and per-molecule
        outcomes. As in v04, the cluster is built even on best-effort failure
        (v04 "Placing it anyway") so callers can decide whether to retry.
    """
    if not molecules:
        raise GeometryError("initialize_cluster: no molecules to place")
    if box_length <= 0:
        raise GeometryError(f"initialize_cluster: invalid box_length={box_length!r}")

    cfg = settings or PlacementSettings()

    # v04 ``placed_atoms_data`` — list of (Z, x, y, z) for every committed atom.
    placed_atoms_data: list = []
    # Indexed by molecule so the frozen-first commit order below does not
    # reorder the cluster: entry ``i`` always belongs to ``molecules[i]``.
    per_molecule_blocks: list = [None] * len(molecules)
    outcomes: list = [None] * len(molecules)

    # Frozen molecules (``*$`` blocks) are committed first, at exactly the
    # coordinates from the file. Doing them up front puts their atoms into
    # ``placed_atoms_data`` before any mobile molecule searches for a
    # clash-free spot, so the mobile ones place themselves around the pinned
    # geometry rather than through it.
    for mol_idx, mol in enumerate(molecules):
        if not getattr(mol, "frozen", False):
            continue
        block = np.array(mol.coords, dtype=np.float64, copy=True)
        for atom_idx in range(mol.num_atoms):
            placed_atoms_data.append((
                mol.atomic_numbers[atom_idx],
                block[atom_idx][0], block[atom_idx][1], block[atom_idx][2],
            ))
        per_molecule_blocks[mol_idx] = block
        outcomes[mol_idx] = PlacementOutcome(
            molecule_index=mol_idx,
            label=mol.label,
            attempts=0,
            succeeded=True,
            final_translation_scale=1.0,
            final_rotation_scale=1.0,
        )

    for mol_idx, mol in enumerate(molecules):
        if getattr(mol, "frozen", False):
            continue
        overlap_found = True
        attempts = 0
        proposed_mol_atoms: list = []  # [(Z, abs_coords ndarray), ...]

        # v04 resets the local scales and the increase threshold per molecule
        # (lines 1254-1256).
        local_translation_range_factor = 1.0
        local_rotation_range_factor = 1.0
        next_increase_threshold = cfg.range_increase_step

        while overlap_found and attempts < cfg.max_attempts:
            attempts += 1

            # v04 lines 1267-1272: widen the placement ranges every
            # ``range_increase_step`` failed attempts (capped).
            if attempts >= next_increase_threshold:
                local_translation_range_factor = min(
                    cfg.max_range_factor,
                    local_translation_range_factor * cfg.range_increase_factor,
                )
                local_rotation_range_factor = min(
                    cfg.max_range_factor,
                    local_rotation_range_factor * cfg.range_increase_factor,
                )
                next_increase_threshold += cfg.range_increase_step

            # v04 line 1275 — one uniform draw of size 3 (consumes 3 doubles).
            translation = rng.uniform(
                -box_length / 2 * local_translation_range_factor,
                box_length / 2 * local_translation_range_factor,
                size=3,
            )

            # v04 lines 1278-1280 — three scalar Euler-angle draws, in order.
            alpha = rng.uniform(0, 2 * np.pi * local_rotation_range_factor)
            beta = rng.uniform(0, 2 * np.pi * local_rotation_range_factor)
            gamma = rng.uniform(0, 2 * np.pi * local_rotation_range_factor)

            rotation_matrix = _euler_rotation_matrix(alpha, beta, gamma)

            # v04 lines 1302-1308 — rotate each raw template atom, translate.
            proposed_mol_atoms = []
            for atom_idx in range(mol.num_atoms):
                atomic_num = mol.atomic_numbers[atom_idx]
                relative_coords_vector = mol.coords[atom_idx]
                rotated_coords = np.dot(rotation_matrix, relative_coords_vector)
                abs_coords = rotated_coords + translation
                proposed_mol_atoms.append((atomic_num, abs_coords))

            # v04 lines 1310-1331 — clash against every committed atom.
            overlap_found = _placement_overlap_found(
                proposed_mol_atoms, placed_atoms_data, cfg.overlap_scale
            )

        if overlap_found and logger is not None:
            # v04 line 1334 — warn and place anyway.
            logger.warning(
                "placement: could not find a non-overlapping placement for "
                "molecule %d (%s) after %d attempts; placing it anyway",
                mol_idx, mol.label, attempts,
            )

        # v04 lines 1336-1341 — commit the molecule's atoms.
        for atomic_num, abs_coords in proposed_mol_atoms:
            placed_atoms_data.append(
                (atomic_num, abs_coords[0], abs_coords[1], abs_coords[2])
            )
        proposed_block = np.array(
            [abs_coords for _, abs_coords in proposed_mol_atoms],
            dtype=np.float64,
        )
        per_molecule_blocks[mol_idx] = proposed_block

        outcomes[mol_idx] = PlacementOutcome(
            molecule_index=mol_idx,
            label=mol.label,
            attempts=attempts,
            succeeded=not overlap_found,
            final_translation_scale=local_translation_range_factor,
            final_rotation_scale=local_rotation_range_factor,
        )

    cluster = Cluster.from_blocks(
        molecules=molecules,
        per_molecule_coords=per_molecule_blocks,
        box_length=box_length,
    )
    return PlacementResult(cluster=cluster, outcomes=tuple(outcomes))


__all__ = [
    "PlacementOutcome",
    "PlacementResult",
    "PlacementSettings",
    "initialize_cluster",
]
