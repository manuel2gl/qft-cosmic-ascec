"""Steric overlap checks.

* :func:`check_intramolecular_overlap` — two atoms *within one molecule*
  clash when their separation is below ``(r_i + r_j) * 0.5`` (covalent
  radii, 0.5 Å default for unknowns). This is the *only* overlap check on
  the live Monte Carlo move path: it gates the conformational (dihedral)
  branch of :func:`~cosmic_ascec.monte_carlo.moves.propose_unified_move`.
  The rigid translation/rotation branch and the QM evaluation do **no**
  overlap check — a clashing geometry is sent straight to the QM program.

* :func:`check_intermolecular_overlap` — clash check between two distinct
  placed atoms with a tunable scale. Placement uses its own inline loop
  (in :mod:`~cosmic_ascec.geometry.placement`), so this helper is currently
  only used by the geometry unit tests; it's kept here as the canonical
  reference rule.

Distances below ``1e-4`` Å are treated as a degenerate self-comparison and
ignored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cosmic_ascec.elements.data import COVALENT_RADII
from cosmic_ascec.elements.radii import get_radius

_NUMERICAL_ZERO_DISTANCE = 1e-4
_INTRAMOLECULAR_FACTOR = 0.5  # ascec-v04.py line 3719.
_INTRAMOLECULAR_RADIUS_DEFAULT = 0.5  # ascec-v04.py lines 3714-3715.

# v04's runtime default (ascec-v04.py line 71). Kept here as a module-level
# constant rather than imported from constants.py so the geometry layer is
# self-contained; tests pin both copies to the same number.
DEFAULT_INTERMOLECULAR_OVERLAP_SCALE = 0.7


def check_intramolecular_overlap(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
) -> bool:
    """Return True if any pair of atoms inside a molecule clashes.

    Verbatim port of v04 ``check_intramolecular_overlap`` (ascec-v04.py lines
    3693-3725): covalent radii from ``COVALENT_RADII`` (v04 ``state.r_atom``)
    with a ``0.5`` default, separation via ``np.linalg.norm``, threshold
    ``(r_i + r_j) * 0.5``, and the ``d > 1e-4`` numerical-noise guard.
    """
    n = len(atomic_numbers)
    if n < 2:
        return False

    coords = np.asarray(coords, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(coords[i] - coords[j])
            radius_i = COVALENT_RADII.get(
                int(atomic_numbers[i]), _INTRAMOLECULAR_RADIUS_DEFAULT
            )
            radius_j = COVALENT_RADII.get(
                int(atomic_numbers[j]), _INTRAMOLECULAR_RADIUS_DEFAULT
            )
            min_distance_allowed = (radius_i + radius_j) * _INTRAMOLECULAR_FACTOR
            if distance < min_distance_allowed and distance > _NUMERICAL_ZERO_DISTANCE:
                return True
    return False


def check_intermolecular_overlap(
    proposed_coords: np.ndarray,
    proposed_atomic_numbers: Sequence[int],
    placed_coords: np.ndarray,
    placed_atomic_numbers: Sequence[int],
    *,
    overlap_scale: float = DEFAULT_INTERMOLECULAR_OVERLAP_SCALE,
    proposed_is_monatomic: bool = False,
    placed_is_monatomic: bool = False,
) -> bool:
    """True if any proposed atom clashes with any already-placed atom.

    **Not on any live code path** — see the module docstring. Retained for the
    geometry unit tests. Mirrors the *shape* of ``config_molecules``' inline
    clash loop (ascec-v04.py lines 1310-1323), though the production placement
    path uses :func:`geometry.placement._placement_overlap_found` instead.
    """
    proposed = np.asarray(proposed_coords, dtype=np.float64)
    placed = np.asarray(placed_coords, dtype=np.float64)
    if placed.size == 0 or proposed.size == 0:
        return False

    for k in range(proposed.shape[0]):
        rk = get_radius(proposed_atomic_numbers[k], monatomic=proposed_is_monatomic)
        for m in range(placed.shape[0]):
            rm = get_radius(placed_atomic_numbers[m], monatomic=placed_is_monatomic)
            distance = np.linalg.norm(proposed[k] - placed[m])
            threshold = (rk + rm) * overlap_scale
            if distance < threshold and distance > _NUMERICAL_ZERO_DISTANCE:
                return True
    return False


# --------------------------------------------------------------------------- #
# Contact gate for substrate runs (``*$`` frozen blocks)                       #
# --------------------------------------------------------------------------- #
#
# The covalent-radius rule above is v04's, and v04 only ever annealed molecular
# clusters. It is far too permissive once a metal substrate is in the box: the
# Au-H floor works out to ``(1.36 + 0.31) * 0.7 = 1.17`` A, so a hydrogen may
# legally sit *inside* a gold lattice. That does not merely give a high energy
# the Metropolis test would reject -- it makes the SCF fail to converge, which
# yields no energy at all and kills the run.
#
# For intermolecular contacts the right length scale is van der Waals, not
# covalent. VDW * 0.60 forbids the pathological geometries while still leaving
# real chemistry reachable:
#
#     Au-N  1.93 A floor   (chemisorbed guanine sits at ~2.2-2.5 A)
#     Au-O  1.91 A floor   (adsorbed water sits at ~2.2-3.0 A)
#     O...H 1.63 A floor   (hydrogen bonds sit at ~1.75-2.0 A)
#     O...O 1.82 A floor   (hydrogen-bonded O...O sits at ~2.7 A)

DEFAULT_SUBSTRATE_CONTACT_SCALE = 0.60
"""VDW scale factor for the substrate-run intermolecular contact gate.

Lower it to allow tighter contacts, raise it to push molecules further apart.
Above roughly ``0.70`` the gate starts rejecting genuine chemisorption.
"""


def vdw_contact_radii(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Per-atom van der Waals radii (A) for the contact gate."""
    return np.array(
        [get_radius(int(z), monatomic=True) for z in atomic_numbers],
        dtype=np.float64,
    )


def has_contact_clash(
    probe_coords: np.ndarray,
    probe_radii: np.ndarray,
    other_coords: np.ndarray,
    other_radii: np.ndarray,
    *,
    scale: float = DEFAULT_SUBSTRATE_CONTACT_SCALE,
) -> bool:
    """True if any probe atom is closer to any other atom than the VDW floor.

    Vectorised over both atom sets -- the gate runs once per molecule per MC
    cycle, so the pairwise block is built with NumPy rather than a Python loop.
    Distances below ``1e-4`` A are ignored as degenerate self-comparisons, the
    same guard the v04 rules use.
    """
    if probe_coords.size == 0 or other_coords.size == 0:
        return False
    deltas = probe_coords[:, None, :] - other_coords[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    floors = (probe_radii[:, None] + other_radii[None, :]) * scale
    return bool(
        np.any((distances < floors) & (distances > _NUMERICAL_ZERO_DISTANCE))
    )


__all__ = [
    "DEFAULT_INTERMOLECULAR_OVERLAP_SCALE",
    "DEFAULT_SUBSTRATE_CONTACT_SCALE",
    "check_intermolecular_overlap",
    "check_intramolecular_overlap",
    "has_contact_clash",
    "vdw_contact_radii",
]
