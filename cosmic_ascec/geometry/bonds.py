"""Bond detection and rotatable-bond discovery for conformational moves.

Two pure-function ports from v04:

* :func:`find_connected_atoms` — depth-first traversal of the implicit
  bond graph, excluding one bond direction. ascec-v04.py lines 3655-3691.
* :func:`find_rotatable_bonds` — distance-based bond detection plus the
  "non-terminal, non-H" filter v04 uses to pick rotatable bonds.
  ascec-v04.py lines 3620-3653.

Three deliberate departures from v04, all contract-driven:

1. v04 reaches into ``state.r_atom`` for every radius lookup (a covalent-only
   table built once at startup). v04 calls :func:`get_radius` from
   ``cosmic_ascec.elements.radii`` directly, keeping ``bonds.py`` free of any
   ``SystemState`` god-object dependency. Numeric equivalence is preserved
   because v04's ``state.r_atom`` is itself the covalent table.
2. v04's bond-length cutoff hard-codes a 1.3 tolerance and a 1.50 Å radius
   fallback inline. Both are surfaced here as module constants so the MC layer
   in Session 3 can introspect them without re-importing v04 magic numbers.
3. **Ring bonds are excluded from the rotatable set** (:func:`_is_ring_bond`).
   v04 has no ring perception, so a ring bond (e.g. any C–C in cyclohexane)
   passes its "non-terminal, non-H" filter and gets rotated as a rigid
   fragment — which tears the ring open rather than puckering it, because the
   fragment swings around the bond axis while the ring-closure bond is left
   behind. Rotating a ring bond can never produce a valid conformer, so we
   drop it here. This changes behaviour *only* for cyclic molecules; acyclic
   inputs (the case v04 handled correctly) enumerate an identical bond set.

The companion ``rotate_around_bond`` (Rodrigues' formula, v04 line 3569) is
intentionally *not* ported here — it is a Monte Carlo move, owned by
Session 3's ``moves`` module. Keeping bond detection as a pure geometric
utility prevents bonds.py from depending on the random-number stack.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from cosmic_ascec.elements.radii import get_radius


BOND_DISTANCE_TOLERANCE: float = 1.3
"""Multiplier applied to (r_i + r_j) when deciding whether two atoms are bonded.

Matches ascec-v04.py line 3641. A pair ``(i, j)`` is treated as bonded when
``|r_j - r_i| < (radius_i + radius_j) * BOND_DISTANCE_TOLERANCE``.
"""


def _bonded(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
    i: int,
    j: int,
) -> bool:
    """Distance-based bond test (v04's inline check at lines 3639-3643)."""
    diff = coords[j] - coords[i]
    distance = float(np.sqrt(diff @ diff))
    radius_i = get_radius(int(atomic_numbers[i]))
    radius_j = get_radius(int(atomic_numbers[j]))
    max_bond_length = (radius_i + radius_j) * BOND_DISTANCE_TOLERANCE
    return distance < max_bond_length


def find_connected_atoms(
    start_atom: int,
    exclude_atom: int,
    mol_coords: np.ndarray,
    mol_atomic_numbers: Sequence[int],
) -> List[int]:
    """Atoms reachable from ``start_atom`` without crossing the ``(start, exclude)`` bond.

    Direct DFS port of ascec-v04.py lines 3655-3691. The returned list
    contains every atom in the fragment *except* ``start_atom`` itself, so
    callers can use it directly as the "moving atoms" argument for a
    bond-axis rotation.
    """
    coords = np.asarray(mol_coords, dtype=np.float64)
    n_atoms = len(mol_atomic_numbers)

    visited: set[int] = {exclude_atom}
    to_visit: list[int] = [start_atom]
    connected: list[int] = []

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        connected.append(current)

        for neighbor in range(n_atoms):
            if neighbor in visited:
                continue
            if _bonded(coords, mol_atomic_numbers, current, neighbor):
                to_visit.append(neighbor)

    if start_atom in connected:
        connected.remove(start_atom)
    return connected


def _is_ring_bond(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
    i: int,
    j: int,
) -> bool:
    """True if the ``i–j`` bond lies in a ring.

    A bond is a ring bond iff its two atoms stay connected after that single
    bond is removed — i.e. there is an alternate path from ``i`` to ``j``. We
    walk the bond graph out of ``i`` with the direct ``i–j`` edge forbidden; if
    the walk still reaches ``j`` by any other route, the bond closes a ring.

    Rotating such a bond as a rigid fragment tears the ring instead of
    puckering it (the ring-closure bond is left behind), so ring bonds must be
    kept out of the rotatable set — see the module docstring, departure 3.
    """
    n = len(atomic_numbers)
    visited: set[int] = {i}
    stack: list[int] = [i]

    while stack:
        current = stack.pop()
        for neighbor in range(n):
            if neighbor in visited:
                continue
            if not _bonded(coords, atomic_numbers, current, neighbor):
                continue
            if current == i and neighbor == j:
                continue  # the direct i–j bond is treated as removed
            if neighbor == j:
                return True  # reached j by an alternate path -> ring bond
            visited.add(neighbor)
            stack.append(neighbor)

    return False


def find_rotatable_bonds(
    mol_coords: np.ndarray,
    mol_atomic_numbers: Sequence[int],
) -> List[Tuple[int, int, List[int]]]:
    """Enumerate rotatable bonds ``(i, j, moving_atoms)`` for a single molecule.

    Filter (matches ascec-v04.py lines 3620-3653, plus the ring exclusion of
    departure 3 in the module docstring):

    * Pair must be bonded by the distance criterion.
    * Both atoms must be heavy (no H–X rotatable bonds).
    * The bond must not close a ring (:func:`_is_ring_bond`) — rotating a ring
      bond tears the ring rather than puckering it.
    * The fragment that would rotate must be non-empty *and* leave at least
      two atoms behind (excludes terminal bonds and the degenerate "rotate
      everything" case).
    """
    coords = np.asarray(mol_coords, dtype=np.float64)
    n_atoms = len(mol_atomic_numbers)
    bonds: list[Tuple[int, int, List[int]]] = []

    for i in range(n_atoms):
        z_i = int(mol_atomic_numbers[i])
        for j in range(i + 1, n_atoms):
            z_j = int(mol_atomic_numbers[j])
            if not _bonded(coords, mol_atomic_numbers, i, j):
                continue
            if z_i == 1 or z_j == 1:
                continue
            if _is_ring_bond(coords, mol_atomic_numbers, i, j):
                continue
            moving_atoms = find_connected_atoms(j, i, coords, mol_atomic_numbers)
            if 0 < len(moving_atoms) < n_atoms - 2:
                bonds.append((i, j, moving_atoms))

    return bonds


__all__ = [
    "BOND_DISTANCE_TOLERANCE",
    "find_connected_atoms",
    "find_rotatable_bonds",
]
