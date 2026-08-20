"""Bond enumeration, fragment decomposition, and polar X–H detection.

``bonds.py`` owns the pairwise bond *criterion* (:func:`~cosmic_ascec.geometry.bonds._bonded`)
and a directed traversal used by the rotatable-bond move. What it does not have
is the two whole-cluster primitives this module adds:

* :func:`iter_bonds` — every bonded pair in a structure, as an index list.
* :func:`find_fragments` — connected components of that bond graph.
* :func:`polar_xh_bonds` — the O–H / N–H / S–H bonds that a coordinating metal
  can steal a proton from.

The last one is what the xTB ``$constrain`` generator consumes. Charged metallic
clusters optimised with GFN2-xTB routinely deprotonate their solvent: the metal
pulls an H off a coordinating water or alcohol, and the optimised structure is no
longer the cluster that was annealed. Biasing just those bonds stops the transfer
while leaving every other coordinate — including the whole intermolecular
landscape — free to relax.

The bond criterion is the one from ``bonds.py``, evaluated vectorised here
because these functions run over every structure in a stage rather than over a
single molecule template. :func:`iter_bonds` is pinned to agree with
:func:`~cosmic_ascec.geometry.bonds._bonded` pair-for-pair.

Radii are looked up *without* ``monatomic=True`` — deliberately. The monatomic
branch of :func:`~cosmic_ascec.elements.radii.get_radius` returns van der Waals
radii, which for a lone metal cation are large enough to "bond" it to half the
cluster. Covalent radii keep a metal–ligand contact a bond and nothing more.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from cosmic_ascec.elements.data import SYMBOL_TO_Z
from cosmic_ascec.geometry.bonds import BOND_DISTANCE_TOLERANCE
from cosmic_ascec.elements.radii import get_radius


POLAR_HEAVY_SYMBOLS = frozenset({"N", "P", "O", "S", "Se", "F", "Cl", "Br", "I"})
"""Heavy atoms whose X–H bond is polar enough for the proton to be transferable.

Same set as the hydrogen-bond acceptor list in ``box.py`` — a good H-bond
acceptor and a deprotonatable donor are the same elements. ``box.py`` imports
this constant rather than keeping a second copy.
"""

POLAR_HEAVY_Z = frozenset(
    SYMBOL_TO_Z[symbol] for symbol in POLAR_HEAVY_SYMBOLS
)
"""Atomic numbers of :data:`POLAR_HEAVY_SYMBOLS`, for coordinate-level code."""

XH_DISTANCE_TOLERANCE: float = 1.6
"""Bond tolerance used by :func:`polar_xh_bonds`, wider than the general 1.3.

Not a cosmetic loosening. At :data:`~cosmic_ascec.geometry.bonds.BOND_DISTANCE_TOLERANCE`
the O–H cutoff is 1.261 Å, so an O–H already stretched past that — a proton
partway onto a metal — is not seen as a bond at all, and would silently receive
no constraint. That is precisely the geometry the constraint exists to rescue,
so the X–H search has to reach further than bond perception does. At 1.6 the
cutoffs are O–H 1.55 Å and N–H 1.63 Å, still comfortably short of a hydrogen
bond contact (1.8–2.0 Å), so ordinary H-bonded neighbours are not swept in.

Only :func:`polar_xh_bonds` uses this. :func:`iter_bonds` and
:func:`find_fragments` stay on the repo-wide criterion.
"""

_HYDROGEN_Z = 1


def _radii(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Covalent radius per atom, in the same order as ``atomic_numbers``."""
    return np.array(
        [get_radius(int(z)) for z in atomic_numbers], dtype=float
    )


def _bond_matrix(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
    tolerance: float = BOND_DISTANCE_TOLERANCE,
) -> np.ndarray:
    """Boolean adjacency matrix under the ``bonds.py`` criterion.

    Vectorised form of :func:`~cosmic_ascec.geometry.bonds._bonded` applied to
    every pair: ``|r_j - r_i| < (radius_i + radius_j) * tolerance``. With the
    default ``tolerance`` the result is pinned to ``_bonded`` pair-for-pair.
    The diagonal is False.
    """
    coords = np.asarray(coords, dtype=float)
    natoms = coords.shape[0]
    if natoms == 0:
        return np.zeros((0, 0), dtype=bool)

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))

    radii = _radii(atomic_numbers)
    cutoffs = (radii[:, None] + radii[None, :]) * tolerance

    adjacency = distances < cutoffs
    np.fill_diagonal(adjacency, False)
    return adjacency


def iter_bonds(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
) -> List[Tuple[int, int]]:
    """Every bonded pair as 0-based ``(i, j)`` with ``i < j``.

    Args:
        coords: ``(N, 3)`` Cartesian coordinates in Ångström.
        atomic_numbers: ``N`` atomic numbers, aligned with ``coords``.

    Returns:
        Bonded index pairs in ascending order.
    """
    adjacency = np.triu(_bond_matrix(coords, atomic_numbers), k=1)
    rows, cols = np.nonzero(adjacency)
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def find_fragments(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
) -> List[List[int]]:
    """Connected components of the bond graph, as sorted 0-based index lists.

    Components come back ordered by their lowest atom index, so the result is
    stable for a given input ordering and usable as a log line.

    A metal cation and the solvent molecules coordinating it land in the *same*
    component — a metal–O contact passes the covalent-radius test. That is the
    physically right answer and it does not affect
    :func:`polar_xh_bonds`, which never consults the component split.
    """
    adjacency = _bond_matrix(coords, atomic_numbers)
    natoms = adjacency.shape[0]

    seen = np.zeros(natoms, dtype=bool)
    fragments: List[List[int]] = []

    for start in range(natoms):
        if seen[start]:
            continue
        # Iterative flood fill — recursion would blow the stack on a large
        # solvent shell, and bonds.find_connected_atoms is directed (it seeds
        # `visited` with an excluded atom) so it does not fit here.
        stack = [start]
        seen[start] = True
        component: List[int] = []
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbour in np.nonzero(adjacency[atom] & ~seen)[0]:
                seen[neighbour] = True
                stack.append(int(neighbour))
        fragments.append(sorted(component))

    return fragments


def polar_xh_bonds(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    tolerance: float = XH_DISTANCE_TOLERANCE,
) -> List[Tuple[int, int]]:
    """``(heavy, hydrogen)`` pairs for every H bonded to a polar heavy atom.

    One entry per hydrogen at most. When an H is bonded to several polar heavy
    atoms — a bridging proton, or one already partway onto a metal — it is
    assigned to its **closest polar** neighbour, not merely its closest
    neighbour. That distinction matters: in a structure where transfer has
    already begun, the nearest bonded heavy atom may be the metal, and binding
    the H there would lock in the very geometry this is meant to prevent.
    Anchoring to the closest polar partner instead pulls it back to the donor.

    Hydrogens with no polar heavy neighbour (C–H, a metal hydride, H2) produce
    no entry — they are not the bonds that break.

    Args:
        coords: ``(N, 3)`` Cartesian coordinates in Ångström.
        atomic_numbers: ``N`` atomic numbers, aligned with ``coords``.
        tolerance: Bond-cutoff multiplier; see :data:`XH_DISTANCE_TOLERANCE`
            for why this is wider than the repo-wide bond tolerance.

    Returns:
        ``(heavy_index, hydrogen_index)`` pairs, 0-based, ordered by hydrogen
        index.
    """
    coords = np.asarray(coords, dtype=float)
    atomic_numbers = [int(z) for z in atomic_numbers]
    adjacency = _bond_matrix(coords, atomic_numbers, tolerance)

    pairs: List[Tuple[int, int]] = []
    for h_index, z in enumerate(atomic_numbers):
        if z != _HYDROGEN_Z:
            continue
        candidates = [
            neighbour
            for neighbour in np.nonzero(adjacency[h_index])[0]
            if atomic_numbers[neighbour] in POLAR_HEAVY_Z
        ]
        if not candidates:
            continue
        closest = min(
            candidates,
            key=lambda n: float(np.linalg.norm(coords[n] - coords[h_index])),
        )
        pairs.append((int(closest), h_index))

    return pairs


def count_fragments(
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
) -> int:
    """Number of connected components — a cheap diagnostic wrapper."""
    return len(find_fragments(coords, atomic_numbers))


__all__ = [
    "POLAR_HEAVY_SYMBOLS",
    "XH_DISTANCE_TOLERANCE",
    "POLAR_HEAVY_Z",
    "count_fragments",
    "find_fragments",
    "iter_bonds",
    "polar_xh_bonds",
]
