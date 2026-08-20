"""Box-length recommender — what cube edge fits a given molecular cluster.

Picking the cube length is a tradeoff: too tight and placement keeps failing
because nothing fits without overlap; too loose and the annealing search
wastes most of its QM calls on configurations of molecules so far apart they
do not interact. This module computes two complementary recommendations:

* **Method A** — convex-hull volume of each molecule, augmented by
  H-bond "ghost cylinders" between donor/acceptor atoms so the box leaves
  room for the expected hydrogen-bond network.
* **Method B** — bounding-box / diagonal-derived volume from per-molecule
  extents. Simpler, used as a sanity-check companion to method A.

Both methods report a table of edge lengths at several packing fractions
(typical values 5-30%), so a user can pick by hand. When ``scipy`` is not
available or a molecule degenerates (fewer than four non-coplanar surface
points), method A falls back to the bounding-box estimate with a fixed 0.52
fill factor — same as the legacy estimator.

The function consumes :class:`~cosmic_ascec.geometry.molecule.Molecule`
instances and returns a structured :class:`BoxLengthReport`. The protocol
workflow's box-stage helper prints the table directly; the single-run
dispatcher uses just the recommended length for one packing fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cosmic_ascec.elements.data import Z_TO_SYMBOL
from cosmic_ascec.elements.radii import get_radius
from cosmic_ascec.geometry.fragments import POLAR_HEAVY_SYMBOLS
from cosmic_ascec.geometry.molecule import Molecule


# v04 defaults: ascec-v04.py line 4454.
_DEFAULT_PACKING_FRACTIONS: Tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
)

# v04: ascec-v04.py lines 4370, 4403. Shared with the xTB constraint generator
# in ``geometry.fragments`` — a good H-bond acceptor and a deprotonatable donor
# are the same elements, so the set lives in one place.
_HB_ACCEPTOR_SYMBOLS = POLAR_HEAVY_SYMBOLS

# v04: ascec-v04.py lines 4278, 4322, 4395, 4399.
_HULL_SURFACE_POINTS_PER_ATOM = 50
_BBOX_OVERESTIMATE_CORRECTION = 0.52
_HB_AVG_LENGTH_ANGSTROM = 2.5
_HB_INTERACTION_RADIUS_ANGSTROM = 1.2


@dataclass(frozen=True)
class HBondAnalysis:
    donors: int
    acceptors: int
    potential_bonds: int
    avg_bond_length: float
    hb_volume_per_bond: float
    total_hb_volume: float


@dataclass(frozen=True)
class BoxLengthRecommendation:
    packing_fraction: float
    box_length_angstrom: float
    box_volume: float
    free_volume: float
    free_volume_fraction: float
    molecular_volume_fraction: float
    hb_network_volume_fraction: float


@dataclass(frozen=True)
class BoxLengthReport:
    """Output of :func:`calculate_optimal_box_length`.

    ``method`` is the same label v04 puts in the results dict
    (``A_hb_volume`` or ``B_diagonal_extent``). ``recommendations`` is keyed by
    a percentage string (``"15.0%"``, ``"20.0%"``, …) to match v04's dict keys
    one-for-one, simplifying the upcoming Session 5 writer port.
    """

    num_molecules: int
    method: str
    has_primary_hbonds: bool
    individual_molecular_volumes: Tuple[Dict[str, object], ...]
    total_molecular_volume: float
    total_hb_network_volume: float
    total_extent_sum: float
    diagonal_box_length: float
    diagonal_derived_volume: float
    total_effective_volume: float
    max_molecular_extent: float
    recommendations: Dict[str, BoxLengthRecommendation]


# --------------------------------------------------------------------------- #
# Per-molecule helpers                                                        #
# --------------------------------------------------------------------------- #


def _atomic_radius(molecule: Molecule, atomic_number: int) -> float:
    return get_radius(atomic_number, monatomic=molecule.is_monatomic)


def calculate_molecular_volume(molecule: Molecule) -> float:
    """Convex-hull volume of a molecule's atomic-sphere surface.

    Uses a 50-point Fibonacci sphere per atom and the scipy QHull
    implementation. Falls back to the bounding-box approximation if scipy is
    unavailable or the hull degenerates — v04 does the same (lines 4272-4304).
    """
    if molecule.num_atoms == 0:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return _bounding_box_volume(molecule)

    n_surf = _HULL_SURFACE_POINTS_PER_ATOM
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    directions: List[Tuple[float, float, float]] = []
    for i in range(n_surf):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n_surf)
        phi = 2.0 * math.pi * i / golden_ratio
        directions.append(
            (math.sin(theta) * math.cos(phi),
             math.sin(theta) * math.sin(phi),
             math.cos(theta))
        )

    points: List[List[float]] = []
    for (z, atom_xyz) in _iter_atoms(molecule):
        radius = _atomic_radius(molecule, z)
        x, y, zc = atom_xyz
        for dx, dy, dz in directions:
            points.append([x + radius * dx, y + radius * dy, zc + radius * dz])

    if len(points) < 4:
        return _bounding_box_volume(molecule)

    try:
        return float(ConvexHull(np.asarray(points)).volume)
    except Exception:
        return _bounding_box_volume(molecule)


def _bounding_box_volume(molecule: Molecule) -> float:
    """Bounding-box fallback (ascec-v04.py line 4307)."""
    if molecule.num_atoms == 0:
        return 0.0
    min_xyz = np.array([np.inf, np.inf, np.inf])
    max_xyz = np.array([-np.inf, -np.inf, -np.inf])
    for (z, atom_xyz) in _iter_atoms(molecule):
        radius = _atomic_radius(molecule, z)
        pos = np.asarray(atom_xyz)
        min_xyz = np.minimum(min_xyz, pos - radius)
        max_xyz = np.maximum(max_xyz, pos + radius)
    extents = max_xyz - min_xyz
    return float(extents[0] * extents[1] * extents[2] * _BBOX_OVERESTIMATE_CORRECTION)


def calculate_molecular_extent(molecule: Molecule) -> float:
    """Longest pair distance including atomic radii (ascec-v04.py line 4326)."""
    if molecule.num_atoms == 0:
        return 0.0
    if molecule.num_atoms == 1:
        return 2.0 * _atomic_radius(molecule, molecule.atomic_numbers[0])

    coords = molecule.coords
    zs = molecule.atomic_numbers
    n = molecule.num_atoms
    max_dist = 0.0
    for i in range(n):
        ri = _atomic_radius(molecule, zs[i])
        xi, yi, zi = coords[i]
        for j in range(i + 1, n):
            rj = _atomic_radius(molecule, zs[j])
            xj, yj, zj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) + ri + rj
            if d > max_dist:
                max_dist = d
    return max_dist


def calculate_hydrogen_bond_potential(molecule: Molecule) -> HBondAnalysis:
    """H-bond-donor/acceptor inventory and cylindrical-volume estimate.

    v04: ``calculate_hydrogen_bond_potential`` line 4384. Donors = ``H`` atoms;
    acceptors = N, P, O, S, Se, F, Cl, Br, I.
    """
    donors = 0
    acceptors = 0
    for z in molecule.atomic_numbers:
        symbol = Z_TO_SYMBOL.get(z, "")
        if symbol == "H":
            donors += 1
        elif symbol in _HB_ACCEPTOR_SYMBOLS:
            acceptors += 1

    potential = min(donors, acceptors)
    hb_volume_per_bond = math.pi * (_HB_INTERACTION_RADIUS_ANGSTROM ** 2) * _HB_AVG_LENGTH_ANGSTROM
    return HBondAnalysis(
        donors=donors,
        acceptors=acceptors,
        potential_bonds=potential,
        avg_bond_length=_HB_AVG_LENGTH_ANGSTROM,
        hb_volume_per_bond=hb_volume_per_bond,
        total_hb_volume=potential * hb_volume_per_bond,
    )


def has_primary_hydrogen_bonds(molecules: Iterable[Molecule]) -> bool:
    """System-level check (ascec-v04.py line 4358).

    True iff the system contains at least one H donor *and* one acceptor.
    """
    total_donors = 0
    total_acceptors = 0
    for mol in molecules:
        for z in mol.atomic_numbers:
            symbol = Z_TO_SYMBOL.get(z, "")
            if symbol == "H":
                total_donors += 1
            elif symbol in _HB_ACCEPTOR_SYMBOLS:
                total_acceptors += 1
    return total_donors >= 1 and total_acceptors >= 1


# --------------------------------------------------------------------------- #
# Top-level entry point                                                       #
# --------------------------------------------------------------------------- #


def calculate_optimal_box_length(
    molecules: Sequence[Molecule],
    *,
    target_packing_fractions: Optional[Sequence[float]] = None,
) -> BoxLengthReport:
    """Compute box-length recommendations using v04's two-method scheme.

    Args:
        molecules: The molecules that will be placed in the cube — one entry
            per *instance* (v04's ``state.molecules_to_add``), not per unique
            template, since a system of two identical waters counts the volume
            and extent twice.
        target_packing_fractions: Defaults to v04's table 5–50 %.

    Returns:
        :class:`BoxLengthReport` containing the per-molecule breakdown and a
        recommendation dict keyed by percentage string (``"15.0%"`` etc.) for
        the v04-format writer in Session 5.
    """
    fractions: Tuple[float, ...] = (
        tuple(target_packing_fractions)
        if target_packing_fractions is not None
        else _DEFAULT_PACKING_FRACTIONS
    )

    if not molecules:
        # v04 returns an error dict; we surface as an empty report so callers
        # can still introspect num_molecules == 0.
        return BoxLengthReport(
            num_molecules=0,
            method="B_diagonal_extent",
            has_primary_hbonds=False,
            individual_molecular_volumes=tuple(),
            total_molecular_volume=0.0,
            total_hb_network_volume=0.0,
            total_extent_sum=0.0,
            diagonal_box_length=0.0,
            diagonal_derived_volume=0.0,
            total_effective_volume=0.0,
            max_molecular_extent=0.0,
            recommendations={},
        )

    system_has_hbonds = has_primary_hydrogen_bonds(molecules)
    method = "A_hb_volume" if system_has_hbonds else "B_diagonal_extent"

    per_mol_volumes: List[float] = []
    per_mol_extents: List[float] = []
    per_mol_hb: List[HBondAnalysis] = []
    individual_rows: List[Dict[str, object]] = []

    for mol in molecules:
        volume = calculate_molecular_volume(mol)
        extent = calculate_molecular_extent(mol)
        hb = calculate_hydrogen_bond_potential(mol)
        per_mol_volumes.append(volume)
        per_mol_extents.append(extent)
        per_mol_hb.append(hb)
        individual_rows.append({
            "molecule_label": mol.label,
            "num_atoms": mol.num_atoms,
            "volume_A3": volume,
            "extent_A": extent,
            "hb_donors": hb.donors,
            "hb_acceptors": hb.acceptors,
            "potential_hb_bonds": hb.potential_bonds,
            "hb_network_volume_A3": hb.total_hb_volume,
        })

    total_molecular_volume = float(sum(per_mol_volumes))
    total_hb_network_volume = float(sum(h.total_hb_volume for h in per_mol_hb))
    total_extent_sum = float(sum(per_mol_extents))

    diagonal_box_length = total_extent_sum / math.sqrt(3.0)
    diagonal_derived_volume = diagonal_box_length ** 3

    if system_has_hbonds:
        total_effective_volume = total_molecular_volume + total_hb_network_volume
    else:
        total_effective_volume = diagonal_derived_volume

    recommendations: Dict[str, BoxLengthRecommendation] = {}
    if total_effective_volume > 0:
        for phi in fractions:
            required_volume = total_effective_volume / phi
            box_length = required_volume ** (1.0 / 3.0)
            recommendations[f"{phi:.1%}"] = BoxLengthRecommendation(
                packing_fraction=phi,
                box_length_angstrom=box_length,
                box_volume=required_volume,
                free_volume=required_volume - total_effective_volume,
                free_volume_fraction=1.0 - (total_effective_volume / required_volume),
                molecular_volume_fraction=total_molecular_volume / required_volume,
                hb_network_volume_fraction=total_hb_network_volume / required_volume,
            )

    max_extent = max(per_mol_extents) if per_mol_extents else 0.0

    return BoxLengthReport(
        num_molecules=len(molecules),
        method=method,
        has_primary_hbonds=system_has_hbonds,
        individual_molecular_volumes=tuple(individual_rows),
        total_molecular_volume=total_molecular_volume,
        total_hb_network_volume=total_hb_network_volume,
        total_extent_sum=total_extent_sum,
        diagonal_box_length=diagonal_box_length,
        diagonal_derived_volume=diagonal_derived_volume,
        total_effective_volume=total_effective_volume,
        max_molecular_extent=max_extent,
        recommendations=recommendations,
    )


def _iter_atoms(molecule: Molecule):
    for z, row in zip(molecule.atomic_numbers, molecule.coords):
        yield z, (float(row[0]), float(row[1]), float(row[2]))


__all__ = [
    "BoxLengthRecommendation",
    "BoxLengthReport",
    "HBondAnalysis",
    "calculate_hydrogen_bond_potential",
    "calculate_molecular_extent",
    "calculate_molecular_volume",
    "calculate_optimal_box_length",
    "has_primary_hydrogen_bonds",
]
