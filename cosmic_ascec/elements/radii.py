"""Atomic-radius lookup used during initial geometry placement.

Ported from v04's `get_radius` (ascec-v04.py line 369). Rule: monatomic species
use the Bondi/Alvarez van-der-Waals radius; everything else uses the Cordero
covalent radius. Defaults match v04 exactly so the box-length calculator and
overlap checker stay parity-clean.

References for the radius tables (in cosmic_ascec.elements.data):
  Covalent radii:
    B. Cordero et al., "Covalent radii revisited," Dalton Trans. 2008,
    2832-2838. doi:10.1039/B801115J
  Van der Waals radii:
    A. Bondi, J. Phys. Chem. 1964, 68(3), 441-451. doi:10.1021/j100785a001
    S. Alvarez, Dalton Trans. 2013, 42, 8617-8636. doi:10.1039/C3DT50599E
    (main-group extensions also match M. Mantina et al., J. Phys. Chem. A
     2009, 113, 5806-5812. doi:10.1021/jp8111556)

The fallback constants below are NOT literature values: they are generic
placeholders for elements absent from the tables. After the covalent table
was extended with Cordero (2008) through Cm, the covalent default (1.50 A)
only fires for Z >= 97; the vdW default fires for Z >= 87 (Alvarez 2013
publishes only sparse Z >= 87 vdW radii, so the table is not extended there).
These placeholders match v04 verbatim.
"""

from __future__ import annotations

from cosmic_ascec.elements.data import COVALENT_RADII, VDW_RADII

_COVALENT_DEFAULT = 1.50
_VDW_FALLBACK_SCALE = 1.2
_VDW_HARD_DEFAULT = 1.70


def get_radius(atomic_number: int, *, monatomic: bool = False) -> float:
    """Return the radius (Å) appropriate for box-length and overlap checks.

    Args:
        atomic_number: Element Z.
        monatomic: True when this Z represents a single-atom species (an atom
            or atomic ion) — then we use the VDW radius. Polyatomic molecules
            use the covalent radius. v04 inferred this from
            ``len(mol_def.atoms_coords) == 1``; passing a flag keeps the
            geometry module decoupled from data structures.

    The fallback chain for an unknown ``Z`` matches v04: VDW lookup falls back
    to ``r_atom[Z] * 1.2`` then to 1.70 Å; covalent lookup falls back to 1.50 Å.
    """
    if monatomic:
        if atomic_number in VDW_RADII:
            return VDW_RADII[atomic_number]
        covalent = COVALENT_RADII.get(atomic_number)
        if covalent is not None:
            return covalent * _VDW_FALLBACK_SCALE
        return _VDW_HARD_DEFAULT
    return COVALENT_RADII.get(atomic_number, _COVALENT_DEFAULT)


__all__ = ["get_radius"]
