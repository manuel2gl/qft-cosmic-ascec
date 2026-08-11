"""Frozen geometric structures used through the geometry/MC/annealing stack.

Two dataclasses live here:

* :class:`Molecule` — one rigid template: atomic numbers and reference
  coordinates. Built once per unique molecule in the ``.asc`` input and
  reused for every placement.
* :class:`Cluster` — a full placed system: every atom from every molecule,
  in one flat array, plus per-molecule slice boundaries.

Both are ``frozen=True`` and their coordinate arrays are marked read-only
(``setflags(write=False)``) on construction, so a caller that holds a
reference cannot mutate the geometry seen by another caller. The MC moves
build *new* :class:`Cluster` instances rather than mutating in place —
keeps the engine easy to reason about and lets callbacks store the
"accepted at step N" cluster without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from cosmic_ascec.exceptions import GeometryError
from cosmic_ascec.file_formats.asc_schema import MoleculeSpec


@dataclass(frozen=True)
class Molecule:
    """A single molecular template: atom kinds and reference coordinates.

    Construct via :meth:`from_spec` from a parsed :class:`MoleculeSpec`; that
    keeps the boundary parsing / numeric paths separate. ``coords`` is an
    ``(N, 3)`` float64 array in Angstroms; ``atomic_numbers`` is a tuple of
    integer ``Z`` values, length ``N``.

    ``is_monatomic`` is the same flag v04 inferred from
    ``len(mol_def.atoms_coords) == 1`` (ascec-v04.py line 383); plumbing it
    through here lets the radius lookup stay agnostic of the dataclass.

    ``frozen`` comes from a ``*$``-delimited block in the ``.asc`` file. A
    frozen molecule is placed at ``coords`` verbatim and is skipped by every
    Monte Carlo move, so it holds still while the rest of the system anneals
    around it.
    """

    label: str
    atomic_numbers: Tuple[int, ...]
    coords: np.ndarray  # (N, 3) float64, read-only
    frozen: bool = False

    def __post_init__(self) -> None:
        # Mutating the write-flag is in-place, so it works on a frozen dataclass.
        self.coords.setflags(write=False)

    @classmethod
    def from_spec(cls, spec: MoleculeSpec) -> "Molecule":
        if spec.num_atoms == 0:
            raise GeometryError(f"molecule {spec.label!r} has zero atoms")
        zs = tuple(atom[0] for atom in spec.atoms)
        coords = np.asarray(
            [[atom[1], atom[2], atom[3]] for atom in spec.atoms],
            dtype=np.float64,
        )
        return cls(
            label=spec.label,
            atomic_numbers=zs,
            coords=coords,
            frozen=getattr(spec, "frozen", False),
        )

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def is_monatomic(self) -> bool:
        return self.num_atoms == 1


@dataclass(frozen=True)
class Cluster:
    """A placed system of one or more molecules inside the simulation cube.

    Attributes:
        coords: ``(natom, 3)`` float64, read-only. v04 calls this ``state.rp``
            (ascec-v04.py line 562).
        atomic_numbers: Tuple of integer Z values, length ``natom``.
            v04 calls this ``state.iznu`` (line 566).
        molecule_offsets: ``(num_molecules + 1,)`` int array; the atoms of
            molecule ``i`` are ``coords[offsets[i]:offsets[i+1]]``. v04 calls
            this ``state.imolec`` (line 565).
        molecules: Tuple of :class:`Molecule` templates indexed *parallel to*
            ``molecule_offsets`` — entry ``i`` is the template that was placed
            as molecule ``i`` in the box.
        box_length: Cube edge in Angstroms (v04's ``state.xbox`` / ``cube_length``).
    """

    coords: np.ndarray
    atomic_numbers: Tuple[int, ...]
    molecule_offsets: np.ndarray
    molecules: Tuple[Molecule, ...]
    box_length: float

    def __post_init__(self) -> None:
        self.coords.setflags(write=False)
        self.molecule_offsets.setflags(write=False)

    @classmethod
    def from_blocks(
        cls,
        molecules: Sequence[Molecule],
        per_molecule_coords: Sequence[np.ndarray],
        box_length: float,
    ) -> "Cluster":
        """Stack a list of per-molecule coordinate blocks into a single Cluster.

        ``per_molecule_coords[i]`` is an ``(n_i, 3)`` array (typically the
        result of placing :class:`molecules` ``[i]`` in the box). Atomic numbers
        are taken from the molecule templates; nothing is recomputed from
        coordinates.
        """
        if len(molecules) != len(per_molecule_coords):
            raise GeometryError(
                "Cluster.from_blocks: got "
                f"{len(per_molecule_coords)} coordinate blocks for "
                f"{len(molecules)} molecules"
            )
        for i, (mol, block) in enumerate(zip(molecules, per_molecule_coords)):
            if block.shape != (mol.num_atoms, 3):
                raise GeometryError(
                    f"Cluster.from_blocks: molecule {i} ({mol.label!r}) expected "
                    f"({mol.num_atoms}, 3) coords, got {block.shape}"
                )

        offsets = np.zeros(len(molecules) + 1, dtype=np.int64)
        for i, mol in enumerate(molecules):
            offsets[i + 1] = offsets[i] + mol.num_atoms
        natom = int(offsets[-1])

        coords = np.empty((natom, 3), dtype=np.float64)
        z_list: list[int] = []
        for i, mol in enumerate(molecules):
            coords[offsets[i] : offsets[i + 1], :] = per_molecule_coords[i]
            z_list.extend(mol.atomic_numbers)

        return cls(
            coords=coords,
            atomic_numbers=tuple(z_list),
            molecule_offsets=offsets,
            molecules=tuple(molecules),
            box_length=float(box_length),
        )

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def num_molecules(self) -> int:
        return len(self.molecules)

    def atoms_for(self, molecule_index: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Return ``(coords_slice, atomic_numbers_tuple)`` for a single molecule."""
        start = int(self.molecule_offsets[molecule_index])
        end = int(self.molecule_offsets[molecule_index + 1])
        return self.coords[start:end], tuple(self.atomic_numbers[start:end])

    def with_molecule(self, molecule_index: int, new_coords: np.ndarray) -> "Cluster":
        """Return a new Cluster whose molecule ``molecule_index`` has ``new_coords``.

        Per **D-004** Monte Carlo moves never mutate in place; they propose a
        fresh :class:`Cluster`. This helper avoids re-stacking every block when
        only one molecule changed. The replacement must have the same atom
        count as the original block; identities (``molecules`` tuple,
        ``atomic_numbers``, ``box_length``) are preserved.
        """
        start = int(self.molecule_offsets[molecule_index])
        end = int(self.molecule_offsets[molecule_index + 1])
        expected_shape = (end - start, 3)
        new_coords = np.asarray(new_coords, dtype=np.float64)
        if new_coords.shape != expected_shape:
            raise GeometryError(
                f"Cluster.with_molecule: expected shape {expected_shape} for "
                f"molecule {molecule_index}, got {new_coords.shape}"
            )
        coords = np.array(self.coords, copy=True)
        coords[start:end, :] = new_coords
        return Cluster(
            coords=coords,
            atomic_numbers=self.atomic_numbers,
            molecule_offsets=np.array(self.molecule_offsets, copy=True),
            molecules=self.molecules,
            box_length=self.box_length,
        )


__all__ = ["Cluster", "Molecule"]
