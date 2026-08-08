"""Carve a solute plus its solvation shell out of an MD trajectory.

A solvated trajectory cannot be clustered as it stands. A box of bisphenol A
in water is 22077 atoms per frame, and the descriptors COSMIC builds are
pairwise, so the cost grows with the square of that. Almost all of it is bulk
solvent that says nothing about the solute's conformation.

This module streams such a trajectory frame by frame -- never holding more than
one frame in memory -- and writes a small multi-frame XYZ holding the solute
and only the solvent nearest to it. The result is an ordinary multi-frame XYZ,
which :func:`~cosmic_ascec.clustering.features.xyz_input.explode_multiframe_xyz`
already knows how to split, so the rest of the pipeline needs no notion of a
trajectory at all.

Two ways to decide "nearest", because they answer different questions:

* a **cutoff** keeps every solvent molecule that comes within *R* angstrom of
  the solute. Physically honest, but the count varies from frame to frame, so
  frames differ in composition as well as in geometry.
* a **fixed count** keeps the *N* closest solvent molecules. Every frame gets
  the same formula and the same atom count, which is what makes frames
  directly comparable -- and it is what the RMSD pass requires, since
  :func:`~cosmic_ascec.clustering.rmsd.calculate_rmsd` returns ``None`` when
  the two structures differ in size.

How a frame is reduced
----------------------

1. The solute's centroid is found and the box is re-centred on it.
2. Every solvent molecule is stitched whole and translated, as a rigid unit, by
   whole lattice vectors into the periodic image beside that centroid. Without
   this a molecule 3 angstrom from the solute across a periodic face would be
   written out 57 angstrom away, and every descriptor derived from it would be
   wrong.
3. The frame is now contiguous in space, so distances are ordinary Euclidean
   ones and a single :class:`scipy.spatial.cKDTree` query over the solute
   answers "how far is each solvent atom from the solute" for the whole frame.
   That is what lets this scale from a 33-atom solute to a protein.

Step 2 is exact for any box shape, triclinic included, which matters because
GROMACS' default box for a solvated protein is a rhombic dodecahedron. It is
valid while the solute's half-extent plus the shell radius stays inside half
the box width -- true of any correctly set-up simulation, since the solute must
not see its own periodic image. :func:`extract_shell` checks it per frame
rather than emitting a quietly wrong shell.

Public API:

* :class:`ShellSpec` — what to extract.
* :func:`extract_shell` — do it, returning a :class:`ShellReport`.
* :class:`Box` — a periodic cell of any shape.
* :func:`open_trajectory` — a reader for a ``.pdb`` / ``.gro`` / ``.xyz`` path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from cosmic_ascec.clustering.features.xyz_input import _frame_stem
from cosmic_ascec.exceptions import TrajectoryError
from cosmic_ascec.file_formats.provenance import (
    MAPPING_FILENAME,
    FrameRecord,
    write_mapping,
)

# Symbols accepted when inferring an element from an atom name. Covers what
# turns up in solvated organic systems; anything outside it has to be named
# explicitly through ShellSpec.element_overrides.
KNOWN_ELEMENTS = frozenset({
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Br",
    "Ru", "Rh", "Pd", "Ag", "I", "Pt", "Au", "Hg",
})

# Atom names whose two-letter reading is a real element but whose intended
# meaning inside a molecule is almost always the one-letter one: CA is an alpha
# carbon far more often than it is calcium.
TWO_LETTER_TRAPS = frozenset({
    "CA", "CB", "CG", "CD", "CE", "CZ", "CH",
    "NE", "ND", "NB", "NZ", "SE", "SD", "SG",
    "HG", "HE", "HD", "HZ", "HB",
})

# Names that are a real element *and* a plausible force-field type, where both
# readings turn up in solvated systems and the name alone cannot settle it: BA
# is barium or an aromatic carbon, IN is indium or AMBER's nitrogen type, NI is
# nickel or a nitrogen. Guessing corrupts every descriptor built from the atom,
# so these are refused and the user is asked.
#
# Deliberately short. The far larger set of two-letter types whose element
# reading is an exotic metal (OS, CF, PM, ...) keeps the organic reading, since
# nobody is running MD on osmium dissolved in water and the resolved table is
# printed at the start of every run for checking.
AMBIGUOUS_NAMES = frozenset({"BA", "CS", "SR", "SN", "PB", "SC", "IN", "NI", "RB"})

# Virtual interaction sites: massless points that carry charge but no nucleus.
# TIP4P water puts one ('MW') on the HOH bisector, TIP5P uses two lone pairs,
# Drude models add a polarisability particle. They have no element, contribute
# nothing a descriptor can use, and must not reach the written geometry — but
# they are common enough that refusing to read the file is the wrong answer.
VIRTUAL_SITE_NAMES = frozenset({
    "MW", "MM", "MN", "MP", "EP", "EPW", "EP1", "EP2", "LP", "LP1", "LP2",
    "DUM", "DU", "DRUD", "DW",
})

# ...and the same idea written positionally: a bare M or D carrying only digits
# ('M1', 'D2') is a dummy in every force field that uses the convention.
_VIRTUAL_SITE_PATTERN = re.compile(r"^[MD]\d*$")

# Tolerance for the --verify self-check, in angstrom. Re-imaging is arithmetic
# on whole lattice vectors, so agreement should be at float precision; anything
# above this means a molecule was translated wrongly.
_VERIFY_TOL = 1e-6

# GROMACS works in nanometres, the rest of this pipeline in angstrom.
_NM_TO_ANGSTROM = 10.0


# --------------------------------------------------------------------------
# Periodic cell
# --------------------------------------------------------------------------

class Box:
    """A periodic cell of any shape.

    Held as a 3x3 matrix whose *rows* are the lattice vectors a, b, c, so a
    cartesian displacement ``r`` has fractional coordinates ``r @ inv(H)``.

    Rectangular cells keep a separate fast path. That is not only for speed:
    ``cos(90 degrees)`` is 6.1e-17 rather than 0 in floating point, so routing
    an orthorhombic box through the general matrix would perturb every
    coordinate in the last bits. Cells that are exactly rectangular are handled
    with the same arithmetic they always were.
    """

    __slots__ = ("matrix", "_inverse", "lengths", "orthorhombic")

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        off_diagonal = self.matrix - np.diag(np.diag(self.matrix))
        self.orthorhombic = not off_diagonal.any()
        self.lengths = np.diag(self.matrix).copy() if self.orthorhombic else None
        self._inverse = None if self.orthorhombic else np.linalg.inv(self.matrix)

        if np.any(np.diag(self.matrix) <= 0) or abs(np.linalg.det(self.matrix)) < 1e-12:
            raise TrajectoryError(f"degenerate periodic box:\n{self.matrix}")

    @classmethod
    def from_edges(cls, lengths: Sequence[float]) -> "Box":
        """A rectangular cell from one or three edge lengths."""
        edges = list(lengths)
        if len(edges) == 1:
            edges = edges * 3
        if len(edges) != 3:
            raise TrajectoryError("a box is given as one edge length or three")
        return cls(np.diag(np.asarray(edges, dtype=float)))

    @classmethod
    def from_parameters(cls, a: float, b: float, c: float,
                        alpha: float, beta: float, gamma: float) -> "Box":
        """A cell from lengths and angles, the CRYST1 convention.

        Angles are in degrees. Exactly 90-degree angles short-circuit to a
        diagonal matrix so a rectangular cell is bit-for-bit what it was
        before triclinic support existed.
        """
        if alpha == 90.0 and beta == 90.0 and gamma == 90.0:
            return cls.from_edges((a, b, c))

        ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
        sg = math.sin(math.radians(gamma))
        if abs(sg) < 1e-9:
            raise TrajectoryError(f"degenerate cell angle gamma={gamma}")

        cz_squared = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        if cz_squared <= 0.0:
            raise TrajectoryError(
                f"cell angles {alpha}/{beta}/{gamma} do not describe a real cell"
            )
        return cls(np.array([
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, c * (ca - cb * cg) / sg, c * math.sqrt(cz_squared) / sg],
        ]))

    def minimum_image(self, delta: np.ndarray) -> np.ndarray:
        """Wrap displacement vectors into the nearest periodic image."""
        if self.orthorhombic:
            return delta - self.lengths * np.round(delta / self.lengths)
        fractional = delta @ self._inverse
        return delta - np.round(fractional) @ self.matrix

    @property
    def min_width(self) -> float:
        """Smallest perpendicular distance between opposite faces.

        For a rectangular cell this is the shortest edge. For a triclinic one
        it is smaller than the shortest lattice vector, and it is the width
        that actually limits the minimum-image convention — which is why the
        validity guard uses it rather than a lattice length.
        """
        if self.orthorhombic:
            return float(self.lengths.min())
        volume = abs(np.linalg.det(self.matrix))
        a, b, c = self.matrix
        faces = [np.cross(b, c), np.cross(c, a), np.cross(a, b)]
        return float(min(volume / np.linalg.norm(f) for f in faces))

    def __repr__(self) -> str:
        if self.orthorhombic:
            lx, ly, lz = self.lengths
            return f"Box(rectangular, {lx:.3f} x {ly:.3f} x {lz:.3f} A)"
        return f"Box(triclinic, min width {self.min_width:.3f} A)"


# --------------------------------------------------------------------------
# What to extract
# --------------------------------------------------------------------------

@dataclass
class ShellSpec:
    """The extraction request.

    Exactly one of *cutoff* and *count* must be set: *cutoff* keeps everything
    within that many angstrom, *count* keeps that many nearest molecules.
    """

    output: Path
    cutoff: Optional[float] = None
    count: Optional[int] = None

    solute_resnames: Sequence[str] = ()
    solute_indices: Optional[str] = None
    solvent_resnames: Sequence[str] = ()
    solvent_size: int = 3

    box: Optional[Sequence[float]] = None
    periodic: bool = True

    first: int = 0
    last: Optional[int] = None
    stride: int = 1

    order: str = "distance"
    element_overrides: Dict[str, str] = field(default_factory=dict)
    verify: bool = False
    quiet: bool = False
    command: str = ""      # recorded verbatim in mapping.dat's header

    def validate(self) -> None:
        if (self.cutoff is None) == (self.count is None):
            raise TrajectoryError("give exactly one of --shell (a radius) or --nearest (a count)")
        if self.cutoff is not None and self.cutoff <= 0:
            raise TrajectoryError("--shell needs a positive radius in angstrom")
        if self.count is not None and self.count < 0:
            raise TrajectoryError("--nearest cannot be negative")
        if self.stride < 1:
            raise TrajectoryError("--stride needs a positive value")
        if self.solvent_size < 1:
            raise TrajectoryError("--solvent-size needs a positive value")
        if self.order not in ("distance", "index"):
            raise TrajectoryError("--order must be 'distance' or 'index'")
        if self.last is not None and self.last < self.first:
            raise TrajectoryError(f"--last {self.last} is before --first {self.first}")


@dataclass
class ShellReport:
    """What the extraction produced, for printing and for callers to assert on."""

    output: Path
    frames_written: int
    solute_atoms: int
    solvent_molecules: List[int]
    solvent_atom_counts: List[int]
    total_solvent_available: int
    single_species: bool
    worst_internal_deviation: Optional[float] = None
    worst_distance_deviation: Optional[float] = None
    frames: List["FrameRecord"] = field(default_factory=list)
    mapping_path: Optional[Path] = None

    @property
    def atom_counts(self) -> Tuple[int, int]:
        """Smallest and largest atom count written."""
        return (self.solute_atoms + min(self.solvent_atom_counts),
                self.solute_atoms + max(self.solvent_atom_counts))

    @property
    def constant_size(self) -> bool:
        """True when every frame has the same atom count, as ``--rmsd`` requires."""
        lo, hi = self.atom_counts
        return lo == hi

    @property
    def solute_only(self) -> bool:
        """True when no solvent was kept at all (``--nearest=0``)."""
        return not any(self.solvent_atom_counts)

    @property
    def verified(self) -> bool:
        """True when the self-check ran and every frame passed it."""
        if self.worst_internal_deviation is None:
            return False
        return (self.worst_internal_deviation <= _VERIFY_TOL
                and (self.worst_distance_deviation or 0.0) <= _VERIFY_TOL)


# --------------------------------------------------------------------------
# Frames and topology
# --------------------------------------------------------------------------

#: ``t=`` / ``step=`` as GROMACS writes them into a PDB TITLE or a .gro title.
_TIME_RE = re.compile(r"\bt\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")
_STEP_RE = re.compile(r"\bstep\s*=\s*(-?\d+)")


def parse_time_step(title: str) -> Tuple[Optional[float], Optional[int]]:
    """Simulation time (ps) and step number out of a frame's title line.

    ``gmx trjconv`` writes ``TITLE  <system> t= 10000.00000 step= 5000000`` per
    MODEL, and the same trailer into a ``.gro`` title. That is the only record
    of where a frame sits in the dynamics, and it is what makes a motif
    traceable back to a point in the simulation. Absent values stay ``None``
    rather than being invented.
    """
    if not title:
        return None, None
    time_match = _TIME_RE.search(title)
    step_match = _STEP_RE.search(title)
    return (float(time_match.group(1)) if time_match else None,
            int(step_match.group(1)) if step_match else None)


class Frame:
    """One snapshot: coordinates, the cell they were recorded in, and when."""

    __slots__ = ("index", "coords", "box", "time_ps", "step")

    def __init__(self, index: int, coords: np.ndarray, box: Optional[Box],
                 time_ps: Optional[float] = None, step: Optional[int] = None):
        self.index = index
        self.coords = coords
        self.box = box
        self.time_ps = time_ps
        self.step = step


class Topology:
    """Per-atom identity, read once from the first frame.

    Every frame of a trajectory holds the same atoms in the same order, so
    parsing names and residues once keeps the per-frame work down to the three
    coordinate columns — the difference between a 10-second extraction and a
    several-minute one on a 400 MB trajectory.
    """

    __slots__ = ("symbols", "names", "resnames", "resids", "is_real")

    def __init__(self, symbols: List[str], names: List[str],
                 resnames: List[str], resids: List[int]):
        self.symbols = symbols
        self.names = names
        self.resnames = resnames
        self.resids = resids
        # Virtual sites stay in the topology so atom indices keep matching the
        # file — '--solute=1-33' counts positions in the trajectory, and
        # silently renumbering them would make that selection mean something
        # else. They are excluded by this mask instead, wherever atoms are
        # grouped, measured or written.
        self.is_real = np.array([s != VIRTUAL_SITE for s in symbols], dtype=bool)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def n_virtual(self) -> int:
        return int((~self.is_real).sum())

    def virtual_site_names(self) -> Dict[str, int]:
        """``{atom name: count}`` for the dropped virtual sites."""
        counts: Dict[str, int] = {}
        for name, real in zip(self.names, self.is_real):
            if not real:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def residue_names(self) -> List[str]:
        """Distinct residue names, in the order they first appear."""
        seen: Dict[str, None] = {}
        for name in self.resnames:
            seen.setdefault(name, None)
        return list(seen)

    def element_table(self) -> List[Tuple[str, str, str]]:
        """``(resname, atom name, symbol)`` once per distinct name, for auditing.

        Printed at the start of a run so a wrong guess shows up immediately
        rather than as a puzzling descriptor an hour later.
        """
        seen = set()
        rows: List[Tuple[str, str, str]] = []
        for resname, name, symbol in zip(self.resnames, self.names, self.symbols):
            key = (resname, name)
            if key not in seen:
                seen.add(key)
                rows.append((resname, name, symbol))
        return rows


#: Returned by :func:`infer_element` for an atom that is not a nucleus at all.
VIRTUAL_SITE = "<virtual>"


def is_virtual_site(atom_name: str) -> bool:
    """True for a massless interaction site rather than a real atom."""
    key = atom_name.strip().upper()
    return key in VIRTUAL_SITE_NAMES or bool(_VIRTUAL_SITE_PATTERN.match(key))


def infer_element(atom_name: str, overrides: Dict[str, str]) -> str:
    """Resolve an atom name to an element symbol, or to :data:`VIRTUAL_SITE`.

    ``gmx trjconv`` leaves the PDB element columns (77-78) blank and a ``.gro``
    has no element field at all, so the atom name is usually all there is — and
    an atom name in an MD file is a *force-field type*, not a symbol. The order
    below is deliberate:

    1. an explicit override always wins, so no guess is ever unavoidable;
    2. known virtual sites resolve to :data:`VIRTUAL_SITE` and get dropped;
    3. leading digits are stripped, because PDB writes ``1HB`` as often as
       ``HB1``;
    4. the two-letter reading is preferred when it is a genuine element — CL
       and NA have to survive — except for the names in
       :data:`TWO_LETTER_TRAPS`, where the one-letter reading is meant;
    5. anything still unresolved raises rather than guesses. Names like ``IN``
       and ``BA`` are the reason: read as elements they are indium and barium,
       read as force-field types they are nitrogen and carbon, and there is no
       way to tell from the name alone. Silently picking one corrupts every
       descriptor downstream, so the user is asked instead.
    """
    key = atom_name.strip().upper()
    if key in overrides:
        return overrides[key]
    if is_virtual_site(key):
        return VIRTUAL_SITE

    # 'HB1' and '1HB' are the same hydrogen written by two different tools.
    match = re.match(r"\d*([A-Za-z]+)", key)
    if not match:
        raise TrajectoryError(
            f"cannot infer an element from atom name {atom_name!r}; "
            f"name it with --elements {key}=<symbol>"
        )
    letters = match.group(1)

    if letters in AMBIGUOUS_NAMES:
        raise TrajectoryError(
            f"atom name {atom_name!r} is ambiguous: it reads as the element "
            f"{letters.capitalize()}, but force fields also use it as a "
            f"{letters[0]} type. Say which with --elements {letters}="
            f"{letters.capitalize()} or --elements {letters}={letters[0]}"
        )

    if len(letters) >= 2 and letters[:2] not in TWO_LETTER_TRAPS:
        two = letters[:2].capitalize()
        if two in KNOWN_ELEMENTS:
            return two

    one = letters[0].upper()
    if one in KNOWN_ELEMENTS:
        return one
    raise TrajectoryError(
        f"cannot resolve atom name {atom_name!r} to an element: neither "
        f"{letters[:2].capitalize()!r} nor {one!r} is one. If it is a force-field type, "
        f"say what it is with --elements {key}=<symbol> (or --elements=@types.map "
        f"for a whole force field); if it is a dummy site, that name is not one "
        f"COSMIC recognises."
    )


def parse_element_overrides(spec: str) -> Dict[str, str]:
    """Read ``--elements``: inline ``NAME=SYM,...`` or ``@file`` for a whole map.

    A force field has hundreds of atom types, which is far more than belongs on
    a command line, so ``--elements=@types.map`` reads them from a file: one
    ``NAME SYMBOL`` (or ``NAME=SYMBOL``) pair per line, ``#`` comments ignored.
    """
    overrides: Dict[str, str] = {}
    if not spec:
        return overrides

    if spec.startswith("@"):
        path = Path(spec[1:]).expanduser()
        if not path.is_file():
            raise TrajectoryError(f"--elements=@{path}: no such file")
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace("=", " ").split()
            if len(parts) != 2:
                raise TrajectoryError(
                    f"{path}:{lineno}: expected 'NAME SYMBOL', got {raw.strip()!r}"
                )
            overrides[parts[0].upper()] = parts[1].capitalize()
        return overrides

    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        name, sep, symbol = item.partition("=")
        if not sep or not symbol:
            raise TrajectoryError(f"--elements entries look like NAME=SYMBOL, not {item!r}")
        overrides[name.upper()] = symbol.capitalize()
    return overrides


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

class TrajectoryReader:
    """Common interface over the trajectory formats.

    A reader exposes the topology once and then streams frames. Keeping the
    two apart is what lets a format be added — a binary backend via MDAnalysis,
    say — without the selection or clustering code changing at all.
    """

    #: Set by subclasses once the first frame has been parsed.
    _topology: Optional[Topology] = None

    def topology(self) -> Topology:
        raise NotImplementedError

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError


class PdbReader(TrajectoryReader):
    """Multi-MODEL PDB.

    The cell comes from the CRYST1 record preceding each MODEL, so an NPT
    trajectory — where the box breathes from frame to frame — gets the right
    cell for every frame instead of one cell applied to all of them.
    """

    def __init__(self, path: Path, overrides: Dict[str, str],
                 fallback_box: Optional[Box] = None):
        self.path = path
        self.overrides = overrides
        self.fallback_box = fallback_box
        self._first_atoms: List[str] = []
        self._first_box: Optional[Box] = None
        self._handle = path.open()
        self._read_first_frame()

    def _read_first_frame(self) -> None:
        box = self.fallback_box
        title = ""
        for line in self._handle:
            if line.startswith("CRYST1"):
                box = _parse_cryst1(line)
            elif line.startswith("TITLE"):
                title = line
            elif line.startswith(("ATOM", "HETATM")):
                self._first_atoms.append(line)
            elif line.startswith(("ENDMDL", "END")) and self._first_atoms:
                # Some writers separate frames with a bare END rather than
                # ENDMDL; both close a frame here. TER does not — it only ends
                # a chain, and a frame may hold several.
                break
        if not self._first_atoms:
            self._handle.close()
            raise TrajectoryError(f"no ATOM/HETATM records found in {self.path}")
        self._first_box = box
        self._first_title = title

        names = [ln[12:16].strip() for ln in self._first_atoms]
        resnames = [ln[17:20].strip() for ln in self._first_atoms]
        resids = [_parse_resid(ln[22:26]) for ln in self._first_atoms]
        # Columns 77-78 hold the element when the writer bothered to fill them
        # in; trjconv does not, so fall back to the atom name. A declared
        # element still has to pass the virtual-site check: TIP4P writers that
        # fill the column at all tend to put 'M' there, which is not an element.
        symbols = []
        for line, name in zip(self._first_atoms, names):
            declared = line[76:78].strip() if len(line) >= 78 else ""
            if declared.isalpha() and declared.capitalize() in KNOWN_ELEMENTS:
                symbols.append(declared.capitalize())
            else:
                symbols.append(infer_element(name, self.overrides))
        self._topology = Topology(symbols, names, resnames, resids)

    def topology(self) -> Topology:
        return self._topology

    def frames(self) -> Iterator[Frame]:
        natoms = len(self._first_atoms)
        try:
            yield Frame(0, _pdb_coords(self._first_atoms), self._first_box,
                        *parse_time_step(self._first_title))
            index = 1
            box = self._first_box
            title = self._first_title
            buffer: List[str] = []
            for line in self._handle:
                if line.startswith("CRYST1"):
                    box = _parse_cryst1(line)
                elif line.startswith("TITLE"):
                    # TITLE precedes its MODEL, so this is the next frame's.
                    title = line
                elif line.startswith(("ATOM", "HETATM")):
                    buffer.append(line)
                elif line.startswith(("ENDMDL", "END")) and buffer:
                    if len(buffer) != natoms:
                        raise TrajectoryError(
                            f"frame {index} has {len(buffer)} atoms, expected {natoms}; "
                            f"the trajectory does not keep a constant atom count"
                        )
                    yield Frame(index, _pdb_coords(buffer), box,
                                *parse_time_step(title))
                    buffer = []
                    index += 1
        finally:
            self._handle.close()


def _parse_cryst1(line: str) -> Box:
    """A :class:`Box` from a CRYST1 record, angles included."""
    try:
        return Box.from_parameters(
            float(line[6:15]), float(line[15:24]), float(line[24:33]),
            float(line[33:40]), float(line[40:47]), float(line[47:54]),
        )
    except ValueError:
        raise TrajectoryError(f"cannot read the cell from CRYST1 record:\n  {line.rstrip()}") from None


def _parse_resid(field_text: str) -> int:
    """Residue sequence number, tolerating the hybrid-36 overflow past 9999."""
    text = field_text.strip()
    try:
        return int(text)
    except ValueError:
        # Hybrid-36 and plain-hex overflow schemes both keep residues in one
        # contiguous run, and grouping only needs values that change at a
        # residue boundary, so any stable mapping does.
        return int(text, 36) if text else 0


def _pdb_coords(lines: Sequence[str]) -> np.ndarray:
    return np.array(
        [(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])) for ln in lines],
        dtype=float,
    )


class GroReader(TrajectoryReader):
    """Multi-frame GROMACS ``.gro``.

    Native GROMACS output, and its trailing box line carries full triclinic
    vectors, so a dodecahedral cell survives the round trip that CRYST1 angles
    only just manage. Coordinates are nanometres and converted on read.
    """

    def __init__(self, path: Path, overrides: Dict[str, str]):
        self.path = path
        self.overrides = overrides
        self._handle = path.open()
        self._first: Optional[Tuple[np.ndarray, Optional[Box]]] = None
        self._read_first_frame()

    def _read_frame(self, first_line: str) -> Tuple[List[str], np.ndarray, Optional[Box]]:
        count_line = self._handle.readline()
        try:
            natoms = int(count_line.split()[0])
        except (IndexError, ValueError):
            raise TrajectoryError(
                f"{self.path}: expected an atom count after the title line, got "
                f"{count_line.strip()!r}"
            ) from None
        lines = [self._handle.readline() for _ in range(natoms)]
        box_line = self._handle.readline()
        return lines, _gro_coords(lines), _parse_gro_box(box_line)

    def _read_first_frame(self) -> None:
        title = self._handle.readline()
        if not title:
            self._handle.close()
            raise TrajectoryError(f"{self.path} is empty")
        lines, coords, box = self._read_frame(title)

        names = [ln[10:15].strip() for ln in lines]
        resnames = [ln[5:10].strip() for ln in lines]
        resids = [int(ln[0:5]) for ln in lines]
        self._topology = Topology(
            [infer_element(n, self.overrides) for n in names], names, resnames, resids
        )
        self._first = (coords, box, title)
        self._natoms = len(lines)

    def topology(self) -> Topology:
        return self._topology

    def frames(self) -> Iterator[Frame]:
        try:
            coords, box, title = self._first
            yield Frame(0, coords, box, *parse_time_step(title))
            index = 1
            while True:
                title = self._handle.readline()
                if not title.strip():
                    break
                lines, coords, box = self._read_frame(title)
                if len(lines) != self._natoms:
                    raise TrajectoryError(
                        f"frame {index} has {len(lines)} atoms, expected {self._natoms}; "
                        f"the trajectory does not keep a constant atom count"
                    )
                yield Frame(index, coords, box, *parse_time_step(title))
                index += 1
        finally:
            self._handle.close()


def _gro_coords(lines: Sequence[str]) -> np.ndarray:
    return np.array(
        [(float(ln[20:28]), float(ln[28:36]), float(ln[36:44])) for ln in lines],
        dtype=float,
    ) * _NM_TO_ANGSTROM


def _parse_gro_box(line: str) -> Optional[Box]:
    """A :class:`Box` from a ``.gro`` box line.

    Three values give a rectangular cell; nine give a triclinic one, in
    GROMACS' order ``v1x v2y v3z v1y v1z v2x v2z v3x v3y``.
    """
    values = [float(v) * _NM_TO_ANGSTROM for v in line.split()]
    if len(values) == 3:
        return Box.from_edges(values)
    if len(values) == 9:
        v = values
        return Box(np.array([[v[0], v[3], v[4]],
                             [v[5], v[1], v[6]],
                             [v[7], v[8], v[2]]]))
    raise TrajectoryError(f"expected 3 or 9 box values in a .gro, got {len(values)}")


class XyzReader(TrajectoryReader):
    """Multi-frame XYZ.

    An XYZ carries neither a cell nor residues, so the box has to be supplied
    by the caller and the solute has to be given as explicit atom indices.
    """

    def __init__(self, path: Path, box: Optional[Box]):
        self.path = path
        self.box = box
        self._handle = path.open()

        header = self._handle.readline()
        if not header.strip():
            self._handle.close()
            raise TrajectoryError(f"{path} is empty")
        self._natoms = int(header.split()[0])
        self._handle.readline()  # comment line
        self._first_lines = [self._handle.readline() for _ in range(self._natoms)]

        symbols = [ln.split()[0] for ln in self._first_lines]
        self._topology = Topology(symbols, list(symbols), ["UNK"] * self._natoms,
                                  list(range(self._natoms)))

    def topology(self) -> Topology:
        return self._topology

    def frames(self) -> Iterator[Frame]:
        try:
            yield Frame(0, _xyz_coords(self._first_lines), self.box)
            index = 1
            while True:
                head = self._handle.readline()
                if not head.strip():
                    break
                count = int(head.split()[0])
                if count != self._natoms:
                    raise TrajectoryError(
                        f"frame {index} has {count} atoms, expected {self._natoms}; "
                        f"the trajectory does not keep a constant atom count"
                    )
                self._handle.readline()
                lines = [self._handle.readline() for _ in range(count)]
                yield Frame(index, _xyz_coords(lines), self.box)
                index += 1
        finally:
            self._handle.close()


def _xyz_coords(lines: Sequence[str]) -> np.ndarray:
    out = np.empty((len(lines), 3))
    for i, line in enumerate(lines):
        parts = line.split()
        out[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return out


#: Suffix -> whether that format carries residue names of its own.
HAS_RESIDUES = {".pdb": True, ".gro": True, ".xyz": False}


def open_trajectory(path: Path, overrides: Dict[str, str], box: Optional[Box],
                    periodic: bool) -> TrajectoryReader:
    """Pick a reader for *path* by its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return PdbReader(path, overrides, fallback_box=box)
    if suffix == ".gro":
        return GroReader(path, overrides)
    if suffix == ".xyz":
        if box is None and periodic:
            raise TrajectoryError(
                "an XYZ carries no box; pass --box L (the CRYST1 edge from the matching "
                "PDB) or --no-pbc. Feeding the .pdb or .gro directly is better: both "
                "carry a cell per frame, which matters for an NPT run"
            )
        return XyzReader(path, box)
    raise TrajectoryError(
        f"unsupported trajectory format {suffix!r}; expected .pdb, .gro or .xyz"
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def parse_index_spec(spec: str) -> np.ndarray:
    """Turn ``1-33,40,55-60`` into zero-based atom indices."""
    indices: List[int] = []
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            if "-" in chunk:
                lo, hi = chunk.split("-", 1)
                if int(hi) < int(lo):
                    raise TrajectoryError(f"atom range {chunk!r} ends before it starts")
                indices.extend(range(int(lo) - 1, int(hi)))
            else:
                indices.append(int(chunk) - 1)
        except ValueError:
            raise TrajectoryError(
                f"cannot read atom selection {chunk!r}; expected 1-based indices like 1-33,40"
            ) from None
    if not indices:
        raise TrajectoryError(f"empty atom selection: {spec!r}")
    if min(indices) < 0:
        raise TrajectoryError("atom indices are 1-based, so 0 and negative values are not valid")
    return np.array(sorted(set(indices)), dtype=int)


def molecules_by_residue(topology: Topology, eligible: np.ndarray) -> List[np.ndarray]:
    """Split *eligible* atoms into molecules using residue numbering.

    Grouped by contiguous runs of the same (residue id, residue name) rather
    than by the id alone, so the wrap-around a PDB suffers past residue 9999
    cannot silently fuse two molecules into one.

    Residue boundaries are read from every atom, but only real ones are kept:
    a molecule's virtual sites must not break the run that defines it, and must
    not appear in it either. A residue left with nothing real is dropped.
    """
    groups: List[List[int]] = []
    key = None
    for i in eligible:
        this = (topology.resids[i], topology.resnames[i])
        if this != key or not groups:
            groups.append([])
            key = this
        if topology.is_real[i]:
            groups[-1].append(int(i))
    return [np.array(g, dtype=int) for g in groups if g]


def molecules_fixed_size(topology: Topology, eligible: np.ndarray,
                         size: int) -> List[np.ndarray]:
    """Split *eligible* atoms into consecutive blocks of *size* atoms.

    *size* counts the atoms as they appear in the file, virtual sites included,
    since that is what a user reading the file would count. The sites are then
    dropped from each block.
    """
    if len(eligible) % size:
        raise TrajectoryError(
            f"{len(eligible)} solvent atoms is not a multiple of --solvent-size {size}; "
            f"check the --solute selection"
        )
    blocks = (eligible[i:i + size] for i in range(0, len(eligible), size))
    kept = [block[topology.is_real[block]] for block in blocks]
    return [block for block in kept if block.size]


def wrap_around(coords: np.ndarray, centre: np.ndarray, groups_flat: np.ndarray,
                group_first_atom: np.ndarray, group_of_atom: np.ndarray,
                box: Optional[Box]) -> np.ndarray:
    """Bring every solvent molecule into the periodic image beside *centre*.

    Each molecule is first stitched whole relative to its own first atom, then
    translated as a rigid unit by whole lattice vectors. Both steps are pure
    minimum-image arithmetic over ragged groups, done in two vectorised calls
    rather than a loop over molecules, so the cost does not depend on how many
    molecules the box holds.

    Once this has run the frame is contiguous around the solute and every
    distance is an ordinary Euclidean one.
    """
    atoms = coords[groups_flat]
    if box is None:
        return atoms

    gidx = group_of_atom[groups_flat]
    reference = coords[group_first_atom]                      # (n_groups, 3)
    stitched = box.minimum_image(atoms - reference[gidx])     # atom offsets within molecule
    shifted = centre + box.minimum_image(reference - centre)  # molecule origins, re-imaged
    return shifted[gidx] + stitched


def solute_extent(solute_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """Centroid of the solute and its half-extent (furthest atom from it)."""
    centroid = solute_xyz.mean(axis=0)
    radius = float(np.linalg.norm(solute_xyz - centroid, axis=1).max())
    return centroid, radius


def distances_to_solute(solute_xyz: np.ndarray,
                        solvent_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from every solvent atom to the nearest solute atom.

    One :class:`scipy.spatial.cKDTree` query rather than a dense
    ``n_solvent x n_solute`` array. The dense form is fine for a small solute
    and hopeless for a protein: 5000 solute atoms against 90000 solvent atoms
    is 450 million distances and several GB of temporaries per frame.

    Requires the frame to be contiguous already — see :func:`wrap_around`.
    """
    from scipy.spatial import cKDTree  # imported lazily, as elsewhere in COSMIC

    tree = cKDTree(solute_xyz)
    distance, nearest = tree.query(solvent_xyz, k=1)
    return distance, nearest


def reduce_to_molecules(atom_distance: np.ndarray, group_of_atom: np.ndarray,
                        groups_flat: np.ndarray, ngroups: int) -> np.ndarray:
    """Closest approach of each molecule, from its atoms' distances."""
    gidx = group_of_atom[groups_flat]
    best = np.full(ngroups, np.inf)
    np.minimum.at(best, gidx, atom_distance)
    return best


def _self_check(wrapped: np.ndarray, group_slices: Sequence[Tuple[int, int]],
                solute_xyz: np.ndarray, source: np.ndarray, groups: Sequence[np.ndarray],
                picked: np.ndarray, distances: np.ndarray,
                box: Optional[Box]) -> Tuple[float, float]:
    """Confirm the re-imaging preserved geometry, for ``--verify``.

    Two properties have to hold, and between them they catch every way the
    translation can go wrong:

    1. Each molecule's internal geometry after wrapping matches its
       minimum-image geometry in the source frame — it moved as a rigid unit
       rather than being torn apart.
    2. Each molecule's closest approach to the solute, measured with **no**
       periodicity applied, equals the distance that selected it — so the
       molecule landed in the right image.

    Returns the worst deviation of each, in angstrom.
    """
    worst_internal = 0.0
    worst_distance = 0.0

    for g, expected in zip(picked, distances):
        start, stop = group_slices[g]
        molecule = wrapped[start:stop]

        if stop - start > 1:
            original = source[groups[g]]
            reference = original - original[0]
            if box is not None:
                reference = box.minimum_image(reference)
            worst_internal = max(
                worst_internal, float(np.abs((molecule - molecule[0]) - reference).max())
            )

        delta = molecule[:, None, :] - solute_xyz[None, :, :]
        plain = float(np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).min())
        worst_distance = max(worst_distance, abs(plain - float(expected)))

    return worst_internal, worst_distance


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_xyz_frame(handle: TextIO, symbols: Sequence[str], coords: np.ndarray,
                    comment: str) -> None:
    handle.write(f"{len(symbols)}\n{comment}\n")
    for sym, (x, y, z) in zip(symbols, coords):
        handle.write(f"{sym:<2s} {x:14.6f} {y:14.6f} {z:14.6f}\n")


def write_pdb_frame(handle: TextIO, names: Sequence[str], resnames: Sequence[str],
                    resids: Sequence[int], coords: np.ndarray, index: int,
                    box: Optional[Box]) -> None:
    """PDB output, so the extraction can be loaded into VMD and checked by eye."""
    if box is not None and box.orthorhombic:
        lx, ly, lz = box.lengths
        handle.write(
            f"CRYST1{lx:9.3f}{ly:9.3f}{lz:9.3f}"
            f"{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n"
        )
    handle.write(f"MODEL     {index + 1:4d}\n")
    for i, (name, resname, resid, (x, y, z)) in enumerate(
        zip(names, resnames, resids, coords), start=1
    ):
        # Column-exact PDB: serial 7-11, name 13-16, altLoc 17, resName 18-20,
        # chainID 22, resSeq 23-26, coordinates from 31. The blank between the
        # name and the residue is altLoc; dropping it shifts every residue name
        # one column left, and a reader then sees "PA" where "BPA" was written.
        handle.write(
            f"ATOM  {i:5d} {name:<4s} {resname:>3s}  {resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        )
    handle.write("TER\nENDMDL\n")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def extract_shell(trajectory: Path, spec: ShellSpec) -> ShellReport:
    """Write *spec.output* holding the solute and its nearest solvent.

    Streams *trajectory*, so memory use is set by one frame, not by file size.
    """
    spec.validate()
    trajectory = Path(trajectory)
    if not trajectory.is_file():
        raise TrajectoryError(f"no such trajectory: {trajectory}")

    # --nearest=0 strips the solvent entirely and clusters the solute's own
    # conformations. Nothing about the solvent is then needed: not its
    # molecules, not the neighbour search, and not even a periodic cell, since
    # no distance is ever measured across one.
    solute_only = spec.count == 0

    cli_box = Box.from_edges(spec.box) if spec.box is not None else None
    reader = open_trajectory(trajectory, spec.element_overrides, cli_box,
                             spec.periodic and not solute_only)
    topology = reader.topology()
    has_residues = HAS_RESIDUES[trajectory.suffix.lower()]

    solute = _resolve_solute(topology, spec, has_residues)
    groups = [] if solute_only else _resolve_solvent(topology, solute, spec, has_residues)

    if spec.count is not None and spec.count > len(groups):
        raise TrajectoryError(
            f"--nearest {spec.count} but the system holds only {len(groups)} solvent molecules"
        )

    if not spec.quiet:
        _print_preamble(topology, solute, groups, spec, has_residues)

    report = _stream(reader, topology, solute, groups, cli_box, spec)

    # mapping.dat lands beside the extracted frames, which for a one-shot run
    # is the directory that gets clustered — so the clustering half of the
    # pipeline finds it without being told where to look.
    report.mapping_path = spec.output.parent / MAPPING_FILENAME
    write_mapping(str(report.mapping_path), report.frames,
                  source=str(trajectory), command=spec.command)

    if not spec.quiet:
        _print_report(report, spec)
    return report


def _resolve_solute(topology: Topology, spec: ShellSpec, has_residues: bool) -> np.ndarray:
    """Which atoms are the solute, from indices or from residue names.

    Selection is expressed over the file's own atom numbering — virtual sites
    included, so ``--solute=1-33`` means what a reader of the file would think
    — and the sites are filtered out afterwards.
    """
    if spec.solute_indices:
        solute = parse_index_spec(spec.solute_indices)
        if solute.max() >= topology.natoms:
            raise TrajectoryError(
                f"--solute selects atom {solute.max() + 1} but the system has "
                f"{topology.natoms} atoms"
            )
    elif not has_residues:
        raise TrajectoryError(
            "XYZ input has no residue names, so the solute must be given as atom "
            "indices, e.g. --solute=1-33"
        )
    else:
        wanted = {r.upper() for r in spec.solute_resnames} or {topology.resnames[0].upper()}
        solute = np.array(
            [i for i, r in enumerate(topology.resnames) if r.upper() in wanted], dtype=int
        )
        if solute.size == 0:
            raise TrajectoryError(
                f"no residue named {', '.join(sorted(wanted))} in this trajectory; "
                f"residues present: {', '.join(topology.residue_names())}"
            )

    solute = solute[topology.is_real[solute]]
    if solute.size == 0:
        raise TrajectoryError("the solute selection holds no real atoms, only virtual sites")
    return solute


def _resolve_solvent(topology: Topology, solute: np.ndarray, spec: ShellSpec,
                     has_residues: bool) -> List[np.ndarray]:
    """Split the eligible non-solute atoms into individual molecules.

    ``--solvent-resname`` narrows what counts as shell material. Without it,
    everything that is not solute is eligible — which is right for a pure
    solvent box and wrong the moment counter-ions are present, since the ions
    then compete for shell slots and make the molecule sizes ragged.
    """
    is_solute = np.zeros(topology.natoms, dtype=bool)
    is_solute[solute] = True
    eligible_mask = ~is_solute

    named_solvent = bool(spec.solvent_resnames)
    if named_solvent:
        if not has_residues:
            raise TrajectoryError(
                "--solvent-resname needs residue names, which an XYZ does not carry; "
                "use --solvent-size instead"
            )
        wanted = {r.upper() for r in spec.solvent_resnames}
        present = {r.upper() for r in topology.residue_names()}
        missing = wanted - present
        if missing:
            raise TrajectoryError(
                f"no residue named {', '.join(sorted(missing))} in this trajectory; "
                f"residues present: {', '.join(topology.residue_names())}"
            )
        selected = np.array([r.upper() in wanted for r in topology.resnames])
        eligible_mask &= selected

    eligible = np.nonzero(eligible_mask)[0]
    if eligible.size == 0:
        raise TrajectoryError(
            "the solute selection leaves no solvent behind; nothing to build a shell from"
        )

    if has_residues:
        groups = molecules_by_residue(topology, eligible)
    else:
        groups = molecules_fixed_size(topology, eligible, spec.solvent_size)

    sizes = {len(g) for g in groups}
    if len(sizes) > 1 and spec.count is not None and not named_solvent:
        raise TrajectoryError(
            f"the solvent is not one species (molecule sizes {sorted(sizes)}), so "
            f"--nearest cannot give every frame the same atom count. Name the solvent "
            f"with --solvent-resname to exclude ions and cosolvents, or use --shell"
        )
    return groups


def _stream(reader: TrajectoryReader, topology: Topology, solute: np.ndarray,
            groups: List[np.ndarray], cli_box: Optional[Box],
            spec: ShellSpec) -> ShellReport:
    """The frame loop: wrap, select, write."""
    # With --nearest=0 there are no molecules at all, so every solvent-derived
    # array is empty rather than absent. np.concatenate rejects an empty list,
    # which is the only special case the empty path needs.
    solute_only = not groups
    groups_flat = np.concatenate(groups) if groups else np.empty(0, dtype=int)
    group_first_atom = np.array([g[0] for g in groups], dtype=int)
    group_of_atom = np.zeros(topology.natoms, dtype=int)
    for g, atoms in enumerate(groups):
        group_of_atom[atoms] = g

    # Where each molecule's atoms land inside the wrapped solvent array, so a
    # molecule can be sliced out without re-deriving its extent every time.
    group_slices: List[Tuple[int, int]] = []
    cursor = 0
    for atoms in groups:
        group_slices.append((cursor, cursor + len(atoms)))
        cursor += len(atoms)

    solvent_symbols = [topology.symbols[i] for i in groups_flat]
    solute_symbols = [topology.symbols[i] for i in solute]
    # 'MEN:44' per molecule, so mapping.dat can name which ones were kept.
    group_labels = [f"{topology.resnames[g[0]]}:{topology.resids[g[0]]}" for g in groups]

    out_pdb = spec.output.suffix.lower() == ".pdb"
    counts: List[int] = []
    atom_counts: List[int] = []
    records: List[FrameRecord] = []
    worst_internal = 0.0 if spec.verify else None
    worst_distance = 0.0 if spec.verify else None
    written = 0

    with spec.output.open("w") as out:
        for frame in reader.frames():
            if frame.index < spec.first:
                continue
            if spec.last is not None and frame.index > spec.last:
                break
            if (frame.index - spec.first) % spec.stride:
                continue

            box = None if not spec.periodic else (cli_box or frame.box)
            solute_xyz = frame.coords[solute]
            centroid, half_extent = solute_extent(solute_xyz)

            if box is not None and 2.0 * half_extent > box.min_width:
                raise TrajectoryError(
                    f"frame {frame.index}: the solute spans {2 * half_extent:.1f} A, wider "
                    f"than the {box.min_width:.1f} A cell. It is almost certainly split "
                    f"across the periodic boundary — make it whole first with "
                    f"'gmx trjconv -pbc mol' (add '-center' to keep it centred)"
                )

            if solute_only:
                # Nothing to wrap and nothing to search: skip the whole
                # neighbour pass rather than running it over empty arrays.
                wrapped = np.empty((0, 3))
                distance = np.empty(0)
                picked = np.empty(0, dtype=int)
            else:
                wrapped = wrap_around(frame.coords, centroid, groups_flat,
                                      group_first_atom, group_of_atom, box)
                atom_distance, _ = distances_to_solute(solute_xyz, wrapped)
                distance = reduce_to_molecules(atom_distance, group_of_atom,
                                               groups_flat, len(groups))

                if spec.cutoff is not None:
                    picked = np.nonzero(distance <= spec.cutoff)[0]
                else:
                    picked = np.argpartition(distance, spec.count - 1)[:spec.count]
                picked = picked[np.argsort(distance[picked], kind="stable")]
                if spec.order == "index":
                    picked = np.sort(picked)

            if picked.size and box is not None:
                reach = float(distance[picked].max()) + half_extent
                if 2.0 * reach > box.min_width:
                    raise TrajectoryError(
                        f"frame {frame.index}: the shell reaches {reach:.1f} A from the "
                        f"solute centre, more than half the {box.min_width:.1f} A cell "
                        f"width. The selection would wrap onto itself; use a smaller "
                        f"--shell / --nearest, or a trajectory with more solvent padding"
                    )

            if spec.verify:
                internal, mismatch = _self_check(wrapped, group_slices, solute_xyz,
                                                 frame.coords, groups, picked,
                                                 distance[picked], box)
                worst_internal = max(worst_internal, internal)
                worst_distance = max(worst_distance, mismatch)

            pieces = [solute_xyz]
            symbols = list(solute_symbols)
            for g in picked:
                start, stop = group_slices[g]
                pieces.append(wrapped[start:stop])
                symbols += solvent_symbols[start:stop]
            coords = np.vstack(pieces)

            counts.append(int(picked.size))
            atom_counts.append(len(symbols) - len(solute))
            records.append(FrameRecord(
                ordinal=written + 1,
                stem="",  # assigned below, once the frame count is known
                traj_frame=frame.index,
                atoms=len(symbols),
                time_ps=frame.time_ps,
                step=frame.step,
                solvent=[group_labels[g] for g in picked],
            ))

            if out_pdb:
                names, resnames, resids = _label(topology, solute,
                                                 [groups[g] for g in picked])
                write_pdb_frame(out, names, resnames, resids, coords, written, box)
            else:
                outermost = float(distance[picked].max()) if picked.size else 0.0
                write_xyz_frame(out, symbols, coords,
                                _frame_comment(frame, picked.size, distance, picked))
            written += 1
            if not spec.quiet and written % 50 == 0:
                print(f"    {written} frames written", flush=True)

    if not written:
        spec.output.unlink(missing_ok=True)
        raise TrajectoryError(
            f"no frames matched --first {spec.first}"
            + (f" --last {spec.last}" if spec.last is not None else "")
            + f" --stride {spec.stride}"
        )

    _assign_stems(records, spec.output)
    return ShellReport(
        output=spec.output,
        frames_written=written,
        solute_atoms=int(solute.size),
        solvent_molecules=counts,
        solvent_atom_counts=atom_counts,
        total_solvent_available=len(groups),
        single_species=len({len(g) for g in groups}) <= 1,
        worst_internal_deviation=worst_internal,
        worst_distance_deviation=worst_distance,
        frames=records,
    )


def _frame_comment(frame: Frame, kept: int, distance: np.ndarray,
                   picked: np.ndarray) -> str:
    """Comment line for an extracted frame, carrying its place in the dynamics."""
    parts = [f"frame {frame.index}"]
    if frame.time_ps is not None:
        parts.append(f"t= {frame.time_ps:g} ps")
    if frame.step is not None:
        parts.append(f"step= {frame.step}")
    parts.append(f"{kept} solvent molecules")
    if picked.size:
        parts.append(f"outermost {float(distance[picked].max()):.2f} A")
    return "  ".join(parts)


def _assign_stems(records: List[FrameRecord], output: Path) -> None:
    """Name each record the way the clustering pipeline will name its file.

    The per-frame files do not exist yet — ``explode_multiframe_xyz`` creates
    them later — so the names are derived with its own ``_frame_stem`` helper
    rather than re-guessed here, which is what keeps mapping.dat pointing at
    files that actually turn up. A single-frame extraction is never exploded,
    so it keeps the output's own stem.
    """
    if len(records) == 1:
        records[0].stem = output.stem
        return
    width = len(str(len(records)))
    for record in records:
        record.stem = _frame_stem(output.stem, record.ordinal, width)


def _label(topology: Topology, solute: np.ndarray,
           chosen: Sequence[np.ndarray]) -> Tuple[List[str], List[str], List[int]]:
    """Atom names, residue names and fresh residue ids for the written cluster."""
    names = [topology.names[i] for i in solute]
    resnames = [topology.resnames[i] for i in solute]
    resids = [1] * len(solute)
    for n, atoms in enumerate(chosen, start=2):
        names += [topology.names[i] for i in atoms]
        resnames += [topology.resnames[i] for i in atoms]
        resids += [n] * len(atoms)
    return names, resnames, resids


def _print_preamble(topology: Topology, solute: np.ndarray, groups: List[np.ndarray],
                    spec: ShellSpec, has_residues: bool) -> None:
    if has_residues:
        residues = sorted({topology.resnames[i] for i in solute})
        n_residues = len({(topology.resids[i], topology.resnames[i]) for i in solute})
        label = f"{', '.join(residues)}; {n_residues} residue(s)"
    else:
        label = "selected by index"

    if not groups:
        mode = "the solute alone — no solvent at all"
        print(f"  System      {topology.natoms} atoms; solvent discarded")
    else:
        mode = (f"every solvent molecule within {spec.cutoff} A" if spec.cutoff is not None
                else f"the {spec.count} nearest solvent molecules")
        sizes = sorted({len(g) for g in groups})
        species = f"{sizes[0]} atoms" if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]} atoms"
        print(f"  System      {topology.natoms} atoms, {len(groups)} solvent molecules "
              f"of {species}")
    print(f"  Solute      {solute.size} atoms ({label})")

    if topology.n_virtual:
        dropped = ", ".join(f"{name} x{count}"
                            for name, count in sorted(topology.virtual_site_names().items()))
        print(f"  Dropped     {topology.n_virtual} virtual sites ({dropped}) — massless, "
              f"no element")

    # An atom name in an MD file is a force-field type, and resolving it to an
    # element is a guess often enough to be worth showing. One line per residue
    # keeps it short while still covering every species.
    by_residue: Dict[str, Dict[str, str]] = {}
    for resname, name, symbol in topology.element_table():
        if symbol != VIRTUAL_SITE:
            by_residue.setdefault(resname, {}).setdefault(name, symbol)
    for resname, mapping in by_residue.items():
        elements = sorted({v for v in mapping.values()})
        shown = ", ".join(f"{k}->{v}" for k, v in list(mapping.items())[:6])
        print(f"  Elements    {resname:<5s} {''.join(elements):<10s} {shown}"
              f"{' ...' if len(mapping) > 6 else ''}")

    # Flushed so a failure on frame 0, written to stderr, cannot overtake the
    # preamble describing the selection that failed.
    print(f"  Keeping     {mode}, ordered by {spec.order}", flush=True)


def _print_report(report: ShellReport, spec: ShellSpec) -> None:
    counts = np.array(report.solvent_molecules)
    lo, hi = report.atom_counts
    print(f"\n  Wrote {report.frames_written} frames to '{report.output}'")
    if report.solute_only:
        print(f"  Solvent: none kept — clustering the solute's own conformations")
        print(f"  Atoms per frame: {lo}")
    else:
        print(f"  Solvent molecules per frame: min {counts.min()}  max {counts.max()}  "
              f"mean {counts.mean():.1f}")
        if report.constant_size:
            print(f"  Atoms per frame: {lo} (constant — the equal atom count "
                  f"--rmsd requires)")
        else:
            reason = ("the solvent is a mixture, so the count varies with which species "
                      "are nearest" if not report.single_species
                      else "--shell keeps a different number of molecules each frame")
            print(f"  Atoms per frame: {lo} to {hi} (variable — {reason}; --rmsd cannot "
                  f"compare frames of different size, use --nearest for that)")

    if report.mapping_path is not None:
        print(f"  Traceability: {report.mapping_path.name} "
              f"(frame -> trajectory frame, time, step, solvent kept)")

    if spec.verify:
        if report.verified:
            print(f"  Self-check: passed on all {report.frames_written} frames "
                  f"(molecules intact, every one re-imaged beside the solute)")
        else:
            print(f"  Self-check FAILED: worst molecule distortion "
                  f"{report.worst_internal_deviation:.2e} A, worst distance mismatch "
                  f"{report.worst_distance_deviation:.2e} A")
