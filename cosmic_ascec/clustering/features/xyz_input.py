"""Plain-XYZ input handling for the clustering pipeline.

COSMIC's two original input types are quantum-chemistry outputs (``*.out`` /
``*.log``). This module adds the third: a bare list of Cartesian coordinates.
For large systems that is the only thing available early — the geometry exists
long before a converged QM output does — and eight of the fifteen clustering
columns are pure functions of ``(atomic numbers, coordinates)``, so they can be
computed without any QM calculation at all.

Two responsibilities live here:

* :func:`read_xyz_frames` — parse a standard XYZ file into one or more frames.
  A single-structure file yields one frame; a concatenated ensemble or an MD
  trajectory yields many.
* :func:`explode_multiframe_xyz` — split any multi-frame file into one
  single-frame file per frame.

The explosion step is what keeps the rest of the pipeline untouched. COSMIC is
built on a one-file-one-record contract: the pickle cache is keyed by
``filename``, :mod:`~cosmic_ascec.clustering.filters` copies skipped structures
by looking their filename up on disk, motif folders are named after the source
file, and the ``--sp`` runner needs a real file to hand to xTB. Normalising
multi-frame input to single-frame files on the way in satisfies all of that at
once, instead of teaching each of those layers to address a frame index.

The ``_read_xyz_geometry`` / ``_read_last_xyz_frame`` closures inside
:func:`~cosmic_ascec.clustering.features.parsers.extract_properties_with_xtb`
are deliberately *not* refactored to call this module. They are part of the
verbatim cosmic-v01 port that the ``.out`` parity contract rests on; rewiring
them would risk that contract for no gain.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cosmic_ascec.elements import SYMBOL_TO_Z

# Files this module claims. Kept as a module constant so the CLI, the parser
# dispatch and the skipped-structure copier all agree on one spelling.
XYZ_GLOB = "*.xyz"
XYZ_EXTENSION = ".xyz"

# Directory (relative to the run's output base) that receives the normalised
# structure set — frames carved out of multi-frame files, and the files kept
# when an input/result pair had to be resolved.
FRAMES_SUBDIR = "xyz_frames"

# Optimisation *trajectories*: many frames tracing the path to a minimum, not
# candidate structures. Splitting one of these per frame would flood a run with
# hundreds of near-identical points, so they are rejected outright.
_XYZ_TRAJECTORY_SUFFIXES = (
    ".xtbtraj.xyz",
    "_trj.xyz",
)
_XYZ_TRAJECTORY_NAMES = frozenset({
    "xtbtraj.xyz",
    "xtbscreen.xyz",
})

# xTB's *optimised geometry* — a single structure, and normally the one worth
# clustering. It is deliberately NOT rejected: a directory of `.xtbopt.xyz`
# files collected out of an optimisation stage is exactly the right input. What
# it must not do is duplicate a structure, so when both ``<base>.xyz`` (the xTB
# input) and ``<base>.xtbopt.xyz`` (its result) sit in one directory, only the
# result is kept — see :func:`resolve_optimized_duplicates`.
XTB_OPTIMIZED_SUFFIX = ".xtbopt.xyz"
XTB_OPTIMIZED_BARE = "xtbopt.xyz"


@dataclass(frozen=True)
class XyzFrame:
    """One structure read out of an XYZ file.

    Attributes
    ----------
    index:
        1-based position of this frame within its source file.
    comment:
        The frame's comment line, verbatim and unparsed. COSMIC deliberately
        reads no energy out of it: the format is free-form, most producers put
        nothing meaningful there, and silently ranking structures on a scraped
        string is worse than declining to rank them. Use ``--sp`` to obtain
        real energies.
    atomnos:
        ``(N,)`` array of atomic numbers. Unrecognised element symbols map to
        ``0``, which the mass and repulsion routines treat as weightless.
    coords:
        ``(N, 3)`` array of Cartesian coordinates in Angstrom.
    """

    index: int
    comment: str
    atomnos: np.ndarray
    coords: np.ndarray

    @property
    def natoms(self) -> int:
        return int(self.atomnos.shape[0])


def is_xyz_byproduct(path: os.PathLike[str] | str) -> bool:
    """Return whether *path* is an optimisation trajectory rather than a structure.

    Only trajectories are rejected. An optimised geometry (``*.xtbopt.xyz``) is
    a perfectly good clustering candidate — usually the *best* one — and a
    folder full of them is a normal way to feed COSMIC the results of an
    optimisation stage. Duplicate structures are handled by pairing instead
    (:func:`resolve_optimized_duplicates`), not by filtering on the name.
    """
    lowered = os.path.basename(str(path)).lower()
    if lowered in _XYZ_TRAJECTORY_NAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in _XYZ_TRAJECTORY_SUFFIXES)


def is_xtb_optimized_name(path: os.PathLike[str] | str) -> bool:
    """Return whether *path* is an xTB optimised geometry."""
    name = os.path.basename(str(path))
    return name.endswith(XTB_OPTIMIZED_SUFFIX) or name == XTB_OPTIMIZED_BARE


def resolve_optimized_duplicates(
    paths: Sequence[os.PathLike[str] | str],
) -> Tuple[List[str], List[str]]:
    """Drop xTB *input* geometries that are superseded by their own result.

    xTB leaves ``<base>.xyz`` as the input and writes ``<base>.xtbopt.xyz`` as
    the optimised result. A directory holding both describes one structure
    twice — once before and once after optimisation — and clustering both would
    invent a conformer that does not exist. The optimised geometry wins.

    A directory containing only results, or only inputs, is left alone: there is
    nothing to disambiguate, and both are legitimate things to cluster.

    Returns ``(kept, superseded)``.
    """
    optimized_bases = set()
    for path in paths:
        directory, name = os.path.split(str(path))
        if name.endswith(XTB_OPTIMIZED_SUFFIX):
            optimized_bases.add((directory, name[: -len(XTB_OPTIMIZED_SUFFIX)]))
        elif name == XTB_OPTIMIZED_BARE:
            # Non-namespaced xTB: pair it with the only other .xyz in the
            # directory, which is the layout an optimisation stage produces.
            siblings = [
                os.path.splitext(os.path.basename(str(p)))[0]
                for p in paths
                if os.path.dirname(str(p)) == directory and not is_xtb_optimized_name(p)
            ]
            if len(siblings) == 1:
                optimized_bases.add((directory, siblings[0]))

    kept: List[str] = []
    superseded: List[str] = []
    for path in paths:
        path = str(path)
        directory, name = os.path.split(path)
        if not is_xtb_optimized_name(path) and (directory, os.path.splitext(name)[0]) in optimized_bases:
            superseded.append(path)
        else:
            kept.append(path)
    return kept, superseded


def _parse_atom_line(line: str) -> Optional[Tuple[int, List[float]]]:
    """Parse one ``Symbol x y z`` line into ``(Z, [x, y, z])``, or ``None``.

    Accepts an atomic number in the symbol column as well, which some MD codes
    emit. Extra trailing columns (forces, velocities, charges) are ignored.
    """
    parts = line.split()
    if len(parts) < 4:
        return None
    label = parts[0]
    try:
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None

    if label.isdigit():
        atomic_number = int(label)
    else:
        # Normalise "FE" / "fe" to "Fe" before the lookup; unknown symbols
        # become 0 rather than aborting the whole frame.
        atomic_number = SYMBOL_TO_Z.get(label.capitalize(), 0)
    return atomic_number, [x, y, z]


def read_xyz_frames(
    path: os.PathLike[str] | str,
    max_frames: Optional[int] = None,
) -> List[XyzFrame]:
    """Read every frame in the XYZ file at *path*.

    The format is ``natoms`` line, comment line, then ``natoms`` atom lines,
    repeated until end of file. A frame that is truncated or malformed is
    skipped with a warning rather than aborting the file: a trailing partial
    frame is the usual way an interrupted trajectory ends, and the complete
    frames before it are still perfectly good structures.

    Returns an empty list when nothing parseable was found.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        print(f"  ERROR: could not read {path.name}: {exc}")
        return []

    frames: List[XyzFrame] = []
    cursor = 0
    total = len(lines)

    while cursor < total:
        # Tolerate blank lines between frames.
        if not lines[cursor].strip():
            cursor += 1
            continue

        try:
            natoms = int(lines[cursor].strip())
        except ValueError:
            print(
                f"  WARNING: {path.name} line {cursor + 1}: expected an atom "
                f"count, got {lines[cursor].strip()!r}. Stopping."
            )
            break

        if natoms <= 0:
            print(f"  WARNING: {path.name} frame {len(frames) + 1}: atom count {natoms}. Stopping.")
            break

        comment_idx = cursor + 1
        first_atom = cursor + 2
        last_atom = first_atom + natoms
        if last_atom > total:
            print(
                f"  WARNING: {path.name} frame {len(frames) + 1}: truncated "
                f"({total - first_atom} of {natoms} atom lines). Skipping."
            )
            break

        atomnos: List[int] = []
        coords: List[List[float]] = []
        malformed = False
        for line in lines[first_atom:last_atom]:
            parsed = _parse_atom_line(line)
            if parsed is None:
                malformed = True
                break
            atomnos.append(parsed[0])
            coords.append(parsed[1])

        if malformed:
            print(f"  WARNING: {path.name} frame {len(frames) + 1}: malformed atom line. Skipping frame.")
        else:
            frames.append(
                XyzFrame(
                    index=len(frames) + 1,
                    comment=lines[comment_idx] if comment_idx < total else "",
                    atomnos=np.array(atomnos, dtype=int),
                    coords=np.array(coords, dtype=float),
                )
            )
            if max_frames is not None and len(frames) >= max_frames:
                break

        cursor = last_atom

    return frames


def count_xyz_frames(path: os.PathLike[str] | str) -> int:
    """Number of frames in an XYZ file, without holding them all in memory.

    Reads only the atom-count lines, seeking past the coordinate blocks, so a
    large trajectory can be classified as single- or multi-frame cheaply.
    """
    path = Path(path)
    frames = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            while True:
                header = handle.readline()
                if not header:
                    break
                if not header.strip():
                    continue
                try:
                    natoms = int(header.strip())
                except ValueError:
                    break
                if natoms <= 0:
                    break
                # Comment line plus the atom block.
                skipped = 0
                for _ in range(natoms + 1):
                    if not handle.readline():
                        break
                    skipped += 1
                if skipped < natoms + 1:
                    break  # truncated trailing frame
                frames += 1
    except OSError:
        return 0
    return frames


def write_single_frame(frame: XyzFrame, destination: os.PathLike[str] | str) -> None:
    """Write one :class:`XyzFrame` as a standard single-structure XYZ file."""
    from cosmic_ascec.clustering.features.geometric import atomic_number_to_symbol

    destination = Path(destination)
    with open(destination, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(f"{frame.natoms}\n")
        handle.write(f"{frame.comment}\n")
        for atomic_number, (x, y, z) in zip(frame.atomnos, frame.coords):
            symbol = atomic_number_to_symbol(int(atomic_number))
            handle.write(f"{symbol:<2} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def _frame_stem(source_stem: str, index: int, width: int) -> str:
    """Name for frame *index* carved out of ``source_stem``.

    Zero-padded so a directory listing sorts the same way the frames appear in
    the file, which keeps the run reports readable for large ensembles.
    """
    return f"{source_stem}_{index:0{width}d}"


def explode_multiframe_xyz(
    paths: Sequence[os.PathLike[str] | str],
    workdir: os.PathLike[str] | str,
) -> Tuple[List[str], Optional[str]]:
    """Normalise *paths* so every returned file holds exactly one structure.

    When no input holds more than one frame — the ordinary case of a directory
    of one-structure files — nothing is written and the original paths are
    returned unchanged, so those runs keep their own filenames in the reports
    and cost nothing.

    As soon as any input is multi-frame, *every* structure is materialised into
    *workdir*: frames are written as ``<stem>_<i>.xyz`` and already-single
    files are copied across. Making the directory complete rather than mixed
    is what lets the caller simply re-point ``input_source`` at it, so the
    globbing, caching and skipped-file layers downstream need no notion of a
    frame index.

    Returns ``(single_frame_paths, materialised_dir_or_None)``.
    """
    workdir = Path(workdir)
    paths = [Path(p) for p in paths]

    frame_counts = {p: count_xyz_frames(p) for p in paths}
    multi = [p for p in paths if frame_counts[p] > 1]

    if not multi:
        # Zero frames means unparseable; leave those in place so the parser
        # reports the failure against the real filename.
        return [str(p) for p in paths], None

    workdir.mkdir(parents=True, exist_ok=True)
    resolved: List[str] = []
    exploded_frames = 0

    for path in paths:
        if frame_counts[path] <= 1:
            target = workdir / path.name
            if os.path.abspath(target) != os.path.abspath(path):
                shutil.copyfile(path, target)
            resolved.append(str(target))
            continue

        frames = read_xyz_frames(path)
        width = len(str(len(frames)))
        for frame in frames:
            target = workdir / f"{_frame_stem(path.stem, frame.index, width)}{XYZ_EXTENSION}"
            write_single_frame(frame, target)
            resolved.append(str(target))
        exploded_frames += len(frames)

    print(
        f"  Split {len(multi)} multi-frame XYZ file(s) into {exploded_frames} "
        f"single-structure files in '{workdir.name}/'"
    )
    return resolved, str(workdir)


def natural_sort_key(path: os.PathLike[str] | str):
    """Sort key ordering ``conf_2`` before ``conf_10``.

    The clustering front-end feeds files to a process pool in sorted order and
    the reports echo that order, so a plain lexicographic sort would interleave
    the structures of a large ensemble confusingly.
    """
    name = os.path.basename(str(path))
    return tuple(
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", name)
    )


__all__ = [
    "FRAMES_SUBDIR",
    "XTB_OPTIMIZED_BARE",
    "XTB_OPTIMIZED_SUFFIX",
    "XYZ_EXTENSION",
    "XYZ_GLOB",
    "XyzFrame",
    "count_xyz_frames",
    "explode_multiframe_xyz",
    "is_xtb_optimized_name",
    "is_xyz_byproduct",
    "natural_sort_key",
    "read_xyz_frames",
    "resolve_optimized_duplicates",
    "write_single_frame",
]
