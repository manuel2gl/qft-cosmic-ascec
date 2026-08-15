"""MDL ``.mol`` input for the clustering pipeline, via OpenBabel.

COSMIC clusters ``.out`` / ``.log`` QM outputs and plain ``.xyz`` coordinates.
A ``.mol`` file carries the same two things this pipeline needs from a
coordinate file — element symbols and Cartesian coordinates — plus a title line
that COSMIC's own writers already use to carry the energy annotation
(``motif_16_opt (G = -458.836007 Hartree ...)``). Structures that have been
round-tripped through GaussView or through COSMIC's own ``combine_xyz_files``
often exist *only* in that form, so refusing them forced a manual obabel pass
before every run.

Rather than teach the parsers a second coordinate format, ``.mol`` input is
normalised to ``.xyz`` at the CLI boundary and everything downstream sees the
input type it already understands. That preserves the one-file-one-record
contract described in :mod:`~cosmic_ascec.clustering.features.xyz_input`: the
pickle cache stays keyed by filename, the skipped-structure copier still finds
its source on disk, and ``--sp`` still has a real file to hand to xTB.

The conversion is OpenBabel's, not ours. ``obabel -imol x.mol -oxyz`` preserves
the title line verbatim and reproduces the coordinates to the precision the
``.mol`` block stores them at, so the converted file is the same structure the
user would have gotten by converting by hand.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import List, Sequence, Tuple

from cosmic_ascec.exceptions import ClusteringError

# Files this module claims. Mirrors XYZ_GLOB / XYZ_EXTENSION so the CLI can
# treat the two coordinate front-ends symmetrically.
MOL_GLOB = "*.mol"
MOL_EXTENSION = ".mol"

#: Directory (relative to the run's output base) that receives the converted
#: structures. Kept apart from ``xyz_frames`` so a run never reads and writes
#: the same place, and named for what it holds rather than for the run.
MOL_CONVERTED_SUBDIR = "mol_as_xyz"


def _obabel_exe() -> str:
    """Resolved path to the obabel binary, or ``""`` when it is not installed.

    Resolved rather than invoked bare: conda-forge ships obabel on Windows as a
    ``.bat`` shim, which ``CreateProcess`` will not find from the name alone
    (``shutil.which`` consults ``PATHEXT``; ``subprocess`` does not).
    """
    return shutil.which("obabel") or ""


def is_mol_byproduct(path: str) -> bool:
    """Whether *path* is a ``.mol`` COSMIC itself wrote, rather than an input.

    ``filters`` and ``motifs`` convert their outputs to ``.mol`` beside the
    ``.xyz`` they came from, so a folder that has been clustered once holds
    ``all_candidates_combined.mol`` and friends. Feeding those back in on a
    re-run would cluster a run's own output alongside its input — the same trap
    the ``.xyz`` byproduct filter exists to close.

    Only the *combined* files and optimisation trajectories are rejected. A
    per-cluster ``cluster_1_2.mol`` is left alone: running COSMIC inside an
    ``extracted_clusters`` folder to re-examine one cluster is a normal thing to
    do, and the ``.xyz`` front-end does not reject its counterpart either.
    """
    name = os.path.basename(path).lower()
    return "combined" in name or name.endswith(("_trj.mol", ".xtbtraj.mol"))


def find_mol_files(folder: str) -> List[str]:
    """Clusterable ``.mol`` files in *folder*, byproducts excluded."""
    return sorted(f for f in glob.glob(os.path.join(folder, MOL_GLOB))
                  if not is_mol_byproduct(f))


def convert_mol_files(mol_files: Sequence[str], dest_dir: str) -> Tuple[List[str], List[str]]:
    """Convert *mol_files* to ``.xyz`` inside *dest_dir*.

    Returns ``(converted_paths, failed_names)``. *dest_dir* is emptied of any
    ``.xyz`` left by an earlier conversion first, so a stale structure from a
    previous run can never join the current set — the folder handed to the
    orchestrator must contain exactly the structures asked for and nothing else.

    Raises :class:`ClusteringError` when OpenBabel is missing or when it cannot
    convert a single file: a coordinate set that silently shrank to nothing is
    worse than a run that stops and says why.
    """
    exe = _obabel_exe()
    if not exe:
        raise ClusteringError(
            "OpenBabel ('obabel') is required to read .mol input but was not "
            "found on PATH. Install it (conda install -c conda-forge openbabel, "
            "or apt install openbabel) or convert the files to .xyz yourself."
        )

    os.makedirs(dest_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(dest_dir, "*.xyz")):
        os.remove(stale)

    converted: List[str] = []
    failed: List[str] = []
    for mol_path in mol_files:
        stem = os.path.splitext(os.path.basename(mol_path))[0]
        xyz_path = os.path.join(dest_dir, stem + ".xyz")
        try:
            result = subprocess.run(
                [exe, "-imol", mol_path, "-oxyz", "-O", xyz_path],
                capture_output=True, text=True, timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            failed.append(os.path.basename(mol_path))
            continue
        # obabel reports a failed conversion on stderr while still exiting 0 in
        # some builds, so the written file is the thing that gets trusted.
        if result.returncode != 0 or not os.path.isfile(xyz_path) or os.path.getsize(xyz_path) == 0:
            failed.append(os.path.basename(mol_path))
            continue
        converted.append(xyz_path)

    if not converted:
        raise ClusteringError(
            f"OpenBabel could not convert any of the {len(mol_files)} .mol "
            f"file(s) to .xyz. Check that they are valid MDL molfiles."
        )
    return converted, failed


def stage_mol_as_xyz(mol_files: Sequence[str], base_dir: str, *,
                     label: str = "") -> str:
    """Convert *mol_files* and return the directory holding the ``.xyz``.

    *label* names a subdirectory under :data:`MOL_CONVERTED_SUBDIR`, so
    processing several folders in one run keeps each folder's converted set
    separate instead of the last one overwriting the rest.
    """
    dest_dir = os.path.join(base_dir, MOL_CONVERTED_SUBDIR)
    if label:
        dest_dir = os.path.join(dest_dir, label)
    converted, failed = convert_mol_files(mol_files, dest_dir)
    print(f"  Converted {len(converted)} .mol file(s) to .xyz with OpenBabel "
          f"({os.path.relpath(dest_dir, base_dir)}/).")
    if failed:
        print(f"  WARNING: OpenBabel could not convert: {', '.join(failed)}")
    return dest_dir
