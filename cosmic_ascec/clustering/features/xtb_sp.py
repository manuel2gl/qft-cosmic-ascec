"""Optional xTB single points for XYZ input (``cosmic --sp``).

A plain coordinate file determines eight of COSMIC's fifteen clustering
columns. The other seven need a wavefunction. ``--sp`` closes most of that gap
cheaply: one xTB single point per structure yields the electronic energy, the
HOMO, the HOMO-LUMO gap and the dipole moment, taking the feature vector from
eight columns to twelve for one subprocess call each. The last three (Gibbs
energy and the two vibrational frequencies) need a frequency calculation, which
a single point is not.

The single point is parsed by the *existing*
:func:`~cosmic_ascec.clustering.features.parsers.extract_properties_with_xtb`
rather than by a bespoke energy regex. That parser already knows how to read
every scalar xTB prints, so routing through a real output file is both less
code and strictly more informative than scraping ``TOTAL ENERGY``.

Only the QM-derived scalars are merged onto the record. Geometry, rotational
constants, V_NN and the hydrogen-bond block stay with the values computed from
the source XYZ, so those columns come from one consistent source across the
whole pool even when a single point fails for some structures. A failed single
point degrades that structure to a geometry-only record; it is never dropped.

Thermochemistry is deliberately out of scope: ``--sp`` is a single point, so
``gibbs_free_energy`` and the vibrational frequencies stay ``None`` and the
pipeline correctly stays in opt-only mode.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Accepted ``--sp`` values mapped to the xTB command-line flags that select the
# Hamiltonian. GFN-FF is the force field — orders of magnitude faster, and the
# only practical choice for the very large systems this input mode exists for,
# but it has no electronic structure, so it yields an energy and nothing else.
SP_METHODS: Dict[str, List[str]] = {
    "gfn2": ["--gfn", "2"],
    "gfn1": ["--gfn", "1"],
    "gfn0": ["--gfn", "0"],
    "gfnff": ["--gfnff"],
}

DEFAULT_SP_METHOD = "gfn2"

# Directory (relative to the run's output base) holding the single-point runs.
SP_SUBDIR = "sp_xtb"

# Scalars taken from the single point. Everything else on the record keeps the
# value computed from the source geometry.
_MERGED_KEYS = (
    "final_electronic_energy",
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "dipole_moment",
    "method",
    "functional",
    "charge",
    "multiplicity",
)

# Per-structure wall-clock ceiling. A single point that has not finished in
# this long is stuck; the structure degrades to geometry-only rather than
# stalling the whole run.
_SP_TIMEOUT_S = 3600


def resolve_sp_method(value: Optional[str]) -> Optional[str]:
    """Normalise a ``--sp`` argument, or raise ``ValueError`` if unknown.

    ``--sp`` with no value arrives as the empty string and means the default.
    """
    if value is None:
        return None
    key = (value or DEFAULT_SP_METHOD).strip().lower().replace("-", "").replace("_", "")
    if key in ("", "xtb"):
        key = DEFAULT_SP_METHOD
    if key.endswith("xtb") and key != "xtb":  # "gfn2xtb" -> "gfn2"
        key = key[:-3]
    if key not in SP_METHODS:
        raise ValueError(
            f"unknown --sp method '{value}'. Choose one of: {', '.join(sorted(SP_METHODS))}"
        )
    return key


def xtb_executable() -> Optional[str]:
    """Path to the ``xtb`` binary, or ``None`` when it is not on ``PATH``."""
    return shutil.which("xtb")


def _run_one(job):
    """Run one xTB single point. Returns ``(filename, output_path_or_None)``."""
    xyz_path, sp_root, method, charge, uhf, exe = job
    xyz_path = Path(xyz_path)
    stem = xyz_path.stem
    workdir = Path(sp_root) / stem

    try:
        workdir.mkdir(parents=True, exist_ok=True)
        local_xyz = workdir / f"{stem}.xyz"
        shutil.copyfile(xyz_path, local_xyz)

        command = [exe, local_xyz.name, *SP_METHODS[method]]
        if charge is not None and charge != 0:
            command += ["--chrg", str(charge)]
        if uhf is not None and uhf != 0:
            command += ["--uhf", str(uhf)]
        # One thread per xTB process: the fan-out is over structures, and
        # letting each call grab every core would oversubscribe the machine.
        command += ["--parallel", "1"]

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"

        out_path = workdir / f"{stem}.out"
        with open(out_path, "w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=_SP_TIMEOUT_S,
                env=env,
                check=False,
            )

        if completed.returncode != 0:
            return (xyz_path.name, None, f"xtb exited with status {completed.returncode}")
        return (xyz_path.name, str(out_path), None)

    except subprocess.TimeoutExpired:
        return (xyz_path.name, None, f"xtb timed out after {_SP_TIMEOUT_S}s")
    except Exception as exc:  # noqa: BLE001 - one bad structure must not kill the run
        return (xyz_path.name, None, str(exc))


def run_single_points(
    xyz_paths: Sequence[str],
    output_base_dir: str,
    method: str = DEFAULT_SP_METHOD,
    charge: Optional[int] = None,
    uhf: Optional[int] = None,
    num_cores: int = 1,
) -> Dict[str, str]:
    """Run one xTB single point per structure.

    Returns ``{source_xyz_filename: path_to_xtb_output}`` for the calls that
    succeeded. Failures are reported and omitted; their structures keep their
    geometry-only feature vectors.
    """
    exe = xtb_executable()
    if exe is None:
        print("  ERROR: --sp requested but 'xtb' was not found on PATH. "
              "Continuing with geometry-only descriptors.")
        return {}

    sp_root = os.path.join(output_base_dir, SP_SUBDIR)
    os.makedirs(sp_root, exist_ok=True)

    jobs = [(p, sp_root, method, charge, uhf, exe) for p in xyz_paths]
    effective_cores = max(1, min(num_cores, len(jobs)))
    print(f"  Running {len(jobs)} xTB single point(s) ({method.upper()}) "
          f"on {effective_cores} core(s)...")

    if effective_cores > 1:
        with mp.Pool(processes=effective_cores) as pool:
            results = pool.map(_run_one, jobs)
    else:
        results = [_run_one(job) for job in jobs]

    outputs: Dict[str, str] = {}
    failures: List[str] = []
    for filename, out_path, error in results:
        if out_path is not None:
            outputs[filename] = out_path
        else:
            failures.append(f"{filename}: {error}")

    if failures:
        print(f"  WARNING: {len(failures)} single point(s) failed; those structures "
              f"keep geometry-only descriptors.")
        for line in failures[:10]:
            print(f"    - {line}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    return outputs


def merge_single_point_properties(
    records: Sequence[Dict[str, Any]],
    outputs: Dict[str, str],
) -> int:
    """Merge single-point scalars onto the matching geometry-only records.

    Returns the number of records that gained an electronic energy.
    """
    from cosmic_ascec.clustering.features.parsers import extract_properties_with_xtb

    merged = 0
    for record in records:
        out_path = outputs.get(record.get("filename", ""))
        if not out_path:
            continue
        parsed = extract_properties_with_xtb(out_path)
        if not parsed:
            continue
        for key in _MERGED_KEYS:
            value = parsed.get(key)
            if value is not None:
                record[key] = value
        if record.get("final_electronic_energy") is not None:
            merged += 1
    return merged


__all__ = [
    "DEFAULT_SP_METHOD",
    "SP_METHODS",
    "SP_SUBDIR",
    "merge_single_point_properties",
    "resolve_sp_method",
    "run_single_points",
    "xtb_executable",
]
