"""QM-output parser dispatch — verbatim port of cosmic-v01's front-end loop.

R5 (D-039 faithful decomposition). cosmic-v01 picks a parser for each QM output
by sniffing the file header (``detect_file_type``), resolving the ORCA version
(``detect_orca_version``), and choosing ``opi`` / ``cclib`` / ``xtb``
(``choose_parser``); ``extract_properties_from_logfile`` then dispatches to the
matching parser in :mod:`cosmic_ascec.clustering.features.parsers`.

The pre-R5 v05 ``extractor.py`` routed clustering feature extraction through the
``quantum_chemistry`` adapters' ``parse_result`` / ``QMResult.extras`` and
assembled a ``pandas`` DataFrame. Neither exists in cosmic-v01 — that was a
divergence. This module restores cosmic-v01's dispatch verbatim.

Ported verbatim from ``cosmic-v01.py``:

* ``detect_file_type`` — lines 684-708.
* ``detect_orca_version`` — lines 710-736.
* ``choose_parser`` — lines 738-808.
* ``process_file_parallel_wrapper`` — lines 810-828.
* ``extract_properties_from_logfile`` — lines 830-854.

``list_qm_outputs`` is a v05 directory-discovery helper (a natural-sorted
``glob``); the R5c-verbatim ``perform_clustering_and_analysis`` does its own
``glob.glob`` inline exactly as cosmic-v01 does, so this helper now serves only
its own unit tests.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cosmic_ascec.clustering.features.parsers import (
    CCLIB_AVAILABLE,
    OPI_AVAILABLE,
    extract_properties_with_cclib,
    extract_properties_with_opi,
    extract_properties_with_xtb,
    extract_properties_with_xyz,
)
from cosmic_ascec.clustering.features.xyz_input import XYZ_EXTENSION, XYZ_GLOB

# Input-file extensions the directory loop considers clustering candidates by
# default. ``*.xyz`` is the geometry-only front-end (no QM output required).
DEFAULT_OUTPUT_GLOBS = ("*.out", "*.log", XYZ_GLOB)


def _looks_like_xyz(path) -> bool:
    """Return whether *path* opens with a bare atom-count line.

    Extension alone is not proof — an ``.xyz`` file that is really a QM output
    under a misleading name should still reach its proper parser — so the
    first non-blank line has to be a positive integer on its own.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    return False
                if not line.strip():
                    continue
                try:
                    return int(line.strip()) > 0
                except ValueError:
                    return False
    except Exception:
        pass
    return False


def detect_file_type(logfile_path):
    """
    Detect whether the output file is from ORCA or Gaussian.

    Args:
        logfile_path: Path to output file (.out, .log, or .xyz)

    Returns:
        'orca', 'gaussian', 'xtb', 'xyz', or None if not detected
    """
    # Plain coordinate files are recognised by shape, not by a program banner:
    # a .xyz whose first line is a bare atom count has no header to sniff.
    if str(logfile_path).lower().endswith(XYZ_EXTENSION) and _looks_like_xyz(logfile_path):
        return 'xyz'

    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first 100 lines to detect file type
            for i, line in enumerate(f):
                if i > 100:
                    break
                if 'O   R   C   A' in line:
                    return 'orca'
                if 'Gaussian' in line:
                    return 'gaussian'
                if 'x T B' in line or 'xtb version' in line.lower() or 'normal termination of xtb' in line.lower():
                    return 'xtb'
    except Exception:
        pass
    return None


def detect_orca_version(logfile_path):
    """
    Detect ORCA version from output file.

    Args:
        logfile_path: Path to ORCA output file (.out or .log)

    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Scan header section until we find version or hit input file section
            # This is robust to header size changes in future ORCA versions
            for i, line in enumerate(f):
                # Stop if we've reached the input file section (version should be before this)
                if 'INPUT FILE' in line or i > 500:
                    break
                # Look for version pattern like "Program Version 6.1.0" or "Program Version 7.0.0"
                match = re.search(r'Program Version\s+(\d+)\.(\d+)', line, re.IGNORECASE)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return (major, minor)
    except Exception:
        pass
    return None


def choose_parser(logfile_path):
    """
    Automatically choose the appropriate parser based on file type and available libraries.

    Args:
        logfile_path: Path to ORCA or Gaussian output file

    Returns:
        'opi', 'cclib', or 'xtb', or raises error if no suitable parser
    """
    # First, detect the file type (ORCA or Gaussian)
    file_type = detect_file_type(logfile_path)

    # Plain coordinates: everything computable comes from geometry, so no
    # third-party output parser is involved at all.
    if file_type == 'xyz':
        return 'xyz'

    # Standalone xTB output parsing does not depend on cclib/OPI.
    if file_type == 'xtb':
        return 'xtb'

    # For Gaussian files, always use cclib (OPI only supports ORCA)
    if file_type == 'gaussian':
        if CCLIB_AVAILABLE:
            return 'cclib'
        else:
            raise RuntimeError(
                "Gaussian output file detected, but cclib is not installed. "
                "Please install cclib: pip install cclib"
            )

    # For ORCA files, check version and choose appropriate parser
    if file_type == 'orca':
        version = detect_orca_version(logfile_path)

        # If version detected
        if version:
            major, minor = version

            # ORCA 6.1+ requires OPI (use tuple comparison for correct 7.0, 8.0 etc.)
            if (major, minor) >= (6, 1):
                if OPI_AVAILABLE:
                    return 'opi'
                else:
                    raise RuntimeError(
                        f"ORCA version {major}.{minor} detected, which requires OPI. "
                        "Please install OPI: pip install orca-pi"
                    )

            # ORCA 6.0 not supported by either
            elif major == 6 and minor == 0:
                raise RuntimeError(
                    f"ORCA version 6.0 is not supported. Please use ORCA 5.0.x or ORCA 6.1+. "
                    "cclib supports up to ORCA 5.0.x, OPI supports ORCA 6.1+"
                )

            # ORCA <=5.0 can use cclib
            else:
                if CCLIB_AVAILABLE:
                    return 'cclib'
                elif OPI_AVAILABLE:
                    warnings.warn(
                        f"ORCA version {major}.{minor} detected. OPI will be used, but cclib is recommended for ORCA <=5.0"
                    )
                    return 'opi'
                else:
                    raise RuntimeError("Neither cclib nor OPI available. Please install one of them.")

    # If file type not detected, use available parser (prefer cclib for compatibility)
    if CCLIB_AVAILABLE:
        return 'cclib'
    elif OPI_AVAILABLE:
        return 'opi'
    else:
        raise RuntimeError("Neither cclib nor OPI available. Please install one of them.")


def process_file_parallel_wrapper(file_path):
    """
    Wrapper function for parallel processing of files.
    Returns a tuple of (success, result, filename) to handle both successful extractions and skipped files.
    """
    try:
        filename = os.path.basename(file_path)
        # Avoid noisy parser fallback errors for empty output files.
        if os.path.getsize(file_path) == 0:
            print(f"  WARNING: {filename} is empty (0 bytes). Skipping.")
            return (False, None, filename)
        extracted_props = extract_properties_from_logfile(file_path)
        if extracted_props:
            return (True, extracted_props, filename)  # Success
        else:
            return (False, None, filename)  # Skipped (likely imaginary frequencies)
    except Exception as e:
        print(f"  ERROR: Failed to process {os.path.basename(file_path)}: {e}")
        return (False, None, os.path.basename(file_path))  # Error


def extract_properties_from_logfile(logfile_path):
    """
    Extract molecular properties from quantum chemistry log file.
    Supports both cclib (ORCA <=5.0) and OPI (ORCA >=6.1).

    Args:
        logfile_path (str): Path to .log or .out file

    Returns:
        dict: Extracted properties, or None if parsing fails
    """
    # Determine which parser to use
    try:
        parser_type = choose_parser(logfile_path)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return None

    # Use the appropriate parser
    if parser_type == 'opi':
        return extract_properties_with_opi(logfile_path)
    elif parser_type == 'xtb':
        return extract_properties_with_xtb(logfile_path)
    elif parser_type == 'xyz':
        return extract_properties_with_xyz(logfile_path)
    else:
        return extract_properties_with_cclib(logfile_path)


# --------------------------------------------------------------------------- #
# Directory discovery — v05 helper for the orchestrator                         #
# --------------------------------------------------------------------------- #


def _natural_key(name: str):
    """Sort key: first integer in the filename, then the name itself."""
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else 0, name)


def list_qm_outputs(
    directory: Path,
    globs: Sequence[str] = DEFAULT_OUTPUT_GLOBS,
) -> List[Path]:
    """Return the QM-output files in ``directory``, in natural-sorted order."""
    from cosmic_ascec.exceptions import ClusteringError

    from cosmic_ascec.clustering.features.xyz_input import is_xyz_byproduct

    directory = Path(directory)
    if not directory.is_dir():
        raise ClusteringError(f"not a directory: {directory}")
    found: Dict[str, Path] = {}
    for pattern in globs:
        for candidate in directory.glob(pattern):
            # Engine-written geometries (xtbopt.xyz and friends) are output, not
            # input; ingesting them would let a re-run cluster its own results.
            if candidate.is_file() and not is_xyz_byproduct(candidate):
                found[candidate.name] = candidate
    return [found[name] for name in sorted(found, key=_natural_key)]


def extract_features_from_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse one QM output into cosmic-v01's ``extracted_props`` dict.

    Thin wrapper over :func:`extract_properties_from_logfile` accepting a
    :class:`~pathlib.Path`; returns cosmic-v01's property dict, or ``None`` when
    the file could not be parsed.
    """
    return extract_properties_from_logfile(str(path))


__all__ = [
    "DEFAULT_OUTPUT_GLOBS",
    "choose_parser",
    "detect_file_type",
    "detect_orca_version",
    "extract_features_from_file",
    "extract_properties_from_logfile",
    "list_qm_outputs",
    "process_file_parallel_wrapper",
]
