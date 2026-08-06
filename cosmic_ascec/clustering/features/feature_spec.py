"""Feature-vector specification — the parity-critical column contract.

The COSMIC clustering pipeline turns a directory of quantum-chemistry outputs
into one numeric feature vector per candidate structure. *Which* features, in
*what order*, and under *what weights* is a contract: cluster IDs and motif
folders are only reproducible if every run builds the matrix with the same
columns in the same order. This module is the single source of truth for that
contract.

**v04 column-set revision (deliberate deviation from cosmic-v01.py).**
The set below is a non-redundant rework, not the verbatim cosmic-v01 list:

* Dropped ``lumo_energy`` — recoverable as ``homo_energy + homo_lumo_gap``;
  also the noisiest orbital energy in semi-empirical methods.
* Dropped ``radius_of_gyration`` — its information is folded into the
  nuclear-repulsion / rotational-constant block.
* Added ``vnn_nuclear_repulsion`` — Coulomb shape descriptor V_NN.
* Added ``num_hydrogen_bonds`` and ``std_hbond_distance`` — counts and spread
  of the H-bond network, already parsed by cosmic-v01 but never exposed as
  clustering columns.

The mechanics (:func:`parse_weights_argument`, :func:`labelled_column`, the
``(key=value)`` weights syntax) still trace back to cosmic-v01.py 3632-3648.
"""

from __future__ import annotations

import logging
import re
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature-name mapping                                                         #
# --------------------------------------------------------------------------- #
#
# Maps a clustering feature name → the key it is stored under in cosmic-v01's
# parsed ``extracted_props`` dict. Most names are identical;
# ``electronic_energy`` is the exception (the parsers call it
# ``final_electronic_energy``). The three ``rotational_constants_*`` entries
# are documented here for completeness but are handled specially — they index
# into the 3-element ``rotational_constants`` array rather than naming a key.

FEATURE_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "electronic_energy": "final_electronic_energy",
        "gibbs_free_energy": "gibbs_free_energy",
        "homo_energy": "homo_energy",
        "homo_lumo_gap": "homo_lumo_gap",
        "dipole_moment": "dipole_moment",
        "vnn_nuclear_repulsion": "vnn_nuclear_repulsion",
        # Array elements — special-cased in the extractor (see ``rotational_constants``).
        "rotational_constants_A": "rotational_constants_0",
        "rotational_constants_B": "rotational_constants_1",
        "rotational_constants_C": "rotational_constants_2",
        "first_vib_freq": "first_vib_freq",
        "last_vib_freq": "last_vib_freq",
        "num_hydrogen_bonds": "num_hydrogen_bonds",
        "average_hbond_distance": "average_hbond_distance",
        "std_hbond_distance": "std_hbond_distance",
        "average_hbond_angle": "average_hbond_angle",
    }
)


# --------------------------------------------------------------------------- #
# Column order — the parity contract                                           #
# --------------------------------------------------------------------------- #

CLUSTERING_NUMERICAL_FEATURES: Tuple[str, ...] = (
    "electronic_energy",
    "gibbs_free_energy",
    "homo_energy",
    "homo_lumo_gap",
    "dipole_moment",
    "vnn_nuclear_repulsion",
    "first_vib_freq",
    "last_vib_freq",
    "num_hydrogen_bonds",
    "average_hbond_distance",
    "std_hbond_distance",
    "average_hbond_angle",
)
"""The 12 scalar clustering features, v04 non-redundant ordering."""

ROTATIONAL_CONSTANT_SUBFEATURES: Tuple[str, ...] = (
    "rotational_constants_A",
    "rotational_constants_B",
    "rotational_constants_C",
)
"""The 3 rotational-constant axes, appended after the scalar features."""

HBOND_GEOMETRY_FEATURES: Tuple[str, ...] = (
    "average_hbond_distance",
    "std_hbond_distance",
    "average_hbond_angle",
)
"""The H-bond geometry columns — the mean/spread/angle of the detected network.

These are real clustering features and discriminate isomers that share every
energetic descriptor (water-hexamer prism vs cage vs book, for instance).
What they must *not* do is decide whether a structure is usable: a structure
with no eligible H-bond has none of them, and that is a property of the
structure rather than a failed calculation. Two places honour that:

* the criticality / full-feature gate skips them, so a non-H-bonded structure
  is never demoted to the reduced tier for lacking them;
* :data:`FEATURE_ABSENT_DEFAULTS` gives them a defined value when absent, so a
  handful of non-H-bonded structures cannot delete the columns for everyone.
"""

FEATURE_ABSENT_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {name: 0.0 for name in HBOND_GEOMETRY_FEATURES}
)
"""Value a feature takes in the vector when the structure genuinely lacks it.

Only H-bond geometry qualifies. "No hydrogen bonds" is a determinate state,
not missing information — ``num_hydrogen_bonds`` already records it as 0, and
a mean distance of 0 Å / mean angle of 0° is the consistent reading of the
same fact. Every other feature stays strict: a missing electronic energy or
rotational constant means the calculation did not provide it, and inventing a
number there would be a fabrication.

Without this, ``select_complete_group_scalar_features`` — which keeps a column
only when *every* member has it — deletes all three H-bond geometry columns
from the whole pool as soon as one structure has no H-bond. A handful of
non-H-bonded structures would then silently strip H-bond information from the
hundreds that do have it.
"""

FEATURE_COLUMNS: Tuple[str, ...] = (
    CLUSTERING_NUMERICAL_FEATURES + ROTATIONAL_CONSTANT_SUBFEATURES
)
"""Pinned feature-DataFrame column order (**D-027**).

15 columns. The extractor builds every DataFrame with exactly these columns
in this order; the clustering stage (Session 8) depends on it not drifting.
Drops ``lumo_energy`` and ``radius_of_gyration`` from cosmic-v01; adds
``vnn_nuclear_repulsion``, ``num_hydrogen_bonds``, ``std_hbond_distance``.
"""


# --------------------------------------------------------------------------- #
# Display units                                                                #
# --------------------------------------------------------------------------- #
#
# cosmic-v01's ``_EXTRACT_FEATURE_UNITS`` (lines 3717-3733). Used for labelled
# CSV headers (``electronic_energy_(Hartree)``, …). Note: the parsers store
# ``homo_lumo_gap`` in eV (xTB prints eV; cclib computes ``lumo - homo`` on its
# eV orbital energies; OPI reads the ``:: HOMO-LUMO gap … eV ::`` line). The
# clustering matrix Z-scores every feature, so the eV/Hartree choice does not
# affect cluster IDs; how ``run_data_extraction`` labels the ``--data`` dump is
# audited in R5b (the scaling region, cosmic-v01.py 3892-4543).

FEATURE_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "electronic_energy": "Hartree",
        "gibbs_free_energy": "Hartree",
        "homo_energy": "Hartree",
        "homo_lumo_gap": "Hartree",
        "dipole_moment": "Debye",
        "vnn_nuclear_repulsion": "Hartree",
        "first_vib_freq": "cm^-1",
        "last_vib_freq": "cm^-1",
        "num_hydrogen_bonds": "count",
        "average_hbond_distance": "A",
        "std_hbond_distance": "A",
        "average_hbond_angle": "deg",
        "rotational_constants_A": "cm^-1",
        "rotational_constants_B": "cm^-1",
        "rotational_constants_C": "cm^-1",
    }
)


def labelled_column(name: str) -> str:
    """Return ``name_(unit)`` for CSV headers, or bare ``name`` if no unit.

    Mirrors cosmic-v01's ``_extract_labeled`` (lines 3736-3738).
    """
    unit = FEATURE_UNITS.get(name)
    return f"{name}_({unit})" if unit else name


# --------------------------------------------------------------------------- #
# Feature weights                                                              #
# --------------------------------------------------------------------------- #
#
# Tuned per-feature weights for semiempirical / standalone xTB output
# (cosmic-v01.py lines 3693-3712). Values < 1.0 down-weight noisy features so
# geometrically identical structures do not look distinct in the
# Z-standardised feature space. Activated by ``--partialweights``; user
# ``--weights`` overrides layer on top.

SEMIEMPIRICAL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        # Reliability-based partial weights. The axis is *measurement noise of the
        # feature*, not dataset-specific correlations: a feature is only down-
        # weighted if its computation is method-dependent, indexing-sensitive,
        # or cutoff-driven. Correlations observed in any specific dataset
        # (e.g. VNN ↔ rotC ≈ 0.94 in W5) are properties of that chemistry, not
        # of the features themselves — VNN is Coulombic Σ Z_i Z_j / r_ij while
        # rotational constants are eigenvalues of the mass-weighted inertia
        # tensor; they share only the geometry, not the mathematical object.
        "electronic_energy": 1.0,        # Direct SCF output, most reliable
        "gibbs_free_energy": 0.9,        # Thermal corrections add noise
        "homo_energy": 0.85,             # Semi-empirical methods show variance
        "homo_lumo_gap": 0.9,            # Orbital-spread observable
        "dipole_moment": 0.6,            # Sensitive to atom indexing and coord frame
        "vnn_nuclear_repulsion": 1.0,    # Pure geometry+Z, deterministic, no method dependence
        "first_vib_freq": 0.9,           # Generally stable
        "last_vib_freq": 0.85,           # High-freq modes vary with method/basis
        "num_hydrogen_bonds": 0.7,       # Depends on H-bond detection cutoffs
        "average_hbond_distance": 0.7,   # Depends on H-bond detection cutoffs
        "std_hbond_distance": 0.7,       # Depends on H-bond detection cutoffs
        "average_hbond_angle": 0.7,      # Sensitive to H-bond detection geometry
        "rotational_constants_A": 1.0,   # Geometric, index-independent
        "rotational_constants_B": 1.0,   # Geometric, index-independent
        "rotational_constants_C": 1.0,   # Geometric, index-independent
    }
)

DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {key: 1.0 for key in SEMIEMPIRICAL_WEIGHTS}
)
"""Uniform default weights — keeps COSMIC method-agnostic unless ``--partialweights``."""


_WEIGHT_PAIR_RE = re.compile(r"\(([^=]+)=([\d.]+)\)")


def parse_weights_argument(weight_str: str | None) -> Dict[str, float]:
    """Parse a ``--weights`` string into ``{feature_name: weight}``.

    The format is a run of ``(key=value)`` pairs, e.g.
    ``"(electronic_energy=0.1)(homo_lumo_gap=0.2)"``. An unparseable value is
    logged at WARNING and skipped — exactly cosmic-v01's behaviour
    (lines 3632-3648), except v04's ``print`` becomes a ``logging`` call
    (**D-003**).
    """
    weights: Dict[str, float] = {}
    if not weight_str:
        return weights
    for key, value in _WEIGHT_PAIR_RE.findall(weight_str):
        try:
            weights[key.strip()] = float(value.strip())
        except ValueError:
            logger.warning("Could not parse weight for '%s=%s'; skipping.", key, value)
    return weights


def parse_abs_tolerance_argument(tolerance_str: str | None) -> Dict[str, float]:
    """Parse a ``--abs-tolerance`` string into ``{feature_name: tolerance}``.

    The format is a run of ``(key=value)`` pairs, e.g.
    ``"(electronic_energy=1e-5)(dipole_moment=1e-3)"``.

    Verbatim port of cosmic-v01's ``parse_abs_tolerance_argument``
    (lines 3650-3665) — its ``print`` warning is kept verbatim (**D-003**:
    cosmic-v01's stdout is a contract).
    """
    tolerances: Dict[str, float] = {}
    if not tolerance_str:
        return tolerances

    matches = re.findall(r'\(([^=]+)=([\d\.eE-]+)\)', tolerance_str)
    for key, value in matches:
        try:
            tolerances[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse absolute tolerance for '{key}={value}'. Skipping this tolerance.")
    return tolerances


__all__ = [
    "CLUSTERING_NUMERICAL_FEATURES",
    "DEFAULT_WEIGHTS",
    "FEATURE_ABSENT_DEFAULTS",
    "FEATURE_COLUMNS",
    "FEATURE_MAPPING",
    "FEATURE_UNITS",
    "HBOND_GEOMETRY_FEATURES",
    "ROTATIONAL_CONSTANT_SUBFEATURES",
    "SEMIEMPIRICAL_WEIGHTS",
    "labelled_column",
    "parse_abs_tolerance_argument",
    "parse_weights_argument",
]
