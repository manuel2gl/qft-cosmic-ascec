"""Feature-matrix scaling — Z-standardisation and per-feature weighting.

The COSMIC distance metric is a *weighted Euclidean* distance over a
Z-standardised feature space: every feature is centred and scaled to unit
variance so that a 1 cm⁻¹ rotational constant and a 1 Hartree energy are
comparable, then multiplied by a per-feature weight. Constant (or
within-tolerance) features carry no discriminating information and are dropped
entirely — keeping them as zero columns would still bias ``n_eff``.

Ported verbatim from cosmic-v01.py (values, ordering, edge cases unchanged):

* :func:`is_valid_scalar` — ``is_valid_scalar`` (lines 3892-3897).
* :func:`group_has_any_clustering_feature_data` —
  ``group_has_any_clustering_feature_data`` (lines 3900-3915).
* :func:`has_valid_rotational_constants` — ``has_valid_rotational_constants``
  (lines 3918-3925).
* :func:`select_complete_group_scalar_features` —
  ``select_complete_group_scalar_features`` (lines 3928-3937).
* :func:`build_feature_vectors` — ``_build_feature_vectors`` (lines 3940-3962).
* :func:`zscore_scale` — ``_zscore_scale`` (lines 3965-4004).
* :func:`apply_weights` — ``_apply_weights`` (lines 4007-4013).
* :func:`effective_n_features` — ``_effective_n_features`` (lines 4016-4026).
* :func:`median_pairwise_distance` — ``_median_pairwise_distance`` (4029-4041).

The only deliberate change is that the leading underscore of the private
cosmic-v01 names (``_build_feature_vectors`` …) is dropped — these are the
clustering layer's public surface in v05; the four originally-public helpers
keep their cosmic-v01 names verbatim.

:func:`pool_has_hydrogen_bonds` and :func:`apply_absent_defaults` are new in
v05. They run once over the whole pool, before any of the ported functions, and
settle what the H-bond columns mean for this dataset — see
:data:`FEATURE_ABSENT_DEFAULTS`.
"""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from cosmic_ascec.clustering.features.feature_spec import (
    CLUSTERING_NUMERICAL_FEATURES,
    FEATURE_ABSENT_DEFAULTS,
    FEATURE_MAPPING,
)

# The three rotational-constant axis names, in cosmic-v01's order. Kept local
# so :func:`build_feature_vectors` matches cosmic-v01.py line 3955 verbatim.
_ROTATIONAL_AXIS_NAMES = (
    "rotational_constants_A",
    "rotational_constants_B",
    "rotational_constants_C",
)


def is_valid_scalar(value: Any) -> bool:
    """Return whether *value* is usable as a scalar clustering feature.

    A finite float / numpy float passes; ``None`` and non-finite values fail;
    any non-float (e.g. ``int``) passes unconditionally.

    Verbatim port of cosmic-v01's ``is_valid_scalar`` (lines 3892-3897).
    """
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        return np.isfinite(value)
    return True


def group_has_any_clustering_feature_data(
    group_data: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether *group_data* carries any scalar or rotational feature.

    Verbatim port of cosmic-v01's ``group_has_any_clustering_feature_data``
    (lines 3900-3915).
    """
    scalar_internal_keys = [
        FEATURE_MAPPING[feature_name]
        for feature_name in CLUSTERING_NUMERICAL_FEATURES
    ]

    has_scalar_feature = any(
        is_valid_scalar(molecule_data.get(feature_key))
        for molecule_data in group_data
        for feature_key in scalar_internal_keys
    )
    has_rotational_constants = any(
        molecule_data.get('rotational_constants') is not None
        and isinstance(molecule_data.get('rotational_constants'), np.ndarray)
        and molecule_data.get('rotational_constants').ndim == 1
        and len(molecule_data.get('rotational_constants')) == 3
        for molecule_data in group_data
    )
    return has_scalar_feature or has_rotational_constants


def has_valid_rotational_constants(molecule_data: Mapping[str, Any]) -> bool:
    """Return whether *molecule_data* has a length-3 1-D rotational-constant array.

    Verbatim port of cosmic-v01's ``has_valid_rotational_constants``
    (lines 3918-3925).
    """
    rot_consts = molecule_data.get('rotational_constants')
    return (
        rot_consts is not None
        and isinstance(rot_consts, np.ndarray)
        and rot_consts.ndim == 1
        and len(rot_consts) == 3
    )


def pool_has_hydrogen_bonds(pool: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether any structure in *pool* has a detected hydrogen bond.

    The test is over the whole pool, not one structure: it decides whether the
    H-bond columns describe anything at all. A Li₅ cluster has no hydrogen to
    donate; a set of hydrocarbon conformers has hydrogens but never satisfies
    the criterion. Either way the four :data:`HBOND_FEATURES` are constant and
    say nothing about how the structures differ.
    """
    return any(
        (molecule_data.get('num_hydrogen_bonds') or 0) > 0
        for molecule_data in pool
    )


def apply_absent_defaults(pool: Sequence[MutableMapping[str, Any]]) -> bool:
    """Materialise :data:`FEATURE_ABSENT_DEFAULTS` across *pool*; report if applied.

    When at least one structure has a hydrogen bond, every structure that has
    none gets the defined absent value (0 Å / 0 Å / 0°) written onto its record,
    and the function returns ``True``. "No hydrogen bonds" is a determinate
    state rather than missing data, so without this a non-H-bonded minority
    would delete the three H-bond geometry columns for the whole pool —
    :func:`select_complete_group_scalar_features` keeps a column only when
    *every* member has it.

    When no structure has a hydrogen bond the records are left untouched and
    the function returns ``False``. There is no majority to protect, and a
    column of identical zeros carries no information; the caller drops the four
    :data:`HBOND_FEATURES` instead of clustering on them.

    Writing the values onto the records rather than substituting them at each
    read keeps the decision in one place: everything downstream — the feature
    selection, the vectors, ``matching``, the ``.dat`` reports — sees one
    consistent set of values through a plain ``.get()``.
    """
    if not pool_has_hydrogen_bonds(pool):
        return False
    for molecule_data in pool:
        for feature_name, default in FEATURE_ABSENT_DEFAULTS.items():
            internal_key = FEATURE_MAPPING.get(feature_name, feature_name)
            if not is_valid_scalar(molecule_data.get(internal_key)):
                molecule_data[internal_key] = default
    return True


def select_complete_group_scalar_features(
    group_data: Sequence[Mapping[str, Any]],
    candidate_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split *candidate_features* into those valid for every member, and the rest.

    Returns ``(active_features, dropped_features)``.

    Verbatim port of cosmic-v01's ``select_complete_group_scalar_features``
    (lines 3928-3937). The H-bond geometry columns survive a non-H-bonded
    member because :func:`apply_absent_defaults` has already written their
    defined absent value onto the record — the substitution happens once, at
    the pool, not here.
    """
    active_features: List[str] = []
    dropped_features: List[str] = []
    for feature_name in candidate_features:
        internal_key = FEATURE_MAPPING.get(feature_name, feature_name)
        if all(is_valid_scalar(molecule_data.get(internal_key)) for molecule_data in group_data):
            active_features.append(feature_name)
        else:
            dropped_features.append(feature_name)
    return active_features, dropped_features


def build_feature_vectors(
    mols: Sequence[Mapping[str, Any]],
    scalar_features: Sequence[str],
    use_rotconsts: bool,
    weights: Mapping[str, float],
) -> Tuple[List[List[Any]], List[str]]:
    """Build raw feature vectors for a list of molecule records.

    Returns ``(vectors, ordered_feature_names)``. A feature whose weight is
    exactly ``0.0`` is omitted from the vector entirely (cosmic-v01 line 3949):
    a zero-weight column would contribute nothing to the weighted distance, so
    dropping it keeps ``n_eff`` honest.

    Verbatim port of cosmic-v01's ``_build_feature_vectors`` (lines 3940-3962).
    A structure with no hydrogen bonds already carries the defined absent value
    on its record — :func:`apply_absent_defaults` put it there.
    """
    vectors: List[List[Any]] = []
    feature_names: List[str] = []
    for mol in mols:
        vec: List[Any] = []
        names: List[str] = []
        for feat in scalar_features:
            if weights.get(feat, 1.0) != 0.0:
                key = FEATURE_MAPPING.get(feat, feat)
                vec.append(mol.get(key))
                names.append(feat)
        if use_rotconsts:
            rc = mol.get("rotational_constants")
            for i, rc_name in enumerate(_ROTATIONAL_AXIS_NAMES):
                if weights.get(rc_name, 1.0) != 0.0:
                    vec.append(rc[i] if rc is not None and len(rc) > i else None)
                    names.append(rc_name)
        vectors.append(vec)
        if not feature_names:
            feature_names = names
    return vectors, feature_names


def zscore_scale(
    raw_np: np.ndarray,
    feature_names: Sequence[str],
    min_std_threshold: float,
    abs_tolerances: Mapping[str, float],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Z-score standardise a feature matrix column-wise, dropping dead columns.

    A column is dropped when (i) the feature varies by less than its
    user-specified absolute tolerance across the pool, or (ii) the population
    standard deviation is below ``min_std_threshold``. Such a column would
    contribute nothing to pairwise distances and is removed entirely rather
    than retained as zeros.

    Returns ``(scaled, active_feature_names, dropped_feature_names)``.

    Verbatim port of cosmic-v01's ``_zscore_scale`` (lines 3965-4004) — the
    ``sklearn.preprocessing.StandardScaler`` call is kept so the standardised
    values match cosmic-v01 to floating-point precision.
    """
    from sklearn.preprocessing import StandardScaler

    scaled_cols: List[np.ndarray] = []
    active_feature_names: List[str] = []
    dropped_feature_names: List[str] = []
    for col in range(raw_np.shape[1]):
        col_data = raw_np[:, col]
        fname = feature_names[col]
        max_diff = np.max(col_data) - np.min(col_data)
        if fname in abs_tolerances and max_diff < abs_tolerances[fname]:
            dropped_feature_names.append(fname)
            continue
        std = np.std(col_data)
        if std < min_std_threshold:
            dropped_feature_names.append(fname)
            continue
        scaler = StandardScaler()
        scaled_cols.append(scaler.fit_transform(col_data.reshape(-1, 1)).flatten())
        active_feature_names.append(fname)
    if scaled_cols:
        scaled = np.column_stack(scaled_cols)
    else:
        scaled = np.zeros((raw_np.shape[0], 0), dtype=raw_np.dtype)
    return scaled, active_feature_names, dropped_feature_names


def apply_weights(
    scaled: np.ndarray,
    feature_names: Sequence[str],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Multiply each column of *scaled* by its per-feature weight, in place.

    Verbatim port of cosmic-v01's ``_apply_weights`` (lines 4007-4013).
    """
    for col_idx, fname in enumerate(feature_names):
        w = weights.get(fname, 1.0)
        if w != 1.0:
            scaled[:, col_idx] *= w
    return scaled


def effective_n_features(scaled_matrix: np.ndarray) -> float:
    """Effective number of features ``N_f`` in the weighted Pearson identity.

    ``d² = 2·N_f·(1 − r)``. Constant/within-tolerance columns are already
    dropped by :func:`zscore_scale`, so every column has unit variance before
    weighting; after weighting by ``w_k`` the pool variance of column *k* is
    ``w_k²``, and the sum over the remaining columns is exactly the ``N_f``
    that appears in the identity.

    Verbatim port of cosmic-v01's ``_effective_n_features`` (lines 4016-4026).
    """
    if scaled_matrix.size == 0:
        return 0.0
    return float(np.sum(np.var(scaled_matrix, axis=0)))


def median_pairwise_distance(scaled_matrix: np.ndarray) -> float:
    """Median pairwise Euclidean distance over the rows of *scaled_matrix*.

    A stage-invariant scale anchor for the ``--th=opt-spread`` transform.

    Verbatim port of cosmic-v01's ``_median_pairwise_distance`` (4029-4041).
    """
    if (
        scaled_matrix is None
        or scaled_matrix.size == 0
        or scaled_matrix.shape[0] < 2
    ):
        return 0.0
    from scipy.spatial.distance import pdist

    d = pdist(scaled_matrix)
    if d.size == 0:
        return 0.0
    return float(np.median(d))


__all__ = [
    "apply_absent_defaults",
    "apply_weights",
    "build_feature_vectors",
    "effective_n_features",
    "group_has_any_clustering_feature_data",
    "has_valid_rotational_constants",
    "is_valid_scalar",
    "median_pairwise_distance",
    "pool_has_hydrogen_bonds",
    "select_complete_group_scalar_features",
    "zscore_scale",
]
