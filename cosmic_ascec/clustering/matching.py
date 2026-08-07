"""Reduced-feature-tier matcher — absorb partial structures into full clusters.

A structure whose QM output is missing some properties (a job that converged
but produced no frequencies, say) cannot be placed on the full feature
vector. Rather than dropping it, COSMIC matches it against the clusters that
*were* formed from the fullest tier:

1. Take only the features the reduced structure actually has.
2. Rebuild a Z-scored matrix over just those features for the existing
   cluster representatives.
3. Attach the reduced structure to its nearest representative if it falls
   within the same cut distance used to form the clusters; otherwise leave
   it unassigned (it will appear in the report as a singleton).

This keeps the cluster set "complete" — every QM output that converged ends
up in either a cluster or the unassigned bucket, with the distance-based
admission rule preventing low-quality matches from polluting otherwise tight
clusters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, MutableMapping, Mapping, Sequence, Tuple

import numpy as np

from cosmic_ascec.clustering.energies import EnergyMode, sorting_energy
from cosmic_ascec.clustering.motifs import (
    geometric_representative,
    pool_has_energies,
)
from cosmic_ascec.clustering.scaling import (
    apply_weights,
    build_feature_vectors,
    effective_n_features,
    zscore_scale,
)
from cosmic_ascec.clustering.thresholds import (
    pearson_r_from_distance,
    pearson_similarity_pct,
)

Record = MutableMapping[str, Any]


def match_reduced_to_clusters(
    reduced_mols: Sequence[Record],
    fullest_mols: Sequence[Record],
    cluster_labels_fullest: Sequence[Any],
    primary_scalar_features: Sequence[str],
    use_primary_rotconsts: bool,
    weights: Mapping[str, float],
    threshold: float,
    min_std_threshold: float,
    abs_tolerances: Mapping[str, float],
    mode: EnergyMode,
) -> Tuple[Dict[Any, List[Record]], List[Record]]:
    """Match reduced-vector structures against existing clusters from the fullest tier.

    Returns ``(matched_by_label, unmatched)``:
        matched_by_label: dict {cluster_label: [mol, ...]}
        unmatched: list of mols that didn't match any cluster

    Verbatim port of cosmic-v01's ``_match_reduced_to_clusters`` (4108-4214).
    cosmic-v01's ``_sorting_energy`` global becomes the explicit *mode*
    argument (**D-007**).
    """
    if not reduced_mols:
        return {}, []

    # Build cluster → members mapping and pick representatives
    clusters: Dict[Any, List[Record]] = defaultdict(list)
    for mol, lbl in zip(fullest_mols, cluster_labels_fullest):
        clusters[lbl].append(mol)
    # Same representative rule as motifs / attach_pearson_to_rep: lowest energy,
    # or centroid-closest when the pool carries none (plain XYZ, no --sp). There
    # every sorting_energy is inf, so the energy rule would anchor the match on
    # whichever member sorts first by filename.
    geometry_only = not pool_has_energies([list(fullest_mols)])
    representatives: Dict[Any, Record] = {}
    for lbl, members in clusters.items():
        representatives[lbl] = (
            geometric_representative(members) if geometry_only
            else min(members, key=lambda x: (sorting_energy(x, mode), x['filename']))
        )

    # Index fullest mols for quick lookup
    fullest_idx = {id(mol): i for i, mol in enumerate(fullest_mols)}

    # Group reduced mols by their available feature set
    reduced_by_features: Dict[Any, List[Record]] = defaultdict(list)
    for mol in reduced_mols:
        key = frozenset(mol['_available_features'])
        reduced_by_features[key].append(mol)

    matched_by_label: Dict[Any, List[Record]] = defaultdict(list)
    unmatched: List[Record] = []

    all_primary_features = set(primary_scalar_features)
    if use_primary_rotconsts:
        all_primary_features.update(['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'])

    for feat_set, tier_mols in reduced_by_features.items():
        # Shared scalar features (preserving order)
        shared_scalar = [f for f in primary_scalar_features if f in feat_set]
        use_shared_rot = (
            use_primary_rotconsts
            and 'rotational_constants_A' in feat_set
            and 'rotational_constants_B' in feat_set
            and 'rotational_constants_C' in feat_set
        )

        if not shared_scalar and not use_shared_rot:
            unmatched.extend(tier_mols)
            continue

        # Build combined matrix: fullest + reduced tier
        combined = list(fullest_mols) + tier_mols
        n_fullest = len(fullest_mols)

        raw_vecs, feat_names = build_feature_vectors(
            combined, shared_scalar, use_shared_rot, weights
        )
        if not feat_names:
            unmatched.extend(tier_mols)
            continue

        raw_np = np.array(raw_vecs, dtype=float)
        scaled, active_feat_names, _ = zscore_scale(
            raw_np, feat_names, min_std_threshold, abs_tolerances)
        if scaled.shape[1] == 0:
            unmatched.extend(tier_mols)
            continue
        scaled = apply_weights(scaled, active_feat_names, weights)

        n_eff_reduced = effective_n_features(scaled)

        # Match each reduced structure to nearest representative
        for idx, mol in enumerate(tier_mols):
            mol_vec = scaled[n_fullest + idx]
            best_dist = float('inf')
            best_label = None
            best_rep = None
            for lbl, rep in representatives.items():
                rep_pos = fullest_idx[id(rep)]
                dist = np.linalg.norm(mol_vec - scaled[rep_pos])
                if dist < best_dist:
                    best_dist = dist
                    best_label = lbl
                    best_rep = rep
            if best_dist <= threshold:
                mol['_pearson_rep_filename'] = best_rep['filename']
                mol['_pearson_rep_distance'] = float(best_dist)
                mol['_pearson_n_eff'] = n_eff_reduced
                mol['_pearson_rep_r'] = pearson_r_from_distance(best_dist, n_eff_reduced)
                mol['_pearson_rep_pct'] = pearson_similarity_pct(best_dist, n_eff_reduced)
                mol['_pearson_threshold_tau'] = float(threshold) if threshold is not None else None
                mol['_pearson_rep_rule'] = 'centroid' if geometry_only else 'energy'
                matched_by_label[best_label].append(mol)
            else:
                unmatched.append(mol)

    return dict(matched_by_label), unmatched


__all__ = ["match_reduced_to_clusters"]
