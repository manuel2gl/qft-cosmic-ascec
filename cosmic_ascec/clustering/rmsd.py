"""RMSD post-processing — geometric refinement of property-based clusters.

Two structures can have nearly identical physicochemical feature vectors yet
be genuinely different geometric arrangements (stereoisomers, mirror images,
loosely-coupled conformers). The optional RMSD pass catches that: within each
property cluster, Cartesian RMSD is measured between every member and the
lowest-energy representative; members beyond the threshold are split out and
re-clustered on a pure RMSD distance matrix.

RMSD is all-atom (hydrogens included) to match CREST and ORCA GOAT, the codes
COSMIC is benchmarked against; ``heavy_only=True`` (CLI ``--rmsd-heavy``)
restores the heavy-atom-only measure cosmic-v01 always used.

Public functions:

* :func:`calculate_rmsd` — Cartesian RMSD via Kabsch alignment.
* :func:`post_process_clusters_with_rmsd` — first pass: detect outliers
  inside each property cluster.
* :func:`perform_second_rmsd_clustering` — re-cluster the outliers from the
  first pass into their own groups.
* :func:`cluster_by_rmsd` — standalone geometry-only partition used by
  ``cosmic --rmsd-only`` (no feature vector involved at all).

The clustering record dicts are mutated in place; energy selection is
threaded explicitly via an
:class:`~cosmic_ascec.clustering.energies.EnergyMode` argument so the same
function works for runs with and without thermochemistry.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import numpy as np

from cosmic_ascec.clustering.energies import EnergyMode, sorting_energy

Record = MutableMapping[str, Any]
Cluster = List[Record]


def calculate_rmsd(
    atomnos1: Sequence[int],
    coords1: np.ndarray,
    atomnos2: Sequence[int],
    coords2: np.ndarray,
    heavy_only: bool = False,
) -> Any:
    """Cartesian RMSD between two structures via Kabsch alignment.

    All atoms are used by default, hydrogens included — this is the convention
    of the codes COSMIC is benchmarked against: CREST's ``--rmsd`` is all-atom
    (heavy-atom is the separate ``--rmsdheavy``) and ORCA's GOAT filters on the
    RMSD of atomic positions, both at a 0.125 Å default threshold. Pass
    *heavy_only* to drop hydrogens (Z = 1) instead.

    Returns the minimised RMSD, or ``None`` when the compared atom counts
    differ or alignment fails.

    Ported from cosmic-v01's ``calculate_rmsd`` (lines 1961-2026); that version
    was heavy-atom-only with no choice.
    """
    from scipy.spatial.transform import Rotation as R  # Import only when needed

    if heavy_only:
        indices1 = [i for i, z in enumerate(atomnos1) if z != 1]
        indices2 = [i for i, z in enumerate(atomnos2) if z != 1]
    else:
        indices1 = list(range(len(atomnos1)))
        indices2 = list(range(len(atomnos2)))

    if len(indices1) == 0 or len(indices2) == 0:
        return None

    if len(indices1) != len(indices2):
        return None

    # Ensure coordinates are numpy arrays and float64
    coords1_filtered = np.asarray(coords1[indices1], dtype=np.float64)
    coords2_filtered = np.asarray(coords2[indices2], dtype=np.float64)

    if coords1_filtered.shape[0] == 0 or coords2_filtered.shape[0] == 0:
        return None

    try:
        # Step 1: Center the coordinates (move to origin)
        center1 = np.mean(coords1_filtered, axis=0)
        centered_coords1 = coords1_filtered - center1

        center2 = np.mean(coords2_filtered, axis=0)
        centered_coords2 = coords2_filtered - center2

        # Step 2: Perform Kabsch alignment to find the optimal rotation
        # R.align_vectors(a, b) finds rotation + RMSD to transform a onto b.
        result = R.align_vectors(centered_coords2, centered_coords1)
        rmsd_value = result[1]  # The RMSD is always the second value returned

        return rmsd_value

    except Exception as e:
        print(f"  ERROR during RMSD calculation: {e}")
        return None


def post_process_clusters_with_rmsd(
    initial_clusters: Sequence[Cluster],
    rmsd_validation_threshold: float,
    mode: EnergyMode,
    heavy_only: bool = False,
) -> Tuple[List[Cluster], List[Record]]:
    """First RMSD pass — validate each property cluster against its representative.

    Returns ``(validated_main_clusters, individual_outliers)``: members whose
    RMSD to the lowest-energy representative exceeds the threshold are pulled
    out as individual outliers (to be re-clustered by
    :func:`perform_second_rmsd_clustering`).

    Verbatim port of cosmic-v01's ``post_process_clusters_with_rmsd``
    (lines 2028-2121). cosmic-v01's ``_sorting_energy`` global becomes the
    explicit *mode* argument (**D-007**).
    """
    validated_main_clusters: List[Cluster] = []
    individual_outliers: List[Record] = []

    print(f"  Initiating first pass RMSD validation with threshold: {rmsd_validation_threshold:.3f} Å...")

    for cluster_idx, current_property_cluster in enumerate(initial_clusters):
        if not current_property_cluster:
            continue

        if len(current_property_cluster) == 1:
            # Single member clusters are passed directly to validated_main_clusters

            current_property_cluster[0]['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        print(f"    Validating initial property cluster {current_property_cluster[0].get('_parent_global_cluster_id', 'N/A')} with {len(current_property_cluster)} configurations...")

        # Select the lowest energy configuration as the representative for this property cluster

        representative_conf = min(current_property_cluster,
                                  key=lambda x: (sorting_energy(x, mode), x['filename']))

        current_validated_sub_cluster = [representative_conf]  # Start new validated cluster with representative
        processed_members_filenames = {representative_conf['filename']}

        representative_conf['_rmsd_pass_origin'] = 'first_pass_validated'

        coords_rep = representative_conf.get('final_geometry_coords')
        atomnos_rep = representative_conf.get('final_geometry_atomnos')

        if coords_rep is None or atomnos_rep is None:
            print(f"    WARNING: Representative {representative_conf['filename']} has missing geometry. Skipping RMSD validation for this property cluster. All members kept together for now.")
            # If skipping, mark all as first_pass_validated
            for conf_member in current_property_cluster:
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        other_members = [conf for conf in current_property_cluster if conf != representative_conf]

        for conf_member in other_members:
            if conf_member['filename'] in processed_members_filenames:
                continue

            coords_member = conf_member.get('final_geometry_coords')
            atomnos_member = conf_member.get('final_geometry_atomnos')

            if coords_member is None or atomnos_member is None:
                print(f"    WARNING: {conf_member['filename']} has missing geometry data. Treating as an individual outlier for now.")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member)  # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])
                continue

            rmsd_val = calculate_rmsd(
                atomnos_rep, coords_rep,
                atomnos_member, coords_member,
                heavy_only=heavy_only
            )

            if rmsd_val is not None and rmsd_val <= rmsd_validation_threshold:
                current_validated_sub_cluster.append(conf_member)
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
                processed_members_filenames.add(conf_member['filename'])
            else:
                # rmsd_val is None when the two structures are not comparable
                # (differing heavy-atom count) — formatting it as a float there
                # raised TypeError and aborted the whole run.
                _rmsd_str = f"{rmsd_val:.3f}" if rmsd_val is not None else "N/A"
                print(f"    {conf_member['filename']} (RMSD={_rmsd_str} Å) is an outlier from {representative_conf['filename']} (Threshold={rmsd_validation_threshold:.3f} Å).")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member)  # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])

        if current_validated_sub_cluster:
            validated_main_clusters.append(current_validated_sub_cluster)

    return validated_main_clusters, individual_outliers


def perform_second_rmsd_clustering(
    cluster_members_to_refine: Cluster,
    rmsd_threshold: float,
    mode: EnergyMode,
    heavy_only: bool = False,
) -> List[Cluster]:
    """Second RMSD pass — re-cluster first-pass outliers on an RMSD distance matrix.

    A pairwise Cartesian RMSD matrix feeds a UPGMA linkage; the tree is cut at
    *rmsd_threshold*. Each resulting sub-cluster records its representative and
    the RMSD of every member to it.

    Verbatim port of cosmic-v01's ``perform_second_rmsd_clustering``
    (lines 2604-2693). cosmic-v01's ``_sorting_energy`` global becomes the
    explicit *mode* argument (**D-007**).
    """
    from scipy.cluster.hierarchy import linkage, fcluster  # Import only when needed

    if len(cluster_members_to_refine) <= 1:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    # Outliers with no geometry at all reach this pass (the first pass routes
    # them here), so the pairwise distances go through the guarded helper:
    # indexing None coordinates raised TypeError, and the previous float('inf')
    # for an incomparable pair propagated into every later merge height.
    condensed_distances = rmsd_condensed_distances(cluster_members_to_refine, heavy_only=heavy_only)

    if len(condensed_distances) == 0:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    linkage_matrix = linkage(condensed_distances, method='average', metric='euclidean')
    second_cluster_labels = fcluster(linkage_matrix, t=rmsd_threshold, criterion='distance')

    second_level_clusters_data: Dict[Any, Cluster] = {}
    for i, label in enumerate(second_cluster_labels):
        cluster_members_to_refine[i]['_second_rmsd_sub_cluster_id'] = label
        cluster_members_to_refine[i]['_rmsd_pass_origin'] = 'second_pass_formed'
        second_level_clusters_data.setdefault(label, []).append(cluster_members_to_refine[i])

    final_sub_clusters: List[Cluster] = []
    for label, sub_cluster_members in second_level_clusters_data.items():
        if not sub_cluster_members:
            continue

        sub_cluster_rep = min(sub_cluster_members,
                              key=lambda x: (sorting_energy(x, mode), x['filename']))

        sub_cluster_rmsd_listing = []
        if sub_cluster_rep.get('final_geometry_coords') is not None and sub_cluster_rep.get('final_geometry_atomnos') is not None:
            for member_conf in sub_cluster_members:
                if member_conf == sub_cluster_rep:
                    rmsd_val = 0.0
                else:
                    # Guarded: a member without geometry is reported as N/A
                    # rather than raising on None coordinates.
                    rmsd_val = _pair_rmsd(sub_cluster_rep, member_conf, heavy_only=heavy_only)
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})
        else:
            for member_conf in sub_cluster_members:
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': None})

        for member_conf in sub_cluster_members:
            member_conf['_second_rmsd_context_listing'] = sub_cluster_rmsd_listing
            member_conf['_second_rmsd_rep_filename'] = sub_cluster_rep['filename']

        final_sub_clusters.append(sub_cluster_members)

    return final_sub_clusters


def _pair_rmsd(conf1: Record, conf2: Record, heavy_only: bool = False) -> Any:
    """Cartesian RMSD between two records, or ``None`` if geometry is missing."""
    coords1 = conf1.get('final_geometry_coords')
    atomnos1 = conf1.get('final_geometry_atomnos')
    coords2 = conf2.get('final_geometry_coords')
    atomnos2 = conf2.get('final_geometry_atomnos')

    if coords1 is None or atomnos1 is None or coords2 is None or atomnos2 is None:
        return None

    return calculate_rmsd(atomnos1, coords1, atomnos2, coords2, heavy_only=heavy_only)


def rmsd_condensed_distances(members: Cluster, heavy_only: bool = False) -> np.ndarray:
    """Condensed pairwise Cartesian RMSD vector for *members*.

    Ordering matches :func:`scipy.spatial.distance.squareform`, so the result
    can be handed straight to :func:`scipy.cluster.hierarchy.linkage`.

    A pair that cannot be compared at all (Kabsch failure) gets a large
    *finite* sentinel rather than ``inf``: it must never merge at any sensible
    threshold, but ``inf`` in a linkage matrix poisons the merge heights of
    every later merge. Callers are expected to have split *members* into
    comparable blocks first (see :func:`rmsd_comparability_blocks`), so the
    sentinel is a safety net, not the normal path.
    """
    n = len(members)
    dists: List[float] = []
    incomparable: List[int] = []

    for i in range(n):
        for j in range(i + 1, n):
            rmsd = _pair_rmsd(members[i], members[j], heavy_only=heavy_only)
            if rmsd is None:
                incomparable.append(len(dists))
                dists.append(0.0)  # placeholder, replaced below
            else:
                dists.append(float(rmsd))

    arr = np.asarray(dists, dtype=float)
    if incomparable:
        finite = np.delete(arr, incomparable)
        sentinel = (float(np.max(finite)) * 10.0 + 1000.0) if finite.size else 1000.0
        arr[incomparable] = sentinel

    return arr


def rmsd_comparability_blocks(members: Cluster, heavy_only: bool = False) -> List[Cluster]:
    """Split *members* into blocks whose geometries can be RMSD-compared.

    Kabsch RMSD is only defined between structures with the same number of
    compared atoms, so that count (all atoms, or heavy atoms under
    *heavy_only*) plus "has a geometry at all" partitions the pool into
    independent blocks. Structures without geometry are their own block of
    one — nothing can be measured against them.

    Blocks are returned largest first, member order preserved within a block.
    """
    blocks: Dict[int, Cluster] = {}
    orphans: List[Cluster] = []

    for member in members:
        coords = member.get('final_geometry_coords')
        atomnos = member.get('final_geometry_atomnos')
        if coords is None or atomnos is None:
            orphans.append([member])
            continue
        n_compared = (sum(1 for z in atomnos if z != 1) if heavy_only
                      else len(atomnos))
        if n_compared == 0:
            orphans.append([member])
            continue
        blocks.setdefault(n_compared, []).append(member)

    ordered = sorted(blocks.values(), key=len, reverse=True)
    return ordered + orphans


def cluster_by_rmsd(
    members: Cluster,
    rmsd_threshold: float,
    mode: EnergyMode,
    heavy_only: bool = False,
) -> Tuple[List[Cluster], Any, Cluster]:
    """Partition *members* on Cartesian RMSD alone — no property features.

    This is the whole clustering step for ``cosmic --rmsd-only``: a pairwise
    Cartesian RMSD matrix feeds a UPGMA linkage that is cut at
    *rmsd_threshold* (in Å, so the cut height is directly interpretable).

    Structures that cannot be compared to each other (different atom
    count, or no geometry) are clustered in separate blocks rather than being
    forced into one tree with a fake huge distance — a mixed pool stays correct
    and the dendrogram keeps a readable Å scale.

    Each member is annotated exactly as the property path annotates it, so all
    downstream reporting (``.dat`` files, motifs, Boltzmann) works unchanged:
    ``_initial_cluster_label``, ``_rmsd_pass_origin`` and
    ``_first_rmsd_context_listing`` (RMSD of every member to its cluster's
    lowest-energy representative).

    Returns ``(clusters, linkage_matrix, linkage_members)``, clusters sorted by
    representative energy. The linkage is the one for the largest comparable
    block — the tree worth plotting — and *linkage_members* is the member order
    it was built from (leaf order for the dendrogram). Both are ``None`` / empty
    when no block has two or more members.
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # Import only when needed

    clusters: List[Cluster] = []
    linkage_matrix = None
    linkage_members: Cluster = []
    next_label = 1

    for block in rmsd_comparability_blocks(members, heavy_only=heavy_only):
        if len(block) < 2:
            for m in block:
                m['_initial_cluster_label'] = next_label
            clusters.append(list(block))
            next_label += 1
            continue

        condensed = rmsd_condensed_distances(block, heavy_only=heavy_only)
        block_linkage = linkage(condensed, method='average')
        block_labels = fcluster(block_linkage, t=rmsd_threshold, criterion='distance')

        if linkage_matrix is None:
            # First (largest) multi-member block — the tree worth plotting.
            linkage_matrix = block_linkage
            linkage_members = list(block)

        grouped: Dict[int, Cluster] = {}
        for member, label in zip(block, block_labels):
            grouped.setdefault(int(label), []).append(member)

        for label in sorted(grouped):
            for member in grouped[label]:
                member['_initial_cluster_label'] = next_label
            clusters.append(grouped[label])
            next_label += 1

    for cluster in clusters:
        _annotate_rmsd_cluster(cluster, mode, heavy_only=heavy_only)

    clusters.sort(key=lambda c: (min(sorting_energy(m, mode) for m in c),
                                 min(m['filename'] for m in c)))

    return clusters, linkage_matrix, linkage_members


def _annotate_rmsd_cluster(cluster: Cluster, mode: EnergyMode, heavy_only: bool = False) -> None:
    """Attach the representative RMSD listing consumed by the ``.dat`` writer."""
    if not cluster:
        return

    representative = min(cluster, key=lambda x: (sorting_energy(x, mode), x['filename']))

    listing = []
    for member in cluster:
        if member is representative:
            rmsd_val: Any = 0.0
        else:
            rmsd_val = _pair_rmsd(representative, member, heavy_only=heavy_only)
        listing.append({'filename': member['filename'], 'rmsd_to_rep': rmsd_val})

    for member in cluster:
        member['_rmsd_pass_origin'] = 'first_pass_validated'
        member['_first_rmsd_context_listing'] = listing
        member['_second_rmsd_sub_cluster_id'] = None
        member['_second_rmsd_context_listing'] = None
        member['_second_rmsd_rep_filename'] = None


__all__ = [
    "calculate_rmsd",
    "cluster_by_rmsd",
    "perform_second_rmsd_clustering",
    "post_process_clusters_with_rmsd",
    "rmsd_comparability_blocks",
    "rmsd_condensed_distances",
]
