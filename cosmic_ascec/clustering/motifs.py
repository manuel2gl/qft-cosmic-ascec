"""Motif folder and structure-XYZ writers — the final clustering outputs.

A *motif* is the lowest-energy representative of a cluster — the unique
conformational family COSMIC discovered. This module produces the on-disk
artifacts that downstream tools (and the user) consume:

* ``extracted_clusters/cluster_NN/`` — one ``.xyz`` per structure in each
  cluster (:func:`write_xyz_file`).
* A combined multi-frame ``.xyz`` per cluster, plus an optional ``.mol``
  conversion via OpenBabel (:func:`combine_xyz_files`).
* ``motifs_NN/`` — directory of cluster representatives, numbered by
  Boltzmann population (highest population → ``motif_01``), with a
  combined XYZ and a motif-level dendrogram
  (:func:`create_unique_motifs_folder`).

The choice of "lowest energy" for the representative depends on what
thermochemistry is available in the QM outputs — passed in via an
:class:`~cosmic_ascec.clustering.energies.EnergyMode`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from cosmic_ascec.clustering.console import print_step, vprint
from cosmic_ascec.clustering.energies import (
    EnergyMode,
    hartree_to_ev,
    hartree_to_kcal_mol,
    sorting_energy,
)
from cosmic_ascec.clustering.features.feature_spec import (
    CLUSTERING_NUMERICAL_FEATURES,
    FEATURE_MAPPING,
    HBOND_FEATURES,
    ROTATIONAL_CONSTANT_SUBFEATURES,
)
from cosmic_ascec.clustering.features.geometric import atomic_number_to_symbol
from cosmic_ascec.clustering.scaling import pool_has_hydrogen_bonds
from cosmic_ascec.file_formats.provenance import describe, load_mapping, update_mapping
from cosmic_ascec import levels as _levels

Record = MutableMapping[str, Any]
Cluster = List[Record]


def eligible_representatives(cluster_members: Cluster, *, dataset_has_freq: bool,
                             geometry_only: bool = False) -> Cluster:
    """Members of a cluster that may stand for it.

    No representative may carry imaginary frequencies or non-converged data, and
    it must have whatever quantity this run ranks on. The rule lives here, in
    one place, because the orchestrator has to apply exactly the same test when
    it decides which clusters get published and numbered — if the two ever
    disagree, a cluster is handed an id and then yields nothing, which is how
    the published sequence used to end up with holes in it (cluster_24 present
    with no motif_24 beside it).

    Geometry-only input (plain XYZ, no ``--sp``) has nothing to rank on, so the
    energy test is dropped there and the caller falls back to a geometric rule.
    In the energy-refinement stage a representative may carry only a composite
    Gibbs energy — the high-level single point has no frequencies of its own —
    so either form of Gibbs counts.
    """
    if geometry_only:
        return [m for m in cluster_members
                if not m.get('_has_imaginary_freqs', False)
                and m.get('_is_full_feature', True)]
    if dataset_has_freq:
        return [m for m in cluster_members
                if not m.get('_has_imaginary_freqs', False)
                and (m.get('gibbs_free_energy') is not None
                     or m.get('composite_gibbs') is not None)
                and m.get('_is_full_feature', True)]
    return [m for m in cluster_members
            if m.get('final_electronic_energy') is not None
            and m.get('_is_full_feature', True)]


#: Per-pass index of representatives, written beside them.
STAGE_INDEX_FILENAME = "stage_index.dat"


def write_stage_index(motifs_dir: str, output_prefix: str,
                      rows: Sequence[Tuple[str, Any, str, str]]) -> None:
    """Write ``stage_index.dat``: what each representative is, and where it came from.

    One row per representative, joining the four things that otherwise live in
    four different files: its label, the cluster it stands for, the structure it
    was carved from, and its ranking energy. Reading one of these per stage
    walks the whole chain back to the annealing geometry —
    ``u_motif_13 <- motif_12_opt``, then in the previous stage's index
    ``motif_12 <- candidate_25_opt``.

    The label alone was never enough for this. Before the levels were split, the
    last two passes both wrote ``umotif_NN``, so a chain step could only be
    resolved by knowing which directory a file sat in; the source column makes
    the link explicit and local.
    """
    if not rows:
        return
    try:
        path = os.path.join(motifs_dir, STAGE_INDEX_FILENAME)
        w_label = max(len(str(r[0])) for r in rows)
        w_label = max(w_label, len("# label"))
        w_src = max(max(len(str(r[2])) for r in rows), len("source"))
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(f"# COSMIC stage index - level '{output_prefix}'\n")
            fh.write("# Each row: this pass's representative, the cluster it "
                     "represents,\n")
            fh.write("# the structure it came from, and its ranking energy "
                     "(Hartree).\n")
            fh.write(f"{'# label'.ljust(w_label)}  {'cluster':>7}  "
                     f"{'source'.ljust(w_src)}  energy\n")
            for label, cluster_id, source, energy in rows:
                fh.write(f"{str(label).ljust(w_label)}  {str(cluster_id):>7}  "
                         f"{str(source).ljust(w_src)}  {energy}\n")
    except OSError as exc:
        # An index is a convenience; losing it must not sink a finished run.
        print(f"  WARNING: could not write {STAGE_INDEX_FILENAME}: {exc}")


def _level_of(output_prefix: str) -> "_levels.Level":
    """:class:`~cosmic_ascec.levels.Level` for a prefix, tolerating old names.

    Display strings and folder stems all hang off this, so an unrecognised
    prefix degrades to the bottom rung rather than raising in the middle of
    writing output.
    """
    return _levels.resolve(output_prefix) or _levels.CANDIDATE



def detect_motif_input_level(filenames: Sequence[str]) -> Tuple[str, str, bool]:
    """Guess which rung to emit from what the input files are called.

    Fallback only. A protocol run passes ``--level`` explicitly, because the
    runner knows which stage it just finished; this is for a bare
    ``cosmic <dir>`` invocation, where the filenames are the only clue.

    The rungs are ``candidate -> motif -> u_motif``
    (:mod:`cosmic_ascec.levels`), and each pass emits the one *above* whatever
    it was handed: unlabelled input (``conf_*``, plain xyz) is raw geometry and
    yields ``candidate``; ``candidate_*`` yields ``motif``; ``motif_*`` yields
    ``u_motif``. ``u_motif_*`` is the top, so it stays there and warns — a
    fourth clustering pass has nothing left to promote to, and silently reusing
    the label is what made ``umotif_23`` ambiguous across stages in the first
    place.

    Legacy ``umotif_*`` input counts as the top rung, since that is what the old
    two-rung scheme meant by it on output.

    Returns ``(output_prefix, folder_prefix, is_second_step)``.
    """
    if not filenames:
        return _levels.CANDIDATE.label, _levels.CANDIDATE.folder, False

    # Highest rung present wins: a folder mixing motif_* with a stray candidate_*
    # is still a motif-level folder. label_of() matches longest-first, so the
    # "motif" inside "u_motif" never registers as the middle rung.
    counts = {}
    for filename in filenames:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        label = _levels.label_of(base_name)
        if label:
            counts[label] = counts.get(label, 0) + 1

    if not counts:
        return _levels.CANDIDATE.label, _levels.CANDIDATE.folder, False

    highest = max(counts, key=lambda k: _levels.LEVELS.index(_levels.resolve(k)))
    current = _levels.resolve(highest)

    if current is _levels.U_MOTIF:
        print(f"  Warning: input is already at the top level "
              f"('{_levels.U_MOTIF.label}'); reclustering it emits that level "
              f"again, so labels will repeat across passes. Pass --level to "
              f"name this pass explicitly.")

    out = _levels.next_level(highest)
    return out.label, out.folder, out is not _levels.CANDIDATE


def pool_has_energies(all_clusters_data: Sequence[Cluster]) -> bool:
    """Return whether any structure anywhere carries a usable ranking energy.

    ``False`` means geometry-only input (plain XYZ with no ``--sp``). The
    energy-based representative rule cannot apply there, and without a fallback
    every cluster would be reported as having no converged minimum and the
    motifs folder would come out empty.
    """
    return any(
        member.get('final_electronic_energy') is not None
        or member.get('gibbs_free_energy') is not None
        or member.get('composite_gibbs') is not None
        for cluster in all_clusters_data
        for member in cluster
    )


def geometric_representative(cluster_members: Sequence[Record]) -> Record:
    """Pick the member closest to the cluster's centroid in descriptor space.

    Used when the pool carries no energy at all — the XYZ front-end without
    ``--sp``, where the usual lowest-energy rule has nothing to rank on. Public
    because ``thresholds.attach_pearson_to_rep`` selects the same representative
    for its similarity report: the ``.dat`` block must name the structure that
    becomes the motif, not a different one. The
    axes are the same clustering features the partition was built from, so the
    representative is central in the space the cluster actually lives in.

    Each descriptor is standardised across the cluster before the distance is
    taken, so a column with a large numeric range (V_NN, in Hartree) cannot
    outvote one with a small range purely through its units — the same
    normalisation the clustering distance metric uses. Members missing a
    descriptor that others have are penalised on that axis rather than
    excluded, so a cluster always yields a representative. Ties break on
    filename, keeping the choice reproducible.
    """
    columns: List[List[Optional[float]]] = []

    for name in CLUSTERING_NUMERICAL_FEATURES:
        key = FEATURE_MAPPING.get(name, name)
        columns.append([_as_float(m.get(key)) for m in cluster_members])
    for axis in range(3):
        column: List[Optional[float]] = []
        for member in cluster_members:
            rc = member.get('rotational_constants')
            column.append(
                _as_float(rc[axis])
                if rc is not None and hasattr(rc, '__len__') and len(rc) > axis
                else None
            )
        columns.append(column)

    distances = np.zeros(len(cluster_members), dtype=float)
    for column in columns:
        present = [v for v in column if v is not None]
        if len(present) < 2:
            continue
        mean = float(np.mean(present))
        std = float(np.std(present))
        if std <= 0.0:
            continue
        # A missing value sits one standard deviation off the centroid: enough
        # to disfavour it against a fully-described member, not enough to
        # dominate the sum.
        deviations = [
            abs(v - mean) / std if v is not None else 1.0
            for v in column
        ]
        distances += np.square(deviations)

    order = sorted(
        range(len(cluster_members)),
        key=lambda i: (distances[i], cluster_members[i]['filename']),
    )
    return cluster_members[order[0]]


def _as_float(value: Any) -> Optional[float]:
    """Coerce *value* to a finite float, or ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def representative_energy_comment(mol_data: Record, mode: EnergyMode) -> str:
    """Energy fragment for an XYZ comment line, e.g. ``G = -37.40 Hartree (...)``.

    Prefers the composite Gibbs energy (energy-refinement stage) when present,
    otherwise falls back to the stage's own Gibbs (freq mode) or electronic
    (opt-only) energy.  This mirrors :func:`sorting_energy`, so the value shown
    in the file matches the one used to rank the structure.
    """
    composite = mol_data.get('composite_gibbs')
    if composite is not None:
        label, value = 'G', composite
    elif mode.has_freq:
        label, value = 'G', mol_data.get('gibbs_free_energy')
    else:
        label, value = 'E', mol_data.get('final_electronic_energy')

    if value is None:
        # Nothing to say. A geometry-only run has no energy at all, and writing
        # "E = N/A" into every comment line of every file states that at length
        # without informing anyone. Callers omit the parenthetical entirely
        # when this is empty; with --sp, or with QM outputs, it comes back.
        return ""
    return (f"{label} = {value:.6f} Hartree "
            f"({hartree_to_kcal_mol(value):.2f} kcal/mol, {hartree_to_ev(value):.2f} eV)")


def structure_comment(mol_data: Record, mode: EnergyMode, *,
                      prefix: str = "", provenance: Optional[Mapping[str, Any]] = None) -> str:
    """The comment line for one structure's XYZ record.

    ``<prefix><name> (E = ...) | traj frame 3  t= 40000 ps``, with each part
    dropped when it has nothing to say: no energy without a QM result, no
    provenance outside a trajectory-derived run. Every writer of a structure
    goes through here so the four places COSMIC emits geometry cannot drift
    into four different formats.
    """
    name = os.path.splitext(os.path.basename(mol_data.get('filename', '')))[0]
    comment = f"{prefix}{name}"

    energy = representative_energy_comment(mol_data, mode)
    if energy:
        comment += f" ({energy})"

    trace = describe(provenance, mol_data.get('filename', ''))
    if trace:
        comment += f" | {trace}"
    return comment


def write_xyz_file(mol_data: Record, filename: str, mode: EnergyMode,
                   provenance: Optional[Mapping[str, Any]] = None) -> None:
    """
    Writes atomic coordinates to an XYZ file with energy in the comment line.
    Freq mode: Gibbs free energy (original).  Opt-only mode: electronic energy.
    Energy-refinement mode: composite Gibbs energy.

    Verbatim port of cosmic-v01's ``write_xyz_file`` (3157-3186); cosmic-v01's
    ``_DATASET_HAS_FREQ`` global becomes the explicit *mode* argument.
    """
    atomnos = mol_data.get('final_geometry_atomnos')
    atomcoords = mol_data.get('final_geometry_coords')

    if atomnos is None or atomcoords is None or len(atomnos) == 0:
        print(f"  WARNING: Cannot write XYZ for {os.path.basename(filename)}: Missing geometry data.")
        return

    comment_line = structure_comment(mol_data, mode, provenance=provenance)

    symbols = [atomic_number_to_symbol(n) for n in atomnos]

    with open(filename, 'w', newline='\n', encoding='utf-8') as f:
        f.write(f"{len(atomnos)}\n")
        f.write(f"{comment_line}\n")
        for i in range(len(atomnos)):
            f.write(f"{symbols[i]:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")


def create_unique_motifs_folder(
    all_clusters_data: Sequence[Cluster],
    output_base_dir: str,
    mode: EnergyMode,
    openbabel_alias: str = "obabel",
    cluster_id_mapping: Optional[Dict[int, int]] = None,
    output_prefix: str = 'motif',
    folder_prefix: str = 'motifs',
    boltzmann_data: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[int, int]:
    """
    Creates a motifs/umotifs folder containing the lowest energy representative
    structure from each cluster.

    Returns the ``{motif_number: cluster_id}`` mapping.

    Verbatim port of cosmic-v01's ``create_unique_motifs_folder`` (3188-3491);
    cosmic-v01's ``dataset_has_freq`` parameter and ``_sorting_energy`` global
    are both carried by *mode*.
    """
    dataset_has_freq = mode.has_freq

    if not all_clusters_data:
        print("  No clusters found. Skipping motifs creation.")
        return {}

    # Determine display name based on prefix
    display_name = _level_of(output_prefix).display

    representatives = []
    representative_cluster_ids = []

    # Geometry-only input (plain XYZ, no --sp) has no energy to rank on. The
    # energy predicate below would then reject every member and the folder
    # would come out empty, so the rule switches to a geometric one instead.
    geometry_only = not pool_has_energies(all_clusters_data)
    if geometry_only:
        print("  No energies available: representatives chosen as the structure "
              "closest to each cluster's centroid in descriptor space.")

    for cluster_idx, cluster_members in enumerate(all_clusters_data):
        if not cluster_members:
            continue

        valid_members = eligible_representatives(
            cluster_members, dataset_has_freq=dataset_has_freq,
            geometry_only=geometry_only)

        if not valid_members:
            # All members are invalid for representative selection.
            print(f"  WARNING: Cluster {cluster_idx + 1} has no converged minima - skipping motif creation")
            continue

        # Find the lowest energy representative from valid (non-imaginary) members only
        if geometry_only:
            representative = geometric_representative(valid_members)
        else:
            representative = min(valid_members,
                               key=lambda x: (sorting_energy(x, mode), x['filename']))

        # Get the cluster ID for this representative
        cluster_id = cluster_id_mapping[cluster_idx] if cluster_id_mapping else cluster_idx + 1

        representatives.append(representative)
        representative_cluster_ids.append(cluster_id)

    # Name the folder by the number of representatives ACTUALLY created — one per
    # cluster that yielded a converged minimum. Clusters whose only members are
    # non-converged / imaginary produce no representative (skipped above), so this
    # count matches the umotif files written and the reported motifs_created,
    # rather than the raw cluster count (which over-counts those empty clusters).
    num_motifs = len(representatives)
    motifs_dir = os.path.join(output_base_dir, f"{folder_prefix}_{num_motifs:02d}")
    os.makedirs(motifs_dir, exist_ok=True)

    print()
    print()
    print_step(f"Creating {num_motifs} {display_name} from cluster representatives...")
    vprint(f"  Output directory: {motifs_dir}")

    # Sort representatives by Boltzmann population (if available) or Gibbs free energy as fallback
    representatives_with_ids = list(zip(representatives, representative_cluster_ids))

    if boltzmann_data:
        # Create a filename-to-population mapping for sorting
        filename_to_population = {}
        for cluster_id_key, data in boltzmann_data.items():
            filename_to_population[data['filename']] = data['population']

        def get_population_for_rep(rep_tuple):
            """Get Boltzmann population for a representative, or -inf if not found (to sort last)."""
            rep, cid = rep_tuple
            filename = rep.get('filename', '')
            return filename_to_population.get(filename, -float('inf'))

        # Sort by population descending (highest population = motif_01)
        sorted_representatives_with_ids = sorted(
            representatives_with_ids,
            key=lambda x: (-get_population_for_rep(x), x[0]['filename'])  # Negative for descending
        )
    else:
        # Fallback: sort by electronic energy (lowest = motif_01)
        sorted_representatives_with_ids = sorted(
            representatives_with_ids,
            key=lambda x: (sorting_energy(x[0], mode), x[0]['filename'])
        )

    # Trajectory provenance, when this run came from one. load_mapping returns
    # None for every ordinary COSMIC run, so the comment lines below are built
    # exactly as they always were unless a mapping.dat sits in the output dir.
    provenance = load_mapping(output_base_dir)
    if provenance is not None:
        cluster_of = {}
        for cluster_idx, members in enumerate(all_clusters_data):
            resolved = cluster_id_mapping[cluster_idx] if cluster_id_mapping else cluster_idx + 1
            for member in members:
                cluster_of[member.get('filename', '')] = resolved
        motif_of = {rep['filename']: idx for idx, (rep, _)
                    in enumerate(sorted_representatives_with_ids, 1)}
        update_mapping(output_base_dir, cluster_of, motif_of)
        # Re-read so the records carry the columns just written, and the motif
        # files below can quote a structure's cluster as well as its frame.
        provenance = load_mapping(output_base_dir)

    stage_index_rows = []

    for motif_idx, (representative, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
        base_name = os.path.splitext(representative['filename'])[0]

        # One naming rule for every rung: <label>_<rank>.xyz, where the rank is
        # this pass's own cluster ordering. The previous rule only cleaned up
        # the top rung and, for the others, either kept the *input's* number
        # when it happened to equal the new rank or glued both together
        # ("motif_03_motif_17_opt.xyz"). Both variants make the number on the
        # file mean something other than the cluster it represents, which is
        # precisely the confusion this is meant to end. The source structure is
        # not lost — it goes in the XYZ comment line below, where it cannot be
        # mistaken for an index.
        motif_filename = f"{output_prefix}_{motif_idx:02d}.xyz"

        motif_path = os.path.join(motifs_dir, motif_filename)

        write_xyz_file(representative, motif_path, mode, provenance=provenance)

        display_prefix = _level_of(output_prefix).display_one.title()
        if dataset_has_freq:
            gibbs_str = f"{representative['gibbs_free_energy']:.6f}" if representative.get('gibbs_free_energy') is not None else "N/A"
            vprint(f"  {display_prefix} {motif_idx:02d}: {base_name} (Gibbs Energy: {gibbs_str} Hartree, Cluster ID: {cluster_id})")
        else:
            elec_str = f"{representative['final_electronic_energy']:.6f}" if representative.get('final_electronic_energy') is not None else "N/A"
            vprint(f"  {display_prefix} {motif_idx:02d}: {base_name} (Electronic Energy: {elec_str} Hartree, Cluster ID: {cluster_id})")

        _e = (representative.get('composite_gibbs')
              if representative.get('composite_gibbs') is not None
              else representative.get('gibbs_free_energy')
              if representative.get('gibbs_free_energy') is not None
              else representative.get('final_electronic_energy'))
        stage_index_rows.append((
            f"{output_prefix}_{motif_idx:02d}", cluster_id, base_name,
            "-" if _e is None else f"{_e:.6f}",
        ))

    # Use appropriate filename based on prefix
    combined_xyz_filename = f"all_{folder_prefix}_combined.xyz"
    combined_xyz_path = os.path.join(motifs_dir, combined_xyz_filename)

    with open(combined_xyz_path, "w", newline='\n', encoding='utf-8') as outfile:
        for motif_idx, (rep_data, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
            atomnos = rep_data.get('final_geometry_atomnos')
            atomcoords = rep_data.get('final_geometry_coords')

            if atomnos is None or atomcoords is None or len(atomnos) == 0:
                print(f"    WARNING: Skipping representative {rep_data['filename']} due to missing geometry data.")
                continue

            base_name = os.path.splitext(rep_data['filename'])[0]
            # Every rung records where it came from. This used to be top-rung
            # only, which is why a mid-protocol file could not be traced back
            # to the structure it represents without opening the stage folder
            # it happened to live in.
            energy_comment = representative_energy_comment(rep_data, mode)
            source = f"from {base_name}"
            detail = f"{source}, {energy_comment}" if energy_comment else source
            comment_line = f"{output_prefix}_{motif_idx:02d} ({detail})"
            trace = describe(provenance, rep_data.get('filename', ''))
            if trace:
                comment_line += f" | {trace}"

            outfile.write(f"{len(atomnos)}\n")
            outfile.write(f"{comment_line}\n")
            for i in range(len(atomnos)):
                symbol = atomic_number_to_symbol(atomnos[i])
                outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")

    # Return the mapping from motif number to cluster ID for use in Boltzmann analysis
    motif_to_cluster_mapping = {}
    for motif_idx, (rep_data, cluster_id) in enumerate(sorted_representatives_with_ids, 1):
        motif_to_cluster_mapping[motif_idx] = cluster_id

    vprint(f"  Created combined XYZ file: {os.path.basename(combined_xyz_path)}")

    # Attempt to create MOL file using OpenBabel
    mol_filename = f"all_{folder_prefix}_combined.mol"
    mol_output_path = os.path.join(motifs_dir, mol_filename)
    openbabel_full_path = shutil.which(openbabel_alias)

    if openbabel_full_path:
        try:
            # Use the correct OpenBabel syntax: obabel -i<format> input_file -o<format> -O output_file
            # Resolved path, not the bare alias: conda-forge ships obabel on
            # Windows as a .bat shim that shutil.which finds (via PATHEXT) but
            # subprocess cannot launch by name. The sibling call site below
            # already used openbabel_full_path; this one did not.
            result = subprocess.run([openbabel_full_path, "-ixyz", combined_xyz_path, "-omol", "-O", mol_output_path],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                vprint(f"  Successfully created MOL file: {os.path.basename(mol_output_path)}")
            else:
                print(f"  WARNING: OpenBabel conversion to MOL failed. Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: OpenBabel conversion to MOL timed out after 30 seconds.")
        except Exception as e:
            print(f"  WARNING: Error during OpenBabel conversion to MOL: {e}")
    else:
        print(f"  WARNING: OpenBabel ({openbabel_alias}) not found. Skipping MOL conversion.")
        print("  Please ensure OpenBabel is installed and added to your system's PATH.")

    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        if len(sorted_representatives_with_ids) > 1:
            # Get representatives data for clustering using complete feature set (same as main clustering)
            representatives_data = []
            motif_labels = []

            _freq_dep = {'gibbs_free_energy', 'first_vib_freq', 'last_vib_freq'}
            _all_num_features = [
                'electronic_energy', 'gibbs_free_energy', 'homo_energy', 'homo_lumo_gap',
                'dipole_moment', 'vnn_nuclear_repulsion',
                'first_vib_freq', 'last_vib_freq',
                'num_hydrogen_bonds', 'average_hbond_distance', 'std_hbond_distance',
                'average_hbond_angle'
            ]
            all_potential_numerical_features = _all_num_features if dataset_has_freq else [f for f in _all_num_features if f not in _freq_dep]
            # No representative has a hydrogen bond: the four H-bond descriptors
            # are constant and were not clustering features in the main run
            # either (orchestrator's _pool_has_hbonds). Keep this dendrogram's
            # feature set matching the one that produced the motifs.
            if not pool_has_hydrogen_bonds([rep_data for rep_data, _ in sorted_representatives_with_ids]):
                all_potential_numerical_features = [f for f in all_potential_numerical_features
                                                    if f not in HBOND_FEATURES]
            rotational_constant_subfeatures = ROTATIONAL_CONSTANT_SUBFEATURES

            # Check which features are globally available across all representatives
            globally_missing_features = []
            for feature in all_potential_numerical_features:
                internal_key = FEATURE_MAPPING.get(feature, feature)
                if all(d.get(internal_key) is None for d in [rep_data for rep_data, _ in sorted_representatives_with_ids]):
                    globally_missing_features.append(feature)

            # Check rotational constants availability
            is_rot_const_globally_missing = True
            for rep_data, _ in sorted_representatives_with_ids:
                rot_consts = rep_data.get('rotational_constants')
                if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                    is_rot_const_globally_missing = False
                    break

            if is_rot_const_globally_missing:
                globally_missing_features.extend(rotational_constant_subfeatures)

            active_features = [f for f in all_potential_numerical_features if f not in globally_missing_features]

            for motif_idx, (rep_data, motif_id) in enumerate(sorted_representatives_with_ids, 1):
                # Build feature vector using same logic as main clustering
                feature_vector = []

                # Add standard numerical features
                for feature_name in active_features:
                    value = rep_data.get(feature_name)
                    if value is None:
                        value = 0.0
                    feature_vector.append(value)

                # Add rotational constants if available
                if not is_rot_const_globally_missing:
                    rot_consts = rep_data.get('rotational_constants')
                    if rot_consts is not None and isinstance(rot_consts, np.ndarray) and len(rot_consts) == 3:
                        feature_vector.extend([rot_consts[0], rot_consts[1], rot_consts[2]])
                    else:
                        feature_vector.extend([0.0, 0.0, 0.0])

                if feature_vector:
                    representatives_data.append(feature_vector)
                    # Use just the motif number (e.g., "01" instead of "motif_01")
                    motif_labels.append(f"{motif_idx:02d}")

            if len(representatives_data) > 1:

                linkage_matrix = linkage(representatives_data, method='average', metric='euclidean')

                plt.figure(figsize=(12, 8))
                dendrogram(linkage_matrix, labels=motif_labels, orientation='top',
                          distance_sort=True, show_leaf_counts=True)
                # Use appropriate title based on prefix
                dendrogram_title = _level_of(output_prefix).display.title()
                plt.title(f'{dendrogram_title} Dendrogram')
                plt.xlabel(dendrogram_title)
                plt.ylabel('Distance')
                plt.xticks(rotation=0)  # Keep horizontal since labels are short
                plt.tight_layout()

                # Save dendrogram in the motifs directory
                dendrogram_filename = f"{folder_prefix}_dendrogram.png"
                dendrogram_path = os.path.join(motifs_dir, dendrogram_filename)
                plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  Created {folder_prefix} dendrogram: {os.path.basename(dendrogram_path)}")

    except ImportError:
        print(f"  WARNING: matplotlib not available. Skipping {folder_prefix} dendrogram creation.")
    except Exception as e:
        print(f"  WARNING: Error creating {folder_prefix} dendrogram: {e}")

    write_stage_index(motifs_dir, output_prefix, stage_index_rows)

    display_name = _level_of(output_prefix).display.capitalize()
    print_step(f"{display_name} created: {len(sorted_representatives_with_ids)} representatives saved to {os.path.basename(motifs_dir)}\n")

    return motif_to_cluster_mapping


def combine_xyz_files(
    cluster_members_data: Cluster,
    input_dir: str,
    mode: EnergyMode,
    output_base_name: Optional[str] = None,
    openbabel_alias: str = "obabel",
    prefix_template: Optional[str] = None,
    motif_numbers: Optional[Sequence[int]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Combines relevant .xyz data from cluster members into a single multi-frame
    .xyz file and attempts to convert the resulting file to a .mol file.

    Verbatim port of cosmic-v01's ``combine_xyz_files`` (3494-3610); cosmic-v01's
    ``_DATASET_HAS_FREQ`` / ``_sorting_energy`` globals become *mode*.
    """
    final_xyz_source_path = None  # This will be the path to the XYZ file used for MOL conversion

    if not cluster_members_data:
        return

    if len(cluster_members_data) == 1:
        # For a single configuration, the XYZ file has already been written by write_xyz_file.
        single_mol_data = cluster_members_data[0]
        original_filename_base = os.path.splitext(single_mol_data['filename'])[0]
        final_xyz_source_path = os.path.join(input_dir, f"{original_filename_base}.xyz")
        # The output_base_name for MOL should be the original filename base
        final_output_mol_name_base = original_filename_base
        vprint(f"  Single configuration found in cluster. Using existing '{os.path.basename(final_xyz_source_path)}' for .mol conversion.")

    else:
        # For multiple configurations, create a new combined multi-frame XYZ file.
        if output_base_name is None:

            output_base_name = "combined_cluster"

        full_combined_xyz_path = os.path.join(input_dir, f"{output_base_name}.xyz")
        final_output_mol_name_base = output_base_name  # Base name for the .mol file

        # Sort members by Gibbs free energy (lowest to highest), with filename as a tie-breaker
        if motif_numbers and len(motif_numbers) == len(cluster_members_data):

            paired_data = list(zip(cluster_members_data, motif_numbers))
            sorted_pairs = sorted(
                paired_data,
                key=lambda x: (sorting_energy(x[0], mode), x[0]['filename'])
            )
            sorted_members_data = [pair[0] for pair in sorted_pairs]
            sorted_motif_numbers = [pair[1] for pair in sorted_pairs]
        else:
            sorted_members_data = sorted(
                cluster_members_data,
                key=lambda x: (sorting_energy(x, mode), x['filename'])
            )
            sorted_motif_numbers = None

        with open(full_combined_xyz_path, "w", newline='\n', encoding='utf-8') as outfile:
            for frame_idx, mol_data in enumerate(sorted_members_data, 1):  # Iterate over sorted data
                atomnos = mol_data.get('final_geometry_atomnos')
                atomcoords = mol_data.get('final_geometry_coords')
                if atomnos is None or atomcoords is None or len(atomnos) == 0:
                    print(f"    WARNING: Skipping {mol_data['filename']} in combined XYZ due to missing geometry data.")
                    continue

                # Apply prefix template with actual motif number if provided
                if prefix_template and sorted_motif_numbers:
                    prefix = prefix_template.format(sorted_motif_numbers[frame_idx - 1])
                elif prefix_template:
                    prefix = prefix_template.format(frame_idx)
                else:
                    prefix = ""
                comment_line = structure_comment(mol_data, mode, prefix=prefix,
                                                 provenance=provenance)

                outfile.write(f"{len(atomnos)}\n")
                outfile.write(f"{comment_line}\n")
                for i in range(len(atomnos)):
                    symbol = atomic_number_to_symbol(atomnos[i])
                    outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")

        vprint(f"  Successfully created combined multi-frame .xyz file: '{os.path.basename(full_combined_xyz_path)}'")
        final_xyz_source_path = full_combined_xyz_path

    # Section for Open Babel Integration (always attempts for MOL conversion)
    if final_xyz_source_path:
        mol_output_filename = f"{final_output_mol_name_base}.mol"
        full_mol_output_path = os.path.join(input_dir, mol_output_filename)

        openbabel_full_path = shutil.which(openbabel_alias)
        openbabel_installed = False

        if openbabel_full_path:
            openbabel_installed = True
        else:
            print(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion.")
            print("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.")
            print(f"  You can change the alias using the 'openbabel_alias' parameter in the function call, e.g., combine_xyz_files(..., openbabel_alias='obabel').")

        if openbabel_installed:
            try:
                conversion_command = [openbabel_full_path, "-i", "xyz", final_xyz_source_path, "-o", "mol", "-O", full_mol_output_path]
                subprocess.run(conversion_command, check=True, capture_output=True, text=True)

                if os.path.exists(full_mol_output_path):
                    vprint(f"  Successfully converted '{os.path.basename(final_xyz_source_path)}' to '{os.path.basename(full_mol_output_path)}' using Open Babel.")

            except subprocess.CalledProcessError as e:
                print(f"  Open Babel conversion failed for '{os.path.basename(final_xyz_source_path)}'.")
                print(f"  Error details: {e.stderr.strip()}")
            except Exception as e:
                print(f"  An unexpected error occurred during Open Babel conversion for '{final_xyz_source_path}': {e}")


__all__ = [
    "combine_xyz_files",
    "create_unique_motifs_folder",
    "detect_motif_input_level",
    "geometric_representative",
    "pool_has_energies",
    "write_xyz_file",
]
