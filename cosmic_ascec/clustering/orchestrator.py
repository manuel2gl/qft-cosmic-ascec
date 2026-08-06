"""Top-level driver for the COSMIC clustering pipeline.

This module owns the single entry point ``perform_clustering_and_analysis``,
which threads every step of the pipeline together. The flow is:

1. Discover candidate QM output files (``cosmic_*.out``) in the run dir.
2. Extract feature vectors in parallel via
   :mod:`~cosmic_ascec.clustering.features.extractor`.
3. Apply quality-control filters (drop structures with imaginary frequencies
   or non-converged geometries) via :mod:`~cosmic_ascec.clustering.filters`.
4. Build a feature matrix, Z-standardise it, apply per-feature weights
   (:mod:`~cosmic_ascec.clustering.scaling`).
5. Run UPGMA hierarchical clustering; pick a cut height
   (:mod:`~cosmic_ascec.clustering.thresholds`); slice into clusters.
6. Absorb structures that were missing some features back into the full
   clusters via the reduced-tier matcher
   (:mod:`~cosmic_ascec.clustering.matching`).
7. Pick a representative per cluster, compute Boltzmann populations
   (:mod:`~cosmic_ascec.clustering.energies`), and write the
   ``clustering_summary.txt`` / ``boltzmann_distribution.txt`` reports.
8. Materialise per-motif directories and ``.xyz`` files
   (:mod:`~cosmic_ascec.clustering.motifs`).

This is intentionally one large function — the steps are tightly coupled
through shared dataframes and indexing schemes, and splitting it would
require multiplying intermediate types without clarifying anything. To
*change* a step, edit the corresponding submodule; this file only sequences
them.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys

import numpy as np

from cosmic_ascec.clustering import console
from cosmic_ascec.clustering.composite_energies import apply_composite_energies
from cosmic_ascec.clustering.console import print_step, version, vprint
from cosmic_ascec.clustering.dat_writer import write_cluster_dat_file
from cosmic_ascec.clustering.dendrogram import plot_annotated_dendrogram
from cosmic_ascec.clustering.energies import (
    BOLTZMANN_CONSTANT_HARTREE_PER_K,
    EnergyMode,
    hartree_to_ev,
    hartree_to_kcal_mol,
    sorting_energy,
)
from cosmic_ascec.clustering.features.extractor import process_file_parallel_wrapper
from cosmic_ascec.clustering.features.feature_spec import (
    CLUSTERING_NUMERICAL_FEATURES,
    FEATURE_MAPPING,
    HBOND_GEOMETRY_FEATURES,
)
from cosmic_ascec.clustering.features.xtb_sp import (
    merge_single_point_properties,
    run_single_points,
)
from cosmic_ascec.clustering.features.xyz_input import (
    FRAMES_SUBDIR,
    XYZ_EXTENSION,
    XYZ_GLOB,
    explode_multiframe_xyz,
    is_xyz_byproduct,
    resolve_optimized_duplicates,
)
from cosmic_ascec.clustering.filters import (
    filter_imaginary_freq_structures,
    filter_non_converged_structures,
    filter_redundant_reduced_structures,
    save_non_converged_critical_structures,
    save_redundant_reduced_structures,
)
from cosmic_ascec.clustering.matching import match_reduced_to_clusters
from cosmic_ascec.clustering.motifs import (
    combine_xyz_files,
    create_unique_motifs_folder,
    detect_motif_input_level,
    write_xyz_file,
)
from cosmic_ascec.clustering.rmsd import (
    calculate_rmsd,
    cluster_by_rmsd,
    perform_second_rmsd_clustering,
    post_process_clusters_with_rmsd,
)
from cosmic_ascec.clustering.scaling import (
    apply_weights,
    build_feature_vectors,
    effective_n_features,
    group_has_any_clustering_feature_data,
    has_valid_rotational_constants,
    is_valid_scalar,
    select_complete_group_scalar_features,
    zscore_scale,
)
from cosmic_ascec.clustering.thresholds import (
    attach_pearson_to_rep,
    compute_knee_threshold,
    compute_mojena_threshold,
    resolve_clustering_threshold,
    threshold_entry,
)


def _stage_xyz_selection(paths, workdir):
    """Copy the chosen structures into *workdir* and return it.

    Used when discovery narrowed the input set but nothing had to be split.
    The pipeline re-globs its input folder several times downstream, so the
    only way to make a narrowed selection stick is to give it a folder that
    contains exactly the selection.
    """
    import shutil

    os.makedirs(workdir, exist_ok=True)
    for path in paths:
        target = os.path.join(workdir, os.path.basename(path))
        if os.path.abspath(target) != os.path.abspath(path):
            shutil.copyfile(path, target)
    return workdir


# cosmic-v01.py line 87 — CPU-count cache for get_cpu_count_fast.
_CPU_COUNT_CACHE = None


def get_cpu_count_fast():
    """
    Get CPU count using fast methods with caching.
    Tries multiple detection methods for maximum compatibility.

    Returns:
        int: Number of available CPU cores

    Verbatim port of cosmic-v01's ``get_cpu_count_fast`` (lines 180-234).
    """
    global _CPU_COUNT_CACHE

    if _CPU_COUNT_CACHE is not None:
        return _CPU_COUNT_CACHE

    # Try os.cpu_count() first (fastest method)
    try:
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 0:
            _CPU_COUNT_CACHE = cpu_count
            return cpu_count
    except (OSError, AttributeError):
        pass

    # Try /proc/cpuinfo (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_count = sum(1 for line in f if line.startswith('processor'))
            if cpu_count > 0:
                _CPU_COUNT_CACHE = cpu_count
                return cpu_count
    except (FileNotFoundError, IOError):
        pass

    # Try nproc command (Linux)
    try:
        result = subprocess.run(['nproc'], capture_output=True, text=True, timeout=1.0)
        if result.returncode == 0:
            cpu_count = int(result.stdout.strip())
            if cpu_count > 0:
                _CPU_COUNT_CACHE = cpu_count
                return cpu_count
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # Use multiprocessing.cpu_count() as fallback
    try:
        cpu_count = mp.cpu_count()
        if cpu_count > 0:
            _CPU_COUNT_CACHE = cpu_count
            return cpu_count
    except (OSError, AttributeError):
        pass

    # Final fallback
    _CPU_COUNT_CACHE = 24
    return 24


# Modified to accept rmsd_threshold and output_base_dir
def perform_clustering_and_analysis(input_source, threshold="auto", file_extension_pattern=None, rmsd_threshold=None, output_base_dir=None, force_reprocess_cache=False, weights=None, is_compare_mode=False, min_std_threshold=1e-6, abs_tolerances=None, num_cores=None, temperature_k=298.15, group_hb=False, prev_out_dir=None, partialweights=False, rmsd_only=False, rmsd_heavy=False, sp_method=None, sp_charge=None, sp_uhf=None):
    """
    Performs hierarchical clustering and comprehensive analysis on molecular structures.
    This is the main analysis function that orchestrates the entire clustering workflow.

    Verbatim port of cosmic-v01's ``perform_clustering_and_analysis`` (4853-6294).
    cosmic-v01's ``_DATASET_HAS_FREQ`` / ``_DATASET_HAS_COMPOSITE`` /
    ``_sorting_energy`` globals are carried by the local :class:`EnergyMode`
    *mode* (**D-007**); ``VERBOSE`` is :data:`console.VERBOSE`.
    """
    from sklearn.preprocessing import StandardScaler  # type: ignore # Import only when needed
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  # type: ignore # Import only when needed
    import matplotlib.pyplot as plt  # type: ignore # Import only when needed


    if num_cores is None:
        num_cores = get_cpu_count_fast()
    """
    Performs hierarchical clustering and analysis on the extracted molecular properties,
    and saves .dat and .xyz files for each cluster.
    Includes an optional RMSD post-processing step and caching of extracted data.
    Output files will be saved relative to output_base_dir.
    `input_source` can be a folder path (normal mode) or a list of file paths (compare mode).
    `weights` is a dictionary mapping feature names (user-friendly) to their weights.
    `min_std_threshold` (float): Minimum standard deviation for a feature to be scaled.
                                 Features with std dev below this are treated as constant (0.0).
    `abs_tolerances` (dict): Dictionary of feature_name: absolute_tolerance. If the max difference
                             for a feature within a group is less than its tolerance, it's zeroed out.
    """
    # Default weights for all clustering features
    # These can be adjusted using the --weights flag, e.g., --weights "(first_vib_freq=1.0)(homo_lumo_gap=1.5)"

    # Available features:
    # - electronic_energy: Final electronic energy (Hartree)
    # - gibbs_free_energy: Gibbs free energy (Hartree)
    # - homo_energy: HOMO energy (Hartree)
    # - homo_lumo_gap: HOMO-LUMO gap (Hartree)
    # - dipole_moment: Dipole moment (Debye)
    # - vnn_nuclear_repulsion: Nuclear-nuclear repulsion energy V_NN (Hartree)
    # - rotational_constants_A/B/C: Rotational constants (cm^-1)
    # - first_vib_freq: First vibrational frequency (cm⁻¹)
    # - last_vib_freq: Last vibrational frequency (cm⁻¹)
    # - num_hydrogen_bonds: Number of hydrogen bonds
    # - average_hbond_distance: Average hydrogen bond distance (Å)
    # - std_hbond_distance: Std-dev of hydrogen bond distances (Å)
    # - average_hbond_angle: Average hydrogen bond angle (degrees)
    default_weights = {
        'electronic_energy': 1.0,          # Final electronic energy
        'gibbs_free_energy': 1.0,          # Gibbs free energy
        'homo_energy': 1.0,                # HOMO energy
        'homo_lumo_gap': 1.0,              # HOMO-LUMO gap
        'dipole_moment': 1.0,              # Dipole moment
        'vnn_nuclear_repulsion': 1.0,      # V_NN nuclear repulsion
        'rotational_constants_A': 1.0,     # Rotational constant A
        'rotational_constants_B': 1.0,     # Rotational constant B
        'rotational_constants_C': 1.0,     # Rotational constant C
        'first_vib_freq': 1.0,             # First vibrational frequency
        'last_vib_freq': 1.0,              # Last vibrational frequency
        'num_hydrogen_bonds': 1.0,         # Number of hydrogen bonds
        'average_hbond_distance': 1.0,     # Average hydrogen bond distance
        'std_hbond_distance': 1.0,         # Std-dev of hydrogen bond distances
        'average_hbond_angle': 1.0,        # Average hydrogen bond angle
    }

    # Track which features the user explicitly set via --weights
    _user_explicit_weights = set(weights.keys()) if weights else set()

    if weights is None:
        weights = default_weights.copy() # Use default weights if none provided
    else:
        # Merge user weights with defaults, user weights take priority
        merged_weights = default_weights.copy()
        merged_weights.update(weights)
        weights = merged_weights

    if abs_tolerances is None:
        abs_tolerances = {} # Ensure abs_tolerances is a dict if not provided

    # --rmsd-only: skip the physicochemical feature vector entirely and build the
    # partition from heavy-atom RMSD alone (cut in Å). Comparison mode always
    # forces a single output cluster, so the geometry-only partition is moot
    # there — it still reports the RMSD context in the .dat file.
    if rmsd_only:
        if is_compare_mode:
            rmsd_only = False
        elif rmsd_threshold is None:
            rmsd_threshold = 1.0

    # Ensure output_base_dir is set, default to current working directory if None
    if output_base_dir is None:
        output_base_dir = os.getcwd()

    # NEW: Adjust output_base_dir for comparison mode and handle unique naming
    if is_compare_mode:
        base_comparison_dir_name = "comparison"
        final_comparison_dir = base_comparison_dir_name
        counter = 0
        while os.path.exists(os.path.join(output_base_dir, final_comparison_dir)):
            counter += 1
            final_comparison_dir = f"{base_comparison_dir_name}_{counter}"
        output_base_dir = os.path.join(output_base_dir, final_comparison_dir)
        os.makedirs(output_base_dir, exist_ok=True) # Ensure this new base directory exists
        print(f"  Comparison mode: All outputs will be placed in '{output_base_dir}'")
    else:
        # For normal mode, output directly to the working directory (no subfolder)
        os.makedirs(output_base_dir, exist_ok=True) # Ensure this base directory exists
        print(f"  All outputs will be placed in the current working directory")


    # --- Normalise XYZ input to one structure per file ---------------------
    # COSMIC is built on a one-file-one-record contract: the cache is keyed by
    # filename, skipped structures are copied by looking their filename up on
    # disk, and motif folders are named after the source file. A concatenated
    # ensemble or a trajectory therefore gets split into single-frame files
    # here, up front, rather than teaching every downstream layer about frame
    # indices. Directories of one-structure files are left completely alone.
    xyz_mode = str(file_extension_pattern) == XYZ_GLOB or (
        is_compare_mode
        and isinstance(input_source, list)
        and all(str(p).lower().endswith(XYZ_EXTENSION) for p in input_source)
    )

    if xyz_mode:
        frames_dir = os.path.join(output_base_dir, FRAMES_SUBDIR)
        if is_compare_mode:
            input_source, _ = explode_multiframe_xyz(input_source, frames_dir)
        else:
            source_xyz = [
                p for p in glob.glob(os.path.join(str(input_source), XYZ_GLOB))
                if not is_xyz_byproduct(p)
            ]
            # An xTB input and its own optimised result describe one structure
            # twice; keep the optimised one.
            source_xyz, _superseded = resolve_optimized_duplicates(source_xyz)
            if _superseded:
                print(f"  {len(_superseded)} xTB input geometry(ies) superseded by their "
                      f"own .xtbopt.xyz result; clustering the optimised structures.")

            _resolved, _materialised = explode_multiframe_xyz(source_xyz, frames_dir)
            if _materialised or _superseded:
                if not _materialised:
                    # Nothing was split, but the folder still holds the inputs
                    # we just discarded. Stage the survivors so the globs that
                    # follow cannot pick them back up.
                    _materialised = _stage_xyz_selection(_resolved, frames_dir)
                # The frames directory now holds every structure, so the rest
                # of the pipeline can simply glob it like any input folder.
                input_source = _materialised

    # Provide early feedback before expensive operations
    if not is_compare_mode:
        print_step("Initializing data extraction...")

    # Define a generic cache file path with random seed (like ASCEC protocol cache)
    # Check for existing cache file first.
    # (cosmic-v01 re-imported glob here; that shadowed the module-level import
    # and made every earlier use in this function an UnboundLocalError.)
    existing_caches = glob.glob(os.path.join(output_base_dir, "data_cache_*.pkl"))

    if existing_caches:
        # Use the most recent cache file if multiple exist
        cache_file_path = max(existing_caches, key=os.path.getmtime)
        vprint(f"Found existing cache file: {os.path.basename(cache_file_path)}")
    else:
        # Create new cache file with random seed
        import random
        cache_seed = random.randint(0, 999999)
        cache_file_name = f"data_cache_{cache_seed:06d}.pkl"
        cache_file_path = os.path.join(output_base_dir, cache_file_name)

    all_extracted_data = []
    skipped_files = set()  # Initialize skipped_files early to avoid unbound variable warnings

    files_to_process = []
    if is_compare_mode:
        files_to_process = input_source # input_source is already the list of files
        # For compare mode, we should probably bypass cache loading/saving, or make it specific to the comparison.
        # For now, let's assume compare mode always re-processes the two files.
        print(f"Starting parallel data extraction for comparison mode from {len(files_to_process)} files...")

        # Use parallel processing for comparison mode
        effective_cores = num_cores  # Use all available cores
        print(f"  Using {effective_cores} CPU cores for parallel processing")

        with mp.Pool(processes=effective_cores) as pool:
            results = pool.map(process_file_parallel_wrapper, sorted(files_to_process))

        # Process results
        for success, extracted_props, filename in results:
            if success and extracted_props:
                all_extracted_data.append(extracted_props)
            elif not success:
                skipped_files.add(filename)
    else:

        # INCREMENTAL CACHE UPDATE MODE (for redo)
        update_cache_file = None
        incremental_update_done = False  # Flag to track if update happened

        if len(sys.argv) > 1:  # Check if there are arguments
            for i, arg in enumerate(sys.argv):
                if arg == '--update-cache' and i + 1 < len(sys.argv):
                    update_cache_file = sys.argv[i + 1]
                    break

        if update_cache_file and os.path.exists(update_cache_file):
            # Read list of files to update
            with open(update_cache_file, 'r', encoding='utf-8') as f:
                basenames_to_update = {line.strip() for line in f if line.strip()}

            # Determine file extension (check if .out or .log exists for first basename)
            file_ext = None
            if basenames_to_update:
                first_basename = next(iter(basenames_to_update))
                for candidate_ext in ('.out', '.log', XYZ_EXTENSION):
                    if os.path.exists(os.path.join(str(input_source), first_basename + candidate_ext)):
                        file_ext = candidate_ext
                        break

            if file_ext:
                # Load existing cache
                cache_exists = os.path.exists(cache_file_path)
                if cache_exists:
                    with open(cache_file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    if isinstance(cached_data, list):
                        successful_data = cached_data
                        skipped_files = set()
                    else:
                        successful_data = cached_data.get('successful', [])
                        skipped_files = set(cached_data.get('skipped', []))
                else:
                    successful_data = []
                    skipped_files = set()

                # Remove old data for files being updated
                filenames_to_update = {b + file_ext for b in basenames_to_update}
                successful_data = [d for d in successful_data if d['filename'] not in filenames_to_update]
                skipped_files = skipped_files - filenames_to_update

                # Reprocess only the specified files
                files_to_reprocess = [os.path.join(str(input_source), f) for f in filenames_to_update
                                     if os.path.exists(os.path.join(str(input_source), f))]

                effective_cores = min(num_cores, len(files_to_reprocess)) if len(files_to_reprocess) > 0 else 1

                with mp.Pool(processes=effective_cores) as pool:
                    results = pool.map(process_file_parallel_wrapper, sorted(files_to_reprocess))

                # Update with new data
                for success, extracted_props, filename in results:
                    if success and extracted_props:
                        successful_data.append(extracted_props)
                    elif not success:
                        skipped_files.add(filename)

                # Save updated cache
                cache_data_to_save = {
                    'successful': successful_data,
                    'skipped': list(skipped_files)
                }
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(cache_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

                all_extracted_data = successful_data
                incremental_update_done = True
                print(f"    Cache updated with {len(results)} file(s)")
            else:
                print(f"  Warning: Could not determine file extension for incremental update")
                force_reprocess_cache = True  # Fallback to full reprocess

        # Existing cache logic for normal mode (skip if incremental update was done)
        if not incremental_update_done:
            if os.path.exists(cache_file_path) and not force_reprocess_cache:
                print(f"Attempting to load data from cache: '{os.path.basename(cache_file_path)}'")
                try:
                    with open(cache_file_path, 'rb') as f:
                        cached_data = pickle.load(f)

                    # Handle both old format (list) and new format (dict with 'successful' and 'skipped')
                    if isinstance(cached_data, list):
                        # Old format - assume all were successful, no skipped info
                        successful_data = cached_data
                        skipped_files = set()
                    elif isinstance(cached_data, dict) and 'successful' in cached_data:
                        # New format with skipped file tracking
                        successful_data = cached_data['successful']
                        skipped_files = set(cached_data.get('skipped', []))
                    else:
                        # Unknown format
                        raise ValueError("Unknown cache format")

                    # Only scan filesystem if cache has data (lightweight check first)
                    if len(successful_data) > 0 or len(skipped_files) > 0:
                        # Defer expensive glob until we know cache exists and has data
                        current_files_in_folder = {os.path.basename(f) for f in glob.glob(os.path.join(str(input_source), str(file_extension_pattern)))} if input_source and file_extension_pattern else set()
                        retained_cached_data = [d for d in successful_data if d['filename'] in current_files_in_folder]

                        # Files that were processed (either successfully or skipped)
                        processed_files = {d['filename'] for d in successful_data} | skipped_files
                        unprocessed_files = current_files_in_folder - processed_files

                        vprint(f"  Cache contains {len(successful_data)} successful entries and {len(skipped_files)} skipped files")
                        vprint(f"  Current folder has {len(current_files_in_folder)} files")
                        vprint(f"  Unprocessed files: {len(unprocessed_files)}")

                        if len(unprocessed_files) == 0:
                            # All files have been processed before
                            all_extracted_data = retained_cached_data
                            vprint(f"Data loaded from cache successfully. ({len(all_extracted_data)} entries)")
                            print_step("Using cached data")
                        else:
                            # Some files haven't been processed yet
                            if len(retained_cached_data) > 0:
                                vprint(f"  Cache partial: {len(unprocessed_files)} files need processing")
                                all_extracted_data = retained_cached_data.copy()
                            else:
                                vprint("Cache data incomplete or outdated. Re-extracting all files.")
                                all_extracted_data = []
                                skipped_files = set()
                                if os.path.exists(cache_file_path):
                                    os.remove(cache_file_path)
                    else:
                        # Cache is empty, invalidate it
                        vprint("Cache is empty. Re-extracting all files.")
                        all_extracted_data = []
                        skipped_files = set()
                        if os.path.exists(cache_file_path):
                            os.remove(cache_file_path)

                except Exception as e:
                    vprint(f"Error loading data from cache: {e}. Re-extracting all files.")
                    all_extracted_data = []
                    skipped_files = set()
                    if os.path.exists(cache_file_path):
                        os.remove(cache_file_path)
        else:
            # Incremental update was done, so we trust the current state
            # all_extracted_data is already set
            pass

        files_to_process = glob.glob(os.path.join(str(input_source), str(file_extension_pattern))) if input_source and file_extension_pattern else []
        if not files_to_process:
            print(f"No files matching '{file_extension_pattern}' found in '{input_source}'. Skipping this folder.")
            return

        # Determine if we need to process any files
        cached_filenames = {d['filename'] for d in all_extracted_data}
        current_filenames = {os.path.basename(f) for f in files_to_process}

        # Files that have been processed (either successfully extracted or skipped)
        processed_files = cached_filenames | skipped_files
        unprocessed_files = current_filenames - processed_files

        if len(unprocessed_files) == 0:
            # All files have been processed before
            files_to_actually_process = []
        elif not processed_files:
            # No cached data, process all files
            print_step(f"\nExtracting data from {len(files_to_process)} files...")
            files_to_actually_process = files_to_process
        else:
            # We have some cached data, only process unprocessed files
            print_step(f"\nProcessing {len(unprocessed_files)} new/unprocessed files...")
            files_to_actually_process = [f for f in files_to_process if os.path.basename(f) in unprocessed_files]

        # Process files if needed
        if files_to_actually_process:
            # Use parallel processing for normal mode
            effective_cores = num_cores  # Use all available cores
            print(f"  Using {effective_cores} CPU cores for parallel processing")

            with mp.Pool(processes=effective_cores) as pool:
                results = pool.map(process_file_parallel_wrapper, sorted(files_to_actually_process))

            # Process results
            for success, extracted_props, filename in results:
                if success and extracted_props:
                    all_extracted_data.append(extracted_props)
                elif not success:
                    # File was skipped (likely due to imaginary frequencies)
                    skipped_files.add(filename)

            # Save updated cache with both successful and skipped files
            if not is_compare_mode:
                cache_data = {
                    'successful': all_extracted_data,
                    'skipped': list(skipped_files)
                }
                vprint(f"Updating cache with {len(all_extracted_data)} successful and {len(skipped_files)} skipped entries: '{os.path.basename(cache_file_path)}'")
                try:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                    vprint("Cache updated successfully.")
                except Exception as e:
                    vprint(f"  Error updating cache: {e}")

    if not all_extracted_data:
        print_step("No data was successfully extracted from files. Skipping clustering.")
        return

    # --- Optional xTB single points on geometry-only records ---------------
    # A plain XYZ determines 7 of the 15 clustering columns; one single point
    # per structure adds the electronic energy, HOMO, gap and dipole, taking
    # the vector to 11. Records that already carry an energy are skipped, so a
    # cached run does not recompute, and a failed call simply leaves that
    # structure geometry-only rather than dropping it.
    if sp_method:
        needs_sp = [
            d for d in all_extracted_data
            if d.get('_from_xyz') and d.get('final_electronic_energy') is None
        ]
        if not needs_sp:
            vprint("--sp: every record already carries an electronic energy; nothing to do.")
        else:
            print_step(f"Running xTB single points on {len(needs_sp)} structure(s)...")
            sp_sources = {
                os.path.basename(p): p
                for p in glob.glob(os.path.join(str(input_source), XYZ_GLOB))
            } if not isinstance(input_source, list) else {
                os.path.basename(p): p for p in input_source
            }
            sp_inputs = [sp_sources[d['filename']] for d in needs_sp if d['filename'] in sp_sources]
            sp_outputs = run_single_points(
                sp_inputs,
                output_base_dir,
                method=sp_method,
                charge=sp_charge,
                uhf=sp_uhf,
                num_cores=num_cores,
            )
            n_merged = merge_single_point_properties(needs_sp, sp_outputs)
            print_step(f"Single-point properties merged for {n_merged}/{len(needs_sp)} structure(s).")

            if n_merged and not is_compare_mode:
                # Fold the new scalars into the cache so a re-run does not pay
                # for the same single points twice.
                try:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(
                            {'successful': all_extracted_data, 'skipped': list(skipped_files)},
                            f,
                        )
                except Exception as e:
                    vprint(f"  Error updating cache with single-point data: {e}")

    print_step("Data extraction complete. Proceeding to clustering.\n")
    print() # Add extra blank line for readability

    clean_data_for_clustering = []
    essential_base_features = ['final_geometry_atomnos', 'final_geometry_coords', 'num_hydrogen_bonds']

    for mol_data in all_extracted_data:
        is_essential_missing = False
        missing_essential_info = []
        for f in essential_base_features:
            if mol_data.get(f) is None:
                is_essential_missing = True
                missing_essential_info.append(f"Missing essential feature '{f}'")

        if not is_essential_missing:
            clean_data_for_clustering.append(mol_data)
        else:
            print(f"Skipping '{mol_data.get('filename', 'Unknown')}' for clustering due to: {'; '.join(missing_essential_info)}")


    if not clean_data_for_clustering:
        print(f"No complete data entries to cluster after filtering. Exiting clustering step.")
        return

    # --- Detect whether the dataset has frequency calculations ---
    # If ANY structure has gibbs_free_energy, we consider the dataset as having freq data.
    # Otherwise, we operate in "opt-only" mode with a reduced feature vector.
    _dataset_has_freq = any(
        is_valid_scalar(_mol.get('gibbs_free_energy'))
        for _mol in clean_data_for_clustering
    )

    # cosmic-v01 sets module globals _DATASET_HAS_FREQ / _DATASET_HAS_COMPOSITE
    # here (lines 5231-5236) so helper functions pick up the mode automatically.
    # v05 carries that state in the explicit EnergyMode `mode` (D-007).
    mode = EnergyMode(has_freq=_dataset_has_freq, has_composite=False)

    # --- Apply composite energies from previous stage (eref workflow) ---
    if prev_out_dir and os.path.isdir(prev_out_dir):
        n_matched = apply_composite_energies(clean_data_for_clustering, prev_out_dir)
        if n_matched > 0:
            _dataset_has_freq = True  # update local too (controls Boltzmann & opt-only marker)
            mode = EnergyMode(has_freq=True, has_composite=True)  # treat composite as freq mode for sorting/output
            print_step(f"Composite energies applied to {n_matched}/{len(clean_data_for_clustering)} structures (prev: {prev_out_dir})")
        else:
            print(f"  Warning: No composite energies could be matched from {prev_out_dir}")

    # --- H-bond grouping decision ---
    # By default, all structures go into a single pool and property-based
    # clustering decides the grouping.  H-bond detection is sensitive to small
    # geometric changes — two nearly identical structures can differ by 1-2
    # H-bonds — so pre-grouping by exact H-bond count can split genuinely
    # similar structures into separate families.
    #
    # The --group-hb flag restores the old behavior of pre-grouping by exact
    # H-bond count, which produces separate dendrograms per HB family.  This
    # can be useful for visualization when the user wants to inspect each
    # H-bond family independently.
    if is_compare_mode and len(clean_data_for_clustering) >= 2:
        hbond_groups = {0: sorted(clean_data_for_clustering,
                                  key=lambda x: (sorting_energy(x, mode), x['filename']))}
        print("  Comparison mode: Running clustering to generate dendrogram, then forcing a single output cluster.")
    elif group_hb:
        # Group by exact H-bond count (separate dendrograms per HB family)
        hbond_groups = {}
        for item in clean_data_for_clustering:
            hbond_groups.setdefault(item['num_hydrogen_bonds'], []).append(item)
        print(f"  H-bond pre-grouping enabled: {len(hbond_groups)} HB families detected")
    else:
        # Default: single pool — let property-based clustering decide
        hbond_groups = {0: sorted(clean_data_for_clustering,
                                  key=lambda x: (sorting_energy(x, mode), x['filename']))}

    # Output directory paths
    dendrogram_images_folder = os.path.join(output_base_dir, "dendrogram_images")
    extracted_data_folder = os.path.join(output_base_dir, "extracted_data")
    extracted_clusters_folder = os.path.join(output_base_dir, "extracted_clusters")



    os.makedirs(dendrogram_images_folder, exist_ok=True)
    os.makedirs(extracted_data_folder, exist_ok=True)
    os.makedirs(extracted_clusters_folder, exist_ok=True)

    if console.VERBOSE:
        print(f"Dendrogram images will be saved to '{dendrogram_images_folder}'")
        print(f"Extracted data files will be saved to '{extracted_data_folder}'")
        print(f"Extracted cluster XYZ/MOL files will be saved to '{extracted_clusters_folder}'")
    else:
        print_step("Setting up output directories...")

    summary_file_content_lines = []
    comparison_specific_summary_lines = [] # New list for comparison-specific details
    resolved_threshold_entries = []

    total_clusters_outputted = 0
    total_rmsd_outliers_first_pass = 0

    # Helper function to center text within 75 characters
    def center_text(text, width=75):
        return text.center(width)

    # Add ASCII art header similar to ASCEC but for COSMIC
    summary_file_content_lines.append("=" * 75)
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append(center_text("*       C O S M I C       *"))
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("                             √≈≠==≈                                  ")
    summary_file_content_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
    summary_file_content_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
    summary_file_content_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
    summary_file_content_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
    summary_file_content_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
    summary_file_content_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
    summary_file_content_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
    summary_file_content_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
    summary_file_content_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
    summary_file_content_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Universidad de Antioquia - Medellín - Colombia"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Clustering Analysis for Quantum Chemistry Calculations"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text(version))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Química Física Teórica - QFT"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("=" * 75 + "\n")
    if is_compare_mode:
        summary_file_content_lines.append(f"Comparison Results for: {', '.join([os.path.basename(f) for f in input_source])}")
    else:
        summary_file_content_lines.append(f"Clustering Results for: {os.path.basename(input_source)}")

    summary_file_content_lines.append("<COSMIC_THRESHOLD_PLACEHOLDER>")

    if rmsd_only:
        summary_file_content_lines.append("Clustering mode: RMSD only (geometry, no feature vector)")
        summary_file_content_lines.append(f"RMSD clustering threshold: {rmsd_threshold:.3f} Å")
        summary_file_content_lines.append(
            f"RMSD atom set: {'heavy atoms only (--rmsd-heavy)' if rmsd_heavy else 'all atoms, hydrogens included'}")
    elif rmsd_threshold is not None:
        summary_file_content_lines.append(f"RMSD validation threshold: {rmsd_threshold:.3f} Å")
        summary_file_content_lines.append(
            f"RMSD atom set: {'heavy atoms only (--rmsd-heavy)' if rmsd_heavy else 'all atoms, hydrogens included'}")

    total_files_attempted = len(clean_data_for_clustering) + len(skipped_files)
    if total_files_attempted > 0:
        skipped_percentage = (len(skipped_files) / total_files_attempted) * 100
        skipped_info = f"{len(skipped_files)} ({skipped_percentage:.1f}%)"
    else:
        skipped_info = f"{len(skipped_files)}"

    summary_file_content_lines.append(f"Total configurations processed: {len(clean_data_for_clustering)}")
    summary_file_content_lines.append(f"Total files skipped: <TOTAL_SKIPPED_PLACEHOLDER>")
    summary_file_content_lines.append(f"Critical skipped files: <IMAG_NEED_RECALC_PLACEHOLDER>")
    summary_file_content_lines.append(f"Total number of final clusters: <TOTAL_CLUSTERS_PLACEHOLDER>")
    if rmsd_threshold is not None and not rmsd_only:
        summary_file_content_lines.append(f"Total RMSD moved configurations: <TOTAL_RMSD_OUTLIERS_PLACEHOLDER>")
    summary_file_content_lines.append("")
    # Report the active weight profile and any non-default weights
    if rmsd_only:
        summary_file_content_lines.append("Weight profile: not applicable (RMSD-only clustering)")
    elif partialweights:
        summary_file_content_lines.append("Weight profile: semiempirical (--partialweights)")
    else:
        summary_file_content_lines.append("Weight profile: uniform (1.0)")
    if weights and not rmsd_only:
        non_default = {k: v for k, v in weights.items() if v != 1.0}
        if non_default:
            summary_file_content_lines.append(f"Feature weights (non-default): {non_default}")
    summary_file_content_lines.append("\n" + "=" * 75 + "\n")


    previous_hbond_group_processed = False
    total_imag_clustered_with_normal = 0
    total_imag_need_recalc = 0
    total_non_converged_critical = 0
    all_skipped_clustered_with_normal = []
    all_skipped_need_recalc = []
    all_non_converged_critical = []

    # --- Dynamic feature vector: all 15 features are always candidates ---
    # Per-structure availability determines the actual vector used.
    _all_scalar_features = list(CLUSTERING_NUMERICAL_FEATURES)
    _scalar_features = list(_all_scalar_features)

    # H-bond geometry features (distance/angle/std) are EXCLUDED from the criticality /
    # full-feature gate. Their absence means "no eligible H-bond" — real structural
    # data, not a failed calculation. Letting them gate criticality discards every
    # non-H-bonded structure as reduced/critical: e.g. a single-fragment conformational
    # search where only the intramolecularly H-bonded conformers carry these features
    # loses every other conformer (95.9% skipped on glycolaldehyde). The gate is
    # therefore computed from non-H-bond-geometry features only. The H-bond features
    # remain in `_scalar_features`, so they are still used as clustering features when
    # present (e.g. water-hexamer isomer discrimination via H-bond geometry).
    # num_hydrogen_bonds is always a valid scalar (0+) so it never gates anyway.
    _hbond_geom_features = set(HBOND_GEOMETRY_FEATURES)
    _gate_features = [f for f in _scalar_features if f not in _hbond_geom_features]

    # Compute available features per structure (gate uses non-H-bond-geometry features)
    _vector_size_hist = {}
    for _mol in clean_data_for_clustering:
        _available = set()
        for _fname in _gate_features:
            _key = FEATURE_MAPPING.get(_fname, _fname)
            if is_valid_scalar(_mol.get(_key)):
                _available.add(_fname)
        if has_valid_rotational_constants(_mol):
            _available.update(['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'])
        _mol['_available_features'] = _available
        _mol['_feature_vector_size'] = len(_available)
        _vector_size_hist[_mol['_feature_vector_size']] = _vector_size_hist.get(_mol['_feature_vector_size'], 0) + 1

    # The "full" vector is the maximum feature count found in the pool
    _pool_max_features = max(_mol['_feature_vector_size'] for _mol in clean_data_for_clustering)
    for _mol in clean_data_for_clustering:
        _mol['_is_full_feature'] = (_mol['_feature_vector_size'] == _pool_max_features)

    _full_count = sum(1 for _mol in clean_data_for_clustering if _mol['_is_full_feature'])
    _non_full_count = len(clean_data_for_clustering) - _full_count
    print(f"Full-feature structures: {_full_count} ({_pool_max_features} features)")
    if _non_full_count > 0:
        print(f"  Reduced-vector structures: {_non_full_count}")
        for _vec_size in sorted([k for k in _vector_size_hist.keys() if k < _pool_max_features], reverse=True):
            _count = _vector_size_hist[_vec_size]
            print(f"    - {_count} with {_vec_size} features")

    # --- Boltzmann Population Calculation (based on initial property clusters) ---
    all_initial_property_clusters = []
    pseudo_global_cluster_id_counter = 1 # This counter is for assigning unique IDs to initial clusters for Boltzmann calc

    # Track reduced structures that could not match any fullest-tier cluster
    all_reduced_unmatched = []
    # Track reduced structures that DID match a full-feature cluster. These have no
    # own energy (e.g. non-converged opt with no freq/thermo), so they never become
    # representatives; they are redundant with a true minimum already in their cluster.
    # Reported separately for visibility — they are neither skipped nor critical.
    all_reduced_matched = []

    # --rmsd-only: the geometry-only partition is built once here and reused by
    # the output loop below (the pairwise Kabsch matrix is O(n²) — no reason to
    # pay for it twice), keyed by H-bond group.
    rmsd_only_partitions = {}

    for hbond_count, group_data in sorted(hbond_groups.items()):
        if rmsd_only:
            print_step(f"RMSD-only clustering of {len(group_data)} configurations "
                       f"at {rmsd_threshold:.3f} Å...")
            _rmsd_clusters, _rmsd_linkage, _rmsd_linkage_members = cluster_by_rmsd(
                group_data, rmsd_threshold, mode, heavy_only=rmsd_heavy)
            rmsd_only_partitions[hbond_count] = (_rmsd_clusters, _rmsd_linkage,
                                                 _rmsd_linkage_members)
            _rmsd_off_tree = len(group_data) - len(_rmsd_linkage_members)
            if _rmsd_off_tree > 0:
                print(f"  {_rmsd_off_tree} structure(s) not RMSD-comparable with the main "
                      f"block (different heavy-atom count or no geometry); clustered "
                      f"separately and left off the dendrogram.")
            for _rmsd_cluster in _rmsd_clusters:
                for member_conf in _rmsd_cluster:
                    member_conf['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                all_initial_property_clusters.append(_rmsd_cluster)
                pseudo_global_cluster_id_counter += 1
            continue

        if len(group_data) < 2 or not group_has_any_clustering_feature_data(group_data):
            for single_mol_data in group_data:
                single_mol_data['_initial_cluster_label'] = hbond_count
                single_mol_data['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                all_initial_property_clusters.append([single_mol_data])
                pseudo_global_cluster_id_counter += 1
        else:
            # --- Tier-based dynamic vector clustering ---
            # Separate fullest-tier from reduced-tier structures
            fullest_tier = [m for m in group_data if m['_is_full_feature']]
            reduced_tier = [m for m in group_data if not m['_is_full_feature']]

            # If no fullest tier structures, promote the local maximum
            if not fullest_tier:
                _local_max = max(m['_feature_vector_size'] for m in group_data)
                fullest_tier = [m for m in group_data if m['_feature_vector_size'] == _local_max]
                reduced_tier = [m for m in group_data if m['_feature_vector_size'] < _local_max]

            # If fullest tier too small to cluster, treat all as singletons
            if len(fullest_tier) < 2:
                for mol in fullest_tier + reduced_tier:
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    pseudo_global_cluster_id_counter += 1
                continue

            # Feature selection on fullest tier only
            active_numerical_features_for_group, dropped_scalar_features = select_complete_group_scalar_features(fullest_tier, list(_scalar_features))
            use_rotational_constants = all(has_valid_rotational_constants(m) for m in fullest_tier)

            _active_display = list(active_numerical_features_for_group)
            if use_rotational_constants:
                _active_display += ['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C']
            print(f"  Active features ({len(_active_display)}): {_active_display}")

            # Build vectors for fullest tier
            features_for_scaling_raw, ordered_feature_names_for_scaling = build_feature_vectors(
                fullest_tier, active_numerical_features_for_group, use_rotational_constants, weights
            )

            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                for mol in fullest_tier + reduced_tier:
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    pseudo_global_cluster_id_counter += 1
                continue

            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            features_scaled, active_feat_names, dropped_constant = zscore_scale(
                features_for_scaling_raw_np, ordered_feature_names_for_scaling,
                min_std_threshold, abs_tolerances)
            if dropped_constant:
                vprint(f"  Dropped constant/within-tolerance features: {', '.join(dropped_constant)}")
            features_scaled = apply_weights(features_scaled, active_feat_names, weights)

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')
            effective_t, _k_eff, t_source = resolve_clustering_threshold(
                linkage_matrix, threshold,
                scaled_matrix=features_scaled, verbose=console.VERBOSE)
            initial_cluster_labels = fcluster(linkage_matrix, t=effective_t, criterion='distance')
            vprint(f"Clustering threshold: τ={effective_t:.4f} ({t_source}), "
                   f"n_c={len(set(initial_cluster_labels))}")

            # Build cluster data from fullest tier
            initial_clusters_data = {}
            for i, label in enumerate(initial_cluster_labels):
                fullest_tier[i]['_initial_cluster_label'] = label
                initial_clusters_data.setdefault(label, []).append(fullest_tier[i])

            # Match reduced-tier structures against fullest-tier clusters
            if reduced_tier:
                matched, unmatched = match_reduced_to_clusters(
                    reduced_tier, fullest_tier, initial_cluster_labels,
                    active_numerical_features_for_group, use_rotational_constants,
                    weights, effective_t, min_std_threshold, abs_tolerances, mode
                )
                for label, mols in matched.items():
                    for mol in mols:
                        mol['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).extend(mols)
                # Unmatched reduced structures → singletons flagged as critical
                for mol in unmatched:
                    mol['_reduced_unmatched'] = True
                    mol['_initial_cluster_label'] = hbond_count
                    mol['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([mol])
                    all_reduced_unmatched.append(mol)
                    pseudo_global_cluster_id_counter += 1

            initial_clusters_list_unsorted = list(initial_clusters_data.values())
            initial_clusters_list_sorted_by_energy = sorted(
                initial_clusters_list_unsorted,
                key=lambda cluster: (min(sorting_energy(m, mode) for m in cluster),
                                     min(m['filename'] for m in cluster))
            )

            for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                parent_id = pseudo_global_cluster_id_counter
                for member_conf in initial_prop_cluster:
                    member_conf['_parent_global_cluster_id'] = parent_id
                all_initial_property_clusters.append(initial_prop_cluster)
                pseudo_global_cluster_id_counter += 1

    boltzmann_g1_data = {}
    global_min_gibbs_energy = None
    global_min_rep_filename = "N/A"
    global_min_cluster_id = "N/A"

    if _dataset_has_freq and all_initial_property_clusters:
        # Find the global minimum Gibbs energy among all representatives
        # Also store the filename and cluster ID of this global minimum representative
        valid_reps_for_emin = []
        for initial_prop_cluster in all_initial_property_clusters:
            # Select the lowest energy member as the representative for this initial property cluster
            rep_conf = min(initial_prop_cluster,
                           key=lambda x: (sorting_energy(x, mode), x['filename']))

            if rep_conf.get('gibbs_free_energy') is not None:
                valid_reps_for_emin.append({
                    'energy': rep_conf['gibbs_free_energy'],
                    'filename': rep_conf['filename'],
                    'cluster_id': rep_conf['_parent_global_cluster_id']
                })

        if valid_reps_for_emin:
            global_min_info = min(valid_reps_for_emin, key=lambda x: x['energy'])
            global_min_gibbs_energy = global_min_info['energy']
            global_min_rep_filename = global_min_info['filename']
            global_min_cluster_id = global_min_info['cluster_id']

            sum_factors_g1 = 0.0

            for initial_prop_cluster in all_initial_property_clusters:
                rep_conf = min(initial_prop_cluster,
                               key=lambda x: (sorting_energy(x, mode), x['filename']))

                if rep_conf.get('gibbs_free_energy') is None:
                    continue

                rep_gibbs_energy = rep_conf['gibbs_free_energy']
                cluster_id = rep_conf['_parent_global_cluster_id']
                cluster_size = len(initial_prop_cluster)

                delta_e = rep_gibbs_energy - global_min_gibbs_energy

                if BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k == 0:
                    factor_g1 = 1.0 if delta_e == 0 else 0.0
                else:
                    factor_g1 = np.exp(-delta_e / (BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k))

                boltzmann_g1_data[cluster_id] = {
                    'energy': rep_gibbs_energy,
                    'filename': rep_conf['filename'],
                    'population': factor_g1,
                    'cluster_size': cluster_size
                }

                sum_factors_g1 += factor_g1

            if sum_factors_g1 > 0:
                for cluster_id, data in boltzmann_g1_data.items():
                    data['population'] = (data['population'] / sum_factors_g1) * 100.0
            else:
                for cluster_id in boltzmann_g1_data:
                    boltzmann_g1_data[cluster_id]['population'] = 0.0
    # --- End Boltzmann Population Calculation ---


    # Collect all final clusters for unique motifs creation
    all_final_clusters = []
    cluster_id_mapping = {}  # Maps cluster index to cluster ID for motifs

    # Now iterate through the hbond_groups again to perform the clustering and write files
    # This loop is responsible for generating the actual clusters and writing their files.
    # The Boltzmann data calculated above will be passed to write_cluster_dat_file.
    cluster_global_id_counter = 1
    for hbond_count, group_data in sorted(hbond_groups.items()):

        hbond_group_summary_lines = []

        if previous_hbond_group_processed:
            hbond_group_summary_lines.append("\n" + "-" * 75 + "\n")

        if group_hb and not is_compare_mode:
            hbond_group_summary_lines.append(f"Hydrogen bonds: {hbond_count}\n")
        else:
            hbond_group_summary_lines.append(f"All configurations\n")
        hbond_group_summary_lines.append(f"Configurations: {len(group_data)}")

        current_hbond_group_clusters_for_final_output = []

        if rmsd_only:
            # Geometry-only mode: reuse the partition already built above. No
            # feature vector, no auto-threshold, no first/second RMSD pass —
            # the RMSD cut IS the clustering.
            _rmsd_clusters, _rmsd_linkage, _rmsd_linkage_members = rmsd_only_partitions[hbond_count]
            current_hbond_group_clusters_for_final_output.extend(_rmsd_clusters)

            if _rmsd_linkage is not None:
                import re
                _rmsd_basenames = [os.path.splitext(m['filename'])[0] for m in _rmsd_linkage_members]
                _rmsd_conf_labels = []
                for _fn in _rmsd_basenames:
                    _m = re.search(r'(\d+)', _fn)
                    _rmsd_conf_labels.append(_m.group(1) if _m else _fn)
                # conf_### names shorten to their index; anything else (t2_10,
                # t2_11, ...) collapses to the same digits, so keep full names.
                if len(set(_rmsd_conf_labels)) < len(_rmsd_conf_labels):
                    _rmsd_conf_labels = _rmsd_basenames

                _hb_tag = f" (H-bonds={hbond_count})" if group_hb else ""
                if group_hb:
                    _rmsd_dendro_file = os.path.join(dendrogram_images_folder,
                                                     f"dendrogram_H{hbond_count}.png")
                else:
                    _rmsd_dendro_file = os.path.join(dendrogram_images_folder, "dendrogram.png")

                plot_annotated_dendrogram(
                    _rmsd_linkage, len(_rmsd_clusters), rmsd_threshold,
                    _rmsd_dendro_file,
                    title_suffix=f"RMSD only{_hb_tag}",
                    conf_labels=_rmsd_conf_labels,
                    ylabel=("Heavy-atom RMSD (Å)" if rmsd_heavy else "RMSD (Å, all atoms)"),
                    standard_threshold=None)
                vprint(f"Dendrogram saved as '{os.path.basename(_rmsd_dendro_file)}'")

        elif len(group_data) < 2 or not group_has_any_clustering_feature_data(group_data):
            vprint(f"\nSkipping detailed clustering: Less than 2 configurations or no valid numerical features left after filtering. Treating each as a single-configuration cluster.")

            for single_mol_data in group_data:
                single_mol_data['_rmsd_pass_origin'] = 'first_pass_validated'
                current_hbond_group_clusters_for_final_output.append([single_mol_data])

        else: # Proceed with actual clustering
            # --- Tier-based dynamic vector clustering ---
            # In compare mode, use all structures together (common features);
            # otherwise, cluster fullest tier first, then match reduced structures.
            if is_compare_mode:
                _clustering_pool = group_data
                _reduced_pool = []
            else:
                _clustering_pool = [m for m in group_data if m['_is_full_feature']]
                _reduced_pool = [m for m in group_data if not m['_is_full_feature']]
                if not _clustering_pool:
                    _local_max = max(m['_feature_vector_size'] for m in group_data)
                    _clustering_pool = [m for m in group_data if m['_feature_vector_size'] == _local_max]
                    _reduced_pool = [m for m in group_data if m['_feature_vector_size'] < _local_max]

            # If clustering pool is too small, treat everything as singletons
            if len(_clustering_pool) < 2:
                for mol in group_data:
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])
                    if not mol['_is_full_feature']:
                        mol['_reduced_unmatched'] = True
                        all_reduced_unmatched.append(mol)
                continue

            filenames_base = [os.path.splitext(item['filename'])[0] for item in _clustering_pool]

            # Feature selection on clustering pool (fullest tier or all in compare mode)
            active_numerical_features_for_group, dropped_scalar_features = select_complete_group_scalar_features(_clustering_pool, list(_scalar_features))
            use_rotational_constants = all(has_valid_rotational_constants(m) for m in _clustering_pool)

            # Build feature vectors for the clustering pool
            features_for_scaling_raw, ordered_feature_names_for_scaling = build_feature_vectors(
                _clustering_pool, active_numerical_features_for_group, use_rotational_constants, weights
            )

            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                vprint(f"  WARNING: No numerical features left for clustering after applying weights. Treating each as a single-configuration cluster.")
                print_step(f"{len(group_data)} config(s) - no features for clustering")
                for mol in group_data:
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])
                continue

            # Announce clustering
            _hb_tag = f" (H-bonds={hbond_count})" if group_hb else ""
            _tier_info = f" (fullest tier: {len(_clustering_pool)}/{len(group_data)})" if _reduced_pool else ""
            print_step(f"Clustering {len(_clustering_pool)} configurations{_hb_tag}{_tier_info}...")

            _active_display = list(active_numerical_features_for_group)
            if use_rotational_constants:
                _active_display += ['rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C']
            print(f"  Active features ({len(_active_display)}): {_active_display}")
            if dropped_scalar_features:
                vprint(f"  Dropped (missing in some structures): {', '.join(dropped_scalar_features)}")
            if not use_rotational_constants:
                vprint(f"  Rotational constants excluded (not available for all structures)")

            # Z-score scaling
            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            features_scaled, active_feat_names, dropped_constant = zscore_scale(
                features_for_scaling_raw_np, ordered_feature_names_for_scaling,
                min_std_threshold, abs_tolerances)
            if dropped_constant:
                vprint(f"  Dropped constant/within-tolerance features: {', '.join(dropped_constant)}")
            features_scaled = apply_weights(features_scaled, active_feat_names, weights)

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')

            # --- Resolve threshold (auto-knee / legacy / user / opt-*) and cluster ---
            effective_t, _k_eff, t_source = resolve_clustering_threshold(
                linkage_matrix, threshold,
                scaled_matrix=features_scaled, verbose=console.VERBOSE)
            initial_cluster_labels = fcluster(linkage_matrix, t=effective_t, criterion='distance')
            _main_optimal_k = len(set(initial_cluster_labels))
            _main_cut_height = effective_t
            vprint(f"Clustering threshold: τ={effective_t:.4f} ({t_source}), n_c={_main_optimal_k}")

            if not is_compare_mode:
                attach_pearson_to_rep(_clustering_pool, features_scaled,
                                      initial_cluster_labels, effective_t, mode)
                resolved_threshold_entries.append(threshold_entry(
                    effective_t, features_scaled, t_source,
                    group_label=(hbond_count if group_hb else None)))

            _mojena_t, _mojena_k = compute_mojena_threshold(linkage_matrix, verbose=console.VERBOSE)

            # Always surface the detected knee on the diagnostic plot — drawn as
            # its own line/legend entry like the applied cut — so the elbow is
            # visible whether or not it was capped to the empirical τ=2.0 ceiling.
            _knee_t = _knee_k = None
            _kt, _kk, _kok, _kr, _ki, _kh = compute_knee_threshold(linkage_matrix)
            if _kok:
                _knee_t, _knee_k = _kt, _kk

            # --- Extract configuration labels for dendrogram ---
            import re
            conf_labels = []
            for filename in filenames_base:
                match = re.search(r'(\d+)', filename)
                if match:
                    conf_labels.append(match.group(1))
                else:
                    conf_labels.append(filename)

            if is_compare_mode:
                dendrogram_title_suffix = "Comparison"
            elif group_hb:
                dendrogram_title_suffix = f"H-bonds = {hbond_count}"
            else:
                dendrogram_title_suffix = "All configurations"

            if group_hb and not is_compare_mode:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram_H{hbond_count}.png")
            else:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram.png")

            try:
                _diag_n_eff = effective_n_features(features_scaled)
            except Exception:
                _diag_n_eff = None

            plot_annotated_dendrogram(
                linkage_matrix, _main_optimal_k, _main_cut_height,
                dendrogram_filename, title_suffix=dendrogram_title_suffix,
                conf_labels=conf_labels,
                mojena_threshold=_mojena_t, mojena_k=_mojena_k,
                n_eff=_diag_n_eff,
                knee_threshold=_knee_t, knee_k=_knee_k)
            vprint(f"Dendrogram saved as '{os.path.basename(dendrogram_filename)}'")

            # --- Match reduced-tier structures against fullest-tier clusters ---
            _matched_reduced = {}
            _unmatched_reduced = []
            if _reduced_pool:
                _matched_reduced, _unmatched_reduced = match_reduced_to_clusters(
                    _reduced_pool, _clustering_pool, initial_cluster_labels,
                    active_numerical_features_for_group, use_rotational_constants,
                    weights, effective_t, min_std_threshold, abs_tolerances, mode
                )
                if _matched_reduced:
                    _total_matched = sum(len(v) for v in _matched_reduced.values())
                    print(f"  Reduced-vector matching: {_total_matched} matched, {len(_unmatched_reduced)} unmatched (critical)")
                    for _matched_mols in _matched_reduced.values():
                        for _m in _matched_mols:
                            _m['_reduced_matched'] = True
                        all_reduced_matched.extend(_matched_mols)
                if _unmatched_reduced:
                    all_reduced_unmatched.extend(_unmatched_reduced)

            if is_compare_mode and len(group_data) >= 2:
                print("  Comparison mode: Overriding clustering to output a single combined cluster.")
                for i, member in enumerate(group_data):
                    member['_initial_cluster_label'] = 1
                    member['_parent_global_cluster_id'] = 1
                    member['_rmsd_pass_origin'] = 'first_pass_validated'
                    member['_second_rmsd_sub_cluster_id'] = None
                    member['_second_rmsd_context_listing'] = None
                    member['_second_rmsd_rep_filename'] = None

                prop_rep_conf = group_data[0]
                first_rmsd_listing = []
                for member_conf in group_data:
                    if member_conf == prop_rep_conf:
                        rmsd_val = 0.0
                    elif prop_rep_conf.get('final_geometry_coords') is not None and prop_rep_conf.get('final_geometry_atomnos') is not None and \
                         member_conf.get('final_geometry_coords') is not None and \
                         member_conf.get('final_geometry_atomnos') is not None:
                        rmsd_val = calculate_rmsd(
                            prop_rep_conf['final_geometry_atomnos'], prop_rep_conf['final_geometry_coords'],
                            member_conf['final_geometry_atomnos'], member_conf['final_geometry_coords'],
                            heavy_only=rmsd_heavy
                        )
                    else:
                        rmsd_val = None
                    first_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})
                for member_conf in group_data:
                    member_conf['_first_rmsd_context_listing'] = first_rmsd_listing

                current_hbond_group_clusters_for_final_output.append(group_data)
            else:
                # Normal clustering: build cluster data from fullest tier + absorbed reduced
                initial_clusters_data = {}
                for i, label in enumerate(initial_cluster_labels):
                    _clustering_pool[i]['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).append(_clustering_pool[i])

                # Absorb matched reduced structures into their clusters
                for label, mols in _matched_reduced.items():
                    for mol in mols:
                        mol['_initial_cluster_label'] = label
                    initial_clusters_data.setdefault(label, []).extend(mols)

                # Unmatched reduced → singleton clusters flagged as critical
                for mol in _unmatched_reduced:
                    mol['_reduced_unmatched'] = True
                    mol['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.append([mol])

                initial_clusters_list_unsorted = list(initial_clusters_data.values())
                initial_clusters_list_sorted_by_energy = sorted(
                    initial_clusters_list_unsorted,
                    key=lambda cluster: (min(sorting_energy(m, mode) for m in cluster),
                                         min(m['filename'] for m in cluster))
                )

                for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                    if initial_prop_cluster and initial_prop_cluster[0].get('_parent_global_cluster_id') is None:
                        parent_id = pseudo_global_cluster_id_counter
                        for member_conf in initial_prop_cluster:
                            member_conf['_parent_global_cluster_id'] = parent_id
                        pseudo_global_cluster_id_counter += 1

                if rmsd_threshold is not None:
                    print(f"  Performing first RMSD validation...")

                    validated_main_clusters, individual_outliers_from_first_pass = \
                        post_process_clusters_with_rmsd(initial_clusters_list_sorted_by_energy, rmsd_threshold, mode,
                                                        heavy_only=rmsd_heavy)

                    current_hbond_group_clusters_for_final_output.extend(validated_main_clusters)
                    total_rmsd_outliers_first_pass += len(individual_outliers_from_first_pass)

                    if individual_outliers_from_first_pass:
                        print(f"    Attempting second RMSD clustering on {len(individual_outliers_from_first_pass)} outliers from first pass...")

                        outliers_grouped_by_parent_global_cluster = {}
                        for outlier_conf in individual_outliers_from_first_pass:
                            parent_global_id = outlier_conf.get('_parent_global_cluster_id')
                            if parent_global_id is not None:
                                outliers_grouped_by_parent_global_cluster.setdefault(parent_global_id, []).append(outlier_conf)

                        for parent_global_id_for_outlier_group, outlier_group in outliers_grouped_by_parent_global_cluster.items():
                            if len(outlier_group) > 1:
                                print(f"      Re-clustering {len(outlier_group)} outliers from original Cluster {parent_global_id_for_outlier_group}...")
                                second_level_clusters = perform_second_rmsd_clustering(outlier_group, rmsd_threshold, mode,
                                                                                       heavy_only=rmsd_heavy)
                                current_hbond_group_clusters_for_final_output.extend(second_level_clusters)
                            else:
                                single_member_processed = perform_second_rmsd_clustering(outlier_group, rmsd_threshold, mode,
                                                                                          heavy_only=rmsd_heavy)
                                current_hbond_group_clusters_for_final_output.extend(single_member_processed)
                else:
                    for cluster in initial_clusters_list_sorted_by_energy:
                        for member in cluster:
                            member['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.extend(initial_clusters_list_sorted_by_energy)

        current_hbond_group_clusters_for_final_output.sort(key=lambda cluster: (min(sorting_energy(m, mode) for m in cluster),
                                                                                  min(m['filename'] for m in cluster)))

        # Filter out structures with imaginary frequencies AFTER clustering
        current_hbond_group_clusters_for_final_output, hbond_skipped_info = filter_imaginary_freq_structures(
            current_hbond_group_clusters_for_final_output,
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering)
        )

        # Remove non-converged structures from clusters after imaginary filtering.
        # These are always critical and must never become representatives.
        current_hbond_group_clusters_for_final_output, hbond_non_converged = filter_non_converged_structures(
            current_hbond_group_clusters_for_final_output, dataset_has_freq=_dataset_has_freq
        )

        # Remove reduced-vector structures that matched a full-feature cluster.
        # They carry no own energy (e.g. a non-converged opt: geometry but no
        # freq/thermo) and are redundant with the converged representative already
        # in their cluster, so — exactly like an imaginary-frequency structure that
        # clustered with a true minimum — they are dropped here and counted as
        # skipped (redundant, not critical). Comparison mode keeps every requested
        # structure, so it is exempt.
        if not is_compare_mode:
            current_hbond_group_clusters_for_final_output, _ = filter_redundant_reduced_structures(
                current_hbond_group_clusters_for_final_output
            )

        # Track and accumulate skipped structures
        total_imag_clustered_with_normal += len(hbond_skipped_info.get('clustered_with_normal', []))
        total_imag_need_recalc += len(hbond_skipped_info.get('need_recalculation', []))
        all_skipped_clustered_with_normal.extend(hbond_skipped_info.get('clustered_with_normal', []))
        all_skipped_need_recalc.extend(hbond_skipped_info.get('need_recalculation', []))
        total_non_converged_critical += len(hbond_non_converged)
        all_non_converged_critical.extend(hbond_non_converged)

        all_final_clusters.extend(current_hbond_group_clusters_for_final_output)

        hbond_group_summary_lines.append(f"Number of clusters: {len(current_hbond_group_clusters_for_final_output)}\n\n")

        # Print info only if there are valid clusters after filtering
        if len(current_hbond_group_clusters_for_final_output) > 0:
            # Check if this was a single-config group (before any potential RMSD processing)
            original_group_size = len(group_data)
            if original_group_size < 2:
                print()
                print()
                print_step(f"{original_group_size} config(s) - treating as single-config clusters")
                print()  # Blank line after single-config group

        # Add blank line before multi-config group
        if len(current_hbond_group_clusters_for_final_output) > 0 and len(group_data) >= 2:
            print()

        for members_data in current_hbond_group_clusters_for_final_output:
            current_global_cluster_id = cluster_global_id_counter

            summary_line_prefix = f"Cluster {current_global_cluster_id} ({len(members_data)} configurations)"

            if rmsd_threshold is not None and members_data[0].get('_rmsd_pass_origin') == 'second_pass_formed':
                parent_global_cluster_id_for_tag = members_data[0].get('_parent_global_cluster_id')

                if len(members_data) == 1:
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"
                else:
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"

            hbond_group_summary_lines.append(summary_line_prefix + ":")
            hbond_group_summary_lines.append("Files:")
            for m_data in members_data:
                if mode.has_freq:
                    if m_data['gibbs_free_energy'] is not None:
                        gibbs_str = f"{m_data['gibbs_free_energy']:.6f} Hartree ({hartree_to_kcal_mol(m_data['gibbs_free_energy']):.2f} kcal/mol, {hartree_to_ev(m_data['gibbs_free_energy']):.2f} eV)"
                    else:
                        gibbs_str = "N/A"
                    hbond_group_summary_lines.append(f"  - {m_data['filename']} (Gibbs Energy: {gibbs_str})")
                else:
                    elec = m_data.get('final_electronic_energy')
                    elec_str = f"{elec:.6f} Hartree" if elec is not None else "N/A"
                    hbond_group_summary_lines.append(f"  - {m_data['filename']} (Electronic Energy: {elec_str})")
            hbond_group_summary_lines.append("\n")

            # Add newline after cluster info
            if members_data == current_hbond_group_clusters_for_final_output[-1]:
                print()

            # Print cluster info - verbose shows all files, non-verbose shows summary
            if console.VERBOSE:
                print(f"\n{summary_line_prefix}:")
                for m_data in members_data:
                    print(f"  - {m_data['filename']}")
            else:
                print_step(f"{summary_line_prefix}")

            cluster_name_prefix = ""
            num_configurations_in_cluster = len(members_data)

            if num_configurations_in_cluster == 1:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}"
            else:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}_{num_configurations_in_cluster}"

            write_cluster_dat_file(cluster_name_prefix, members_data, output_base_dir, rmsd_threshold,
                                   hbond_count_for_original_cluster=hbond_count if group_hb else None, weights=weights, tolerances=abs_tolerances,
                                   rmsd_only=rmsd_only, rmsd_heavy=rmsd_heavy)
            vprint(f"Wrote combined data for Cluster '{cluster_name_prefix}' to '{cluster_name_prefix}.dat'")

            cluster_xyz_subfolder = os.path.join(extracted_clusters_folder, cluster_name_prefix)
            os.makedirs(cluster_xyz_subfolder, exist_ok=True)
            vprint(f"  Saving .xyz files to '{cluster_xyz_subfolder}'")

            # Store cluster ID in each member for later motif mapping
            for m_data in members_data:
                m_data['_cluster_global_id'] = current_global_cluster_id
                xyz_filename = os.path.join(cluster_xyz_subfolder, os.path.splitext(m_data['filename'])[0] + ".xyz")
                write_xyz_file(m_data, xyz_filename, mode)

            combine_xyz_files(members_data, cluster_xyz_subfolder, mode, output_base_name=cluster_name_prefix)

            total_clusters_outputted += 1
            cluster_global_id_counter += 1

        if len(current_hbond_group_clusters_for_final_output) > 0:
            summary_file_content_lines.extend(hbond_group_summary_lines)
            previous_hbond_group_processed = True

    # Write combined skipped structures summary after clustering
    if all_skipped_clustered_with_normal or all_skipped_need_recalc:
        combined_skipped_info = {
            'clustered_with_normal': all_skipped_clustered_with_normal,
            'need_recalculation': all_skipped_need_recalc
        }
        filter_imaginary_freq_structures(
            [],  # Empty cluster list since we're using precomputed data
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering),
            write_summary=True,
            precomputed_skipped=combined_skipped_info
        )

    # Persist non-converged critical structures for redo mode.
    if all_non_converged_critical:
        save_non_converged_critical_structures(
            all_non_converged_critical,
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering)
        )

    # Reduced-vector structures that matched a full-feature cluster have no own
    # energy and are redundant with the converged representative already in their
    # cluster. They are counted as skipped (not critical), mirroring imaginary-
    # frequency structures that clustered with a true minimum. The unmatched
    # reduced structures are already inside total_non_converged_critical.
    # Comparison mode keeps every requested structure, so it neither strips nor
    # counts them as skipped.
    redundant_reduced = [] if is_compare_mode else list(all_reduced_matched)

    # Persist redundant reduced-vector structures for traceability.
    if redundant_reduced:
        save_redundant_reduced_structures(
            redundant_reduced,
            output_base_dir,
            input_source,
            total_processed=len(clean_data_for_clustering)
        )

    total_skipped_all = (len(skipped_files) + total_imag_clustered_with_normal
                         + total_imag_need_recalc + total_non_converged_critical
                         + len(redundant_reduced))
    if total_files_attempted > 0:
        total_skipped_percentage = (total_skipped_all / total_files_attempted) * 100
        critical_total = total_imag_need_recalc + total_non_converged_critical
        critical_skipped_percentage = (critical_total / total_files_attempted) * 100
        total_skipped_str = f"{total_skipped_all} ({total_skipped_percentage:.1f}%)"
        critical_skipped_str = f"{critical_total} ({critical_skipped_percentage:.1f}%)"
    else:
        total_skipped_str = str(total_skipped_all)
        critical_skipped_str = str(total_imag_need_recalc + total_non_converged_critical)

    # Reduced-vector structures that could not match any fullest-tier cluster
    # are flagged as critical for redo (recalculation / Hessian / imaginary displacement).
    # These were collected during tier-based matching in all_reduced_unmatched.
    reduced_unmatched_critical = list(all_reduced_unmatched)
    if reduced_unmatched_critical:
        print(f"\nReduced-vector criticals: {len(reduced_unmatched_critical)} structure(s) did not match any full-feature cluster.")
        print(f"  These structures need recalculation (redo mode).")

    if total_files_attempted > 0:
        reduced_unmatched_percentage = (len(reduced_unmatched_critical) / total_files_attempted) * 100
        reduced_unmatched_str = f"{len(reduced_unmatched_critical)} ({reduced_unmatched_percentage:.1f}%)"
    else:
        reduced_unmatched_str = str(len(reduced_unmatched_critical))

    _method_label = {"knee": "knee detection",
                     "knee-capped": "knee detection, capped to empirical 2.0",
                     # No nested parens: the summary wraps this in "(...)" and
                     # --th=opt's fallback regex reads up to the first ')'.
                     "knee-uncapped": "knee detection, uncapped by --th=knee"}
    if rmsd_only:
        # No feature-space tau exists in geometry-only mode; the cut is the RMSD
        # itself. Deliberately not emitting the machine-readable 'details:' line
        # so a later --th=opt stage never picks an Å value up as a feature tau.
        _cosmic_threshold_text = (f"COSMIC threshold: RMSD = {rmsd_threshold:.3f} Å "
                                  f"(geometry-only clustering, --rmsd-only)")
    elif is_compare_mode or not resolved_threshold_entries:
        _cosmic_threshold_text = "COSMIC threshold: N/A"
    elif len(resolved_threshold_entries) == 1:
        _e = resolved_threshold_entries[0]
        _tau = _e['tau']
        _src = _e.get('source', '')
        _method = _method_label.get(_src, _src)
        if _method:
            _cosmic_threshold_text = f"COSMIC threshold: tau = {_tau:.4f} ({_method})"
        else:
            _cosmic_threshold_text = f"COSMIC threshold: tau = {_tau:.4f}"
        # Machine-readable detail line consumed by --th=opt on later refinement
        # stages (parse_opt_params_from_summary / _OPT_DETAIL_RE). Emitting it
        # lets a post-refinement cosmic reuse this stage's exact tau.
        _r = _e.get('r_thresh'); _neff = _e.get('n_eff'); _dmed = _e.get('d_med')
        if _r is not None and _neff is not None and _dmed is not None and _src:
            _cosmic_threshold_text += (
                f"\n  details: tau = {_tau:.4f}, r >= {_r:.4f}, "
                f"N_f = {_neff:.2f}, d_med = {_dmed:.4f}, source = {_src}")
    else:
        _lines = ["COSMIC threshold (per H-bond group):"]
        for _e in resolved_threshold_entries:
            _label = (f"H = {_e['group_label']}" if _e['group_label'] is not None else "all")
            _tau = _e['tau']
            _src = _e.get('source', '')
            _method = _method_label.get(_src, _src)
            if _method:
                _lines.append(f"  {_label:<6} : tau = {_tau:.4f} ({_method})")
            else:
                _lines.append(f"  {_label:<6} : tau = {_tau:.4f}")
        _cosmic_threshold_text = "\n".join(_lines)

    # For a manually specified numeric threshold there is no auto-detection method
    if not is_compare_mode and not rmsd_only and not resolved_threshold_entries and isinstance(threshold, (int, float)):
        _cosmic_threshold_text = f"COSMIC threshold: tau = {threshold}"

    for i, line in enumerate(summary_file_content_lines):
        if "<COSMIC_THRESHOLD_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<COSMIC_THRESHOLD_PLACEHOLDER>", _cosmic_threshold_text)
        if "<TOTAL_CLUSTERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_CLUSTERS_PLACEHOLDER>", str(total_clusters_outputted))
        if "<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>", str(total_rmsd_outliers_first_pass))
        if "<TOTAL_SKIPPED_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_SKIPPED_PLACEHOLDER>", total_skipped_str)
        if "<IMAG_NEED_RECALC_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<IMAG_NEED_RECALC_PLACEHOLDER>", critical_skipped_str)

    # Add comparison-specific details at the very end if in comparison mode
    if is_compare_mode:
        comparison_specific_summary_lines.append("\n" + "=" * 75 + "\n")
        comparison_specific_summary_lines.append("Comparison Parameters:\n")
        if weights:
            comparison_specific_summary_lines.append("  Applied Feature Weights:")
            for key, value in weights.items():
                comparison_specific_summary_lines.append(f"    - {key}: {value}")
        if abs_tolerances:
            comparison_specific_summary_lines.append("  Applied Absolute Tolerances:")
            for key, value in abs_tolerances.items():
                # Format the float to a fixed number of decimal places to avoid scientific notation
                # Adjust precision as needed, e.g., for 1e-5 use 5 decimal places, for 0.5 use 1 decimal place
                # A general approach is to find the number of decimal places or use a high fixed number.
                # For simplicity and to cover common cases, I'll use a fixed high precision like 7.
                formatted_value = f"{value:.7f}".rstrip('0').rstrip('.') if '.' in f"{value:.7f}" else f"{int(value)}"
                comparison_specific_summary_lines.append(f"    - {key}: {formatted_value}")
        summary_file_content_lines.extend(comparison_specific_summary_lines)

    # Create cluster ID mapping for motifs BEFORE Boltzmann analysis
    for cluster_idx, cluster_members in enumerate(all_final_clusters):
        if cluster_members:
            # Get the cluster ID from the first member (all members have the same cluster ID)
            cluster_id = cluster_members[0].get('_cluster_global_id', cluster_idx + 1)
            cluster_id_mapping[cluster_idx] = cluster_id

    # --- RE-COMPUTE Boltzmann using FINAL cluster IDs ---
    # This ensures traceability: cluster IDs in Boltzmann match cluster IDs in summary
    # Boltzmann distribution is only meaningful with Gibbs free energy (freq mode).
    boltzmann_final_data = {}
    final_global_min_energy = None
    final_global_min_filename = "N/A"
    final_global_min_cluster_id = "N/A"

    # First pass: find representatives and global minimum
    final_representatives = []
    if _dataset_has_freq:
        for cluster_members in all_final_clusters:
            if not cluster_members:
                continue

            # Get the final cluster ID
            final_cluster_id = cluster_members[0].get('_cluster_global_id')
            if final_cluster_id is None:
                continue

            # Find lowest energy representative in this cluster (excluding imaginary freq structures)
            valid_members = [m for m in cluster_members if not m.get('_has_imaginary_freqs', False)]
            if not valid_members:
                valid_members = cluster_members  # Use all if none are valid

            rep = min(valid_members,
                      key=lambda x: (sorting_energy(x, mode), x['filename']))

            energy_for_boltz = rep.get('composite_gibbs') or rep.get('gibbs_free_energy')
            if energy_for_boltz is not None:
                final_representatives.append({
                    'cluster_id': final_cluster_id,
                    'filename': rep['filename'],
                    'energy': energy_for_boltz,
                    'dft_gibbs': rep.get('composite_dft_gibbs'),
                    'ccsdt_elec': rep.get('composite_ccsdt_elec'),
                    'thermal': rep.get('composite_thermal'),
                    'cluster_size': len(cluster_members)
                })

    # Find global minimum
    if final_representatives:
        min_rep = min(final_representatives, key=lambda x: x['energy'])
        final_global_min_energy = min_rep['energy']
        final_global_min_filename = min_rep['filename']
        final_global_min_cluster_id = min_rep['cluster_id']

        # Calculate Boltzmann factors
        sum_factors = 0.0
        for rep in final_representatives:
            delta_e = rep['energy'] - final_global_min_energy
            if BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k == 0:
                factor = 1.0 if delta_e == 0 else 0.0
            else:
                factor = np.exp(-delta_e / (BOLTZMANN_CONSTANT_HARTREE_PER_K * temperature_k))

            boltzmann_final_data[rep['cluster_id']] = {
                'energy': rep['energy'],
                'dft_gibbs': rep.get('dft_gibbs'),
                'ccsdt_elec': rep.get('ccsdt_elec'),
                'thermal': rep.get('thermal'),
                'filename': rep['filename'],
                'population': factor,
                'cluster_size': rep['cluster_size']
            }
            sum_factors += factor

        # Normalize populations
        if sum_factors > 0:
            for cid in boltzmann_final_data:
                boltzmann_final_data[cid]['population'] = (boltzmann_final_data[cid]['population'] / sum_factors) * 100.0
        else:
            for cid in boltzmann_final_data:
                boltzmann_final_data[cid]['population'] = 0.0

    # Detect if inputs are from a previous motif step to determine output naming
    # This enables the workflow: conf_### → motif_## → umotif_##
    all_input_filenames = [m.get('filename', '') for m in clean_data_for_clustering]
    output_prefix, folder_prefix, is_second_step = detect_motif_input_level(all_input_filenames)

    if is_second_step:
        print_step(f"Detected motif inputs - using '{output_prefix}_##' naming for unique motifs")

    # Create motifs folder with representative structures from each cluster
    # Pass boltzmann_final_data to sort motifs by population (highest population = motif_01)
    motif_to_cluster_mapping = create_unique_motifs_folder(all_final_clusters, output_base_dir, mode,
                                                          cluster_id_mapping=cluster_id_mapping,
                                                          output_prefix=output_prefix,
                                                          folder_prefix=folder_prefix,
                                                          boltzmann_data=boltzmann_final_data)

    # --- Create separate Boltzmann Distribution Analysis file ---
    boltzmann_file_content_lines = []
    if final_global_min_energy is not None:
        # Helper function to center text within 75 characters (same as summary)
        def center_text_boltzmann(text, width=75):
            return text.center(width)

        # Header for Boltzmann file - using the same beautiful ASCII art as summary
        boltzmann_file_content_lines.append("=" * 75)
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("***************************"))
        boltzmann_file_content_lines.append(center_text_boltzmann("* C O S M I C *"))
        boltzmann_file_content_lines.append(center_text_boltzmann("***************************"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("                             √≈≠==≈                                  ")
        boltzmann_file_content_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
        boltzmann_file_content_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
        boltzmann_file_content_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
        boltzmann_file_content_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
        boltzmann_file_content_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
        boltzmann_file_content_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
        boltzmann_file_content_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
        boltzmann_file_content_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
        boltzmann_file_content_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
        boltzmann_file_content_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Universidad de Antioquia - Medellín - Colombia"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Boltzmann Population Distribution Analysis"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann(version))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append(center_text_boltzmann("Química Física Teórica - QFT"))
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("")
        boltzmann_file_content_lines.append("=" * 75 + "\n")

        # Reference configuration info
        boltzmann_file_content_lines.append("Reference Configuration:")
        boltzmann_file_content_lines.append(f"  Structure: {os.path.splitext(final_global_min_filename)[0]} from cluster_{final_global_min_cluster_id}")
        boltzmann_file_content_lines.append(f"  Reference Energy (Emin): {final_global_min_energy:.6f} Hartree ({hartree_to_kcal_mol(final_global_min_energy):.2f} kcal/mol, {hartree_to_ev(final_global_min_energy):.2f} eV)")
        boltzmann_file_content_lines.append(f"  Temperature (T): {temperature_k:.2f} K")
        boltzmann_file_content_lines.append("")

        # Population by Energy Minimum (gi = 1)
        boltzmann_file_content_lines.append("=" * 60)
        boltzmann_file_content_lines.append("Population by Energy Minimum")
        boltzmann_file_content_lines.append("(assuming non-degeneracy, gi = 1)")
        boltzmann_file_content_lines.append("=" * 60)
        boltzmann_file_content_lines.append("")

        # Sort by population percentage descending for better readability
        # Number motifs by population rank: highest population = motif_01/umotif_01
        sorted_final_data = sorted(boltzmann_final_data.items(), key=lambda item: item[1]['population'], reverse=True)

        # Add section header for motif assignment
        display_name = "Unique Motif (umotif)" if output_prefix == 'umotif' else "Motif"
        boltzmann_file_content_lines.append(f"{display_name} Assignment Summary")
        boltzmann_file_content_lines.append("(sorted by Boltzmann population)\n")

        for motif_rank, (cluster_id, data) in enumerate(sorted_final_data, 1):
            # Motif number is based on population rank (most populated = 01)
            cluster_line = f"cluster_{cluster_id} ({output_prefix}_{motif_rank:02d})"

            boltzmann_file_content_lines.append(cluster_line)
            boltzmann_file_content_lines.append(f"  From structure: {os.path.splitext(data['filename'])[0]}")
            # For composite (eref) runs, break the composite Gibbs energy down
            # into its contributions before the combined value: the Gibbs energy
            # from the geometry-refinement stage (thermal-correction source) and
            # the electronic energy from the energy-refinement stage. Labels are
            # stage-based so they stay method-agnostic.
            if mode.has_composite:
                if data.get('dft_gibbs') is not None:
                    dft_g = data['dft_gibbs']
                    boltzmann_file_content_lines.append(f"  Geometry Refinement Gibbs Energy: {dft_g:.6f} Hartree ({hartree_to_kcal_mol(dft_g):.2f} kcal/mol, {hartree_to_ev(dft_g):.2f} eV)")
                if data.get('thermal') is not None:
                    therm = data['thermal']
                    boltzmann_file_content_lines.append(f"  Thermal Correction: {therm:.6f} Hartree ({hartree_to_kcal_mol(therm):.2f} kcal/mol, {hartree_to_ev(therm):.2f} eV)")
                if data.get('ccsdt_elec') is not None:
                    cc_e = data['ccsdt_elec']
                    boltzmann_file_content_lines.append(f"  Energy Refinement: {cc_e:.6f} Hartree ({hartree_to_kcal_mol(cc_e):.2f} kcal/mol, {hartree_to_ev(cc_e):.2f} eV)")
            energy_label = "Composite Gibbs Energy" if mode.has_composite else "Gibbs Energy"
            boltzmann_file_content_lines.append(f"  {energy_label}: {data['energy']:.6f} Hartree ({hartree_to_kcal_mol(data['energy']):.2f} kcal/mol, {hartree_to_ev(data['energy']):.2f} eV)")
            # Relative Gibbs energy (ΔG) referenced to the global minimum (umotif_01).
            delta_g = data['energy'] - final_global_min_energy
            boltzmann_file_content_lines.append(f"  Relative Gibbs Energy (ΔG): {delta_g:.6f} Hartree ({hartree_to_kcal_mol(delta_g):.2f} kcal/mol, {hartree_to_ev(delta_g):.2f} eV)")
            boltzmann_file_content_lines.append(f"  Population: {data['population']:.2f} %")
            boltzmann_file_content_lines.append("")

        boltzmann_file_content_lines.append("=" * 75)

    # Write separate files
    summary_file = os.path.join(output_base_dir, "clustering_summary.txt")
    with open(summary_file, "w", newline='\n', encoding='utf-8') as f:
        f.write("\n".join(summary_file_content_lines))

    # Write Boltzmann distribution file only if we have the analysis data
    if boltzmann_file_content_lines:
        boltzmann_file = os.path.join(output_base_dir, "boltzmann_distribution.txt")
        with open(boltzmann_file, "w", newline='\n', encoding='utf-8') as f:
            f.write("\n".join(boltzmann_file_content_lines))
        print_step(f"Boltzmann distribution saved to '{os.path.basename(boltzmann_file)}'")

    print()
    print_step(f"Clustering summary saved to '{os.path.basename(summary_file)}'")
    vprint(f"   Full path: {output_base_dir}")

    # Emit marker for workflow detection: opt-only mode means no true minima.
    if not _dataset_has_freq:
        print("COSMIC_OPT_ONLY_MODE")


__all__ = [
    "get_cpu_count_fast",
    "perform_clustering_and_analysis",
]
