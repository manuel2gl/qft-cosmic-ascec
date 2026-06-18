"""``cosmic`` console entry point — clustering on a directory of QM outputs.

Run ``cosmic <folder>`` to cluster the QM outputs in ``<folder>`` into
representative motifs. With no positional argument, an interactive folder
picker is launched.

Flag surface (full list in ``cosmic --help``):

* ``--threshold/--th`` — cut height for the UPGMA tree; default is automatic
  (knee detection on the sorted merge-height curve).
* ``--rmsd`` — enable geometric RMSD post-processing.
* ``--cores/-j N`` — parallel workers for feature extraction.
* ``--reprocess-files/-r`` — invalidate any existing cache and re-parse.
* ``--weights`` / ``--partialweights`` — per-feature weights.
* ``--compare`` — two-stage compare mode.
* ``-T/--temperature`` — temperature for Boltzmann populations (K).
* ``--prev-out-dir`` — sibling stage for composite-Gibbs energy lookup.
* ``--data`` — dump the feature matrix and exit.

The CLI is intentionally a thin shell over
:func:`cosmic_ascec.clustering.perform_clustering_and_analysis` — adding a
new flag means updating both the argparse spec here and the parameter the
orchestrator consumes.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from cosmic_ascec.clustering import console
from cosmic_ascec.clustering.console import print_step, print_version_banner
from cosmic_ascec.clustering.data_extraction import run_data_extraction
from cosmic_ascec.clustering.features.feature_spec import (
    DEFAULT_WEIGHTS,
    SEMIEMPIRICAL_WEIGHTS,
    parse_abs_tolerance_argument,
    parse_weights_argument,
)
from cosmic_ascec.clustering.orchestrator import (
    get_cpu_count_fast,
    perform_clustering_and_analysis,
)
from cosmic_ascec.clustering.thresholds import resolve_opt_params_from_sibling_cosmic


def preprocess_j_argument(argv):
    """
    Preprocesses command line arguments:
    - Handle -j8 format (no space) by converting it to -j 8.
    - Extract boolean flags (--verbose, etc.) that appear after --compare
      so they are not consumed as file arguments by nargs='+'.

    Verbatim port of cosmic-v01's ``preprocess_j_argument`` (lines 4282-4308).
    """
    # First pass: extract standalone boolean flags that could be trapped by --compare nargs='+'
    _bool_flags = {'-v', '--verbose', '-r', '--reprocess-files', '-V', '--version'}
    extracted_flags = []
    remaining = []
    for arg in argv:
        if arg in _bool_flags:
            extracted_flags.append(arg)
        else:
            remaining.append(arg)

    # Second pass: handle -j8 → -j 8
    processed_argv = []
    for arg in remaining:
        if arg.startswith('-j') and len(arg) > 2 and arg[2:].isdigit():
            processed_argv.extend(['-j', arg[2:]])
        else:
            processed_argv.append(arg)

    # Put boolean flags at the front so they are parsed before --compare
    return extracted_flags + processed_argv


def main(argv=None):
    """COSMIC clustering CLI — verbatim port of cosmic-v01.py lines 6296-6744.

    cosmic-v01's top-level ``__main__`` block; ``sys.exit``/``exit`` become
    ``return`` so the root ``cosmic.py`` shim can propagate the exit code.
    """
    parser = argparse.ArgumentParser(
        description="COSMIC (COnfigurational Similarity via Motif Identification Code) - Hierarchical clustering for quantum chemistry structures\nPhysicochemical feature-based discrimination of conformational families",
        usage="cosmic [OPTIONS] [FOLDER]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""DESCRIPTION:
  COSMIC performs topological clustering of quantum chemistry outputs
  using a multi-dimensional physicochemical feature vector (energy, HOMO-LUMO
  gap, dipole moment, rotational constants, vibrational frequencies, H-bond geometry).
  Hierarchical clustering with optional RMSD refinement identifies unique
  conformational families and filters redundant structures.

METHODOLOGY:
  1. Feature Extraction: Parse QM outputs (.log/.out) for scalar descriptors
  2. Z-score Standardization: Normalize features across different units
  3. Weighted Euclidean Distance: Calculate cosmic matrix
  4. Hierarchical Clustering: UPGMA linkage with 2-sigma threshold on Z-standardized features
     (Calinski-Harabasz + Silhouette score optimization)
  5. RMSD Refinement (optional): Distinguish geometric stereoisomers
  6. Quality Control: Flag imaginary frequencies and convergence failures

OPTIONS:
  Manual Override (deprecated):
    --threshold=FLOAT, --th=FLOAT Manual distance threshold (overrides statistical
                                  consensus method). Consider removing this flag.

  Geometric Validation:
    --rmsd=FLOAT                  Enable RMSD validation in Ångström
                                  If no value given, defaults to 1.0 Å
                                  Recommended: 0.5-1.0 for tight geometric control

  Processing Control:
    --cores=INT, -j=INT           Number of CPU cores (default: auto-detect)
    --reprocess-files             Ignore cache and force feature re-extraction
    --output-dir=PATH             Output directory (default: current directory)

  Advanced Features:
    --weights=STRING              Custom feature weights in format:
                                  '(energy=0.1)(gap=0.2)(dipole=0.15)'
    --compare FILE [FILE ...]     Direct comparison mode (minimum 2 files)
    -T=FLOAT, --temperature=FLOAT Temperature for Boltzmann analysis in K
                                  (default: 298.15)
    --group-hb                    Group structures by H-bond count before
                                  clustering (separate dendrograms per family)

  Output:
    -v, --verbose                 Enable detailed progress output
    -V, --version                 Display version information and exit

INPUT:
  FOLDER                          Directory containing QM output files
                                  (.log for Gaussian, .out for ORCA)
                                  If omitted, interactive folder selection

OUTPUT FILES:
  clustering_summary.txt          Comprehensive clustering report with statistics
  data_cache.pkl                  Cache file for output data
  dendrogram_images/              Hierarchical clustering dendrograms
    └── dendrogram.png            Single dendrogram (or dendrogram_H{N}.png with --group-hb)
  extracted_data/                 Raw data files (.dat) for each cluster
    └── cluster_*.dat
  extracted_clusters/             Individual cluster directories
    ├── cluster_1/                Single-member cluster (no combined file)
    │   ├── structure.xyz         Individual structure file
    │   └── structure.mol         MOL format (if OpenBabel available)
    └── cluster_2_5/              Multi-member cluster (5 members)
        ├── structure1.xyz        Individual XYZ files for each member
        ├── structure2.xyz
        ├── cluster_2_5.xyz       Combined multi-frame XYZ file
        └── cluster_2_5.mol       Combined MOL file
  skipped_structures/             Structures with imaginary frequencies (if any)
    ├── skipped_summary.txt       Details of skipped structures
    ├── clustered_with_normal/    Imaginary freq. clustered with valid structures
    └── need_recalculation/       Isolated imaginary freq. structures

EXAMPLES:
  Basic clustering (2-sigma threshold - recommended):
    cosmic                          Default threshold=2.0 (moderate)
    cosmic --rmsd=1                 Add RMSD validation at 1.0 Å
    cosmic calculation/             Process specific folder

  Manual threshold (deprecated):
    cosmic --th=2                   Manual clustering at threshold 2.0
    cosmic --th=1 --rmsd=0.5        Tight clustering with RMSD control

  Performance optimization:
    cosmic --th=2 -j=8              Use 8 CPU cores for parallel processing
    cosmic --reprocess-files        Force cache refresh after updates

  Direct comparison:
    cosmic --compare s1.log s2.log s3.log  Compare specific structures

  Custom analysis:
    cosmic --th=2 --weights='(energy=0.3)(gap=0.2)'  Weighted features
    cosmic --th=2 -T=350.0          Boltzmann analysis at 350 K

WORKFLOW INTEGRATION:
  COSMIC is typically used after ASCEC sampling and QM optimization:
    1. ascec input.in r5        → 5 replicated annealing runs
    2. ascec calc template.inp  → generate QM inputs
    3. [Run ORCA/Gaussian calculations]
    4. ascec sort               → organize results
    5. cosmic --th=2        → identify unique conformers

RECOMMENDATIONS:
  - Start with --th=2 for initial exploration
  - Use --rmsd=1 for systems with subtle geometric differences
  - Adjust --th value based on desired clustering granularity
  - Use --reprocess-files after modifying QM outputs or settings
  - Check dendrogram.png to validate threshold selection

SUPPORTED FORMATS:
  - Gaussian: .log files (via cclib parser)
  - ORCA 5.0.x: .out files (via cclib parser)
  - ORCA 6.1+: .out files (via OPI parser)
  Note: ORCA 6.0 is not supported; use 5.0.x or upgrade to 6.1+

CITATION:
  If you use COSMIC in your research, please cite:
  Manuel, G.; Sara, G.; Albeiro, R. Universidad de Antioquia (2026)

MORE INFORMATION:
    Repository:     https://github.com/manuel2gl/qft-cosmic-ascec
  Documentation:  See user manual for theoretical background
  Support:        Química Física Teórica - Universidad de Antioquia
""")
    # Clustering threshold: default 'auto' detects the elbow of the merge-height
    # curve per case; pass a float to override (e.g. 2.0 for legacy 2-sigma rule).
    parser.add_argument("--threshold", "--th", type=str, default="auto",
                        metavar="FLOAT|auto|opt",
                        help="UPGMA distance threshold for dendrogram cut. Default 'auto' "
                             "detects the elbow of the merge-height curve per case "
                             "(recommended for atomic clusters and van der Waals systems). "
                             "Pass a float to override: 2.0 for the legacy 2-sigma rule, "
                             "0.5 for tight, 3.0-4.0 for loose clustering. "
                             "'opt' reuses the raw τ resolved by the sibling post-opt cosmic "
                             "(read from its clustering_summary.txt) — the recommended mode "
                             "for post-refinement cosmic stages so the partition stays "
                             "consistent with the preliminary one. "
                             "('opt-pearson'/'opt-spread' are deprecated: they rebuilt τ from "
                             "the current run's N_f / median spread, which is unstable on the "
                             "small refined set; both now behave as 'opt'.)")

    # Geometric validation
    parser.add_argument("--rmsd", type=float, nargs='?', const=1.0, default=None, metavar="FLOAT",
                        help="RMSD geometric validation in Ångström (default: 1.0)")

    # Processing control
    parser.add_argument("--cores", "-j", type=int, default=None, metavar="INT",
                        help="number of CPU cores (default: auto-detect)")
    parser.add_argument("--reprocess-files", "-r", action="store_true",
                        help="ignore cache and force re-extraction")
    parser.add_argument("--output-dir", type=str, default=None, metavar="PATH",
                        help="output directory (default: current directory)")
    parser.add_argument("--weights", type=str, default="", metavar="STRING",
                        help="custom feature weights: '(energy=0.1)(gap=0.2)'")
    parser.add_argument("--compare", nargs='+', metavar="FILE",
                        help="direct comparison mode (minimum 2 files)")
    parser.add_argument("-T", "--temperature", type=float, default=298.15, metavar="FLOAT",
                        help="temperature for Boltzmann analysis in K (default: 298.15)")
    parser.add_argument("--prev-out-dir", type=str, default=None, metavar="PATH",
                        help="previous stage COSMIC base directory for composite energy: G = E_eref + (G_prev - E_prev)")

    # Output control
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable detailed progress output")
    parser.add_argument("-V", "--version", action="store_true",
                        help="display version and exit")

    parser.add_argument("--group-hb", action="store_true",
                        help="group structures by H-bond count before clustering (separate dendrograms per HB family)")

    parser.add_argument("--partialweights", action="store_true",
                        help="apply tuned weights for semiempirical / standalone xTB output "
                             "(down-weights noisy orbital, dipole, and H-bond features). "
                             "Recommended for PM3/AM1/xTB; leave off for DFT/post-HF.")

    parser.add_argument("--data", type=str, default=None, metavar="PKL",
                        help="extract per-configuration feature vectors from the given "
                             "data_cache_*.pkl file and write features.csv (labeled with units), "
                             "matrix.csv, and matrix.npy next to it (override with --output-dir). "
                             "All-NaN columns are dropped; cluster column only emitted when labels "
                             "are available. Exits after writing; skips clustering.")

    # Hidden/advanced options
    parser.add_argument("--min-std-threshold", type=float, default=1e-6,
                        help=argparse.SUPPRESS)
    parser.add_argument("--abs-tolerance", type=str, default="",
                        help=argparse.SUPPRESS)
    parser.add_argument("--update-cache", type=str, default=None,
                        help=argparse.SUPPRESS)

    # Positional argument
    parser.add_argument('input_source', nargs='?', default=None, metavar="FOLDER",
                        help='directory containing QM output files')


    # Preprocess arguments to handle -j8 format
    raw_args = sys.argv[1:] if argv is None else list(argv)
    processed_args = preprocess_j_argument(raw_args)
    args = parser.parse_args(processed_args)

    # Check if version is requested
    if args.version:
        print_version_banner()
        return 0

    # --data: dump feature vectors from the given cache file and exit.
    if args.data:
        return run_data_extraction(args.data, out_dir=args.output_dir)

    # Validate --threshold: accept "auto", "opt", or a float string.
    # "opt-pearson"/"opt-spread" are deprecated (they rebuilt τ from the current
    # run's N_f / median spread, which is unstable on the small refined set);
    # they are still accepted but transparently aliased to "opt" (raw τ reuse).
    _DEPRECATED_OPT_MODES = {"opt-pearson", "opt-spread"}
    if isinstance(args.threshold, str) and args.threshold.lower() == "auto":
        args.threshold = "auto"
    elif isinstance(args.threshold, str) and args.threshold.lower() == "opt":
        args.threshold = "opt"
    elif isinstance(args.threshold, str) and args.threshold.lower() in _DEPRECATED_OPT_MODES:
        print(f"WARNING: --th={args.threshold.lower()} is deprecated and unstable on "
              f"refined sets; using --th=opt (raw τ reuse) instead.")
        args.threshold = "opt"
    else:
        try:
            args.threshold = float(args.threshold)
        except (TypeError, ValueError):
            parser.error("--threshold must be 'auto', 'opt', or a number")

    clustering_threshold = args.threshold

    # Resolve --th=opt by parsing the sibling post-opt cosmic's
    # clustering_summary.txt for the raw resolved τ_opt and reusing it directly,
    # so a post-refinement cosmic keeps the same partition the preliminary found.
    if isinstance(clustering_threshold, str) and clustering_threshold == "opt":
        _opt_params = resolve_opt_params_from_sibling_cosmic(os.getcwd())
        if _opt_params is None:
            print(f"WARNING: --th=opt requested but no sibling "
                  f"cosmic*/clustering_summary.txt with parseable trust-score details "
                  f"was found; falling back to --th=auto.")
            clustering_threshold = "auto"
        else:
            _src_dir = os.path.basename(_opt_params.get("source_dir", "?"))
            def _fmt(v, nd=4):
                return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"
            print(f"--th=opt reading from sibling '{_src_dir}': "
                  f"τ_opt={_fmt(_opt_params['tau'])}, r_opt={_fmt(_opt_params['r'])}, "
                  f"N_f_opt={_fmt(_opt_params['n_eff'], 2)}, "
                  f"d_med_opt={_fmt(_opt_params['d_med'])}, source={_opt_params['source']}")
            clustering_threshold = float(_opt_params["tau"])
    rmsd_validation_threshold = args.rmsd
    output_directory = args.output_dir
    force_reprocess_cache = args.reprocess_files
    user_weights_dict = parse_weights_argument(args.weights)
    # Pick tuned semiempirical baseline only when --partialweights is passed;
    # otherwise stay method-agnostic with a flat 1.0 baseline.
    base_weights = SEMIEMPIRICAL_WEIGHTS if args.partialweights else DEFAULT_WEIGHTS
    weights_dict = dict(base_weights)
    weights_dict.update(user_weights_dict)  # user --weights override the baseline
    min_std_threshold_val = args.min_std_threshold
    abs_tolerances_dict = parse_abs_tolerance_argument(args.abs_tolerance)
    num_cores = args.cores if args.cores is not None else get_cpu_count_fast()
    temperature_k = args.temperature

    # Update the global verbose flag
    console.VERBOSE = args.verbose

    # Set default absolute tolerances if not provided via command line
    if not abs_tolerances_dict:
        abs_tolerances_dict = {
            "electronic_energy": 5e-6,
            "gibbs_free_energy": 5e-6,
            "homo_energy": 3e-4,
            "homo_lumo_gap": 3e-4,
            "dipole_moment": 1.5e-3,
            "vnn_nuclear_repulsion": 1e-4,   # V_NN is in Hartree, geometry-driven
            "rotational_constants_A": 7e-5,
            "rotational_constants_B": 3.5e-4,
            "rotational_constants_C": 3e-4,
            "first_vib_freq": 1e-2,
            "last_vib_freq": 0.3,
            "num_hydrogen_bonds": 0.5,        # integer-valued in practice
            "average_hbond_distance": 1e-3,
            "std_hbond_distance": 1e-3,
            "average_hbond_angle": 0.1
        }

    current_dir = os.getcwd()


    if args.compare:
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 files.")
            return 1

        compare_files = args.compare

        # Check that all files exist
        for file_path in compare_files:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return 1

        # Determine file extensions and check compatibility
        extensions = [os.path.splitext(f)[1].lower() for f in compare_files]
        unique_extensions = set(extensions)

        if len(unique_extensions) > 1:
            print(f"Warning: Comparing files with different extensions ({', '.join(unique_extensions)}). Proceeding, but ensure they are compatible.")

        # Use the extension of the first file for pattern
        file_extension_pattern_for_compare = extensions[0] if extensions[0] in ['.log', '.out'] else None
        if not file_extension_pattern_for_compare:
            print("Error: Provided files do not have .log or .out extensions.")
            return 1

        file_names = [os.path.basename(f) for f in compare_files]
        print(f"\n--- Comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")
        perform_clustering_and_analysis(
            input_source=compare_files,
            threshold=clustering_threshold,
            file_extension_pattern=file_extension_pattern_for_compare, # Pass for consistency, though not used for glob
            rmsd_threshold=rmsd_validation_threshold,
            output_base_dir=output_directory,
            force_reprocess_cache=True, # Always reprocess for comparison
            weights=weights_dict,
            is_compare_mode=True,
            min_std_threshold=min_std_threshold_val,
            abs_tolerances=abs_tolerances_dict,
            num_cores=num_cores,
            temperature_k=temperature_k,
            group_hb=args.group_hb,
            partialweights=args.partialweights,
        )
        print(f"\n--- Finished comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")

    else: # Normal mode (folder processing)
        if args.input_source:
            # Non-interactive mode
            if not os.path.isdir(args.input_source):
                print(f"Error: Input source '{args.input_source}' is not a directory.")
                return 1

            selected_folders = [args.input_source]

            # Auto-detect file extension
            log_files_input = [f for f in glob.glob(os.path.join(args.input_source, "*.log")) if not f.endswith('.xtbopt.log')]
            has_log = bool(log_files_input)
            has_out = bool(glob.glob(os.path.join(args.input_source, "*.out")))

            if has_out:
                file_extension_pattern = "*.out"
            elif has_log:
                file_extension_pattern = "*.log"
            else:
                print(f"Error: No .log or .out files found in '{args.input_source}'.")
                return 1

        else:
            # Interactive mode
            all_potential_folders = [current_dir] + [d for d in glob.glob(os.path.join(current_dir, '*')) if os.path.isdir(d)]

            folders_with_log_files = []
            folders_with_out_files = []

            for folder in all_potential_folders:
                # Exclude xTB trajectory files (*.xtbopt.log) — not calculation outputs
                log_files = [f for f in glob.glob(os.path.join(folder, "*.log")) if not f.endswith('.xtbopt.log')]
                has_log = bool(log_files)
                has_out = bool(glob.glob(os.path.join(folder, "*.out")))

                if has_log:
                    folders_with_log_files.append(folder)
                if has_out:
                    folders_with_out_files.append(folder)

            all_valid_folders_to_display = sorted(list(set(folders_with_log_files + folders_with_out_files)))

            if not all_valid_folders_to_display:
                print("No subdirectories containing .log or .out files found, or files are organized directly in the current directory.")
                return 0

            print("\nFound the following folder(s) containing quantum chemistry log/out files:\n")
            for i, folder in enumerate(all_valid_folders_to_display):
                display_name = os.path.basename(folder)
                if folder == current_dir:
                    display_name = "./"

                folder_types_present = []
                if folder in folders_with_log_files: folder_types_present.append(".log")
                if folder in folders_with_out_files: folder_types_present.append(".out")

                print(f"  [{i+1}] {display_name} (Contains: {', '.join(folder_types_present)})")

            selected_folders = []
            while True:
                choice = input("\nEnter the number of the folder to process, or type 'a' to process all: ").strip().lower()

                if choice == 'a':
                    selected_folders = all_valid_folders_to_display
                    break
                try:
                    folder_index = int(choice) - 1
                    if 0 <= folder_index < len(all_valid_folders_to_display):
                        selected_folders = [all_valid_folders_to_display[folder_index]]
                        break
                    else:
                        print("\nInvalid number. Please enter a valid number from the list.")
                except ValueError:
                    print("\nInvalid input. Please enter a number or 'a'.")

            selected_set_has_log = False
            selected_set_has_out = False
            for folder_path in selected_folders:
                if folder_path in folders_with_log_files:
                    selected_set_has_log = True
                if folder_path in folders_with_out_files:
                    selected_set_has_out = True
                if selected_set_has_log and selected_set_has_out:
                    break

            file_extension_pattern = None
            if selected_set_has_log and selected_set_has_out:
                while file_extension_pattern is None:
                    type_choice = input("\nBoth .log and .out files are present in the selected folder(s).\nWhich file type would you like to process?\n  [1] .log files\n  [2] .out files\n  Enter your choice (1 or 2): ").strip()
                    if type_choice == '1':
                        file_extension_pattern = "*.log"
                    elif type_choice == '2':
                        file_extension_pattern = "*.out"
                    else:
                        print("Invalid choice. Please enter '1' or '2'.")
            elif selected_set_has_log:
                file_extension_pattern = "*.log"
                print("\nOnly .log files found in the selected folder(s). Processing .log files.")
            elif selected_set_has_out:
                file_extension_pattern = "*.out"
                print("\nOnly .out files found in the selected folder(s). Processing .out files.")
            else:
                print("\nNo .log or .out files found in the selected folder(s) that match available types. Exiting.")
                return 0

        print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_pattern}'...")
        for folder_path in selected_folders:
            display_name = os.path.basename(folder_path)
            if folder_path == current_dir:
                display_name = "./"
            print(f"\nProcessing folder: {display_name}\n")

            perform_clustering_and_analysis(folder_path, clustering_threshold, file_extension_pattern, rmsd_validation_threshold, output_directory, force_reprocess_cache, weights_dict, is_compare_mode=False, min_std_threshold=min_std_threshold_val, abs_tolerances=abs_tolerances_dict, num_cores=num_cores, temperature_k=temperature_k, group_hb=args.group_hb, prev_out_dir=args.prev_out_dir, partialweights=args.partialweights)

            print(f"\nFinished processing folder: {display_name}\n")

    print()
    print_step("All selected molecular analyses complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
