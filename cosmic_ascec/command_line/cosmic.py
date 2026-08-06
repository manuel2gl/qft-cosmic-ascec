"""``cosmic`` console entry point — clustering on a directory of structures.

Run ``cosmic <folder>`` to cluster the structures in ``<folder>`` into
representative motifs. With no positional argument, an interactive folder
picker is launched.

Three input types are accepted, in this precedence order when a folder holds
more than one: ``*.out``, ``*.log``, and ``*.xyz``. The first two are QM
outputs. The third is plain Cartesian coordinates — the entry point for large
systems, where a geometry exists long before a converged QM output does.
Eight of the fifteen clustering columns follow from coordinates alone; ``--sp``
adds four more via one xTB single point per structure. The remaining three
(Gibbs energy, first/last vibrational frequency) need a frequency calculation.
A ``.xyz`` file holding several frames (a concatenated ensemble, a trajectory)
is split into one structure per record automatically.

Flag surface (full list in ``cosmic --help``):

* ``--threshold/--th`` — cut height for the UPGMA tree; default is automatic
  (knee detection on the sorted merge-height curve, capped at the empirical
  τ=2.0); ``--th=knee`` applies the detected knee uncapped.
* ``--rmsd`` — enable geometric RMSD post-processing.
* ``--rmsd-only`` — cluster on Cartesian RMSD alone (no feature vector).
* ``--rmsd-heavy`` — exclude hydrogens from RMSD (default is all-atom).
* ``--cores/-j N`` — parallel workers for feature extraction.
* ``--reprocess-files/-r`` — invalidate any existing cache and re-parse.
* ``--weights`` / ``--partialweights`` — per-feature weights.
* ``--compare`` — two-stage compare mode.
* ``-T/--temperature`` — temperature for Boltzmann populations (K).
* ``--prev-out-dir`` — sibling stage for composite-Gibbs energy lookup.
* ``--data`` — dump the feature matrix and exit.
* ``--sp`` / ``--charge`` / ``--uhf`` — xTB single points for XYZ input.

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
from cosmic_ascec.clustering.features.xtb_sp import SP_METHODS, resolve_sp_method
from cosmic_ascec.clustering.features.xyz_input import (
    XYZ_EXTENSION,
    XYZ_GLOB,
    is_xyz_byproduct,
)
from cosmic_ascec.clustering.orchestrator import (
    get_cpu_count_fast,
    perform_clustering_and_analysis,
)
from cosmic_ascec.clustering.thresholds import resolve_opt_params_from_sibling_cosmic


# Optional-value flags: `--rmsd`/`--rmsd-only` take a float, or none at all
# (in which case this default applies). Kept here so the preprocessor and the
# argparse `const=` stay in sync.
_OPTIONAL_FLOAT_FLAGS = ('--rmsd', '--rmsd-only', '--only-rmsd')
_OPTIONAL_FLOAT_DEFAULT = 1.0


def _is_number(token):
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


# Input globs in precedence order. A folder that holds QM outputs is clustered
# on those; ``*.xyz`` is the fallback, so adding the geometry-only front-end
# cannot change how any existing run directory is interpreted.
_INPUT_PRECEDENCE = ("*.out", "*.log", XYZ_GLOB)


def _available_inputs(folder):
    """Return ``{glob_pattern: [paths]}`` for the clustering inputs in *folder*.

    Only two things are excluded, and both are excluded because they are not
    single structures: xTB optimisation trajectories (``*.xtbopt.log``, and
    ``*.xtbtraj.xyz`` / ``*_trj.xyz`` via :func:`is_xyz_byproduct`).

    Optimised geometries (``*.xtbopt.xyz``) are *kept* — a folder of them is
    the natural way to hand COSMIC the results of an optimisation stage. When
    a folder holds both an xTB input and its own result the pair is resolved
    later, in favour of the optimised structure.
    """
    found = {}
    for pattern in _INPUT_PRECEDENCE:
        matches = glob.glob(os.path.join(folder, pattern))
        if pattern == "*.log":
            matches = [f for f in matches if not f.endswith('.xtbopt.log')]
        elif pattern == XYZ_GLOB:
            matches = [f for f in matches if not is_xyz_byproduct(f)]
        if matches:
            found[pattern] = sorted(matches)
    return found


def _preferred_pattern(available):
    """Pick the glob to process from an ``_available_inputs`` mapping."""
    for pattern in _INPUT_PRECEDENCE:
        if available.get(pattern):
            return pattern
    return None


def preprocess_j_argument(argv):
    """
    Preprocesses command line arguments:
    - Handle -j8 format (no space) by converting it to -j 8.
    - Extract boolean flags (--verbose, etc.) that appear after --compare
      so they are not consumed as file arguments by nargs='+'.
    - Pin the default on a bare `--rmsd` / `--rmsd-only` that is followed by
      the FOLDER positional, so `cosmic --rmsd-only mydir` does not fail with
      "invalid float value: 'mydir'".

    Verbatim port of cosmic-v01's ``preprocess_j_argument`` (lines 4282-4308),
    plus the optional-value pinning above.
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
    for i, arg in enumerate(remaining):
        if arg.startswith('-j') and len(arg) > 2 and arg[2:].isdigit():
            processed_argv.extend(['-j', arg[2:]])
        elif arg in _OPTIONAL_FLOAT_FLAGS:
            nxt = remaining[i + 1] if i + 1 < len(remaining) else None
            takes_next = nxt is not None and not nxt.startswith('-') and _is_number(nxt)
            if takes_next:
                processed_argv.append(arg)
            else:
                processed_argv.append(f"{arg}={_OPTIONAL_FLOAT_DEFAULT}")
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
  COSMIC clusters molecular structures by a physicochemical feature vector
  (up to 15 descriptors: electronic energy, HOMO-LUMO gap, dipole, rotational
  constants, vibrational frequencies, H-bond geometry). The vector is dynamic —
  it uses whatever each input provides and reports motifs from the largest
  feature pool available. Redundant structures collapse into unique conformational
  families; each family's lowest-energy structure is the representative motif.

  Input is a folder of .out/.log QM outputs, or of .xyz coordinate files. Plain
  coordinates give the 8 geometry-derived descriptors (nuclear repulsion,
  rotational constants A/B/C, and the four H-bond columns) with no QM run at
  all — the entry point for large systems. '--sp' adds 4 more via one xTB
  single point per structure. A multi-frame .xyz is split per frame.

HOW IT WORKS:
  1. Parse QM outputs (.log/.out) or coordinates (.xyz) into the feature vector
  2. Z-standardize each feature (drop near-constant columns)
  3. Build a UPGMA tree (SciPy average linkage, Euclidean distance)
  4. Cut it at the threshold (default 'auto' = knee of the merge-height curve,
     capped at the empirical 2.0 — '--th=knee' lifts that cap; Mojena is
     plotted only as a diagnostic)
  5. Optional RMSD pass to split geometric look-alikes within a family
  6. Flag imaginary frequencies and convergence failures

KEY OPTIONS:
  --th=auto|knee|opt|FLOAT
                        Dendrogram cut (default auto). 'knee' is auto with the
                        empirical τ=2.0 ceiling disabled: the detected knee is
                        applied even above 2.0. 'opt' reuses the τ from the
                        sibling post-opt run — use it for refinement-stage cosmic
                        so the partition stays consistent. A float overrides
                        (2.0 = legacy 2-sigma; <1 tight; 3-4 loose).
  --rmsd[=FLOAT]        Geometric validation in Å (default 1.0 if bare).
  --rmsd-only[=FLOAT]   Cluster on Cartesian RMSD alone (default 1.0 Å): no
                        feature vector, the tree is cut directly in Å.
  --rmsd-heavy          Exclude hydrogens from RMSD. Default is all-atom RMSD,
                        as used by CREST and ORCA GOAT (both 0.125 Å default).
  -j, --cores INT       CPU cores (default: auto-detect).
  --partialweights      Tuned weights for semiempirical/xTB (down-weights noisy
                        orbital/dipole/H-bond features). Added by the web GUI for
                        preliminary runs; leave off for DFT/post-HF.
  --weights STRING      Manual weights, e.g. '(energy=0.3)(gap=0.2)'.
  --group-hb            Cluster separately per H-bond count (one dendrogram each).
  --sp[=METHOD]         XYZ input only: one xTB single point per structure, adding
                        electronic energy, HOMO, HOMO-LUMO gap and dipole moment
                        to the 8 geometry-derived descriptors (8 of 15 columns
                        become 12). METHOD is gfn2 (default), gfn1, gfn0, or
                        gfnff. Gibbs and the vibrational frequencies still need a
                        frequency calculation, which a single point is not.
  --charge INT          Total charge for the --sp single points (default 0).
  --uhf INT             Unpaired electrons for the --sp single points (default 0).
  -T FLOAT              Temperature (K) for Boltzmann populations (default 298.15).
  --compare FILE...     Direct pairwise comparison of ≥2 files (no folder).
  --reprocess-files     Ignore the descriptor cache and re-parse outputs.
  FOLDER                Directory of .out/.log QM outputs or .xyz coordinates —
                        or a single .xyz file (default: current / interactive).

MAIN OUTPUTS:
  clustering_summary.txt   Full report (clusters, τ source, similarity floors)
  dendrogram_images/       Annotated dendrogram(s)
  extracted_clusters/      One folder per family + representative motif
  skipped_structures/      Imaginary-frequency / non-converged structures

EXAMPLES:
  cosmic -j4                       Auto threshold, 4 cores (typical run)
  cosmic --rmsd=1 -j4              Add 1.0 Å geometric split
  cosmic --rmsd-only -j4           Cluster on RMSD alone (1.0 Å cut)
  cosmic --rmsd-only=0.125 -j4     RMSD-only at the CREST / GOAT default cut
  cosmic --th=opt -j4              Refinement stage: reuse the post-opt τ
  cosmic --th=knee -j4             Knee detection, uncapped (allow τ > 2.0)
  cosmic --th=2.0                  Force the legacy 2-sigma cut
  cosmic --partialweights -j4      Preliminary xTB / semiempirical screening
  cosmic xyz_dir -j4               Cluster plain coordinates (8 descriptors)
  cosmic xyz_dir --sp gfn2 -j4     Add xTB GFN2 single points (8 → 12 descriptors)
  cosmic xyz_dir --sp gfnff -j4    Same via the GFN-FF force field: far faster on
                                   very large systems, but energy only (no HOMO,
                                   gap or dipole — it has no electronic structure)
  cosmic ensemble.xyz -j4          Cluster a multi-frame file (split per frame)
  cosmic --compare a.out b.out     Compare two structures directly

TYPICAL PIPELINE (or use the automated protocol in one .asc file):
  ascec input.asc r5 --concurrent=5   → 5 replicate annealing runs
  ascec opt template.inp launcher.sh  → build + run optimization inputs
  ascec sort                          → collect and rank
  cosmic -j4                          → unique motifs

SUPPORTED FORMATS:
  Gaussian .log (cclib) · ORCA 5.0.x .out (cclib) · ORCA 6.1+ .out (OPI)
  ORCA 6.0 is not supported — use 5.0.x or 6.1+.
  Plain .xyz coordinates, single- or multi-frame — no QM output needed. Gives the
  8 geometry-derived descriptors; add --sp for the 4 electronic ones. Precedence
  in a mixed folder is .out > .log > .xyz. An xTB input superseded by its own
  .xtbopt.xyz result is dropped, so a structure is never clustered twice.

CITATION:
  Manuel, G.; Sara, G.; Albeiro, R. Universidad de Antioquia (2026)
  Repository: https://github.com/manuel2gl/qft-cosmic-ascec
""")
    # Clustering threshold: default 'auto' detects the elbow of the merge-height
    # curve per case; pass a float to override (e.g. 2.0 for legacy 2-sigma rule).
    parser.add_argument("--threshold", "--th", type=str, default="auto",
                        metavar="FLOAT|auto|knee|opt",
                        help="UPGMA distance threshold for dendrogram cut. Default 'auto' "
                             "detects the elbow of the merge-height curve per case "
                             "(recommended for atomic clusters and van der Waals systems); "
                             "a knee above the empirical ceiling τ=2.0 is capped to 2.0. "
                             "'knee' is the same detection with that ceiling disabled — the "
                             "detected knee is applied even when it is higher than 2.0. "
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
    parser.add_argument("--rmsd-only", "--only-rmsd", type=float, nargs='?', const=1.0,
                        default=None, metavar="FLOAT", dest="rmsd_only",
                        help="cluster on Cartesian RMSD alone in Ångström (default: 1.0): "
                             "skips the physicochemical feature vector entirely, so the "
                             "dendrogram is cut directly in Å. Ignores --threshold, --weights, "
                             "--partialweights.")
    parser.add_argument("--rmsd-heavy", action="store_true", dest="rmsd_heavy",
                        help="measure RMSD over heavy atoms only (exclude hydrogens). "
                             "Default is all-atom RMSD, matching CREST (--rmsd) and ORCA GOAT.")

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
                        help="apply tuned weights for preliminary semiempirical / xTB runs "
                             "(down-weights noisy orbital, dipole, and H-bond features). "
                             "Added by the web GUI for preliminary screening; leave off for "
                             "refined DFT/post-HF output.")

    parser.add_argument("--data", type=str, default=None, metavar="PKL",
                        help="extract per-configuration feature vectors from the given "
                             "data_cache_*.pkl file and write features.csv (labeled with units), "
                             "matrix.csv, and matrix.npy next to it (override with --output-dir). "
                             "All-NaN columns are dropped; cluster column only emitted when labels "
                             "are available. Exits after writing; skips clustering.")

    parser.add_argument("--sp", type=str, nargs='?', const='', default=None,
                        metavar="METHOD",
                        help="run one xTB single point per structure to add electronic "
                             "energy, HOMO, HOMO-LUMO gap and dipole moment to the "
                             "geometry-only feature vector (8 of the 15 columns become 12). "
                             f"METHOD is one of {', '.join(sorted(SP_METHODS))} "
                             "(default gfn2; gfnff is the force field, fastest for very "
                             "large systems, energy only). XYZ input only.")

    parser.add_argument("--charge", type=int, default=None, metavar="N",
                        help="total charge passed to the --sp single points (default 0).")

    parser.add_argument("--uhf", type=int, default=None, metavar="N",
                        help="number of unpaired electrons passed to the --sp single "
                             "points (default 0).")

    # Hidden/advanced options
    parser.add_argument("--min-std-threshold", type=float, default=1e-6,
                        help=argparse.SUPPRESS)
    parser.add_argument("--abs-tolerance", type=str, default="",
                        help=argparse.SUPPRESS)
    parser.add_argument("--update-cache", type=str, default=None,
                        help=argparse.SUPPRESS)

    # Positional argument
    parser.add_argument('input_source', nargs='?', default=None, metavar="FOLDER",
                        help='directory containing .out/.log QM outputs or .xyz '
                             'coordinate files (a single .xyz file is also accepted)')


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
    elif isinstance(args.threshold, str) and args.threshold.lower() == "knee":
        # Like 'auto', but the empirical τ=2.0 ceiling is not applied: the
        # detected knee is used even when it sits above it.
        args.threshold = "knee"
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
            parser.error("--threshold must be 'auto', 'knee', 'opt', or a number")

    clustering_threshold = args.threshold

    # --rmsd-only replaces the property-based clustering with a pure geometry
    # partition; its value is the RMSD cut, so it also supplies the RMSD
    # threshold when --rmsd was not given separately. Resolved before --th=opt
    # so no sibling summary is parsed for a threshold that is never used.
    rmsd_only_mode = args.rmsd_only is not None
    if rmsd_only_mode:
        if clustering_threshold != "auto":
            print("WARNING: --rmsd-only ignores --threshold (the cut is the RMSD value in Å).")
        clustering_threshold = "auto"

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
    if rmsd_only_mode:
        rmsd_validation_threshold = args.rmsd_only
        _atom_set = "heavy atoms only" if args.rmsd_heavy else "all atoms"
        print(f"RMSD-only mode: clustering on Cartesian RMSD ({_atom_set}) at "
              f"{rmsd_validation_threshold:.3f} Å (feature vector not used).")

    try:
        sp_method = resolve_sp_method(args.sp)
    except ValueError as exc:
        parser.error(str(exc))

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
        file_extension_pattern_for_compare = (
            extensions[0] if extensions[0] in ['.log', '.out', XYZ_EXTENSION] else None
        )
        if not file_extension_pattern_for_compare:
            print("Error: Provided files do not have .log, .out or .xyz extensions.")
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
            rmsd_only=rmsd_only_mode,
            rmsd_heavy=args.rmsd_heavy,
            sp_method=sp_method if file_extension_pattern_for_compare == XYZ_EXTENSION else None,
            sp_charge=args.charge,
            sp_uhf=args.uhf,
        )
        print(f"\n--- Finished comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")

    else: # Normal mode (folder processing)
        if args.input_source:
            # Non-interactive mode. A bare .xyz file is accepted as well as a
            # folder — a single ensemble or trajectory file is a natural way to
            # hand over a large set of structures, and it is split per frame
            # downstream.
            if os.path.isfile(args.input_source):
                if not args.input_source.lower().endswith(XYZ_EXTENSION):
                    print(f"Error: '{args.input_source}' is a file; only .xyz files can be "
                          f"given directly. Pass the containing folder instead.")
                    return 1
                selected_folders = [os.path.dirname(os.path.abspath(args.input_source)) or current_dir]
                file_extension_pattern = XYZ_GLOB
            elif not os.path.isdir(args.input_source):
                print(f"Error: Input source '{args.input_source}' is not a directory.")
                return 1
            else:
                selected_folders = [args.input_source]

                # Auto-detect input type: QM outputs win over coordinates, so an
                # existing run directory is interpreted exactly as before.
                file_extension_pattern = _preferred_pattern(_available_inputs(args.input_source))
                if file_extension_pattern is None:
                    print(f"Error: No .out, .log or .xyz files found in '{args.input_source}'.")
                    return 1

        else:
            # Interactive mode
            all_potential_folders = [current_dir] + [d for d in glob.glob(os.path.join(current_dir, '*')) if os.path.isdir(d)]

            folder_inputs = {}
            for folder in all_potential_folders:
                available = _available_inputs(folder)
                if available:
                    folder_inputs[folder] = available

            all_valid_folders_to_display = sorted(folder_inputs)

            if not all_valid_folders_to_display:
                print("No subdirectories containing .out, .log or .xyz files found, or files are organized directly in the current directory.")
                return 0

            def _folder_label(folder):
                name = "./" if folder == current_dir else os.path.basename(folder)
                types = ', '.join(
                    pattern.lstrip('*')
                    for pattern in _INPUT_PRECEDENCE
                    if folder_inputs[folder].get(pattern)
                )
                return f"{name} (Contains: {types})"

            if len(all_valid_folders_to_display) == 1:
                # Nothing to choose between — a menu with one entry is just a
                # keystroke tax. Announce the folder and get on with it.
                selected_folders = list(all_valid_folders_to_display)
                print(f"\nProcessing the only folder found: {_folder_label(selected_folders[0])}")
            else:
                print("\nFound the following folder(s) containing structures:\n")
                for i, folder in enumerate(all_valid_folders_to_display):
                    print(f"  [{i+1}] {_folder_label(folder)}")

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

            # Which input types the chosen folder(s) actually offer.
            selected_patterns = [
                pattern for pattern in _INPUT_PRECEDENCE
                if any(folder_inputs[f].get(pattern) for f in selected_folders)
            ]

            file_extension_pattern = None
            if len(selected_patterns) > 1:
                menu = "\n".join(
                    f"  [{i + 1}] {pattern.lstrip('*')} files"
                    for i, pattern in enumerate(selected_patterns)
                )
                prompt = (
                    f"\nSeveral input types are present in the selected folder(s).\n"
                    f"Which would you like to process?\n{menu}\n"
                    f"  Enter your choice (1-{len(selected_patterns)}): "
                )
                while file_extension_pattern is None:
                    type_choice = input(prompt).strip()
                    try:
                        idx = int(type_choice) - 1
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(selected_patterns):
                        file_extension_pattern = selected_patterns[idx]
                    else:
                        print(f"Invalid choice. Please enter a number from 1 to {len(selected_patterns)}.")
            elif selected_patterns:
                file_extension_pattern = selected_patterns[0]
                print(f"\nOnly {file_extension_pattern.lstrip('*')} files found in the selected "
                      f"folder(s). Processing {file_extension_pattern.lstrip('*')} files.")
            else:
                print("\nNo .out, .log or .xyz files found in the selected folder(s) that match available types. Exiting.")
                return 0

        # Single points only make sense for coordinate input: a QM output
        # already carries its own energy, and recomputing it at a different
        # level would mix two methods in one feature column.
        if sp_method and file_extension_pattern != XYZ_GLOB:
            print(f"WARNING: --sp applies to .xyz input only; ignoring it for "
                  f"'{file_extension_pattern}' files.")
            sp_method = None

        print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_pattern}'...")
        for folder_path in selected_folders:
            display_name = os.path.basename(folder_path)
            if folder_path == current_dir:
                display_name = "./"
            print(f"\nProcessing folder: {display_name}\n")

            perform_clustering_and_analysis(folder_path, clustering_threshold, file_extension_pattern, rmsd_validation_threshold, output_directory, force_reprocess_cache, weights_dict, is_compare_mode=False, min_std_threshold=min_std_threshold_val, abs_tolerances=abs_tolerances_dict, num_cores=num_cores, temperature_k=temperature_k, group_hb=args.group_hb, prev_out_dir=args.prev_out_dir, partialweights=args.partialweights, rmsd_only=rmsd_only_mode, rmsd_heavy=args.rmsd_heavy, sp_method=sp_method, sp_charge=args.charge, sp_uhf=args.uhf)

            print(f"\nFinished processing folder: {display_name}\n")

    print()
    print_step("All selected molecular analyses complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
