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
* ``--shell`` / ``--nearest`` — reduce a solvated MD trajectory to the solute
  and its solvation shell, then cluster that. See
  :mod:`cosmic_ascec.file_formats.md_trajectory`.

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
from pathlib import Path

from cosmic_ascec.clustering import console
from cosmic_ascec.clustering.console import print_step, print_version_banner
from cosmic_ascec.clustering.data_extraction import run_data_extraction
from cosmic_ascec.clustering.features.feature_spec import (
    DEFAULT_ABS_TOLERANCES,
    DEFAULT_WEIGHTS,
    SEMIEMPIRICAL_WEIGHTS,
    parse_abs_tolerance_argument,
    parse_weights_argument,
)
from cosmic_ascec.clustering.features.xtb_sp import SP_METHODS, resolve_sp_method
from cosmic_ascec.clustering.features.mol_input import (
    MOL_CONVERTED_SUBDIR,
    MOL_EXTENSION,
    MOL_GLOB,
    find_mol_files,
    is_mol_byproduct,
    stage_mol_as_xyz,
)
from cosmic_ascec.clustering.features.xyz_input import (
    FRAMES_SUBDIR,
    XYZ_EXTENSION,
    XYZ_GLOB,
    is_xyz_byproduct,
)
from cosmic_ascec.clustering.orchestrator import (
    get_cpu_count_fast,
    perform_clustering_and_analysis,
)
from cosmic_ascec.clustering.thresholds import resolve_opt_params_from_sibling_cosmic
from cosmic_ascec import levels as _levels
from cosmic_ascec.exceptions import ClusteringError


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
# cannot change how any existing run directory is interpreted. ``*.mol`` sits
# below both: it carries no more information than the ``.xyz`` it converts to,
# so a folder holding a structure in both forms is still read as .xyz and the
# OpenBabel round-trip is skipped entirely.
_INPUT_PRECEDENCE = ("*.out", "*.log", XYZ_GLOB, MOL_GLOB)


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
        elif pattern == MOL_GLOB:
            matches = [f for f in matches if not is_mol_byproduct(f)]
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


#: Where a single named .xyz is staged so the downstream globs see only it.
SELECTED_SUBDIR = "xyz_selected"

#: Directories COSMIC creates to hold a normalised copy of its own input:
#: frames exploded out of a multi-frame file, a single named .xyz staged on its
#: own, and .mol converted to .xyz. Interactive discovery skips them — offering
#: one as a folder to process would present the same structures a second time,
#: under a name the user never chose. Naming one explicitly still works; only
#: the menu is filtered.
_STAGING_SUBDIRS = frozenset({SELECTED_SUBDIR, FRAMES_SUBDIR, MOL_CONVERTED_SUBDIR})


def _stage_single_xyz(path, base_dir):
    """Copy one named .xyz into a directory of its own and return that directory.

    ``cosmic ensemble.xyz`` has to mean *that* ensemble. The clustering pipeline
    works in terms of an input folder and re-globs it as it goes
    (:func:`~cosmic_ascec.clustering.orchestrator.perform_clustering_and_analysis`),
    so a lone file is honoured by giving it a folder containing exactly itself —
    the same device :func:`_stage_xyz_selection` uses when discovery narrows a
    set. Kept out of the frames directory the orchestrator writes into, so
    nothing reads and writes the same place.
    """
    import shutil

    staged = os.path.join(base_dir, SELECTED_SUBDIR)
    os.makedirs(staged, exist_ok=True)
    for stale in glob.glob(os.path.join(staged, XYZ_GLOB)):
        os.remove(stale)
    target = os.path.join(staged, os.path.basename(path))
    if os.path.abspath(target) != os.path.abspath(path):
        shutil.copyfile(path, target)
    return staged


def _clear_work_dir(work_dir, parser, replace):
    """Make *work_dir* hold nothing left over from an earlier extraction.

    Only the frames just extracted may live in the directory that gets
    clustered, and that has to hold one level down as well: the orchestrator
    explodes the multi-frame file into ``xyz_frames/`` and then globs *that*,
    so 23 frames left there by a previous selection would join a new run of 5
    and be clustered alongside them.

    Deletes precisely what a re-extraction invalidates — the staged frames, the
    exploded copies and the descriptor cache — rather than emptying the
    directory, which may be one the user named and put other things in.
    """
    import shutil

    if not work_dir.is_dir():
        work_dir.mkdir(parents=True, exist_ok=True)
        return

    stale = sorted(work_dir.glob(XYZ_GLOB))
    frames_dir = work_dir / FRAMES_SUBDIR
    caches = sorted(work_dir.glob("data_cache_*.pkl"))

    if (stale or frames_dir.is_dir()) and not replace:
        parser.error(
            f"'{work_dir}' already holds an earlier extraction. Re-run with "
            f"--reprocess-files to replace it, or point --work-dir somewhere else."
        )

    for path in stale + caches:
        path.unlink()
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)


def _output_dir_for(input_source):
    """Where a run's results belong: *beside* the input that produced them.

    Naming an input on the command line means "analyse that", and the answer
    belongs next to it — so a batch fired off from one place,

        for f in serie*/w6_*/w6_*.pdb; do cosmic "$f"; done

    leaves each result in its own directory instead of every run's output
    collapsing into whichever directory the loop was launched from.

    "Beside" is meant literally in both cases, which makes them differ:

    * a **file** puts its results in the directory holding it, next to the file;
    * a **folder** puts them in that folder's parent, next to the folder —
      never inside it, so a directory of inputs stays a directory of inputs and
      does not accumulate motifs and caches among the structures.

    Returns None, meaning "the working directory" as COSMIC has always done,
    whenever that is where the results would land anyway: no input named, an
    input that *is* the working directory (``cosmic .``), or the ordinary
    ``cosmic xyz_dir`` whose parent is simply where you already are.
    """
    if not input_source:
        return None

    here = os.path.abspath(os.getcwd())
    if os.path.isfile(input_source):
        directory = os.path.dirname(os.path.abspath(input_source))
    else:
        folder = os.path.abspath(input_source)
        if folder == here:
            return None
        directory = os.path.dirname(folder)

    return None if directory == here else directory


#: Default name for a trajectory run's output directory, before numbering.
WORK_DIR_BASENAME = "cosmic"


def _next_work_dir(base_dir, basename=WORK_DIR_BASENAME):
    """``cosmic``, then ``cosmic_2``, ``cosmic_3`` … — the first name not taken.

    Every trajectory run gets a directory of its own, so a second run never
    lands on top of the first. That matters more here than for ordinary
    clustering: the pipeline re-globs its input folder, so frames left over
    from an earlier selection would silently join the new ensemble rather than
    being overwritten.

    Created in the working directory rather than beside the trajectory. The old
    name was derived from the trajectory's own (``traj_dt90_shell``) and could
    safely sit next to it; a name as generic as ``cosmic`` belongs where the
    user is working, not scattered through shared data directories.
    """
    base = Path(base_dir)
    candidate = base / basename
    if not candidate.exists():
        return candidate
    suffix = 2
    while (base / f"{basename}_{suffix}").exists():
        suffix += 1
    return base / f"{basename}_{suffix}"


def _parse_name_list(value):
    """Split a comma-separated residue-name argument into a tuple of names."""
    if not value:
        return ()
    return tuple(part for part in (p.strip() for p in value.split(",")) if part)


def run_shell_extraction(args, parser):
    """``--shell`` / ``--nearest``: carve a solute + its solvation shell.

    A solvated MD trajectory is not a clustering input — a box of bisphenol A in
    water is 22077 atoms per frame, and the descriptors are pairwise. This
    reduces every frame to the solute and its nearest solvent, and writes the
    result into a directory holding nothing else.

    That directory is the return value, and ``main`` then points the ordinary
    folder-clustering path at it, so the whole job is one command and there is
    still only one call into
    :func:`~cosmic_ascec.clustering.perform_clustering_and_analysis`. Returns an
    exit status instead when the run should stop after extracting.
    """
    from cosmic_ascec.exceptions import TrajectoryError
    from cosmic_ascec.file_formats.md_trajectory import (
        ShellSpec,
        extract_shell,
        parse_element_overrides,
    )

    if args.shell is not None and args.nearest is not None:
        parser.error("--shell and --nearest are alternatives; pass only one")
    if not args.input_source:
        parser.error("--shell / --nearest need a trajectory, e.g. "
                     "cosmic traj.pdb --solute-resname=LIG --nearest=30")

    trajectory = Path(args.input_source)

    # A .pdb output can only be looked at, not clustered, so asking for one is
    # itself a request to stop after extracting.
    output_is_pdb = bool(args.output) and args.output.lower().endswith(".pdb")
    extract_only = args.extract_only or output_is_pdb

    # A named --work-dir is the user's choice and may already hold a run, so it
    # is checked and cleared. The default picks a fresh numbered name instead,
    # which cannot collide and so needs neither.
    if args.work_dir:
        work_dir = Path(args.work_dir)
        if not extract_only:
            _clear_work_dir(work_dir, parser, replace=args.reprocess_files)
    else:
        # Beside the trajectory, not in the working directory. The point is to
        # be able to fire a batch off from one place —
        #   for f in serie*/w6_*/w6_*.pdb; do cosmic "$f" --nearest=5; done
        # — and have every result land with the input it came from, instead of
        # thirty numbered directories piling up wherever the loop was launched.
        work_dir = _next_work_dir(trajectory.parent)

    output = Path(args.output) if args.output else work_dir / "shell.xyz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not extract_only:
        # Only when it is about to be clustered — an --extract-only run writing
        # elsewhere should not leave an empty directory behind.
        work_dir.mkdir(parents=True, exist_ok=True)

    box = None
    if args.box:
        try:
            box = [float(v) for v in args.box.replace(",", " ").split()]
        except ValueError:
            parser.error(f"--box must be a number or 'Lx,Ly,Lz', not {args.box!r}")

    try:
        overrides = parse_element_overrides(args.elements)
    except TrajectoryError as exc:
        parser.error(str(exc))

    spec = ShellSpec(
        output=output,
        cutoff=args.shell,
        count=args.nearest,
        solute_resnames=_parse_name_list(args.solute_resname),
        solute_indices=args.solute,
        solvent_resnames=_parse_name_list(args.solvent_resname),
        solvent_size=args.solvent_size,
        box=box,
        periodic=not args.no_pbc,
        first=args.first,
        last=args.last,
        stride=args.stride,
        order=args.order,
        element_overrides=overrides,
        verify=args.verify,
        command=" ".join(["cosmic"] + sys.argv[1:]),
    )

    # Flushed so the header cannot appear after an error written to stderr.
    print_step("Extracting solute + solvation shell from trajectory", flush=True)
    try:
        extract_shell(trajectory, spec)
    except TrajectoryError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    if extract_only:
        print(f"\n  Stopping after extraction. Cluster it with:  cosmic {output}")
        return 0
    return str(work_dir)


# Output subfolders a COSMIC stage directory writes its QM jobs into. Used to
# recognise a stage directory when resolving a relative --prev-out-dir against
# the ancestors of the working directory.
_STAGE_OUT_GLOBS = ("orca_out_*", "opt_out_*", "gaussian_out_*", "calc_out_*", "xtb_out_*")


def _looks_like_stage_dir(path):
    """True when *path* holds QM output subfolders, i.e. is a COSMIC stage dir."""
    return any(
        os.path.isdir(hit)
        for pattern in _STAGE_OUT_GLOBS
        for hit in glob.glob(os.path.join(path, pattern))
    )


def resolve_prev_out_dir(path, parser):
    """Resolve ``--prev-out-dir`` to an absolute directory, or abort.

    Two failures used to be silent, because the orchestrator guards the
    composite-energy call with a bare ``os.path.isdir`` and no ``else``:

    * A path that does not exist was ignored, and the run silently ranked by
      the bare eref electronic energy instead of the composite Gibbs energy.
    * A *relative* sibling-stage name is the natural thing to type, but the
      working directory is usually the stage's own output folder
      (``cosmic_3/orca_out_29``), so ``--prev-out-dir cosmic_2`` resolved to a
      non-existent ``cosmic_3/orca_out_29/cosmic_2`` and hit the first case.

    So a relative name is tried against the working directory first, then
    against its ancestors — an ancestor hit must look like a stage directory
    (:func:`_looks_like_stage_dir`) so a common name cannot match some
    unrelated folder further up the tree. Nothing found is a hard error: a run
    that asked for composite energies and got electronic ones is worse than a
    run that stopped.
    """
    expanded = os.path.expanduser(path)

    if os.path.isabs(expanded):
        if os.path.isdir(expanded):
            return os.path.abspath(expanded)
        parser.error(
            f"--prev-out-dir: no such directory: {expanded}"
        )

    candidates = []
    directory = os.getcwd()
    while True:
        candidates.append(os.path.join(directory, expanded))
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent

    for index, candidate in enumerate(candidates):
        if not os.path.isdir(candidate):
            continue
        # The working-directory candidate is what the user literally asked for;
        # ancestors are inferred, so they have to prove they are a stage dir.
        if index > 0 and not _looks_like_stage_dir(candidate):
            continue
        resolved = os.path.abspath(candidate)
        if index > 0:
            print_step(f"--prev-out-dir '{path}' resolved to {resolved}")
        return resolved

    parser.error(
        f"--prev-out-dir: could not find '{path}' in {os.getcwd()} or any "
        f"parent directory. Composite energies need the previous stage's "
        f"COSMIC base directory (the one holding orca_out_*/, umotifs_*/ …); "
        f"pass its full path."
    )

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

  Input is a folder of .out/.log QM outputs, or of .xyz or .mol coordinate
  files. Plain coordinates give the 8 geometry-derived descriptors (nuclear
  repulsion, rotational constants A/B/C, and the four H-bond columns) with no QM
  run at all — the entry point for large systems. '--sp' adds 4 more via one xTB
  single point per structure. A multi-frame .xyz is split per frame. .mol input
  is converted to .xyz with OpenBabel ('obabel', required for it) into
  'mol_as_xyz/'; a folder holding both forms is read as .xyz.

  An MD trajectory (.pdb/.gro/.xyz) can be given directly. Say which part of it
  matters and cosmic reduces every frame to the solute plus its solvation shell
  before clustering, in the same command:

      cosmic traj.pdb --solute-resname=BPA --nearest=30 -j4

  Without that, a solvated box is far too large to cluster — the descriptors
  are pairwise, so tens of thousands of atoms per frame is hopeless while the
  solute and its 30 nearest waters is 123 atoms. See 'MD TRAJECTORY PRE-FILTER'
  under KEY OPTIONS.

HOW IT WORKS:
  1. Check every input has the same composition — same elements, same counts —
     and stop if not, naming the files that differ. Descriptors and RMSD are
     only defined within one system; mixing two would cluster molecules rather
     than conformers. Skipped for --shell / --nearest, where the number of
     solvent molecules varies by design.
  2. Parse QM outputs (.log/.out) or coordinates (.xyz) into the feature vector
  3. Z-standardize each feature (drop near-constant columns)
  4. Build a UPGMA tree (SciPy average linkage, Euclidean distance)
  5. Cut it at the threshold (default 'auto' = knee of the merge-height curve,
     capped at the empirical 2.0 — '--th=knee' lifts that cap; Mojena is
     plotted only as a diagnostic)
  6. Optional RMSD pass to split geometric look-alikes within a family
  7. Flag imaginary frequencies and convergence failures

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
                        Charge and spin come from --charge / --uhf below; both
                        default to a neutral singlet, so an ion or a radical must
                        set them or every single point is solved for the wrong
                        system.
  --charge INT          Total charge for the --sp single points (default 0).
                        Passed to xTB as --chrg; ignored without --sp.
  --uhf INT             Unpaired electrons for the --sp single points (default 0,
                        closed shell). A doublet is 1, a triplet 2. Passed to xTB
                        as --uhf; ignored without --sp.
  -T FLOAT              Temperature (K) for Boltzmann populations (default 298.15).
  --compare FILE...     Direct pairwise comparison of ≥2 files (no folder).
                        Writes only clustering_summary.txt, the single
                        cluster_*.dat and extracted_clusters/ — no
                        dendrogram, Boltzmann report or motif folder.
  --reprocess-files     Ignore the descriptor cache and re-parse outputs.
  FOLDER                Directory of .out/.log QM outputs or .xyz/.mol
                        coordinates — or a single .xyz/.mol file
                        (default: current / interactive).

MD TRAJECTORY PRE-FILTER (--shell / --nearest):
  A solvated trajectory cannot be clustered as it stands — a solute in a box of
  water is tens of thousands of atoms per frame, and the descriptors are
  pairwise. These flags carve out the solute plus the solvent actually touching
  it and then cluster the result, in one command. Input is .pdb or .gro (both
  carry a cell per frame and residue names) or .xyz (needs --box and --solute).

  --nearest N           Keep the N nearest solvent molecules. Constant atom
                        count, so frames differ in geometry only, and --rmsd
                        has the equal-size input it requires. Note that solvent
                        identity still turns over between frames, so an
                        all-atom RMSD across the shell measures that turnover
                        as much as it measures the solute.
  --nearest 0           Drop the solvent entirely — cluster the solute's own
                        conformations across the trajectory. No cell needed,
                        since no distance is measured across one.
  --shell R             Keep every solvent molecule within R Å. Physically
                        honest, but frames vary in size and composition — so
                        the composition check under HOW IT WORKS does not apply
                        to --shell or --nearest runs, including an
                        --extract-only file clustered later (mapping.dat in the
                        work directory is what marks it).
  --solute-resname NAMES  Residue name(s) of the solute: BPA, or LIG,HEM.
  --solvent-resname NAMES Residue name(s) allowed into the shell. Keeps
                        counter-ions out; everything unnamed is ignored.
  --solute 1-33         Solute as atom indices (required for .xyz input).
  --stride N            Keep every Nth frame — thins a long trajectory.
  --verify              Check every frame: molecules intact, correctly re-imaged.
  --extract-only        Write the frames and stop, without clustering.
  --elements MAP        Name the element behind a force-field atom type, e.g.
                        'IN=N', or '@types.map' for a whole force field.
  --work-dir DIR        Where the frames and the clustering output go (default
                        'cosmic' beside the trajectory, then 'cosmic_2',
                        'cosmic_3' … so a re-run never lands on top of an
                        earlier one).
  -o FILE               Name the extracted file. A .pdb name implies
                        --extract-only, since only .xyz can be clustered.

  Selection is molecule-whole and periodic: a molecule counts if any of its
  atoms reaches the solute, it is never cut in half, and it is translated into
  the periodic image beside the solute. Skipping that last step writes a
  neighbour 3 Å away as one 57 Å away and quietly ruins every descriptor. Any
  cell shape is handled, including the rhombic dodecahedron GROMACS defaults to
  for solvated proteins, and the selection uses a KD-tree so a protein-sized
  solute costs no more than a small molecule.

  Atom names in an MD file are force-field types, not element symbols. COSMIC
  reads the PDB element columns when they are filled in, drops virtual sites
  (TIP4P's MW, lone pairs, Drude particles) rather than choking on them, and
  prints the name->element table it resolved so a wrong guess is visible at
  once. Genuinely ambiguous types — IN is indium and a nitrogen type — are
  refused rather than guessed; name them with --elements.

  TRACEABILITY: every trajectory run writes mapping.dat into the work
  directory, joining each clustered structure to the trajectory frame it came
  from, its simulation time and step, which solvent molecules were in its
  shell, and the cluster and motif it ended up in. The same frame and time are
  appended to the comment lines of shell.xyz, the motif files,
  extracted_clusters/ and clustering_summary.txt. Runs that are not from a
  trajectory have no mapping.dat and are not annotated at all.

WHERE RESULTS GO:
  Naming an input puts its results beside it, so a batch launched from one
  place leaves each answer with the data it came from rather than piling
  everything into the directory the loop ran in:
    cosmic runs/a/traj.pdb --nearest=5   → runs/a/cosmic/ (then cosmic_2, ...)
    cosmic runs/a/w6.xyz                 → runs/a/
    cosmic runs/a                        → runs/  (beside the folder, never
                                            inside it — a folder of inputs
                                            stays a folder of inputs)
  With no input, or with '.', results land in the working directory as always.
  --output-dir overrides all of this.

MAIN OUTPUTS:
  mapping.dat              Trajectory runs only: frame → time, step, solvent,
                           cluster and motif (see --shell / --nearest below)
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
  cosmic xyz_dir --sp --charge -1  Single points on an anion (neutral singlet
                                   is the default, so charged systems must say so)
  cosmic xyz_dir --sp --uhf 2      Single points on a triplet (2 unpaired electrons)
  cosmic ensemble.xyz -j4          Cluster a multi-frame file (split per frame)
  cosmic --compare a.out b.out     Compare two structures directly

MD TRAJECTORY EXAMPLES (extract + cluster in one command):
  cosmic traj.pdb --solute-resname=BPA --nearest=30 -j4
                                   Solute + its 30 nearest waters, every frame
                                   the same size, clustered. Everything lands
                                   in cosmic/ (cosmic_2/ next time)
  cosmic traj.pdb --solute-resname=BPA --nearest=0 -j4
                                   Solute only, solvent discarded — pure
                                   conformational clustering of the molecule
  cosmic traj.pdb --solute-resname=BPA --shell=5.0 -j4
                                   Everything within 5 Å instead (variable size)
  cosmic traj.pdb --solute-resname=LIG,HEM --solvent-resname=SOL --nearest=40 -j4
                                   Two-residue solute, ions excluded from the
                                   shell by naming the solvent
  cosmic traj.pdb --solute-resname=BPA --nearest=30 --stride=5 -j4
                                   Same, keeping every 5th frame
  cosmic traj.pdb --solute-resname=BPA --nearest=30 --verify --extract-only
                                   Extract and self-check, cluster later
  cosmic traj.pdb --solute-resname=BPA --nearest=30 -o look.pdb
                                   Write a PDB to open in VMD (extract only)
  cosmic traj.gro --solute-resname=BPA --nearest=30 -j4
                                   GROMACS .gro, triclinic cells included
  cosmic traj.xyz --solute=1-33 --box=60.729 --nearest=30 -j4
                                   XYZ input: no residues and no box, so both
                                   have to be given explicitly

TYPICAL PIPELINE (or use the automated protocol in one .asc file):
  ascec input.asc r5 --concurrent=5   → 5 replicate annealing runs
  ascec opt template.inp launcher.sh  → build + run optimization inputs
  ascec sort                          → collect and rank
  cosmic -j4                          → unique motifs

FROM AN MD TRAJECTORY (one command, nothing to prepare):
  gmx trjconv -f run.xtc -o traj.pdb -pbc mol    → text trajectory, solute whole
  cosmic traj.pdb --solute-resname=BPA --nearest=30 -j4
                                                 → shell extracted, then motifs

SUPPORTED FORMATS:
  Gaussian .log (cclib) · ORCA 5.0.x .out (cclib) · ORCA 6.1+ .out (OPI)
  ORCA 6.0 is not supported — use 5.0.x or 6.1+.
  Plain .xyz coordinates, single- or multi-frame — no QM output needed. Gives the
  8 geometry-derived descriptors; add --sp for the 4 electronic ones. Precedence
  in a mixed folder is .out > .log > .xyz. An xTB input superseded by its own
  .xtbopt.xyz result is dropped, so a structure is never clustered twice.
  MD trajectories: .pdb and .gro (both carry a cell per frame and residue names,
  any cell shape), or .xyz with --box and --solute. Needs --shell or --nearest to
  say what to keep. Convert a binary trajectory first: gmx trjconv -pbc mol.

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
                        help="previous stage COSMIC base directory for composite energy: "
                             "G = E_eref + (G_prev - E_prev). A relative name is looked up "
                             "in the working directory and its parents, so a sibling stage "
                             "(e.g. cosmic_2) works from inside cosmic_3/orca_out_29")
    parser.add_argument("--level", type=str, default=None, metavar="LEVEL",
                        choices=[lv.key for lv in _levels.LEVELS],
                        help="name this pass's representatives explicitly: "
                             "candidate (after geometry optimization), motif "
                             "(after geometry refinement) or u_motif (after "
                             "energy refinement). Omitted, the level is guessed "
                             "from the input filenames.")

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

    # --- MD trajectory pre-filter -----------------------------------------
    # A solvated trajectory cannot be clustered as it stands: hundreds of bulk
    # solvent molecules swamp the pairwise descriptors and say nothing about
    # the solute's conformation. Passing --shell or --nearest turns cosmic into
    # a one-shot extractor that writes a small multi-frame XYZ and exits; that
    # file is then clustered by an ordinary second cosmic call.
    md = parser.add_argument_group(
        "MD trajectory pre-filter",
        "Carve a solute plus its solvation shell out of a solvated trajectory "
        "(.pdb, .gro or .xyz), then cluster it — one command, one call."
    )
    md.add_argument("--shell", type=float, default=None, metavar="R",
                    help="keep every solvent molecule coming within R Å of the solute. "
                         "Physically honest, but the count varies per frame, so frames "
                         "differ in composition as well as geometry (and --rmsd cannot "
                         "compare them).")
    md.add_argument("--nearest", type=int, default=None, metavar="N",
                    help="keep the N nearest solvent molecules. Every frame gets the same "
                         "formula and atom count, which is what makes frames comparable "
                         "and is required for --rmsd. Recommended for clustering. "
                         "--nearest=0 drops the solvent entirely and clusters the solute's "
                         "own conformations.")
    md.add_argument("-o", "--output", type=str, default=None, metavar="FILE",
                    help="where to write the extracted frames (default: shell.xyz inside "
                         "the work directory). Give a .pdb name to inspect the result in "
                         "VMD instead — that stops after extraction, since only .xyz can "
                         "be clustered.")
    md.add_argument("--extract-only", action="store_true",
                    help="write the extracted frames and stop, without clustering them. "
                         "Use it to check a selection before committing to a long run.")
    md.add_argument("--work-dir", type=str, default=None, metavar="DIR",
                    help="directory for the extracted frames and the clustering output "
                         "(default: 'cosmic' beside the trajectory, then 'cosmic_2', "
                         "'cosmic_3' and so on, so a re-run never overwrites an earlier "
                         "one). It holds exactly the extracted frames, which is what "
                         "keeps the clustering from picking up unrelated .xyz files.")
    md.add_argument("--solute-resname", type=str, default=None, metavar="NAMES",
                    help="residue name(s) of the solute, comma-separated: BPA, or "
                         "LIG,HEM for a multi-residue solute. Default: the residue of "
                         "atom 1. .pdb / .gro input only.")
    md.add_argument("--solvent-resname", type=str, default=None, metavar="NAMES",
                    help="residue name(s) eligible as shell material, comma-separated. "
                         "Anything neither solute nor solvent is ignored entirely — this "
                         "is what keeps counter-ions out of the shell. Default: "
                         "everything that is not solute.")
    md.add_argument("--solute", type=str, default=None, metavar="SPEC",
                    help="solute as explicit 1-based atom indices, e.g. 1-33 or 1-33,58. "
                         "Required for .xyz input, which carries no residue names.")
    md.add_argument("--solvent-size", type=int, default=3, metavar="N",
                    help="atoms per solvent molecule for .xyz input (default 3, water). "
                         ".pdb / .gro input read this from the residue numbering instead.")
    # A single string rather than nargs='+': with nargs the FOLDER positional
    # gets swallowed by `cosmic --box 60.7 traj.xyz`.
    md.add_argument("--box", type=str, default=None, metavar="L",
                    help="periodic box: one cubic edge, or three as 'Lx,Ly,Lz', in Å. "
                         "Read per frame from CRYST1 (.pdb) or the box line (.gro), so "
                         "both an NPT cell and a non-orthogonal one are handled; "
                         "required only for .xyz input.")
    md.add_argument("--no-pbc", action="store_true",
                    help="treat the system as non-periodic (no minimum-image distances, "
                         "no re-imaging of the selected molecules).")
    md.add_argument("--first", type=int, default=0, metavar="I",
                    help="first trajectory frame to read, 0-based (default 0).")
    md.add_argument("--last", type=int, default=None, metavar="I",
                    help="last trajectory frame to read, inclusive (default: to the end).")
    md.add_argument("--stride", type=int, default=1, metavar="N",
                    help="keep only every Nth frame (default 1). The cheapest way to "
                         "thin a long trajectory down to a workable ensemble.")
    md.add_argument("--order", choices=("distance", "index"), default="distance",
                    help="write solvent molecules ordered by distance to the solute "
                         "(default) or by their original atom index.")
    md.add_argument("--elements", type=str, default="", metavar="MAP",
                    help="name the element behind an atom name, e.g. 'OS=O,IN=N'. Use "
                         "'@types.map' to read a whole force field from a file, one "
                         "'NAME SYMBOL' pair per line. Needed when a force-field atom "
                         "type is ambiguous — 'IN' is indium and a nitrogen type, and "
                         "COSMIC asks rather than guessing.")
    md.add_argument("--verify", action="store_true",
                    help="after building each frame, check that every kept molecule is "
                         "geometrically intact and landed in the periodic image beside "
                         "the solute. Reports the worst deviation found.")

    # Hidden/advanced options
    parser.add_argument("--min-std-threshold", type=float, default=1e-6,
                        help=argparse.SUPPRESS)
    parser.add_argument("--abs-tolerance", type=str, default="",
                        help=argparse.SUPPRESS)
    parser.add_argument("--update-cache", type=str, default=None,
                        help=argparse.SUPPRESS)

    # Positional argument
    parser.add_argument('input_source', nargs='?', default=None, metavar="FOLDER|TRAJECTORY",
                        help='directory containing .out/.log QM outputs or .xyz/.mol '
                             'coordinate files (a single .xyz or .mol file is also '
                             'accepted), or an MD trajectory (.pdb/.gro/.xyz) together '
                             'with --shell/--nearest saying which part of it to cluster')


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

    # Pin --prev-out-dir to an absolute directory here, once, so every mode
    # below (--compare and the folder scan alike) gets the same resolution and
    # a missing previous stage stops the run instead of quietly degrading it to
    # bare electronic energies.
    if args.prev_out_dir:
        args.prev_out_dir = resolve_prev_out_dir(args.prev_out_dir, parser)

    # --shell / --nearest: pre-filter an MD trajectory, then carry on and
    # cluster what came out. Extraction happens here, before any clustering
    # setup, and simply re-points input_source at the directory it filled — so
    # every clustering flag below applies unchanged and there is still only one
    # call into the orchestrator.
    #
    # A trajectory shell is also the one input whose composition legitimately
    # varies from frame to frame, so it is exempt from the composition check the
    # orchestrator runs. That exemption needs both halves: the orchestrator also
    # detects it from mapping.dat, which covers an --extract-only file clustered
    # later, and this flag covers `-od elsewhere`, where mapping.dat is not in
    # the directory the orchestrator looks in.
    from_md_shell = False
    if args.shell is not None or args.nearest is not None:
        outcome = run_shell_extraction(args, parser)
        if isinstance(outcome, int):
            return outcome
        args.input_source = outcome
        from_md_shell = True
        # Keep the clustering output beside the frames it came from rather than
        # scattering it through the user's working directory.
        if args.output_dir is None:
            args.output_dir = outcome

    # A trajectory that arrived without a selection would otherwise fall
    # through to the folder scan and be reported as "no structures found".
    if args.input_source and str(args.input_source).lower().endswith((".pdb", ".gro")):
        parser.error(
            f"'{args.input_source}' is an MD trajectory, not a structure folder. "
            f"Say which part of it to cluster, e.g.\n"
            f"    cosmic {args.input_source} --solute-resname=<RES> --nearest=30\n"
            f"which extracts the solute plus its 30 nearest solvent molecules and "
            f"clusters them in one go."
        )

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

    output_directory = args.output_dir or _output_dir_for(args.input_source)
    force_reprocess_cache = args.reprocess_files
    user_weights_dict = parse_weights_argument(args.weights)
    # Pick tuned semiempirical baseline only when --partialweights is passed;
    # otherwise stay method-agnostic with a flat 1.0 baseline.
    base_weights = SEMIEMPIRICAL_WEIGHTS if args.partialweights else DEFAULT_WEIGHTS
    weights_dict = dict(base_weights)
    weights_dict.update(user_weights_dict)  # user --weights override the baseline
    min_std_threshold_val = args.min_std_threshold
    # --abs-tolerance overrides layer on top of the defaults, the same way
    # --weights layers over its baseline above. Naming one feature used to
    # replace the whole table, which silently switched the tolerance gate off
    # for the other fourteen and left their component-difference scores with no
    # reference to measure against.
    abs_tolerances_dict = dict(DEFAULT_ABS_TOLERANCES)
    abs_tolerances_dict.update(parse_abs_tolerance_argument(args.abs_tolerance))
    num_cores = args.cores if args.cores is not None else get_cpu_count_fast()
    temperature_k = args.temperature

    # Update the global verbose flag
    console.VERBOSE = args.verbose

    current_dir = os.getcwd()


    if args.compare:
        # A stage run keeps each structure in its own folder
        # (energy_refinement/umotif_12/umotif_12_opt.out), so naming two of them
        # meant spelling the stem out twice per path. Accept the folder and
        # resolve the output inside it, so `--compare a_dir b_dir` works as well
        # as `--compare a.out b.out`. Leading/trailing whitespace is stripped:
        # a pasted backslash continuation collapses to " path", which used to
        # fail as a not-found file for no visible reason.
        compare_files = []
        for entry in (e.strip() for e in args.compare):
            if os.path.isdir(entry):
                found = sorted(
                    f for ext in ('.out', '.log', XYZ_EXTENSION)
                    for f in glob.glob(os.path.join(entry, '*' + ext))
                )
                # .mol only when the folder offers nothing better. Adding it
                # unconditionally would list a structure twice in the folder
                # COSMIC itself wrote both forms into.
                if not found:
                    found = find_mol_files(entry)
                if not found:
                    print(f"Error: No .out, .log, .xyz or .mol file in directory: {entry}")
                    return 1
                compare_files.extend(found)
            elif os.path.exists(entry):
                compare_files.append(entry)
            else:
                print(f"Error: File not found: {entry}")
                return 1

        if len(compare_files) < 2:
            print("Error: --compare requires at least 2 files.")
            return 1

        # Normalise .mol to .xyz before anything inspects extensions, so the
        # compatibility check below and the parser dispatch downstream both see
        # a single coordinate format rather than a mixture that only looks like
        # one. Named .xyz and .out files are passed through untouched.
        mol_named = [f for f in compare_files if f.lower().endswith(MOL_EXTENSION)]
        if mol_named:
            try:
                staged = stage_mol_as_xyz(mol_named, output_directory or current_dir,
                                          label="compare")
            except ClusteringError as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)
                return 1
            converted = {f: os.path.join(staged, os.path.splitext(os.path.basename(f))[0] + XYZ_EXTENSION)
                         for f in mol_named}
            compare_files = [converted.get(f, f) for f in compare_files]

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
            print("Error: Provided files do not have .log, .out, .xyz or .mol extensions.")
            return 1

        file_names = [os.path.basename(f) for f in compare_files]
        print(f"\n--- Comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")
        try:
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
                # Composite Gibbs needs the previous stage here too: comparing
                # two energy-refinement outputs is exactly the case where the
                # ranking energy is E_eref + (G_prev - E_prev), and dropping
                # this argument silently fell back to the bare electronic
                # energy with no warning.
                prev_out_dir=args.prev_out_dir,
                partialweights=args.partialweights,
                rmsd_only=rmsd_only_mode,
                rmsd_heavy=args.rmsd_heavy,
                sp_method=sp_method if file_extension_pattern_for_compare == XYZ_EXTENSION else None,
                sp_charge=args.charge,
                sp_uhf=args.uhf,
                allow_mixed_stoichiometry=from_md_shell,
            )
        except ClusteringError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
        print(f"\n--- Finished comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")

    else: # Normal mode (folder processing)
        if args.input_source:
            # Non-interactive mode. A bare .xyz file is accepted as well as a
            # folder — a single ensemble or trajectory file is a natural way to
            # hand over a large set of structures, and it is split per frame
            # downstream.
            if os.path.isfile(args.input_source):
                lowered = args.input_source.lower()
                if not lowered.endswith((XYZ_EXTENSION, MOL_EXTENSION)):
                    print(f"Error: '{args.input_source}' is a file; only .xyz and .mol "
                          f"files can be given directly. Pass the containing folder instead.")
                    return 1
                # Name one file and that is the file you get. Handing over its
                # directory instead would sweep in every other .xyz beside it,
                # because the pipeline re-globs its input folder several times
                # downstream — so the only way to make a one-file selection
                # stick is to give it a folder holding exactly that file. A
                # named .mol reaches the same place through obabel: the
                # conversion writes into a directory of its own, which is
                # already the one-file folder the staging step would have made.
                if lowered.endswith(MOL_EXTENSION):
                    try:
                        selected_folders = [stage_mol_as_xyz(
                            [args.input_source], args.output_dir or current_dir)]
                    except ClusteringError as exc:
                        print(f"\nERROR: {exc}", file=sys.stderr)
                        return 1
                else:
                    selected_folders = [_stage_single_xyz(args.input_source, args.output_dir or current_dir)]
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
                    print(f"Error: No .out, .log, .xyz or .mol files found in '{args.input_source}'.")
                    return 1

        else:
            # Interactive mode
            all_potential_folders = [current_dir] + [
                d for d in glob.glob(os.path.join(current_dir, '*'))
                if os.path.isdir(d) and os.path.basename(d) not in _STAGING_SUBDIRS
            ]

            folder_inputs = {}
            for folder in all_potential_folders:
                available = _available_inputs(folder)
                if available:
                    folder_inputs[folder] = available

            all_valid_folders_to_display = sorted(folder_inputs)

            if not all_valid_folders_to_display:
                print("No subdirectories containing .out, .log, .xyz or .mol files found, or files are organized directly in the current directory.")
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
                print("\nNo .out, .log, .xyz or .mol files found in the selected folder(s) that match available types. Exiting.")
                return 0

        # Single points only make sense for coordinate input: a QM output
        # already carries its own energy, and recomputing it at a different
        # level would mix two methods in one feature column.
        if sp_method and file_extension_pattern not in (XYZ_GLOB, MOL_GLOB):
            print(f"WARNING: --sp applies to coordinate input only; ignoring it "
                  f"for '{file_extension_pattern}' files.")
            sp_method = None

        print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_pattern}'...")
        for folder_path in selected_folders:
            display_name = os.path.basename(folder_path)
            if folder_path == current_dir:
                display_name = "./"
            print(f"\nProcessing folder: {display_name}\n")

            # .mol input is normalised to .xyz here and nowhere else: from this
            # point the pipeline is handed a folder of coordinates in the format
            # it already parses, caches and copies by filename. The converted
            # set lives under the output directory, not beside the .mol files,
            # so the user's input folder is left exactly as it was found.
            cluster_folder, cluster_pattern = folder_path, file_extension_pattern
            if file_extension_pattern == MOL_GLOB:
                try:
                    cluster_folder = stage_mol_as_xyz(
                        find_mol_files(folder_path), output_directory or current_dir,
                        label="" if len(selected_folders) == 1 else (display_name.strip("./") or "cwd"),
                    )
                except ClusteringError as exc:
                    print(f"\nERROR: {exc}", file=sys.stderr)
                    return 1
                cluster_pattern = XYZ_GLOB

            try:
                perform_clustering_and_analysis(cluster_folder, clustering_threshold, cluster_pattern, rmsd_validation_threshold, output_directory, force_reprocess_cache, weights_dict, is_compare_mode=False, min_std_threshold=min_std_threshold_val, abs_tolerances=abs_tolerances_dict, num_cores=num_cores, temperature_k=temperature_k, group_hb=args.group_hb, prev_out_dir=args.prev_out_dir, partialweights=args.partialweights, rmsd_only=rmsd_only_mode, rmsd_heavy=args.rmsd_heavy, sp_method=sp_method, sp_charge=args.charge, sp_uhf=args.uhf, allow_mixed_stoichiometry=from_md_shell, level=args.level)
            except ClusteringError as exc:
                # Stop the run rather than skipping the folder: the exit status
                # has to mean "this did not do what you asked", and a batch that
                # kept going would print "all analyses complete" over the top of
                # a failure and leave nothing but the code to tell them apart.
                print(f"\nERROR: {exc}", file=sys.stderr)
                return 1

            print(f"\nFinished processing folder: {display_name}\n")

    print()
    print_step("All selected molecular analyses complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
