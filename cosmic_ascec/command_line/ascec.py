"""``ascec`` console entry point.

The top-level ``ascec`` command does a lot. It is the front door for:

* **Single-run annealing** — ``ascec <file.asc>`` (no protocol embedded in
  the file).
* **Protocol workflows** — ``ascec <file.asc> protocol [stage] [-i]`` or any
  ``.asc`` that carries an embedded ``.asc, r3, opt, cosmic`` line.
* **Subcommands** — ``ascec box``, ``ascec opt``, ``ascec ref``,
  ``ascec sort``, ``ascec merge``, ``ascec update``, ``ascec cleanup``,
  ``ascec launcher``, ``ascec diagram``, ``ascec cosmic ...`` (passthrough).
* **Auxiliary modes** — ``ascec status`` (job registry / progress viewer),
  ``ascec analyze_box``, ``ascec test_box``, ``ascec input`` (WebGUI input
  generator). These are routed from the root-shim before this module is
  imported.
* **Per-file modes** — ``ascec <file> exclude ...``, ``ascec <file> box``,
  ``ascec <file> rN [--boxP]``.

The dispatch ordering (``--maxprint`` strip → ``--version`` → ``status`` →
``<file> exclude`` → ``<file> protocol`` → embedded-protocol auto-detect →
comma/``then`` workflow → single-command parser → bare-input single-run) is
load-bearing: arguments resolved earlier in the chain are stripped before
the parser sees them.

Single-run annealing routes through
:func:`cosmic_ascec.annealing.anneal`; protocol workflows route through
:func:`cosmic_ascec.workflow.stages.execute_workflow_stages`; the cosmic
passthrough shells to ``cosmic.py``.

Each annealing run draws one 6-digit seed, echoes ``Seed = NNNNNN``, and
uses it both as the output-file suffix and as the seed for the run's single
``numpy.random.RandomState``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import re
import shlex
import shutil
import sys
from pathlib import Path

from cosmic_ascec.annealing import (
    AnnealingCallback,
    TemperatureStart,
    anneal,
)
from cosmic_ascec.elements import Z_TO_SYMBOL
from cosmic_ascec.exceptions import CosmicAscecError
from cosmic_ascec.file_formats import (
    QMProgram,
    SimulationMode,
    SummaryWriter,
    TrajectoryWriter,
    TvseWriter,
    parse_asc,
)
from cosmic_ascec.geometry.molecule import Molecule
from cosmic_ascec.geometry.placement import initialize_cluster
from cosmic_ascec.monte_carlo import (
    MoveParams,
    initialize_rotatable_bond_cache,
    trial,
    zero_energy,
)
from cosmic_ascec.logging_setup import ROOT_LOGGER_NAME, configure_logging
from cosmic_ascec.quantum_chemistry import get_adapter, list_adapters
from cosmic_ascec.random_numbers import make_rng, resolve_run_seed
from cosmic_ascec.workflow import stages as _stages_module
from cosmic_ascec.workflow.protocol import (
    contains_workflow_separator,
    parse_workflow_stages,
)
from cosmic_ascec.workflow.protocol_cache import (
    load_protocol_cache,
    save_protocol_cache,
)
from cosmic_ascec.workflow.stages import (
    ASCEC_VERSION,
    consume_protocol_maxprint_flag,
    create_refinement_system,
    create_simple_optimization_system,
    execute_box_analysis,
    execute_cosmic_analysis,
    execute_diagram_generation,
    execute_merge_command,
    execute_merge_result_command,
    execute_sort_command,
    execute_summary_only,
    execute_workflow_stages,
    extract_protocol_from_input,
    get_box_size_recommendation,
    parse_exclusion_pattern,
    print_version_banner,
    show_ascec_status,
    update_existing_input_files,
)
from cosmic_ascec.workflow.replicas import (
    create_replicated_runs,
    merge_launcher_scripts,
)


# v04 line 9 → adapter name. Used by the single-run wiring below.
_PROGRAM_TO_ADAPTER = {
    QMProgram.GAUSSIAN: "gaussian",
    QMProgram.ORCA: "orca",
    QMProgram.XTB: "xtb",
}

class _AscecStepWriter(AnnealingCallback):
    """Write the current temperature step into ``run_dir/.ascec_step``.

    The parent replication aggregator (``execute_replication_stage`` in
    ``workflow/stages.py``) reads this file every 2 s and aggregates
    progress across concurrent replicas — the format and semantics are
    fixed by that reader, so this writer matches v04's inline write at
    ``ascec-v04.py`` line 20640 byte-for-byte: a single line of the form
    ``"{step}/{total}\\n"`` (1-based step). Writes are best-effort;
    OS errors are swallowed so a transient I/O issue cannot crash the
    annealing run.
    """

    def __init__(self, run_dir: Path) -> None:
        self._step_file = Path(run_dir) / ".ascec_step"

    def on_temperature_start(self, event: TemperatureStart) -> None:
        try:
            with open(self._step_file, "w") as f:
                f.write(f"{event.step_index}/{event.total_steps}\n")
        except OSError:
            pass


def _resolve_adapter_name(alias: str, program: QMProgram) -> str:
    registered = {name.lower() for name in list_adapters()}
    if alias and alias.lower() in registered:
        return alias.lower()
    return _PROGRAM_TO_ADAPTER.get(program, alias.lower() if alias else "")


def _run_random_configurations(
    config,
    input_path: Path,
    run_dir: Path,
    run_seed: int,
    rng,
    logger: logging.Logger,
) -> int:
    """Mode-0 path: same MC code as annealing, minus QM and the schedule.

    Mode 0 reuses the annealing trial code exactly:
    :func:`~cosmic_ascec.monte_carlo.trial.trial` with
    :data:`~cosmic_ascec.monte_carlo.trial.zero_energy` as the energy
    function. With zero energies the Metropolis criterion always returns
    ``LwE`` (downhill accept), so every proposed geometry becomes the
    next current geometry and the trajectory chains exactly like
    annealing would — only without spending QM calls and without a
    temperature schedule wrapping the loop.

    One ``initialize_cluster`` seeds the trajectory (the big-bang
    scatter), then ``n_configs`` ``trial(...)`` calls produce the output
    frames.

    Two trajectories are emitted: ``mto_<seed>.xyz`` (atoms only) and
    ``mtobox_<seed>.xyz`` (atoms + eight ``X`` markers at the cube
    corners), mirroring the annealing pair ``result_*`` / ``resultbox_*``.
    """
    from dataclasses import replace as _replace

    n_configs = int(config.num_configurations)
    if n_configs <= 0:
        print(
            f"ascec: random mode requires num_configurations > 0, got {n_configs}",
            file=sys.stderr,
        )
        return 2

    molecules = [Molecule.from_spec(spec) for spec in config.molecules]
    box_length = float(config.box.cube_length_angstrom)
    xyz_path = run_dir / f"mto_{run_seed}.xyz"
    box_path = run_dir / f"mtobox_{run_seed}.xyz"

    # Move knobs come straight from line 7+8 of the .asc — exactly the same
    # MoveParams the annealing path builds. Mode 0 must run the same move
    # code as annealing, only without the QM cost and the temperature schedule.
    move_params = MoveParams.from_config(config)

    # One big-bang scatter: doubles as the rotatable-bond cache seed and as
    # the starting state for the chained trajectory below. Atom indices are
    # stable across the templates, so the cache stays valid for every
    # subsequent trial() call.
    seed_placement = initialize_cluster(
        molecules,
        box_length=box_length,
        rng=rng,
        logger=logger,
    )
    cache, effective_prob = initialize_rotatable_bond_cache(
        seed_placement.cluster,
        conformational_move_prob=move_params.conformational_move_prob,
        max_dihedral_angle_rad=move_params.max_dihedral_angle_rad,
        logger=logger,
    )
    if effective_prob != move_params.conformational_move_prob:
        move_params = _replace(move_params, conformational_move_prob=effective_prob)

    print(
        f"Generating {n_configs} random configurations on {input_path.name} "
        f"(no energy evaluation)..."
    )
    if effective_prob > 0.0:
        import math as _math
        max_deg = _math.degrees(move_params.max_dihedral_angle_rad)
        print(
            f"  Conformational sampling: {effective_prob * 100:.1f}% probability, "
            f"max dihedral ±{max_deg:.1f}°"
        )
    else:
        print("  Conformational sampling: disabled (no rotatable bonds or max dihedral = 0)")

    # Eight cube-corner markers for the box file (v04 dummy-atom convention).
    half_box = box_length / 2.0
    corners = [
        (sx * half_box, sy * half_box, sz * half_box)
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    ]

    move_counts = {"conformational": 0, "translate_rotate": 0}

    # Mode 0 = mode 1 minus the QM call and the temperature schedule. Every
    # frame is one ``trial(...)`` call against the previous frame's cluster
    # with ``energy_fn=zero_energy``: delta is identically 0 so the Metropolis
    # criterion always returns ``LwE`` (downhill accept) and the proposed
    # geometry becomes the next current geometry. Same propose + accept code
    # path as annealing — the only thing that changes between modes is the
    # energy function and whether a temperature schedule wraps the loop.
    current_cluster = seed_placement.cluster
    current_energy = 0.0
    dummy_temperature = 1.0  # any T > 0; delta == 0 makes the value irrelevant

    xyz_path.write_text("")
    box_path.write_text("")
    with xyz_path.open("a") as fh, box_path.open("a") as fhb:
        for i in range(1, n_configs + 1):
            result = trial(
                current_cluster,
                current_energy=current_energy,
                energy_fn=zero_energy,
                rng=rng,
                temperature_kelvin=dummy_temperature,
                params=move_params,
                cache=cache,
            )
            current_cluster = result.cluster
            cluster = current_cluster
            move_counts[result.move_type] = move_counts.get(result.move_type, 0) + 1

            atom_lines = [
                f"{Z_TO_SYMBOL.get(int(z), 'X'):<3}{x: 13.6f}{y: 13.6f}{zc: 13.6f}"
                for z, (x, y, zc) in zip(cluster.atomic_numbers, cluster.coords)
            ]
            header = (
                f"Configuration: {i} | random placement (mode 0) | "
                f"last move: {result.move_type}"
            )
            fh.write(f"{cluster.num_atoms}\n{header}\n")
            fh.write("\n".join(atom_lines) + "\n")

            corner_lines = [
                f"X  {x: 13.6f}{y: 13.6f}{z: 13.6f}" for x, y, z in corners
            ]
            box_header = (
                f"Configuration: {i} | random placement (mode 0) | "
                f"BoxL={box_length:.1f} A (X box markers)"
            )
            fhb.write(f"{cluster.num_atoms + len(corner_lines)}\n{box_header}\n")
            fhb.write("\n".join(atom_lines + corner_lines) + "\n")

    # Mirror the annealing path: convert each multi-frame xyz to mol via obabel
    # so downstream viewers (and the COSMIC stage) have the .mol siblings.
    # The box file holds non-standard 'X' markers — obabel emits those as '*'
    # in the MOL atom block, so we patch them back via the trajectory_writer
    # helper that TrajectoryWriter uses for resultbox_*.mol. Silent no-op when
    # obabel isn't on PATH.
    import shutil as _shutil
    import subprocess as _subprocess
    from cosmic_ascec.file_formats.trajectory_writer import _restore_dummy_atom_symbol
    if _shutil.which("obabel"):
        for path in (xyz_path, box_path):
            if not path.exists() or path.stat().st_size == 0:
                continue
            mol_path = path.with_suffix(".mol")
            try:
                _subprocess.run(
                    ["obabel", "-ixyz", str(path), "-omol", "-O", str(mol_path)],
                    capture_output=True, check=False, timeout=60,
                )
            except (OSError, _subprocess.SubprocessError):
                continue
            if path is box_path and mol_path.exists():
                _restore_dummy_atom_symbol(mol_path)

    print(
        f"Done. {n_configs} configurations written to "
        f"{xyz_path.name} (+ {box_path.name})."
    )
    print(
        f"  Move breakdown: {move_counts['conformational']} conformational, "
        f"{move_counts['translate_rotate']} rigid-body."
    )
    print(f"Output written to {run_dir}/")
    return 0


def _run_single_simulation(input_file: str, args: argparse.Namespace,
                            unknown_args, logger: logging.Logger) -> int:
    """Bare-input-file single annealing run (v04 main_ascec_integrated 20257-20982).

    Wired (per the R7 prompt) to v05's R3 live annealing engine. v04's
    inline body opens the ``.out`` file, runs the initial-QM retry loop and
    the temperature-schedule MC loop, and emits ``result_<seed>.xyz`` /
    ``rless_<seed>.out`` / ``tvse_<seed>.dat``; v05's
    :func:`cosmic_ascec.annealing.anneal` runs the same loop with the same
    semantics, fed by the existing :class:`SummaryWriter` / ``TvseWriter`` /
    ``TrajectoryWriter`` callbacks. The seed flow matches v04 line 20360
    exactly via :func:`cosmic_ascec.random_numbers.resolve_run_seed`.
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"\nCRITICAL ERROR: Input file '{input_file}' not found.",
              file=sys.stderr)
        sys.exit(1)
    try:
        config = parse_asc(input_path)
    except CosmicAscecError as exc:
        print(f"ascec: failed to parse {input_path}: {exc}", file=sys.stderr)
        return 2

    # v04 lines 20269-20273: outputs land next to the input file, not in the
    # caller's cwd. The launcher script runs replicas via relative paths from
    # the parent dir, so without this every replica would spill its result_*,
    # rless_*, tvse_*, anneal.* into the parent.
    run_dir = input_path.resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # --box<percent> override for single-run mode (v04 lines 20325-20347).
    # v05 reports the recommendation but does not yet rewrite ``config.box`` —
    # the AscConfig dataclass is frozen and a re-port of v04's
    # ``state.cube_length`` / ``state.xbox`` mutation is **carry-over for R8**.
    box_candidates = list(unknown_args or []) + list(sys.argv[1:])
    for candidate in box_candidates:
        if not isinstance(candidate, str) or not candidate.lower().startswith("--box"):
            continue
        packing_str = candidate.lower().replace("--box", "")
        if not packing_str:
            continue
        try:
            packing_percent = float(packing_str)
        except ValueError:
            continue
        recommended = get_box_size_recommendation(input_file, packing_percent)
        if recommended is not None:
            print(
                f"Using recommended box size: {recommended:.1f} Å "
                f"({packing_percent}% effective packing)"
            )
        break

    run_seed = resolve_run_seed()
    rng = make_rng(run_seed)
    print(f"Seed = {run_seed}")

    # v04 lines 20412-20447 — mode 0 generates N random placements and exits
    # without any QM call, MC loop, or annealing. The port never branched on
    # mode, so mode-0 inputs silently ran annealing instead.
    if config.mode == SimulationMode.RANDOM:
        return _run_random_configurations(
            config, input_path, run_dir, run_seed, rng, logger
        )

    adapter_name = _resolve_adapter_name(config.qm.alias, config.qm.program)
    try:
        adapter = get_adapter(adapter_name)
    except CosmicAscecError as exc:
        print(f"ascec: {exc}", file=sys.stderr)
        return 2
    energy_fn = adapter.energy_function(config.qm, run_dir, logger=logger)

    callbacks: list[AnnealingCallback] = [
        SummaryWriter(run_dir, run_seed, input_path.stem),
        TvseWriter(run_dir, run_seed),
        TrajectoryWriter(run_dir, run_seed),
        # Progress writer: drops "<step>/<total>" into run_dir/.ascec_step
        # every temperature so the replication aggregator (the parent process
        # spawned by ``execute_replication_stage``) can poll progress every
        # 2 s and update the workflow progress bar. This mirrors the inline
        # write in the v04 MC loop (ascec-v04.py line 20640) — without it,
        # the progress bar freezes at "0/1 ..." for a single replica run.
        _AscecStepWriter(run_dir),
    ]
    print(
        f"Running annealing on {input_path.name} "
        f"({config.num_molecules} molecules, adapter '{adapter.name}')..."
    )
    try:
        result = anneal(
            config, energy_fn, rng=rng, callbacks=callbacks, run_logger=logger
        )
    except CosmicAscecError as exc:
        print(f"ascec: annealing failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done. {result.total_qm_calls} QM evaluations, "
        f"{result.total_accepted} configurations accepted."
    )
    print(
        f"Lowest energy = {result.lowest_energy:.8f} u.a. "
        f"(Config. {result.lowest_config_index})"
    )
    print(f"Output written to {run_dir}/")
    return 0


def _build_single_command_parser() -> argparse.ArgumentParser:
    """v04 lines 19868-20031 — the single-command argparse spec, verbatim."""
    parser = argparse.ArgumentParser(
        description=(
            "ASCEC - Annealing Simulado con Energía Cuántica\n"
            "(Simulated Annealing with Quantum Energy)\n"
            "Configurational sampling via Monte Carlo with quantum mechanical "
            "evaluation"
        ),
        usage="ascec [OPTIONS] COMMAND [ARGUMENTS]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
COMMANDS:
  Run an annealing job (the COMMAND is the .asc input file itself):
    ascec input.asc                Run simulated annealing on input.asc
    ascec input.asc rN             Build N replicate runs (e.g. r3 = 3 replicas)
    ascec input.asc rN --box10     Replicas sized for 10% effective packing
    ascec input.asc box            Box / packing analysis for the input
    ascec box input.asc            Same box analysis, box as the command

  Build downstream QM input decks from annealing results:
    ascec opt TEMPLATE [LAUNCHER]  Create geometry-optimization inputs
    ascec ref TEMPLATE [LAUNCHER]  Create refinement (single-point) inputs
    ascec update TEMPLATE [PAT]    Re-stamp existing inputs from a new template
       (LAUNCHER is optional; without it only the input files are written)

  Collect, rank and analyse results:
    ascec sort                     Sort optimized structures + write summary
       --nosum                       sort but skip the summary file
       --justsum                     write the summary only, do not sort
       --nobox                       skip box-visualization XYZ files
    ascec cosmic [FOLDER] [OPTS]   Run COSMIC clustering (see: cosmic -h)
    ascec diagram                  Energy diagram from tvse_*.dat
       --scaled                      scaled-energy variant
    ascec merge [result]           Merge launcher outputs (or merged results)

  Queue a run behind another (workflow/protocol runs only — the ones that
  appear in 'ascec status'):
    ascec input.asc after PID         Hold until job PID finishes, then start
    ascec input.asc , r3 after PID    Same, with an explicit r3 stage
       (PID is taken from 'ascec status'. Launched from a terminal the queued
        run auto-backgrounds — your prompt returns immediately and output goes
        to <input>_queue.log. It shows as HOLDING in 'ascec status' until PID
        exits, then starts automatically. Cancel it there with K, or Ctrl+C if
        you ran it in the foreground.)

  Housekeeping:
    ascec status                   Show status of the current run
    ascec launcher                 Merge launcher scripts in this folder
    ascec cleanup                  Remove temporary calculation/cosmic folders
    ascec <file> exclude [STAGE] [PATTERN]   Exclude structures from a protocol
    ascec <file> protocol          Run a multi-stage protocol workflow

  Chaining: separate commands with a comma to run them in sequence, e.g.
    ascec input.asc r3 , sort , cosmic

EXAMPLES:
    ascec water.asc                # anneal a single input
    ascec water.asc r5 --box15     # 5 replicas at 15% packing
    ascec opt template.inp run.sh  # optimization inputs + launcher
    ascec sort --justsum           # summary only
    ascec cosmic ./outputs --th auto

Run 'ascec cosmic -h' for the full COSMIC clustering option reference.
""",
    )
    parser.add_argument(
        "command", metavar="COMMAND",
        help="Input file or command (opt, ref, sort, cosmic, diagram, etc.)"
    )
    parser.add_argument(
        "arg1", nargs='?', default=None, metavar="ARG1",
        help="Command-specific argument (e.g., template file, mode)"
    )
    parser.add_argument(
        "arg2", nargs='?', default=None, metavar="ARG2",
        help="Additional command-specific argument"
    )
    parser.add_argument("-v", action="count", default=0,
                        help="Increase verbosity level (use -v, -v2, -v3, etc.)")
    parser.add_argument("--standard", action="store_true",
                        help="Use standard Metropolis criterion instead of modified")
    parser.add_argument("--nosum", action="store_true",
                        help="Skip summary file generation during sort")
    parser.add_argument("--justsum", action="store_true",
                        help="Generate summary file only without sorting structures")
    # Hidden flags (v04 lines 20023-20024).
    parser.add_argument("--target-sim-folder", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--reuse-existing", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--nobox", action="store_true",
                        help="Disable generation of box-visualization XYZ files")
    parser.add_argument("--maxprint", action="store_true",
                        help="Keep all intermediate files from every stage "
                             "(default: miniprint, clean up at end)")
    parser.add_argument("-V", "--version", action="store_true",
                        help="Display version information and exit")
    return parser


def _dispatch_exclude(argv: list[str]) -> int:
    """v04 lines 19277-19434 — ``ascec <file> exclude [stage] [pattern]``."""
    protocol_file = argv[1]
    existing_caches = sorted(glob.glob("protocol_*.pkl"))
    if not existing_caches:
        print("Error: No active protocol found")
        print("No protocol_*.pkl cache files found in current directory")
        print("This command is used to exclude stage inputs from a paused protocol.")
        return 1
    cache_file = None
    for cache_path in existing_caches:
        test_cache = load_protocol_cache(cache_path)
        if test_cache and test_cache.get("input_file") == protocol_file:
            cache_file = cache_path
            break
    if cache_file is None:
        print(f"Error: No protocol cache found for {protocol_file}")
        print("\nAvailable protocol caches:")
        for cache_path in existing_caches:
            test_cache = load_protocol_cache(cache_path)
            associated_input = test_cache.get("input_file", "unknown") if test_cache else "unknown"
            print(f"  {cache_path} -> {associated_input}")
        return 1
    print(f"Using cache file: {cache_file} (for {protocol_file})\n")
    cache = load_protocol_cache(cache_file)

    if len(argv) == 3:
        print("\n" + "=" * 60)
        print(f"Protocol Exclusion Manager - {protocol_file}")
        print("=" * 60)
        optimization_excluded = cache.get("excluded_optimizations", [])
        optimization2_excluded = cache.get("excluded_optimizations_2", [])
        if optimization_excluded:
            print(f"\nExcluded optimizations (1st stage): {optimization_excluded}")
        else:
            print("\nNo optimizations (1st stage) excluded")
        if optimization2_excluded:
            print(f"Excluded optimizations (2nd stage): {optimization2_excluded}")
        else:
            print("No optimizations (2nd stage) excluded")
        opt_excluded = cache.get("excluded_refinements", [])
        if opt_excluded:
            print(f"Excluded optimizations: {opt_excluded}")
        else:
            print("No optimizations excluded")
        print("\nUsage:")
        print(f"  ascec {protocol_file} exclude opt <pattern>     - Exclude 1st optimization files")
        print(f"  ascec {protocol_file} exclude opt2 <pattern>    - Exclude 2nd optimization files")
        print(f"  ascec {protocol_file} exclude ref <pattern>     - Exclude refinement files")
        print(f"  ascec {protocol_file} exclude clear             - Clear all exclusions")
        print(f"  ascec {protocol_file} exclude opt clear         - Clear 1st optimization exclusions")
        print(f"  ascec {protocol_file} exclude opt2 clear        - Clear 2nd optimization exclusions")
        print(f"  ascec {protocol_file} exclude ref clear         - Clear refinement exclusions")
        print("\nPattern examples:")
        print("  2               -> Exclude conf_2 (for opt) or motif_02 (for ref)")
        print("  2,5-9           -> Exclude 2, 5, 6, 7, 8, 9")
        print("  3-15            -> Exclude 3 through 15")
        print("  1,3,5-10        -> Exclude 1, 3, and 5 through 10")
        print(f"\nAfter adding exclusions, resume with: ascec {protocol_file} protocol")
        return 0

    if len(argv) < 4:
        print("Error: Missing stage type (opt, opt2, or ref)")
        return 1

    stage_type = argv[3].lower()
    if stage_type == "clear":
        cache["excluded_optimizations"] = []
        cache["excluded_optimizations_2"] = []
        cache["excluded_refinements"] = []
        save_protocol_cache(cache, cache_file)
        print("✓ All exclusions cleared")
        return 0

    if stage_type not in {"opt", "opt2", "ref"}:
        print(f"Error: Invalid stage type '{stage_type}'. Use 'opt', 'opt2', or 'ref'")
        return 1

    if len(argv) < 5:
        print("Error: Missing exclusion pattern or 'clear' command")
        print("Examples:")
        print(f"  ascec {protocol_file} exclude opt 3")
        print(f"  ascec {protocol_file} exclude opt2 4-5")
        print(f"  ascec {protocol_file} exclude ref 03-15")
        print(f"  ascec {protocol_file} exclude opt clear   # Clear opt exclusions")
        print(f"  ascec {protocol_file} exclude ref clear   # Clear ref exclusions")
        return 1

    pattern = argv[4]
    if pattern.lower() == "clear":
        if stage_type == "opt":
            cache["excluded_optimizations"] = []
            save_protocol_cache(cache, cache_file)
            print("✓ Cleared exclusions for opt (1st stage)")
        elif stage_type == "opt2":
            cache["excluded_optimizations_2"] = []
            save_protocol_cache(cache, cache_file)
            print("✓ Cleared exclusions for opt2 (2nd stage)")
        else:
            cache["excluded_refinements"] = []
            save_protocol_cache(cache, cache_file)
            print("✓ Cleared exclusions for ref")
        return 0

    try:
        excluded_numbers = parse_exclusion_pattern(pattern)
        if stage_type == "opt":
            key = "excluded_optimizations"
            existing = cache.get(key, [])
            existing.extend(excluded_numbers)
            cache[key] = sorted(list(set(existing)))
            print(f"✓ Excluded optimizations (1st stage) matching: {excluded_numbers}")
            print(f"  Total excluded: {cache[key]}")
        elif stage_type == "opt2":
            key = "excluded_optimizations_2"
            existing = cache.get(key, [])
            existing.extend(excluded_numbers)
            cache[key] = sorted(list(set(existing)))
            print(f"✓ Excluded optimizations (2nd stage) matching: {excluded_numbers}")
            print(f"  Total excluded: {cache[key]}")
        else:
            key = "excluded_refinements"
            existing = cache.get(key, [])
            existing.extend(excluded_numbers)
            cache[key] = sorted(list(set(existing)))
            print(f"✓ Excluded refinements matching: {excluded_numbers}")
            print(f"  Total excluded: {cache[key]}")
        save_protocol_cache(cache, cache_file)
        print(f"\nResume protocol with: ascec {protocol_file} protocol")
    except ValueError as e:
        print(f"Error parsing exclusion pattern: {e}")
        return 1
    return 0


def _center_text(text: str, width: int = 75) -> str:
    return text.center(width)


def _print_protocol_logo() -> None:
    """v04 lines 19487-19520 — the ASCII protocol-mode logo."""
    print("\n===========================================================================")
    print(_center_text("*********************"))
    print(_center_text("*     A S C E C     *"))
    print(_center_text("*********************"))
    print("")
    print("                             √≈≠==≈                                  ")
    print("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
    print("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
    print("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
    print("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
    print("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
    print("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
    print("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
    print("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
    print("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
    print("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
    print("")
    print("")
    print(_center_text("Universidad de Antioquia - Medellín - Colombia"))
    print("")
    print("")
    print(_center_text("Annealing Simulado Con Energía Cuántica"))
    print("")
    print(_center_text(ASCEC_VERSION))
    print("")
    print(_center_text("Química Física Teórica - QFT"))
    print("")
    print("===========================================================================")


def _stage_key_num(key: str) -> int:
    """Return the trailing stage index from a cache key.

    Stage keys are ``<type>_<num>`` but ``<type>`` may itself contain
    underscores (``energy_refinement_6``, ``geometry_optimization_2``), so the
    number is always the LAST underscore-separated field — not ``split("_")[1]``,
    which yields ``"refinement"`` for two-word types and raises ValueError.
    """
    m = re.search(r"_(\d+)$", key)
    return int(m.group(1)) if m else -1


def _dispatch_protocol(argv: list[str]) -> int:
    """v04 lines 19445-19775 — ``ascec <file> protocol [stage] [-i]``."""
    input_file = argv[1]
    restart_stage = None
    incomplete_mode = False
    if len(argv) >= 4:
        restart_stage = argv[3].lower()
        if len(argv) >= 5 and argv[4] == "-i":
            incomplete_mode = True
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return 1
    protocol = extract_protocol_from_input(input_file)
    if protocol is None:
        print("Error: No protocol found in input file")
        print("\nExpected format in input file:")
        print(".asc,")
        print("r1 --box10,")
        print("opt -c --redo=3 ../preopt_input.inp ../launcher_orca.sh,")
        print("cosmic --th=2")
        print("\nFlag meanings:")
        print("  --redo=N: Redo entire stage (opt/ref + cosmic) up to N times")
        print("\nNote: Launch failures are automatically retried up to 10 times.")
        print("\nStage restart:")
        print("  ascec input.asc protocol opt    - Restart optimization stage (deletes files)")
        print("  ascec input.asc protocol opt1   - Restart first optimization stage")
        print("  ascec input.asc protocol opt2   - Restart second optimization stage")
        print("  ascec input.asc protocol ref    - Restart refinement stage")
        print("  ascec input.asc protocol 2 -i   - Mark stage 2 as incomplete (keeps files)")
        print("\nThe -i flag marks a stage as incomplete without deleting files,")
        print("allowing the workflow to continue from where it left off.")
        return 1
    _print_protocol_logo()
    print("\nProtocol mode activated")

    protocol_text = protocol.strip()
    protocol, protocol_has_maxprint = consume_protocol_maxprint_flag(protocol)
    if protocol_has_maxprint:
        _stages_module._ascec_maxprint_requested = True

    protocol = re.sub(r"\.asc,?", input_file + ",", protocol, count=1)
    protocol = protocol.replace(",,", ",")
    protocol_args = shlex.split(protocol)
    stages = parse_workflow_stages(protocol_args[1:])
    if not stages:
        print("Error: No valid workflow stages found in protocol")
        return 1

    if restart_stage:
        existing_caches = sorted(glob.glob("protocol_*.pkl"))
        cache_file = None
        for cache_path in existing_caches:
            test_cache = load_protocol_cache(cache_path)
            if test_cache and test_cache.get("input_file") == input_file:
                cache_file = cache_path
                break
        if cache_file:
            cache = load_protocol_cache(cache_file)
            all_stage_keys = sorted(
                cache.get("stages", {}).keys(),
                key=_stage_key_num,
            )
            optimization_stages = [k for k in all_stage_keys if k.startswith("optimization_")]
            refinement_stages = [k for k in all_stage_keys if k.startswith("refinement_")]
            stages_to_restart: list[str] = []
            try:
                stage_num = int(restart_stage)
                for stage_key in cache.get("stages", {}).keys():
                    if _stage_key_num(stage_key) == stage_num:
                        stages_to_restart.append(stage_key)
            except ValueError:
                if restart_stage in ("opt", "opt1") and optimization_stages:
                    stages_to_restart.append(optimization_stages[0])
                elif restart_stage == "opt2" and len(optimization_stages) >= 2:
                    stages_to_restart.append(optimization_stages[1])
                elif restart_stage.startswith("opt") and len(restart_stage) > 3:
                    try:
                        opt_idx = int(restart_stage[3:]) - 1
                        if 0 <= opt_idx < len(optimization_stages):
                            stages_to_restart.append(optimization_stages[opt_idx])
                    except ValueError:
                        pass
                elif restart_stage in ("ref", "ref1") and refinement_stages:
                    stages_to_restart.append(refinement_stages[0])
                elif restart_stage == "ref2" and len(refinement_stages) >= 2:
                    stages_to_restart.append(refinement_stages[1])
                elif restart_stage.startswith("ref") and len(restart_stage) > 3:
                    try:
                        ref_idx = int(restart_stage[3:]) - 1
                        if 0 <= ref_idx < len(refinement_stages):
                            stages_to_restart.append(refinement_stages[ref_idx])
                    except ValueError:
                        pass
            if stages_to_restart:
                if incomplete_mode:
                    print(f"\nMarking stage(s) as incomplete: {', '.join(stages_to_restart)}")
                    print("  → Keeping existing directories (incomplete mode)")
                else:
                    print(f"\nRestarting stage(s): {', '.join(stages_to_restart)}")
                min_restart_num = min(
                    _stage_key_num(key) for key in stages_to_restart
                )
                if not incomplete_mode:
                    print(
                        f"  → Deleting all directories and cache entries from "
                        f"stage {min_restart_num} onwards\n"
                    )
                    if "optimization" in [k.split("_")[0] for k in stages_to_restart]:
                        for dir_name in ("geometry_optimization", "Geom Optimization"):
                            if os.path.exists(dir_name):
                                print(f"     Removing {dir_name}/")
                                shutil.rmtree(dir_name)
                        for pattern in (
                            "geometry_optimization_*",
                            "Geom Optimization_*",
                        ):
                            for dir_path in glob.glob(pattern):
                                if os.path.isdir(dir_path):
                                    print(f"     Removing {dir_path}/")
                                    shutil.rmtree(dir_path)
                        cosmic_dirs = sorted(
                            glob.glob("cosmic*") + glob.glob("COSMIC*")
                        )
                        for cosmic_dir in cosmic_dirs:
                            if os.path.isdir(cosmic_dir) and "_" in cosmic_dir:
                                try:
                                    cosmic_num = int(cosmic_dir.split("_")[1])
                                    if cosmic_num > min_restart_num:
                                        print(f"     Removing {cosmic_dir}/")
                                        shutil.rmtree(cosmic_dir)
                                except Exception:
                                    pass
                all_stage_keys = sorted(
                    cache.get("stages", {}).keys(),
                    key=_stage_key_num,
                )
                stages_to_remove = [
                    k for k in all_stage_keys
                    if _stage_key_num(k) >= min_restart_num
                ]
                if stages_to_remove:
                    print(f"\n  → Clearing cache entries: {', '.join(stages_to_remove)}")
                    for stage_key in stages_to_remove:
                        del cache["stages"][stage_key]
                cache["completed"] = False
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)
                print(f"Cache updated: {cache_file}\n")
            else:
                print(f"\nError: Stage '{restart_stage}' not found in cache")
                return 1
        else:
            print(f"\nError: No protocol cache found for {input_file}")
            print("Run the protocol first before trying to restart a stage")
            return 1

    result = 1
    try:
        result = execute_workflow_stages(
            input_file, stages, use_cache=True, protocol_text=protocol_text
        )
    except KeyboardInterrupt:
        result = 130
    finally:
        _pf = os.path.join(os.getcwd(), ".ascec_progress.json")
        for _p in (_pf, _pf + ".tmp"):
            try:
                os.remove(_p)
            except OSError:
                pass
    return result


def _is_auto_protocol_flag(token: str) -> bool:
    """v04 lines 19782-19783 — tokens permitted on the auto-detect path."""
    return token in {"-v", "-v2", "-v3", "--maxprint", "--nobox", "--standard"}


def _dispatch_auto_protocol(argv: list[str]) -> int | None:
    """v04 lines 19777-19826 — auto-detect embedded protocol after a bare file."""
    if len(argv) < 2:
        return None
    if not all(_is_auto_protocol_flag(tok) for tok in argv[2:]):
        return None
    if os.environ.get("ASCEC_DISABLE_EMBEDDED_PROTOCOL") == "1":
        return None
    input_file = argv[1]
    if not os.path.exists(input_file):
        return None
    if "--maxprint" in argv[2:]:
        _stages_module._ascec_maxprint_requested = True
    protocol = extract_protocol_from_input(input_file)
    if not protocol:
        return None
    protocol_text = protocol
    protocol, protocol_has_maxprint = consume_protocol_maxprint_flag(protocol)
    if protocol_has_maxprint:
        _stages_module._ascec_maxprint_requested = True
    protocol = re.sub(r"\.asc,?", input_file + ",", protocol, count=1)
    protocol = protocol.replace(",,", ",")
    protocol_args = shlex.split(protocol)
    stages = parse_workflow_stages(protocol_args[1:])
    if not stages:
        print("Error: No valid workflow stages found in embedded protocol")
        return 1
    result = 1
    try:
        result = execute_workflow_stages(
            input_file, stages, use_cache=True, protocol_text=protocol_text
        )
    except KeyboardInterrupt:
        result = 130
    finally:
        _pf = os.path.join(os.getcwd(), ".ascec_progress.json")
        for _p in (_pf, _pf + ".tmp"):
            try:
                os.remove(_p)
            except OSError:
                pass
    return result


def _dispatch_comma_workflow(argv: list[str]) -> int:
    """v04 lines 19828-19864 — comma/``then``-separated workflow mode."""
    if len(argv) < 4:
        print("Error: Workflow mode requires input file and at least one stage")
        print("Usage: ascec input.asc , r3 , opt template.inp launcher.sh , cosmic --th=2")
        print("   or: ascec input.asc then r3 then opt template.inp launcher.sh then cosmic --th=2")
        return 1
    input_file = argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return 1
    stages = parse_workflow_stages(argv[2:])
    if not stages:
        print("Error: No valid workflow stages found")
        print("Usage: ascec input.asc , r3 , opt template.inp launcher.sh , cosmic --th=2")
        print("   or: ascec input.asc then r3 then opt template.inp launcher.sh then cosmic --th=2")
        return 1
    result = 1
    try:
        result = execute_workflow_stages(input_file, stages, use_cache=True)
    except KeyboardInterrupt:
        result = 130
    finally:
        _pf = os.path.join(os.getcwd(), ".ascec_progress.json")
        for _p in (_pf, _pf + ".tmp"):
            try:
                os.remove(_p)
            except OSError:
                pass
    return result


def _extract_after_pid(argv: list[str]) -> tuple[int | None, list[str]]:
    """Pull a queue-wait target PID out of ``argv``.

    Recognised forms (anywhere after the program name):

        ascec input.asc r3 after 12345
        ascec input.asc r3 --after 12345
        ascec input.asc r3 --after=12345

    The PID is the one shown in ``ascec status`` for the run to wait on.
    Returns ``(pid_or_None, argv_without_the_tokens)``; the bare ``after``
    keyword is only consumed when immediately followed by a numeric PID so it
    cannot swallow an unrelated argument.
    """
    pid: int | None = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        low = tok.lower()
        if low in ("after", "--after") and i + 1 < len(argv) and argv[i + 1].isdigit():
            pid = int(argv[i + 1])
            i += 2
            continue
        if low.startswith("--after=") and tok.split("=", 1)[1].isdigit():
            pid = int(tok.split("=", 1)[1])
            i += 1
            continue
        out.append(tok)
        i += 1
    return pid, out


def _spawn_detached_after_run(orig_argv: list[str], after_pid: int) -> int:
    """Re-launch this exact command as a detached background process.

    Used so a queued run (``ascec input.asc after <PID>``) holds in the
    background without tying up the terminal — the user gets their prompt back
    immediately and the child blocks until ``after_pid`` exits, then runs.

    The child is launched with the *original* argv (still carrying ``after
    <PID>``) plus ``ASCEC_AFTER_DETACHED=1`` so it does not detach again; its
    stdin is ``/dev/null`` (so the Ctrl+D detach watcher never starts) and its
    console output is redirected to ``<input>_queue.log`` in the cwd. Returns
    the child PID, or 0 if the spawn failed (caller then runs in foreground).
    """
    import subprocess

    try:
        entry = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if entry and os.path.isfile(entry):
            cmd = [sys.executable, entry] + orig_argv[1:]
        else:
            cmd = [sys.executable, "-m", "cosmic_ascec.command_line.ascec"] + orig_argv[1:]

        # Name the log after the input file when we can spot it (first bare,
        # non-keyword token); fall back to a generic name otherwise.
        input_token = ""
        for tok in orig_argv[1:]:
            low = tok.lower()
            if tok.startswith("-") or low in ("after", "then", ",", "protocol"):
                continue
            input_token = tok
            break
        stem = os.path.splitext(os.path.basename(input_token))[0] if input_token else "ascec"
        log_path = os.path.join(os.getcwd(), f"{stem}_queue.log")

        env = os.environ.copy()
        env["ASCEC_AFTER_DETACHED"] = "1"

        log_fh = open(log_path, "a", buffering=1)
        try:
            child = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log_fh.close()

        print(
            f"\nASCEC queued in background (PID {child.pid}) — holding until "
            f"PID {after_pid} finishes, then it starts automatically."
        )
        print(f"  Output: {log_path}")
        print(f"  Track or cancel:  ascec status")
        return int(child.pid)
    except Exception as exc:
        print(
            f"ascec: could not background queued run ({exc}); running in foreground.",
            file=sys.stderr,
        )
        return 0


def main_ascec_integrated(argv=None) -> int:
    """v04 ``main_ascec_integrated`` (lines 19255-20982) — verbatim dispatcher.

    Pre-R0's ``run`` / ``replicate`` / ``cancel`` subcommand driver is gone.
    The dispatch chain is the exact v04 sequence of checks; the single-run
    body is wired to v05's R3 ``annealing.anneal`` engine and the workflow
    paths to the R6b ``execute_workflow_stages`` orchestrator.
    """
    if argv is None:
        argv = sys.argv
    else:
        argv = ["ascec"] + list(argv)

    # Preserve the user's original argv (with ``after <PID>`` intact) so a
    # queued run can relaunch itself detached below.
    _orig_argv = list(argv)

    # Strip --maxprint from argv (v04 lines 19262-19265). Stash the flag on the
    # stages module global the orchestrator reads via ``globals().get(...)``.
    if "--maxprint" in argv:
        _stages_module._ascec_maxprint_requested = True
    argv = [a for a in argv if a != "--maxprint"]

    # Queue support: ``... after <PID>`` / ``--after <PID>`` makes a workflow
    # run hold until that PID exits (see execute_workflow_stages). Strip the
    # tokens here — before separator detection and argparse — and stash the
    # target on the stages module the orchestrator reads via globals().get().
    _after_pid, argv = _extract_after_pid(argv)
    _stages_module._ascec_after_pid = _after_pid

    # Auto-background a queued run so the terminal returns immediately (no ``&``
    # needed). Only when launched interactively (stdout is a TTY) and not
    # already the detached child — scripts / pipes / the child itself run inline
    # so schedulers and redirects behave predictably.
    if (
        _after_pid
        and os.environ.get("ASCEC_AFTER_DETACHED") != "1"
        and sys.stdout.isatty()
    ):
        if _spawn_detached_after_run(_orig_argv, int(_after_pid)):
            return 0
        # Spawn failed → fall through and hold in the foreground instead.

    sys.argv = argv

    # v04 lines 19267-19270 — version banner.
    if len(argv) >= 2 and argv[1] in ("--version", "-V", "version"):
        print_version_banner("ASCEC")
        return 0

    # v04 lines 19272-19275 — interactive status viewer.
    if len(argv) >= 2 and argv[1].lower() == "status":
        show_ascec_status()
        return 0

    # v04 lines 19277-19434 — exclude command.
    if len(argv) >= 3 and argv[2].lower() == "exclude":
        return _dispatch_exclude(argv)

    # v04 lines 19436-19775 — protocol mode.
    if len(argv) >= 3 and argv[2].lower() == "protocol":
        return _dispatch_protocol(argv)

    # v04 lines 19777-19826 — auto-detect embedded protocol.
    auto = _dispatch_auto_protocol(argv)
    if auto is not None:
        return auto

    # v04 lines 19828-19864 — comma/then-workflow mode.
    if len(argv) >= 2 and contains_workflow_separator(argv[1:]):
        return _dispatch_comma_workflow(argv)

    # v04 lines 19866-20034 — single-command argparse mode (incl. bare input).
    parser = _build_single_command_parser()
    if len(argv) < 2:
        parser.print_help()
        return 0
    args, unknown_args = parser.parse_known_args(argv[1:])

    if args.version:
        print_version_banner("ASCEC")
        return 0

    cmd = args.command.lower()

    if cmd in ("help", "commands"):
        parser.print_help()
        return 0

    if cmd == "cosmic":
        # v04 lines 20046-20051 — ``ascec cosmic …`` passthrough.
        cosmic_args = argv[2:]
        execute_cosmic_analysis(*cosmic_args)
        return 0

    if cmd == "sort":
        # v04 lines 20053-20062.
        if args.justsum:
            execute_summary_only()
        else:
            target_sim = getattr(args, "target_sim_folder", None)
            reuse = getattr(args, "reuse_existing", False)
            execute_sort_command(
                include_summary=not args.nosum,
                target_cosmic_folder=target_sim,
                reuse_existing=reuse,
            )
        return 0

    if cmd == "box":
        # v04 lines 20064-20076 — box analysis as a primary command.
        if not args.arg1:
            print("Error: box command requires an input file.")
            print("Usage: python ascec.py box input_file.inp")
            print("   or: python ascec.py box input_file.inp > box_info.txt")
            return 1
        execute_box_analysis(args.arg1)
        return 0

    if cmd == "opt":
        # v04 lines 20078-20090.
        if not args.arg1:
            print("Error: opt command requires a template file.")
            print("Usage: python ascec.py opt template_file [launcher_template]")
            print("Example: python ascec.py opt example_input.inp launcher_orca.sh")
            print("         python ascec.py opt example_input.inp  # Creates inputs only")
            return 1
        result = create_simple_optimization_system(args.arg1, args.arg2)
        if result:
            print(result)
        return 0

    if cmd in ("ref", "refinement"):
        # v04 lines 20092-20105.
        if not args.arg1:
            print("Error: ref command requires a template file.")
            print("Usage: python ascec.py ref template_file [launcher_template]")
            print("Example: python ascec.py ref example_input.inp launcher_orca.sh")
            print("         python ascec.py ref example_input.inp  # Creates inputs only")
            return 1
        result = create_refinement_system(args.arg1, args.arg2)
        if result:
            print(result)
        return 0

    if cmd == "merge":
        # v04 lines 20107-20113.
        if args.arg1 and args.arg1.lower() == "result":
            execute_merge_result_command()
        else:
            execute_merge_command()
        return 0

    if cmd == "update":
        # v04 lines 20115-20137.
        if not args.arg1:
            print("Error: update command requires a template file.")
            print("Usage: python ascec.py update new_template.inp")
            print("   or: python ascec.py update new_template.inp pattern")
            print("This will search for files with the same extension as the template")
            return 1
        if unknown_args:
            print(
                f"Detected shell expansion (found {len(unknown_args) + 2} total "
                f"arguments). Switching to interactive mode..."
            )
            target_pattern = ""
        elif args.arg2 is None:
            target_pattern = ""
        else:
            target_pattern = args.arg2
        result = update_existing_input_files(args.arg1, target_pattern)
        print(result)
        return 0

    if cmd == "cleanup":
        # v04 lines 20139-20171.
        print("Cleaning up temporary folders...")
        temp_calc_folders = glob.glob("calculation_tmp_*")
        temp_cosmic_folders = glob.glob("cosmic_tmp_*") + glob.glob("COSMIC_tmp_*")
        retry_input = ["retry_input"] if os.path.exists("retry_input") else []
        good_structures = ["good_structures"] if os.path.exists("good_structures") else []
        all_temp = temp_calc_folders + temp_cosmic_folders + retry_input + good_structures
        if all_temp:
            if temp_calc_folders or temp_cosmic_folders:
                print("\n  Note: Found legacy _tmp_ folders from old runs")
                print("  (Current redo logic unsorts folders instead of creating _tmp_ copies)\n")
            for folder in all_temp:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"  Removed: {folder}")
            print(f"\n✓ Cleaned {len(all_temp)} temporary folder(s)")
        else:
            print("  No temporary folders found")
        return 0

    if cmd == "launcher":
        # v04 lines 20173-20177.
        merge_launcher_scripts(".")
        return 0

    if cmd == "diagram":
        # v04 lines 20179-20189.
        scaled = False
        if args.arg1 and args.arg1.lower() == "--scaled":
            scaled = True
        elif unknown_args and "--scaled" in [a.lower() for a in unknown_args]:
            scaled = True
        execute_diagram_generation(scaled=scaled)
        return 0

    # v04 lines 20192-20255 — bare input-file or ``<file> rN`` or ``<file> box``.
    input_file = args.command
    replication = args.arg1

    if replication is not None and replication.lower() == "box":
        execute_box_analysis(input_file)
        return 0

    if replication is not None:
        # ``<file> rN [--boxP]`` — v04 lines 20201-20255.
        if replication.lower().startswith("r") and len(replication) > 1:
            try:
                num_replicas = int(replication[1:])
                if num_replicas <= 0:
                    raise ValueError("Number of replicas must be positive")
                box_size_override = None
                candidates: list[str] = []
                if args.arg2 is not None:
                    candidates.append(args.arg2)
                if unknown_args:
                    candidates.extend(unknown_args)
                candidates.extend(argv[1:])
                for candidate in candidates:
                    try:
                        if isinstance(candidate, str) and candidate.lower().startswith("--box"):
                            packing_str = candidate.lower().replace("--box", "")
                            if packing_str:
                                packing_percent = float(packing_str)
                                recommended_box = get_box_size_recommendation(
                                    input_file, packing_percent
                                )
                                if recommended_box is not None:
                                    box_size_override = recommended_box
                                    print(
                                        f"Using recommended box size: "
                                        f"{box_size_override:.1f} Å "
                                        f"({packing_percent}% effective packing)"
                                    )
                                else:
                                    print(
                                        f"Warning: Could not determine box size for "
                                        f"{packing_percent}% packing. Using original box size."
                                    )
                                break
                            else:
                                print(
                                    "Warning: Invalid --box flag format. "
                                    "Expected --box<number> (e.g., --box10)"
                                )
                    except ValueError:
                        print(
                            f"Warning: Could not parse packing percentage from "
                            f"'{candidate}'. Using original box size."
                        )
                    except Exception:
                        continue
                create_replicated_runs(input_file, num_replicas, box_size=box_size_override)
                return 0
            except ValueError as e:
                print(f"Error: Invalid replication argument '{replication}'. {e}")
                print("Usage: python ascec.py input_file.in r<number> [--box<percentage>]")
                print("Example: python ascec.py example.in r3")
                print("Example: python ascec.py example.in r3 --box10  # Use 10% packing")
                return 1
        print(f"Error: Invalid replication format '{replication}'.")
        print("Usage: python ascec.py input_file.in r<number> [--box<percentage>]")
        print("Example: python ascec.py example.in r3")
        print("Example: python ascec.py example.in r3 --box10  # Use 10% packing")
        return 1

    # Bare input file → single annealing run. v04 inlines the whole loop here
    # (lines 20257-20982); v05 wires the path to the R3 live engine.
    configure_logging(verbosity=args.v if hasattr(args, "v") else 0,
                       quiet=False)
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    return _run_single_simulation(input_file, args, unknown_args, logger)


def main(argv=None) -> int:
    """Console-script entry point. Wraps :func:`main_ascec_integrated`."""
    return main_ascec_integrated(argv)


if __name__ == "__main__":
    raise SystemExit(main())
