"""Workflow stages — the COSMIC ASCEC multi-stage protocol driver.

v04's workflow is ``execute_workflow_stages`` plus five stage runners across
~8000 deeply-interleaved lines. R6 split the re-port into four sub-sessions;
**this module is filled in across all four**:

* **R6** — the cross-stage *helper* functions, ported verbatim:
  ``find_cosmic_script``, ``parse_cosmic_percentages`` / ``parse_cosmic_summary``
  / ``parse_cosmic_output``, ``get_critical_count`` / ``get_critical_files_list``,
  and the resume gates ``check_workflow_pause`` /
  ``validate_cached_optimization_cosmic`` / ``validate_cached_refinement_cosmic``.
* **R6b** — the orchestrator ``execute_workflow_stages`` (v04 11404-14207),
  ``execute_replication_stage`` (14208-14597) and the QM-concurrency machinery
  ``find_out_file_in_subdirs`` / ``process_redo_structures`` /
  ``process_optimization_redo`` / ``check_qm_output_completed`` /
  ``_run_single_qm_job`` / ``_run_qm_calculations_with_concurrency``
  (14598-16025) — ported **verbatim**.
* **R6c** — the four ``execute_*_stage`` runners (16026-18351), the
  Hessian-rescue protocol (7939-8480: ``parse_rescue_method`` /
  ``generate_rescue_hessian_input`` / ``enable_hessian_restart`` /
  ``_run_xtb_rescue_hessian`` / ``run_rescue_hessian_calculation``) and the
  eight cross-subsystem helpers the orchestrator calls
  (``generate_protocol_summary``, ``get_box_size_recommendation``,
  ``detect_convergence_status``, ``check_orca_terminated_normally_opi``,
  ``create_qm_input_file`` / ``create_xyz_input_file``,
  ``plot_annealing_diagrams`` / ``plot_combined_replicas_diagram``) — all
  ported **verbatim**.
* **R6d (this slice)** — the stage-runner *helper closure* the four R6c
  runners call by bare name (~56 helpers). 10 module-level constants
  (``r_atom`` / ``r_vdw`` / ``atomic_weights`` / ``element_symbols`` /
  ``atomic_number_to_symbol`` / ``max_mole`` / ``overlap_scale_factor`` /
  ``qm_program_details`` / ``XTB_SYNONYMS`` / ``XTB_NATIVE_MAP``) plus 54
  functions/classes (the 21 directly-referenced helpers — ``SystemState``,
  ``read_input_file``, ``calculate_optimal_box_length``, ``calculate_input_files``,
  the ``*_with_tracking`` collectors, the ORCA/xtb option resolvers — and
  the 33 they reach transitively). Extracted programmatically by
  ``scripts/extract_r6d_helpers.py``, byte-identical to ``ascec-v04.py``;
  see the "R6d verbatim ports" section below.

Everything here is a **verbatim port** of ``ascec-v04.py`` (**D-039**: faithful
decomposition — no redesign). v04's ``print`` output is reproduced byte for
byte (**D-003**).

Forward references
------------------
v04 is a single module, so ``execute_workflow_stages`` and the four stage
runners call dozens of helpers by bare name. Every port keeps those calls
verbatim; the callees fall into two groups:

* **Already re-ported elsewhere** — imported below from the ``workflow``
  package (``WorkflowContext``, the ``protocol_*.pkl`` cache, the SQLite
  ``jobs.db`` registry, the rescue/imag-freq helpers, ``create_replicated_runs``).
* **Ported into this module** — the tiny generic helpers
  (``parse_verbosity_level`` / ``natural_sort_key`` / ``_cosmic_base_name`` /
  ``_xtb_thread_env_prefix``); the R6d closure constants and helpers
  (the "R6d verbatim ports" section); the R6 cross-stage helpers; the R6b
  orchestrator + QM-concurrency machinery; the R6c stage runners + rescue
  protocol + cross-subsystem helpers. Python resolves bare names at call time,
  so the R6d constants and helpers defined *above* the R6c functions that
  call them are picked up correctly.

The optimization, refinement, replication and clustering stage paths — and
the Hessian-rescue calculation — are end-to-end runnable from this module.
The subprocess-driven replica execution inside ``execute_replication_stage``
invokes ``python sys.argv[0] <replica.asc>``, so a *full* multi-stage run
also needs the R7 root shim ``ascec.py``.
"""

from __future__ import annotations

# Imports — v04 lines 1-27 verbatim (the R6d port closure references numpy as
# ``np``, ``math``, ``Counter``/``defaultdict``, ``timedelta``,
# ``multiprocessing``, ``tempfile``, ``shlex``, ``dataclasses``,
# ``ProcessPoolExecutor``; the orchestrator already needed ``threading`` /
# ``ThreadPoolExecutor`` / ``as_completed``).
import argparse
import dataclasses
import glob
import json
import math
import multiprocessing
import os
import pickle
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cosmic_ascec.workflow.context import WorkflowContext
from cosmic_ascec.workflow.job_registry import (
    _adopt_ascec_job,
    _ascec_state_dir,
    _atomic_claim_ascec_job,
    _get_recent_jobs,
    _is_pid_alive,
    _PDEATHSIG_PREEXEC,
    _register_ascec_job,
    _remove_progress_artifacts,
    _set_job_holding,
    _update_ascec_job,
)
from cosmic_ascec.workflow.protocol_cache import (
    accumulate_stage_wall_time,
    invalidate_stage_cache,
    load_protocol_cache,
    save_protocol_cache,
    update_protocol_cache,
)
from cosmic_ascec.workflow.replicas import (
    create_replicated_runs,
    is_protocol_marker_line,
)
from cosmic_ascec.workflow.rescue import (
    count_imaginary_frequencies,
    displace_along_imaginary_mode,
    extract_final_geometry,
)

# v04 lines 29-45 (verbatim): optional / conditional imports. matplotlib backs
# the annealing-diagram plotters (R6c ``plot_annealing_diagrams`` /
# ``plot_combined_replicas_diagram``); OPI (the ORCA Python Interface) backs the
# ORCA-6.1+ output check (R6c ``check_orca_terminated_normally_opi``).
try:
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

# OPI (ORCA Python Interface) for ORCA 6.1+ support
try:
    from opi.output.core import Output as OPIOutput  # type: ignore
    OPI_AVAILABLE = True
except ImportError:
    OPI_AVAILABLE = False
    OPIOutput = None  # type: ignore

# v04 module global set by the CLI when ``--maxprint`` is passed (v04 lines
# 19262-19265). ``execute_workflow_stages`` reads it via ``globals().get(...)``;
# the v05 CLI (R7) sets ``stages._ascec_maxprint_requested`` the same way.
_ascec_maxprint_requested = False

# Queue target: when set (by the CLI from ``... after <PID>``), a workflow run
# holds in ``execute_workflow_stages`` until this PID exits before launching.
_ascec_after_pid = None

__all__ = [
    # tiny generic helpers (verbatim)
    "parse_verbosity_level",
    "natural_sort_key",
    # R6 cross-stage helpers (verbatim)
    "find_cosmic_script",
    "parse_cosmic_percentages",
    "parse_cosmic_summary",
    "parse_cosmic_output",
    "get_critical_count",
    "get_critical_files_list",
    "check_workflow_pause",
    "validate_cached_optimization_cosmic",
    "validate_cached_refinement_cosmic",
    # R6b orchestrator + QM-concurrency machinery (verbatim)
    "execute_workflow_stages",
    "execute_replication_stage",
    "find_out_file_in_subdirs",
    "process_redo_structures",
    "process_optimization_redo",
    "check_qm_output_completed",
    # R6c stage runners (verbatim)
    "execute_optimization_stage",
    "execute_cosmic_stage",
    "execute_refinement_stage",
    "execute_energy_refinement_stage",
    # R6c Hessian-rescue protocol (verbatim)
    "parse_rescue_method",
    "generate_rescue_hessian_input",
    "enable_hessian_restart",
    "run_rescue_hessian_calculation",
    # R6c cross-subsystem helpers (verbatim)
    "generate_protocol_summary",
    "get_box_size_recommendation",
    "detect_convergence_status",
    "check_orca_terminated_normally_opi",
    "create_qm_input_file",
    "create_xyz_input_file",
    "plot_annealing_diagrams",
    "plot_combined_replicas_diagram",
    # R6d stage-runner helper closure (verbatim)
    "SystemState",
    "MoleculeData",
    "read_input_file",
    "resolve_template_reference",
    "match_exclusion",
    "extract_embedded_qm_template",
    "_process_xyz_file_for_opt",
    "calculate_optimal_box_length",
    "calculate_molecular_volume",
    "calculate_molecular_extent",
    "has_primary_hydrogen_bonds",
    "calculate_hydrogen_bond_potential",
    "calculate_input_files",
    "parse_xtb_options_from_template",
    "parse_xtb_options_from_launcher",
    "build_xtb_runtime_options",
    "is_xtb_method",
    "convert_xtb_for_orca_version",
    "detect_xtb_in_template",
    "detect_orca_version",
    "detect_orca_version_from_executable",
    "detect_orca_version_from_launcher",
    "detect_orca_version_combined",
    "detect_orca_executable",
    "create_auto_launcher",
    "resolve_orca_executable_from_launcher",
    "combine_xyz_files",
    "convert_xyz_to_mol_simple",
    "create_combined_mol",
    "find_out_files",
    "group_files_by_base_with_tracking",
    "create_summary_with_tracking",
    "collect_out_files_with_tracking",
    "summarize_calculations",
    "extract_configurations_from_xyz",
    "extract_qm_executable_from_launcher",
    "get_radius",
    "get_element_symbol",
    "get_optimal_workers",
    "initialize_element_symbols",
    "initialize_element_weights",
    # R7 CLI helper closure (verbatim — see "R7 verbatim ports" section)
    "ASCEC_VERSION",
    "B2",
    "create_box_xyz_copy",
    "print_version_banner",
    "extract_protocol_from_input",
    "consume_protocol_maxprint_flag",
    "parse_exclusion_pattern",
    "provide_box_length_advice",
    "get_molecular_formula",
    "interactive_directory_selection_with_pattern",
    "update_existing_input_files",
    "extract_config_from_input_file",
    "create_simple_optimization_system",
    "create_refinement_system",
    "execute_merge_command",
    "execute_merge_result_command",
    "capture_current_state",
    "execute_summary_only",
    "execute_sort_command",
    "execute_cosmic_analysis",
    "execute_diagram_generation",
    "execute_box_analysis",
    "show_ascec_status",
]


# =========================================================================== #
# Tiny generic helpers — verbatim ports of ascec-v04.py                        #
# =========================================================================== #


def parse_verbosity_level(argv: List[str]) -> int:
    """Parse verbosity level from command line arguments.
    Returns: 0 (silent), 1 (-v), 2 (-v2), 3 (-v3)
    """
    for arg in argv:
        if arg == '-v':
            return 1
        elif arg == '-v2':
            return 2
        elif arg == '-v3':
            return 3
    return 0


def natural_sort_key(s):
    """
    Generate a sort key for natural (numerical) sorting.
    Converts 'opt_conf_10.inp' to ['opt_conf_', 10, '.inp'] for proper numerical sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def _cosmic_base_name(path: str) -> str:
    """Extract the cosmic base folder name from a path.

    For absolute paths like '/home/user/project/cosmic/orca_out_126',
    returns 'cosmic'.  For relative paths like 'cosmic/orca_out_126',
    returns 'cosmic'.  For plain names like 'cosmic', returns as-is.
    Case-insensitive (legacy 'COSMIC' directories still match) for resume safety.
    """
    if os.path.isabs(path):
        # Walk up from the leaf until we find a 'cosmic*' component
        parts = path.split(os.sep)
        for part in reversed(parts):
            if part.lower().startswith('cosmic'):
                return part
        # Fallback: parent of last component
        return os.path.basename(os.path.dirname(path))
    return path.split('/')[0] if '/' in path else path


def _canonical_calc_key(name: str) -> str:
    """Reduce any spelling of a calculation to a single structure identity.

    Optimization/refinement scans discover the same structure under several
    spellings depending on where and when they look — ``motif_24``,
    ``motif_24_opt``, ``motif_24_opt.inp``, ``geometry_refinement/motif_24/…``,
    or the redo artifact ``motif_24_opt_rescue``. Counting these as distinct
    inflates the N/M denominator past the real number of motifs (and makes redo
    passes grow the total). Collapsing every variant to its base structure key
    (``motif_24``) keeps the count anchored to the actual motif set.
    """
    base = os.path.splitext(os.path.basename(str(name)))[0]
    # Redo artifacts belong to their parent structure, not a new one.
    for suffix in ('_rescue', '_redo'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.replace('_opt', '').replace('_calc', '')


def _xtb_thread_env_prefix() -> str:
    """Compatibility prefix used by legacy launcher branches; does not set thread count."""
    return 'ASCEC_XTB_RUNTIME=1'


# =========================================================================== #
# R6d verbatim ports — the stage-runner helper closure.                        #
#                                                                             #
# The four R6c stage runners (see the "R6c verbatim ports" section below)     #
# call ~56 further v04 helpers by bare name. R6 was scoped foundation (R6) +  #
# orchestrator (R6b) + stage runners (R6c); R6d is the helper closure: 10     #
# module-level constants (``r_atom`` / ``r_vdw`` / ``atomic_weights`` /        #
# ``element_symbols`` / ``atomic_number_to_symbol`` / ``max_mole`` /           #
# ``overlap_scale_factor`` / ``qm_program_details`` / ``XTB_SYNONYMS`` /       #
# ``XTB_NATIVE_MAP``) plus 54 functions/classes (the 21 directly-referenced   #
# stubs + the 33 they reach transitively).                                    #
#                                                                             #
# Every constant + function/class below is a byte-identical extract of        #
# ``ascec-v04.py`` (**D-039**: faithful decomposition — no redesign).         #
# Extracted programmatically by ``scripts/extract_r6d_helpers.py``, not       #
# retyped. v04 source order is preserved by ``--- label  (lines N-M) ---``    #
# banners between definitions.                                                #
#                                                                             #
# v04's ``read_input_file`` / ``SystemState`` / ``calculate_optimal_box_length`` #
# overlap the pre-R0 clean-refactor ``file_formats/asc_parser.py`` /          #
# ``geometry/box.py`` (diverged signatures, never verbatim-re-ported). They   #
# are ported here verbatim "wherever they land"; a later session re-homes any #
# duplicates without altering v04 logic.                                      #
# =========================================================================== #


# --- const max_mole  (ascec-v04.py 67-67) ---
max_mole = 100  # Increase this if you have more than 100 molecules


# --- const overlap_scale_factor  (ascec-v04.py 71-71) ---
overlap_scale_factor = 0.7 # Factor to make overlap check slightly more lenient (e.g., allow partial overlap)


# --- const r_atom  (ascec-v04.py 163-271) ---
r_atom = {
    # Covalent radii for elements in Angstroms (for volume calculations and steric interactions)
    # Based on Cordero et al. (2008) "Covalent radii revisited" Dalton Trans. 2832-2838
    #   doi:10.1039/B801115J  (Carbon uses the sp2 value, 0.73 A.)
    # and optimized values from previous ASCEC usage
    
    # Period 1
    1: 0.31,   # H (Hydrogen)
    2: 0.28,   # He (Helium) - Van der Waals approximation
    
    # Period 2
    3: 1.28,   # Li (Lithium)
    4: 0.96,   # Be (Beryllium)
    5: 0.84,   # B (Boron)
    6: 0.73,   # C (Carbon) - Standard covalent radius
    7: 0.71,   # N (Nitrogen)
    8: 0.66,   # O (Oxygen)
    9: 0.57,   # F (Fluorine)
    10: 0.58,  # Ne (Neon) - Van der Waals approximation
    
    # Period 3
    11: 1.66,  # Na (Sodium)
    12: 1.41,  # Mg (Magnesium)
    13: 1.21,  # Al (Aluminum)
    14: 1.11,  # Si (Silicon)
    15: 1.07,  # P (Phosphorus)
    16: 1.05,  # S (Sulfur)
    17: 1.02,  # Cl (Chlorine)
    18: 1.06,  # Ar (Argon) - Van der Waals approximation
    
    # Period 4
    19: 2.03,  # K (Potassium)
    20: 1.76,  # Ca (Calcium)
    21: 1.70,  # Sc (Scandium)
    22: 1.60,  # Ti (Titanium)
    23: 1.53,  # V (Vanadium)
    24: 1.39,  # Cr (Chromium)
    25: 1.39,  # Mn (Manganese) - low spin
    26: 1.32,  # Fe (Iron) - low spin
    27: 1.26,  # Co (Cobalt) - low spin
    28: 1.24,  # Ni (Nickel)
    29: 1.32,  # Cu (Copper)
    30: 1.22,  # Zn (Zinc)
    31: 1.22,  # Ga (Gallium)
    32: 1.20,  # Ge (Germanium)
    33: 1.19,  # As (Arsenic)
    34: 1.20,  # Se (Selenium)
    35: 1.20,  # Br (Bromine)
    36: 1.16,  # Kr (Krypton) - Van der Waals approximation
    
    # Period 5
    37: 2.20,  # Rb (Rubidium)
    38: 1.95,  # Sr (Strontium)
    39: 1.90,  # Y (Yttrium)
    40: 1.75,  # Zr (Zirconium)
    41: 1.64,  # Nb (Niobium)
    42: 1.54,  # Mo (Molybdenum)
    43: 1.47,  # Tc (Technetium)
    44: 1.46,  # Ru (Ruthenium)
    45: 1.42,  # Rh (Rhodium)
    46: 1.39,  # Pd (Palladium)
    47: 1.45,  # Ag (Silver)
    48: 1.44,  # Cd (Cadmium)
    49: 1.42,  # In (Indium)
    50: 1.39,  # Sn (Tin)
    51: 1.39,  # Sb (Antimony)
    52: 1.38,  # Te (Tellurium)
    53: 1.39,  # I (Iodine)
    54: 1.40,  # Xe (Xenon) - Van der Waals approximation
    
    # Period 6
    55: 2.44,  # Cs (Cesium)
    56: 2.15,  # Ba (Barium)
    57: 2.07,  # La (Lanthanum)
    
    # Period 6 - Lanthanides (f-block, elements 58-71)
    58: 2.04,  # Ce (Cerium)
    59: 2.03,  # Pr (Praseodymium)
    60: 2.01,  # Nd (Neodymium)
    61: 1.99,  # Pm (Promethium)
    62: 1.98,  # Sm (Samarium)
    63: 1.98,  # Eu (Europium)
    64: 1.96,  # Gd (Gadolinium)
    65: 1.94,  # Tb (Terbium)
    66: 1.92,  # Dy (Dysprosium)
    67: 1.92,  # Ho (Holmium)
    68: 1.89,  # Er (Erbium)
    69: 1.90,  # Tm (Thulium)
    70: 1.87,  # Yb (Ytterbium)
    71: 1.87,  # Lu (Lutetium)
    
    # Period 6 - Transition metals (d-block, elements 72-80)
    72: 1.75,  # Hf (Hafnium)
    73: 1.70,  # Ta (Tantalum)
    74: 1.62,  # W (Tungsten)
    75: 1.51,  # Re (Rhenium)
    76: 1.44,  # Os (Osmium)
    77: 1.41,  # Ir (Iridium)
    78: 1.36,  # Pt (Platinum)
    79: 1.36,  # Au (Gold)
    80: 1.32,  # Hg (Mercury)
    
    # Period 6 - Main group (p-block, elements 81-86)
    81: 1.45,  # Tl (Thallium)
    82: 1.46,  # Pb (Lead)
    83: 1.48,  # Bi (Bismuth)
    84: 1.40,  # Po (Polonium)
    85: 1.50,  # At (Astatine)
    86: 1.50,  # Rn (Radon)

    # Period 7 - actinides etc. (Cordero et al. 2008, doi:10.1039/B801115J;
    # table ends at Cm, Z=96. Fr is the paper's extrapolated value. Z>=97 are
    # absent from Cordero and fall back to get_radius's 1.50 A default.)
    87: 2.60,  # Fr (Francium) - extrapolated
    88: 2.21,  # Ra (Radium)
    89: 2.15,  # Ac (Actinium)
    90: 2.06,  # Th (Thorium)
    91: 2.00,  # Pa (Protactinium)
    92: 1.96,  # U (Uranium)
    93: 1.90,  # Np (Neptunium)
    94: 1.87,  # Pu (Plutonium)
    95: 1.80,  # Am (Americium)
    96: 1.69,  # Cm (Curium)
}


# --- const r_vdw  (ascec-v04.py 274-367) ---
r_vdw = {
    # Van der Waals radii in Angstroms (used for monatomic species). Sources:
    #   A. Bondi, "van der Waals Volumes and Radii," J. Phys. Chem. 1964,
    #     68(3), 441-451. doi:10.1021/j100785a001  (base set)
    #   S. Alvarez, "A cartography of the van der Waals territories,"
    #     Dalton Trans. 2013, 42, 8617-8636. doi:10.1039/C3DT50599E
    #     (heavier/transition elements; main-group extensions also match
    #      Mantina et al., J. Phys. Chem. A 2009, 113, 5806-5812,
    #      doi:10.1021/jp8111556).
    # NOTE: several transition-metal entries are a flat 2.00 A placeholder.
    # Period 1
    1: 1.20,   # H
    2: 1.40,   # He
    # Period 2
    3: 1.82,   # Li
    4: 1.53,   # Be
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    10: 1.54,  # Ne
    # Period 3
    11: 2.27,  # Na
    12: 1.73,  # Mg
    13: 1.84,  # Al
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    18: 1.88,  # Ar
    # Period 4
    19: 2.75,  # K
    20: 2.31,  # Ca
    21: 2.11,  # Sc
    22: 2.00,  # Ti
    23: 2.00,  # V
    24: 2.00,  # Cr
    25: 2.00,  # Mn
    26: 2.00,  # Fe
    27: 2.00,  # Co
    28: 1.63,  # Ni
    29: 1.40,  # Cu
    30: 1.39,  # Zn
    31: 1.87,  # Ga
    32: 2.11,  # Ge
    33: 1.85,  # As
    34: 1.90,  # Se
    35: 1.85,  # Br
    36: 2.02,  # Kr
    # Period 5
    37: 3.03,  # Rb
    38: 2.49,  # Sr
    39: 2.31,  # Y
    40: 2.23,  # Zr
    41: 2.18,  # Nb
    42: 2.17,  # Mo
    43: 2.16,  # Tc
    44: 2.13,  # Ru
    45: 2.10,  # Rh
    46: 1.63,  # Pd
    47: 1.72,  # Ag
    48: 1.58,  # Cd
    49: 1.93,  # In
    50: 2.17,  # Sn
    51: 2.06,  # Sb
    52: 2.06,  # Te
    53: 1.98,  # I
    54: 2.16,  # Xe
    # Period 6
    55: 3.43,  # Cs
    56: 2.68,  # Ba
    57: 2.43,  # La
    58: 2.42,  # Ce
    59: 2.40,  # Pr
    60: 2.39,  # Nd
    61: 2.38,  # Pm
    62: 2.36,  # Sm
    63: 2.35,  # Eu
    64: 2.34,  # Gd
    65: 2.33,  # Tb
    66: 2.31,  # Dy
    67: 2.30,  # Ho
    68: 2.29,  # Er
    69: 2.27,  # Tm
    70: 2.26,  # Yb
    71: 2.24,  # Lu
    72: 2.23,  # Hf
    73: 2.22,  # Ta
    74: 2.18,  # W
    75: 2.16,  # Re
    76: 2.16,  # Os
    77: 2.13,  # Ir
    78: 1.75,  # Pt
    79: 1.66,  # Au
    80: 1.55,  # Hg
    81: 1.96,  # Tl
    82: 2.02,  # Pb
    83: 2.07,  # Bi
    84: 1.97,  # Po
    85: 2.02,  # At
    86: 2.20,  # Rn
}


# --- def get_radius  (ascec-v04.py 369-386) ---
def get_radius(atomic_num: int, mol_def=None) -> float:
    """Return the appropriate atomic radius for box-length calculations.

    Uses van der Waals radius for monatomic species (single atom or ion) and
    covalent radius for multi-atom molecules.

    Args:
        atomic_num: Atomic number of the element.
        mol_def: MoleculeData object (optional). If provided and has only one atom,
                 the VDW radius is used.

    Returns:
        float: Radius in Angstroms.
    """
    is_monatomic = mol_def is not None and len(mol_def.atoms_coords) == 1
    if is_monatomic:
        return r_vdw.get(atomic_num, r_atom.get(atomic_num, 1.70) * 1.2)
    return r_atom.get(atomic_num, 1.50)


# --- const element_symbols  (ascec-v04.py 390-403) ---
element_symbols = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}


# --- const atomic_number_to_symbol  (ascec-v04.py 407-420) ---
atomic_number_to_symbol = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}


# --- const atomic_weights  (ascec-v04.py 425-450) ---
atomic_weights = {
    1: 1.008,    2: 4.003,    3: 6.940,    4: 9.012,    5: 10.81,
    6: 12.01,    7: 14.01,    8: 16.00,    9: 19.00,   10: 20.18,
    11: 22.99,   12: 24.31,   13: 26.98,   14: 28.09,   15: 30.97,
    16: 32.06,   17: 35.45,   18: 39.95,   19: 39.10,   20: 40.08,
    21: 44.96,   22: 47.87,   23: 50.94,   24: 52.00,   25: 54.94,
    26: 55.85,   27: 58.93,   28: 58.69,   29: 63.55,   30: 65.38,
    31: 69.72,   32: 72.63,   33: 74.92,   34: 78.97,   35: 79.90,
    36: 83.80,   37: 85.47,   38: 87.62,   39: 88.91,   40: 91.22,
    41: 92.91,   42: 95.95,   43: 97.00,   44: 101.1,   45: 102.9,
    46: 106.4,   47: 107.9,   48: 112.4,   49: 114.8,   50: 118.7,
    51: 121.8,   52: 127.6,   53: 126.9,   54: 131.3,   55: 132.9,
    56: 137.3,   57: 138.9,   58: 140.1,   59: 140.9,   60: 144.2,
    61: 145.0,   62: 150.4,   63: 152.0,   64: 157.2,   65: 158.9,
    66: 162.5,   67: 164.9,   68: 167.3,   69: 168.9,   70: 173.0,
    71: 175.0,   72: 178.5,   73: 180.9,   74: 183.8,   75: 186.2,
    76: 190.2,   77: 192.2,   78: 195.1,   79: 197.0,   80: 200.6,
    81: 204.4,   82: 207.2,   83: 209.0,   84: 209.0,   85: 210.0,
    86: 222.0,   87: 223.0,   88: 226.0,   89: 227.0,   90: 232.0,
    91: 231.0,   92: 238.0,   93: 237.0,   94: 244.0,   95: 243.0,
    96: 247.0,   97: 247.0,   98: 251.0,   99: 252.0,  100: 257.0,
    101: 258.0,  102: 259.0,  103: 262.0,  104: 267.0,  105: 270.0,
    106: 269.0,  107: 270.0,  108: 270.0,  109: 278.0,  110: 281.0,
    111: 281.0,  112: 285.0,  113: 286.0,  114: 289.0,  115: 289.0,
    116: 293.0,  117: 293.0,  118: 294.0
}


# --- class SystemState  (ascec-v04.py 478-669) ---
class SystemState:
    def __init__(self):
        # Configuration parameters read from input file
        self.random_generate_config: int = 0      # 0: Annealing; 1: Random configurations
        self.num_random_configs: int = 0          # Used if random_generate_config is 1
        self.cube_length: float = 0.0             # Simulation Cube Length (Angstroms)
        self.quenching_routine: int = 0           # 1: Linear, 2: Geometrical
        self.linear_temp_init: float = 0.0
        self.linear_temp_decrement: float = 0.0
        self.linear_num_steps: int = 0
        self.geom_temp_init: float = 0.0
        self.geom_temp_factor: float = 0.0
        self.geom_num_steps: int = 0
        self.max_cycle: int = 0                   # Maximum Monte Carlo Cycles per Temperature (initial value from input)
        self.max_cycle_floor: int = 10            # Floor value for maxstep reduction (default: 10)
        self.max_displacement_a: float = 0.0      # Maximum Displacement of each mass center (ds)
        self.max_rotation_angle_rad: float = 0.0  # Maximum Rotation angle in radians (dphi)
        self.conformational_move_prob: float = 0.0  # Probability of conformational move vs rigid-body move (read from input)
        self.max_dihedral_angle_rad: float = 0.0  # Maximum dihedral rotation angle in radians (read from input)
        self.rotatable_bonds_by_molecule: List[List[Tuple[int, int, List[int]]]] = []  # Cached rotatable bond definitions per molecule
        self.molecules_with_rotatable_bonds: List[bool] = []  # Per-molecule availability flag for conformational moves
        self.total_rotatable_bonds: int = 0  # Total number of cached rotatable bonds in the system
        self.ia: int = 0                          # QM program type (1: Gaussian, 2: ORCA, etc.)
        self.alias: str = ""                      # Program alias/executable name (e.g., "g09")
        self.qm_method: Optional[str] = None      # (e.g., "pm3", "hf") - Renamed from hamiltonian for clarity
        self.qm_basis_set: Optional[str] = None   # (e.g., "6-31G*", "STO-3G") - Renamed from basis_set for clarity
        self.charge: int = 0                      # (iQ)
        self.multiplicity: int = 0                # (iS2) - Renamed from spin_multiplicity for clarity
        self.qm_memory: Optional[str] = None      # memory - No default, will be None if not in input
        self.qm_nproc: Optional[int] = None       # nprocs
        self.qm_additional_keywords: str = ""     # if necessary
        self._orca_exe_checked: bool = False      # Cache ORCA executable check per run
        self._orca_full_path: Optional[str] = None  # Cached resolved ORCA executable path
        
        # Parallel processing settings for ASCEC
        self.ascec_parallel_cores: int = 1       # Number of cores for ASCEC operations (file I/O, coordinate transformations)
        self.use_ascec_parallel: bool = False    # Enable parallel processing within ASCEC operations
        
        self.num_molecules: int = 0               # (nmo) from input file
        self.output_dir: str = "."                # Directory for output files
        self.ivalE: int = 0                       # Evaluate Energy flag (0: No energy evaluation, just mto movements)
        self.mto: int = 0                         # Number of random configurations if ivalE=0 (Fortran's nrandom)
        self.qm_program: Optional[str] = None     # Will store "gaussian", "orca", etc., derived from alias or ia
        self.molecules_to_add = []                # Initialize as an empty list
        self.all_molecule_definitions = []        # Also good to initialize this if not already
        self.openbabel_alias = "obabel"           # Default Open Babel executable name
        
        # Add these from global constants
        self.r_atom = r_atom.copy() 
        self.overlap_scale_factor = overlap_scale_factor
        self.max_overlap_placement_attempts = 100000 # Max attempts to place a single molecule without significant overlap

        # Random number generator state (internal to ran0_method)
        # This MUST be an integer. It will be updated by ran0_method.
        self.random_seed: int = -1
        self.IY: int = 0                          # Internal state for ran0_method
        self.IVEC: List[int] = [0] * 32           # Internal state for ran0_method (NTAB=32 from ran0_method)

        # Simulation State Variables (updated during simulation)
        self.nstep: int = 0                        # Total energy evaluation steps (Fortran's cycle)
        self.icount: int = 0                       # Counter for steps at current temperature (Fortran's icount)
        self.iboltz: int = 0                       # Count of configurations accepted by Boltzmann (Fortran's iboltz)
        self.current_energy: float = 0.0          # Ep (current energy of the system)
        self.lowest_energy: float = float('inf')  # oldE (stores the lowest energy found, initialize to large value)
        self.lowest_energy_rp: Optional[np.ndarray] = None # Coordinates of the lowest energy config
        self.lowest_energy_iznu: Optional[List[int]] = None # Atomic numbers of the lowest energy config
        self.lowest_energy_config_idx: int = 0    # Index of the lowest energy config
        self.current_temp: float = 0.0            # Ti (current annealing temperature)
        self.maxstep: int = 0                     # Current maximum steps at a temperature (derived from max_cycle initially, then reduced)
        self.natom: int = 0                       # Total number of atoms, calculated from molecule definitions
        self.nmo: int = 0                         # Number of molecules (often kept in sync with num_molecules)
        self.jdum: str = ""                       # Timestamp/label for output files (can be populated from alias or datetime)
        self.xbox: float = 0.0                    # Length of the simulation cube (synced with cube_length)
        self.ds: float = 0.0                      # Max displacement (synced with max_displacement_a)
        self.dphi: float = 0.0                    # Max rotation angle (synced with max_rotation_angle_rad)
        self.last_accepted_qm_call_count: int = 0 # Stores qm_call_count at last accepted configuration for history
        self.total_accepted_configs: int = 0      # Counter for total accepted configurations written to XYZ
        self.lower_energy_configs: int = 0        # Counter for configurations with lower energy than previous lowest
        self.consecutive_na_steps: int = 0        # Counter for consecutive N/A steps in annealing
        
        # New variable to track the global QM call count at the last history line written
        self.last_history_qm_call_count: int = 0

        # Molecular and Atomic Data
        self.rp: np.ndarray = np.empty((0, 3), dtype=np.float64) # Current atomic coordinates
        self.rf: np.ndarray = np.empty((0, 3), dtype=np.float64) # Proposed atomic coordinates (not directly used as state, but as return value)
        self.rcm: np.ndarray = np.empty((max_mole, 3), dtype=np.float64) # Center of mass for each molecule
        self.imolec: List[int] = []               # Indices defining molecules (which atoms belong to which molecule)
        self.iznu: List[int] = []                 # Atomic numbers for each atom in the system
        self.all_molecule_definitions: List['MoleculeData'] = [] # Stores parsed molecule data from input
        self.coords_per_molecule: np.ndarray = np.empty((0,3), dtype=np.float64) # Template coordinates for one molecule
        self.atomic_numbers_per_molecule: List[int] = [] # Template atomic numbers for one molecule
        self.natom_per_molecule: int = 0          # Number of atoms in the template molecule
        self.molecule_label: str = ""             # Label of the template molecule

        # Element Maps
        self.atomic_number_to_symbol: Dict[int, str] = {}
        self.atomic_number_to_weight: Dict[int, float] = {} # Populated from atomic_weights
        self.atomic_number_to_mass: Dict[int, float] = {}   # Alias for atomic_number_to_weight, used for clarity in COM calcs

        # These Fortran-style arrays might be redundant if the dicts above are the primary source.
        self.sym: List[str] = [''] * 120          # From Fortran's symelem (symbol for atomic number index)
        self.wt: List[float] = [0.0] * 120        # From Fortran's wtelem (weight for atomic number index)
        
        # Additional QM related attributes
        self.qm_call_count: int = 0 # Counter for QM calculation calls (cumulative global count)
        self.initial_qm_retries: int = 100 # Number of retries for initial QM calculation, default to 100
        self.verbosity_level: int = 0 # 0: default, 1: --v (every 10 steps), 2: --va (all steps)
        # QM attempt debug preservation mode:
        #   'none'          -> do not copy per-attempt files
        #   'first_success' -> keep per-attempt files until first successful QM, then stop (recommended)
        #   'all'           -> keep per-attempt files for all attempts (slowest)
        self.qm_attempt_debug_mode: str = "first_success"
        self._first_success_qm_preserved: bool = False

        self.use_standard_metropolis: bool = False # Flag for Metropolis criterion
        
        # Additional attributes for proper type checking
        self.input_file_path: str = ""            # Path to the input file
        self.max_molecular_extent: float = 0.0    # Maximum molecular extent
        self.num_elements_defined: int = 0        # Number of unique elements defined
        self.element_types: List[Tuple[int, int]] = []  # List of (atomic_number, count) tuples
        self.iz_types: List[int] = []             # List of atomic numbers
        self.nnu_types: List[int] = []            # List of atom counts per element type
        self.volume_based_recommendations: Dict = {}  # Volume-based box recommendations

    def ran0_method(self) -> float:
        """
        A Python implementation of Numerical Recipes' ran0 random number generator.
        This method is kept for historical context/compatibility with the original Fortran.
        For new Python code, `random.random()` or `np.random.rand()` are generally preferred.
        """
        IA = 16807
        IM = 2147483647 # 2^31 - 1
        AM = 1.0 / IM
        IQ = 127773
        IR = 2836
        NTAB = 32
        NDIV = 1 + (IM - 1) // NTAB
        EPS = 1.2e-7
        RNMX = 1.0 - EPS

        if self.random_seed == 0:
            self.random_seed = 1

        if self.random_seed < 0:
            self.random_seed = -self.random_seed
            for j in range(NTAB + 7):
                k = self.random_seed // IQ
                self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
                if self.random_seed < 0:
                    self.random_seed += IM
                if j < NTAB:
                    self.IVEC[j] = self.random_seed
            self.IY = self.IVEC[0]

        k = self.random_seed // IQ
        self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
        if self.random_seed < 0:
            self.random_seed += IM
        j = self.random_seed // NDIV

        self.IVEC[j] = self.random_seed # This line was incorrect in previous version (used self.IVEC[j] = self.IY)
        self.IY = self.IVEC[j] # Correct usage: self.IY becomes the value from IVEC[j]

        if self.IY > RNMX * IM:
            return RNMX
        else:
            return float(self.IY) * AM

    def get_molecule_label_from_atom_index(self, atom_abs_idx: int) -> str:
        """
        Determines the label of the molecule to which a given atom belongs.
        Assumes `imolec` and `all_molecule_definitions` are correctly populated.
        """
        if not self.imolec or not self.all_molecule_definitions:
            return "Unknown"

        # Find which molecule this atom belongs to based on imolec
        # imolec stores the starting atom index for each molecule definition,
        # plus the total number of atoms at the end.
        for i in range(self.num_molecules):
            start_idx = self.imolec[i]
            end_idx = self.imolec[i+1] # This is the start of the *next* molecule, or total atoms

            if start_idx <= atom_abs_idx < end_idx:
                # The molecule_definitions list is indexed from 0 to (num_molecules - 1)
                # This needs to correctly map to the *instance* of the molecule.
                # Since molecules_to_add now stores the correct definition index for each instance, use that.
                mol_def_idx = self.molecules_to_add[i]
                return self.all_molecule_definitions[mol_def_idx].label
        return "Unknown"


# --- def _print_verbose  (ascec-v04.py 672-681) ---
def _print_verbose(message: str, level: int, state: Optional[SystemState], file=sys.stderr):
    """
    Prints a message to stderr if the state's verbosity level meets or exceeds the required level.
    level 0: always print (critical errors, final summary, accepted configs, and key annealing steps)
    level 1: --v (every 10 steps), plus level 0)
    level 2: --va (all steps, plus level 0 and 1)
    """
    if state is None or state.verbosity_level >= level:
        print(message, file=file)
        file.flush()


# --- def get_optimal_workers  (ascec-v04.py 690-713) ---
def get_optimal_workers(task_type: str, num_items: int) -> int:
    """
    Calculate optimal number of workers for different task types.
    
    Args:
        task_type: Type of task ('cpu_intensive', 'io_intensive', 'mixed')
        num_items: Number of items to process
    
    Returns:
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    if task_type == 'cpu_intensive':
        # For CPU-intensive tasks like parsing files
        return min(cpu_count, num_items, 8)
    elif task_type == 'io_intensive':
        # For I/O-intensive tasks like file copying
        return min(cpu_count * 2, num_items, 6)  
    elif task_type == 'mixed':
        # For mixed tasks like file creation
        return min(cpu_count, num_items, 4)
    else:
        return min(cpu_count, num_items, 4)  # Default


def _process_xyz_file_for_calc(xyz_file_data):
    """
    Process a single XYZ file for calculation system creation.
    This is a module-level function to support multiprocessing.
    Used by both simple and standard calculation workflows.
    """
    xyz_file, template_content, optimization_dir_path, qm_program, input_ext, total_xyz_files, use_combined_naming = xyz_file_data

    configurations = extract_configurations_from_xyz(xyz_file)
    if not configurations:
        return [], f"Warning: No configurations found in {xyz_file}"

    dir_name = os.path.dirname(xyz_file)
    if dir_name == ".":
        run_num = 1
    else:
        parts = os.path.basename(dir_name).split('_')
        run_num = 1
        for part in reversed(parts):
            try:
                run_num = int(part)
                break
            except ValueError:
                continue

    file_input_files = []
    xyz_file_fallback = os.path.basename(xyz_file).replace('.xyz', '')

    for config in configurations:
        original_comment = config['comment']
        energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
        config_match = re.search(r'Configuration: (\d+)', original_comment)

        source_name = xyz_file_fallback
        if '|' in original_comment:
            parts = original_comment.split('|')
            last_part = parts[-1].strip()
            if '|' in last_part:
                source_name = last_part.split('|')[0].strip()
            else:
                source_name = last_part

        energy = energy_match.group(1) if energy_match else "unknown"
        config_num = config_match.group(1) if config_match else config['config_num']

        if energy == "unknown":
            config['comment'] = f"Configuration: {config_num} | {source_name}"
        else:
            config['comment'] = f"Configuration: {config_num} | E = {energy} a.u. | {source_name}"

        if use_combined_naming or total_xyz_files == 1:
            input_name = f"opt_conf_{config['config_num']}{input_ext}"
        else:
            input_name = f"opt{run_num}_conf_{config['config_num']}{input_ext}"
        input_path = os.path.join(optimization_dir_path, input_name)

        if qm_program == 'xtb':
            if create_xyz_input_file(config, input_path):
                file_input_files.append(input_name)
        else:
            if create_qm_input_file(config, template_content, input_path, qm_program):
                file_input_files.append(input_name)

    if total_xyz_files == 1:
        return file_input_files, f"Processed {xyz_file} with {len(configurations)} configurations"
    else:
        return file_input_files, f"Processed {xyz_file} (run {run_num}) with {len(configurations)} configurations"


# --- def _process_xyz_file_for_opt  (ascec-v04.py 799-856) ---
def _process_xyz_file_for_opt(xyz_file_data):
    """
    Process a single XYZ file for optimization system creation.
    This is a module-level function to support multiprocessing.
    """
    xyz_file, template_content, opt_dir, qm_program, input_ext = xyz_file_data
    
    # Extract configurations
    configurations = extract_configurations_from_xyz(xyz_file)
    if not configurations:
        return [], f"Warning: No configurations found in {xyz_file}"
    
    file_input_files = []
    base_name = os.path.basename(xyz_file).replace('.xyz', '')
    
    # Create input files for each configuration
    for config in configurations:
        # Extract motif name from comment if available, otherwise use base filename
        import re
        comment = config['comment']
        
        # Check for umotif first (must come before motif check since umotif contains 'motif')
        umotif_match = re.search(r'(umotif_\d+)', comment, re.IGNORECASE)
        motif_match = re.search(r'(?<!u)(motif_\d+)', comment, re.IGNORECASE)
        
        # If not found in comment, try to extract from filename
        if not umotif_match and not motif_match:
            umotif_match = re.search(r'(umotif_\d+)', base_name, re.IGNORECASE)
            motif_match = re.search(r'(?<!u)(motif_\d+)', base_name, re.IGNORECASE)
        
        if umotif_match:
            # Keep umotif prefix, just add _opt suffix
            source_name = umotif_match.group(1).lower()
            input_name = f"{source_name}_opt{input_ext}"
        elif motif_match:
            # Keep motif prefix, just add _opt suffix
            # COSMIC will later promote motif_##_opt → umotif_##
            source_name = motif_match.group(1).lower()
            input_name = f"{source_name}_opt{input_ext}"
        else:
            # For non-motif files, use simple opt_conf_X naming
            input_name = f"opt_conf_{config['config_num']}{input_ext}"
            source_name = base_name
        
        input_path = os.path.join(opt_dir, input_name)
        
        # Update config comment with source file info
        config['comment'] = f"Configuration: {config['config_num']} | Source: {source_name}"
        
        # Create input file
        if qm_program == 'xtb':
            if create_xyz_input_file(config, input_path):
                file_input_files.append(input_name)
        else:
            if create_qm_input_file(config, template_content, input_path, qm_program):
                file_input_files.append(input_name)
    
    return file_input_files, f"Processed {xyz_file} with {len(configurations)} configurations"


# --- def initialize_element_symbols  (ascec-v04.py 1105-1114) ---
def initialize_element_symbols(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to element symbols
    and populates the Fortran-style list state.sym.
    """
    state.atomic_number_to_symbol = atomic_number_to_symbol.copy()

    for z, symbol in atomic_number_to_symbol.items():
        if z < len(state.sym):
            state.sym[z] = symbol.strip()


# --- def get_element_symbol  (ascec-v04.py 1117-1119) ---
def get_element_symbol(atomic_number: int) -> str:
    """Retrieves the element symbol for a given atomic number."""
    return atomic_number_to_symbol.get(atomic_number, 'X') # Default to 'X' for unknown/dummy


# --- def initialize_element_weights  (ascec-v04.py 1167-1177) ---
def initialize_element_weights(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to their atomic weights
    and populates the list state.wt. Also populates atomic_number_to_mass.
    """
    state.atomic_number_to_weight = atomic_weights.copy()
    state.atomic_number_to_mass = atomic_weights.copy() # Populate this for COM calculations

    for atomic_num, weight in atomic_weights.items():
        if 0 < atomic_num < len(state.wt):
            state.wt[atomic_num] = weight


# --- class MoleculeData  (ascec-v04.py 1349-1356) ---
class MoleculeData:
    """
    A data structure to hold information for a single molecule parsed from input.
    """
    def __init__(self, label: str, num_atoms: int, atoms_coords: List[Tuple[int, float, float, float]]):
        self.label = label
        self.num_atoms = num_atoms
        self.atoms_coords = atoms_coords


# --- def read_input_file  (ascec-v04.py 1422-1743) ---
def read_input_file(state: SystemState, source) -> List[MoleculeData]:
    """
    Reads and parses the input. Can read from a file path or a file-like object (like sys.stdin).
    Populates SystemState and returns a list of MoleculeData objects.
    This version uses a two-phase parsing approach for robustness and correctly handles '*' separators.
    """
    molecule_definitions: List[MoleculeData] = []
    
    lines = []
    if isinstance(source, str):
        state.input_file_path = source
        with open(source, 'r') as f_obj:
            lines = f_obj.readlines()
    else:
        state.input_file_path = "stdin_input.in" # Placeholder name for stdin
        lines = source.readlines()

    def clean_line(line: str) -> str:
        """Removes comments and leading/trailing whitespace from a line, including non-breaking spaces."""
        if '#' in line:
            line = line.split('#')[0]
        if '!' in line:
            line = line.split('!')[0]
        line = line.replace('\xa0', ' ')
        return line.strip()

    lines_iterator = iter(lines) 

    # PHASE 1: Read Fixed Configuration Parameters (Lines 1-13)
    config_lines_parsed = 0 

    # We expect 13 lines for fixed configuration parameters (including conformational sampling)
    while config_lines_parsed < 13:
        try:
            raw_line = next(lines_iterator)
        except StopIteration:
            raise EOFError(f"Unexpected end of input while reading configuration parameters. Expected 13 lines, but found only {config_lines_parsed}.")
        
        line_num = lines.index(raw_line) + 1 
        line = clean_line(raw_line)

        if not line: 
            continue

        parts = line.split()
        if not parts: 
            continue

        # Parsing config lines based on count
        if config_lines_parsed == 0: # Line 1: Simulation Mode & Number of Config
            state.random_generate_config = int(parts[0])
            state.num_random_configs = int(parts[1])
            
            if state.random_generate_config == 0: 
                state.ivalE = 0 
                state.mto = state.num_random_configs 
            elif state.random_generate_config == 1: 
                state.ivalE = 1 
                state.mto = 0   

        elif config_lines_parsed == 1: # Line 2: Simulation Cube Length
            state.cube_length = float(parts[0])
            state.xbox = state.cube_length

        elif config_lines_parsed == 2: # Line 3: Annealing Quenching Routine
            state.quenching_routine = int(parts[0])

        elif config_lines_parsed == 3: # Line 4: Linear Quenching Parameters
            state.linear_temp_init = float(parts[0])
            state.linear_temp_decrement = float(parts[1])
            state.linear_num_steps = int(parts[2])

        elif config_lines_parsed == 4: # Line 5: Geometric Quenching Parameters
            state.geom_temp_init = float(parts[0])
            # Parse the geometric factor. If quenching_routine is 2, interpret as percentage decrease.
            if state.quenching_routine == 2:
                # User specified "decrease a 5%". So 5.0 in input means 0.95 factor.
                factor_percentage = float(parts[1])
                if not (0.0 < factor_percentage < 100.0):
                    raise ValueError(f"Error parsing geometric temperature factor on line {line_num}: Expected a percentage between 0 and 100, got '{parts[1]}'.")
                state.geom_temp_factor = 1.0 - (factor_percentage / 100.0)
            else:
                # If not geometric, just read it directly (though it shouldn't be used).
                state.geom_temp_factor = float(parts[1])
            state.geom_num_steps = int(parts[2])

        elif config_lines_parsed == 5: # Line 6: Maximum Monte Carlo Cycles per T and floor value (optional)
            state.max_cycle = int(parts[0])
            # Check if floor value is provided (optional second parameter)
            if len(parts) >= 2:
                state.max_cycle_floor = int(parts[1])
            # If not provided, default value (10) is already set in __init__
            state.maxstep = state.max_cycle # Initialize maxstep with the initial max_cycle from input

        elif config_lines_parsed == 6: # Line 7: Maximum Displacement & Rotation
            state.max_displacement_a = float(parts[0])
            state.max_rotation_angle_rad = float(parts[1])
            state.ds = state.max_displacement_a
            state.dphi = state.max_rotation_angle_rad

        elif config_lines_parsed == 7: # Line 8: Conformational sampling (%) & Maximum dihedral rotation (degrees)
            # Parse conformational move probability as percentage (0-100)
            conformational_percent = float(parts[0])
            if not (0.0 <= conformational_percent <= 100.0):
                raise ValueError(f"Error parsing conformational move percentage on line {line_num}: Expected a value between 0 and 100, got '{parts[0]}'.")
            state.conformational_move_prob = conformational_percent / 100.0  # Convert to 0.0-1.0 range
            
            # Parse maximum dihedral rotation angle in degrees
            max_dihedral_degrees = float(parts[1])
            if not (0.0 <= max_dihedral_degrees <= 180.0):
                raise ValueError(f"Error parsing maximum dihedral angle on line {line_num}: Expected a value between 0 and 180 degrees, got '{parts[1]}'.")
            state.max_dihedral_angle_rad = np.radians(max_dihedral_degrees)  # Convert to radians

        elif config_lines_parsed == 8: # Line 9: QM Program Index & Alias (e.g., "1 g09")
            state.ia = int(parts[0])
            if state.ia not in qm_program_details:
                raise ValueError(
                    f"Unknown QM program index {state.ia} on line {line_num}. "
                    f"Valid: 1 (Gaussian), 2 (ORCA), 3 (xtb)."
                )
            state.qm_program = parts[1]
            state.alias = parts[1]
            state.jdum = state.alias

        elif config_lines_parsed == 9: # Line 10: Hamiltonian & Basis Set (e.g., "pm3 zdo")
            state.qm_method = parts[0]      
            # Handle case where only method is provided (basis set is optional)
            if len(parts) > 1:
                state.qm_basis_set = parts[1]   
            else:
                # If no basis set provided, leave it as None - respect user's input
                state.qm_basis_set = None   

        elif config_lines_parsed == 10:      # Line 11: nprocs (QM calculations and ASCEC evaluation)
            state.qm_nproc = int(parts[0])   
            
            # Check for ASCEC parallel cores (optional second parameter)
            if len(parts) > 1:
                # User explicitly specified ASCEC cores
                ascec_cores = int(parts[1])
                if ascec_cores > 1:
                    state.ascec_parallel_cores = ascec_cores
                    state.use_ascec_parallel = True
                    _print_verbose(f"ASCEC parallel processing: {state.ascec_parallel_cores} cores (user-specified)", 1, state)
                else:
                    state.ascec_parallel_cores = 1
                    state.use_ascec_parallel = False
                    _print_verbose(f"ASCEC parallel processing disabled (user-specified)", 1, state)
            else:
                # Only one value provided - let ASCEC auto-decide based on available resources
                cpu_count = multiprocessing.cpu_count()
                if state.qm_nproc and state.qm_nproc < cpu_count:
                    # Use remaining cores for ASCEC operations
                    remaining_cores = cpu_count - state.qm_nproc
                    if remaining_cores >= 2:
                        state.ascec_parallel_cores = min(4, remaining_cores)  # Cap at 4 cores for ASCEC
                        state.use_ascec_parallel = True
                        _print_verbose(f"ASCEC parallel processing: {state.ascec_parallel_cores} cores (auto-detected)", 1, state)
                        _print_verbose(f"  (System: {cpu_count} cores, QM: {state.qm_nproc}, ASCEC: {state.ascec_parallel_cores})", 1, state)
                    else:
                        state.ascec_parallel_cores = 1
                        state.use_ascec_parallel = False
                        _print_verbose(f"ASCEC parallel processing disabled (insufficient remaining cores)", 1, state)
                else:
                    # QM uses all or more cores than available - no parallel ASCEC
                    state.ascec_parallel_cores = 1
                    state.use_ascec_parallel = False

        elif config_lines_parsed == 11:     # Line 12: Charge & Spin Multiplicity
            state.charge = int(parts[0])
            state.multiplicity = int(parts[1]) 

        elif config_lines_parsed == 12:     # Line 13: Number of Molecules
            try:
                state.num_molecules = int(parts[0])
            except ValueError:
                raise ValueError(f"Error parsing 'Number of Molecules' on line {line_num}: Expected an integer, but found '{parts[0]}'. "
                                 "Please ensure line 13 of your input file contains the total number of molecules.")
            state.nmo = state.num_molecules
        else:
            _print_verbose(f"Warning: Unexpected configuration line at index {config_lines_parsed}. Line: {line}", 1, state)

        config_lines_parsed += 1 

    # PHASE 2: Read Molecule Definitions
    reading_molecule = False
    current_molecule_num_atoms_expected = 0
    current_molecule_label = ""
    current_molecule_atoms: List[Tuple[int, float, float, float]] = []
    atoms_read_in_current_molecule = 0

    for raw_line in lines_iterator: 
        line_num = lines.index(raw_line) + 1 
        
        # Stop parsing if we hit the Protocol section (old or new format)
        stripped = raw_line.strip()
        if '# Protocol' in raw_line or '# protocol' in raw_line.lower() or is_protocol_marker_line(raw_line):
            break
        
        line = clean_line(raw_line)

        if not line:
            continue

        parts = line.split()

        if parts[0] == "*":
            if reading_molecule: # Found '*' and was reading a molecule -> this '*' closes the previous molecule
                if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
                    molecule_definitions.append(
                        MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
                    )
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected {current_molecule_num_atoms_expected} atoms but read {atoms_read_in_current_molecule} for molecule {current_molecule_label}.")
                
                current_molecule_num_atoms_expected = 0
                current_molecule_label = ""
                current_molecule_atoms = []
                atoms_read_in_current_molecule = 0
                
                reading_molecule = True 
                continue 

            else: # Found '*' and was NOT reading a molecule -> this must be the very first '*' opening the first molecule
                reading_molecule = True
                continue 

        elif reading_molecule: # We are inside a molecule block 
            if current_molecule_num_atoms_expected == 0: 
                try:
                    current_molecule_num_atoms_expected = int(parts[0])
                except ValueError:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected number of atoms, got '{parts[0]}'.")
            elif not current_molecule_label: 
                current_molecule_label = parts[0]
            else: # Expecting atom coordinates
                if atoms_read_in_current_molecule < current_molecule_num_atoms_expected:
                    if len(parts) < 4:
                        raise ValueError(f"Error parsing atom coordinates near line {line_num}: Expected element symbol and 3 coordinates, got '{line}'.")
                    
                    symbol = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    atomic_num = element_symbols.get(symbol)
                    if atomic_num is None:
                        raise ValueError(f"Error: Unknown element symbol '{symbol}' near line {line_num}.")
                    
                    current_molecule_atoms.append((atomic_num, x, y, z))
                    atoms_read_in_current_molecule += 1
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Read more atoms than expected for molecule {current_molecule_label}. Missing '*' delimiter?")
        else: 
            _print_verbose(f"Warning: Unexpected content '{raw_line.strip()}' found outside of defined blocks near line {line_num}.", 1, state)
            continue 
            
    if reading_molecule and current_molecule_num_atoms_expected > 0: 
        if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
            molecule_definitions.append(
                MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
            )
        else:
            raise ValueError(f"Error: Last molecule block not properly closed or incomplete for molecule {current_molecule_label}.")

    if len(molecule_definitions) != state.num_molecules:
        raise ValueError(f"Error: Number of molecules defined in input ({len(molecule_definitions)}) does not match expected ({state.num_molecules}).")

    state.all_molecule_definitions = molecule_definitions
    # Populate molecules_to_add with indices corresponding to the order they were defined
    # This ensures that each instance (water1, water2, etc.) refers to its specific definition.
    state.molecules_to_add = list(range(state.num_molecules))

    state.natom = sum(mol.num_atoms for mol in molecule_definitions)
    
    if molecule_definitions: 
        # The template molecule for initial placement is still the first one defined
        first_molecule_data = molecule_definitions[0]
        state.natom_per_molecule = first_molecule_data.num_atoms
        state.molecule_label = first_molecule_data.label 

        coords_list = [[atom[1], atom[2], atom[3]] for atom in first_molecule_data.atoms_coords]
        state.coords_per_molecule = np.array(coords_list, dtype=np.float64)

        state.atomic_numbers_per_molecule = [atom[0] for atom in first_molecule_data.atoms_coords]
        
        # Calculate max molecular extent for suggestion
        min_coords_all_mols = np.min(state.coords_per_molecule, axis=0)
        max_coords_all_mols = np.max(state.coords_per_molecule, axis=0)
        max_extent = np.max(max_coords_all_mols - min_coords_all_mols)
        state.max_molecular_extent = max_extent
    else:
        raise ValueError("No molecule definitions found in the input file.")

    state.iznu = [0] * state.natom 
    
    unique_elements = {}
    atom_idx = 0
    for mol_def_idx in state.molecules_to_add: # Iterate through the *indices* of molecules to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition based on its index
        for atomic_num, _, _, _ in mol_data.atoms_coords:
            state.iznu[atom_idx] = atomic_num
            unique_elements[atomic_num] = unique_elements.get(atomic_num, 0) + 1
            atom_idx += 1

    state.num_elements_defined = len(unique_elements) 
    state.element_types = [(z, unique_elements[z]) for z in sorted(list(unique_elements.keys()))]
    state.iz_types = [z for z, _ in state.element_types] 
    state.nnu_types = [count for _, count in state.element_types] 

    initialize_element_symbols(state)
    initialize_element_weights(state) # This populates atomic_number_to_weight and atomic_number_to_mass

    state.imolec = [0] * (state.num_molecules + 1)
    current_atom_for_imolec = 0
    for i, mol_def_idx in enumerate(state.molecules_to_add): # Iterate through instances to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition
        state.imolec[i] = current_atom_for_imolec
        current_atom_for_imolec += mol_data.num_atoms
    state.imolec[state.num_molecules] = current_atom_for_imolec

    return molecule_definitions


# --- def extract_embedded_qm_template  (ascec-v04.py 1821-1881) ---
def extract_embedded_qm_template(input_file: str, template_label: str) -> Optional[Tuple[str, str]]:
    """
    Extract an embedded QM template block by label from an input file.

    Supported headers:
        #orca <label>
        #gaussian <label>
        #xtb <label>

    Returns:
        Tuple (template_content, extension) where extension is ".inp" for ORCA,
        ".com" for Gaussian, and ".xtb" for standalone xTB metadata blocks,
        or None if no matching block is found.
    """
    if not input_file or not template_label:
        return None

    label_norm = template_label.strip().lower()
    if not label_norm:
        return None

    header_re = re.compile(r'^\s*#\s*(orca|gaussian|xtb)\s+(\S+)\s*$', re.IGNORECASE)

    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return None

    collecting = False
    detected_program = None
    collected: List[str] = []

    for raw_line in lines:
        match = header_re.match(raw_line)
        if match:
            program = match.group(1).lower()
            label = match.group(2).strip().lower()

            if collecting:
                # Reached next template header, stop current block.
                break

            if label == label_norm:
                collecting = True
                detected_program = program
            continue

        if collecting:
            collected.append(raw_line)

    if not collecting or not collected or not detected_program:
        return None

    if detected_program == 'orca':
        extension = '.inp'
    elif detected_program == 'gaussian':
        extension = '.com'
    else:
        extension = '.xtb'
    return ''.join(collected).strip() + '\n', extension


# --- def resolve_template_reference  (ascec-v04.py 1884-1923) ---
def resolve_template_reference(context: 'WorkflowContext', template_token: str) -> Optional[str]:
    """
    Resolve template reference for workflow stages.

    Resolution order:
    1) Existing file path (as provided).
    2) Embedded template label in context.input_file (#orca/#gaussian/#xtb blocks).

    Returns absolute path to a resolved template file, or None if unresolved.
    """
    if not template_token:
        return None

    token = template_token.strip()
    if not token:
        return None

    # Direct file path resolution first.
    token_abs = os.path.abspath(token)
    if os.path.exists(token_abs):
        return token_abs

    input_file = getattr(context, 'input_file', '')
    extracted = extract_embedded_qm_template(input_file, token)
    if not extracted:
        return None

    template_content, extension = extracted
    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', token)
    out_name = f"{safe_label}{extension}"
    out_path = os.path.abspath(out_name)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        if hasattr(context, 'generated_template_files'):
            context.generated_template_files.append(out_path)
        return out_path
    except Exception:
        return None


# --- def match_exclusion  (ascec-v04.py 1984-2008) ---
def match_exclusion(filename: str, excluded_numbers: List[int]) -> bool:
    """
    Check if a filename matches any excluded number.
    
    Examples:
        "motif_01_opt.inp" matches [1]
        "opt_conf_3.inp" matches [3]
        "motif_15_opt.inp" matches [15]
    
    Args:
        filename: Input filename
        excluded_numbers: List of numbers to exclude
        
    Returns:
        True if filename should be excluded
    """
    # Extract all numbers from filename
    numbers = re.findall(r'\d+', filename)
    
    for num_str in numbers:
        num = int(num_str)
        if num in excluded_numbers:
            return True
    
    return False


# --- const qm_program_details  (ascec-v04.py 2963-2989) ---
qm_program_details = {
    1: {
        "name": "gaussian",
        "default_exe": "g09", # Common executable name.
        "input_ext": ".gjf",
        "output_ext": ".log",
        "energy_regex": r"SCF Done:\s+E\(.+\)\s*=\s*([-\d.]+)\s+A\.U\.",
        "termination_string": "Normal termination",
    },
    2: {
        "name": "orca",
        "default_exe": "orca", # Common executable name.
        "input_ext": ".inp",
        "output_ext": ".out",
        "energy_regex": r"FINAL SINGLE POINT ENERGY\s*:?\s*([-+]?\d+\.\d+)",  # Works for ORCA 5 and 6 (with or without colon)
        "termination_string": "ORCA TERMINATED NORMALLY",
        "alternative_termination": ["****ORCA TERMINATED NORMALLY****", "OPTIMIZATION RUN DONE"],  # Additional termination patterns
    },
    3: {
        "name": "xtb",
        "default_exe": "xtb",
        "input_ext": ".xyz",
        "output_ext": ".out",
        "energy_regex": r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh",
        "termination_string": "normal termination of xtb",
    },
}


# --- def calculate_molecular_volume  (ascec-v04.py 4242-4304) ---
def calculate_molecular_volume(mol_def, method='coordinate_based') -> float:
    """
    Calculates the approximate volume of a molecule using its xyz coordinates.

    Uses the convex hull of atom positions expanded by their covalent radii to
    compute a coordinate-aware volume. Falls back to a bounding-box approach
    if scipy is not available.

    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers
        method (str): 'coordinate_based' (default) or 'covalent_spheres' (legacy)

    Returns:
        float: Estimated molecular volume in Angstroms^3
    """
    if not mol_def.atoms_coords:
        return 0.0

    if method == 'covalent_spheres':
        # Legacy: sum of individual atomic volumes (not coordinate-aware)
        total_volume = 0.0
        for atomic_num, x, y, z in mol_def.atoms_coords:
            radius = get_radius(atomic_num, mol_def)
            atomic_volume = (4.0/3.0) * np.pi * (radius ** 3)
            total_volume += atomic_volume
        overlap_factor = 0.74
        return total_volume * overlap_factor

    # Default: coordinate-based volume using convex hull of atomic spheres
    # Uses VDW radii for monatomic species, covalent for molecules.
    try:
        from scipy.spatial import ConvexHull

        # Generate surface points on each atomic sphere using Fibonacci sphere (50 points)
        # Fibonacci sphere produces nearly uniform distribution with no coplanar clusters,
        # reducing algorithm sensitivity between scipy Qhull and JS incremental hull
        N_SURF = 50
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        directions = []
        for i in range(N_SURF):
            theta = math.acos(1.0 - 2.0 * (i + 0.5) / N_SURF)
            phi = 2.0 * math.pi * i / golden_ratio
            dx = math.sin(theta) * math.cos(phi)
            dy = math.sin(theta) * math.sin(phi)
            dz = math.cos(theta)
            directions.append((dx, dy, dz))

        all_surface_points = []
        for atomic_num, x, y, z in mol_def.atoms_coords:
            radius = get_radius(atomic_num, mol_def)
            for ddx, ddy, ddz in directions:
                all_surface_points.append([x + radius*ddx, y + radius*ddy, z + radius*ddz])

        if len(all_surface_points) < 4:
            return _bounding_box_volume(mol_def)

        hull = ConvexHull(np.array(all_surface_points))
        return hull.volume

    except ImportError:
        return _bounding_box_volume(mol_def)
    except Exception:
        return _bounding_box_volume(mol_def)


# --- def _bounding_box_volume  (ascec-v04.py 4307-4323) ---
def _bounding_box_volume(mol_def) -> float:
    """Fallback volume calculation using bounding box of atom positions + radii."""
    if not mol_def.atoms_coords:
        return 0.0

    min_xyz = np.array([np.inf, np.inf, np.inf])
    max_xyz = np.array([-np.inf, -np.inf, -np.inf])

    for atomic_num, x, y, z in mol_def.atoms_coords:
        radius = get_radius(atomic_num, mol_def)
        pos = np.array([x, y, z])
        min_xyz = np.minimum(min_xyz, pos - radius)
        max_xyz = np.maximum(max_xyz, pos + radius)

    extents = max_xyz - min_xyz
    # Bounding box overestimates; apply correction factor (~0.52 for typical molecules)
    return extents[0] * extents[1] * extents[2] * 0.52


# --- def calculate_molecular_extent  (ascec-v04.py 4326-4355) ---
def calculate_molecular_extent(mol_def) -> float:
    """
    Calculate the longest molecular extent (max distance between any two atoms
    including their radii). This is the molecular 'diameter'.
    Uses VDW radii for monatomic species (atoms/ions), covalent for molecules.

    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers

    Returns:
        float: Longest extent in Angstroms
    """
    if not mol_def.atoms_coords or len(mol_def.atoms_coords) < 2:
        if mol_def.atoms_coords:
            r = get_radius(mol_def.atoms_coords[0][0], mol_def)
            return 2.0 * r
        return 0.0

    max_dist = 0.0
    coords = mol_def.atoms_coords
    for i in range(len(coords)):
        ai, xi, yi, zi = coords[i]
        ri = get_radius(ai, mol_def)
        for j in range(i + 1, len(coords)):
            aj, xj, yj, zj = coords[j]
            rj = get_radius(aj, mol_def)
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2) + ri + rj
            if dist > max_dist:
                max_dist = dist
    return max_dist


# --- def has_primary_hydrogen_bonds  (ascec-v04.py 4358-4381) ---
def has_primary_hydrogen_bonds(mol_defs) -> bool:
    """
    Detect whether the system has significant primary hydrogen bond donors/acceptors.
    A system has primary H-bonds if at least one molecule has both H-bond donors
    (H atoms) and acceptors (N, P, O, S, Se, F, Cl, Br, I — each counting as 1).

    Args:
        mol_defs: List of MoleculeData objects

    Returns:
        bool: True if the system has primary hydrogen bonding potential
    """
    HB_ACCEPTORS = {'N', 'P', 'O', 'S', 'Se', 'F', 'Cl', 'Br', 'I'}
    total_donors = 0
    total_acceptors = 0
    for mol_def in mol_defs:
        for atomic_num, x, y, z in mol_def.atoms_coords:
            element = get_element_symbol(atomic_num)
            if element == 'H':
                total_donors += 1
            elif element in HB_ACCEPTORS:
                total_acceptors += 1
    # Need at least 1 donor AND 1 acceptor for primary H-bonds
    return total_donors >= 1 and total_acceptors >= 1


# --- def calculate_hydrogen_bond_potential  (ascec-v04.py 4384-4431) ---
def calculate_hydrogen_bond_potential(mol_def) -> Dict:
    """
    Calculate the potential hydrogen bonding volume based on donor/acceptor counts.

    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers

    Returns:
        Dictionary with hydrogen bond analysis and volume estimation
    """
    # Average hydrogen bond length in organic clusters
    avg_hb_length = 2.5  # Angstroms
    
    # Volume estimation: cylindrical volume around H-bond
    # Using radius of ~1.2 Å (approximate H-bond interaction zone)
    hb_interaction_radius = 1.2
    hb_volume_per_bond = math.pi * (hb_interaction_radius ** 2) * avg_hb_length
    
    # Count potential donors and acceptors
    HB_ACCEPTORS = {'N', 'P', 'O', 'S', 'Se', 'F', 'Cl', 'Br', 'I'}
    donors = 0
    acceptors = 0

    for atomic_num, x, y, z in mol_def.atoms_coords:
        element = get_element_symbol(atomic_num)

        # Hydrogen bond donors (H atoms)
        if element == 'H':
            donors += 1

        # Hydrogen bond acceptors (N, P, O, S, Se, F, Cl, Br, I — each counts as 1)
        elif element in HB_ACCEPTORS:
            acceptors += 1
    
    # Estimate potential hydrogen bonds (limited by the smaller of donors/acceptors)
    potential_hb_bonds = min(donors, acceptors)
    
    # Total volume needed for hydrogen bonding network
    hb_network_volume = potential_hb_bonds * hb_volume_per_bond
    
    return {
        'donors': int(donors),
        'acceptors': int(acceptors),
        'potential_bonds': potential_hb_bonds,
        'avg_bond_length': avg_hb_length,
        'hb_volume_per_bond': hb_volume_per_bond,
        'total_hb_volume': hb_network_volume
    }


# --- def calculate_optimal_box_length  (ascec-v04.py 4434-4568) ---
def calculate_optimal_box_length(state: SystemState, target_packing_fractions: Optional[List[float]] = None) -> Dict:
    """
    Calculates optimal box lengths based on packing fraction.

    Two methods (both use L = (V_eff / phi)^(1/3)):
      Method A (HB systems):  V_eff = V_mol + V_HB  (hull volumes + H-bond ghost cylinders)
      Method B (non-HB):      V_eff = L_diag³        (diagonal-derived volume from extents)
                               where L_diag = sum_of_extents / sqrt(3)

    Method A is preferred when the system has primary hydrogen bond potential.
    Method B is used otherwise.

    Args:
        state: SystemState object containing molecule definitions
        target_packing_fractions: List of target packing fractions to calculate box sizes for

    Returns:
        Dict: Results containing volumes, box lengths for different packing fractions, and recommendations
    """
    if target_packing_fractions is None:
        target_packing_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if not state.all_molecule_definitions:
        return {'error': 'No molecule definitions found'}

    # Detect whether system has primary hydrogen bonds
    system_has_hbonds = has_primary_hydrogen_bonds(state.all_molecule_definitions)

    results = {
        'individual_molecular_volumes': [],
        'total_molecular_volume': 0.0,
        'num_molecules': state.num_molecules,
        'box_length_recommendations': {},
        'packing_analysis': {},
        'method': 'A_hb_volume' if system_has_hbonds else 'B_diagonal_extent'
    }

    # Calculate volume, extent, and hydrogen bond potential for each molecule
    unique_molecular_volumes = []
    unique_hb_analyses = []
    unique_extents = []

    for mol_def in state.all_molecule_definitions:
        volume = calculate_molecular_volume(mol_def)
        hb_analysis = calculate_hydrogen_bond_potential(mol_def)
        extent = calculate_molecular_extent(mol_def)

        unique_molecular_volumes.append(volume)
        unique_hb_analyses.append(hb_analysis)
        unique_extents.append(extent)

        results['individual_molecular_volumes'].append({
            'molecule_label': mol_def.label,
            'num_atoms': mol_def.num_atoms,
            'volume_A3': volume,
            'extent_A': extent,
            'hb_donors': hb_analysis['donors'],
            'hb_acceptors': hb_analysis['acceptors'],
            'potential_hb_bonds': hb_analysis['potential_bonds'],
            'hb_network_volume_A3': hb_analysis['total_hb_volume']
        })

    # Calculate totals across all molecules to be placed
    total_molecular_volume = 0.0
    total_hb_network_volume = 0.0
    total_extent_sum = 0.0

    for i, mol_def_idx in enumerate(state.molecules_to_add):
        if mol_def_idx < len(unique_molecular_volumes):
            total_molecular_volume += unique_molecular_volumes[mol_def_idx]
            total_hb_network_volume += unique_hb_analyses[mol_def_idx]['total_hb_volume']
            total_extent_sum += unique_extents[mol_def_idx]

    results['total_molecular_volume'] = total_molecular_volume
    results['total_hb_network_volume'] = total_hb_network_volume
    results['has_primary_hbonds'] = system_has_hbonds

    # Diagonal-based reference length (sum of extents / √3)
    diagonal = total_extent_sum * 1.0
    diagonal_box_length = diagonal / math.sqrt(3.0)
    diagonal_derived_volume = diagonal_box_length ** 3
    results['diagonal_sum_extents'] = total_extent_sum
    results['diagonal_value'] = diagonal
    results['diagonal_box_length'] = diagonal_box_length
    results['diagonal_derived_volume'] = diagonal_derived_volume

    # Select effective volume based on method
    if system_has_hbonds:
        # Method A: hull volumes + H-bond ghost cylinders
        total_effective_volume = total_molecular_volume + total_hb_network_volume
    else:
        # Method B: diagonal-derived volume from molecular extents
        total_effective_volume = diagonal_derived_volume

    results['total_effective_volume'] = total_effective_volume

    if total_effective_volume <= 0:
        return {'error': 'Total effective volume is zero or negative'}

    # Calculate box lengths for different packing fractions (same formula for both cases;
    # V_eff = V_mol + V_HB when HB present, V_eff = V_mol when no HB)
    for packing_fraction in target_packing_fractions:
        required_box_volume = total_effective_volume / packing_fraction
        box_length = required_box_volume ** (1.0/3.0)

        results['box_length_recommendations'][f'{packing_fraction:.1%}'] = {
            'packing_fraction': packing_fraction,
            'box_length_A': box_length,
            'box_volume_A3': required_box_volume,
            'free_volume_A3': required_box_volume - total_effective_volume,
            'free_volume_fraction': 1.0 - (total_effective_volume / required_box_volume),
            'molecular_volume_fraction': total_molecular_volume / required_box_volume,
            'hb_network_volume_fraction': total_hb_network_volume / required_box_volume,
        }

    # Current box analysis
    if hasattr(state, 'cube_length') and state.cube_length > 0:
        current_box_volume = state.cube_length ** 3
        current_packing_fraction = total_effective_volume / current_box_volume

        results['current_box_analysis'] = {
            'current_box_length_A': state.cube_length,
            'current_box_volume_A3': current_box_volume,
            'current_packing_fraction': current_packing_fraction,
            'current_free_volume_A3': current_box_volume - total_effective_volume,
            'current_free_volume_fraction': 1.0 - current_packing_fraction,
            'molecular_packing_fraction': total_molecular_volume / current_box_volume,
            'hb_network_fraction': total_hb_network_volume / current_box_volume
        }

    # Largest single molecular extent (for reference)
    max_molecular_extent = max(unique_extents) if unique_extents else 0.0
    results['max_molecular_extent_A'] = max_molecular_extent

    return results


# --- def extract_configurations_from_xyz  (ascec-v04.py 4982-5104) ---
def extract_configurations_from_xyz(xyz_file_path: str) -> List[Dict]:
    """
    Extracts all configurations from an XYZ file.
    
    Args:
        xyz_file_path (str): Path to the XYZ file
    
    Returns:
        List[Dict]: List of configuration dictionaries with atoms, energy, and config number
    """
    configurations = []
    
    try:
        with open(xyz_file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
            
            # Read number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            
            # Read comment line with configuration info
            if i + 1 >= len(lines):
                break
            
            comment_line = lines[i + 1].strip()
            
            # Extract configuration number and energy from comment
            config_num = 1
            energy = 0.0
            
            if "Configuration:" in comment_line:
                parts = comment_line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Configuration:"):
                        try:
                            config_num = int(part.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "E =" in part:
                        try:
                            energy_str = part.split("=")[1].strip().split()[0]
                            energy = float(energy_str)
                        except (ValueError, IndexError):
                            pass
            else:
                # Handle other formats like "Motif_01_opt_conf_4 (G = -55.537389 hartree)"
                import re
                # Try to extract configuration number from patterns like "conf_4" or "Motif_01"
                conf_match = re.search(r'conf_(\d+)', comment_line)
                if conf_match:
                    try:
                        config_num = int(conf_match.group(1))
                    except ValueError:
                        pass
                else:
                    # Try to extract from Motif_XX or motif_XX pattern
                    motif_match = re.search(r'[Mm]otif_(\d+)', comment_line)
                    if motif_match:
                        try:
                            config_num = int(motif_match.group(1))
                        except ValueError:
                            pass
                
                # Try to extract energy from patterns like "G = -55.537389" or "E = -55.537389"
                energy_match = re.search(r'[GE]\s*=\s*([-\d.]+)', comment_line)
                if energy_match:
                    try:
                        energy = float(energy_match.group(1))
                    except ValueError:
                        pass
            
            # Read atom coordinates (preserve original string format for precision)
            atoms = []
            for j in range(num_atoms):
                if i + 2 + j >= len(lines):
                    break
                
                atom_line = lines[i + 2 + j].strip()
                if atom_line:
                    parts = atom_line.split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x_str = parts[1]  # Keep original string format
                        y_str = parts[2]  # Keep original string format
                        z_str = parts[3]  # Keep original string format
                        # Also store float values for numerical operations if needed
                        try:
                            x_float = float(x_str)
                            y_float = float(y_str)
                            z_float = float(z_str)
                            atoms.append((symbol, x_str, y_str, z_str, x_float, y_float, z_float))
                        except ValueError:
                            # Skip malformed coordinates
                            continue
            
            if len(atoms) == num_atoms:
                # Convert Motif_ to motif_ in comment for consistency
                processed_comment = comment_line.replace('Motif_', 'motif_')
                configurations.append({
                    'config_num': config_num,
                    'energy': energy,
                    'atoms': atoms,
                    'comment': processed_comment
                })
            
            i += num_atoms + 2
    
    except IOError as e:
        print(f"Error reading XYZ file '{xyz_file_path}': {e}")
        return []
    
    return configurations


# --- def interactive_xyz_file_selection  (ascec-v04.py 5650-5798) ---
def interactive_xyz_file_selection(xyz_files: List[str], optimization_dir_path: str = ".", auto_select: Optional[str] = None,
                                   quiet: bool = False) -> List[str]:
    """
    Provides interactive selection for XYZ files to process.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        optimization_dir_path (str): Directory where combined files should be created
        auto_select (Optional[str]): Auto-selection mode:
            - 'all': Process all result_*.xyz files (excludes combined files)
            - 'combined': Combine all result_*.xyz files into combined_r{N}.xyz, then process that
            - None: Interactive prompt for user selection
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate result_*.xyz files from combined_results.xyz and combined_r*.xyz
    result_files = [f for f in xyz_files if not (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    combined_files = [f for f in xyz_files if (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    
    # Handle auto-selection
    if auto_select == 'all':
        if not quiet:
            print(f"\nAuto-selected: All result_*.xyz files ({len(result_files)} files)")
        return result_files
    elif auto_select == 'combined':
        if len(result_files) == 1:
            # Single file: just use it directly (treat as "combined")
            if not quiet:
                print(f"\nAuto-selected: Single result file (treating as combined)")
                print(f"Using: {os.path.basename(result_files[0])}")
            return result_files
        elif len(result_files) > 1:
            # Multiple files: combine them
            if not quiet:
                print(f"\nAuto-combining {len(result_files)} result_*.xyz files...")
            combined_filename = os.path.join(optimization_dir_path, f"combined_r{len(result_files)}.xyz")
            success = merge_xyz_files(result_files, combined_filename, quiet=quiet)
            if success:
                if not quiet:
                    print(f"Successfully created {os.path.basename(combined_filename)}")
                return [combined_filename]
            else:
                if not quiet:
                    print("Failed to combine files. Using individual files instead.")
                return result_files
        else:
            # No files
            if not quiet:
                print("\nNo result files found")
            return []
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("XYZ file selection".center(60))
    print("=" * 60)
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # First show result_*.xyz files
    if result_files:
        print("\nResult files:")
        for xyz_file in result_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Then show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(result_files) > 1:
        options["a"] = "All result files"
        print(f"a. Process all result_*.xyz files ({len(result_files)} files, excluding combined)")
        options["c"] = "Combine result files"
        print(f"c. Combine all result_*.xyz files first, then process the combined file (combined_r{len(result_files)}.xyz)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'c' to combine): ").strip()
            
            if choice.lower() == 'q':
                print("Operation cancelled.")
                return []
            
            if choice.lower() == 'a' and len(result_files) > 1:
                print(f"Selected: All result_*.xyz files ({len(result_files)} files)")
                return result_files
            
            if choice.lower() == 'c' and len(result_files) > 1:
                print(f"Combining {len(result_files)} result_*.xyz files...")
                # Use merge_xyz_files to combine result files only
                combined_filename = os.path.join(optimization_dir_path, f"combined_r{len(result_files)}.xyz")
                success = merge_xyz_files(result_files, combined_filename)
                if success:
                    print(f"Successfully created {os.path.basename(combined_filename)}")
                    return [combined_filename]
                else:
                    print("Failed to combine files. Using individual files instead.")
                    return result_files
            
            # Handle numbered selections (single or multiple)
            try:
                numbers = [int(x.strip()) for x in choice.split() if x.strip().isdigit()]
                selected_files = []
                
                for num in numbers:
                    if str(num) in options:
                        selected_files.append(options[str(num)])
                    else:
                        print(f"Invalid number: {num}")
                        raise ValueError
                
                if selected_files:
                    print(f"Selected {len(selected_files)} file(s):")
                    for f in selected_files:
                        print(f"  - {f}")
                    return selected_files
                else:
                    print("No valid files selected.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all result files.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return []
        except EOFError:
            print("\nOperation cancelled.")
            return []


# --- def extract_qm_executable_from_launcher  (ascec-v04.py 5915-6013) ---
def extract_qm_executable_from_launcher(launcher_content: str, qm_program_idx: int) -> str:
    """
    Extract the QM executable path from the launcher template by looking for
    any exported variable ending in _ROOT and constructing the executable path,
    or by finding existing run commands in the template.
    
    Args:
        launcher_content (str): Content of the launcher template
        qm_program_idx (int): QM program index (1: Gaussian, 2: ORCA)
    
    Returns:
        str: The executable path or command to use
    """
    import re
    
    qm_name = qm_program_details.get(qm_program_idx, {}).get('name', 'unknown')
    
    # Get default executable based on program type
    if qm_name == 'orca':
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'orca')
    elif qm_name == 'gaussian':
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'g16')
    else:
        default_exe = qm_program_details.get(qm_program_idx, {}).get('default_exe', 'unknown')
    
    if not launcher_content:
        return default_exe
    
    lines = launcher_content.split('\n')
    
    # Strategy 1: Look for existing run commands after the ### separator
    # This extracts the actual command the user intends to use
    after_separator = False
    for line in lines:
        if line.strip() == '###':
            after_separator = True
            continue
        
        if after_separator:
            # Look for command patterns like: $VAR/exe file > output or exe file > output
            # Match patterns: <executable> <input_file> [>] [output_file] [&&]
            match = re.search(r'^\s*(\S+)\s+\S+\.(inp|gjf|com)\s*(?:>|\s)', line)
            if match:
                return match.group(1)  # Return the executable part
    
    # Strategy 2: Look for any *_ROOT variable exports and construct executable path
    # This is generic and works with any naming convention (ORCA5_ROOT, GAUSSIAN_ROOT, QM_ROOT, etc.)
    root_pattern = r'export\s+([A-Z_0-9]+ROOT)\s*=\s*(.+)'
    root_var = None
    root_value = None
    
    for line in lines:
        match = re.search(root_pattern, line)
        if match:
            var_name = match.group(1)
            var_value = match.group(2).strip().strip('"\'')
            
            # For ORCA, any variable containing "ORCA" in the name
            if qm_name == 'orca' and 'ORCA' in var_name.upper():
                root_var = var_name
                root_value = var_value
                break
            # For Gaussian, look for variables containing G09, G16, or GAUSS
            elif qm_name == 'gaussian' and any(x in var_name.upper() for x in ['G09', 'G16', 'GAUSS']):
                root_var = var_name
                root_value = var_value
                # Try to detect version from variable name
                if 'G09' in var_name.upper():
                    default_exe = 'g09'
                elif 'G16' in var_name.upper():
                    default_exe = 'g16'
                break
    
    # If we found a ROOT variable, construct the executable path
    if root_var and root_value:
        if '$' in root_value or not root_value.startswith('/'):
            # Variable expansion in path, use the variable
            return f"${root_var}/{default_exe}"
        else:
            # Absolute path
            return f"{root_value}/{default_exe}"
    
    # Strategy 3: Check if executable appears to be in PATH
    # Look for any sourcing of profiles or PATH modifications
    for line in lines:
        # If we see the QM program mentioned in PATH or source commands
        if 'PATH=' in line or 'source' in line.lower() or '. ' in line:
            if qm_name == 'orca' and 'orca' in line.lower():
                return default_exe
            elif qm_name == 'gaussian':
                if 'g09' in line.lower():
                    return 'g09'
                elif 'g16' in line.lower():
                    return 'g16'
                elif 'gaussian' in line.lower():
                    return default_exe
    
    # Default fallback
    return default_exe


# --- def parse_xtb_options_from_template  (ascec-v04.py 6016-6076) ---
def parse_xtb_options_from_template(template_content: str) -> str:
    """
    Parse xTB options from a standalone xTB metadata/template block.

    The parser is intentionally permissive and only extracts the minimum
    flags needed for geometry optimization workflows.
    """
    # Strip #rescue(...) directives before keyword scanning — those describe
    # rescue-mode parameters (e.g. "#rescue(GFN2-xTB/freq)") and must not
    # leak into normal-opt flag detection (would otherwise force --hess /
    # wrong --gfn level on every standard optimization).
    template_for_keywords = re.sub(r'#\s*rescue\s*\([^)]*\)', '', template_content or '', flags=re.IGNORECASE)
    content_upper = template_for_keywords.upper()

    flags: List[str] = []

    def _normalize_xtb_opt_level(raw_level: str) -> str:
        level = re.sub(r'\s+', '', raw_level.strip().lower())
        return 'vtight' if level in ('vtight', 'verytight') else level

    if 'GFN-FF' in content_upper or 'GFNFF' in content_upper:
        flags.append('--gfnff')
    elif 'GFN0-XTB' in content_upper or 'GFN0' in content_upper:
        flags.extend(['--gfn', '0'])
    elif 'GFN1-XTB' in content_upper or 'GFN1' in content_upper:
        flags.extend(['--gfn', '1'])
    else:
        flags.extend(['--gfn', '2'])

    opt_level_match = re.search(r'\bOPT\b(?:\s+(NORMAL|TIGHT|VTIGHT|VERY\s+TIGHT))?', template_for_keywords, re.IGNORECASE)
    if opt_level_match:
        opt_level = opt_level_match.group(1)
        if opt_level:
            flags.extend(['--opt', _normalize_xtb_opt_level(opt_level)])
        else:
            flags.append('--opt')

    if 'FREQ' in content_upper or '/FREQ' in content_upper or '/NUM' in content_upper:
        flags.append('--hess')

    # Parse nprocs from template metadata (e.g. "# nprocs 4")
    nprocs_match = re.search(r'#\s*nprocs\s+(\d+)', template_content or '', re.IGNORECASE)
    if nprocs_match:
        flags.extend(['--parallel', nprocs_match.group(1)])

    # Parse maxiter/cycles from template metadata (e.g. "# maxiter 200")
    maxiter_match = re.search(r'#\s*max(?:iter|cycles?)\s+(\d+)', template_content or '', re.IGNORECASE)
    if maxiter_match:
        flags.extend(['--cycles', maxiter_match.group(1)])

    # Parse charge from template metadata (e.g. "# charge -1")
    # These are the total system values from the annealing parameters, embedded at input generation time.
    charge_match = re.search(r'#\s*charge\s+(-?\d+)', template_content or '', re.IGNORECASE)
    if charge_match:
        charge_val = int(charge_match.group(1))
        if charge_val != 0:
            flags.extend(['--chrg', str(charge_val)])

    # Parse unpaired electrons from template metadata (e.g. "# uhf 2" for triplet, mult=3)
    uhf_match = re.search(r'#\s*uhf\s+(\d+)', template_content or '', re.IGNORECASE)
    if uhf_match:
        uhf_val = int(uhf_match.group(1))
        if uhf_val > 0:
            flags.extend(['--uhf', str(uhf_val)])

    return ' '.join(flags).strip()


# --- def parse_xtb_options_from_launcher  (ascec-v04.py 6079-6096) ---
def parse_xtb_options_from_launcher(launcher_content: str) -> str:
    """Extract xTB command-line options from an existing launcher."""
    if not launcher_content:
        return '--gfn 2 --opt'

    for line in launcher_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or ' xtb ' not in f' {stripped} ':
            continue
        match = re.search(r'\bxtb\s+\S+\.xyz\s+(.*?)\s*>', stripped)
        if match:
            opts = ' '.join(match.group(1).split()).strip()
            if opts:
                # Namespace is injected by command builders; keep launcher-parsed options portable.
                opts = re.sub(r'\s--namespace(?:\s+\S+|=\S+)?', '', f' {opts}').strip()
                return opts

    return '--gfn 2 --opt'


# --- def build_xtb_runtime_options  (ascec-v04.py 6104-6139) ---
def build_xtb_runtime_options(xtb_options: str, qm_nproc: Optional[int] = None,
                              xtb_cycles: Optional[int] = None,
                              charge: Optional[int] = None,
                              multiplicity: Optional[int] = None) -> str:
    """Ensure xTB options use workflow/index parallel and cycle settings when provided.

    charge and multiplicity are the total system values from the annealing parameters
    (SystemState.charge / SystemState.multiplicity).  When provided they override any
    existing --chrg / --uhf flags so the single source-of-truth is always respected.
    """
    opts = ' '.join((xtb_options or '').split()).strip()

    # Remove pre-existing runtime controls so index/workflow settings can take precedence.
    opts = re.sub(r'\s--parallel(?:\s+\S+|=\S+)?', '', f' {opts}').strip()
    opts = re.sub(r'\s--cycles(?:\s+\S+|=\S+)?', '', f' {opts}').strip()
    opts = re.sub(r'\s--namespace(?:\s+\S+|=\S+)?', '', f' {opts}').strip()

    if qm_nproc and qm_nproc > 0:
        opts = f"{opts} --parallel {qm_nproc}".strip()

    if xtb_cycles and xtb_cycles > 0:
        opts = f"{opts} --cycles {xtb_cycles}".strip()

    # Inject charge and multiplicity from annealing parameters (total system values).
    if charge is not None:
        opts = re.sub(r'\s--chrg(?:\s+\S+|=\S+)?', '', f' {opts}').strip()
        if charge != 0:
            opts = f"{opts} --chrg {charge}".strip()

    if multiplicity is not None:
        uhf = multiplicity - 1
        opts = re.sub(r'\s--uhf(?:\s+\S+|=\S+)?', '', f' {opts}').strip()
        if uhf > 0:
            opts = f"{opts} --uhf {uhf}".strip()

    return opts


# Files a QM engine drops next to a real calculation: ORCA's optimization
# trajectory/final geometry, xtb's restart files, rescue reruns. Their names match
# the motif_*.xyz / *.out patterns the stage scans for, but they are not
# structures. Feeding a _trj.xyz back in makes ORCA parse a raw XYZ as an input
# deck and abort; counting a _trj.out inflates the cosmic input population.
_QM_BYPRODUCT_STEMS = ('_trj', '_property', '_rescue', '_engrad', '_xtbrestart', '_xtboptok')


def is_qm_byproduct(filename: str) -> bool:
    """True if filename is a QM engine byproduct rather than a real structure."""
    base = os.path.basename(filename).lower()
    if '.xtbopt.' in base or base.startswith('xtbopt'):
        return True
    stem = os.path.splitext(base)[0]
    # Match the marker as the trailing stem (motif_01_opt_trj.out) and as an
    # interior segment (motif_01_rescue.tmp.out).
    return any(stem.endswith(marker) or f'{marker}.' in base
               for marker in _QM_BYPRODUCT_STEMS)


def is_stage_output_dir(dirname: str) -> bool:
    """True for a QM stage's own output folder (geometry_optimization, geometry_refinement, ...).

    Scans for candidate structures must not descend into these: on a resumed run
    they already hold the previous pass's outputs and byproducts.
    """
    lower = os.path.basename(dirname.rstrip('/')).lower()
    return lower.startswith(('geometry_optimization', 'geometry_refinement',
                             'energy_refinement', 'geom optimization'))


# --- def calculate_input_files  (ascec-v04.py 6142-6509) ---
def calculate_input_files(template_file: str, launcher_template: Optional[str] = None,
                          auto_select: str = 'interactive', stage_type: str = "optimization",
                          workflow_mode: bool = False, qm_alias: str = "orca",
                          qm_nproc: Optional[int] = None,
                          xtb_cycles: Optional[int] = None,
                          charge: Optional[int] = None,
                          multiplicity: Optional[int] = None) -> str:
    """
    Unified function to create QM input files and launcher scripts for both
    optimization and refinement stages.
    
    Args:
        template_file (str): Path to the QM input template file.
        launcher_template (Optional[str]): Path to the launcher script template.
        auto_select (str): 'interactive', 'all', or 'combined' for file selection.
        stage_type (str): "optimization" or "refinement".
        workflow_mode (bool): True if called from a workflow, suppresses some print statements.
        
    Returns:
        str: Status message indicating success or failure.
    """
    # Determine QM program and file extension from template
    if template_file.lower().endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
        output_ext = '.out'
        qm_program_idx = 2 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    elif template_file.lower().endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
        output_ext = '.log'
        qm_program_idx = 1 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    elif template_file.lower().endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
        output_ext = '.log'
        qm_program_idx = 1 # For extract_qm_executable_from_launcher (matches qm_program_details key)
    elif template_file.lower().endswith('.xtb'):
        qm_program = 'xtb'
        input_ext = '.xyz'
        output_ext = '.out'
        qm_program_idx = 0
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA, .com/.gjf for Gaussian, or .xtb for standalone xTB metadata."
    
    # Check if template files exist
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    launcher_content = None
    if launcher_template:
        if not os.path.exists(launcher_template):
            return f"Error: Launcher template '{launcher_template}' not found."
        try:
            with open(launcher_template, 'r') as f:
                launcher_content = f.read()
        except IOError as e:
            return f"Error reading launcher template '{launcher_template}': {e}"
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Normalize accepted stage aliases.
    stage_type_aliases = {
        'opt': 'optimization',
        'optimization': 'optimization',
        'ref': 'refinement',
        'refinement': 'refinement',
    }
    stage_type = stage_type_aliases.get(stage_type.lower(), stage_type.lower())

    # Determine output directory name
    def get_next_dir(base):
        """Find the next available directory (base, base_2, etc.)"""
        if not os.path.exists(base):
            return base
        counter = 2
        while True:
            dir_name = f"{base}_{counter}"
            if not os.path.exists(dir_name):
                return dir_name
            counter += 1
    
    # Map internal stage type to filesystem directory name
    _stage_dir_names = {
        'optimization': 'geometry_optimization',
        'refinement': 'geometry_refinement',
    }
    output_dir = get_next_dir(_stage_dir_names.get(stage_type, stage_type))
    
    # File Search Logic
    xyz_files = []
    if stage_type == "optimization":
        # FIRST: Check for retry_input folder (structures from need_recalculation)
        if os.path.exists("retry_input"):
            if not workflow_mode:
                print("Found retry_input folder, using structures from previous cosmic analysis")
            for file in os.listdir("retry_input"):
                if file.endswith(".xyz") and not file.startswith("combined") and ".xtbopt." not in file:
                    xyz_files.append(os.path.join("retry_input", file))
        else:
            # Normal flow: Check for combined files in current directory
            for file in os.listdir("."):
                if (file.startswith("combined_results") or file.startswith("combined_r")) and file.endswith(".xyz"):
                    xyz_files.append(file)
            
            # Look for result_*.xyz files recursively in subdirectories
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.startswith("result_") and file.endswith(".xyz") and not file.startswith("resultbox_") and ".xtbopt." not in file:
                        xyz_files.append(os.path.join(root, file))
                        
        if not xyz_files:
            return "No result_*.xyz, combined_results.xyz, or combined_r*.xyz files found."
            
        # Sort XYZ files by annealing number
        def get_annealing_number(file_path):
            import re
            directory = os.path.dirname(file_path)
            match = re.search(r'_(\d+)$', directory)
            if match: return int(match.group(1))
            match = re.search(r'result_(\d+)', os.path.basename(file_path))
            if match: return int(match.group(1))
            return float('inf')
        xyz_files.sort(key=get_annealing_number)
        
    elif stage_type == "refinement":
        # Check for combined files
        for file in os.listdir("."):
            if file.endswith(".xyz") and "combined" in file.lower():
                xyz_files.append(file)

        # Look for motif_*.xyz files. Prune the QM stage output folders: on a
        # resumed run geometry_refinement/motif_NN/ already holds ORCA's
        # motif_NN_opt_trj.xyz, which matches motif_*.xyz but is a byproduct.
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not is_stage_output_dir(d)]
            for file in files:
                if is_qm_byproduct(file):
                    continue
                if (file.startswith("motif_") and file.endswith(".xyz")) or \
                   (file.endswith(".xyz") and "combined" in file.lower() and root != "."):
                    xyz_files.append(os.path.join(root, file))

        if not xyz_files:
            return "No files with 'combined' in name or motif_*.xyz files found."
            
    # File Selection Logic
    if stage_type == "optimization":
        if not workflow_mode:
            print(f"Found {len(xyz_files)} XYZ file(s) to process:")
            for xyz_file in xyz_files:
                print(f"  - {xyz_file}")
            
        # Store first XYZ source for protocol summary
        if hasattr(sys, '_current_workflow_context') and sys._current_workflow_context is not None:  # type: ignore[attr-defined]
            if xyz_files:
                first_file = xyz_files[0]
                if 'result' in first_file or 'annealing' in first_file.lower():
                    sys._current_workflow_context.optimization_xyz_source = "annealing"  # type: ignore[attr-defined]
                else:
                    sys._current_workflow_context.optimization_xyz_source = first_file  # type: ignore[attr-defined]
            else:
                sys._current_workflow_context.optimization_xyz_source = "annealing"  # type: ignore[attr-defined]
                
        if auto_select == 'combined':
            os.makedirs(output_dir, exist_ok=True)
            
        selected_xyz_files = interactive_xyz_file_selection(
            xyz_files,
            output_dir,
            auto_select=auto_select,
            quiet=workflow_mode,
        )
        
    elif stage_type == "refinement":
        selected_xyz_files = interactive_optimization_file_selection(xyz_files, output_dir)
    else:
        selected_xyz_files = []
        
    if not selected_xyz_files:
        return "No XYZ files selected for processing."
    xyz_files = selected_xyz_files
    
    # Create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not workflow_mode:
            print(f"Created {stage_type} directory: {output_dir}")
        
    # Process files in parallel
    all_input_files = []
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    total_xyz_files = len(xyz_files)
    use_combined_naming = (auto_select == 'combined')
    
    # Prepare arguments
    if stage_type == "optimization":
        xyz_file_args = [(xyz_file, template_content, output_dir, qm_program, input_ext, total_xyz_files, use_combined_naming) 
                         for xyz_file in xyz_files]
        process_func = _process_xyz_file_for_calc
    else: # refinement
        xyz_file_args = [(xyz_file, template_content, output_dir, qm_program, input_ext) 
                         for xyz_file in xyz_files]
        process_func = _process_xyz_file_for_opt
        
    max_workers = get_optimal_workers('mixed', len(xyz_files))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_func, args): args[0] for args in xyz_file_args}
        
        for future in as_completed(future_to_file):
            xyz_file = future_to_file[future]
            try:
                file_input_files, status_msg = future.result()
                all_input_files.extend(file_input_files)
                if not workflow_mode:
                    print(f"  {status_msg}")
                    for input_file in file_input_files:
                        print(f"    Created: {input_file}")
            except Exception as e:
                if not workflow_mode:
                    print(f"  Error processing {xyz_file}: {e}")
                    
    if not workflow_mode:
        print(f"\nCompleted processing. Created {len(all_input_files)} input files total.")
    
    if not all_input_files:
        return "No input files were created successfully."
        
    # Deduplicate input files: keep only flat filenames (basenames)
    # This prevents duplicate commands like motif_29/motif_29_opt.inp AND motif_29_opt.inp
    seen_basenames = set()
    launcher_input_files = []
    for input_file in all_input_files:
        basename = os.path.basename(input_file)
        if basename not in seen_basenames:
            seen_basenames.add(basename)
            launcher_input_files.append(basename)

    launcher_input_files.sort(key=natural_sort_key)

    # Decide launcher source: user template or auto-generated ORCA environment
    launcher_base_content = None
    launcher_generated_automatically = False

    xtb_options = parse_xtb_options_from_template(template_content) if qm_program == 'xtb' else ''
    # Ensure --opt is included for xTB optimization/refinement stages
    if qm_program == 'xtb' and '--opt' not in xtb_options:
        xtb_options = (xtb_options + ' --opt').strip()

    # Use explicit values first; in workflow mode, fall back to current workflow context.
    if qm_program == 'xtb':
        wf_ctx = getattr(sys, '_current_workflow_context', None)
        effective_qm_nproc = qm_nproc if qm_nproc is not None else getattr(wf_ctx, 'qm_nproc', None)
        effective_xtb_cycles = xtb_cycles if xtb_cycles is not None else getattr(wf_ctx, 'xtb_cycles', None)
        # Charge and multiplicity are total system properties from the annealing parameters.
        # Explicit args take priority; fall back to workflow context (SystemState).
        effective_charge = charge if charge is not None else getattr(wf_ctx, 'charge', None)
        effective_multiplicity = multiplicity if multiplicity is not None else getattr(wf_ctx, 'multiplicity', None)
        xtb_options = build_xtb_runtime_options(
            xtb_options, effective_qm_nproc, effective_xtb_cycles,
            charge=effective_charge, multiplicity=effective_multiplicity
        )

    if launcher_template and launcher_content:
        launcher_base_content = launcher_content
    elif qm_program == 'orca':
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # qm_alias on line 9 is for annealing only; ORCA stages always look
            # up the standard "orca" binary unless a launcher template is given.
            auto_launcher_tmp = create_auto_launcher(tmpdir, qm_program, "orca", quiet=workflow_mode)
            if auto_launcher_tmp and os.path.exists(auto_launcher_tmp):
                with open(auto_launcher_tmp, 'r') as f:
                    launcher_base_content = f.read()
                launcher_generated_automatically = True

    if launcher_base_content:
        launcher_path = os.path.join(output_dir, f"launcher_{qm_program}.sh")
        try:
            qm_executable = extract_qm_executable_from_launcher(launcher_base_content, qm_program_idx)
            # Ensure ORCA uses full path - fall back to detect_orca_executable if needed
            if qm_program == 'orca' and qm_executable in ('orca', qm_alias) and '/' not in qm_executable and '$' not in qm_executable:
                detected_path = detect_orca_executable(qm_alias)
                if detected_path:
                    qm_executable = detected_path
            elif qm_program == 'xtb' and qm_executable in ('unknown', ''):
                qm_executable = 'xtb'
            if not workflow_mode:
                print(f"Using QM executable: {qm_executable}")

            with open(launcher_path, 'w') as f:
                # Header - copy everything up to ### separator, skip any existing run commands
                launcher_lines = launcher_base_content.rstrip().split('\n')
                separator_found = False
                import re

                for line in launcher_lines:
                    if line.strip() == '###':
                        separator_found = True
                        f.write(line + "\n\n# Run QM using the full path\n")
                        break
                    else:
                        # Skip lines that look like run commands (generic pattern)
                        if re.search(r'\S+\s+\S+\.(inp|gjf|com|xyz)\s*(?:>|;|&&|\s*$)', line):
                            continue
                        # Skip standalone continuations
                        if line.strip() in ('&&', '&& \\', ';', '; \\', '\\'):
                            continue
                        f.write(line + "\n")

                if not separator_found:
                    f.write("\n###\n\n# Run QM using the full path\n")

                for i, input_file in enumerate(launcher_input_files):
                    output_file = input_file.replace(input_ext, output_ext)
                    if qm_program == 'gaussian':
                        cmd = f"{qm_executable} < {input_file} > {output_file}"
                    elif qm_program == 'xtb':
                        xtb_namespace = os.path.splitext(input_file)[0]
                        cmd = f"{qm_executable} {input_file} {xtb_options} --namespace {xtb_namespace} > {output_file} 2>&1"
                    else:  # orca
                        cmd = f"{qm_executable} {input_file} > {output_file}"

                    if i < len(launcher_input_files) - 1:
                        f.write(f"{cmd} ; \\\n")
                    else:
                        f.write(f"{cmd}\n")

            os.chmod(launcher_path, 0o755)

            msg = f"\nCreated {stage_type} system in '{output_dir}' directory:\n"
            msg += f"  Input files: {len(all_input_files)}\n"
            if launcher_generated_automatically:
                msg += f"  Launcher script: launcher_{qm_program}.sh (auto-generated)"
            else:
                msg += f"  Launcher script: launcher_{qm_program}.sh"
            if not workflow_mode:
                msg += f"\n\nTo run all calculations, use:\n"
                msg += f"  cd {output_dir}\n"
                msg += f"  ./launcher_{qm_program}.sh"
            return msg

        except IOError as e:
            return f"Error creating launcher script: {e}"

    if qm_program == 'xtb':
        launcher_path = os.path.join(output_dir, 'launcher_xtb.sh')
        try:
            with open(launcher_path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('set -e\n')
                # Cap BLAS threads so --parallel 1 actually means one thread per xtb.
                f.write('export OMP_NUM_THREADS=1\n')
                f.write('export MKL_NUM_THREADS=1\n')
                f.write('export OPENBLAS_NUM_THREADS=1\n\n')
                for i, input_file in enumerate(launcher_input_files):
                    output_file = input_file.replace(input_ext, output_ext)
                    xtb_namespace = os.path.splitext(input_file)[0]
                    cmd = f"xtb {input_file} {xtb_options} --namespace {xtb_namespace} > {output_file} 2>&1"
                    if i < len(launcher_input_files) - 1:
                        f.write(f"{cmd} ; \\\n")
                    else:
                        f.write(f"{cmd}\n")
            os.chmod(launcher_path, 0o755)
            return f"\nCreated {stage_type} system in '{output_dir}' directory:\n  Input files: {len(all_input_files)}\n  Launcher script: launcher_xtb.sh"
        except IOError as e:
            return f"Error creating launcher script: {e}"

    return f"Created {stage_type} system in '{output_dir}' with {len(all_input_files)} input files (no launcher)."


# --- def interactive_optimization_file_selection  (ascec-v04.py 6556-6663) ---
def interactive_optimization_file_selection(xyz_files: List[str], opt_dir: str = ".") -> List[str]:
    """
    Provides interactive selection for optimization XYZ files.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        opt_dir (str): Directory where files will be processed
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    print("\n" + "=" * 60)
    print("Optimization XYZ file selection".center(60))
    print("=" * 60)
    
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate combined files from motif/umotif files
    combined_files = [f for f in xyz_files if "combined" in os.path.basename(f).lower()]
    # Match both motif_ and umotif_ files
    motif_files = [f for f in xyz_files if ("motif_" in os.path.basename(f) or "umotif_" in os.path.basename(f)) and "combined" not in os.path.basename(f).lower()]
    
    # Sort motif/umotif files by number
    def extract_motif_number(filepath):
        import re
        filename = os.path.basename(filepath)
        # Match both motif_XX and umotif_XX patterns
        match = re.search(r'u?motif_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    motif_files.sort(key=extract_motif_number)
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 0
    
    # First show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Then show motif files
    if motif_files:
        print("\nMotif files:")
        for xyz_file in motif_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(xyz_files) > 1:
        options["a"] = "All files"
        print(f"a. Process all files ({len(xyz_files)} files total)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Optimization system creation cancelled.")
                return []
            
            if choice.lower() == 'a':
                print(f"Selected: All files ({len(xyz_files)} files)")
                return xyz_files
            
            # Handle numbered selections (single or multiple)
            try:
                choices = choice.split()
                selected_files = []
                
                for ch in choices:
                    if ch in options and ch not in ['a', 'q']:
                        selected_files.append(options[ch])
                    else:
                        print(f"Invalid choice: {ch}")
                        break
                else:
                    if selected_files:
                        print("Selected %s file(s):" % len(selected_files))
                        for f in selected_files:
                            print(f"  - {f}")
                        return selected_files
                    else:
                        print("No valid files selected.")
                        
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all files.")
                
        except KeyboardInterrupt:
            print("\nOptimization system creation cancelled by user.")
            return []
        except EOFError:
            print("\nOptimization system creation cancelled.")
            return []


# --- def merge_xyz_files  (ascec-v04.py 6831-6938) ---
def merge_xyz_files(xyz_files: List[str], output_filename: str, quiet: bool = False) -> bool:
    """
    Merge a list of XYZ files into a single output file.
    Renumbers configurations sequentially and adds source file information.
    
    Args:
        xyz_files (List[str]): List of XYZ file paths to merge
        output_filename (str): Name of the output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not xyz_files:
        return False
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Sort files by the number in the filename for consistent ordering
        def get_sort_key(filepath):
            filename = os.path.basename(filepath)
            # Try multiple patterns to extract configuration numbers
            patterns = [
                r'result_(\d+)\.\w+',           # result_123.xyz
                r'conf_(\d+)\.\w+',             # conf_20.xyz
                r'opt\d+_conf_(\d+)\.\w+',      # opt1_conf_20.xyz
                r'_(\d+)\.\w+',                 # any_123.xyz (general pattern)
                r'(\d+)\.\w+'                   # 123.xyz (number only)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
            
            # If no pattern matches, sort alphabetically by filename
            return (float('inf'), filename)
        
        sorted_files = sorted(xyz_files, key=get_sort_key)
        
        total_configs = 0
        
        with open(output_filename, 'w') as outfile:
            for file_idx, xyz_file in enumerate(sorted_files):
                try:
                    # Extract configurations from this file
                    configurations = extract_configurations_from_xyz(xyz_file)
                    
                    if not configurations:
                        print(f"Warning: No configurations found in {xyz_file}")
                        continue
                    
                    source_name = os.path.basename(xyz_file).replace('.xyz', '')
                    
                    for config in configurations:
                        total_configs += 1
                        
                        # Write atom count
                        outfile.write(f"{len(config['atoms'])}\n")
                        
                        # Parse original comment to extract energy
                        original_comment = config['comment']
                        energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
                        
                        energy = energy_match.group(1) if energy_match else "unknown"
                        
                        # Create new comment with sequential numbering and source info (no temperature, no "Source:" label)
                        if energy == "unknown":
                            new_comment = f"Configuration: {total_configs} | {source_name}"
                        else:
                            new_comment = f"Configuration: {total_configs} | E = {energy} a.u. | {source_name}"
                        
                        outfile.write(f"{new_comment}\n")
                        
                        # Write atoms (preserve original coordinate precision with proper column alignment)
                        for atom in config['atoms']:
                            if len(atom) == 7:  # New format with string coordinates
                                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                                # Right-align coordinates to maintain column alignment while preserving original precision
                                outfile.write(f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n")
                            else:  # Old format compatibility (fallback to 6 decimal places)
                                symbol, x, y, z = atom
                                outfile.write(f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n")
                        
                except IOError as e:
                    print(f"Warning: Could not read {xyz_file}: {e}")
                    continue
        
        if not quiet:
            print(f"Merged {len(sorted_files)} files into {output_filename} with {total_configs} configurations")
        
        # Create .mol file if obabel is available
        if shutil.which("obabel"):
            success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
            if success:
                if not quiet:
                    print(f"Also created {output_filename.replace('.xyz', '.mol')}")
            else:
                if not quiet:
                    print(f"Warning: Could not create .mol file: {error_msg}")
        
        return True
        
    except IOError as e:
        print(f"Error writing to {output_filename}: {e}")
        return False


# --- def extract_base  (ascec-v04.py 7104-7122) ---
def extract_base(filename):
    """Extract base name from filename by removing extension and known suffixes."""
    # Define optional suffixes that can appear before extensions
    # Note: _rescue must come before _inp/_out to handle _rescue.inp properly
    KNOWN_SUFFIXES = ['_trj', '_opt', '_property', '_gu', '_xtbrestart', '_xtboptok', '_engrad', '_xyz', '_out', '_inp', '_tmp', '_rescue']

    # Remove extension; handle dot-prefixed (hidden) files
    working = filename.lstrip('.') if filename.startswith('.') else filename
    name, *_ = working.split('.', 1)
    # Remove all known suffixes (not just the last one) by repeatedly checking
    changed = True
    while changed:
        changed = False
        for suffix in KNOWN_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break  # Start over to handle multiple suffixes
    return name


# --- def get_sort_key  (ascec-v04.py 7231-7258) ---
def get_sort_key(filename):
    """Extract the configuration number from filename for sorting.
    
    Handles various patterns:
    - umotif_28_opt.xyz -> 28
    - motif_28_opt.xyz -> 28
    - opt_conf_3.xyz -> 3
    - result_123.xyz -> 123
    - any_456.xyz -> 456
    """
    import re
    # Try multiple patterns in order of specificity
    patterns = [
        r'u?motif_(\d+)_opt\.\w+',   # umotif_28_opt.xyz or motif_28_opt.xtboptok
        r'u?motif_(\d+)\.\w+',        # umotif_28.xyz or motif_28.xyz
        r'opt_conf_(\d+)\.\w+',      # opt_conf_3.xyz or opt_conf_1.xtboptok
        r'result_(\d+)\.\w+',        # result_123.xyz
        r'conf_(\d+)\.\w+',          # conf_20.xyz
        r'_(\d+)\.\w+',              # any_123.xyz (general pattern)
        r'(\d+)\.\w+'                # 123.xyz (number only)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return float('inf')


# --- def combine_xyz_files  (ascec-v04.py 7263-7305) ---
def combine_xyz_files(output_filename="combined_results.xyz", exclude_pattern="_trj.xyz"):
    """Combine all relevant .xyz files into a single .xyz file."""
    
    all_xyz_files = []
    for root, _, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            if not filepath.endswith(".xyz"):
                continue
            # Never merge existing aggregate files into combined_results.
            if file == output_filename or file.startswith("combined_r") or file.startswith("combined_results"):
                continue
            if exclude_pattern in file:
                continue
            if ".xtbopt." in file:
                continue
            all_xyz_files.append(filepath)

    if not all_xyz_files:
        print(f"No relevant .xyz files found (excluding '{exclude_pattern}' and '{output_filename}').")
        return False

    # Sort the files based on the first configuration number found
    sorted_xyz_files = sorted(all_xyz_files, key=lambda x: get_sort_key(os.path.basename(x)))

    with open(output_filename, "w") as outfile:
        for xyz_file in sorted_xyz_files:
            print(f"Processing: {xyz_file}")
            with open(xyz_file, "r") as infile:
                lines = infile.readlines()
                outfile.writelines(lines)

    print(f"\nSuccessfully combined {len(sorted_xyz_files)}.xyz files into: {output_filename}")
    
    # Create .mol file if obabel is available
    if shutil.which("obabel"):
        success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
        if success:
            print(f"Also created {output_filename.replace('.xyz', '.mol')}")
        else:
            print(f"Warning: Could not create .mol file: {error_msg}")
    
    return True


# --- def convert_xyz_to_mol_simple  (ascec-v04.py 7308-7328) ---
def convert_xyz_to_mol_simple(input_xyz, output_mol):
    """Convert an XYZ file to a MOL file using Open Babel."""
    try:
        # Make paths absolute
        input_xyz = os.path.abspath(input_xyz)
        output_mol = os.path.abspath(output_mol)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_mol)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        command = ["obabel", "-i", "xyz", input_xyz, "-o", "mol", "-O", output_mol]
        subprocess.run(command, check=True, capture_output=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Error converting {input_xyz}: {e.stderr.decode()}"
    except FileNotFoundError:
        return False, "Error: Open Babel ('obabel') command not found. Make sure it's installed and in your system's PATH."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


# --- def create_combined_mol  (ascec-v04.py 7330-7342) ---
def create_combined_mol():
    """Create MOL file from combined_results.xyz."""
    if os.path.exists("combined_results.xyz"):
        success, error_message = convert_xyz_to_mol_simple("combined_results.xyz", "combined_results.mol")
        if success:
            print("Successfully created combined_results.mol")
            return True
        else:
            print(f"Failed to create MOL file: {error_message}")
            return False
    else:
        print("No combined_results.xyz file found to convert.")
        return False


# --- def detect_orca_version  (ascec-v04.py 7349-7375) ---
def detect_orca_version(logfile_path: str) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from output file.
    
    Args:
        logfile_path: Path to ORCA output file (.out)
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Scan header section until we find version or hit input file section
            # This is robust to header size changes in future ORCA versions
            for i, line in enumerate(f):
                # Stop if we've reached the input file section (version should be before this)
                if 'INPUT FILE' in line or i > 500:
                    break
                # Look for version pattern like "Program Version 6.1.0" or "Program Version 7.0.0"
                match = re.search(r'Program Version\s+(\d+)\.(\d+)', line, re.IGNORECASE)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return (major, minor)
    except Exception:
        pass
    return None


# --- def detect_orca_version_from_executable  (ascec-v04.py 7539-7571) ---
def detect_orca_version_from_executable(orca_path: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from the executable by running 'orca --version'.
    
    Args:
        orca_path: Path to ORCA executable. If None, uses 'orca' from PATH.
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    if orca_path is None:
        # Try to find ORCA in PATH
        orca_path = shutil.which("orca")
        if not orca_path:
            return None
    
    try:
        result = subprocess.run(
            [orca_path, "--version"],
            capture_output=True,
            text=True
        )
        # Parse version from output (works for both stdout and stderr)
        output = result.stdout + result.stderr
        # Look for patterns like "Program Version 6.1.0" or just "6.1.0"
        match = re.search(r'(?:Program Version\s+)?(\d+)\.(\d+)', output, re.IGNORECASE)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return (major, minor)
    except (FileNotFoundError, Exception):
        pass
    return None


# --- def detect_orca_version_from_launcher  (ascec-v04.py 7574-7615) ---
def detect_orca_version_from_launcher(launcher_content: str) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version from launcher variable names like ORCA611_ROOT.
    
    This is a fallback when the ORCA executable is not yet in PATH.
    Parses variable names following the pattern ORCA{major}{minor}[patch]_ROOT.
    
    Examples:
        ORCA611_ROOT -> (6, 1)  # ORCA 6.1.1
        ORCA61_ROOT  -> (6, 1)  # ORCA 6.1
        ORCA6_ROOT   -> (6, 0)  # ORCA 6.x (minor assumed 0)
        ORCA5_ROOT   -> (5, 0)  # ORCA 5.x
        ORCA_ROOT    -> None    # No version info
        
    Args:
        launcher_content: Content of the launcher script
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    if not launcher_content:
        return None
    
    # Look for ORCA version variable patterns: ORCA611_ROOT, ORCA61_ROOT, ORCA6_ROOT
    # Pattern: ORCA followed by digits, then _ROOT
    pattern = r'ORCA(\d)(\d*)_ROOT'
    
    for line in launcher_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            major = int(match.group(1))
            minor_str = match.group(2)
            if minor_str:
                # First digit after major is the minor version
                # e.g., ORCA611_ROOT: major=6, minor_str="11" -> minor=1
                # e.g., ORCA61_ROOT: major=6, minor_str="1" -> minor=1
                minor = int(minor_str[0])
            else:
                minor = 0
            return (major, minor)
    
    return None


# --- def detect_orca_version_combined  (ascec-v04.py 7618-7643) ---
def detect_orca_version_combined(orca_path: Optional[str] = None, 
                                  launcher_content: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Detect ORCA version using multiple methods:
    1. From executable (orca --version)
    2. From launcher variable names (ORCA611_ROOT)
    
    Args:
        orca_path: Path to ORCA executable (optional)
        launcher_content: Content of the launcher script (optional)
        
    Returns:
        Tuple of (major, minor) version numbers, or None if not detected
    """
    # Try executable first (most reliable)
    version = detect_orca_version_from_executable(orca_path)
    if version:
        return version
    
    # Fall back to launcher variable parsing
    if launcher_content:
        version = detect_orca_version_from_launcher(launcher_content)
        if version:
            return version
    
    return None


# --- const XTB_SYNONYMS  (ascec-v04.py 7647-7663) ---
XTB_SYNONYMS = {
    # Native xTB (ORCA 6.1+)
    'NATIVE-GFN-XTB': 'NATIVE-GFN-XTB',
    'NATIVE-GFN1-XTB': 'NATIVE-GFN-XTB',
    'NATIVE-XTB1': 'NATIVE-GFN-XTB',
    'NATIVE-GFN2-XTB': 'NATIVE-GFN2-XTB',
    'NATIVE-XTB2': 'NATIVE-GFN2-XTB',
    # Non-native xTB (ORCA 5.x, requires external xtb)
    'GFN0-XTB': 'GFN0-XTB',
    'XTB0': 'GFN0-XTB',
    'GFN-XTB': 'GFN-XTB',
    'XTB1': 'GFN-XTB',
    'GFN2-XTB': 'GFN2-XTB',
    'XTB2': 'GFN2-XTB',
    'GFN-FF': 'GFN-FF',
    'XTBFF': 'GFN-FF',
}


# --- const XTB_NATIVE_MAP  (ascec-v04.py 7666-7671) ---
XTB_NATIVE_MAP = {
    'GFN-XTB': 'Native-GFN-xTB',
    'XTB1': 'Native-GFN-xTB',
    'GFN2-XTB': 'Native-GFN2-xTB',
    'XTB2': 'Native-GFN2-xTB',
}


# --- def is_xtb_method  (ascec-v04.py 7674-7685) ---
def is_xtb_method(method: str) -> bool:
    """
    Check if a method string is an xTB method.
    
    Args:
        method: Method string (e.g., "Native-GFN2-xTB", "GFN2-xTB", "XTB2")
        
    Returns:
        True if it's an xTB method, False otherwise
    """
    method_upper = method.upper().replace('_', '-')
    return method_upper in XTB_SYNONYMS or any(kw in method_upper for kw in ['GFN', 'XTB'])


# --- def convert_xtb_for_orca_version  (ascec-v04.py 7688-7727) ---
def convert_xtb_for_orca_version(method: str, orca_path: Optional[str] = None, 
                                   launcher_content: Optional[str] = None) -> str:
    """
    Convert xTB method to the appropriate version for the detected ORCA version.
    
    - ORCA 6.1+: Convert to native xTB methods (Native-GFN-xTB, Native-GFN2-xTB)
    - ORCA 5.x: Keep non-native xTB methods (GFN-xTB, GFN2-xTB)
    
    Args:
        method: xTB method string
        orca_path: Path to ORCA executable (for version detection)
        launcher_content: Launcher script content (for version detection from ORCA611_ROOT etc.)
        
    Returns:
        Appropriate xTB method for the ORCA version
    """
    method_upper = method.upper().replace('_', '-')
    
    # Detect ORCA version (tries executable first, then launcher variable names)
    version = detect_orca_version_combined(orca_path, launcher_content)
    is_orca_61_plus = version and version[0] >= 6 and version[1] >= 1
    
    # If already native and ORCA 6.1+, keep it
    if 'NATIVE' in method_upper and is_orca_61_plus:
        return method
    
    # If native but ORCA < 6.1, convert to non-native
    if 'NATIVE' in method_upper and not is_orca_61_plus:
        if 'GFN2' in method_upper or 'XTB2' in method_upper:
            return 'GFN2-xTB'
        elif 'GFN' in method_upper or 'XTB1' in method_upper:
            return 'GFN-xTB'
        return 'GFN2-xTB'  # Default to GFN2
    
    # If non-native and ORCA 6.1+, convert to native
    if method_upper in XTB_NATIVE_MAP and is_orca_61_plus:
        return XTB_NATIVE_MAP[method_upper]
    
    # Return as-is
    return method


# --- def detect_xtb_in_template  (ascec-v04.py 7730-7754) ---
def detect_xtb_in_template(template_content: str) -> Optional[str]:
    """
    Detect if the template uses an xTB method in the ! line.
    
    Args:
        template_content: Content of the ORCA input template
        
    Returns:
        The xTB method found (canonical form), or None if not xTB
    """
    # Look for the ! line(s) in the template
    for line in template_content.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('!'):
            # Extract all tokens from the ! line
            tokens = line_stripped[1:].split()
            for token in tokens:
                token_upper = token.upper().replace('_', '-')
                # Check if this token is an xTB method
                if token_upper in XTB_SYNONYMS:
                    return token  # Return original case
                # Also check for partial matches
                if any(kw in token_upper for kw in ['GFN', 'NATIVE-GFN', 'NATIVE-XTB']):
                    return token
    return None


# --- def detect_orca_executable  (ascec-v04.py 7757-7808) ---
def detect_orca_executable(alias: str = "orca") -> Optional[str]:
    """
    Detect ORCA executable path using 'which <alias>' or shutil.which.

    Args:
        alias: The command alias to search for (default: "orca")
               This can come from line 9 of input.in (e.g., "orca", "orca6", etc.)

    Returns:
        Full path to ORCA executable, or None if not found
    """
    # The alias on line 9 of input.in describes the *main* QM program. When the
    # main program isn't ORCA (e.g. "xtb", "g16", "g09", "gaussian"), the alias
    # must not be used to locate ORCA — it would resolve to the wrong binary
    # and silently produce empty .out files. Fall back to the standard "orca".
    non_orca_aliases = {"xtb", "g16", "g09", "g03", "gaussian"}
    candidates = []
    if alias and alias.lower() not in non_orca_aliases:
        candidates.append(alias)
    if "orca" not in candidates:
        candidates.append("orca")

    for cand in candidates:
        # Try shutil.which first (works cross-platform)
        orca_path = shutil.which(cand)
        if not orca_path:
            # Fallback: try running 'which <alias>' in shell
            try:
                result = subprocess.run(
                    ["which", cand],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    candidate_path = result.stdout.strip()
                    if os.path.exists(candidate_path):
                        orca_path = candidate_path
            except (FileNotFoundError, Exception):
                pass

        if not orca_path:
            continue

        # Validate the resolved binary actually looks like ORCA. The directory
        # name is checked too because some sites install orca next to other
        # tools and the basename alone could be ambiguous.
        bn = os.path.basename(orca_path).lower()
        parent = os.path.basename(os.path.dirname(orca_path)).lower()
        if "orca" in bn or "orca" in parent:
            return orca_path

    return None


# --- def create_auto_launcher  (ascec-v04.py 7811-7936) ---
def create_auto_launcher(output_dir: str, qm_program: str = "orca", orca_alias: str = "orca",
                         quiet: bool = False) -> Optional[str]:
    """
    Create an auto-generated launcher script when no launcher file is provided.
    
    This function detects the ORCA installation path using the alias from input.in
    and creates a proper launcher script with OpenMPI environment setup.
    
    Args:
        output_dir: Directory to create the launcher script in
        qm_program: QM program type ('orca' or 'gaussian')
        orca_alias: The ORCA executable alias from input.in line 9 (default: "orca")
        
    Returns:
        Path to the created launcher script, or None if ORCA not found
    """
    if qm_program != 'orca':
        if not quiet:
            print(f"  Warning: Auto-launcher only supported for ORCA, not {qm_program}")
        return None
    
    # Detect ORCA executable using the alias. detect_orca_executable() ignores
    # non-ORCA aliases (e.g. "xtb") to avoid pointing the launcher at the wrong
    # binary, so the message below names what was actually searched for.
    orca_path = detect_orca_executable(orca_alias)
    if not orca_path:
        if not quiet:
            non_orca_aliases = {"xtb", "g16", "g09", "g03", "gaussian"}
            searched = "orca" if (not orca_alias or orca_alias.lower() in non_orca_aliases) else f"{orca_alias}/orca"
            print(f"  Warning: ORCA executable '{searched}' not found in PATH. Please provide a launcher script.")
        return None
    
    if not quiet:
        print(f"  Auto-detected ORCA at: {orca_path}")
    
    # Create launcher script
    launcher_path = os.path.join(output_dir, "launcher_orca.sh")
    
    # Get ORCA directory (for parallel execution, ORCA needs full path)
    orca_dir = os.path.dirname(orca_path)
    
    # Try to detect OpenMPI directory (commonly in same parent as ORCA or in software folder)
    orca_base = os.path.dirname(orca_dir)  # e.g., /home/user/software
    
    # Look for OpenMPI installation in common locations
    openmpi_dir = None
    openmpi_search_patterns = [
        os.path.join(orca_base, "openmpi*"),
        os.path.join(orca_base, "OpenMPI*"),
        "/usr/lib/openmpi*",
        "/usr/local/lib/openmpi*",
    ]
    for pattern in openmpi_search_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Sort and take the latest version
            matches.sort(reverse=True)
            for match in matches:
                if os.path.isdir(match):
                    openmpi_dir = match
                    break
            if openmpi_dir:
                break
    
    # Try to detect ORCA version from directory name (e.g., orca_6_1_1 -> 611)
    orca_version_str = ""
    orca_dir_name = os.path.basename(orca_dir)
    version_match = re.search(r'orca[_-]?(\d+)[_.]?(\d+)?[_.]?(\d+)?', orca_dir_name, re.IGNORECASE)
    if version_match:
        major = version_match.group(1) or ""
        minor = version_match.group(2) or ""
        patch = version_match.group(3) or ""
        orca_version_str = f"{major}{minor}{patch}"  # e.g., "611"
    
    # Build the launcher content using the default template structure
    if openmpi_dir:
        openmpi_version = os.path.basename(openmpi_dir)
        launcher_content = f'''#!/bin/bash
# Auto-generated launcher script for ORCA
# Created by ASCEC

# Define the paths to your ORCA installation
export ORCA_BASE="{orca_base}"
export ORCA{orca_version_str}_ROOT="{orca_dir}"
export OPENMPI_ROOT="{openmpi_dir}"

# Save system paths to prevent infinite nesting if sourced multiple times
_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Prepend ORCA and OpenMPI to the system paths
export PATH="$ORCA{orca_version_str}_ROOT:$OPENMPI_ROOT/bin:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA{orca_version_str}_ROOT:$OPENMPI_ROOT/lib:$_SYSTEM_LD_LIBRARY_PATH"

###
'''
    else:
        # No OpenMPI found - create simpler launcher
        launcher_content = f'''#!/bin/bash
# Auto-generated launcher script for ORCA
# Created by ASCEC

# Define the paths to your ORCA installation
export ORCA_BASE="{orca_base}"
export ORCA{orca_version_str}_ROOT="{orca_dir}"

# Save system paths to prevent infinite nesting if sourced multiple times
_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Prepend ORCA to the system paths
export PATH="$ORCA{orca_version_str}_ROOT:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA{orca_version_str}_ROOT:$_SYSTEM_LD_LIBRARY_PATH"

###
'''
    
    try:
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        os.chmod(launcher_path, 0o755)
        return launcher_path
    except Exception as e:
        if not quiet:
            print(f"  Warning: Could not create launcher script: {e}")
        return None


# --- def resolve_orca_executable_from_launcher  (ascec-v04.py 8647-8688) ---
def resolve_orca_executable_from_launcher(launcher_content: str, qm_alias: str = "orca") -> str:
    """
    Resolve the full ORCA executable path from launcher content.

    Tries multiple strategies:
    1. Extract ORCA_ROOT variable value from launcher and construct full path
    2. Use detect_orca_executable() to find the actual binary
    3. Fall back to the alias

    Args:
        launcher_content: Content of the launcher script
        qm_alias: ORCA alias from input file (default: "orca")

    Returns:
        Full path to ORCA executable, or shell variable reference, or alias as last resort
    """
    # Strategy 1: Extract ORCA_ROOT value from launcher to get absolute path
    root_pattern = r'^export\s+(ORCA[A-Z0-9_]*ROOT)\s*=\s*["\']?([^"\'\s]+)["\']?'
    for line in launcher_content.split('\n'):
        line = line.strip()
        match = re.match(root_pattern, line)
        if match:
            var_name = match.group(1)
            var_value = match.group(2)
            if var_value.startswith('/') and '$' not in var_value:
                # Absolute path - use it directly
                full_path = os.path.join(var_value, "orca")
                if os.path.exists(full_path):
                    return full_path
                # Path from launcher doesn't exist on disk, use variable reference
                return f"${{{var_name}}}/orca"
            else:
                # Variable reference or relative path
                return f"${{{var_name}}}/orca"

    # Strategy 2: Detect ORCA executable from system PATH
    detected = detect_orca_executable(qm_alias)
    if detected:
        return detected

    # Strategy 3: Fall back to alias
    return qm_alias


# --- def detect_output_file_type  (ascec-v04.py 8763-8782) ---
def detect_output_file_type(filepath):
    """Detect whether an output file is from ORCA, Gaussian, or xTB.

    Returns:
        'orca', 'gaussian', 'xtb', or None if not detected
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                if 'O   R   C   A' in line:
                    return 'orca'
                if 'Gaussian' in line:
                    return 'gaussian'
                if 'x T B' in line or 'xtb version' in line.lower():
                    return 'xtb'
    except Exception:
        pass
    return None


def parse_orca_output(filepath):
    """Parse an ORCA output file to extract energy, cycles, convergence, run time."""
    results = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    orca_signatures = [
        "O   R   C   A",
        "ORCA - Electronic Structure Program",
        "Program Version 5.",
        "Program Version 6.",
    ]
    if not any(sig in content for sig in orca_signatures):
        print(f"File {filepath} is not an ORCA output file.")
        return None

    results['input_file'] = os.path.splitext(os.path.basename(filepath))[0]

    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY:?\s*([-+]?\d+\.\d+)", content)
    if energy_matches:
        optimization_done_index = content.find("*** OPTIMIZATION RUN DONE ***")
        if optimization_done_index != -1:
            last_valid_energy_index = -1
            for i, _match in enumerate(energy_matches):
                match_index = content.find(r"FINAL SINGLE POINT ENERGY\s*(-?\d+\.\d+)", 0)
                if match_index < optimization_done_index:
                    last_valid_energy_index = i
            results['energy'] = float(energy_matches[last_valid_energy_index]) if last_valid_energy_index != -1 else None
        else:
            results['energy'] = float(energy_matches[-1])
    else:
        results['energy'] = None

    cycles_match = re.search(r"\(AFTER\s+(\d+)\s+CYCLES\)", content)
    results['cycles'] = int(cycles_match.group(1)) if cycles_match else None

    # An energy-refinement single point (e.g. DLPNO-CCSD(T)) runs no geometry
    # optimization, so "THE OPTIMIZATION HAS CONVERGED" is never printed and it
    # has no opt cycles.  Judging such a job by that string wrongly flags it as
    # non-converged.  Only apply the optimization criterion to actual opt jobs;
    # for single points, convergence == ORCA terminating normally.
    is_optimization = (
        "GEOMETRY OPTIMIZATION CYCLE" in content
        or "*** OPTIMIZATION RUN DONE ***" in content
    )
    if is_optimization:
        results['converged'] = "THE OPTIMIZATION HAS CONVERGED" in content
    else:
        results['converged'] = "ORCA TERMINATED NORMALLY" in content

    time_match = re.search(r"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds\s*(\d+)\s*msec", content)
    if time_match:
        d, h, m, s, ms = (int(time_match.group(i)) for i in range(1, 6))
        results['time'] = d * 86400 + h * 3600 + m * 60 + s + ms / 1000.0
    else:
        results['time'] = None
    return results


def parse_xtb_output(filepath):
    """Parse an xTB output file to extract energy, cycles, convergence, wall time."""
    results = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    if 'x T B' not in content and 'xtb version' not in content.lower():
        print(f"File {filepath} is not an xTB output file.")
        return None

    results['input_file'] = os.path.splitext(os.path.basename(filepath))[0]

    energy_matches = re.findall(r'TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh', content)
    results['energy'] = float(energy_matches[-1]) if energy_matches else None

    if 'optimized geometry written to' in content.lower() or 'GEOMETRY OPTIMIZATION CONVERGED' in content:
        results['converged'] = True
    elif 'FAILED TO CONVERGE' in content:
        results['converged'] = False
    else:
        results['converged'] = None

    results['cycles'] = None
    cycle_patterns = [
        r'GEOMETRY\s+OPTIMIZATION\s+CONVERGED[^\n]*?(?:AFTER|IN)\s+(\d+)\s+(?:ITERATIONS?|CYCLES?|STEPS?)',
        r'optimized\s+geometry\s+written\s+to[^\n]*?after\s+(\d+)\s+(?:iterations?|cycles?|steps?)',
        r'(?:total\s+number\s+of|number\s+of)\s+(?:optimization\s+)?cycles?\s*[:=]\s*(\d+)',
        r'optimization\s+(?:took|finished|converged)[^\n]{0,40}?(\d+)\s+(?:iterations?|cycles?|steps?)',
        r'ANC[^\n]{0,80}?(?:steps?|cycles?)\s*[:=]\s*(\d+)',
    ]
    for pattern in cycle_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                results['cycles'] = int(matches[-1])
                break
            except (TypeError, ValueError):
                continue

    if results['cycles'] is None:
        step_matches = re.findall(
            r'^\s*(\d+)\s+[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?\s+[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?',
            content, re.MULTILINE,
        )
        if step_matches:
            try:
                results['cycles'] = int(step_matches[-1])
            except (TypeError, ValueError):
                results['cycles'] = None

    time_match = re.search(
        r'total:\s*\n\s*\*\s*wall-time:\s*(\d+)\s*d,\s*(\d+)\s*h,\s*(\d+)\s*min,\s*([\d.]+)\s*sec',
        content,
    )
    if time_match:
        d, h, m = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3))
        s = float(time_match.group(4))
        results['time'] = d * 86400 + h * 3600 + m * 60 + s
    else:
        results['time'] = None

    return results


def parse_gaussian_output(filepath):
    """Parse a Gaussian .log file for energy, cycles, convergence, wall-equivalent time."""
    results = {'input_file': os.path.splitext(os.path.basename(filepath))[0]}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    energy_matches = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*(-?\d+\.\d+)", content)
    results['energy'] = float(energy_matches[-1]) if energy_matches else None

    nproc = 1
    nproc_match = re.search(r"Will use up to\s+(\d+)\s+processors", content)
    if nproc_match:
        nproc = int(nproc_match.group(1))
    else:
        nproc_match = re.search(r"%NProcShared\s*=\s*(\d+)", content, re.IGNORECASE)
        if nproc_match:
            nproc = int(nproc_match.group(1))

    cycles_match = re.search(r"Step number\s+(\d+)\s+out of", content)
    if cycles_match:
        results['cycles'] = int(cycles_match.group(1))
    else:
        opt_steps = re.findall(r"Step number\s+(\d+)", content)
        results['cycles'] = int(opt_steps[-1]) if opt_steps else None

    results['converged'] = "Stationary point found" in content or "Optimization completed" in content

    time_matches = re.findall(r"Job cpu time:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+\.\d+)\s*seconds", content)
    if time_matches:
        total_cpu_time = 0.0
        for tm in time_matches:
            d, h, m, s = int(tm[0]), int(tm[1]), int(tm[2]), float(tm[3])
            total_cpu_time += d * 86400 + h * 3600 + m * 60 + s
        results['time'] = total_cpu_time / nproc
    else:
        results['time'] = None

    return results


# --- def format_time_summary  (ascec-v04.py 8868-8879) ---
def format_time_summary(seconds, include_days=False):
    """Format time for summary output."""
    if include_days:
        days, rem = divmod(seconds, 24 * 3600)
        hours, rem = divmod(rem, 3600)
        minutes, sec = divmod(rem, 60)
        return f"{int(days)} days, {int(hours)}:{int(minutes)}:{sec:.3f}"
    else:
        # Convert all time to hours (even if >24), minutes, seconds
        total_hours, rem = divmod(seconds, 3600)
        minutes, sec = divmod(rem, 60)
        return f"{int(total_hours)}:{int(minutes)}:{sec:.3f}"


# --- def format_total_time  (ascec-v04.py 8881-8885) ---
def format_total_time(seconds):
    """Format the total execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"


# --- def format_mean_time  (ascec-v04.py 8887-8891) ---
def format_mean_time(seconds):
    """Format mean execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"


# --- def format_wall_time  (ascec-v04.py 8893-8946) ---
def format_wall_time(seconds):
    """Format the wall time showing only the two most significant time units."""
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    
    # Build time string showing only the two most significant units
    time_parts = []
    
    # Check for weeks and days
    if days >= 7:
        weeks, remaining_days = divmod(days, 7)
        time_parts.append(f"{int(weeks)} week{'s' if weeks != 1 else ''}")
        if remaining_days > 0:
            time_parts.append(f"{int(remaining_days)} day{'s' if remaining_days != 1 else ''}")
        elif hours > 0:
            # If no remaining days but we have hours, show hours
            time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        elif minutes > 0:
            # If no days or hours but we have minutes, show minutes
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    elif days > 0:
        # Days and hours/minutes
        time_parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
        if hours > 0:
            time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        elif minutes > 0:
            # If no hours but we have minutes, show minutes
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    elif hours > 0:
        # Hours and minutes
        time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
        elif sec > 0:
            # If no minutes but we have seconds, show seconds
            time_parts.append(f"{int(sec)} second{'s' if sec != 1 else ''}")
    elif minutes > 0:
        # Minutes and seconds
        time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
        if sec > 0:
            time_parts.append(f"{int(sec)} second{'s' if sec != 1 else ''}")
        # Note: For sub-second precision, we could add milliseconds here if needed
    else:
        # Just seconds (including fractional seconds)
        if sec >= 1:
            time_parts.append(f"{sec:.1f} second{'s' if sec != 1 else ''}")
        else:
            # Show milliseconds for very short times
            milliseconds = sec * 1000
            time_parts.append(f"{milliseconds:.0f} millisecond{'s' if milliseconds != 1 else ''}")
    
    # Return only the first two parts (most significant)
    return ", ".join(time_parts[:2])


# --- def summarize_calculations  (ascec-v04.py 8948-9103) ---
def summarize_calculations(directory=".", file_types=None, actual_wall_time=None):
    """Create summary of calculations for ORCA (.out), Gaussian (.log), and/or xTB (.out) files."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if file_types is None:
        file_types = ['orca', 'gaussian']  # Default: process both types

    results_by_type = {}

    # Process each requested file type
    for file_type in file_types:
        if file_type == 'orca':
            summary_file = "orca_summary.txt"
            file_extension = ".out"
            parse_function = parse_orca_output
        elif file_type == 'gaussian':
            summary_file = "gaussian_summary.txt"
            file_extension = ".log"
            parse_function = parse_gaussian_output
        elif file_type == 'xtb':
            summary_file = "xtb_summary.txt"
            file_extension = ".out"
            parse_function = parse_xtb_output
        else:
            continue
        
        job_summaries = []
        all_results = {
            'job_count': 0,
            'total_time': 0,
            'min_time': None,
            'max_time': None,
            'total_cycles': 0,
            'min_cycles': None,
            'max_cycles': None,
            'cycles_count': 0,  # Count of jobs with cycle data
            'non_converged': 0,  # Count of non-converged optimizations
        }

        # Find files of this type (exclude rescue files like _rescue.out, _rescue.log)
        found_files = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(file_extension):
                    # Skip rescue files - they are intermediate files from redo workflow
                    if '_rescue' in filename:
                        continue
                    # Skip aggregate combined files (combined_results.out, combined_r*.out)
                    if filename.startswith('combined_results') or filename.startswith('combined_r'):
                        continue
                    filepath = os.path.join(root, filename)
                    # For .out files, disambiguate ORCA vs xTB by content
                    if file_extension == '.out':
                        detected = detect_output_file_type(filepath)
                        if file_type == 'xtb' and detected != 'xtb':
                            continue
                        if file_type == 'orca' and detected == 'xtb':
                            continue
                    found_files.append(filepath)
        
        if not found_files:
            continue  # Skip this type if no files found
            
        print(f"Processing {len(found_files)} {file_type.upper()} files in parallel...")
        
        # Use parallel processing to parse files
        max_workers = get_optimal_workers('cpu_intensive', len(found_files))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks - use keyword argument to avoid type checker issues
            future_to_file = {executor.submit(parse_function, filepath=filepath): filepath for filepath in found_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    results = future.result()
                    if results:
                        job_summaries.append(results)
                        all_results['job_count'] += 1
                        if results.get('time') is not None:
                            all_results['total_time'] += results['time']
                            if all_results['min_time'] is None or results['time'] < all_results['min_time']:
                                all_results['min_time'] = results['time']
                            if all_results['max_time'] is None or results['time'] > all_results['max_time']:
                                all_results['max_time'] = results['time']
                        if results.get('cycles') is not None:
                            all_results['total_cycles'] += results['cycles']
                            all_results['cycles_count'] += 1
                            if all_results['min_cycles'] is None or results['cycles'] < all_results['min_cycles']:
                                all_results['min_cycles'] = results['cycles']
                            if all_results['max_cycles'] is None or results['cycles'] > all_results['max_cycles']:
                                all_results['max_cycles'] = results['cycles']
                        if results.get('converged') is False:
                            all_results['non_converged'] += 1
                except Exception as e:
                    print(f"Error processing file: {e}")
        
        print(f"Completed processing {len(job_summaries)} {file_type.upper()} files successfully.")
        
        # Write summary file with collected results
        with open(summary_file, 'w', encoding='utf-8') as outfile:

            # Sort the job summaries by time, then by cycles
            job_summaries.sort(key=lambda x: (x.get('time') or float('inf'), x.get('cycles') or float('inf')))

            # Write summary
            outfile.write("=" * 40 + "\n")
            outfile.write("Summary of all calculations:\n")
            outfile.write(f"  Number of jobs: {all_results['job_count']}\n")
            outfile.write("-" * 40 + "\n")
            if all_results['total_time']:
                outfile.write(f"  Total execution time: {format_total_time(all_results['total_time'])}\n")
                outfile.write(f"  Mean execution time: {format_mean_time(all_results['total_time'] / all_results['job_count'])}\n")
                outfile.write(f"  Shortest execution time: {format_time_summary(all_results['min_time'], include_days=False)}\n")
                outfile.write(f"  Longest execution time: {format_time_summary(all_results['max_time'], include_days=False)}\n")
                if actual_wall_time is not None:
                    outfile.write(f"  Total wall time: {format_wall_time(actual_wall_time)}\n")
            outfile.write("-" * 40 + "\n")
            if all_results['cycles_count'] > 0:
                outfile.write(f"  Mean cycles: {all_results['total_cycles'] // all_results['cycles_count']}\n")
                outfile.write(f"  Min cycles: {all_results['min_cycles']}\n")
                outfile.write(f"  Max cycles: {all_results['max_cycles']}\n")
            else:
                # Keep cycle metrics explicit even when all jobs are non-converged.
                outfile.write("  Mean cycles: N/A\n")
                outfile.write("  Min cycles: N/A\n")
                outfile.write("  Max cycles: N/A\n")
            outfile.write(f"  Non-converged: {all_results['non_converged']}\n")

            outfile.write("=" * 40 + "\n\n")

            # Write individual job details
            job_index = 1
            for result in job_summaries:
                if file_type == 'gaussian':
                    outfile.write(f"=> {job_index}. {result['input_file']}.log\n")
                else:  # orca or xtb
                    outfile.write(f"=> {job_index}. {result['input_file']}.out\n")
                job_index += 1
                # Write in specific order: energy, cycles, time
                if 'energy' in result and result['energy'] is not None:
                    outfile.write(f"  energy = {result['energy']}\n")
                if 'cycles' in result and result['cycles'] is not None:
                    outfile.write(f"  cycles = {result['cycles']}\n")
                else:
                    outfile.write("  cycles = N/A\n")
                if 'time' in result and result['time'] is not None:
                    outfile.write(f"  time = {format_time_summary(result['time'], include_days=False)}\n")
                outfile.write("\n")

        print(f"Summary written to {summary_file}")
        results_by_type[file_type] = len(job_summaries)

    # Return total number of summaries created
    return sum(results_by_type.values())


# --- def find_out_files  (ascec-v04.py 9105-9146) ---
def find_out_files(root_dir, include_orca: bool = True, include_gaussian: bool = True):
    """Find calculation output files in the directory tree using parallel processing."""
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    out_files = []
    directories_to_search = []
    
    # First, collect all directories to search
    for root, dirs, files in os.walk(root_dir):
        directories_to_search.append((root, files))
    
    # Define function to search a single directory
    def search_directory(dir_data):
        root, files = dir_data
        local_out_files = []
        for file in files:
            if (include_orca and file.endswith('.out')) or (include_gaussian and file.endswith('.log') and not file.endswith('.xtbopt.log')):
                local_out_files.append(os.path.join(root, file))
        return local_out_files
    
    # Use parallel processing to search directories
    if len(directories_to_search) > 1:
        max_workers = get_optimal_workers('io_intensive', len(directories_to_search))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all directory search tasks
            future_to_dir = {executor.submit(search_directory, dir_data): dir_data[0] 
                            for dir_data in directories_to_search}
            
            # Collect results as they complete
            for future in as_completed(future_to_dir):
                try:
                    local_files = future.result()
                    out_files.extend(local_files)
                except Exception as e:
                    print(f"Error searching directory: {e}")
    else:
        # For single directory, just do it directly
        out_files = search_directory(directories_to_search[0]) if directories_to_search else []
    
    return out_files


# --- def get_unique_folder_name  (ascec-v04.py 9149-9159) ---
def get_unique_folder_name(base_name, current_dir):
    """Generate a unique folder name if base_name already exists."""
    folder_path = os.path.join(current_dir, base_name)
    counter = 1
    
    while os.path.exists(folder_path):
        new_name = f"{base_name}_{counter}"
        folder_path = os.path.join(current_dir, new_name)
        counter += 1
    
    return os.path.basename(folder_path)


# --- def group_files_by_base_with_tracking  (ascec-v04.py 9306-9365) ---
def group_files_by_base_with_tracking(directory='.'):
    """Group files by base name and track what was moved."""
    # Filter out ORCA intermediate files right from the start
    excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) 
             and not any(pattern in f for pattern in excluded_patterns)]
    base_map = defaultdict(list)
    
    for file in files:
        # Skip combined results to avoid moving them to subfolders
        if file.startswith("combined_results"):
             continue
        base = extract_base(file)
        if base:
            base_map[base].append(file)
    
    # Track moved files and created folders
    tracking = {'folders': [], 'moved_files': {}}
    moved_count = 0
    
    for base, grouped_files in base_map.items():
        if len(grouped_files) > 1:
            folder_path = os.path.join(directory, base)
            os.makedirs(folder_path, exist_ok=True)
            tracking['folders'].append(folder_path)
            
            for file in grouped_files:
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                tracking['moved_files'][dest] = src  # destination -> original
                shutil.move(src, dest)
            
            print(f"Moved {len(grouped_files)} files to folder: {base}")
            moved_count += len(grouped_files)
    
    if moved_count == 0:
        print("No files needed to be grouped.")
    else:
        print(f"Total files moved: {moved_count}")
    
    # Clean up orphaned .inp/.com/.gjf files at root if their subfolder exists
    # This handles case where files were already organized but input files were left behind
    # CRITICAL: Filter out ORCA intermediate files (.scfgrad.inp, .scfp.inp, etc.)
    excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
    for file in os.listdir(directory):
        if file.endswith(('.inp', '.com', '.gjf')) and os.path.isfile(os.path.join(directory, file)):
            # Skip ORCA intermediate files
            if any(pattern in file for pattern in excluded_patterns):
                continue
            base = extract_base(file)
            folder_path = os.path.join(directory, base)
            if base and os.path.isdir(folder_path):
                # Subfolder exists, move the orphaned input file there
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                if not os.path.exists(dest):  # Don't overwrite if already exists
                    shutil.move(src, dest)
    
    return tracking


# --- def create_summary_with_tracking  (ascec-v04.py 9368-9419) ---
def create_summary_with_tracking(directory, file_types_override: Optional[List[str]] = None, actual_wall_time=None):
    """Create summaries and return list of created files."""
    created_files = []

    # Classify .out files by actual content (ORCA vs xTB) and collect .log files
    orca_files = []
    xtb_files = []
    gaussian_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith(".out"):
                # Skip rescue files
                if '_rescue' in filename:
                    continue
                detected = detect_output_file_type(filepath)
                if detected == 'xtb':
                    xtb_files.append(filepath)
                else:
                    orca_files.append(filepath)
            elif filename.endswith(".log") and not filename.endswith(".xtbopt.log"):
                gaussian_files.append(filepath)

    try:
        # Determine which file types to process
        if file_types_override is not None:
            file_types_to_process = [ft for ft in file_types_override if ft in ('orca', 'gaussian', 'xtb')]
        else:
            file_types_to_process = []
            if orca_files:
                file_types_to_process.append('orca')
            if xtb_files:
                file_types_to_process.append('xtb')
            if gaussian_files:
                file_types_to_process.append('gaussian')

        # Create summaries for found file types
        if file_types_to_process:
            num_summaries = summarize_calculations(directory, file_types_to_process, actual_wall_time=actual_wall_time)

            # Check which summary files were created
            if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
                created_files.append("orca_summary.txt")
            if 'xtb' in file_types_to_process and os.path.exists("xtb_summary.txt"):
                created_files.append("xtb_summary.txt")
            if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
                created_files.append("gaussian_summary.txt")
    except Exception:
        pass

    return created_files


# --- def collect_out_files_with_tracking  (ascec-v04.py 9422-9618) ---
def collect_out_files_with_tracking(reuse_existing=False, target_cosmic_folder=None, include_gaussian: bool = True):
    """Collect output files and return the created cosmic folder path."""
    try:
        current_directory = os.getcwd()
        all_out_files = find_out_files(current_directory, include_orca=True, include_gaussian=include_gaussian)
        
        # Filter out backup files, ORCA intermediate files, and non-calculation output files
        all_out_files = [f for f in all_out_files if not (
            '.backup' in f or
            f.endswith('.out.backup') or
            f.endswith('.log.backup') or
            'orca_summary' in os.path.basename(f).lower() or
            'xtb_summary' in os.path.basename(f).lower() or
            'gaussian_summary' in os.path.basename(f).lower() or
            'combined_results' in os.path.basename(f).lower() or
            is_qm_byproduct(f) or
            '.scfhess.' in os.path.basename(f) or
            '.scfgrad.' in os.path.basename(f) or
            '.scfp.' in os.path.basename(f) or
            '.tmp.' in os.path.basename(f)
        )]
        
        if not all_out_files:
            return None

        num_files = len(all_out_files)
        
        # Detect file types to name folder appropriately
        out_files = [f for f in all_out_files if f.endswith('.out')]
        has_gaussian = any(f.endswith('.log') for f in all_out_files)
        has_xtb = False
        has_orca = False
        for f in out_files:
            detected = detect_output_file_type(f)
            if detected == 'xtb':
                has_xtb = True
            else:
                has_orca = True
            if has_xtb and has_orca:
                break

        if sum([has_orca, has_gaussian, has_xtb]) > 1:
            base_destination_folder_name = f"calc_out_{num_files}"
        elif has_gaussian:
            base_destination_folder_name = f"gaussian_out_{num_files}"
        elif has_xtb:
            base_destination_folder_name = f"xtb_out_{num_files}"
        else:
            base_destination_folder_name = f"orca_out_{num_files}"

        # Create cosmic folder with incremental numbering at parent level
        parent_directory = os.path.dirname(current_directory)
        
        def get_next_cosmic_dir():
            """Find the next available cosmic directory (cosmic, cosmic_2, etc.)"""
            # If target folder is explicitly provided, use it
            if target_cosmic_folder:
                # Handle both full path and relative path
                if os.path.isabs(target_cosmic_folder):
                    return target_cosmic_folder
                else:
                    return os.path.join(parent_directory, target_cosmic_folder)

            base_name = "cosmic"
            # Resume safety: if a legacy uppercase "COSMIC" tree already exists
            # in this project, keep using uppercase so we don't split state
            # across two trees.
            if os.path.exists(os.path.join(parent_directory, "COSMIC")):
                base_name = "COSMIC"
            cosmic_path = os.path.join(parent_directory, base_name)

            # If reuse_existing is True, return the base path if it exists
            if reuse_existing and os.path.exists(cosmic_path):
                return cosmic_path

            if not os.path.exists(cosmic_path):
                return cosmic_path

            counter = 2
            while True:
                cosmic_dir_name = f"{base_name}_{counter}"
                cosmic_path = os.path.join(parent_directory, cosmic_dir_name)
                if not os.path.exists(cosmic_path):
                    return cosmic_path
                counter += 1
        
        cosmic_dir = get_next_cosmic_dir()
        os.makedirs(cosmic_dir, exist_ok=True)
        
        # Reuse existing destination folder when possible, otherwise create one.
        # This prevents redo/resume flows from producing orca_out_N_1/_2/_3.
        existing_exact_dest = os.path.join(cosmic_dir, base_destination_folder_name)

        def _clear_destination_outputs(dest_dir: str) -> None:
            """Remove previous copied outputs so destination reflects current state."""
            try:
                for item in os.listdir(dest_dir):
                    item_path = os.path.join(dest_dir, item)
                    if os.path.isfile(item_path) and item.endswith(('.out', '.log')):
                        os.remove(item_path)
            except Exception:
                pass

        # Create the calc_out/orca_out/gaussian_out subfolder inside cosmic folder
        if reuse_existing:
            # In redo mode: always reuse the same output folder (cleanup if exists)
            # Look for any of the possible folder patterns
            existing_orca_dirs = glob.glob(os.path.join(cosmic_dir, f"orca_out_{num_files}*"))
            existing_gaussian_dirs = glob.glob(os.path.join(cosmic_dir, f"gaussian_out_{num_files}*"))
            existing_calc_dirs = glob.glob(os.path.join(cosmic_dir, f"calc_out_{num_files}*"))
            existing_orca_dirs.extend(existing_gaussian_dirs)
            existing_orca_dirs.extend(existing_calc_dirs)

            # Prefer exact folder name (e.g., orca_out_180) over suffixed variants.
            if os.path.isdir(existing_exact_dest):
                destination_path = existing_exact_dest
                destination_folder_name = os.path.basename(destination_path)
                _clear_destination_outputs(destination_path)
            elif existing_orca_dirs:
                existing_orca_dirs = sorted(existing_orca_dirs, key=lambda p: (0 if os.path.basename(p) == base_destination_folder_name else 1, p))
                destination_path = existing_orca_dirs[0]
                destination_folder_name = os.path.basename(destination_path)
                _clear_destination_outputs(destination_path)
            else:
                # First time - create the folder
                destination_folder_name = base_destination_folder_name
                destination_path = os.path.join(cosmic_dir, destination_folder_name)
                os.makedirs(destination_path, exist_ok=True)

            # Cleanup old artifacts before reusing cosmic folder
            print(f"Cleaning up previous cosmic results in {os.path.basename(cosmic_dir)}...")
            items_to_remove = [
                'dendrogram_images', 'extracted_clusters', 'extracted_data',
                'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
            ]

            # Also remove motifs folders
            for item in os.listdir(cosmic_dir):
                if item.startswith('motifs_') or item.startswith('umotifs_'):
                    items_to_remove.append(item)

            # Remove stale sibling output folders (e.g., orca_out_18 left over
            # when num_files inflated previously). Keep only the canonical
            # destination we are about to fill so cosmic sees a single source.
            canonical_dest_name = os.path.basename(destination_path)
            out_prefixes = ('orca_out_', 'gaussian_out_', 'calc_out_', 'xtb_out_', 'opt_out_')
            for item in os.listdir(cosmic_dir):
                if item == canonical_dest_name:
                    continue
                if any(item.startswith(p) for p in out_prefixes):
                    items_to_remove.append(item)

            for item in items_to_remove:
                item_path = os.path.join(cosmic_dir, item)
                if os.path.exists(item_path):
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {item}: {e}")
        else:
            # Default mode: if the exact destination exists, update it in-place.
            # This keeps workflow outputs stable instead of creating *_1 variants.
            if os.path.isdir(existing_exact_dest):
                destination_folder_name = base_destination_folder_name
                destination_path = existing_exact_dest
                _clear_destination_outputs(destination_path)
            else:
                destination_folder_name = get_unique_folder_name(base_destination_folder_name, cosmic_dir)
                destination_path = os.path.join(cosmic_dir, destination_folder_name)
                os.makedirs(destination_path, exist_ok=True)
        
        # Copy files to the output subfolder
        for file_path in all_out_files:
            shutil.copy2(file_path, destination_path)
        
        # Get just the folder name for display (without full path)
        cosmic_folder_name = os.path.basename(cosmic_dir)
        file_type_desc = "output files"
        if sum([has_orca, has_gaussian, has_xtb]) > 1:
            file_type_desc = "output files (mixed)"
        elif has_gaussian:
            file_type_desc = "Gaussian files"
        elif has_xtb:
            file_type_desc = "xTB files"
        else:
            file_type_desc = "ORCA files"
        print(f"Copied {num_files} {file_type_desc} to {cosmic_folder_name}/{destination_folder_name}")
        return destination_path
    except Exception as e:
        import traceback
        print(f"Error collecting output files: {e}")
        traceback.print_exc()
        return None


# =========================================================================== #
# R6 cross-stage helpers — verbatim ports of ascec-v04.py 10338-10531,         #
# 11173-11402.                                                                 #
# =========================================================================== #


def find_cosmic_script() -> "str | None":
    """
    Locate cosmic-v01.py script.
    Searches in: current dir, same dir as ascec script, scripts/ dir, parent dir, and PATH.

    Returns:
        Path to cosmic script, or None if not found
    """
    # __file__ here is cosmic_ascec/workflow/stages.py — three levels deep —
    # so the historical os.path.dirname(__file__) lookup misses the repo root.
    # sys.argv[0] is the entry-point script (ascec-v04.py at the repo root),
    # which always sits next to cosmic-v01.py.
    ascec_dir = os.path.dirname(os.path.abspath(__file__))
    entry_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else ascec_dir

    search_locations = [
        'cosmic-v01.py',  # Current working directory
        'cosmic.py',
        os.path.join(entry_dir, 'cosmic-v01.py'),  # Same dir as entry-point script
        os.path.join(entry_dir, 'cosmic.py'),
        os.path.join(ascec_dir, 'cosmic-v01.py'),  # Legacy: monolith colocation
        os.path.join(ascec_dir, 'cosmic.py'),
        'scripts/cosmic-v01.py',
        '../scripts/cosmic-v01.py',
        os.path.expanduser('~/scripts/cosmic-v01.py'),
    ]

    # Check each location
    for location in search_locations:
        if os.path.exists(location):
            return os.path.abspath(location)

    # Check if in PATH (shutil.which is cross-platform)
    found = shutil.which('cosmic-v01.py')
    if found:
        return found

    return None


def parse_cosmic_percentages(stdout_text: str) -> Tuple[float, float]:
    """
    Parse cosmic stdout to extract critical and skipped percentages.

    Example output:
        Total files skipped: 47 (11.2%)
        Critical skipped files: 4 (1.0%)

    Returns:
        (critical_percentage, skipped_percentage)
    """
    import re

    critical_pct = 0.0
    skipped_pct = 0.0

    try:
        # Look for critical percentages from both legacy and new cosmic outputs.
        critical_matches = re.findall(
            r'(?:Critical skipped files|Critical reduced-vector unmatched):.*?\(([0-9.]+)%\)',
            stdout_text,
            re.IGNORECASE,
        )
        if critical_matches:
            critical_pct = max(float(val) for val in critical_matches)

        # Look for "Total files skipped: 47 (11.2%)"
        skipped_match = re.search(r'Total files skipped:.*?\(([0-9.]+)%\)', stdout_text, re.IGNORECASE)
        if skipped_match:
            skipped_pct = float(skipped_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse cosmic percentages: {e}")

    return (critical_pct, skipped_pct)


def parse_cosmic_summary(summary_file: str) -> Tuple[float, float]:
    """
    Parse clustering_summary.txt file to extract critical and skipped percentages.

    Example content:
        Total files skipped: 1 (50.0%)
        Critical skipped files: 1 (50.0%)

    Returns:
        (critical_percentage, skipped_percentage)
    """
    import re

    critical_pct = 0.0
    skipped_pct = 0.0

    try:
        with open(summary_file, 'r') as f:
            content = f.read()

        # Capture critical percentages from both formats and use the strictest value.
        critical_matches = re.findall(
            r'(?:Critical skipped files|Critical reduced-vector unmatched):.*?\(([0-9.]+)%\)',
            content,
            re.IGNORECASE,
        )
        if critical_matches:
            critical_pct = max(float(val) for val in critical_matches)

        # Look for "Total files skipped: 1 (50.0%)"
        skipped_match = re.search(r'Total files skipped:.*?\(([0-9.]+)%\)', content, re.IGNORECASE)
        if skipped_match:
            skipped_pct = float(skipped_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse cosmic summary: {e}")

    return (critical_pct, skipped_pct)


def parse_cosmic_output(cosmic_dir: str) -> Tuple[int, int]:
    """
    Parse cosmic output to extract critical and skipped file counts.

    Returns:
        (critical_count, skipped_count)
    """
    summary_file = os.path.join(cosmic_dir, 'clustering_summary.txt')

    if not os.path.exists(summary_file):
        print(f"Warning: clustering_summary.txt not found in {cosmic_dir}")
        return (0, 0)

    critical_count = 0
    skipped_count = 0

    try:
        with open(summary_file, 'r') as f:
            content = f.read()

            # Critical count can come from skipped critical files or reduced-vector unmatched.
            import re
            critical_matches = re.findall(
                r'(?:Critical skipped files|Critical reduced-vector unmatched):\s*(\d+)',
                content,
                re.IGNORECASE,
            )
            if critical_matches:
                critical_count = max(int(val) for val in critical_matches)

            # Look for total skipped count like "Total files skipped: 97 (19.9%)"
            skipped_match = re.search(r'Total files skipped:\s*(\d+)', content, re.IGNORECASE)
            if skipped_match:
                skipped_count = int(skipped_match.group(1))

    except Exception as e:
        print(f"Warning: Error parsing cosmic output: {e}")

    return (critical_count, skipped_count)


def get_critical_count(optimization_dir_path: str) -> int:
    """
    Count files with imaginary frequencies in calculation directory.

    Returns:
        Number of critical files found
    """
    critical_count = 0

    # Look for ORCA (.out) and Gaussian (.log) output files
    for root, dirs, files in os.walk(optimization_dir_path):
        for file in files:
            if file.endswith('.out') or file.endswith('.log'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # Check for imaginary frequencies (ORCA and Gaussian)
                        if 'imaginary mode' in content.lower() or '***imaginary' in content.lower():
                            critical_count += 1
                except:
                    pass

    return critical_count


def get_critical_files_list(optimization_dir_path: str) -> List[str]:
    """
    Get list of files with imaginary frequencies.

    Returns:
        List of filepaths for files with imaginary frequencies
    """
    critical_files = []

    for root, dirs, files in os.walk(optimization_dir_path):
        for file in files:
            if file.endswith('.out') or file.endswith('.log'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'imaginary mode' in content.lower() or '***imaginary' in content.lower():
                            critical_files.append(filepath)
                except:
                    pass

    return critical_files


def check_workflow_pause(stage: Dict[str, Any], stage_num: int, total_stages: int,
                        cache_file: str, use_cache: bool) -> bool:
    """
    Check if workflow should pause after this stage.

    Args:
        stage: Stage dictionary that was just completed
        stage_num: Current stage number (1-indexed)
        total_stages: Total number of stages in workflow
        cache_file: Path to protocol cache file
        use_cache: Whether protocol caching is enabled

    Returns:
        True if workflow should continue, False if paused (will exit)
    """
    if not stage.get('pause_after', False):
        return True  # No pause requested, continue

    # Save pause state in cache
    if use_cache:
        cache = load_protocol_cache(cache_file)
        if not cache:
            cache = {}
        cache['paused_at_stage'] = stage_num
        cache['pause_timestamp'] = datetime.now().isoformat()
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

    # Display pause message
    print(f"\n{'='*70}")
    print(f"⏸  Workflow paused for manual review")
    print(f"{'='*70}")
    print(f"\nCompleted stage {stage_num}/{total_stages}: {stage['type'].capitalize()}")
    print(f"\nPlease review the results before continuing.")
    print(f"\nTo resume the workflow, run the same command again:")
    print(f"  The workflow will automatically continue from stage {stage_num + 1}")
    if use_cache:
        print(f"\nCache file: {cache_file}")
    print(f"\n{'='*70}\n")

    return False  # Signal to stop execution


def validate_cached_optimization_cosmic(cache: dict, stage: Dict[str, Any], stage_num: int,
                                            stages: List[Dict[str, Any]], stage_idx: int,
                                            cache_file: str) -> Tuple[bool, int]:
    """
    Validate cached optimization+cosmic results against thresholds.

    Returns:
        Tuple of (should_skip, new_stage_idx)
        - should_skip: True if cache is valid and should be skipped
        - new_stage_idx: Updated stage index if validation changes flow
    """
    stage_type = stage['type']
    stage_key = f"{stage_type}_{stage_num}"

    # Check if next stage is cosmic AND it's also cached
    next_is_cosmic = (stage_idx < len(stages) and stages[stage_idx]['type'] == 'cosmic')
    if not next_is_cosmic:
        return True, stage_idx

    next_stage_num = stage_idx + 1
    next_stage_key = f"cosmic_{next_stage_num}"
    if next_stage_key not in cache.get('stages', {}) or \
       cache['stages'][next_stage_key].get('status') != 'completed':
        return True, stage_idx

    # Check if optimization stage has redo parameters
    optimization_args = stage['args']
    max_critical = 0  # Default: 0% critical structures allowed (retry all)
    max_skipped = None

    for arg in optimization_args:
        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])

    # If no thresholds set, cache is valid
    if max_critical is None and max_skipped is None:
        # Print skipped cosmic stage
        cosmic_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] cosmic (cached)")
        print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1

    # Get cosmic directory from cache and validate
    cosmic_cache = cache['stages'][next_stage_key]
    cosmic_dir = cosmic_cache.get('result', {}).get('working_dir', 'cosmic')
    summary_file = os.path.join(cosmic_dir, "clustering_summary.txt")

    if not os.path.exists(summary_file):
        # Can't validate, accept cache
        cosmic_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] cosmic (cached)")
        print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1

    critical_pct, skipped_pct = parse_cosmic_summary(summary_file)

    threshold_met = True
    if max_critical is not None:
        threshold_met = critical_pct <= max_critical
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: critical {critical_pct:.1f}% > {max_critical}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    elif max_skipped is not None:
        threshold_met = skipped_pct <= max_skipped
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: skipped {skipped_pct:.1f}% > {max_skipped}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")

    if not threshold_met:
        # Remove both optimization and cosmic from cache
        if stage_key in cache.get('stages', {}):
            del cache['stages'][stage_key]
        if next_stage_key in cache.get('stages', {}):
            del cache['stages'][next_stage_key]
        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        # Go back to re-execute this stage
        return False, stage_idx - 1

    # Threshold met - skip both stages
    cosmic_stage_cache = cache['stages'][next_stage_key]
    print(f"\n{'-' * 60}")
    print(f"[{next_stage_num}/{len(stages)}] COSMIC (cached)")
    print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
    print('-' * 60)
    return True, stage_idx + 1


def validate_cached_refinement_cosmic(cache: dict, stage: Dict[str, Any], stage_num: int,
                                   stages: List[Dict[str, Any]], stage_idx: int,
                                   cache_file: str) -> Tuple[bool, int]:
    """
    Validate cached optimization+cosmic results against thresholds.

    Returns:
        Tuple of (should_skip, new_stage_idx)
        - should_skip: True if cache is valid and should be skipped
        - new_stage_idx: Updated stage index if validation changes flow
    """
    stage_type = stage['type']
    stage_key = f"{stage_type}_{stage_num}"

    # Check if next stage is cosmic AND it's also cached
    next_is_cosmic = (stage_idx < len(stages) and stages[stage_idx]['type'] == 'cosmic')
    if not next_is_cosmic:
        return True, stage_idx

    next_stage_num = stage_idx + 1
    next_stage_key = f"cosmic_{next_stage_num}"
    if next_stage_key not in cache.get('stages', {}) or \
       cache['stages'][next_stage_key].get('status') != 'completed':
        return True, stage_idx

    # Check if optimization stage has redo parameters
    opt_args = stage['args']
    max_critical = 0  # Default: 0% critical structures allowed (retry all)
    max_skipped = None

    for arg in opt_args:
        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])

    # If no thresholds set, cache is valid
    if max_critical is None and max_skipped is None:
        # Print skipped cosmic stage
        cosmic_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] cosmic (cached)")
        print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1

    # Get cosmic directory from cache and validate
    cosmic_cache = cache['stages'][next_stage_key]
    cosmic_dir = cosmic_cache.get('result', {}).get('working_dir', 'cosmic')
    summary_file = os.path.join(cosmic_dir, "clustering_summary.txt")

    if not os.path.exists(summary_file):
        # Can't validate, accept cache
        cosmic_stage_cache = cache['stages'][next_stage_key]
        print(f"\n{'-' * 60}")
        print(f"[{next_stage_num}/{len(stages)}] cosmic (cached)")
        print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
        print('-' * 60)
        return True, stage_idx + 1

    critical_pct, skipped_pct = parse_cosmic_summary(summary_file)

    threshold_met = True
    if max_critical is not None:
        threshold_met = critical_pct <= max_critical
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: critical {critical_pct:.1f}% > {max_critical}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")
    elif max_skipped is not None:
        threshold_met = skipped_pct <= max_skipped
        if not threshold_met:
            print(f"\n⚠ Cached result invalid: skipped {skipped_pct:.1f}% > {max_skipped}%")
            print(f"  Invalidating cache and re-running optimization with redo logic...")

    if not threshold_met:
        # Remove both opt and cosmic from cache
        if stage_key in cache.get('stages', {}):
            del cache['stages'][stage_key]
        if next_stage_key in cache.get('stages', {}):
            del cache['stages'][next_stage_key]
        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        # Go back to re-execute this stage
        return False, stage_idx - 1

    # Threshold met - skip both stages
    cosmic_stage_cache = cache['stages'][next_stage_key]
    print(f"\n{'-' * 60}")
    print(f"[{next_stage_num}/{len(stages)}] COSMIC (cached)")
    print(f"  ✓ Skipped (completed at {cosmic_stage_cache.get('timestamp', 'unknown')})")
    print('-' * 60)
    return True, stage_idx + 1


# =========================================================================== #
# R6b orchestrator + QM-concurrency machinery — verbatim ports of              #
# ascec-v04.py 11404-16025.                                                    #
# =========================================================================== #


def _record_redo_attempt(cache_file: str, stage_key: str, attempt: int,
                         max_redos: int, use_cache: bool = True) -> None:
    """Persist the *current* redo attempt into the stage cache entry.

    The cache only records the final ``attempts`` count when a stage completes,
    so a live ``ascec status`` view (and the in-progress protocol summary) has
    no way to tell which redo is currently running. Writing ``current_redo`` /
    ``redo_max`` here (merged into the in-progress result) lets those views show
    e.g. ``(Redo 1/3)``. The field is naturally dropped when the stage is later
    marked completed (which replaces the result wholesale).

    ``attempt`` 0 (the initial run) is recorded too, so the views can show
    ``(Redo 0/3)`` — otherwise "no tag" is ambiguous between "not a redo yet"
    and "redo reporting is broken". Stages without ``--redo=`` write nothing.
    """
    if not use_cache or not cache_file or not stage_key or attempt < 0 or max_redos <= 0:
        return
    try:
        update_protocol_cache(
            stage_key, 'in_progress',
            {'current_redo': attempt, 'redo_max': max_redos},
            cache_file,
        )
    except Exception:
        pass


def execute_workflow_stages(input_file: str, stages: List[Dict[str, Any]],
                           use_cache: bool = False, protocol_text: str = "") -> int:
    """
    Execute all workflow stages in sequence with context passing.
    
    Args:
        input_file: Path to initial input file
        stages: List of stage dictionaries from parse_workflow_stages()
        use_cache: If True, use protocol_cache for resumability
        protocol_text: Original protocol text from .in file for summary
        
    Returns:
        0 on success, non-zero on failure
    """
    workflow_start_dt = datetime.now()
    cache_file: Optional[str] = None

    def _is_background_tty_run() -> bool:
        """Return True when attached to a TTY but not running in the foreground."""
        try:
            if not sys.stdin.isatty():
                return False
            tty_fd = sys.stdin.fileno()
            return os.tcgetpgrp(tty_fd) != os.getpgrp()
        except Exception:
            return False

    _background_tty_launch = use_cache and os.environ.get("ASCEC_DETACHED_CHILD") != "1" and _is_background_tty_run()

    # ── Atomic duplicate-run guard + early job claim (protocol/cached runs) ──
    # Combine the duplicate check and DB insert in one transaction so two
    # ascecs launched simultaneously cannot both pass before either registers.
    # The placeholder row gets its cache_file/log_file/progress_file filled in
    # below at the registration finalization point.
    _early_claimed_job_id = 0
    if use_cache and os.environ.get("ASCEC_DETACHED_CHILD") != "1":
        _early_claimed_job_id, _conflict = _atomic_claim_ascec_job(input_file, os.getcwd())
        if _conflict:
            print(f"\nWARNING: '{input_file}' is already running")
            print(f"  Job ID: {_conflict['id']}  PID: {_conflict['pid']}  Started: {_conflict['started_at']}")
            print(f"  Run 'ascec status' to view or kill it.")
            # Caller exits via os._exit which does not flush; flush now so the
            # warning is visible when stdout is redirected to a file.
            try:
                sys.stdout.flush()
            except Exception:
                pass
            return 1

    # ── Queue / hold-until-PID-exits ────────────────────────────────────────
    # ``ascec <input> r3 after <PID>`` (set by the CLI as
    # ``_ascec_after_pid``): block here until the target run finishes, then
    # fall through to the normal launch. The placeholder row claimed above is
    # flipped to 'holding' so the wait is visible in ``ascec status``; the
    # later ``_adopt_ascec_job`` call flips it back to 'running'.
    _after_pid = globals().get('_ascec_after_pid', None)
    if _after_pid and os.environ.get("ASCEC_DETACHED_CHILD") != "1":
        try:
            _after_pid = int(_after_pid)
        except (TypeError, ValueError):
            _after_pid = 0
        if _after_pid > 0 and _after_pid != os.getpid() and _is_pid_alive(_after_pid):
            _set_job_holding(_early_claimed_job_id, _after_pid)
            print(
                f"\nHolding: waiting for PID {_after_pid} to finish before starting "
                f"'{input_file}'.\n  Run 'ascec status' to view (HOLDING) or cancel it.",
                flush=True,
            )
            try:
                while _is_pid_alive(_after_pid):
                    time.sleep(5)
            except KeyboardInterrupt:
                _update_ascec_job(_early_claimed_job_id, 'killed')
                print("\nHold cancelled.", flush=True)
                return 130
            print(f"\nPID {_after_pid} finished — starting '{input_file}'.", flush=True)

    # If launched in the background from the shell, immediately re-exec as a
    # detached child so the shell job does not get stopped by terminal I/O.
    # ─────────────────────────────────────────────────────────────────────────

    def format_compact_wall_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total_seconds = int(seconds)
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        millis = int(round((seconds - total_seconds) * 1000))
        return f"{secs}s {millis}ms"

    def render_final_workflow_summary() -> None:
        nonlocal progress_lines, last_progress_render

        total = len(stages)
        bar = render_progress_bar(total, total, width=30)

        # Prefer cache-backed stage data so redo mode reports FINAL cosmic counts.
        final_cache: Dict[str, Any] = {}
        if use_cache and isinstance(cache_file, str) and cache_file and os.path.exists(cache_file):
            final_cache = load_protocol_cache(cache_file) or {}
        cache_stages = final_cache.get('stages', {}) if isinstance(final_cache, dict) else {}

        cosmic_counter = 0
        summary_lines: List[str] = []

        for idx, stage_def in enumerate(stages, start=1):
            stage_type = stage_def.get('type', '')
            stage_name = stage_display_map.get(stage_type, str(stage_type).capitalize())
            stage_line = f"[{idx}/{total}] {stage_name} ✓"

            # Append redo count to every QM stage line when redo occurred
            _redo_attr = {
                'optimization': 'last_opt_redo_count',
                'refinement': 'last_ref_redo_count',
                'energy_refinement': 'last_eref_redo_count',
            }.get(stage_type)
            if _redo_attr:
                _redo_n = getattr(context, _redo_attr, 0)
                if _redo_n and _redo_n > 0:
                    stage_line = f"[{idx}/{total}] {stage_name} (redo: {_redo_n}) ✓"

            if stage_type == 'cosmic':
                cosmic_counter += 1
                stage_name = "cosmic" if cosmic_counter == 1 else f"cosmic_{cosmic_counter}"
                stage_key = f"cosmic_{idx}"
                stage_data = cache_stages.get(stage_key, {}) if isinstance(cache_stages, dict) else {}
                stage_result = stage_data.get('result', {}) if isinstance(stage_data, dict) else {}

                motifs_created = None
                inputs_count = None
                if isinstance(stage_result, dict):
                    motifs_created = stage_result.get('motifs_created')
                    inputs_count = stage_result.get('input_count')

                # Fallback to last known count if cache is unavailable.
                if motifs_created is None:
                    motifs_created = context.cosmic_motifs_created
                if inputs_count is None:
                    stage_input_counts = getattr(context, 'cosmic_stage_input_counts', {})
                    inputs_count = stage_input_counts.get(idx)

                if motifs_created is not None:
                    if inputs_count is not None and inputs_count > 0:
                        stage_line = f"[{idx}/{total}] {stage_name} ({inputs_count}→{motifs_created}) ✓"
                    else:
                        stage_line = f"[{idx}/{total}] {stage_name} ({motifs_created}) ✓"
                else:
                    stage_line = f"[{idx}/{total}] {stage_name} ✓"

            summary_lines.append(stage_line)

        workflow_end_dt = datetime.now()
        wall_time_str = format_compact_wall_time((workflow_end_dt - workflow_start_dt).total_seconds())

        # Build the final panel as a single block so it repaints the live progress
        # in-place instead of printing a duplicate header below it.
        lines = [
            "",
            "=== COSMIC ASCEC ===",
            "-" * 60,
            f"Progress [{bar}] 100.0%",
            "-" * 60,
        ] + summary_lines + [
            "",
            "Workflow finished",
            f"Start: {workflow_start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End:   {workflow_end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total wall time: {wall_time_str}",
            "-" * 60,
        ]

        if supports_ansi_repaint and progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A")
            for _ in range(progress_lines):
                sys.stdout.write("\033[2K\033[1B")
            sys.stdout.write(f"\033[{progress_lines}A")
            sys.stdout.flush()

        for line in lines:
            print(line)
        progress_lines = len(lines)
        last_progress_render = tuple(lines)

    def miniprint_cleanup() -> None:
        """
        Reduce disk usage so each stage folder keeps only final-motif outputs
        and useful files. Layout after cleanup:
        - annealing/ as-is.
        - geometry_optimization/: combined_results.*, *_summary.txt,
          geom_opt_out/{motif,umotif}_NN/ (reps mapped to the final ensemble).
        - geometry_refinement/: combined_results.*, *_summary.txt,
          geom_ref_out/{motif,umotif}_NN/.
        - energy_refinement/: combined_results.*, *_summary.txt,
          energy_ref_out/{motif,umotif}_NN/.
        - cosmic*/: left untouched (miniprint never removes anything from
          cosmic folders).
        - Root: final_ensemble.* or possible_final_ensemble.*, boltzmann_distribution.txt,
          protocol_summary.txt, protocol_*.pkl (kept for resume), .asc file.
        """
        import re as _re

        input_root = os.path.dirname(os.path.abspath(input_file)) or "."
        verbose = context.workflow_verbose_level >= 1

        if verbose:
            print(f"\n{'─' * 60}")
            print("Miniprint cleanup: reducing disk usage...")
            print(f"{'─' * 60}")

        # --- Identify stage directories and their types ---
        # Walk the stages list to figure out which directories correspond to what
        opt_dirs = []      # (dir_path, stage_index, is_refinement)
        cosmic_dirs = []      # (dir_path, stage_index)
        cosmic_counter = 0

        for idx, stage in enumerate(stages):
            stype = stage['type']
            if stype == 'optimization':
                # First optimization uses 'calculation' or 'optimization' dir
                calc_dir = context.optimization_stage_dir if context.optimization_stage_dir else "geometry_optimization"
                if not os.path.isdir(calc_dir):
                    calc_dir = "calculation"
                if os.path.isdir(calc_dir):
                    opt_dirs.append((calc_dir, idx, False))
            elif stype == 'refinement':
                ref_dir = context.refinement_stage_dir if context.refinement_stage_dir else "geometry_refinement"
                if os.path.isdir(ref_dir):
                    opt_dirs.append((ref_dir, idx, True))
            elif stype == 'energy_refinement':
                eref_dir = getattr(context, 'energy_refinement_stage_dir', None) or "energy_refinement"
                if os.path.isdir(eref_dir):
                    opt_dirs.append((eref_dir, idx, True))
            elif stype == 'cosmic':
                cosmic_counter += 1
                if cosmic_counter == 1:
                    cosmic_dir = "cosmic"
                    # Legacy fallback
                    if not os.path.isdir(cosmic_dir) and os.path.isdir("COSMIC"):
                        cosmic_dir = "COSMIC"
                else:
                    cosmic_dir = f"cosmic_{cosmic_counter}"
                    if not os.path.isdir(cosmic_dir) and os.path.isdir(f"COSMIC_{cosmic_counter}"):
                        cosmic_dir = f"COSMIC_{cosmic_counter}"
                if os.path.isdir(cosmic_dir):
                    cosmic_dirs.append((cosmic_dir, idx))

        # If context has better info from cache, also scan for dirs directly
        if not opt_dirs:
            for d in sorted(glob.glob("geometry_optimization*") + glob.glob("Geom Optimization*")) + sorted(glob.glob("calculation*")) + sorted(glob.glob("geometry_refinement*") + glob.glob("Geom Refinement*")) + sorted(glob.glob("energy_refinement*") + glob.glob("Energy Refinement*")):
                if os.path.isdir(d):
                    is_ref = "refine" in d.lower()
                    opt_dirs.append((d, -1, is_ref))
        if not cosmic_dirs:
            for d in sorted(glob.glob("cosmic*") + glob.glob("COSMIC*")):
                if os.path.isdir(d) and not d.lower().startswith("cosmic_tmp"):
                    cosmic_dirs.append((d, -1))

        # --- Determine the LAST optimization/refinement stage ---
        last_opt_idx = len(opt_dirs) - 1 if opt_dirs else -1

        # --- Find the final motif names from the last cosmic stage ---
        # These map back to the original calculation directories
        final_motif_mapping = {}  # motif_rank -> original_base_name (e.g. "opt_conf_5" or "motif_03_opt")
        final_cosmic_dir = cosmic_dirs[-1][0] if cosmic_dirs else None
        final_motifs_dir = None

        if final_cosmic_dir:
            # Find the final motifs/umotifs directory (highest numbered)
            for pattern in ["umotifs_*", "motifs_*"]:
                candidates = sorted(glob.glob(os.path.join(final_cosmic_dir, pattern)))
                candidates = [c for c in candidates if os.path.isdir(c)]
                if candidates:
                    final_motifs_dir = candidates[-1]
                    break

        # --- Helper: find motif representative folders in a calculation directory ---
        def find_representative_folders(calc_dir, motif_mapping):
            """Map motif ranks to actual subdirectory paths in calc_dir.
            Returns dict: rank -> (subdir_path, original_base_name)"""
            result = {}
            if not os.path.isdir(calc_dir):
                return result

            subdirs = {d: os.path.join(calc_dir, d) for d in os.listdir(calc_dir)
                       if os.path.isdir(os.path.join(calc_dir, d))}

            for rank, stem in motif_mapping.items():
                # Generate candidate folder names from the filename stem
                # stem examples: "motif_01_opt_conf_5", "motif_01_opt", "umotif_01_motif_03_opt"
                candidates = [stem]
                # Strip trailing _opt suffix (calculation folders don't include it)
                if stem.endswith('_opt'):
                    candidates.append(stem[:-4])
                # Try just the part after the output prefix (e.g. "opt_conf_5" from "motif_01_opt_conf_5")
                m = _re.match(r'(?:u?motif)_\d+_(.*)', stem)
                if m:
                    rest = m.group(1)
                    candidates.append(rest)
                    if rest.endswith('_opt'):
                        candidates.append(rest[:-4])
                # For refinement: "motif_01_opt" -> folder "motif_01"
                m2 = _re.match(r'((?:u?motif)_\d+)_opt$', stem)
                if m2:
                    candidates.append(m2.group(1))

                for variant in candidates:
                    if variant in subdirs:
                        result[rank] = (subdirs[variant], variant)
                        break
            return result

        # --- Helper: parse Boltzmann distribution file for rank -> source mapping ---
        def parse_boltzmann_sources(cosmic_dir_path):
            """Parse boltzmann_distribution.txt to get rank -> source_stem mapping."""
            mapping = {}
            boltz = os.path.join(cosmic_dir_path, "boltzmann_distribution.txt")
            if os.path.exists(boltz):
                try:
                    with open(boltz, 'r') as bf:
                        lines = bf.readlines()
                    cur_rank = None
                    for line in lines:
                        mr = _re.search(r'\((?:u?motif)_(\d+)\)', line)
                        if mr:
                            cur_rank = int(mr.group(1))
                        ms = _re.search(r'(?:From s|S)tructure:\s*(\S+)', line)
                        if ms and cur_rank is not None:
                            mapping[cur_rank] = ms.group(1)
                            cur_rank = None
                except (IOError, OSError):
                    pass
            return mapping

        # --- Helper: parse a cosmic stage's motif-to-source mapping ---
        def parse_sim_mapping(cosmic_dir_path):
            """Parse motif filenames or Boltzmann file to get rank -> source_stem mapping.
            For clean names (umotif_01.xyz with no suffix), prefers Boltzmann source info
            since filenames don't encode the original structure name."""
            mapping = {}
            motifs_d = None
            for pat in ["umotifs_*", "motifs_*"]:
                cands = sorted(glob.glob(os.path.join(cosmic_dir_path, pat)))
                cands = [c for c in cands if os.path.isdir(c)]
                if cands:
                    motifs_d = cands[-1]
                    break
            if not motifs_d:
                return mapping

            # Check if filenames encode source info (have suffix beyond prefix_NN)
            has_informative_names = False
            for fname in sorted(os.listdir(motifs_d)):
                if not fname.endswith('.xyz') or 'combined' in fname:
                    continue
                stem = fname[:-4]
                m = _re.match(r'(?:u?motif)_(\d+)(?:_(.+))?\.xyz', fname)
                if m:
                    rank = int(m.group(1))
                    rest = m.group(2)
                    if rest:
                        # Has suffix like "opt_conf_5" or "motif_03_opt" — informative
                        has_informative_names = True
                        mapping[rank] = stem

            if has_informative_names and mapping:
                return mapping

            # Clean names (umotif_01.xyz) — use Boltzmann for source traceability
            boltz_mapping = parse_boltzmann_sources(cosmic_dir_path)
            if boltz_mapping:
                return boltz_mapping

            # Last resort: use filename stems even if not informative
            mapping = {}
            for fname in sorted(os.listdir(motifs_d)):
                if not fname.endswith('.xyz') or 'combined' in fname:
                    continue
                stem = fname[:-4]
                m = _re.match(r'(?:u?motif)_(\d+)', fname)
                if m:
                    mapping[int(m.group(1))] = stem
            return mapping

        # --- Helper: extract rank number from a motif/umotif stem ---
        def extract_rank_from_stem(stem):
            """Extract the rank number from stems like 'umotif_05_opt', 'motif_03_opt', 'umotif_05'."""
            m = _re.match(r'(?:u?motif)_(\d+)', stem)
            return int(m.group(1)) if m else None

        # --- Helper: copy motif folders from calc_dir to out_dir with final naming ---
        def extract_motif_folders(calc_dir, rank_to_stem, out_dir, prefix):
            """Extract representative folders, renaming to final Boltzmann ordering.
            rank_to_stem maps final_rank -> stem that should exist as folder in calc_dir."""
            rep_folders = find_representative_folders(calc_dir, rank_to_stem)
            if not rep_folders:
                return 0
            os.makedirs(out_dir, exist_ok=True)
            for rank in sorted(rep_folders.keys()):
                src_path, _ = rep_folders[rank]
                dest_path = os.path.join(out_dir, f"{prefix}_{rank:02d}")
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                # Strip only non-output files: temp/intermediate, redo artifacts,
                # and runtime scratch.  All genuine calculation outputs are kept.
                strip_patterns = [
                    # Temp / intermediate
                    "*.tmp", "*.densitiesinfo", "*.xtbw",
                    # Redo artifacts — not real calculation outputs
                    "*.out.backup", "*_rescue.*",
                    # ORCA runtime scratch (not scientific outputs)
                    "*.bas[0-9]", "*.hostnames",
                    # xTB runtime scratch (not scientific outputs)
                    "*.xtbrestart", "*.xtbtopo.mol",
                ]
                for pat in strip_patterns:
                    for f in glob.glob(os.path.join(dest_path, pat)):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
            return len(rep_folders)

        if final_cosmic_dir:
            final_motif_mapping = parse_sim_mapping(final_cosmic_dir)

        # --- Build cosmic chain mappings ---
        # cosmic_mappings[i] = {rank: source_stem} for cosmic stage i
        cosmic_mappings = []
        for sd, _ in cosmic_dirs:
            cosmic_mappings.append(parse_sim_mapping(sd))

        # --- Identify refinement stages vs optimization-only ---
        ref_stages = [(d, idx, ir) for d, idx, ir in opt_dirs if ir]  # refinement stages only
        opt_only_stages = [(d, idx, ir) for d, idx, ir in opt_dirs if not ir]  # optimization stages

        # --- Determine the final prefix (motif vs umotif) ---
        final_prefix = "umotif" if (final_motifs_dir and "umotif" in os.path.basename(final_motifs_dir)) else "motif"

        # Subdirectories inside each stage dir that must survive the cleaning pass.
        preserve_subdirs: Dict[str, set] = {}

        def _chain_back(start_mapping, from_cosmic_idx, to_cosmic_idx):
            """Translate final_rank -> stem from cosmic index `from` back to `to`."""
            rank_to_stem = dict(start_mapping)
            for chain_idx in range(from_cosmic_idx, to_cosmic_idx, -1):
                prev_mapping = cosmic_mappings[chain_idx - 1] if chain_idx - 1 >= 0 else {}
                new_rank_to_stem = {}
                for final_rank, current_stem in rank_to_stem.items():
                    intermediate_rank = extract_rank_from_stem(current_stem)
                    if intermediate_rank is not None and intermediate_rank in prev_mapping:
                        new_rank_to_stem[final_rank] = prev_mapping[intermediate_rank]
                rank_to_stem = new_rank_to_stem
            return rank_to_stem

        final_cosmic_idx = len(cosmic_mappings) - 1

        if ref_stages and final_motif_mapping:
            # Extract final-rank motif folders inside each refinement stage dir.
            for ref_idx, (ref_dir, _, _) in enumerate(ref_stages):
                cosmic_for_this_ref = ref_idx + len(opt_only_stages)
                rank_to_stem = _chain_back(final_motif_mapping, final_cosmic_idx, cosmic_for_this_ref)

                if len(ref_stages) == 1:
                    out_dir_name = "geom_ref_out"
                elif ref_idx == len(ref_stages) - 1:
                    out_dir_name = "energy_ref_out"
                elif ref_idx == 0:
                    out_dir_name = "geom_ref_out"
                else:
                    out_dir_name = f"ref_{ref_idx + 1}_out"

                out_dir = os.path.join(ref_dir, out_dir_name)
                count = extract_motif_folders(ref_dir, rank_to_stem, out_dir, final_prefix)
                if count > 0:
                    preserve_subdirs.setdefault(ref_dir, set()).add(out_dir_name)
                if verbose and count > 0:
                    print(f"  Created {ref_dir}/{out_dir_name}/ with {count} {final_prefix} folders")

        # Always extract final-rank motif folders inside the last optimization stage.
        if opt_only_stages and final_motif_mapping:
            last_opt_dir = opt_only_stages[-1][0]
            if ref_stages:
                # Rigorous/ultimate: chain back from final cosmic to cosmic_dirs[0] (follows opt).
                opt_rank_to_stem = _chain_back(final_motif_mapping, final_cosmic_idx, 0)
                opt_prefix = final_prefix
            else:
                # Preliminary: final cosmic directly reflects opt stems.
                opt_rank_to_stem = dict(final_motif_mapping)
                opt_prefix = "motif"
            out_dir = os.path.join(last_opt_dir, "geom_opt_out")
            count = extract_motif_folders(last_opt_dir, opt_rank_to_stem, out_dir, opt_prefix)
            if count > 0:
                preserve_subdirs.setdefault(last_opt_dir, set()).add("geom_opt_out")
            if verbose and count > 0:
                print(f"  Created {last_opt_dir}/geom_opt_out/ with {count} {opt_prefix} folders")

        # --- Clean optimization/refinement directories ---
        # Keep only: combined_results.*, *_summary.txt, and the exported motif folder.
        for i, (calc_dir, _, is_ref) in enumerate(opt_dirs):
            if not os.path.isdir(calc_dir):
                continue

            preserve = preserve_subdirs.get(calc_dir, set())
            kept_root_files = set()
            removed_count = 0

            for entry in os.listdir(calc_dir):
                entry_path = os.path.join(calc_dir, entry)

                if os.path.isfile(entry_path):
                    keep = (entry.startswith("combined_results") or
                            entry in ("orca_summary.txt", "xtb_summary.txt", "gaussian_summary.txt") or
                            entry.endswith('_summary.txt'))
                    if keep:
                        kept_root_files.add(entry)
                    else:
                        os.remove(entry_path)
                        removed_count += 1
                elif os.path.isdir(entry_path):
                    if entry in preserve:
                        kept_root_files.add(entry)
                        continue
                    shutil.rmtree(entry_path)
                    removed_count += 1

            if verbose and removed_count > 0:
                label = "geometry_refinement" if is_ref else "geometry_optimization"
                print(f"  Cleaned {calc_dir}/ ({label}): kept {len(kept_root_files)} items")

        # --- Annealing and cosmic directories are preserved as-is by miniprint ---
        # Per project policy, --miniprint MUST NOT touch annealing/ or cosmic/
        # folders.  Annealing produces the result_*/resultbox_*/anneal.*/tvse_*
        # scientific outputs that must remain intact; per-run scratch is already
        # removed during the run by cleanup_run_dir_keeplist().  Cosmic folders
        # are scanned above only to derive the final-motif mapping used when
        # extracting representative folders from optimization/refinement, but
        # their contents are never modified here.

        # --- Report final disk usage ---
        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(input_root)
                for f in filenames
            )
            size_mb = total_size / (1024 * 1024)
            if verbose:
                print(f"\n  Final disk usage: {size_mb:.1f} MB")
                print(f"{'─' * 60}")
            else:
                print(f"\n✓ Miniprint: {size_mb:.1f} MB (use --maxprint to keep all files)")
        except OSError:
            if not verbose:
                print("\n✓ Miniprint cleanup complete (use --maxprint to keep all files)")

    def copy_final_ensemble_to_root() -> None:
        """Copy final ensemble files from the last cosmic output to input root."""
        input_root = os.path.dirname(os.path.abspath(input_file))

        # Resolve the last cosmic directory from context first, then cache fallback.
        cosmic_candidates: List[str] = []
        if getattr(context, 'cosmic_dir', None):
            cosmic_candidates.append(str(context.cosmic_dir))

        if use_cache and isinstance(cache_file, str) and cache_file and os.path.exists(cache_file):
            cache_data = load_protocol_cache(cache_file) or {}
            cache_stages = cache_data.get('stages', {}) if isinstance(cache_data, dict) else {}
            for idx in range(len(stages), 0, -1):
                if stages[idx - 1].get('type') != 'cosmic':
                    continue
                stage_key = f"cosmic_{idx}"
                stage_data = cache_stages.get(stage_key, {}) if isinstance(cache_stages, dict) else {}
                if not isinstance(stage_data, dict):
                    continue
                stage_result = stage_data.get('result', {})
                if not isinstance(stage_result, dict):
                    continue
                working_dir = stage_result.get('working_dir')
                if isinstance(working_dir, str) and working_dir:
                    cosmic_candidates.append(working_dir)
                break

        resolved_cosmic_dir = None
        for candidate in cosmic_candidates:
            candidate_dir = candidate
            base_name = os.path.basename(candidate_dir)
            if base_name.startswith('orca_out_') or base_name.startswith('gaussian_out_') or base_name.startswith('opt_out_') or base_name.startswith('calc_out_') or base_name.startswith('xtb_out_'):
                candidate_dir = os.path.dirname(candidate_dir)

            abs_candidate = os.path.abspath(candidate_dir)
            if os.path.isdir(abs_candidate):
                resolved_cosmic_dir = abs_candidate
                break

        if not resolved_cosmic_dir:
            print("Warning: Could not resolve final cosmic directory for final ensemble copy.")
            return

        umotif_dirs = sorted(glob.glob(os.path.join(resolved_cosmic_dir, 'umotifs_*')))
        motif_dirs = sorted(glob.glob(os.path.join(resolved_cosmic_dir, 'motifs_*')))

        # Prefer the most refined/final ensemble when present.
        source_dir = umotif_dirs[-1] if umotif_dirs else (motif_dirs[-1] if motif_dirs else None)
        if not source_dir:
            print(f"Warning: No motifs_/umotifs_ folder found in {resolved_cosmic_dir} for final ensemble copy.")
            return

        source_xyz = None
        source_mol = None

        preferred_xyz = [
            os.path.join(source_dir, 'all_umotifs_combined.xyz'),
            os.path.join(source_dir, 'all_motifs_combined.xyz'),
        ]
        preferred_mol = [
            os.path.join(source_dir, 'all_umotifs_combined.mol'),
            os.path.join(source_dir, 'all_motifs_combined.mol'),
        ]

        for path in preferred_xyz:
            if os.path.exists(path):
                source_xyz = path
                break
        for path in preferred_mol:
            if os.path.exists(path):
                source_mol = path
                break

        if source_xyz is None:
            xyz_candidates = sorted(glob.glob(os.path.join(source_dir, 'all_*_combined.xyz')))
            if xyz_candidates:
                source_xyz = xyz_candidates[-1]
        if source_mol is None:
            mol_candidates = sorted(glob.glob(os.path.join(source_dir, 'all_*_combined.mol')))
            if mol_candidates:
                source_mol = mol_candidates[-1]

        # Choose output naming: possible_final_ensemble when the protocol has
        # no refinement/eref stage (pure opt+cosmic), final_ensemble otherwise.
        opt_only = getattr(context, 'cosmic_opt_only', False)
        if opt_only:
            ensemble_name = 'possible_final_ensemble'
        else:
            ensemble_name = 'final_ensemble'

        copied_any = False
        copied_xyz = False
        copied_mol = False
        if source_xyz and os.path.exists(source_xyz):
            shutil.copy2(source_xyz, os.path.join(input_root, f'{ensemble_name}.xyz'))
            copied_any = True
            copied_xyz = True
        if source_mol and os.path.exists(source_mol):
            shutil.copy2(source_mol, os.path.join(input_root, f'{ensemble_name}.mol'))
            copied_any = True
            copied_mol = True

        # Copy the last Boltzmann distribution to root alongside the ensemble
        boltz_src = os.path.join(resolved_cosmic_dir, "boltzmann_distribution.txt")
        if os.path.exists(boltz_src):
            boltz_dst = os.path.join(input_root, "boltzmann_distribution.txt")
            shutil.copy2(boltz_src, boltz_dst)
            copied_any = True

        if copied_any:
            if opt_only:
                print("Warning: No true minima can be assured without frequency calculations in the final ensemble.")
            if context.workflow_verbose_level >= 1:
                if copied_xyz:
                    print(f"{ensemble_name.replace('_', ' ').capitalize()} copied to: {os.path.join(input_root, f'{ensemble_name}.xyz')}")
                if copied_mol:
                    print(f"{ensemble_name.replace('_', ' ').capitalize()} copied to: {os.path.join(input_root, f'{ensemble_name}.mol')}")
        else:
            print(f"Warning: No final combined ensemble files found in {source_dir}.")

    context = WorkflowContext(input_file=input_file)
    context.is_workflow = True  # We're in workflow mode
    context.workflow_verbose_level = parse_verbosity_level(sys.argv)
    context.cosmic_opt_only = False  # Determined after stages are parsed (opt-only when no ref/eref)
    context.maxprint = globals().get('_ascec_maxprint_requested', False)  # Default: miniprint (clean up at end)
    
    # Read configuration from input file
    # - Line 9: QM program index and alias (e.g., "2 orca")
    # - Line 11: nprocs (QM calculations and ASCEC evaluation)
    # Default to reasonable values if not specified or if input file doesn't exist
    try:
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                lines = f.readlines()
                config_line_count = 0
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    config_line_count += 1
                    
                    if config_line_count == 9:  # Line 9: QM Program Index & Alias
                        parts = line.split('#')[0].strip().split()
                        if len(parts) > 1:
                            # Second value is the QM program alias (e.g., "orca", "g16")
                            context.qm_alias = parts[1]
                        elif len(parts) == 1:
                            # Only index provided, use default based on index
                            qm_idx = int(parts[0])
                            context.qm_alias = "orca" if qm_idx == 2 else "g16" if qm_idx == 1 else "orca"

                    if config_line_count == 6:  # Line 6: max cycles
                        parts = line.split('#')[0].strip().split()
                        if parts:
                            context.xtb_cycles = int(parts[0])
                    
                    if config_line_count == 11:  # Line 11: nprocs
                        parts = line.split('#')[0].strip().split()
                        if parts:
                            context.qm_nproc = int(parts[0])
                        if len(parts) > 1:
                            # User explicitly specified ASCEC cores
                            context.ascec_parallel_cores = int(parts[1])
                        else:
                            # Only QM procs specified, use auto-detect (0)
                            context.ascec_parallel_cores = 0
                        break
    except (ValueError, IndexError, IOError):
        context.ascec_parallel_cores = 0  # Default to auto-detect if parsing fails
        context.qm_alias = "orca"  # Default alias
        context.qm_nproc = None
        context.xtb_cycles = 200
    
    # Use protocol-specific cache filename with random seed to support parallel protocols
    # First, check if there's an existing protocol cache file for THIS input file
    import glob
    existing_caches = sorted(glob.glob("protocol_*.pkl"))
    cache_file = None
    
    if existing_caches and use_cache:
        # Look for cache that matches this input file
        for cache_path in existing_caches:
            test_cache = load_protocol_cache(cache_path)
            if test_cache and test_cache.get('input_file') == input_file:
                cache_file = cache_path
                if context.workflow_verbose_level >= 1:
                    print(f"Found existing protocol cache: {cache_file} (for {input_file})")
                break
    
    if cache_file is None:
        # Generate new random 6-digit seed (similar to annealing seed)
        import random
        protocol_seed = random.randint(100000, 999999)
        cache_file = f"protocol_{protocol_seed}.pkl"
        if context.workflow_verbose_level >= 1:
            print(f"Creating new protocol cache: {cache_file} (for {input_file})")
    
    context.cache_file = cache_file  # Store in context for use by stages
    
    # Load cache if in protocol mode
    cache = {}
    is_fresh_start = False
    if use_cache:
        cache = load_protocol_cache(cache_file)
        # Distinguish a brand-new run from a resume BEFORE we seed
        # input_file, otherwise the cache always looks non-empty below.
        is_fresh_start = not cache
        if not cache:
            cache = {}
        if 'input_file' not in cache:
            cache['input_file'] = input_file
            # Save immediately so the association is recorded
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        if not is_fresh_start:
            start_time = cache.get('start_time', None)
            completed_stages = cache.get('stages', {})
            
            # Check if protocol is completed
            is_completed = cache.get('completed', False)
            
            if is_completed and completed_stages:
                # Protocol was completed - offer resume from specific stage
                print(f"\n{'='*70}")
                print(f"Protocol was previously completed!")
                if start_time:
                    dt = datetime.fromtimestamp(start_time)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Original run started: {time_str}")
                
                # Show completed stages
                print(f"\nCompleted stages:")
                stage_list = []
                for idx, (stage_key, stage_data) in enumerate(sorted(completed_stages.items()), 1):
                    stage_type = stage_data.get('type', 'Unknown')
                    stage_list.append((idx, stage_key, stage_type))
                    print(f"  [{idx}] {stage_type}")
                
                print(f"\nOptions:")
                print(f"  [0] Start from beginning (will use existing directories)")
                for idx, _, stage_type in stage_list:
                    print(f"  [{idx}] Resume from {stage_type}")
                print(f"  [q] Quit")
                
                choice = input(f"\nSelect starting point [0-{len(stage_list)}, q]: ").strip().lower()
                
                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)
                
                try:
                    choice_idx = int(choice)
                    if choice_idx == 0:
                        # Start from beginning - clear cache but keep directories
                        print(f"Starting from beginning (existing directories will be reused)")
                        cache = {}
                        if protocol_text:
                            cache['protocol_text'] = protocol_text
                        cache['total_stages'] = len(stages)
                        save_protocol_cache(cache, cache_file)
                    elif 1 <= choice_idx <= len(stage_list):
                        # Resume from specific stage - mark previous stages as completed
                        resume_stage_key = stage_list[choice_idx - 1][1]
                        print(f"Resuming from stage: {stage_list[choice_idx - 1][2]}")
                        print(f"Previous stages marked as completed")
                        # Keep cache but mark as not completed (we're resuming)
                        cache['completed'] = False
                    else:
                        print(f"Invalid choice. Starting from beginning.")
                        cache = {}
                        if protocol_text:
                            cache['protocol_text'] = protocol_text
                        cache['total_stages'] = len(stages)
                        save_protocol_cache(cache, cache_file)
                except ValueError:
                    print(f"Invalid input. Starting from beginning.")
                    cache = {}
                    if protocol_text:
                        cache['protocol_text'] = protocol_text
                    cache['total_stages'] = len(stages)
                    save_protocol_cache(cache, cache_file)
                
                print(f"{'='*70}\n")
            elif start_time:
                # Protocol in progress - just resume
                dt = datetime.fromtimestamp(start_time)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if context.workflow_verbose_level >= 1:
                    print(f"Resuming workflow (started: {time_str})")
            else:
                if context.workflow_verbose_level >= 1:
                    print(f"Resuming workflow")
            
            # Backfill total_stages for old cache files (compatibility)
            if 'total_stages' not in cache:
                cache['total_stages'] = len(stages)
                save_protocol_cache(cache, cache_file)
        else:
            # Store protocol text and total stages when first starting
            if protocol_text:
                cache['protocol_text'] = protocol_text
            cache['total_stages'] = len(stages)
            save_protocol_cache(cache, cache_file)

    # Determine opt-only once from the protocol: True only when no
    # refinement or energy_refinement stage exists (pure opt+cosmic workflow).
    _has_ref_stage = any(s.get('type') in ('refinement', 'energy_refinement') for s in stages)
    context.cosmic_opt_only = not _has_ref_stage

    # ── Terminal independence: SIGHUP immunity + tee log + job registry ───────
    # Only set up background tracking for protocol (cached) runs.
    # Simple commands like 'ascec input.asc r3' run without tracking overhead.
    import signal as _signal

    _job_id: int = 0
    _log_fh = None
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    _progress_legacy_file: str = os.path.join(os.getcwd(), ".ascec_progress.json")
    _progress_file: str = _progress_legacy_file
    _progress_stream = None

    def _remove_progress_file():
        """Remove .ascec_progress.json and its temp file from the working directory."""
        # Close process-tied temp stream first (if used).
        if _progress_stream is not None:
            try:
                _progress_stream.close()
            except Exception:
                pass

        # Clean legacy on-disk artifacts (fallback mode / old runs).
        for p in (_progress_legacy_file, _progress_legacy_file + ".tmp"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        # Best-effort cleanup of stale temp progress files from older runs.
        try:
            for p in glob.glob(os.path.join(os.getcwd(), ".ascec_progress_*.json")):
                try:
                    os.remove(p)
                except OSError:
                    pass
        except Exception:
            pass

    if use_cache:
        # Low-overhead process-tied progress file: points to an open FD path.
        # When this process dies (even SIGKILL), the FD vanishes automatically.
        try:
            import tempfile as _tempfile
            _fd, _tmp_path = _tempfile.mkstemp(
                suffix='.json',
                prefix='.ascec_progress_',
                dir=os.getcwd(),
                text=True,
            )
            _progress_stream = os.fdopen(_fd, 'w+')
            try:
                # Remove directory entry immediately; file lives only via open FD.
                os.unlink(_tmp_path)
            except OSError:
                pass
            _progress_file = f"/proc/{os.getpid()}/fd/{_progress_stream.fileno()}"
        except Exception:
            _progress_stream = None
            _progress_file = _progress_legacy_file

        # Survive SSH disconnect / terminal close (POSIX only; Windows has no SIGHUP)
        if hasattr(_signal, 'SIGHUP'):
            try:
                _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass

        # Open a persistent log file for this run (used by 'ascec status' view)
        _state_dir = _ascec_state_dir()
        (_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        _ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        _safe_name = re.sub(r'[^\w.-]', '_', os.path.basename(input_file))
        _log_file = str(_state_dir / "logs" / f"{_safe_name}_{_ts_str}.log")
        try:
            _log_fh = open(_log_file, 'w', buffering=1)
        except OSError:
            _log_fh = None

        # Tee stdout/stderr: write to both terminal and log file.
        # When the terminal dies (SSH disconnect), writes silently continue to log only.
        class _TeeStream:
            def __init__(self, primary, log):
                self._p = primary
                self._l = log
                self._dead = False
                # Propagate attributes that library code may inspect
                self.encoding = getattr(primary, 'encoding', 'utf-8')
                self.errors = getattr(primary, 'errors', 'replace')

            def write(self, data):
                if self._l is not None:
                    try:
                        self._l.write(data)
                        self._l.flush()
                    except Exception:
                        pass
                if not self._dead:
                    try:
                        self._p.write(data)
                        self._p.flush()
                    except (OSError, IOError):
                        self._dead = True

            def flush(self):
                if self._l is not None:
                    try: self._l.flush()
                    except Exception: pass

            def isatty(self):
                return not self._dead and getattr(self._p, 'isatty', lambda: False)()

            def fileno(self):
                try:
                    return self._p.fileno()
                except (AttributeError, OSError):
                    raise OSError("stream does not support fileno()")

        class _LogOnlyStream:
            """Write-only stream that goes to the log file only (no terminal)."""
            def __init__(self, log):
                self._l = log
                self.encoding = 'utf-8'
                self.errors = 'replace'
            def write(self, data):
                try: self._l.write(data); self._l.flush()
                except Exception: pass
            def flush(self):
                try: self._l.flush()
                except Exception: pass
            def isatty(self): return False
            def fileno(self): raise OSError("detached log-only stream")

        def _set_output_streams(log_only: bool = False) -> None:
            if _log_fh is None:
                return
            if log_only:
                sys.stdout = _LogOnlyStream(_log_fh)
                sys.stderr = _LogOnlyStream(_log_fh)
            else:
                sys.stdout = _TeeStream(_orig_stdout, _log_fh)
                sys.stderr = _TeeStream(_orig_stderr, _log_fh)

        _set_output_streams(log_only=False)

        if _background_tty_launch:
            try:
                _signal.signal(_signal.SIGTTIN, _signal.SIG_IGN)
                _signal.signal(_signal.SIGTTOU, _signal.SIG_IGN)
            except Exception:
                pass
            _set_output_streams(log_only=True)

        # Register (or rebind) job in the status database. Priority order:
        #   1. ASCEC_REUSE_JOB_ID — set by Ctrl+D detach handoff from a parent ascec
        #   2. _early_claimed_job_id — placeholder row from atomic claim above
        #   3. fresh INSERT — fallback if neither path applies
        _reuse_job_id = 0
        try:
            _reuse_job_id = int(os.environ.get("ASCEC_REUSE_JOB_ID", "0") or "0")
        except (TypeError, ValueError):
            _reuse_job_id = 0

        _target_job_id = _reuse_job_id if _reuse_job_id > 0 else _early_claimed_job_id

        if _target_job_id > 0 and _adopt_ascec_job(
            _target_job_id,
            os.getpid(),
            _log_file if _log_fh is not None else "",
            _progress_file,
            cache_file=cache_file or "",
        ):
            _job_id = _target_job_id
        else:
            _job_id = _register_ascec_job(
                os.path.abspath(input_file),
                os.getcwd(),
                cache_file or "",
                _log_file if _log_fh is not None else "",
                _progress_file,
            )

        # SIGTERM handler: update job status when killed via 'ascec status K <id>'
        def _handle_sigterm(signum, frame):
            _update_ascec_job(_job_id, 'killed')
            _remove_progress_file()
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
            if _log_fh is not None:
                try: _log_fh.close()
                except Exception: pass
            raise SystemExit(1)

        _signal.signal(_signal.SIGTERM, _handle_sigterm)

        # Ctrl+D detach: background thread watches stdin for EOT byte (0x04).
        # When detected: relaunches this command detached and exits foreground
        # process so the shell prompt returns immediately on any shell.
        import threading as _threading
        import subprocess as _subprocess

        def _do_detach():
            """Detach by spawning a fully detached child and exiting this process."""
            # Flush current log stream before handoff.
            if _log_fh is not None:
                try:
                    _log_fh.flush()
                except Exception:
                    pass

            # Relaunch same command fully detached from terminal/session.
            # NB: ``__file__`` here is ``cosmic_ascec/workflow/stages.py`` —
            # a library module, not a runnable entry point. Use ``sys.argv[0]``
            # (the actual ``ascec``/``ascec-v04.py`` script the user invoked);
            # fall back to ``python -m cosmic_ascec.command_line.ascec`` when
            # argv[0] is missing (e.g. ``-c`` or stripped by a wrapper).
            _child_pid = 0
            try:
                _entry = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
                if _entry and os.path.isfile(_entry):
                    _cmd = [sys.executable, _entry] + sys.argv[1:]
                else:
                    _cmd = [sys.executable, "-m", "cosmic_ascec.command_line.ascec"] + sys.argv[1:]
                _env = os.environ.copy()
                _env["ASCEC_DETACHED_CHILD"] = "1"
                if _job_id:
                    _env["ASCEC_REUSE_JOB_ID"] = str(_job_id)

                _child_log = _subprocess.DEVNULL
                _child_log_handle = None
                if _log_fh is not None:
                    _child_log_handle = open(_log_file, 'a', buffering=1)
                    _child_log = _child_log_handle

                _child = _subprocess.Popen(
                    _cmd,
                    cwd=os.getcwd(),
                    env=_env,
                    stdin=_subprocess.DEVNULL,
                    stdout=_child_log,
                    stderr=_child_log,
                    start_new_session=True,
                    close_fds=True,
                )
                _child_pid = int(_child.pid)

                if _child_log_handle is not None:
                    try:
                        _child_log_handle.close()
                    except Exception:
                        pass
            except Exception:
                _child_pid = 0

            if _child_pid > 0:
                # Print detach notice with the real background PID.
                _orig_stdout.write(
                    f"\n  ASCEC detaching (PID {_child_pid}) — job keeps running.\n"
                    f"  Use 'ascec status' to monitor or kill.\n\n"
                )
                _orig_stdout.flush()

                # Hand off status row to detached child PID.
                if _job_id:
                    _adopt_ascec_job(
                        _job_id,
                        _child_pid,
                        _log_file if _log_fh is not None else "",
                        _progress_file,
                    )

                # Restore terminal streams and close handles before exiting foreground process.
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
                if _log_fh is not None:
                    try:
                        _log_fh.close()
                    except Exception:
                        pass
                os._exit(0)

            # Fallback: stay attached if detach relaunch failed.
            _orig_stdout.write("  Warning: detach failed; continuing in foreground.\n")
            _orig_stdout.flush()

        def _stdin_ctrl_d_watcher():
            """Daemon thread: read raw bytes from stdin; trigger detach on Ctrl+D."""
            try:
                import select as _sel
                _stdin_obj = getattr(sys, '__stdin__', None)
                if _stdin_obj is None:
                    return
                fd = _stdin_obj.fileno()
                while True:
                    if _sel.select([fd], [], [], 1.0)[0]:
                        byte = os.read(fd, 1)
                        # Ctrl+D in cooked mode → empty read (EOF flush) or 0x04
                        if byte in (b'', b'\x04'):
                            _do_detach()
                            break
            except Exception:
                pass

        if sys.stdin.isatty() and not _background_tty_launch:
            _watcher = _threading.Thread(target=_stdin_ctrl_d_watcher, daemon=True)
            _watcher.start()
    # ──────────────────────────────────────────────────────────────────────────

    # Pre-scan stages to extract cosmic args for use in optimization stage
    for stage in stages:
        if stage['type'] == 'cosmic':
            context.cosmic_args = stage.get('args', [])
            break
    
    # Count optimization stages for proper numbering (opt, opt2)
    optimization_stage_counter = 0
    
    # Map stage types to display names
    stage_display_map: Dict[str, str] = {
        'replication': 'annealing',
        'optimization': 'geometry_optimization',
        'cosmic': 'cosmic',
        'refinement': 'geometry_refinement',
        'energy_refinement': 'energy_refinement'
    }

    progress_lines = 0
    last_progress_render: Optional[Tuple[str, ...]] = None
    supports_ansi_repaint = bool(sys.stdout.isatty() or (os.environ.get("TERM") not in (None, "", "dumb")))

    def render_progress_bar(current: float, total: float, width: int = 30) -> str:
        if total <= 0:
            return "░" * width
        ratio = min(current / total, 1.0)
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)

    def _active_redo_tag(stage_index: int) -> str:
        """Return " (Redo n/N)" for a redo-enabled stage, else "".

        Shown from attempt 0 (the initial run) so an absent tag means "this
        stage has no --redo=", never "the counter is not working".
        """
        if getattr(context, 'active_redo_stage_num', 0) != stage_index:
            return ""
        max_redos = getattr(context, 'active_redo_max', 0)
        if not max_redos:
            return ""
        attempt = getattr(context, 'active_redo_attempt', 0)
        return f" (Redo {attempt}/{max_redos})"

    def render_workflow_progress(completed_stages: int, current_stage_num: int, sub_progress: str = "") -> None:
        """Render compact workflow progress with in-place updates (always visible)."""
        nonlocal progress_lines, last_progress_render

        total = len(stages)
        completed_stages = max(0, min(completed_stages, total))
        current_stage_num = max(1, min(current_stage_num, total))

        # Ensure the active stage is always ahead of completed stages to avoid blank panels.
        if completed_stages < total and current_stage_num <= completed_stages:
            current_stage_num = completed_stages + 1

        stage_lines = []
        for i, st in enumerate(stages, start=1):
            name = stage_display_map.get(st['type'], st['type'].capitalize())
            if i <= completed_stages:
                line = f"[{i}/{total}] {name} ✓"
                if st.get('type') == 'cosmic':
                    stage_counts = getattr(context, 'cosmic_stage_counts', {})
                    stage_input_counts = getattr(context, 'cosmic_stage_input_counts', {})
                    stage_total = stage_counts.get(i)
                    stage_inputs = stage_input_counts.get(i)
                    if stage_total is None:
                        motif_count = getattr(context, 'last_cosmic_motif_count', None)
                        umotif_count = getattr(context, 'last_cosmic_umotif_count', None)
                        if motif_count is not None or umotif_count is not None:
                            m_val = motif_count if motif_count is not None else 0
                            u_val = umotif_count if umotif_count is not None else 0
                            stage_total = m_val + u_val
                    if stage_total is not None and stage_total > 0:
                        if stage_inputs is not None and stage_inputs > 0:
                            line += f" ({stage_inputs}→{stage_total})"
                        else:
                            line += f" ({stage_total})"
                stage_lines.append(line)
            elif i == current_stage_num and completed_stages < total:
                suffix = f" {sub_progress}" if sub_progress else " ..."
                stage_lines.append(f"[{i}/{total}] {name}{suffix}{_active_redo_tag(i)}")
                break

        # Smooth progress within a stage using n/N updates (annealing/optimization/refinement only).
        stage_fraction = 0.0
        if completed_stages < total and 1 <= current_stage_num <= total:
            active_stage_type = stages[current_stage_num - 1].get('type')
            if active_stage_type in ('replication', 'optimization', 'refinement', 'energy_refinement') and sub_progress:
                m = re.search(r'(\d+)\s*/\s*(\d+)', sub_progress)
                if m:
                    done = int(m.group(1))
                    stage_total = int(m.group(2))
                    if stage_total > 0:
                        stage_fraction = min(max(done / stage_total, 0.0), 1.0)
                        # For replication: when step info is provided it represents
                        # OVERALL annealing-step progress aggregated across all
                        # replicas (completed + running), so use it directly.
                        # sub_progress format: "N/M (step total_done/total_steps)"
                        if active_stage_type == 'replication':
                            m2 = re.search(r'step\s+(\d+)\s*/\s*(\d+)', sub_progress)
                            if m2:
                                step_done = int(m2.group(1))
                                step_total = int(m2.group(2))
                                if step_total > 0:
                                    stage_fraction = min(max(step_done / step_total, 0.0), 1.0)

        progress_units = min(completed_stages + stage_fraction, float(total))
        pct = ((progress_units / total) * 100.0) if total > 0 else 0.0
        bar = render_progress_bar(progress_units, total, width=30)

        lines = [
            "",
            "=== COSMIC ASCEC ===",
            "-" * 60,
            f"Progress [{bar}] {pct:.1f}%",
            "-" * 60,
        ] + stage_lines + [""]

        # Skip duplicate redraw if the rendered panel text is identical.
        render_snapshot = tuple(lines)
        if render_snapshot == last_progress_render:
            return

        if supports_ansi_repaint and progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A")
            for _ in range(progress_lines):
                sys.stdout.write("\033[2K\033[1B")
            sys.stdout.write(f"\033[{progress_lines}A")
            sys.stdout.flush()

        for line in lines:
            print(line)
        progress_lines = len(lines)
        last_progress_render = render_snapshot

        # Write machine-readable progress state for 'ascec status' re-attachment.
        # Prefer process-tied stream (auto-vanishes on process death); fallback
        # to legacy on-disk file with atomic replace.
        try:
            _prog_state = {
                "job_id": _job_id,
                "input_file": input_file,
                "stages_total": total,
                "stages_completed": completed_stages,
                "current_stage_num": current_stage_num,
                "pct": round(pct, 1),
                "sub_progress": sub_progress,
                "stage_lines": stage_lines,
                "current_redo": getattr(context, 'active_redo_attempt', 0),
                "redo_max": getattr(context, 'active_redo_max', 0),
                "redo_stage_num": getattr(context, 'active_redo_stage_num', 0),
                "updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            if _progress_stream is not None:
                _progress_stream.seek(0)
                _progress_stream.truncate(0)
                json.dump(_prog_state, _progress_stream)
                _progress_stream.flush()
            else:
                _tmp_prog = _progress_file + ".tmp"
                with open(_tmp_prog, 'w') as _pf:
                    json.dump(_prog_state, _pf)
                os.replace(_tmp_prog, _progress_file)
        except Exception:
            pass

    context.completed_stage_count = 0
    last_progress_call: Optional[Tuple[int, int, str]] = None

    def _update_workflow_progress(sub_progress: str = "") -> None:
        nonlocal last_progress_call
        completed = context.completed_stage_count
        current = min(completed + 1, len(stages))
        # Redo state is part of the signature: a redo re-runs the same stage and
        # replays the same n/N strings, so without it the dedupe would swallow
        # the repaint that first surfaces "(Redo n/N)".
        call_sig = (completed, current, sub_progress,
                    getattr(context, 'active_redo_stage_num', 0),
                    getattr(context, 'active_redo_attempt', 0))
        if call_sig == last_progress_call:
            return
        last_progress_call = call_sig
        render_workflow_progress(completed, current, sub_progress)

    context.update_progress = _update_workflow_progress
    
    # Cleaner output for protocol mode
    if use_cache:
        stage_display_names = []
        for s in stages:
            display_name = stage_display_map.get(s['type'], s['type'].capitalize()) or s['type'].capitalize()
            if s.get('pause_after', False):
                display_name += ' ⏸'
            stage_display_names.append(display_name)
        stage_names = ' → '.join(stage_display_names)
        if context.workflow_verbose_level >= 1:
            print(f"\nWorkflow: {stage_names}\n")
    else:
        if context.workflow_verbose_level >= 1:
            print("-" * 60)
            print(f"Workflow: {input_file}")
            stage_display_parts = []
            for s in stages:
                part = s['type']
                if s.get('pause_after', False):
                    part += ' ⏸'
                stage_display_parts.append(part)
            stage_names = ' → '.join(stage_display_parts)
            print(f"Pipeline: {stage_names}")
            print("-" * 60)

    completed_stage_count = 0
    
    # Execute each stage in sequence with optimization+cosmic retry logic
    stage_idx = 0
    while stage_idx < len(stages):
        stage_num = stage_idx + 1
        stage = stages[stage_idx]
        stage_type = stage['type']
        # Expose active stage to helper functions that inspect threshold flags.
        context.current_stage = stage  # type: ignore[attr-defined]
        
        # Check if stage already completed (from cache)
        if use_cache and 'stages' in cache:
            stage_key = f"{stage_type}_{stage_num}"
            if stage_key in cache['stages']:
                stage_cache = cache['stages'][stage_key]
                if stage_cache.get('status') == 'completed':
                    # Display name (use new naming convention)
                    display_name = stage_display_map.get(stage_type, stage_type.capitalize())
                    if context.workflow_verbose_level >= 1:
                        print(f"\n{'-' * 60}")
                        print(f"[{stage_num}/{len(stages)}] {display_name} (cached)")
                        print(f"  ✓ Skipped (completed at {stage_cache.get('timestamp', 'unknown')})")
                        print('-' * 60)

                    completed_stage_count = stage_num
                    context.completed_stage_count = completed_stage_count
                    stage_idx += 1

                    # Validate cached optimization+cosmic if applicable
                    if stage_type == 'optimization':
                        should_skip, new_idx = validate_cached_optimization_cosmic(
                            cache, stage, stage_num, stages, stage_idx, cache_file
                        )
                        stage_idx = new_idx
                        # Keep completed_stage_count in sync with the new position:
                        # new_idx is the 0-based index of the next stage to run, so all
                        # stages before it are done. When the validator also consumes the
                        # cached cosmic stage (should_skip) this credits it; when it loops
                        # back to re-run (not should_skip) it un-credits it. Without this
                        # the next stage's live n/N progress is mislabeled as the previous
                        # stage in the progress bar and 'ascec status'.
                        completed_stage_count = new_idx
                        context.completed_stage_count = completed_stage_count
                        continue
                    # Validate cached refinement+cosmic if applicable
                    elif stage_type == 'refinement':
                        should_skip, new_idx = validate_cached_refinement_cosmic(
                            cache, stage, stage_num, stages, stage_idx, cache_file
                        )
                        stage_idx = new_idx
                        completed_stage_count = new_idx
                        context.completed_stage_count = completed_stage_count
                        continue
                    continue
        
        # Display name for stage (use new naming convention)
        display_name = stage_display_map.get(stage_type, stage_type.capitalize())
        if context.workflow_verbose_level >= 1:
            print(f"\n{'-' * 60}")
            print(f"[{stage_num}/{len(stages)}] {display_name}")
            print('-' * 60)
        else:
            context.update_progress("")
        
        # Update cache - mark stage as in progress and snapshot summary so the
        # protocol_summary.txt reflects the new stage immediately.
        if use_cache:
            stage_key = f"{stage_type}_{stage_num}"
            context.current_stage_key = stage_key  # Store for use in stage execution
            update_protocol_cache(stage_key, 'in_progress', cache_file=cache_file)
            if cache_file:
                generate_protocol_summary(cache_file=cache_file)
        
        try:
            if stage_type == 'replication':
                result = execute_replication_stage(context, stage)
                
                # Update cache on success with detailed results
                if use_cache and result == 0:
                    stage_key = f"{stage_type}_{stage_num}"
                    
                    # Collect detailed annealing results
                    result_data = {'num_replicas': stage['num_replicas']}
                    
                    # Store directories for next stages to use
                    result_data['working_dir'] = '.'  # Annealing works in current dir
                    result_data['output_dir'] = '.'   # XYZ files created in current dir
                    result_data['annealing_dirs'] = context.annealing_dirs  # List of w6_annealing4_N directories
                    
                    # Get box size from stage context if available
                    if hasattr(context, 'annealing_box_size'):
                        result_data['box_size'] = context.annealing_box_size
                    if hasattr(context, 'annealing_packing'):
                        result_data['packing'] = context.annealing_packing
                    
                    # Count total accepted configurations from result files
                    total_accepted = 0
                    if context.annealing_dirs:
                        for adir in context.annealing_dirs:
                            # Count structures in result files
                            result_files = glob.glob(os.path.join(adir, "result_*.xyz"))
                            if result_files:
                                # Read XYZ file to count structures
                                with open(result_files[0], 'r') as f:
                                    lines = f.readlines()
                                    i = 0
                                    while i < len(lines):
                                        if lines[i].strip().isdigit():
                                            total_accepted += 1
                                            natoms = int(lines[i].strip())
                                            i += natoms + 2  # Skip atoms + comment line
                                        else:
                                            i += 1
                    
                    if total_accepted > 0:
                        result_data['total_accepted'] = total_accepted
                    
                    # Extract energy evaluations from annealing.out files
                    total_energy_evals = 0
                    if context.annealing_dirs:
                        for adir in context.annealing_dirs:
                            annealing_out = os.path.join(adir, 'annealing.out')
                            if os.path.exists(annealing_out):
                                try:
                                    with open(annealing_out, 'r') as f:
                                        content = f.read()
                                        match = re.search(r'Energy calculations:\s+(\d+)', content)
                                        if match:
                                            total_energy_evals += int(match.group(1))
                                except:
                                    pass
                    
                    if total_energy_evals > 0:
                        result_data['energy_evals'] = total_energy_evals
                    
                    update_protocol_cache(stage_key, 'completed',
                                        result=result_data,
                                        cache_file=cache_file)
                    if use_cache and cache_file:
                        generate_protocol_summary(cache_file=cache_file)

                if result != 0:
                    print(f"\nError: Annealing failed with code {result}")
                    return result

                # Check if workflow should pause after this stage
                if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                    return 0  # Paused successfully

                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count

            elif stage_type == 'optimization':
                # Increment optimization stage counter
                optimization_stage_counter += 1
                context.optimization_stage_number = optimization_stage_counter
                
                # Check if next stage is cosmic - if so, handle optimization+cosmic with retry
                next_is_cosmic = (stage_idx + 1 < len(stages) and 
                                     stages[stage_idx + 1]['type'] == 'cosmic')
                
                if next_is_cosmic:
                    # Extract redo parameters from optimization stage
                    optimization_args = stage['args']
                    max_redos = 3   # Default redo budget; overridden by --redo=N
                    max_critical = 0  # Default: 0% critical threshold (retry all)
                    max_skipped = None
                    concurrent_jobs = 1
                    _skipped_set = False
                    _concurrent_given = False

                    for arg in optimization_args:
                        if arg.startswith('--redo='):
                            try:
                                max_redos = max(0, int(arg.split('=')[1]))
                            except ValueError:
                                pass
                        elif arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                            _skipped_set = True
                        elif arg.startswith('--concurrent='):
                            try:
                                concurrent_jobs = max(1, int(arg.split('=')[1]))
                                _concurrent_given = True
                            except ValueError:
                                pass

                    # If --skipped given, use skipped threshold instead of default critical=0
                    if _skipped_set:
                        max_critical = None

                    # No --concurrent= in stage args ⇒ silent default of 1.

                    # Show redo configuration
                    if max_redos > 1:
                        if context.workflow_verbose_level >= 1:
                            if max_critical is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target critical ≤ {max_critical}%")
                            elif max_skipped is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target skipped ≤ {max_skipped}%")

                    # Redo loop for optimization+cosmic
                    # attempt 0 = initial run, attempts 1..max_redos = redo attempts
                    final_attempt = 0
                    initial_critical = None  # Track initial critical % from first attempt
                    initial_skipped = None   # Track initial skipped % from first attempt
                    initial_critical_count = None
                    initial_skipped_count = None
                    for attempt in range(max_redos + 1):
                        final_attempt = attempt
                        # Publish the redo counter from attempt 0 so the panel and
                        # 'ascec status' show "(Redo 0/N)" during the initial run.
                        if max_redos > 0:
                            context.active_redo_stage_num = stage_num
                            context.active_redo_attempt = attempt
                            context.active_redo_max = max_redos
                            _upd = getattr(context, 'update_progress', None)
                            if callable(_upd):
                                _upd("")
                            _record_redo_attempt(cache_file, f"optimization_{stage_num}",
                                                 attempt, max_redos, use_cache)
                        if attempt > 0:
                            if context.workflow_verbose_level >= 1:
                                print(f"\n{'-' * 60}")
                                print(f"Redo {attempt}/{max_redos}")

                        # Don't delete cosmic folder - we'll update it with corrected calculations

                        # Run calculation (which includes sort step that creates cosmic/)
                        result = execute_optimization_stage(context, stage)
                        if result != 0:
                            print(f"\nError: Optimization failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"optimization_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                            return result

                        # Note: on redo (attempt > 0), execute_optimization_stage's internal
                        # sort step already re-runs collect_out_files_with_tracking, which (a)
                        # places the freshly recalculated .out files into the canonical orca_out
                        # folder and (b) wipes stale sibling out folders + previous cosmic
                        # artifacts (motifs_*, clustering_summary, skipped_structures, ...).
                        # No outer hand-copy or cleanup needed here.


                        # Run cosmic; stage header is only shown in verbose mode.
                        if attempt == 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"[{stage_idx + 2}/{len(stages)}] cosmic")
                            print('-' * 60)

                        cosmic_stage = stages[stage_idx + 1]
                        if use_cache:
                            cosmic_key = f"cosmic_{stage_num + 1}"
                            update_protocol_cache(cosmic_key, 'in_progress', cache_file=cache_file)
                        result = execute_cosmic_stage(context, cosmic_stage)
                        if result != 0:
                            print(f"\nError: cosmic failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"optimization_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                            return result

                        # Parse cosmic results from clustering_summary.txt
                        summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                        if os.path.exists(summary_file):
                            critical_pct, skipped_pct = parse_cosmic_summary(summary_file)

                            # Capture initial values on first attempt
                            if attempt == 0:
                                initial_critical = critical_pct
                                initial_skipped = skipped_pct
                                # Get counts for initial attempt
                                cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic"
                                init_crit_count, init_skip_count = parse_cosmic_output(cosmic_dir)
                                initial_critical_count = init_crit_count
                                initial_skipped_count = init_skip_count

                            if context.workflow_verbose_level >= 1:
                                print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")

                            # Check thresholds based on which was set
                            threshold_met = True

                            if max_critical is not None:
                                threshold_met = critical_pct <= max_critical

                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")

                            elif max_skipped is not None:
                                threshold_met = skipped_pct <= max_skipped

                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                            else:
                                # No thresholds set - accept results
                                break

                            # If threshold not met and attempts remain, continue to redo logic below
                            if not threshold_met:
                                if attempt < max_redos:
                                    # Loop will continue to next iteration
                                    pass
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"Max attempts reached")
                        else:
                            if context.workflow_verbose_level >= 1:
                                print("⚠ Warning: Could not find clustering_summary.txt")
                            break

                    # Store redo count for terminal summary
                    context.active_redo_stage_num = 0
                    context.active_redo_attempt = 0
                    context.active_redo_max = 0
                    context.last_opt_redo_count = final_attempt

                    # Get final cosmic results for cache
                    final_critical = None
                    final_skipped = None
                    summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        final_critical, final_skipped = parse_cosmic_summary(summary_file)

                    # Update cache for both optimization and cosmic stages
                    if use_cache:
                        calc_key = f"optimization_{stage_num}"
                        cosmic_key = f"cosmic_{stage_num + 1}"

                        calc_result: Dict[str, Any] = {
                            'attempts': final_attempt,
                            'max_redos': max_redos,
                        }
                        
                        # Store directories for stage memory
                        # Get input dir from previous stage (annealing or current dir)
                        input_dir = context.get_previous_stage_output_dir('replication')
                        if input_dir is None:
                            input_dir = '.'  # Default if no previous stage
                        
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        calc_result['input_dir'] = input_dir
                        calc_result['working_dir'] = optimization_dir_path
                        calc_result['output_dir'] = optimization_dir_path  # .out files are in optimization_dir_path
                        
                        if max_critical is not None:
                            calc_result['critical_threshold'] = max_critical
                        if max_skipped is not None:
                            calc_result['skipped_threshold'] = max_skipped
                        
                        # Add XYZ source info if available
                        if hasattr(context, 'optimization_xyz_source'):
                            calc_result['xyz_source'] = context.optimization_xyz_source
                        
                        # Add completion counts if available
                        if hasattr(context, 'optimization_completed'):
                            calc_result['completed'] = context.optimization_completed
                        if hasattr(context, 'optimization_total'):
                            calc_result['total'] = context.optimization_total

                        # Add concurrent jobs count and total CPU time for protocol summary
                        calc_result['concurrent_jobs'] = concurrent_jobs
                        if hasattr(context, 'optimization_total_cpu_time'):
                            calc_result['total_cpu_time'] = context.optimization_total_cpu_time

                        # Add cosmic folder info if available
                        if hasattr(context, 'optimization_cosmic_folder'):
                            calc_result['cosmic_folder'] = context.optimization_cosmic_folder

                        update_protocol_cache(calc_key, 'completed',
                                            result=calc_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                        cosmic_result = {}

                        # Store directories for stage memory
                        cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic"
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        cosmic_result['input_dir'] = optimization_dir_path  # Read from calculation
                        cosmic_result['working_dir'] = cosmic_dir
                        # After calculation: use "motifs" prefix (first level clustering)
                        cosmic_result['output_dir'] = os.path.join(cosmic_dir, "motifs")  # Motifs for opt stage
                        
                        if final_critical is not None:
                            cosmic_result['critical_pct'] = final_critical
                        if final_skipped is not None:
                            cosmic_result['skipped_pct'] = final_skipped
                        
                        # Extract critical and skipped counts from cosmic output
                        critical_count, skipped_count = parse_cosmic_output(cosmic_dir)
                        cosmic_result['critical_count'] = critical_count
                        cosmic_result['skipped_count'] = skipped_count
                        
                        # Extract threshold value from cosmic command args
                        cosmic_stage = stages[stage_idx + 1] if stage_idx + 1 < len(stages) else {}
                        cosmic_args = cosmic_stage.get('args', [])
                        for arg in cosmic_args:
                            if arg.startswith('--th=') or arg.startswith('--threshold='):
                                raw_val = arg.split('=', 1)[1]
                                try:
                                    cosmic_result['threshold'] = float(raw_val)
                                except ValueError:
                                    cosmic_result['threshold'] = raw_val
                            elif arg.startswith('--rmsd='):
                                rmsd_val = float(arg.split('=')[1])
                                cosmic_result['rmsd_threshold'] = rmsd_val

                        # Add cosmic folder and motifs info if available
                        if hasattr(context, 'cosmic_folder'):
                            cosmic_result['cosmic_folder'] = context.cosmic_folder
                        if hasattr(context, 'cosmic_motifs_created'):
                            cosmic_result['motifs_created'] = context.cosmic_motifs_created

                        # Add initial validation values (from first attempt)
                        if initial_critical is not None:
                            cosmic_result['initial_critical'] = initial_critical
                        if initial_skipped is not None:
                            cosmic_result['initial_skipped'] = initial_skipped
                        if initial_critical_count is not None:
                            cosmic_result['initial_critical_count'] = initial_critical_count
                        if initial_skipped_count is not None:
                            cosmic_result['initial_skipped_count'] = initial_skipped_count
                        
                        # Add threshold info and attempts
                        cosmic_result['attempts'] = final_attempt
                        if max_critical is not None:
                            cosmic_result['threshold_type'] = 'critical'
                            cosmic_result['threshold_value'] = max_critical
                            cosmic_result['threshold_met'] = (final_critical is not None and final_critical <= max_critical)
                        elif max_skipped is not None:
                            cosmic_result['threshold_type'] = 'skipped'
                            cosmic_result['threshold_value'] = max_skipped
                            cosmic_result['threshold_met'] = (final_skipped is not None and final_skipped <= max_skipped)

                        # Note: redo is wired up here (max_redos > 0), so threshold validation
                        # is meaningful and should be shown even in cosmic_opt_only mode. The
                        # opt_only suppression flag is intentionally NOT set in this branch.

                        update_protocol_cache(cosmic_key, 'completed',
                                            result=cosmic_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                    # Check if workflow should pause after optimization stage
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused after optimization
                    
                    # Check if workflow should pause after cosmic stage (check next stage for pause marker)
                    if stage_idx + 1 < len(stages):
                        cosmic_stage = stages[stage_idx + 1]
                        if not check_workflow_pause(cosmic_stage, stage_num + 1, len(stages), cache_file, use_cache):
                            return 0  # Paused after cosmic
                    
                    # Skip both optimization and cosmic stages since we handled them
                    stage_idx += 2
                    completed_stage_count = stage_num + 1
                    context.completed_stage_count = completed_stage_count
                else:
                    # Standalone optimization without cosmic
                    result = execute_optimization_stage(context, stage)

                    if result == 0 and use_cache:
                        opt_result: Dict[str, Any] = {}
                        if hasattr(context, 'optimization_completed'):
                            opt_result['completed'] = context.optimization_completed
                        if hasattr(context, 'optimization_total'):
                            opt_result['total'] = context.optimization_total
                        if hasattr(context, 'optimization_total_cpu_time'):
                            opt_result['total_cpu_time'] = context.optimization_total_cpu_time
                        update_protocol_cache(stage_key, 'completed',
                                              result=opt_result, cache_file=cache_file)
                        if cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                    # Check if workflow should pause after this stage
                    if result == 0:
                        if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                            return 0  # Paused successfully

                    stage_idx += 1
                    completed_stage_count = stage_num
                    context.completed_stage_count = completed_stage_count
                    
            elif stage_type == 'cosmic':
                # Standalone cosmic (not after optimization/refinement in combined mode)
                result = execute_cosmic_stage(context, stage)
                
                # Check if previous stage was optimization or refinement with threshold requirements
                if result == 0 and stage_idx > 0:
                    prev_stage = stages[stage_idx - 1]
                    prev_stage_type = prev_stage['type']
                    
                    if prev_stage_type in ['optimization', 'refinement']:
                        # Check if previous stage had redo parameters
                        prev_args = prev_stage['args']
                        max_critical = 0  # Default: 0% critical structures allowed (retry all)
                        max_skipped = None
                        max_redos = 3
                        
                        for arg in prev_args:
                            if arg.startswith('--critical='):
                                max_critical = float(arg.split('=')[1])
                            elif arg.startswith('--skipped='):
                                max_skipped = float(arg.split('=')[1])
                            elif arg.startswith('--redo='):
                                max_redos = int(arg.split('=')[1])
                        
                        # If thresholds are set and redo is enabled, check results
                        if (max_critical is not None or max_skipped is not None) and max_redos > 1:
                            summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                            
                            if os.path.exists(summary_file):
                                critical_pct, skipped_pct = parse_cosmic_summary(summary_file)
                                
                                if context.workflow_verbose_level >= 1:
                                    print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")
                                
                                threshold_met = True
                                if max_critical is not None:
                                    threshold_met = critical_pct <= max_critical
                                    if not threshold_met:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                                        print(f"  Invalidating cache and re-running {prev_stage_type} with redo logic...")
                                        
                                        # Invalidate both previous stage and cosmic stages
                                        prev_key = f"{prev_stage_type}_{stage_idx}"  # Previous stage
                                        cosmic_key = f"cosmic_{stage_num}"          # Current stage
                                        
                                        if use_cache and 'stages' in cache:
                                            if prev_key in cache['stages']:
                                                del cache['stages'][prev_key]
                                            if cosmic_key in cache['stages']:
                                                del cache['stages'][cosmic_key]
                                            
                                            # Save updated cache
                                            with open(cache_file, 'wb') as f:
                                                pickle.dump(cache, f)
                                        
                                        # Go back to re-execute previous stage
                                        stage_idx -= 1
                                        continue
                                    else:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                        
                                elif max_skipped is not None:
                                    threshold_met = skipped_pct <= max_skipped
                                    if not threshold_met:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                                        print(f"  Invalidating cache and re-running {prev_stage_type} with redo logic...")
                                        
                                        # Invalidate both previous stage and cosmic stages
                                        prev_key = f"{prev_stage_type}_{stage_idx}"
                                        cosmic_key = f"cosmic_{stage_num}"
                                        
                                        if use_cache and 'stages' in cache:
                                            if prev_key in cache['stages']:
                                                del cache['stages'][prev_key]
                                            if cosmic_key in cache['stages']:
                                                del cache['stages'][cosmic_key]
                                            
                                            # Save updated cache
                                            with open(cache_file, 'wb') as f:
                                                pickle.dump(cache, f)
                                        
                                        # Go back to re-execute previous stage
                                        stage_idx -= 1
                                        continue
                                    else:
                                        if context.workflow_verbose_level >= 1:
                                            print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                
                # Save cosmic result to cache
                if result == 0 and use_cache:
                    from datetime import datetime as dt_sim
                    cosmic_key = f"cosmic_{stage_num}"
                    cosmic_result: Dict[str, Any] = {
                        'status': 'completed',
                        'end_time': dt_sim.now().isoformat()
                    }
                    
                    # Store directories for stage memory
                    cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic"

                    # Determine input directory and output prefix based on previous stage
                    # Check if previous stage was refinement or optimization
                    prev_was_opt = (stage_idx > 0 and stages[stage_idx - 1]['type'] == 'refinement')
                    
                    if prev_was_opt:
                        # After refinement: input from refinement dir, output to umotifs
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "geometry_refinement"
                        cosmic_result['input_dir'] = opt_dir
                        cosmic_result['output_dir'] = os.path.join(cosmic_dir, "umotifs")
                    else:
                        # After optimization: input from optimization dir, output to motifs
                        optimization_dir_path = context.optimization_stage_dir if context.optimization_stage_dir else "calculation"
                        cosmic_result['input_dir'] = optimization_dir_path
                        cosmic_result['output_dir'] = os.path.join(cosmic_dir, "motifs")
                    
                    cosmic_result['working_dir'] = cosmic_dir
                    
                    # Add cosmic folder and motifs info if available
                    if hasattr(context, 'cosmic_folder') and context.cosmic_folder:
                        cosmic_result['cosmic_folder'] = context.cosmic_folder
                    if hasattr(context, 'cosmic_motifs_created') and context.cosmic_motifs_created is not None:
                        cosmic_result['motifs_created'] = context.cosmic_motifs_created
                    cosmic_stage_num_s = stage_num
                    input_cnt_s = getattr(context, 'cosmic_stage_input_counts', {}).get(cosmic_stage_num_s)
                    if input_cnt_s:
                        cosmic_result['input_count'] = input_cnt_s
                    
                    # Extract critical and skipped percentages and counts
                    summary_file = os.path.join(cosmic_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        critical_pct, skipped_pct = parse_cosmic_summary(summary_file)
                        if critical_pct is not None:
                            cosmic_result['critical_pct'] = critical_pct
                        if skipped_pct is not None:
                            cosmic_result['skipped_pct'] = skipped_pct
                        
                        critical_count, skipped_count = parse_cosmic_output(cosmic_dir)
                        cosmic_result['critical_count'] = critical_count
                        cosmic_result['skipped_count'] = skipped_count
                    
                    # Extract threshold value from command args
                    cosmic_args = stage.get('args', [])
                    for arg in cosmic_args:
                        if arg.startswith('--th=') or arg.startswith('--threshold='):
                            raw_val = arg.split('=', 1)[1]
                            try:
                                cosmic_result['threshold'] = float(raw_val)
                            except ValueError:
                                cosmic_result['threshold'] = raw_val
                            break

                    # Flag opt-only mode so protocol summary suppresses critical/skipped display
                    if getattr(context, 'cosmic_opt_only', False):
                        cosmic_result['opt_only'] = True

                    update_protocol_cache(cosmic_key, 'completed',
                                        result=cosmic_result, cache_file=cache_file)
                    if use_cache and cache_file:
                        generate_protocol_summary(cache_file=cache_file)

                if result != 0:
                    print(f"\nError: Optimization failed with code {result}")
                    return result

                # Check if workflow should pause after this stage
                if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                    return 0  # Paused successfully

                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count

            elif stage_type == 'refinement':
                # Check if next stage is cosmic - if so, handle opt+cosmic with retry
                next_is_cosmic = (stage_idx + 1 < len(stages) and 
                                     stages[stage_idx + 1]['type'] == 'cosmic')
                
                if next_is_cosmic:
                    # Extract redo parameters from opt stage
                    opt_args = stage['args']
                    max_redos = 3   # Default redo budget; overridden by --redo=N
                    max_critical = 0  # Default: 0% critical threshold (retry all)
                    max_skipped = None
                    concurrent_jobs = 1
                    _skipped_set = False
                    _concurrent_given = False

                    for arg in opt_args:
                        if arg.startswith('--redo='):
                            try:
                                max_redos = max(0, int(arg.split('=')[1]))
                            except ValueError:
                                pass
                        elif arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                            _skipped_set = True
                        elif arg.startswith('--concurrent='):
                            try:
                                concurrent_jobs = max(1, int(arg.split('=')[1]))
                                _concurrent_given = True
                            except ValueError:
                                pass

                    # If --skipped given, use skipped threshold instead of default critical=0
                    if _skipped_set:
                        max_critical = None

                    # No --concurrent= in stage args ⇒ silent default of 1.

                    # Show redo configuration
                    if max_redos > 1:
                        if context.workflow_verbose_level >= 1:
                            if max_critical is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target critical ≤ {max_critical}%")
                            elif max_skipped is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target skipped ≤ {max_skipped}%")

                    # Redo loop for opt+cosmic
                    # attempt 0 = initial run, attempts 1..max_redos = redo attempts
                    final_attempt = 0
                    initial_critical = None  # Track initial critical % from first attempt
                    initial_skipped = None   # Track initial skipped % from first attempt
                    initial_critical_count = None
                    initial_skipped_count = None
                    for attempt in range(max_redos + 1):
                        final_attempt = attempt
                        # Publish the redo counter from attempt 0 so the panel and
                        # 'ascec status' show "(Redo 0/N)" during the initial run.
                        if max_redos > 0:
                            context.active_redo_stage_num = stage_num
                            context.active_redo_attempt = attempt
                            context.active_redo_max = max_redos
                            _upd = getattr(context, 'update_progress', None)
                            if callable(_upd):
                                _upd("")
                            _record_redo_attempt(cache_file, f"refinement_{stage_num}",
                                                 attempt, max_redos, use_cache)
                        if attempt > 0:
                            if context.workflow_verbose_level >= 1:
                                print(f"\n{'-' * 60}")
                                print(f"Redo {attempt}/{max_redos}")

                        # Run refinement (includes organizing/copying files to cosmic)
                        result = execute_refinement_stage(context, stage)
                        if result != 0:
                            print(f"\nError: Refinement failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"refinement_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                                print(f"  Cache invalidated for stage {stage_num}")
                            return result
                        
                        # Note: on redo (attempt > 0), execute_refinement_stage's internal
                        # sort step already re-runs collect_out_files_with_tracking, which (a)
                        # places the freshly recalculated .out files into the canonical orca_out
                        # folder and (b) wipes stale sibling out folders + previous cosmic
                        # artifacts (motifs_*, clustering_summary, skipped_structures, ...).
                        # No outer hand-copy or cleanup needed here.


                        # Run cosmic; stage header is only shown in verbose mode.
                        if attempt == 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"[{stage_idx + 2}/{len(stages)}] cosmic")
                            print('-' * 60)

                        cosmic_stage = stages[stage_idx + 1]
                        if use_cache:
                            cosmic_key = f"cosmic_{stage_num + 1}"
                            update_protocol_cache(cosmic_key, 'in_progress', cache_file=cache_file)
                        result = execute_cosmic_stage(context, cosmic_stage)
                        if result != 0:
                            print(f"\nError: cosmic failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"refinement_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                            return result

                        # Parse cosmic results from clustering_summary.txt
                        summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                        if os.path.exists(summary_file):
                            critical_pct, skipped_pct = parse_cosmic_summary(summary_file)

                            # Capture initial values on first attempt
                            if attempt == 0:
                                initial_critical = critical_pct
                                initial_skipped = skipped_pct
                                # Get counts for initial attempt
                                cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic_2"
                                init_crit_count, init_skip_count = parse_cosmic_output(cosmic_dir)
                                initial_critical_count = init_crit_count
                                initial_skipped_count = init_skip_count
                            
                            if context.workflow_verbose_level >= 1:
                                print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")
                            
                            # Check thresholds
                            threshold_met = True
                            
                            if max_critical is not None:
                                threshold_met = critical_pct <= max_critical
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                            elif max_skipped is not None:
                                threshold_met = skipped_pct <= max_skipped
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                            else:
                                break
                            
                            # If threshold not met and attempts remain, prepare for redo
                            if not threshold_met:
                                if attempt < max_redos:
                                    # Loop will continue to next iteration
                                    pass
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"Max attempts reached")
                        else:
                            if context.workflow_verbose_level >= 1:
                                print("⚠ Warning: Could not find clustering_summary.txt")
                            break
                    
                    # Store redo count for terminal summary
                    context.active_redo_stage_num = 0
                    context.active_redo_attempt = 0
                    context.active_redo_max = 0
                    context.last_ref_redo_count = final_attempt

                    # Get final cosmic results for cache
                    final_critical = None
                    final_skipped = None
                    summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        final_critical, final_skipped = parse_cosmic_summary(summary_file)

                    # Update cache for both refinement and cosmic stages
                    if use_cache:
                        opt_key = f"refinement_{stage_num}"
                        cosmic_key = f"cosmic_{stage_num + 1}"
                        
                        # Build result data
                        opt_result: Dict[str, Any] = {
                            'attempts': final_attempt,
                            'max_redos': max_redos,
                        }
                        
                        # Store directories for stage memory
                        # Get input dir from previous cosmic stage
                        motifs_dir = context.get_previous_stage_output_dir('cosmic')
                        if not motifs_dir:
                            motifs_dir = "cosmic/motifs"  # Fallback
                        
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "geometry_refinement"
                        opt_result['input_dir'] = motifs_dir
                        opt_result['working_dir'] = opt_dir
                        opt_result['output_dir'] = opt_dir
                        
                        if max_critical is not None:
                            opt_result['critical_threshold'] = max_critical
                        if max_skipped is not None:
                            opt_result['skipped_threshold'] = max_skipped
                        
                        # Add motifs source info if available
                        if hasattr(context, 'refinement_motifs_source') and context.refinement_motifs_source:
                            opt_result['motifs_source'] = context.refinement_motifs_source
                        
                        # Add completion counts if available
                        if hasattr(context, 'refinement_completed') and context.refinement_completed is not None:
                            opt_result['completed'] = context.refinement_completed
                        if hasattr(context, 'refinement_total') and context.refinement_total is not None:
                            opt_result['total'] = context.refinement_total

                        # Add concurrent jobs count and total CPU time for protocol summary
                        opt_result['concurrent_jobs'] = concurrent_jobs
                        if hasattr(context, 'refinement_total_cpu_time'):
                            opt_result['total_cpu_time'] = context.refinement_total_cpu_time

                        # Add cosmic folder info if available
                        if hasattr(context, 'refinement_cosmic_folder') and context.refinement_cosmic_folder:
                            opt_result['cosmic_folder'] = context.refinement_cosmic_folder

                        update_protocol_cache(opt_key, 'completed', result=opt_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                        cosmic_result = {}

                        # Store directories for stage memory
                        cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic_2"
                        opt_dir = context.refinement_stage_dir if context.refinement_stage_dir else "geometry_refinement"
                        cosmic_result['input_dir'] = opt_dir  # Read from refinement
                        cosmic_result['working_dir'] = cosmic_dir
                        # After optimization: use "umotifs" prefix (unique motifs, second level)
                        cosmic_result['output_dir'] = os.path.join(cosmic_dir, "umotifs")  # Unique motifs
                        
                        if final_critical is not None:
                            cosmic_result['critical_pct'] = final_critical
                        if final_skipped is not None:
                            cosmic_result['skipped_pct'] = final_skipped
                        
                        # Extract critical and skipped counts from cosmic output
                        critical_count, skipped_count = parse_cosmic_output(cosmic_dir)
                        cosmic_result['critical_count'] = critical_count
                        cosmic_result['skipped_count'] = skipped_count
                        
                        # Extract threshold value from cosmic command args
                        cosmic_stage = stages[stage_idx + 1] if stage_idx + 1 < len(stages) else {}
                        cosmic_args = cosmic_stage.get('args', [])
                        for arg in cosmic_args:
                            if arg.startswith('--th=') or arg.startswith('--threshold='):
                                raw_val = arg.split('=', 1)[1]
                                try:
                                    cosmic_result['threshold'] = float(raw_val)
                                except ValueError:
                                    cosmic_result['threshold'] = raw_val
                            elif arg.startswith('--rmsd='):
                                rmsd_val = float(arg.split('=')[1])
                                cosmic_result['rmsd_threshold'] = rmsd_val
                        
                        # Add cosmic folder and motifs info if available
                        if hasattr(context, 'cosmic_folder'):
                            cosmic_result['cosmic_folder'] = context.cosmic_folder
                        if hasattr(context, 'cosmic_motifs_created'):
                            cosmic_result['motifs_created'] = context.cosmic_motifs_created
                        ref_cosmic_stage_num = stage_num + 1
                        input_cnt_r = getattr(context, 'cosmic_stage_input_counts', {}).get(ref_cosmic_stage_num)
                        if input_cnt_r:
                            cosmic_result['input_count'] = input_cnt_r

                        # Add initial validation values (from first attempt)
                        if initial_critical is not None:
                            cosmic_result['initial_critical'] = initial_critical
                        if initial_skipped is not None:
                            cosmic_result['initial_skipped'] = initial_skipped
                        if initial_critical_count is not None:
                            cosmic_result['initial_critical_count'] = initial_critical_count
                        if initial_skipped_count is not None:
                            cosmic_result['initial_skipped_count'] = initial_skipped_count
                        
                        # Add threshold info and attempts
                        cosmic_result['attempts'] = final_attempt
                        if max_critical is not None:
                            cosmic_result['threshold_type'] = 'critical'
                            cosmic_result['threshold_value'] = max_critical
                            cosmic_result['threshold_met'] = (final_critical is not None and final_critical <= max_critical)
                        elif max_skipped is not None:
                            cosmic_result['threshold_type'] = 'skipped'
                            cosmic_result['threshold_value'] = max_skipped
                            cosmic_result['threshold_met'] = (final_skipped is not None and final_skipped <= max_skipped)

                        # Note: redo is wired up here (max_redos > 0), so threshold validation
                        # is meaningful and should be shown even in cosmic_opt_only mode. The
                        # opt_only suppression flag is intentionally NOT set in this branch.

                        update_protocol_cache(cosmic_key, 'completed', result=cosmic_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                    # Check if workflow should pause after refinement stage
                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0  # Paused after refinement
                    
                    # Check if workflow should pause after cosmic stage (check next stage for pause marker)
                    if stage_idx + 1 < len(stages):
                        cosmic_stage = stages[stage_idx + 1]
                        if not check_workflow_pause(cosmic_stage, stage_num + 1, len(stages), cache_file, use_cache):
                            return 0  # Paused after cosmic
                    
                    # Skip the cosmic stage (already executed)
                    stage_idx += 2
                    completed_stage_count = stage_num + 1
                    context.completed_stage_count = completed_stage_count
                    continue
                else:
                    # No cosmic after refinement - just run it once
                    result = execute_refinement_stage(context, stage)
                
                # Build and save refinement result if successful (for standalone refinement)
                if result == 0 and use_cache and not next_is_cosmic:
                    opt_key = f"refinement_{stage_num}"
                    
                    # Extract optimization parameters from stage
                    max_skipped = None
                    for arg in stage.get('args', []):
                        if arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                    
                    from datetime import datetime as dt_now  # Local import to avoid scope issues
                    opt_result = {
                        'status': 'completed',
                        'end_time': dt_now.now().isoformat()
                    }
                    
                    # Add threshold info
                    if max_skipped is not None:
                        opt_result['skipped_threshold'] = max_skipped
                    
                    # Add detailed execution data from context
                    if hasattr(context, 'refinement_motifs_source') and context.refinement_motifs_source:
                        opt_result['motifs_source'] = context.refinement_motifs_source
                    
                    if hasattr(context, 'refinement_completed') and context.refinement_completed is not None:
                        opt_result['completed'] = context.refinement_completed
                    if hasattr(context, 'refinement_total') and context.refinement_total is not None:
                        opt_result['total'] = context.refinement_total
                    
                    if hasattr(context, 'refinement_cosmic_folder') and context.refinement_cosmic_folder:
                        opt_result['cosmic_folder'] = context.refinement_cosmic_folder
                    
                    # Save to cache
                    update_protocol_cache(opt_key, 'completed',
                                        result=opt_result, cache_file=cache_file)
                    if use_cache and cache_file:
                        generate_protocol_summary(cache_file=cache_file)

                if result != 0:
                    print(f"\nError: Refinement failed with code {result}")
                    return result

                # Check if workflow should pause after this stage
                if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                    return 0  # Paused successfully

                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count

            elif stage_type == 'energy_refinement':
                # Energy refinement: single-point calculations on motifs, then composite COSMIC
                next_is_cosmic = (stage_idx + 1 < len(stages) and
                                  stages[stage_idx + 1]['type'] == 'cosmic')

                if next_is_cosmic:
                    # Extract redo parameters
                    opt_args = stage['args']
                    max_redos = 3   # Default redo budget; overridden by --redo=N
                    max_critical = 0  # Default: 0% critical threshold (retry all)
                    max_skipped = None
                    concurrent_jobs = 1
                    _skipped_set = False
                    _concurrent_given = False

                    for arg in opt_args:
                        if arg.startswith('--redo='):
                            try:
                                max_redos = max(0, int(arg.split('=')[1]))
                            except ValueError:
                                pass
                        elif arg.startswith('--skipped='):
                            max_skipped = float(arg.split('=')[1])
                            _skipped_set = True
                        elif arg.startswith('--concurrent='):
                            try:
                                concurrent_jobs = max(1, int(arg.split('=')[1]))
                                _concurrent_given = True
                            except ValueError:
                                pass

                    # If --skipped given, use skipped threshold instead of default critical=0
                    if _skipped_set:
                        max_critical = None

                    # No --concurrent= in stage args ⇒ silent default of 1.

                    if max_redos > 1:
                        if context.workflow_verbose_level >= 1:
                            if max_critical is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target critical ≤ {max_critical}%")
                            elif max_skipped is not None:
                                print(f"  Stage redo enabled: max {max_redos} redo attempts, target skipped ≤ {max_skipped}%")

                    final_attempt = 0
                    initial_critical = None
                    initial_skipped = None
                    initial_critical_count = None
                    initial_skipped_count = None
                    for attempt in range(max_redos + 1):
                        final_attempt = attempt
                        # Publish the redo counter from attempt 0 so the panel and
                        # 'ascec status' show "(Redo 0/N)" during the initial run.
                        if max_redos > 0:
                            context.active_redo_stage_num = stage_num
                            context.active_redo_attempt = attempt
                            context.active_redo_max = max_redos
                            _upd = getattr(context, 'update_progress', None)
                            if callable(_upd):
                                _upd("")
                            _record_redo_attempt(cache_file, f"energy_refinement_{stage_num}",
                                                 attempt, max_redos, use_cache)
                        if attempt > 0:
                            if context.workflow_verbose_level >= 1:
                                print(f"\n{'-' * 60}")
                                print(f"Redo {attempt}/{max_redos}")

                        result = execute_energy_refinement_stage(context, stage)
                        if result != 0:
                            print(f"\nError: Energy refinement failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"energy_refinement_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                                print(f"  Cache invalidated for stage {stage_num}")
                            return result

                        # Redo: copy recalculated files to cosmic orca_out folder
                        if attempt > 0 and hasattr(context, 'recalculated_files') and context.recalculated_files:
                            opt_dir = getattr(context, 'energy_refinement_stage_dir', 'energy_refinement') or 'energy_refinement'
                            cosmic_dir = getattr(context, 'eref_cosmic_folder', None) or context.cosmic_dir

                            # Derive the base cosmic directory (strip orca_out_* suffix if present)
                            _ecf = getattr(context, 'eref_cosmic_folder', None)
                            base_cosmic_dir = _cosmic_base_name(_ecf) if isinstance(_ecf, str) and _ecf else (
                                os.path.dirname(cosmic_dir) if cosmic_dir and 'orca_out' in cosmic_dir else cosmic_dir
                            )

                            # Find the orca_out directory in cosmic (deterministic target).
                            # Use eref_completed to select the matching orca_out_N subdir,
                            # the same way execute_cosmic_stage selects it.
                            cosmic_orca_dir = None
                            if base_cosmic_dir and os.path.isdir(base_cosmic_dir):
                                _expected = getattr(context, 'eref_completed', None)
                                _orca_dirs = sorted(glob.glob(os.path.join(base_cosmic_dir, "orca_out*")), key=natural_sort_key)
                                if _orca_dirs:
                                    if _expected is not None:
                                        _m_dirs = [d for d in _orca_dirs
                                                   if re.search(r'_(\d+)$', os.path.basename(d)) and
                                                   int(re.search(r'_(\d+)$', os.path.basename(d)).group(1)) == _expected]
                                        cosmic_orca_dir = _m_dirs[0] if _m_dirs else _orca_dirs[-1]
                                    else:
                                        cosmic_orca_dir = _orca_dirs[-1]
                            elif cosmic_dir and ('orca_out' in cosmic_dir or 'gaussian_out' in cosmic_dir):
                                cosmic_orca_dir = cosmic_dir

                            if cosmic_orca_dir:
                                if base_cosmic_dir and os.path.exists(base_cosmic_dir):
                                    items_to_remove = [
                                        'dendrogram_images', 'extracted_clusters', 'extracted_data',
                                        'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt'
                                    ]
                                    for item in os.listdir(base_cosmic_dir):
                                        if item.startswith('motifs_') or item.startswith('umotifs_'):
                                            items_to_remove.append(item)
                                    for item in items_to_remove:
                                        item_path = os.path.join(base_cosmic_dir, item)
                                        if os.path.exists(item_path):
                                            try:
                                                if os.path.isdir(item_path):
                                                    shutil.rmtree(item_path)
                                                else:
                                                    os.remove(item_path)
                                            except Exception:
                                                pass

                                for basename in context.recalculated_files:
                                    short_name = basename.replace('_opt', '').replace('_calc', '')
                                    opt_subdir = os.path.join(opt_dir, short_name)
                                    opt_out_file = os.path.join(opt_subdir, f"{basename}.out")
                                    if os.path.exists(opt_out_file):
                                        cosmic_out_file = os.path.join(cosmic_orca_dir, f"{basename}.out")
                                        shutil.copy2(opt_out_file, cosmic_out_file)
                            else:
                                print(f"\n  Warning: No orca_out directory found (cosmic_dir={cosmic_dir})")

                        if attempt == 0 and context.workflow_verbose_level >= 1:
                            print(f"\n{'-' * 60}")
                            print(f"[{stage_idx + 2}/{len(stages)}] cosmic")
                            print('-' * 60)

                        cosmic_stage = stages[stage_idx + 1]
                        if use_cache:
                            cosmic_key = f"cosmic_{stage_num + 1}"
                            update_protocol_cache(cosmic_key, 'in_progress', cache_file=cache_file)
                        result = execute_cosmic_stage(context, cosmic_stage)
                        if result != 0:
                            print(f"\nError: cosmic failed with code {result}")
                            if use_cache:
                                invalidate_stage_cache(
                                    cache_file,
                                    f"energy_refinement_{stage_num}",
                                    f"cosmic_{stage_num + 1}",
                                )
                            return result

                        summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                        if os.path.exists(summary_file):
                            critical_pct, skipped_pct = parse_cosmic_summary(summary_file)

                            if attempt == 0:
                                initial_critical = critical_pct
                                initial_skipped = skipped_pct
                                cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic_3"
                                init_crit_count, init_skip_count = parse_cosmic_output(cosmic_dir)
                                initial_critical_count = init_crit_count
                                initial_skipped_count = init_skip_count

                            if context.workflow_verbose_level >= 1:
                                print(f"\nResults: {critical_pct:.1f}% critical, {skipped_pct:.1f}% skipped")

                            threshold_met = True
                            if max_critical is not None:
                                threshold_met = critical_pct <= max_critical
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (critical ≤ {max_critical}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (critical {critical_pct:.1f}% > {max_critical}%)")
                            elif max_skipped is not None:
                                threshold_met = skipped_pct <= max_skipped
                                if threshold_met:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold met (skipped ≤ {max_skipped}%)")
                                    break
                                else:
                                    if context.workflow_verbose_level >= 1:
                                        print(f"→ Threshold exceeded (skipped {skipped_pct:.1f}% > {max_skipped}%)")
                            else:
                                break

                            if not threshold_met and attempt >= max_redos:
                                if context.workflow_verbose_level >= 1:
                                    print(f"Max attempts reached")
                        else:
                            if context.workflow_verbose_level >= 1:
                                print("⚠ Warning: Could not find clustering_summary.txt")
                            break

                    # Store redo count for terminal summary
                    context.active_redo_stage_num = 0
                    context.active_redo_attempt = 0
                    context.active_redo_max = 0
                    context.last_eref_redo_count = final_attempt

                    # Cache results for energy_refinement and cosmic stages
                    final_critical = None
                    final_skipped = None
                    summary_file = os.path.join(context.cosmic_dir, "clustering_summary.txt")
                    if os.path.exists(summary_file):
                        final_critical, final_skipped = parse_cosmic_summary(summary_file)

                    if use_cache:
                        opt_key = f"energy_refinement_{stage_num}"
                        cosmic_key = f"cosmic_{stage_num + 1}"

                        opt_result: Dict[str, Any] = {
                            'attempts': final_attempt,
                            'max_redos': max_redos,
                        }

                        motifs_dir = context.get_previous_stage_output_dir('cosmic')
                        if not motifs_dir:
                            motifs_dir = "cosmic_2/motifs"

                        opt_dir = getattr(context, 'energy_refinement_stage_dir', 'energy_refinement') or 'energy_refinement'
                        opt_result['input_dir'] = motifs_dir
                        opt_result['working_dir'] = opt_dir
                        opt_result['output_dir'] = opt_dir

                        if max_critical is not None:
                            opt_result['critical_threshold'] = max_critical
                        if max_skipped is not None:
                            opt_result['skipped_threshold'] = max_skipped

                        eref_ms = getattr(context, 'eref_motifs_source', None)
                        if eref_ms:
                            opt_result['motifs_source'] = eref_ms
                        eref_c = getattr(context, 'eref_completed', None)
                        if eref_c is not None:
                            opt_result['completed'] = eref_c
                        eref_t = getattr(context, 'eref_total', None)
                        if eref_t is not None:
                            opt_result['total'] = eref_t

                        opt_result['concurrent_jobs'] = concurrent_jobs
                        eref_cpu = getattr(context, 'eref_total_cpu_time', None)
                        if eref_cpu is not None:
                            opt_result['total_cpu_time'] = eref_cpu

                        eref_cf = getattr(context, 'eref_cosmic_folder', None)
                        if eref_cf:
                            opt_result['cosmic_folder'] = eref_cf

                        update_protocol_cache(opt_key, 'completed', result=opt_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                        cosmic_result = {}
                        cosmic_dir = context.cosmic_dir if context.cosmic_dir else "cosmic_3"
                        cosmic_result['input_dir'] = opt_dir
                        cosmic_result['working_dir'] = cosmic_dir
                        cosmic_result['output_dir'] = os.path.join(cosmic_dir, "umotifs")

                        if final_critical is not None:
                            cosmic_result['critical_pct'] = final_critical
                        if final_skipped is not None:
                            cosmic_result['skipped_pct'] = final_skipped

                        critical_count, skipped_count = parse_cosmic_output(cosmic_dir)
                        cosmic_result['critical_count'] = critical_count
                        cosmic_result['skipped_count'] = skipped_count

                        cosmic_stage = stages[stage_idx + 1] if stage_idx + 1 < len(stages) else {}
                        cosmic_args = cosmic_stage.get('args', [])
                        for arg in cosmic_args:
                            if arg.startswith('--th=') or arg.startswith('--threshold='):
                                raw_val = arg.split('=', 1)[1]
                                try:
                                    cosmic_result['threshold'] = float(raw_val)
                                except ValueError:
                                    cosmic_result['threshold'] = raw_val
                            elif arg.startswith('--rmsd='):
                                cosmic_result['rmsd_threshold'] = float(arg.split('=')[1])

                        if hasattr(context, 'cosmic_folder'):
                            cosmic_result['cosmic_folder'] = context.cosmic_folder
                        if hasattr(context, 'cosmic_motifs_created'):
                            cosmic_result['motifs_created'] = context.cosmic_motifs_created

                        ref_cosmic_stage_num = stage_num + 1
                        input_cnt_r = getattr(context, 'cosmic_stage_input_counts', {}).get(ref_cosmic_stage_num)
                        if input_cnt_r:
                            cosmic_result['input_count'] = input_cnt_r

                        if initial_critical is not None:
                            cosmic_result['initial_critical'] = initial_critical
                        if initial_skipped is not None:
                            cosmic_result['initial_skipped'] = initial_skipped
                        if initial_critical_count is not None:
                            cosmic_result['initial_critical_count'] = initial_critical_count
                        if initial_skipped_count is not None:
                            cosmic_result['initial_skipped_count'] = initial_skipped_count

                        cosmic_result['attempts'] = final_attempt
                        if max_critical is not None:
                            cosmic_result['threshold_type'] = 'critical'
                            cosmic_result['threshold_value'] = max_critical
                            cosmic_result['threshold_met'] = (final_critical is not None and final_critical <= max_critical)
                        elif max_skipped is not None:
                            cosmic_result['threshold_type'] = 'skipped'
                            cosmic_result['threshold_value'] = max_skipped
                            cosmic_result['threshold_met'] = (final_skipped is not None and final_skipped <= max_skipped)

                        # Note: redo is wired up here (max_redos > 0), so threshold validation
                        # is meaningful and should be shown even in cosmic_opt_only mode. The
                        # opt_only suppression flag is intentionally NOT set in this branch.

                        update_protocol_cache(cosmic_key, 'completed', result=cosmic_result, cache_file=cache_file)
                        if use_cache and cache_file:
                            generate_protocol_summary(cache_file=cache_file)

                    if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                        return 0

                    if stage_idx + 1 < len(stages):
                        cosmic_stage = stages[stage_idx + 1]
                        if not check_workflow_pause(cosmic_stage, stage_num + 1, len(stages), cache_file, use_cache):
                            return 0

                    stage_idx += 2
                    completed_stage_count = stage_num + 1
                    context.completed_stage_count = completed_stage_count
                    continue
                else:
                    # No cosmic after eref - standalone run
                    result = execute_energy_refinement_stage(context, stage)

                if result == 0 and use_cache and not next_is_cosmic:
                    opt_key = f"energy_refinement_{stage_num}"
                    from datetime import datetime as dt_now
                    opt_result = {'status': 'completed', 'end_time': dt_now.now().isoformat()}
                    eref_ms = getattr(context, 'eref_motifs_source', None)
                    if eref_ms:
                        opt_result['motifs_source'] = eref_ms
                    eref_c = getattr(context, 'eref_completed', None)
                    if eref_c is not None:
                        opt_result['completed'] = eref_c
                    eref_t = getattr(context, 'eref_total', None)
                    if eref_t is not None:
                        opt_result['total'] = eref_t
                    eref_cf = getattr(context, 'eref_cosmic_folder', None)
                    if eref_cf:
                        opt_result['cosmic_folder'] = eref_cf
                    update_protocol_cache(opt_key, 'completed', result=opt_result, cache_file=cache_file)
                    if use_cache and cache_file:
                        generate_protocol_summary(cache_file=cache_file)

                if result != 0:
                    print(f"\nError: Energy refinement failed with code {result}")
                    return result

                if not check_workflow_pause(stage, stage_num, len(stages), cache_file, use_cache):
                    return 0

                stage_idx += 1
                completed_stage_count = stage_num
                context.completed_stage_count = completed_stage_count
            else:
                print(f"Error: Unknown stage type '{stage_type}'")
                return 1
                
        except KeyboardInterrupt:
            print(f"\nInterrupted by user during stage {stage_num} ({stage_type}).")
            if _job_id:
                _update_ascec_job(_job_id, 'killed')
            _remove_progress_file()
            raise
        except Exception as e:
            print(f"\nError executing stage {stage_num} ({stage_type}): {e}")
            traceback.print_exc()
            _remove_progress_file()
            return 1
    
    if context.workflow_verbose_level >= 1:
        print(f"\n{'-' * 60}")
        print("✓ Workflow completed")
        print(f"{'-' * 60}")
    else:
        render_final_workflow_summary()

    # Export final ensemble at root directory from the last cosmic stage.
    copy_final_ensemble_to_root()
    
    # Clean up temporary folders from retry attempts (final safety cleanup)
    temp_calc_folders = glob.glob("calculation_tmp_*")
    temp_cosmic_folders = glob.glob("cosmic_tmp_*") + glob.glob("COSMIC_tmp_*")
    retry_input = ["retry_input"] if os.path.exists("retry_input") else []
    all_temp = temp_calc_folders + temp_cosmic_folders + retry_input
    
    if all_temp:
        if context.workflow_verbose_level >= 1:
            print("\nCleaning up temporary folders...")
        for folder in all_temp:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                if context.workflow_verbose_level >= 1:
                    print(f"  Removed: {folder}")
        if context.workflow_verbose_level >= 1:
            print(f"  Cleaned {len(all_temp)} temporary folder(s)")
    
    # If using cache (protocol mode), generate summary
    # NOTE: Cache is NOT deleted to allow protocol resume
    if use_cache:
        generate_protocol_summary(cache_file=cache_file)
        if context.workflow_verbose_level >= 1:
            print(f"\n→ Protocol cache saved: {cache_file}")

    # Remove temporary template files extracted from embedded labels.
    # Keep user-provided real template files untouched.
    if getattr(context, 'generated_template_files', None):
        for temp_tpl in context.generated_template_files:
            try:
                if temp_tpl and os.path.isfile(temp_tpl):
                    os.remove(temp_tpl)
            except OSError:
                pass

    # Miniprint cleanup: reduce disk usage unless --maxprint was specified
    if not context.maxprint:
        miniprint_cleanup()

    # Mark job as completed and restore original streams
    if _job_id:
        _update_ascec_job(_job_id, 'completed')
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    if _log_fh is not None:
        try: _log_fh.close()
        except Exception: pass
    _remove_progress_file()

    return 0

def _extract_replica_error(stderr_capture_file: str) -> Optional[str]:
    """Pull a human-readable failure reason out of a replica's captured stderr.

    A replica that aborts cleanly prints ``ascec: annealing failed: <reason>``
    (see command_line/ascec.py); we surface ``<reason>`` so the workflow can
    tell the user *why* it stopped instead of just "exit code 1". Falls back to
    the last non-empty stderr line (or a trailing traceback line) when no
    tagged message is present.
    """
    try:
        with open(stderr_capture_file, 'r', errors='replace') as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines:
        return None
    for ln in reversed(lines):
        marker = 'annealing failed:'
        if marker in ln:
            return ln.split(marker, 1)[1].strip()
    # No tagged message — return the last stderr line (often a raised message
    # or the final traceback line), which is still more useful than a code.
    return lines[-1]

def execute_replication_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute replication stage (r3, r5, etc.) and run annealing simulations."""
    num_replicas = stage['num_replicas']
    context.num_replicas = num_replicas
    verbose = getattr(context, 'workflow_verbose_level', 0) >= 1
    
    # Parse --box and --concurrent flags from stage args
    box_size_override = None
    concurrent_jobs = 1  # Default: serial replica execution
    args = stage.get('args', [])
    for arg in args:
        if arg.startswith('--concurrent='):
            try:
                concurrent_jobs = max(1, int(arg.split('=')[1]))
            except ValueError:
                pass
        elif arg.startswith('--box'):
            # Extract packing percentage from flag (e.g., --box10 -> 10%)
            try:
                packing_str = arg.replace('--box', '')
                if packing_str:
                    packing_percent = float(packing_str)
                    # Get box size recommendation for this packing percentage
                    recommended_box = get_box_size_recommendation(context.input_file, packing_percent)
                    if recommended_box is not None:
                        box_size_override = recommended_box
                        # Store for protocol summary
                        context.annealing_box_size = box_size_override
                        context.annealing_packing = packing_percent
                        if verbose:
                            print(f"Using recommended box size: {box_size_override:.1f} Å ({packing_percent}% effective packing)")
            except ValueError:
                pass
    
    # Clean up old annealing folder from previous runs to avoid duplicate replicas
    # (Only in workflow mode - standalone mode keeps it for reference)
    annealing_folder = os.path.join(os.path.dirname(context.input_file), "annealing")
    if os.path.exists(annealing_folder):
        try:
            shutil.rmtree(annealing_folder)
            if verbose:
                print(f"Cleaned up previous Annealing folder")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not clean old annealing folder: {e}")
    
    # Create replicated runs (without launcher script in workflow mode)
    replicated_files = create_replicated_runs(
        context.input_file,
        num_replicas,
        create_launcher=False,
        box_size=box_size_override,
        verbose=verbose
    )
    
    if not replicated_files:
        print("✗ Failed to create replicated runs")
        return 1
    
    context.annealing_dirs = [os.path.dirname(f) for f in replicated_files]

    # Optional per-replica seed pinning for reproducible re-runs (D-038 flow).
    # COSMIC_ASCEC_SEEDS is a comma-separated list mapped one seed per replica,
    # in replica order (li5_1, li5_2, ...). Each replica subprocess receives its
    # own COSMIC_ASCEC_SEED so the whole protocol can be reproduced in a single
    # command. A single COSMIC_ASCEC_SEED (no plural) would otherwise be shared
    # by every replica subprocess, forcing them all to identical trajectories.
    replica_seeds: Dict[str, str] = {}
    _seeds_raw = os.environ.get("COSMIC_ASCEC_SEEDS", "").strip()
    if _seeds_raw:
        _seed_list = [s.strip() for s in _seeds_raw.split(",") if s.strip()]
        for _s in _seed_list:
            try:
                int(_s)
            except ValueError:
                raise ValueError(
                    f"COSMIC_ASCEC_SEEDS contains a non-integer value: {_s!r}"
                )
        if len(_seed_list) != len(replicated_files):
            raise ValueError(
                f"COSMIC_ASCEC_SEEDS has {len(_seed_list)} seed(s) but the "
                f"replication stage created {len(replicated_files)} replica(s); "
                f"provide exactly one seed per replica."
            )
        replica_seeds = dict(zip(replicated_files, _seed_list))
        if verbose:
            print(f"Pinning per-replica seeds: {', '.join(_seed_list)}")

    # Actually run the annealing simulations
    if verbose:
        print(f"Running {num_replicas} annealing simulation(s)")
    
    failed_runs = []
    failed_reasons: Dict[str, str] = {}
    progress_cb = getattr(context, 'update_progress', None)
    if not verbose and callable(progress_cb):
        progress_cb(f"0/{num_replicas} ...")
    completed_replicas = 0

    def _replica_already_completed(replica_dir: str, replica_input_name: str) -> bool:
        """Return True when a replica has a completed result set.

        Mode 1 (annealing) emits result_*.xyz + tvse_*.dat + a .out file with a
        normal-termination marker. Mode 0 (random configurations) emits only
        mto_*.xyz / mtobox_*.xyz and no .out file, so a mode-0 replica counts
        as completed as soon as the mto_*.xyz output exists.
        """
        if glob.glob(os.path.join(replica_dir, 'mto_*.xyz')):
            return True

        output_file = os.path.join(replica_dir, os.path.splitext(replica_input_name)[0] + '.out')
        has_result = bool(glob.glob(os.path.join(replica_dir, 'result_*.xyz')))
        has_tvse = bool(glob.glob(os.path.join(replica_dir, 'tvse_*.dat')))
        if not (has_result and has_tvse):
            return False
        if not os.path.exists(output_file):
            return False
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            return ('Normal annealing termination' in content) or ('Annealing simulation finished' in content)
        except OSError:
            return False

    def _run_single_replica(input_file: str, on_step=None) -> Dict[str, Any]:
        """Run a single annealing replica and return result dict.

        on_step: optional callable(step_str) called ~every 2 s with "X/Y" progress
                 while the replica is running. When provided, Popen is used instead
                 of subprocess.run so the parent can poll without blocking.
        """
        run_dir = os.path.dirname(input_file)
        run_name = os.path.basename(run_dir)
        input_basename = os.path.basename(input_file)

        result_info = {
            'input_file': input_file,
            'run_dir': run_dir,
            'run_name': run_name,
            'success': False,
            'last_error': None,
        }

        step_progress_file = os.path.join(run_dir, '.ascec_step')
        # Capture the child's stderr so a failed replica can report *why* it
        # failed (e.g. "no rotatable bonds ...") instead of a bare exit code.
        stderr_capture_file = os.path.join(run_dir, '.ascec_stderr')

        try:
            cmd = [sys.executable, os.path.abspath(sys.argv[0]), input_basename]
            env = {**os.environ, "ASCEC_DISABLE_EMBEDDED_PROTOCOL": "1"}
            # Per-replica seed pin (from COSMIC_ASCEC_SEEDS). Overrides any single
            # COSMIC_ASCEC_SEED inherited from the parent so each replica gets its
            # own seed rather than all sharing one.
            _pinned_seed = replica_seeds.get(input_file)
            if _pinned_seed is not None:
                env["COSMIC_ASCEC_SEED"] = _pinned_seed

            if on_step is not None:
                with open(stderr_capture_file, 'wb') as _errfh:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=run_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=_errfh,
                        env=env,
                        preexec_fn=_PDEATHSIG_PREEXEC,
                    )
                    while proc.poll() is None:
                        try:
                            with open(step_progress_file, 'r') as _spf:
                                line = _spf.readline().strip()
                            if line:
                                on_step(line)
                        except OSError:
                            pass
                        time.sleep(2.0)
                    returncode = proc.returncode
                try:
                    os.unlink(step_progress_file)
                except OSError:
                    pass
            else:
                with open(stderr_capture_file, 'wb') as _errfh:
                    proc = subprocess.run(
                        cmd,
                        cwd=run_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=_errfh,
                        env=env,
                        preexec_fn=_PDEATHSIG_PREEXEC,
                    )
                returncode = proc.returncode

            output_file = os.path.join(run_dir, os.path.splitext(input_basename)[0] + '.out')
            artifacts_exist = bool(glob.glob(os.path.join(run_dir, 'result_*.xyz'))) and bool(
                glob.glob(os.path.join(run_dir, 'tvse_*.dat'))
            )
            # Mode 0 (random configurations) emits mto_*.xyz only, no .out file
            # and no tvse. A mode-0 replica is successful as soon as mto_*.xyz
            # exists with a non-zero return code from the subprocess.
            mode0_artifacts_exist = bool(glob.glob(os.path.join(run_dir, 'mto_*.xyz')))

            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    if 'Normal annealing termination' in content or 'Annealing simulation finished' in content:
                        result_info['success'] = True

            if not result_info['success'] and artifacts_exist:
                result_info['success'] = True

            if not result_info['success'] and mode0_artifacts_exist and returncode == 0:
                result_info['success'] = True

            if not result_info['success']:
                # Check if result files were created
                result_files = []
                for pattern in ['result_*.xyz', 'result_*.mol']:
                    result_files.extend(glob.glob(os.path.join(run_dir, pattern)))
                if result_files:
                    result_info['success'] = True

            if not result_info['success']:
                reason = _extract_replica_error(stderr_capture_file)
                if returncode != 0:
                    result_info['last_error'] = (
                        f"{reason} (exit code {returncode})" if reason
                        else f"Exit code {returncode}"
                    )
                elif not os.path.exists(output_file):
                    result_info['last_error'] = reason or "Output file not created"
                else:
                    result_info['last_error'] = (
                        reason or "Annealing finished without normal termination marker"
                    )

        except Exception as e:
            result_info['last_error'] = str(e)
        finally:
            try:
                os.unlink(stderr_capture_file)
            except OSError:
                pass

        return result_info

    def _process_replica_result(result_info: Dict[str, Any]):
        """Process a completed replica result (thread-safe with lock)."""
        nonlocal completed_replicas
        run_name = result_info['run_name']
        run_dir = result_info['run_dir']
        input_basename = os.path.basename(result_info['input_file'])

        if result_info['success']:
            with _replica_lock:
                completed_replicas += 1
                current = completed_replicas
            if verbose:
                print(f"\n  {run_name}... ✓")
            elif callable(progress_cb):
                progress_cb(f"{current}/{num_replicas} ...")
        else:
            if verbose:
                print(f"\n  {run_name}... ✗ (no output files)")
                output_file = os.path.join(run_dir, input_basename.replace('.in', '.out'))
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            content = f.read()
                            if 'Traceback' in content:
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if 'Traceback' in line:
                                        error_lines = lines[i:min(i+10, len(lines))]
                                        print(f"    Traceback found in {run_name}/{input_basename.replace('.in', '.out')}:")
                                        for eline in error_lines:
                                            if eline.strip():
                                                print(f"      {eline}")
                                        break
                            elif result_info['last_error']:
                                print(f"    {result_info['last_error']}")
                    except:
                        if result_info['last_error']:
                            print(f"    {result_info['last_error']}")
                elif result_info['last_error']:
                    print(f"    {result_info['last_error']}")
            failed_runs.append(run_name)
            if result_info.get('last_error'):
                failed_reasons[run_name] = result_info['last_error']

    import threading
    _replica_lock = threading.Lock()

    # Build list of pending replicas (skip already completed)
    pending_replicas = []
    for i, input_file in enumerate(replicated_files, 1):
        run_dir = os.path.dirname(input_file)
        run_name = os.path.basename(run_dir)
        input_basename = os.path.basename(input_file)

        if _replica_already_completed(run_dir, input_basename):
            completed_replicas += 1
            if verbose:
                print(f"\n  {run_name}... ✓ (already completed)")
            elif callable(progress_cb):
                progress_cb(f"{completed_replicas}/{num_replicas} ...")
            continue
        pending_replicas.append(input_file)

    effective_concurrent = min(concurrent_jobs, len(pending_replicas)) if pending_replicas else 1

    if effective_concurrent <= 1:
        # Serial execution
        for input_file in pending_replicas:
            run_name = os.path.basename(os.path.dirname(input_file))
            if verbose:
                print(f"\n  {run_name}...", end=" ", flush=True)
                result_info = _run_single_replica(input_file)
            elif callable(progress_cb):
                def _step_cb(step_str, _cr=completed_replicas, _nr=num_replicas, _pcb=progress_cb):
                    # step_str is "X/Y" (current step / total steps) for the active replica.
                    # Encode OVERALL progress: completed replicas contribute Y each.
                    try:
                        x_str, y_str = step_str.split('/', 1)
                        x = int(x_str.strip())
                        y = int(y_str.strip())
                    except (ValueError, AttributeError):
                        return
                    overall_done = _cr * y + x
                    overall_total = _nr * y
                    _pcb(f"{_cr}/{_nr} (step {overall_done}/{overall_total})")
                result_info = _run_single_replica(input_file, on_step=_step_cb)
            else:
                result_info = _run_single_replica(input_file)
            _process_replica_result(result_info)
    else:
        # Concurrent execution with staggered launches for seed uniqueness
        if verbose:
            print(f"\n  Running {len(pending_replicas)} replicas ({effective_concurrent} concurrent, staggered)...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _staggered_replica(input_file: str, delay: float) -> Dict[str, Any]:
            """Launch a replica after a short delay for seed uniqueness."""
            if delay > 0:
                time.sleep(delay)
            return _run_single_replica(input_file)

        # Background monitor: aggregate step progress across all running replicas
        # by reading their .ascec_step files, and report overall progress.
        _monitor_stop = threading.Event()
        _replica_dirs_to_watch = [os.path.dirname(f) for f in pending_replicas]

        def _aggregate_step_monitor():
            while not _monitor_stop.wait(2.0):
                if not callable(progress_cb):
                    continue
                # Per-replica step readout. Each replica writes "x/y" (current
                # step / total steps) to its .ascec_step; we show every replica
                # so concurrent progress is visible, and never double-count
                # finished replicas (their file already reads ~y/y).
                steps_total_each = 0
                per_replica = []  # (replica_index, current_step or None)
                for i, d in enumerate(_replica_dirs_to_watch, 1):
                    spf = os.path.join(d, '.ascec_step')
                    x = None
                    try:
                        with open(spf, 'r') as f:
                            line = f.readline().strip()
                        x_str, y_str = line.split('/', 1)
                        x = int(x_str.strip())
                        y = int(y_str.strip())
                        if y > 0 and steps_total_each == 0:
                            steps_total_each = y
                    except (OSError, ValueError):
                        x = None
                    per_replica.append((i, x))
                if steps_total_each <= 0:
                    continue
                with _replica_lock:
                    cr = completed_replicas
                parts = " ".join(
                    f"r{i}:{min(x, steps_total_each)}/{steps_total_each}" if x is not None
                    else f"r{i}:-/{steps_total_each}"
                    for i, x in per_replica
                )
                progress_cb(f"{cr}/{num_replicas} ({parts})")

        _monitor_thread = threading.Thread(target=_aggregate_step_monitor, daemon=True)
        if not verbose and callable(progress_cb):
            _monitor_thread.start()

        try:
            with ThreadPoolExecutor(max_workers=effective_concurrent) as executor:
                # Stagger launches by 2 seconds each to ensure unique time-based seeds
                futures = {}
                for idx, input_file in enumerate(pending_replicas):
                    delay = idx * 2.0  # 2-second stagger between launches
                    future = executor.submit(_staggered_replica, input_file, delay)
                    futures[future] = input_file

                for future in as_completed(futures):
                    try:
                        result_info = future.result()
                        _process_replica_result(result_info)
                    except Exception as e:
                        input_file = futures[future]
                        run_name = os.path.basename(os.path.dirname(input_file))
                        if verbose:
                            print(f"\n  {run_name}... ✗ (exception: {e})")
                        failed_runs.append(run_name)
                        failed_reasons[run_name] = f"exception: {e}"
        finally:
            _monitor_stop.set()
            if _monitor_thread.is_alive():
                _monitor_thread.join(timeout=3.0)
    
    if failed_runs:
        print(f"\n✗ {len(failed_runs)} simulation(s) failed")
        # Surface the reason for each failure (unique reasons if several
        # replicas share one) so the user sees *why*, not just the count.
        _seen_reasons = set()
        for _rn in failed_runs:
            _reason = failed_reasons.get(_rn)
            if _reason and _reason not in _seen_reasons:
                _seen_reasons.add(_reason)
                print(f"  Reason: {_reason}")
        return 1  # Fail the stage if any runs didn't complete
    
    # Generate annealing diagrams for each replica
    if MATPLOTLIB_AVAILABLE:
        # Only show messages if not in workflow mode
        is_workflow = getattr(context, 'is_workflow', False)
        
        if not is_workflow:
            print(f"\nGenerating annealing diagrams...")
        
        diagrams_generated = 0
        all_tvse_files = []
        
        for annealing_dir in context.annealing_dirs:
            # Find tvse_*.dat file in this directory
            tvse_files = glob.glob(os.path.join(annealing_dir, 'tvse_*.dat'))
            if tvse_files:
                for tvse_file in tvse_files:
                    # Always add to list for combined diagram, regardless of individual diagram success
                    all_tvse_files.append(tvse_file)
                    if plot_annealing_diagrams(tvse_file, annealing_dir):
                        diagrams_generated += 1
        
        if diagrams_generated > 0 and not is_workflow:
            print(f"  Generated {diagrams_generated} diagram(s)")
            
        # Generate combined replica diagram in parent annealing directory
        if len(all_tvse_files) > 1:
            # Get parent directory (annealing/)
            annealing_parent = os.path.dirname(context.annealing_dirs[0])
            if not annealing_parent:
                annealing_parent = "annealing"
            
            combined_diagram = os.path.join(annealing_parent, f"tvse_r{num_replicas}.png")
            if plot_combined_replicas_diagram(all_tvse_files, combined_diagram, num_replicas):
                if not is_workflow:
                    print(f"  Generated combined: {os.path.basename(combined_diagram)}")
    
    return 0

def _cached_cosmic_input_count(context: "WorkflowContext", stage_num: int) -> Optional[int]:
    """Return the cosmic input-structure count persisted in the protocol cache.

    Used to re-seed the redo lock on a resumed session so a redo-only resume
    cannot re-establish (and inflate) the input maximum recorded by the original
    full run. Returns None when no cache or no stored value is available.
    """
    cache_file = getattr(context, 'cache_file', '')
    if not cache_file:
        return None
    try:
        cache = load_protocol_cache(cache_file) or {}
    except Exception:
        return None
    stages = cache.get('stages', {}) if isinstance(cache, dict) else {}
    stage_data = stages.get(f"cosmic_{stage_num}", {}) if isinstance(stages, dict) else {}
    result = stage_data.get('result', {}) if isinstance(stage_data, dict) else {}
    value = result.get('input_count') if isinstance(result, dict) else None
    return int(value) if isinstance(value, int) and value > 0 else None


def find_out_file_in_subdirs(base_dir: str, basename: str):
    """Find .out file in subdirectories, checking shortened basename variants."""
    # Check root first
    root_file = os.path.join(base_dir, f"{basename}.out")
    if os.path.exists(root_file):
        return root_file
    
    # Build list of subdirectories to check, prioritizing exact and shortened names
    subdirs_to_check = [basename]
    
    # Add shortened basename variants
    if '_opt' in basename:
        subdirs_to_check.insert(0, basename.replace('_opt', ''))
    elif '_calc' in basename:
        subdirs_to_check.insert(0, basename.replace('_calc', ''))
    
    # Add common subdirectory names
    subdirs_to_check.extend(['orca_out', 'gaussian_out', 'completed', 'failed', 'skipped'])
    
    # Check subdirectories
    for subdir in subdirs_to_check:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path):
            out_file = os.path.join(subdir_path, f"{basename}.out")
            if os.path.exists(out_file):
                return out_file
    
    # Last resort: search all subdirectories recursively
    for root, dirs, files in os.walk(base_dir):
        # Skip deep nesting (max 2 levels)
        if root.count(os.sep) - base_dir.count(os.sep) > 2:
            continue
        if f"{basename}.out" in files:
            return os.path.join(root, f"{basename}.out")
    
    return None

def process_redo_structures(context: WorkflowContext, stage_dir: str, template_file: str, concurrent_jobs: int = 1) -> bool:
    """
    Process structures that need recalculation (from cosmic stage).
    Regenerates input files using new geometries and deletes old output files.

    Args:
        context: Workflow context
        stage_dir: Directory of the current stage (calculation or optimization)
        template_file: Path to the template input file
        concurrent_jobs: Number of concurrent jobs for rescue Hessian calculations

    Returns:
        bool: True if any structures were processed, False otherwise
    """
    # Determine cosmic directory (check context or default)
    cosmic_dir = getattr(context, 'cosmic_dir', None)
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1

    def _redo_log(*args, **kwargs):
        if not workflow_concise:
            print(*args, **kwargs)

    if not cosmic_dir:
        # For optimization stages, check for cosmic_2, cosmic_3, etc.
        # For optimization stages, use cosmic
        if 'optimization' in stage_dir.lower():
            # Check for cosmic_2 first (most common for first optimization).
            # Accept legacy uppercase COSMIC_* as fallback for resume safety.
            for candidate in ('cosmic_2', 'COSMIC_2', 'cosmic_3', 'COSMIC_3',
                              'cosmic_4', 'COSMIC_4'):
                if os.path.exists(candidate):
                    cosmic_dir = candidate
                    break
            else:
                # Fallback to cosmic if none found
                cosmic_dir = 'cosmic' if os.path.exists('cosmic') else 'COSMIC'
        else:
            # Calculation stage - use cosmic
            cosmic_dir = 'cosmic' if os.path.exists('cosmic') else 'COSMIC'
    
    if not os.path.exists(cosmic_dir):
        return False
        
    need_recalc_dir = os.path.join(cosmic_dir, "skipped_structures", "need_recalculation")
    critical_non_conv_dir = os.path.join(cosmic_dir, "skipped_structures", "critical_non_converged")
    
    # Check if either directory exists
    has_need_recalc = os.path.exists(need_recalc_dir)
    has_critical_nc = os.path.exists(critical_non_conv_dir)
    
    if not has_need_recalc and not has_critical_nc:
        return False
        
    # Collect XYZ files from both directories
    xyz_files = []
    if has_need_recalc:
        xyz_files.extend(glob.glob(os.path.join(need_recalc_dir, "*.xyz")))
    if has_critical_nc:
        xyz_files.extend(glob.glob(os.path.join(critical_non_conv_dir, "*.xyz")))
    
    if not xyz_files:
        return False
    
    # Get basenames and filter out combined files BEFORE printing count
    need_recalc_basenames = [os.path.splitext(os.path.basename(f))[0] for f in xyz_files]
    filtered_basenames = []
    for b in need_recalc_basenames:
        low = b.lower()
        if low.startswith('combined') or 'combined_' in low or low == 'combined':
            continue
        filtered_basenames.append(b)
    need_recalc_basenames = filtered_basenames
    
    if not need_recalc_basenames:
        return False
    
    # Determine display directory
    display_dir = os.path.join(cosmic_dir, "skipped_structures")
    _redo_log(f"\nProcessing redo structures from: {display_dir}")
    _redo_log(f"\nFound {len(need_recalc_basenames)} structure(s) to retry")

    processed_count = 0
    processed_basenames = []
    
    # Get template content
    if template_file and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            template_content = f.read()
    else:
        _redo_log(f"  Warning: Template file {template_file} not found. Cannot regenerate inputs.")
        return False

    # Find launcher path for ORCA calculations (load early for version detection)
    launcher_path = None
    launcher_content = None
    possible_launchers = [
        os.path.join(stage_dir, 'launcher_xtb.sh'),
        os.path.join(stage_dir, 'launcher_orca.sh'),
        os.path.join(stage_dir, 'launcher_ascec.sh'),
        'launcher_xtb.sh',
        'launcher_orca.sh',
        'launcher_ascec.sh',
    ]
    for lp in possible_launchers:
        if os.path.exists(lp):
            launcher_path = lp
            try:
                with open(lp, 'r') as f:
                    launcher_content = f.read()
            except Exception:
                pass
            break

    # Determine QM program from template extension
    template_lower = template_file.lower().strip() if template_file else ''
    if template_lower.endswith('.inp'):
        qm_program = 'orca'
    elif template_lower.endswith(('.com', '.gjf')):
        qm_program = 'gaussian'
    else:
        qm_program = 'xtb'

    # Parse rescue method from template (for structures with 2+ imaginary frequencies)
    rescue_method, rescue_use_numfreq = parse_rescue_method(
        template_content, launcher_content=launcher_content, qm_program=qm_program
    )

    # Extract charge and multiplicity from template
    charge_val = 0
    mult_val = 1
    xyz_match = re.search(r'\*\s*(?:xyz|xyzfile)\s+(-?\d+)\s+(\d+)', template_content, re.IGNORECASE)
    if xyz_match:
        charge_val = int(xyz_match.group(1))
        mult_val = int(xyz_match.group(2))

    # Extract nprocs from template (ORCA or Gaussian format)
    nprocs_val = 1  # default to 1 if not found
    # ORCA format: nprocs N
    nprocs_match = re.search(r'nprocs\s+(\d+)', template_content, re.IGNORECASE)
    if nprocs_match:
        nprocs_val = int(nprocs_match.group(1))
    else:
        # Gaussian format: %NProcShared=N
        nprocs_match = re.search(r'%NProcShared\s*=\s*(\d+)', template_content, re.IGNORECASE)
        if nprocs_match:
            nprocs_val = int(nprocs_match.group(1))

    # Track structures needing rescue Hessian (2+ imaginary freqs)
    rescue_hessian_tasks = []  # List of (basename, hess_file)
    # Pending rescue Hessian tasks to run concurrently after geometry processing
    pending_rescue = []  # List of dicts with rescue task parameters

    for basename in need_recalc_basenames:
        # Find xyz_file in either need_recalculation or critical_non_converged
        xyz_file: str = ""  # Will be set below
        if has_need_recalc and os.path.exists(os.path.join(need_recalc_dir, f"{basename}.xyz")):
            xyz_file = os.path.join(need_recalc_dir, f"{basename}.xyz")
        elif has_critical_nc and os.path.exists(os.path.join(critical_non_conv_dir, f"{basename}.xyz")):
            xyz_file = os.path.join(critical_non_conv_dir, f"{basename}.xyz")
        
        # Skip if no xyz_file found (shouldn't happen since basenames come from xyz files)
        if not xyz_file:
            _redo_log(f"    {basename}: XYZ file not found, skipping")
            continue
        
        # Check if this structure came from critical_non_converged (non-converged optimization)
        is_critical_non_converged = critical_non_conv_dir in xyz_file

        
        # Possible subdirectory names where .out files may be located
        # Include both standard directories AND the basename itself (where ORCA/Gaussian creates output)
        subdirs_to_check = [basename, 'orca_out', 'gaussian_out', 'completed', 'failed', 'skipped']
        
        # Also add shortened versions of basename (e.g., "motif_01" for "motif_01_opt")
        # ORCA often creates subdirectories without the suffix
        if '_opt' in basename:
            subdirs_to_check.insert(0, basename.replace('_opt', ''))
        elif '_calc' in basename:
            subdirs_to_check.insert(0, basename.replace('_calc', ''))
        
        # Find .out file for this structure
        out_file = None
        
        # Search in critical_non_converged first (if applicable)
        if has_critical_nc and os.path.exists(os.path.join(critical_non_conv_dir, f"{basename}.out")):
            out_file = os.path.join(critical_non_conv_dir, f"{basename}.out")
        # Search in need_recalc_dir
        elif has_need_recalc and os.path.exists(os.path.join(need_recalc_dir, f"{basename}.out")):
            out_file = os.path.join(need_recalc_dir, f"{basename}.out")
        # Search in root
        elif os.path.exists(os.path.join(stage_dir, f"{basename}.out")):
            out_file = os.path.join(stage_dir, f"{basename}.out")
        elif os.path.exists(os.path.join(stage_dir, f"{basename}.out.backup")):
            out_file = os.path.join(stage_dir, f"{basename}.out.backup")
        else:
            # Search in subdirectories
            for subdir in subdirs_to_check:
                subdir_path = os.path.join(stage_dir, subdir)
                if os.path.exists(os.path.join(subdir_path, f"{basename}.out")):
                    out_file = os.path.join(subdir_path, f"{basename}.out")
                    break
                elif os.path.exists(os.path.join(subdir_path, f"{basename}.out.backup")):
                    out_file = os.path.join(subdir_path, f"{basename}.out.backup")
                    break
            
            # If still not found, check cosmic directory (where files are moved after stage completion)
            if not out_file and cosmic_dir and os.path.exists(cosmic_dir):
                # Check for orca_out_*/gaussian_out_*/calc_out_* directories in cosmic folder
                for item in os.listdir(cosmic_dir):
                    if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                        out_dir = os.path.join(cosmic_dir, item)
                        if os.path.isdir(out_dir):
                            if os.path.exists(os.path.join(out_dir, f"{basename}.out")):
                                out_file = os.path.join(out_dir, f"{basename}.out")
                                break

        
        # Determine new geometry based on imaginary frequencies
        new_geometry = None
        if out_file and os.path.exists(out_file):
            imag_count = count_imaginary_frequencies(out_file)
            
            if imag_count == 1:
                # Single imaginary: displace along mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file))
                if xyz_lines:
                    try:
                        natoms = int(xyz_lines[0].strip())
                        if len(xyz_lines) >= natoms + 2:
                            new_geometry = xyz_lines[2:]  # Skip first 2 lines
                            _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✓")
                        else:
                            _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (malformed XYZ)")
                    except (ValueError, IndexError):
                        _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (invalid XYZ format)")
                else:
                    _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode ✗ (displacement failed)")
            
            elif imag_count >= 2:
                # Multiple imaginary: displace along the highest (most negative) imaginary mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file), use_highest_mode=True)
                if xyz_lines:
                    try:
                        natoms = int(xyz_lines[0].strip())
                        if len(xyz_lines) >= natoms + 2:
                            new_geometry = xyz_lines[2:]  # Skip first 2 lines
                            _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✓")
                        else:
                            _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (malformed XYZ)")
                    except (ValueError, IndexError):
                        _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (invalid XYZ format)")
                else:
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗ (displacement failed)")
            else:
                # No imaginary frequencies - check if non-converged (max iterations reached)
                conv_status = detect_convergence_status(out_file)
                
                if conv_status['status'] == 'not_converged' and rescue_method and launcher_path:
                    # Non-converged structure - use final geometry with rescue Hessian
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines:
                        new_geometry = xyz_lines[2:]  # Skip first 2 lines

                        # Save geometry to temporary XYZ for rescue calculation
                        rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                        try:
                            with open(rescue_xyz_path, 'w') as f:
                                natoms = len([line for line in new_geometry if line.strip()])
                                f.write(f"{natoms}\n")
                                f.write(f"{basename} geometry for rescue Hessian\n")
                                for line in new_geometry:
                                    if line.strip():
                                        f.write(line if line.endswith('\n') else line + '\n')

                            # Queue rescue Hessian for concurrent execution
                            pending_rescue.append({
                                'xyz_path': rescue_xyz_path,
                                'rescue_method': rescue_method,
                                'launcher_path': launcher_path,
                                'charge': charge_val,
                                'multiplicity': mult_val,
                                'nprocs': nprocs_val,
                                'basename': basename,
                                'use_numfreq': rescue_use_numfreq,
                                'qm_program': qm_program,
                            })
                            _redo_log(f"    {basename}: non-converged (max iter), rescue Hessian queued")
                        except Exception as e:
                            _redo_log(f"    {basename}: non-converged (max iter), using final geometry (error: {e})")
                    else:
                        # Fallback to cosmic XYZ
                        _redo_log(f"    {basename}: non-converged (max iter), using cosmic XYZ", end='')
                        if os.path.exists(xyz_file):
                            with open(xyz_file, 'r') as f:
                                new_geometry = f.readlines()[2:]
                            _redo_log(f" ✓")
                        else:
                            _redo_log(f" ✗ (extraction failed)")
                elif conv_status['status'] == 'not_converged':
                    # Non-converged but no rescue method - use final geometry
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines:
                        new_geometry = xyz_lines[2:]
                        _redo_log(f"    {basename}: non-converged (max iter), using final geometry ✓")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using cosmic XYZ", end='')
                        if os.path.exists(xyz_file):
                            with open(xyz_file, 'r') as f:
                                new_geometry = f.readlines()[2:]
                            _redo_log(f" ✓")
                        else:
                            _redo_log(f" ✗")
                else:
                    # Converged with no imaginary freqs - use cosmic XYZ
                    _redo_log(f"    {basename}: No imaginary freqs, using cosmic XYZ", end='')
                    if os.path.exists(xyz_file):
                        with open(xyz_file, 'r') as f:
                            new_geometry = f.readlines()[2:]
                        _redo_log(f" ✓")
                    else:
                        _redo_log(f" ✗ (cosmic XYZ not found)")
        else:
            # No .out file found - check if from critical_non_converged (needs rescue hessian)
            if is_critical_non_converged and xyz_file and os.path.exists(xyz_file):
                # Structure from critical_non_converged - use XYZ with rescue Hessian
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]

                if rescue_method and launcher_path:
                    # Queue rescue Hessian for concurrent execution
                    rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                    try:
                        with open(rescue_xyz_path, 'w') as f:
                            natoms = len([line for line in new_geometry if line.strip()])
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} geometry for rescue Hessian\n")
                            for line in new_geometry:
                                if line.strip():
                                    f.write(line if line.endswith('\n') else line + '\n')

                        pending_rescue.append({
                            'xyz_path': rescue_xyz_path,
                            'rescue_method': rescue_method,
                            'launcher_path': launcher_path,
                            'charge': charge_val,
                            'multiplicity': mult_val,
                            'nprocs': nprocs_val,
                            'basename': basename,
                            'use_numfreq': rescue_use_numfreq,
                            'qm_program': qm_program,
                        })
                        _redo_log(f"    {basename}: critical non-converged, rescue Hessian queued")
                    except Exception as e:
                        _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (error: {e})")
                else:
                    _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (no rescue method)")
            elif xyz_file and os.path.exists(xyz_file):
                # Regular case - just use XYZ from cosmic
                _redo_log(f"    {basename}: No .out file, using cosmic XYZ", end='')
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]
                _redo_log(f" ✓")
            else:
                _redo_log(f"    {basename}: No .out file, cosmic XYZ not found ✗")
        
        # Regenerate input file with new geometry
        if new_geometry:
            # Determine QM program and input extension
            template_lower = template_file.lower().strip()
            if template_lower.endswith('.inp'):
                qm_program = 'orca'
                input_ext = '.inp'
            else:
                qm_program = 'gaussian'
                input_ext = '.com' if template_lower.endswith('.com') else '.gjf'
            
            # Define input path
            input_path = os.path.join(stage_dir, f"{basename}{input_ext}")
            
            # Check if original input file exists - if so, update it; otherwise create new
            if os.path.exists(input_path):
                # Original file exists - just update coordinates
                try:
                    with open(input_path, 'r') as f:
                        original_lines = f.readlines()
                    
                    # Find and replace the coordinate block
                    updated_lines = []
                    in_coords = False
                    coords_written = False
                    
                    for line in original_lines:
                        # Detect start of coordinate block (after "* xyz" or similar)
                        if not in_coords and ('* xyz' in line.lower() or '* xyzfile' in line.lower()):
                            updated_lines.append(line)
                            in_coords = True
                            continue
                        
                        # Write new coordinates once we're in the block
                        if in_coords and not coords_written:
                            # Write all new geometry lines
                            for geom_line in new_geometry:
                                updated_lines.append(geom_line)
                            coords_written = True
                            # Skip old coordinates until we hit the end marker
                            if line.strip() == '*':
                                updated_lines.append(line)
                                in_coords = False
                            continue
                        
                        # Skip old coordinate lines
                        if in_coords:
                            if line.strip() == '*':
                                updated_lines.append(line)
                                in_coords = False
                            continue
                        
                        # Keep all other lines as-is
                        updated_lines.append(line)
                    
                    # Write updated file
                    with open(input_path, 'w') as f:
                        f.writelines(updated_lines)
                    
                    processed_count += 1
                    processed_basenames.append(basename)
                except Exception as e:
                    _redo_log(f"\n    Warning: Could not update {basename}: {e}")
            else:
                # Original file doesn't exist - create new from template
                # Parse XYZ lines into atom fields (symbol, x, y, z)
                atoms_list = []
                for line in new_geometry:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Convert coordinates to float
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms_list.append([symbol, x, y, z])
                
                config_data = {
                    'atoms': atoms_list,
                    'comment': f"{basename}"
                }
                
                if create_qm_input_file(config_data, template_content, input_path, qm_program):
                    processed_count += 1
                    processed_basenames.append(basename)

    # Run pending rescue Hessian calculations (concurrently if possible)
    if pending_rescue:
        effective_rescue = min(concurrent_jobs, len(pending_rescue))
        if effective_rescue > 1:
            _redo_log(f"\n  Running {len(pending_rescue)} rescue Hessian calculations ({effective_rescue} concurrent)...")
            with ThreadPoolExecutor(max_workers=effective_rescue) as executor:
                futures = {}
                for task in pending_rescue:
                    future = executor.submit(
                        run_rescue_hessian_calculation,
                        task['xyz_path'], task['rescue_method'], task['launcher_path'],
                        charge=task['charge'], multiplicity=task['multiplicity'],
                        nprocs=task['nprocs'], output_basename=task['basename'],
                        use_numfreq=task['use_numfreq'],
                        qm_program=task.get('qm_program')
                    )
                    futures[future] = task
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        hess_file = future.result()
                        if hess_file and os.path.exists(hess_file):
                            rescue_hessian_tasks.append((task['basename'], hess_file))
                            _redo_log(f"    {task['basename']}: rescue Hessian computed ✓")
                        else:
                            _redo_log(f"    {task['basename']}: rescue Hessian failed, using final geometry")
                    except Exception as e:
                        _redo_log(f"    {task['basename']}: rescue Hessian error: {e}")
                    # Cleanup temporary XYZ
                    if os.path.exists(task['xyz_path']):
                        try:
                            os.remove(task['xyz_path'])
                        except Exception:
                            pass
        else:
            # Serial execution
            for task in pending_rescue:
                try:
                    hess_file = run_rescue_hessian_calculation(
                        task['xyz_path'], task['rescue_method'], task['launcher_path'],
                        charge=task['charge'], multiplicity=task['multiplicity'],
                        nprocs=task['nprocs'], output_basename=task['basename'],
                        use_numfreq=task['use_numfreq'],
                        qm_program=task.get('qm_program')
                    )
                    if hess_file and os.path.exists(hess_file):
                        rescue_hessian_tasks.append((task['basename'], hess_file))
                        _redo_log(f"    {task['basename']}: rescue Hessian computed ✓")
                    else:
                        _redo_log(f"    {task['basename']}: rescue Hessian failed, using final geometry")
                except Exception as e:
                    _redo_log(f"    {task['basename']}: rescue Hessian error: {e}")
                # Cleanup temporary XYZ
                if os.path.exists(task['xyz_path']):
                    try:
                        os.remove(task['xyz_path'])
                    except Exception:
                        pass

    if processed_count > 0:
        _redo_log(f"\n  Regenerated {processed_count} input file(s) with new geometries")

        # Enable Hessian restart for structures with rescue Hessians computed
        if rescue_hessian_tasks:
            _redo_log(f"  Enabling Hessian restart for {len(rescue_hessian_tasks)} structure(s)")
            for task_basename, hess_file in rescue_hessian_tasks:
                # Determine where the calculation runs from
                subdir_path = os.path.join(stage_dir, task_basename)
                if os.path.isdir(subdir_path):
                    calc_dir = subdir_path
                else:
                    calc_dir = stage_dir

                if qm_program == 'xtb':
                    # For standalone xTB: copy rescue hessian with the optimization
                    # namespace name so xTB finds it automatically.
                    # xTB with --namespace <basename> reads <basename>.hessian.
                    hess_dest = os.path.join(calc_dir, f"{task_basename}.hessian")
                    if os.path.exists(hess_file):
                        shutil.copy2(hess_file, hess_dest)
                    _redo_log(f"    {task_basename}: xTB Hessian copied to calc directory ✓")
                else:
                    # For ORCA/Gaussian: modify input file to read Hessian
                    template_lower_ext = template_file.lower().strip()
                    if template_lower_ext.endswith('.inp'):
                        input_ext = '.inp'
                    else:
                        input_ext = '.com' if template_lower_ext.endswith('.com') else '.gjf'
                    input_path = os.path.join(calc_dir, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(calc_dir, os.path.basename(hess_file))

                    if os.path.exists(input_path):
                        same_file = os.path.exists(hess_dest) and os.path.samefile(hess_file, hess_dest)
                        if not same_file and os.path.exists(hess_file):
                            shutil.copy2(hess_file, hess_dest)

                        if enable_hessian_restart(input_path, hess_dest):
                            _redo_log(f"    {task_basename}: Hessian restart enabled ✓")
                        else:
                            _redo_log(f"    {task_basename}: Hessian restart failed ✗")

        # Store recalculated basenames in context for cosmic cache update
        context.recalculated_files = processed_basenames

        # RENAME (not delete) output files for structures needing recalculation
        # This preserves the old .out file in case the new calculation fails
        # We can then extract geometry from the backup on next redo attempt
        _redo_log(f"  Backing up old output files for structures needing recalculation")
        for basename in processed_basenames:
            # Build list of subdirectories to search, including shortened basename variants
            subdirs_to_search = ['orca_out', 'gaussian_out', 'completed', basename]
            
            # Add shortened versions for common patterns (e.g., "motif_01" for "motif_01_opt")
            if '_opt' in basename:
                subdirs_to_search.insert(0, basename.replace('_opt', ''))
            elif '_calc' in basename:
                subdirs_to_search.insert(0, basename.replace('_calc', ''))
            
            # Check in subdirectories
            search_dirs_for_out = [stage_dir] + [os.path.join(stage_dir, sd) for sd in subdirs_to_search]
            
            out_file_found = False
            for search_dir in search_dirs_for_out:
                if not os.path.exists(search_dir):
                    continue
                
                out_file = os.path.join(search_dir, f"{basename}.out")
                if os.path.exists(out_file):
                    # Only create backup if one doesn't already exist (keep first backup only)
                    backup_file = out_file + ".backup"
                    if not os.path.exists(backup_file):
                        shutil.move(out_file, backup_file)
                    else:
                        # Backup exists, just remove the current .out file
                        os.remove(out_file)
                    out_file_found = True
                    break # Only rename the first .out file found
            
            # Also remove auxiliary files from all possible locations
            # Use the broader search_dirs list for auxiliary files
            all_possible_search_dirs = [stage_dir] + [os.path.join(stage_dir, sd) for sd in subdirs_to_search + ['failed', 'skipped']]
            for search_dir in all_possible_search_dirs:
                if not os.path.exists(search_dir):
                    continue
                for ext in ['.gbw', '.prop', '.densities', '.tmp', '_property.txt', '.engrad']:
                    aux_file = os.path.join(search_dir, f"{basename}{ext}")
                    if os.path.exists(aux_file):
                        try:
                            os.remove(aux_file)
                        except Exception:
                            pass
    
    return processed_count > 0
def process_optimization_redo(context: WorkflowContext, stage_dir: str, template_file: str, concurrent_jobs: int = 1) -> bool:
    """
    Specialized redo processing for optimization stage.
    Handles the specific directory structure of optimization/cosmic interactions.
    """
    # 1. Determine cosmic Directory
    # For optimization, we look for the cosmic folder that THIS optimization feeds into.
    # Usually cosmic_2, cosmic_3, etc.
    # Use refinement_cosmic_folder (the dedicated variable for optimization outputs)
    cosmic_dir = getattr(context, 'refinement_cosmic_folder', None)
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1

    def _redo_log(*args, **kwargs):
        if not workflow_concise:
            print(*args, **kwargs)
    
    if not cosmic_dir:
        # Try to guess based on existence (lowercase first, then uppercase for legacy)
        for candidate in ('cosmic_2', 'COSMIC_2', 'cosmic_3', 'COSMIC_3', 'cosmic_4', 'COSMIC_4'):
            if os.path.exists(candidate):
                cosmic_dir = candidate
                break
        else:
            return False
    
    # CRITICAL: If cosmic_dir includes orca_out_X subdirectory, strip it
    # organize step sets context.refinement_cosmic_folder to "cosmic_2/orca_out_5_1"
    # but skipped_structures is at "cosmic_2/skipped_structures/"
    if '/' in cosmic_dir and ('orca_out_' in cosmic_dir or 'gaussian_out_' in cosmic_dir):
        cosmic_dir = _cosmic_base_name(cosmic_dir)
    
    # 2. Locate need_recalculation directory and optionally clustered_with_minima
    need_recalc_dir = os.path.join(cosmic_dir, "skipped_structures", "need_recalculation")
    clustered_dir = os.path.join(cosmic_dir, "skipped_structures", "clustered_with_minima")
    critical_non_conv_dir = os.path.join(cosmic_dir, "skipped_structures", "critical_non_converged")
    
    # Check threshold mode from context
    # --critical: only use need_recalculation (structures with imaginary freqs)
    # --skipped: use both need_recalculation AND clustered_with_minima
    use_skipped_threshold = getattr(context, 'use_skipped_threshold', False)
    
    # 3. Find XYZ files to process
    xyz_files = []
    
    # Always include critical structures (need_recalculation) - these have imaginary frequencies
    if os.path.exists(need_recalc_dir):
        need_recalc_files = glob.glob(os.path.join(need_recalc_dir, "*.xyz"))
        # Filter out combined files
        need_recalc_files = [f for f in need_recalc_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(need_recalc_files)
    
    # Always include critical non-converged structures (these need rescue hessian)
    if os.path.exists(critical_non_conv_dir):
        critical_nc_files = glob.glob(os.path.join(critical_non_conv_dir, "*.xyz"))
        critical_nc_files = [f for f in critical_nc_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(critical_nc_files)
    
    # Only include clustered_with_minima if --skipped threshold is used
    # These are structures that clustered with existing minima (skipped but not critical)
    if use_skipped_threshold and os.path.exists(clustered_dir):
        clustered_files = glob.glob(os.path.join(clustered_dir, "*.xyz"))
        clustered_files = [f for f in clustered_files if "combined" not in os.path.basename(f).lower()]
        xyz_files.extend(clustered_files)
    
    if not xyz_files:
        return False
    
    # Sort files naturally (motif_01, motif_02, ...)
    xyz_files = sorted(xyz_files, key=lambda x: natural_sort_key(os.path.basename(x)))
        
    # Determine which directory to display based on what we found
    display_dir = need_recalc_dir if os.path.exists(need_recalc_dir) else clustered_dir
    if os.path.exists(need_recalc_dir) and os.path.exists(clustered_dir):
        display_dir = os.path.dirname(need_recalc_dir)  # Show parent "skipped_structures"
    if os.path.exists(critical_non_conv_dir):
        display_dir = os.path.dirname(need_recalc_dir)  # Show parent "skipped_structures"
    
    _redo_log(f"\nProcessing redo structures from: {display_dir}")
    _redo_log(f"Found {len(xyz_files)} structure(s) to retry")
    
    # 4. Process each file
    processed_count = 0
    processed_basenames = []
    
    # Read template
    if not os.path.exists(template_file):
        print(f"Error: Template file {template_file} not found")
        return False
        
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Find launcher path for ORCA calculations (load early for version detection)
    launcher_path = None
    launcher_content = None
    possible_launchers = [
        os.path.join(stage_dir, 'launcher_xtb.sh'),
        os.path.join(stage_dir, 'launcher_orca.sh'),
        os.path.join(stage_dir, 'launcher_ascec.sh'),
        'launcher_xtb.sh',
        'launcher_orca.sh',
        'launcher_ascec.sh',
    ]
    for lp in possible_launchers:
        if os.path.exists(lp):
            launcher_path = lp
            try:
                with open(lp, 'r') as f:
                    launcher_content = f.read()
            except Exception:
                pass
            break

    # Determine QM program from template extension
    template_lower = template_file.lower().strip() if template_file else ''
    if template_lower.endswith('.inp'):
        qm_program = 'orca'
    elif template_lower.endswith(('.com', '.gjf')):
        qm_program = 'gaussian'
    else:
        qm_program = 'xtb'

    # Parse rescue method from template (for structures with 2+ imaginary frequencies)
    rescue_method, rescue_use_numfreq = parse_rescue_method(
        template_content, launcher_content=launcher_content, qm_program=qm_program
    )

    # Extract charge and multiplicity from template
    charge_val = 0
    mult_val = 1
    xyz_match = re.search(r'\*\s*(?:xyz|xyzfile)\s+(-?\d+)\s+(\d+)', template_content, re.IGNORECASE)
    if xyz_match:
        charge_val = int(xyz_match.group(1))
        mult_val = int(xyz_match.group(2))

    # Extract nprocs from template (ORCA or Gaussian format)
    nprocs_val = 1  # default to 1 if not found
    # ORCA format: nprocs N
    nprocs_match = re.search(r'nprocs\s+(\d+)', template_content, re.IGNORECASE)
    if nprocs_match:
        nprocs_val = int(nprocs_match.group(1))
    else:
        # Gaussian format: %NProcShared=N
        nprocs_match = re.search(r'%NProcShared\s*=\s*(\d+)', template_content, re.IGNORECASE)
        if nprocs_match:
            nprocs_val = int(nprocs_match.group(1))

    # Track structures needing rescue Hessian (2+ imaginary freqs)
    rescue_hessian_tasks = []  # List of (basename, hess_file)
    # Pending rescue Hessian tasks to run concurrently after geometry processing
    pending_rescue = []  # List of dicts with rescue task parameters

    for xyz_file in xyz_files:
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        
        # Check if this structure came from critical_non_converged (non-converged optimization)
        is_critical_non_converged = os.path.exists(critical_non_conv_dir) and critical_non_conv_dir in xyz_file
        
        # 4a. Find the previous output file (to check imaginary freqs and extract geometry)
        # Optimization outputs are tricky. They could be in:
        # - optimization/umotif_XX_opt.out (root)
        # - optimization/umotif_XX_opt/umotif_XX_opt.out (subdir)
        # - cosmic_2/orca_out_X/umotif_XX_opt.out (moved there)
        # - need_recalc_dir/umotif_XX_opt.out (copied there by cosmic script)
        
        out_file = None
        
        # Priority 1: Check optimization directory (subdirs) - THIS IS THE REAL FILE WE NEED TO BACKUP
        # Check exact basename subdir
        check_path = os.path.join(stage_dir, basename, f"{basename}.out")
        if os.path.exists(check_path):
            out_file = check_path
        else:
            # Check subdir (motif_01 for motif_01_opt)
            short_name = basename.replace('_opt', '').replace('_calc', '')
            check_path = os.path.join(stage_dir, short_name, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
                    
        # Priority 2: Check optimization root
        if not out_file:
            check_path = os.path.join(stage_dir, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
        
        # Priority 3: Check cosmic output folders (if not found in optimization yet)
        if not out_file and os.path.exists(cosmic_dir):
            for item in os.listdir(cosmic_dir):
                if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                    check_path = os.path.join(cosmic_dir, item, f"{basename}.out")
                    if os.path.exists(check_path):
                        out_file = check_path
                        break
        
        # Priority 4: Check need_recalc_dir or critical_non_converged (cosmic script copy - ONLY for reading geometry)
        if not out_file:
            check_path = os.path.join(need_recalc_dir, f"{basename}.out")
            if os.path.exists(check_path):
                out_file = check_path
            elif os.path.exists(critical_non_conv_dir):
                check_path = os.path.join(critical_non_conv_dir, f"{basename}.out")
                if os.path.exists(check_path):
                    out_file = check_path
            
        # 4b. Determine new geometry
        new_geometry = None
        
        if out_file:
            imag_count = count_imaginary_frequencies(out_file)
            
            if imag_count == 1:
                # Displace
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file))
                if xyz_lines and len(xyz_lines) > 2:
                    new_geometry = xyz_lines[2:]
                    _redo_log(f"    {basename}: 1 imaginary freq, displacing along mode \u2713")
            elif imag_count >= 2:
                # Multiple imaginary: displace along the highest (most negative) imaginary mode
                xyz_lines = displace_along_imaginary_mode(out_file, os.path.dirname(out_file), use_highest_mode=True)
                if xyz_lines and len(xyz_lines) > 2:
                    new_geometry = xyz_lines[2:]
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✓")
                else:
                    _redo_log(f"    {basename}: {imag_count} imaginary freqs, displacing along highest mode ✗")
            else:
                # No imaginary frequencies - check if non-converged (max iterations reached)
                conv_status = detect_convergence_status(out_file)
                
                if conv_status['status'] == 'not_converged' and rescue_method and launcher_path:
                    # Non-converged structure - use final geometry with rescue Hessian
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines and len(xyz_lines) > 2:
                        new_geometry = xyz_lines[2:]

                        # Save geometry to temporary XYZ for rescue calculation
                        rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                        try:
                            with open(rescue_xyz_path, 'w') as f:
                                natoms = len([line for line in new_geometry if line.strip()])
                                f.write(f"{natoms}\n")
                                f.write(f"{basename} geometry for rescue Hessian\n")
                                for line in new_geometry:
                                    if line.strip():
                                        f.write(line if line.endswith('\n') else line + '\n')

                            # Queue rescue Hessian for concurrent execution
                            pending_rescue.append({
                                'xyz_path': rescue_xyz_path,
                                'rescue_method': rescue_method,
                                'launcher_path': launcher_path,
                                'charge': charge_val,
                                'multiplicity': mult_val,
                                'nprocs': nprocs_val,
                                'basename': basename,
                                'use_numfreq': rescue_use_numfreq,
                                'qm_program': qm_program,
                            })
                            _redo_log(f"    {basename}: non-converged (max iter), rescue Hessian queued")
                        except Exception as e:
                            _redo_log(f"    {basename}: non-converged (max iter), using final geometry (error: {e})")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using cosmic XYZ", end='')
                elif conv_status['status'] == 'not_converged':
                    # Non-converged but no rescue method - use final geometry
                    xyz_lines = extract_final_geometry(out_file, os.path.dirname(out_file))
                    if xyz_lines and len(xyz_lines) > 2:
                        new_geometry = xyz_lines[2:]
                        _redo_log(f"    {basename}: non-converged (max iter), using final geometry ✓")
                    else:
                        _redo_log(f"    {basename}: non-converged (max iter), using cosmic XYZ", end='')
                else:
                    # Converged with no imaginary freqs - use cosmic XYZ
                    _redo_log(f"    {basename}: No imaginary freqs, using cosmic XYZ", end='')
        else:
            # No .out file found - check if from critical_non_converged (needs rescue hessian)
            if is_critical_non_converged and os.path.exists(xyz_file):
                # Structure from critical_non_converged - use XYZ with rescue Hessian
                with open(xyz_file, 'r') as f:
                    new_geometry = f.readlines()[2:]
                
                if rescue_method and launcher_path:
                    # Queue rescue Hessian for concurrent execution
                    rescue_xyz_path = os.path.join(stage_dir, f"{basename}_rescue_geom.xyz")
                    try:
                        with open(rescue_xyz_path, 'w') as f:
                            natoms = len([line for line in new_geometry if line.strip()])
                            f.write(f"{natoms}\n")
                            f.write(f"{basename} geometry for rescue Hessian\n")
                            for line in new_geometry:
                                if line.strip():
                                    f.write(line if line.endswith('\n') else line + '\n')

                        pending_rescue.append({
                            'xyz_path': rescue_xyz_path,
                            'rescue_method': rescue_method,
                            'launcher_path': launcher_path,
                            'charge': charge_val,
                            'multiplicity': mult_val,
                            'nprocs': nprocs_val,
                            'basename': basename,
                            'use_numfreq': rescue_use_numfreq,
                            'qm_program': qm_program,
                        })
                        _redo_log(f"    {basename}: critical non-converged, rescue Hessian queued")
                    except Exception as e:
                        _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (error: {e})")
                else:
                    _redo_log(f"    {basename}: critical non-converged, using XYZ geometry (no rescue method)")
            else:
                # Regular case - just use XYZ from cosmic
                _redo_log(f"    {basename}: No .out file, using cosmic XYZ", end='')
        
        # Fallback to XYZ file
        if not new_geometry:
            with open(xyz_file, 'r') as f:
                new_geometry = f.readlines()[2:]
            imag_count = 0  # Initialize
            if out_file is None or imag_count == 0:
                _redo_log(f" \u2713")
                
        # 4c. Generate Input File
        if new_geometry:
            # Determine extension
            template_lower = template_file.lower().strip()
            if template_lower.endswith('.inp'):
                qm_program = 'orca'
                input_ext = '.inp'
            elif template_lower.endswith(('.com', '.gjf')):
                qm_program = 'gaussian'
                input_ext = '.com' if template_lower.endswith('.com') else '.gjf'
            else:
                qm_program = 'xtb'
                input_ext = '.xyz'
            
            # Always create redo input files at root level
            # The running loop expects flat files and will move them after sort
            input_path = os.path.join(stage_dir, f"{basename}{input_ext}")
            
            # Parse atoms
            atoms_list = []
            for line in new_geometry:
                parts = line.strip().split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms_list.append([symbol, x, y, z])
                    except ValueError:
                        pass
            
            config_data = {
                'atoms': atoms_list,
                'comment': f"{basename} generated by ASCEC redo"
            }
            
            if qm_program == 'xtb':
                created_ok = create_xyz_input_file(config_data, input_path)
            else:
                created_ok = create_qm_input_file(config_data, template_content, input_path, qm_program)

            if created_ok:
                processed_count += 1
                processed_basenames.append(basename)
                
                # 4d. Backup old output file (CRITICAL for optimization to re-run)
                # We must move the old .out file so the workflow sees it as "not done"
                # Only keep first backup - don't create incrementing backups
                if out_file and os.path.exists(out_file):
                    backup_file = out_file + ".backup"
                    if not os.path.exists(backup_file):
                        shutil.move(out_file, backup_file)
                    else:
                        # Backup exists, just remove the current .out file
                        os.remove(out_file)

    # Run pending rescue Hessian calculations (concurrently if possible)
    if pending_rescue:
        effective_rescue = min(concurrent_jobs, len(pending_rescue))
        if effective_rescue > 1:
            _redo_log(f"\n  Running {len(pending_rescue)} rescue Hessian calculations ({effective_rescue} concurrent)...")
            with ThreadPoolExecutor(max_workers=effective_rescue) as executor:
                futures = {}
                for task in pending_rescue:
                    future = executor.submit(
                        run_rescue_hessian_calculation,
                        task['xyz_path'], task['rescue_method'], task['launcher_path'],
                        charge=task['charge'], multiplicity=task['multiplicity'],
                        nprocs=task['nprocs'], output_basename=task['basename'],
                        use_numfreq=task['use_numfreq'],
                        qm_program=task.get('qm_program')
                    )
                    futures[future] = task
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        hess_file = future.result()
                        if hess_file and os.path.exists(hess_file):
                            rescue_hessian_tasks.append((task['basename'], hess_file))
                            _redo_log(f"    {task['basename']}: rescue Hessian computed ✓")
                        else:
                            _redo_log(f"    {task['basename']}: rescue Hessian failed, using final geometry")
                    except Exception as e:
                        _redo_log(f"    {task['basename']}: rescue Hessian error: {e}")
                    # Cleanup temporary XYZ
                    if os.path.exists(task['xyz_path']):
                        try:
                            os.remove(task['xyz_path'])
                        except Exception:
                            pass
        else:
            # Serial execution
            for task in pending_rescue:
                try:
                    hess_file = run_rescue_hessian_calculation(
                        task['xyz_path'], task['rescue_method'], task['launcher_path'],
                        charge=task['charge'], multiplicity=task['multiplicity'],
                        nprocs=task['nprocs'], output_basename=task['basename'],
                        use_numfreq=task['use_numfreq'],
                        qm_program=task.get('qm_program')
                    )
                    if hess_file and os.path.exists(hess_file):
                        rescue_hessian_tasks.append((task['basename'], hess_file))
                        _redo_log(f"    {task['basename']}: rescue Hessian computed ✓")
                    else:
                        _redo_log(f"    {task['basename']}: rescue Hessian failed, using final geometry")
                except Exception as e:
                    _redo_log(f"    {task['basename']}: rescue Hessian error: {e}")
                # Cleanup temporary XYZ
                if os.path.exists(task['xyz_path']):
                    try:
                        os.remove(task['xyz_path'])
                    except Exception:
                        pass

    if processed_count > 0:
        _redo_log(f"\n  Regenerated {processed_count} input file(s) with new geometries")

        # Enable Hessian restart for structures with rescue Hessians computed
        if rescue_hessian_tasks:
            _redo_log(f"  Enabling Hessian restart for {len(rescue_hessian_tasks)} structure(s)")
            for task_basename, hess_file in rescue_hessian_tasks:
                # Determine where the calculation runs from
                subdir_path = os.path.join(stage_dir, task_basename)
                if os.path.isdir(subdir_path):
                    calc_dir = subdir_path
                else:
                    calc_dir = stage_dir

                if qm_program == 'xtb':
                    # For standalone xTB: copy rescue hessian with the optimization
                    # namespace name so xTB finds it automatically.
                    # xTB with --namespace <basename> reads <basename>.hessian.
                    hess_dest = os.path.join(calc_dir, f"{task_basename}.hessian")
                    if os.path.exists(hess_file):
                        shutil.copy2(hess_file, hess_dest)
                    _redo_log(f"    {task_basename}: xTB Hessian copied to calc directory ✓")
                else:
                    # For ORCA/Gaussian: modify input file to read Hessian
                    template_lower_ext = template_file.lower().strip()
                    if template_lower_ext.endswith('.inp'):
                        input_ext = '.inp'
                    else:
                        input_ext = '.com' if template_lower_ext.endswith('.com') else '.gjf'
                    input_path = os.path.join(calc_dir, f"{task_basename}{input_ext}")
                    hess_dest = os.path.join(calc_dir, os.path.basename(hess_file))

                    if os.path.exists(input_path):
                        same_file = os.path.exists(hess_dest) and os.path.samefile(hess_file, hess_dest)
                        if not same_file and os.path.exists(hess_file):
                            shutil.copy2(hess_file, hess_dest)

                        if enable_hessian_restart(input_path, hess_dest):
                            _redo_log(f"    {task_basename}: Hessian restart enabled ✓")
                        else:
                            _redo_log(f"    {task_basename}: Hessian restart failed ✗")

        # Sort for natural ordering (motif_01, motif_02, etc.)
        context.recalculated_files = sorted(processed_basenames, key=natural_sort_key)
    
    return processed_count > 0


def check_qm_output_completed(qm_program: str, output_path: str) -> bool:
    """Check whether a QM output file finished successfully for a given program."""
    try:
        if qm_program == 'auto':
            if output_path.lower().endswith('.log'):
                return check_qm_output_completed('gaussian', output_path)
            # Disambiguate ORCA vs xTB by content before dispatching. A crashed
            # ORCA job (e.g. MDCI out-of-memory) still prints "Total Energy", and
            # the xTB check's loose fallback only greps for "total energy" — so an
            # ORCA file must NOT fall through to the xTB check, or error-terminated
            # ORCA jobs get misreported as completed. Detect the ORCA banner and
            # use the ORCA-specific normal-termination check for those files.
            try:
                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                    head = f.read(4000)
            except Exception:
                head = ''
            if 'O   R   C   A' in head:
                return check_qm_output_completed('orca', output_path)
            return (
                check_qm_output_completed('orca', output_path) or
                check_qm_output_completed('xtb', output_path)
            )

        if qm_program == 'orca':
            return check_orca_terminated_normally_opi(output_path)
        if qm_program == 'xtb':
            with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                output_content = f.read()
            output_lower = output_content.lower()

            if (
                'normal termination of xtb' in output_lower
                or 'geometry optimization converged' in output_lower
                or 'optimized geometry written to' in output_lower
            ):
                return True

            output_path_obj = Path(output_path)
            workdir = output_path_obj.parent
            basename = output_path_obj.stem
            xtb_artifacts = [
                workdir / f'{basename}.xtbopt.xyz',
                workdir / 'xtbopt.xyz',
                workdir / f'{basename}.xtbopt.log',
                workdir / 'xtbopt.log',
            ]
            if any(path.exists() for path in xtb_artifacts):
                return True

            return (
                'total energy' in output_lower
            )

        with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
            output_content = f.read()
        return 'Normal termination of Gaussian' in output_content
    except Exception:
        return False


def _parse_launcher_env(launcher_content: str) -> Dict[str, str]:
    """Extract literal ``export KEY=VALUE`` assignments from a launcher header.

    Windows-only helper: the POSIX launcher script cannot be executed there
    (no usable ``bash``), so the QM job is run directly and the launcher's
    plain environment exports (``OMP_NUM_THREADS``, ``ASCEC_XTB_RUNTIME``, …)
    must be applied to the subprocess env in-process instead. Lines containing
    a shell substitution (``$PATH`` etc.) are skipped — they are POSIX path
    manipulations that have no meaning on Windows. Never called on POSIX, so
    the bash launcher path stays byte-for-byte unchanged.
    """
    env_overrides: Dict[str, str] = {}
    header = launcher_content.split('###')[0] if launcher_content else ''
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped.startswith('export '):
            continue
        assignment = stripped[len('export '):].strip()
        key, sep, value = assignment.partition('=')
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or '$' in value:
            continue
        env_overrides[key] = value
    return env_overrides


def _run_qm_job_direct(qm_program: str, calc_working_dir: str,
                       input_file_relative: str, output_file_relative: str,
                       script_basename: str, launcher_content: str,
                       xtb_options: str, orca_exe: str) -> None:
    """Run a single QM optimization job directly, without a bash launcher.

    Windows path for :func:`_run_single_qm_job`. On Windows ``bash`` is either
    absent or maps to WSL (which cannot exec the native Windows xtb/ORCA
    binaries), so ``subprocess.run(['bash', script.sh])`` raises
    ``FileNotFoundError`` and every optimization job fails. This rebuilds the
    exact QM command the launcher script would have run and invokes it
    directly, with the launcher's environment exports applied. POSIX never
    takes this branch, so Linux behaviour is unchanged.
    """
    env = {**os.environ, **_parse_launcher_env(launcher_content)}
    output_path = os.path.join(calc_working_dir, output_file_relative)

    if qm_program == 'xtb':
        env.setdefault('ASCEC_XTB_RUNTIME', '1')
        cmd = ['xtb', input_file_relative, *shlex.split(xtb_options or ''),
               '--namespace', script_basename]
    elif qm_program == 'orca':
        cmd = [orca_exe or 'orca', input_file_relative]
    elif qm_program == 'gaussian':
        gauss_root = os.environ.get('GAUSS_ROOT', '')
        g16 = os.path.join(gauss_root, 'g16') if gauss_root else 'g16'
        # Gaussian writes its own .log; no stdout redirect (mirrors the launcher).
        subprocess.run([g16, input_file_relative], cwd=calc_working_dir,
                       env=env, preexec_fn=_PDEATHSIG_PREEXEC)
        return
    else:
        return

    with open(output_path, 'wb') as out_fh:
        subprocess.run(cmd, cwd=calc_working_dir, stdout=out_fh,
                       stderr=subprocess.STDOUT, env=env,
                       preexec_fn=_PDEATHSIG_PREEXEC)


def _run_single_qm_job(job_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single QM calculation job with launch failure retry logic.

    This is the worker function used by both serial and concurrent execution modes.
    It is self-contained and does not modify shared state.

    Args:
        job_info: Dictionary with keys:
            - input_file: relative path to input file
            - base_dir: base directory for calculations
            - launcher_content: launcher script content
            - qm_program: 'orca', 'gaussian', or 'xtb'
            - max_launch_retries: max retry attempts
            - launch_failure_threshold: seconds threshold
            - orca_exe: resolved ORCA executable path

    Returns:
        Dictionary with: input_file, success, launch_attempts, output_path
    """
    input_file = job_info['input_file']
    base_dir = job_info['base_dir']
    launcher_content = job_info['launcher_content']
    qm_program = job_info['qm_program']
    max_launch_retries = job_info.get('max_launch_retries', 10)
    launch_failure_threshold = job_info.get('launch_failure_threshold', 5.0)
    orca_exe = job_info.get('orca_exe', 'orca')
    xtb_options = job_info.get('xtb_options', '--gfn 2 --opt')

    basename = os.path.splitext(input_file)[0]
    output_file = basename + ('.out' if qm_program in ('orca', 'xtb') else '.log')
    output_path = os.path.join(base_dir, output_file)

    # Determine working directory
    if '/' in input_file or '\\' in input_file:
        calc_subdir = os.path.dirname(input_file)
        calc_working_dir = os.path.join(base_dir, calc_subdir)
        input_file_relative = os.path.basename(input_file)
        output_file_relative = os.path.basename(output_file)
        script_basename = os.path.splitext(os.path.basename(input_file))[0]
    else:
        calc_working_dir = base_dir
        input_file_relative = input_file
        output_file_relative = output_file
        script_basename = basename

    success = False
    launch_attempt = 0
    calculation_started = False

    while launch_attempt < max_launch_retries and not calculation_started:
        launch_attempt += 1

        # STEP 1: CLEAN OUTPUT FILES
        # Preserve .hessian/.hess files — they are rescue Hessians placed by
        # redo logic for xTB/ORCA to read as initial Hessian guess.
        for item in os.listdir(calc_working_dir):
            if item.startswith(script_basename) and not item.endswith(('.inp', '.com', '.gjf', '.xyz', '.hessian', '.hess')):
                item_path = os.path.join(calc_working_dir, item)
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                    except:
                        pass

        # STEP 2: RUN CALCULATION
        temp_script = os.path.join(calc_working_dir, f'_run_{script_basename}.sh')
        with open(temp_script, 'w') as f:
            f.write(launcher_content.split('###')[0])
            f.write("\n\n")
            if qm_program == 'orca':
                f.write(f"# Set unique scratch directory for this ORCA process\n")
                f.write(f"export TMPDIR=\"$(pwd)/.orca_tmp_{script_basename}_$$\"\n")
                f.write(f"mkdir -p \"$TMPDIR\"\n")
                f.write(f"trap 'rm -rf \"$TMPDIR\"' EXIT\n\n")
                f.write(f"{orca_exe} {input_file_relative} > {output_file_relative}\n")
            elif qm_program == 'gaussian':
                f.write(f"$GAUSS_ROOT/g16 {input_file_relative}\n")
            elif qm_program == 'xtb':
                f.write(f"export {_xtb_thread_env_prefix()}\n")
                f.write(f"{_xtb_thread_env_prefix()} xtb {input_file_relative} {xtb_options} --namespace {script_basename} > {output_file_relative} 2>&1\n")

        os.chmod(temp_script, 0o755)
        script_name = os.path.basename(temp_script)

        start_time = time.time()
        if sys.platform == 'win32':
            # Windows has no usable POSIX bash for the launcher script (it is
            # absent, or maps to WSL which cannot exec the native Windows QM
            # binaries), so run the QM program directly. POSIX keeps the bash
            # launcher path below unchanged.
            _run_qm_job_direct(
                qm_program, calc_working_dir, input_file_relative,
                output_file_relative, script_basename, launcher_content,
                xtb_options, orca_exe,
            )
        else:
            subprocess.run(
                ['bash', script_name],
                cwd=calc_working_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=_PDEATHSIG_PREEXEC,
            )
        elapsed_time = time.time() - start_time

        if elapsed_time > launch_failure_threshold:
            calculation_started = True

        # Cleanup temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)
        if qm_program == 'orca':
            for tmp_dir in glob.glob(os.path.join(calc_working_dir, f'.orca_tmp_{script_basename}_*')):
                try:
                    shutil.rmtree(tmp_dir)
                except:
                    pass
            # Remove ORCA per-job .tmp files left in the job directory after success or failure
            for tmp_f in glob.glob(os.path.join(calc_working_dir, f'{script_basename}.*.tmp')):
                try:
                    os.remove(tmp_f)
                except:
                    pass
            # Remove stray ORCA .tmp.N files that land in the parent directory
            parent_dir = os.path.dirname(calc_working_dir)
            if parent_dir and parent_dir != calc_working_dir:
                for tmp_f in glob.glob(os.path.join(parent_dir, f'{script_basename}*.tmp*')):
                    if os.path.isfile(tmp_f):
                        try:
                            os.remove(tmp_f)
                        except:
                            pass

        # STEP 3: CHECK SUCCESS
        if os.path.exists(output_path):
            try:
                normal_term = check_qm_output_completed(qm_program, output_path)

                if normal_term:
                    success = True
                    break
            except:
                pass

        if calculation_started:
            break

    return {
        'input_file': input_file,
        'success': success,
        'launch_attempts': launch_attempt,
        'calculation_started': calculation_started,
        'output_path': output_path,
    }


def _run_qm_calculations_with_concurrency(
    pending_jobs: List[Dict[str, Any]],
    concurrent_jobs: int,
    workflow_concise: bool,
    context: 'WorkflowContext',
    initial_completed_count: int,
    num_inputs: int,
    completed_list: List[str],
    all_input_basenames: set,
    cache_file: str,
    stage_key_prefix: str = 'calculation',
) -> Tuple[int, int, List[str]]:
    """
    Run QM calculations with the specified concurrency level.

    Args:
        pending_jobs: List of job_info dicts for _run_single_qm_job
        concurrent_jobs: Number of concurrent jobs (1 = serial)
        workflow_concise: Whether to suppress verbose output
        context: WorkflowContext for progress updates
        initial_completed_count: Number already completed before this run
        num_inputs: Total expected input count
        completed_list: Mutable list of completed file basenames (updated in place)
        all_input_basenames: Mutable set of all input basenames (updated in place)
        cache_file: Path to protocol cache file
        stage_key_prefix: Key prefix for cache updates

    Returns:
        (num_completed, num_failed, failed_files)
    """
    import threading

    num_completed = 0
    num_failed = 0
    failed_files: List[str] = []
    lock = threading.Lock()

    def _process_result(result: Dict[str, Any]):
        nonlocal num_completed, num_failed
        input_file = result['input_file']
        display_name = os.path.basename(input_file)

        if result['success']:
            with lock:
                num_completed += 1
                # Store the canonical structure key, never the raw path/extension
                # (e.g. 'motif_24_opt.inp' -> 'motif_24'); otherwise the same
                # structure is counted twice under two spellings and the N/M total
                # grows past the real motif count.
                basename_only = _canonical_calc_key(input_file)
                if basename_only not in completed_list:
                    completed_list.append(basename_only)
                all_input_basenames.add(basename_only)
                stage_key = getattr(context, 'current_stage_key', stage_key_prefix)
                update_protocol_cache(stage_key, 'in_progress',
                                      result={'completed_files': completed_list,
                                             'total_files': num_inputs,
                                             'num_completed': num_completed},
                                      cache_file=cache_file)
                current_total = initial_completed_count + num_completed
                if current_total > num_inputs:
                    current_total = num_inputs
                progress_cb = context.update_progress
                if workflow_concise and callable(progress_cb):
                    progress_cb(f"{current_total}/{num_inputs} ...")
                else:
                    if result['launch_attempts'] > 1:
                        print(f"\r  Running: {display_name}... ✓ (launch attempt {result['launch_attempts']})\033[K")
                    else:
                        print(f"\r  Running: {display_name}... ✓\033[K")
        else:
            with lock:
                num_failed += 1
                failed_files.append(input_file)
                if not workflow_concise:
                    if result['calculation_started']:
                        print(f"\r  Running: {display_name}... ✗ (no normal termination)\033[K")
                    elif result['launch_attempts'] >= result.get('max_launch_retries', 10):
                        print(f"\r  Running: {display_name}... ✗ (launch failed after {result['launch_attempts']} attempts)\033[K")
                    else:
                        print(f"\r  Running: {display_name}... ✗\033[K")

    if not pending_jobs:
        return 0, 0, []

    effective_concurrent = min(concurrent_jobs, len(pending_jobs))

    if effective_concurrent <= 1:
        # Serial execution
        for job in pending_jobs:
            display_name = os.path.basename(job['input_file'])
            if not workflow_concise:
                print(f"  Running: {display_name}...", end='', flush=True)
            else:
                progress_cb = getattr(context, 'update_progress', None)
                if callable(progress_cb):
                    _curr = initial_completed_count + num_completed
                    if _curr > num_inputs:
                        _curr = num_inputs
                    progress_cb(f"{_curr}/{num_inputs} ...")
            result = _run_single_qm_job(job)
            _process_result(result)
    else:
        # Concurrent execution
        if not workflow_concise:
            print(f"  Running {len(pending_jobs)} calculations ({effective_concurrent} concurrent)...")

        with ThreadPoolExecutor(max_workers=effective_concurrent) as executor:
            futures = {executor.submit(_run_single_qm_job, job): job for job in pending_jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    _process_result(result)
                except Exception as e:
                    job = futures[future]
                    with lock:
                        num_failed += 1
                        failed_files.append(job['input_file'])
                        if not workflow_concise:
                            print(f"  Error running {os.path.basename(job['input_file'])}: {e}")

    return num_completed, num_failed, failed_files


# =========================================================================== #
# R6c verbatim ports — the four execute_*_stage runners, the Hessian-rescue    #
# protocol and the cross-subsystem helpers the orchestrator calls.             #
#                                                                             #
# Every function below is a byte-identical extract of ascec-v04.py (D-039:     #
# faithful decomposition). Extracted programmatically, not retyped. v04 line   #
# ranges: plot_annealing_diagrams 2237-2349, plot_combined_replicas_diagram    #
# 2352-2469, generate_protocol_summary 2472-2959, create_qm_input_file         #
# 5107-5198, create_xyz_input_file 5201-5230, get_box_size_recommendation      #
# 6941-6976, check_orca_terminated_normally_opi 7378-7419,                     #
# detect_convergence_status 7422-7536, parse_rescue_method 7939-8049,          #
# generate_rescue_hessian_input 8052-8103, enable_hessian_restart 8106-8181,   #
# _run_xtb_rescue_hessian 8184-8291, run_rescue_hessian_calculation 8294-8477, #
# execute_optimization_stage 16026-16768, execute_cosmic_stage 16770-17301,    #
# execute_refinement_stage 17303-18332, execute_energy_refinement_stage        #
# 18335-18343.                                                                 #
# =========================================================================== #


def plot_annealing_diagrams(tvse_file: str, output_dir: str, scaled: bool = False):
    """
    Generate Energy vs Step and Energy vs Temperature diagrams from tvse_*.dat file.
    
    Args:
        tvse_file: Path to tvse_*.dat file
        output_dir: Directory to save the plots
        scaled: If True, apply intelligent y-axis scaling (remove initial high-energy points)
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return False
    
    if not os.path.exists(tvse_file):
        return False
    
    try:
        # Read tvse data file
        # Format: Step Temperature Energy
        steps = []
        temperatures = []
        energies = []
        
        with open(tvse_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        step = int(parts[0])
                        temp = float(parts[1])
                        energy = float(parts[2])
                        steps.append(step)
                        temperatures.append(temp)
                        energies.append(energy)
                    except ValueError:
                        continue
        
        if not steps:
            return False
        
        # Calculate intelligent y-axis scaling if requested
        y_min = None
        y_max = None
        if scaled and energies:
            sorted_energies = sorted(energies)
            
            # Find the minimum (most negative) energy - this is critical!
            y_min = min(energies)
            
            # For y_max: exclude only the top 5% highest energies (initial hot configurations)
            percentile_95_idx = int(len(sorted_energies) * 0.95)
            y_max_candidate = sorted_energies[percentile_95_idx]
            
            # Add small margins for visualization
            energy_range = y_max_candidate - y_min
            y_min = y_min - 0.02 * abs(energy_range)  # 2% margin below minimum
            y_max = y_max_candidate + 0.05 * abs(energy_range)  # 5% margin above 95th percentile
        
        # Create figure with two subplots with more separation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'hspace': 0.35})
        
        # Plot 1: Energy vs Step Number
        ax1.plot(steps, energies, 'k-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax1.set_title('Simulated Annealing: Energy Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2, linewidth=0.5)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Apply scaled y-limits if requested
        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # Plot 2: Energy vs Temperature
        ax2.scatter(temperatures, energies, c='black', s=10, alpha=0.5, edgecolors='none')
        ax2.set_xlabel('Temperature [K]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Distribution vs Temperature', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.2, linewidth=0.5)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Apply scaled y-limits to second plot if requested
        if y_min is not None and y_max is not None:
            ax2.set_ylim(y_min, y_max)
        
        # Suppress tight_layout warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        
        # Save figure
        basename = os.path.splitext(os.path.basename(tvse_file))[0]
        output_file = os.path.join(output_dir, f"{basename}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        # Log error for debugging
        if os.environ.get('ASCEC_DEBUG'):
            import sys
            print(f"Warning: Failed to generate diagram for {os.path.basename(tvse_file)}: {e}", file=sys.stderr)
        return False


def plot_combined_replicas_diagram(tvse_files: List[str], output_file: str, num_replicas: int):
    """
    Generate a single combined Energy vs Step diagram for all replicas.
    
    Args:
        tvse_files: List of paths to tvse_*.dat files
        output_file: Output path for the combined diagram
        num_replicas: Number of replicas
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return False
    
    if not tvse_files:
        return False
    
    try:
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect all energies first to determine optimal y-axis range
        all_energies = []
        replica_data = []
        
        # Plot each replica and extract replica number from filename
        for idx, tvse_file in enumerate(tvse_files, 1):
            if not os.path.exists(tvse_file):
                continue
            
            steps = []
            energies = []
            
            with open(tvse_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            step = int(parts[0])
                            energy = float(parts[2])
                            steps.append(step)
                            energies.append(energy)
                            all_energies.append(energy)
                        except ValueError:
                            continue
            
            if steps:
                # Extract replica number from path (e.g., "annealing_1" -> 1)
                replica_label = f"ann_{idx}"
                parent_dir = os.path.basename(os.path.dirname(tvse_file))
                # Try to extract number from directory name
                import re
                match = re.search(r'_(\d+)$', parent_dir)
                if match:
                    replica_label = f"ann_{match.group(1)}"
                
                replica_data.append((steps, energies, replica_label))
        
        # Calculate optimal y-axis range by removing only initial high-energy points
        # Keep ALL final energies (they show convergence to minima - very important!)
        y_min = None
        y_max = None
        
        if all_energies:
            sorted_energies = sorted(all_energies)
            
            # Find the minimum (most negative) energy - this is critical!
            y_min = min(all_energies)
            
            # For y_max: exclude only the top 5% highest energies (initial hot configurations)
            # This removes the initial spike without affecting the convergence region
            percentile_95_idx = int(len(sorted_energies) * 0.95)
            y_max_candidate = sorted_energies[percentile_95_idx]
            
            # Add small margins for visualization
            energy_range = y_max_candidate - y_min
            y_min = y_min - 0.02 * abs(energy_range)  # 2% margin below minimum
            y_max = y_max_candidate + 0.05 * abs(energy_range)  # 5% margin above 95th percentile
        
        # Now plot all replica data with labels
        for steps, energies, label in replica_data:
            ax.plot(steps, energies, linewidth=0.8, alpha=0.6, label=label)
        
        # Apply the calculated y-axis range
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy [Hartrees]', fontsize=12, fontweight='bold')
        ax.set_title(f'Simulated Annealing: Energy Evolution (r{num_replicas})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # Make secondary axes less prominent
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend at top right without box
        if replica_data:
            ax.legend(loc='upper right', frameon=False, fontsize=10)
        
        # Suppress tight_layout warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        # Log error for debugging
        if os.environ.get('ASCEC_DEBUG'):
            import sys
            print(f"Warning: Failed to generate combined diagram: {e}", file=sys.stderr)
        return False


def generate_protocol_summary(cache_file: str = "protocol_cache.pkl", 
                              output_file: str = "protocol_summary.txt"):
    """Generate comprehensive protocol summary from cache data with professional formatting."""
    cache = load_protocol_cache(cache_file)
    
    if not cache:
        print("Warning: No cache data found for summary generation")
        return
    
    def format_duration(seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{days}d {hours}h {mins}m"
    
    def format_wall_time_timing(seconds: float) -> str:
        """Format wall time as H:MM:SS.mmm for timing breakdown."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:06.3f}"
    
    def center_text(text: str, width: int = 75) -> str:
        """Center text within a fixed-width field for box-drawing alignment."""
        if width <= 0:
            return ""
        if len(text) >= width:
            return text[:width]
        return text.center(width)

    def format_concise_path(path_value: str) -> str:
        """Render paths concisely for summary output (e.g., /cosmic_2/orca_out_3)."""
        if not path_value:
            return path_value

        raw = str(path_value).strip()
        if not raw:
            return raw

        # Keep simple labels (e.g., Annealing) untouched.
        if not any(sep in raw for sep in ('/', '\\')):
            return raw

        normalized = os.path.normpath(raw)
        parts = [p for p in normalized.split(os.sep) if p and p != '.']
        if not parts:
            return raw

        concise_parts = parts[-2:] if len(parts) >= 2 else parts
        concise = os.sep + os.sep.join(concise_parts)
        if raw.endswith(os.sep) and not concise.endswith(os.sep):
            concise += os.sep
        return concise
    
    def _stage_sort_key(stage_key: str) -> int:
        """Extract trailing stage number from keys like 'cosmic_3' or 'energy_refinement_6'."""
        match = re.search(r'_(\d+)$', stage_key)
        return int(match.group(1)) if match else 0

    def _stage_type_name(stage_key: str) -> str:
        """Extract stage type from keys like 'cosmic_3' or 'energy_refinement_6'.

        Returns a capitalized type string usable in type_map lookups.
        """
        # Strip trailing _N
        base = re.sub(r'_\d+$', '', stage_key)
        return base.capitalize()  # e.g. 'Energy_refinement', 'Cosmic', 'Refinement'

    def _extract_time_from_orca_summary(summary_file: str) -> Optional[float]:
        """Extract total execution time from orca_summary.txt file."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                match = re.search(r'Total execution time:\s+(\d+):(\d+):(\d+\.\d+)', content)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return None

    def _extract_energy_evals_from_annealing(annealing_dir: str) -> Optional[int]:
        """Extract total energy evaluations from annealing.out file."""
        annealing_out = os.path.join(annealing_dir, 'annealing.out')
        if not os.path.exists(annealing_out):
            return None
        try:
            with open(annealing_out, 'r') as f:
                content = f.read()
                # Look for "Energy calculations: XXXX"
                match = re.search(r'Energy calculations:\s+(\d+)', content)
                if match:
                    return int(match.group(1))
        except:
            pass
        return None

    def _extract_final_clusters_from_summary(summary_file: str) -> Optional[int]:
        """Extract final cluster count from clustering_summary.txt."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            match = re.search(r'Total number of final clusters:\s*(\d+)', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def _resolve_cosmic_summary_file(result: Dict[str, Any]) -> Optional[str]:
        """Resolve clustering_summary.txt for a cosmic result entry."""
        cosmic_path = result.get('working_dir') or result.get('cosmic_folder')
        if not cosmic_path:
            return None

        cosmic_base = cosmic_path
        base_name = os.path.basename(os.path.normpath(cosmic_path))
        if base_name.startswith('orca_out_') or base_name.startswith('opt_out_'):
            cosmic_base = os.path.dirname(cosmic_path)

        summary_file = os.path.join(cosmic_base, 'clustering_summary.txt')
        return summary_file if os.path.exists(summary_file) else None
    
    try:
        # encoding='utf-8' is required: the summary contains Unicode glyphs
        # (≤, ✓, box-drawing, Spanish accents) that the default Windows code
        # page (cp1252 on Spanish-locale installs) cannot encode.
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 75 + "\n")
            f.write(center_text("C O S M I C  A S C E C") + "\n")
            f.write(center_text("Configurational Similarity via Motif Identification Code") + "\n")
            f.write(center_text("Annealing Simulado Con Energía Cuántica") + "\n")
            f.write(center_text("Universidad de Antioquia - QFT") + "\n")
            f.write("=" * 75 + "\n\n")
            
            f.write(center_text("Protocol Workflow Summary") + "\n")
            f.write(center_text("-" * 30) + "\n\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # EXECUTION OVERVIEW
            # ══════════════════════════════════════════════════════════════════════
            f.write("┌" + "─" * 73 + "┐\n")
            f.write("│" + center_text("Execution Overview", 73) + "│\n")
            f.write("└" + "─" * 73 + "┘\n\n")
            
            # Timing info
            all_stages = cache.get('stages', {})
            _any_in_progress = any(v.get('status') == 'in_progress' for v in all_stages.values())
            if 'start_time_str' in cache:
                f.write(f"  Started:    {cache['start_time_str']}\n")
            if 'last_update' in cache:
                _upd_label = "Updated:   " if _any_in_progress else "Completed: "
                f.write(f"  {_upd_label} {cache['last_update']}\n")
            
            total_wall_time = 0
            if 'start_time' in cache:
                total_wall_time = time.time() - cache['start_time']
                f.write(f"  Duration:   {format_duration(total_wall_time)}\n")
            
            f.write("\n")
            
            # Workflow diagram (completed + current in-progress stage)
            if 'stages' in cache:
                sorted_stages = sorted(cache['stages'].items(),
                                     key=lambda x: _stage_sort_key(x[0]))

                _wf_type_map = {'Replication': 'annealing', 'Calculation': 'geometry_optimization',
                                'Cosmic': 'cosmic', 'COSMIC': 'cosmic', 'Optimization': 'geometry_optimization',
                                'Refinement': 'geometry_refinement',
                                'Energy_refinement': 'energy_refinement'}
                workflow_parts = []
                for stage_key, stage_info in sorted_stages:
                    stage_type = _stage_type_name(stage_key)
                    stage_name = _wf_type_map.get(stage_type, stage_type.lower())
                    _st = stage_info.get('status')
                    if _st == 'completed':
                        workflow_parts.append(stage_name)
                    elif _st == 'in_progress':
                        workflow_parts.append(f"{stage_name} (running...)")

                f.write(f"  Workflow:   {' → '.join(workflow_parts)}\n\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # TIMING BREAKDOWN
            # ══════════════════════════════════════════════════════════════════════
            if 'stages' in cache and total_wall_time > 0:
                f.write("┌" + "─" * 73 + "┐\n")
                f.write("│" + center_text("Timing Breakdown", 73) + "│\n")
                f.write("└" + "─" * 73 + "┘\n\n")
                
                f.write(f"  {'Stage':<15} {'Duration':>15} {'% Total':>10}\n")
                f.write(f"  {'-' * 15} {'-' * 15} {'-' * 10}\n")
                
                sorted_stages = sorted(cache['stages'].items(), 
                                     key=lambda x: _stage_sort_key(x[0]))
                
                total_qm_percentage = 0.0
                total_tracked_time = 0.0  # Track sum of all stage times
                for stage_key, stage_info in sorted_stages:
                    if stage_info.get('status') == 'completed':
                        stage_type = _stage_type_name(stage_key)

                        # Skip cosmic stages in timing (no QM time)
                        if stage_type.lower() == 'cosmic':
                            continue

                        wall_time = stage_info.get('wall_time')
                        if not wall_time:
                            _result = stage_info.get('result', {})
                            _wdir = _result.get('working_dir')
                            if _wdir:
                                _sf = os.path.join(_wdir, 'orca_summary.txt')
                                if os.path.exists(_sf):
                                    wall_time = _extract_time_from_orca_summary(_sf)

                        if wall_time:
                            total_tracked_time += wall_time
                            percentage = (wall_time / total_wall_time) * 100
                            total_qm_percentage += percentage
                            type_map = {'Replication': 'Annealing', 'Calculation': 'Optimization',
                                      'Optimization': 'Optimization', 'Refinement': 'Refinement',
                                      'Energy_refinement': 'Eref'}
                            stage_name = type_map.get(stage_type, stage_type)
                            f.write(f"  {stage_name:<15} {format_wall_time_timing(wall_time):>15} {percentage:>9.1f}%\n")
                
                # Calculate and show recovery time if there's a significant difference
                recovery_time = total_wall_time - total_tracked_time
                if recovery_time > 60:  # Only show if recovery time > 1 minute
                    recovery_percentage = (recovery_time / total_wall_time) * 100
                    f.write(f"  {'Recovery':<15} {format_wall_time_timing(recovery_time):>15} {recovery_percentage:>9.1f}%\n")
                
                f.write("\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # PROTOCOL DEFINITION
            # ══════════════════════════════════════════════════════════════════════
            if 'protocol_text' in cache:
                f.write("┌" + "─" * 73 + "┐\n")
                f.write("│" + center_text("Protocol Definition", 73) + "│\n")
                f.write("└" + "─" * 73 + "┘\n\n")
                
                protocol_lines = cache['protocol_text'].split(',')
                for line in protocol_lines:
                    line = line.strip()
                    if line:
                        f.write(f"  {line}\n")
                f.write("\n")
            
            # ══════════════════════════════════════════════════════════════════════
            # STAGE DETAILS
            # ══════════════════════════════════════════════════════════════════════
            f.write("┌" + "─" * 73 + "┐\n")
            f.write("│" + center_text("Stage Details", 73) + "│\n")
            f.write("└" + "─" * 73 + "┘\n\n")
            
            if 'stages' in cache:
                sorted_stages = sorted(cache['stages'].items(), 
                                     key=lambda x: _stage_sort_key(x[0]))
                
                step_num = 1
                total_stages = cache.get('total_stages', len([s for s in sorted_stages if s[1].get('status') in ('completed', 'in_progress')]))

                for stage_key, stage_info in sorted_stages:
                    status = stage_info.get('status', 'unknown')

                    if status not in ('completed', 'in_progress'):
                        continue

                    stage_type = _stage_type_name(stage_key)
                    type_map = {'Replication': 'Annealing', 'Calculation': 'Optimization',
                              'Cosmic': 'cosmic', 'Optimization': 'Optimization',
                              'Refinement': 'Refinement',
                              'Energy_refinement': 'Energy Refinement'}
                    stage_name = type_map.get(stage_type, stage_type)

                    result = stage_info.get('result', {})
                    wall_time = stage_info.get('wall_time')

                    # For in-progress stages: show elapsed time and skip detailed result block
                    if status == 'in_progress':
                        elapsed = time.time() - stage_info.get('start_time', time.time())
                        # Show the active redo attempt instead of plain "(running)".
                        _redo_n = result.get('current_redo')
                        if _redo_n:
                            _redo_max = result.get('redo_max')
                            _run_tag = f"(Redo {_redo_n}/{_redo_max})" if _redo_max else f"(Redo {_redo_n})"
                        else:
                            _run_tag = "(running)"
                        f.write(f"  [{step_num}/{total_stages}] {stage_name} ⟳ {_run_tag}\n")
                        f.write(f"  {'─' * 40}\n")
                        f.write(f"    Elapsed:          {format_duration(elapsed)}\n")
                        f.write("\n")
                        step_num += 1
                        continue

                    # Stage header with status indicator
                    status_icon = "✓"
                    f.write(f"  [{step_num}/{total_stages}] {stage_name} {status_icon}\n")
                    f.write(f"  {'─' * 40}\n")
                    
                    # Stage-specific details
                    if stage_type == 'Replication':  # Annealing
                        if result.get('box_size') is not None:
                            f.write(f"    Box size:         {float(result['box_size']):.1f} Å")
                            if result.get('packing') is not None:
                                f.write(f" ({result['packing']}% packing)")
                            f.write("\n")
                        if 'num_replicas' in result:
                            num_replicas = result['num_replicas']
                            replica_desc = "Duplicated" if num_replicas == 2 else "Triplicated" if num_replicas == 3 else f"{num_replicas}x"
                            f.write(f"    Replicas:         {num_replicas} ({replica_desc})\n")
                        if 'energy_evals' in result:
                            f.write(f"    Evaluations:      {result['energy_evals']} times\n")
                        if 'total_accepted' in result:
                            f.write(f"    Accepted:         {result['total_accepted']} configurations\n")
                    
                    elif stage_type.lower() == 'cosmic':
                        live_critical_pct = None
                        live_skipped_pct = None
                        live_critical_count = None
                        live_skipped_count = None
                        live_clusters = None
                        cosmic_summary_file = _resolve_cosmic_summary_file(result)
                        if cosmic_summary_file:
                            live_critical_pct, live_skipped_pct = parse_cosmic_summary(cosmic_summary_file)
                            cosmic_base = os.path.dirname(cosmic_summary_file)
                            live_critical_count, live_skipped_count = parse_cosmic_output(cosmic_base)
                            live_clusters = _extract_final_clusters_from_summary(cosmic_summary_file)

                        if 'cosmic_folder' in result and result['cosmic_folder']:
                            f.write(f"    Working dir:      {format_concise_path(result['cosmic_folder'])}\n")
                        if 'threshold' in result:
                            f.write(f"    Threshold:        {result['threshold']}\n")
                        if 'rmsd_threshold' in result:
                            f.write(f"    RMSD:             {result['rmsd_threshold']}\n")
                        else:
                            f.write(f"    RMSD:             N/A\n")
                        f.write("\n")
                        
                        motifs_created = result.get('motifs_created') if result.get('motifs_created') is not None else live_clusters
                        if motifs_created is not None:
                            motif_label = "Unique Motifs" if ('output_dir' in result and 'umotif' in str(result.get('output_dir', ''))) else "Motifs"
                            label_col = f"    {motif_label}:"
                            f.write(f"{label_col:<22}{motifs_created} representatives\n")

                        # In opt-only mode there are no true minima, so critical/skipped/validation
                        # metrics are not meaningful and must be suppressed.
                        _sim_opt_only = result.get('opt_only', False)
                        if not _sim_opt_only:
                            # Get threshold info for validation output
                            threshold_met = result.get('threshold_met', True)
                            threshold_type = result.get('threshold_type', 'critical')
                            threshold_value = result.get('threshold_value')
                            attempts = result.get('attempts', 0)

                            # Check if there were redo attempts (initial values differ from final)
                            has_redo = 'initial_critical' in result or 'initial_skipped' in result

                            if has_redo and attempts > 0:
                                # Show Initial validation
                                f.write("\n    Initial validation\n")
                                if 'initial_critical' in result:
                                    init_crit = result['initial_critical']
                                    init_crit_count = result.get('initial_critical_count', 0)
                                    f.write(f"    Critical:         {init_crit}% ({init_crit_count} structures)\n")
                                if 'initial_skipped' in result:
                                    init_skip = result['initial_skipped']
                                    init_skip_count = result.get('initial_skipped_count', 0)
                                    f.write(f"    Skipped:          {init_skip}% ({init_skip_count} structures)\n")

                                if threshold_value is not None:
                                    actual_init = result.get(f'initial_{threshold_type}', 'N/A')
                                    f.write(f"\n    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual_init}%\n")

                                # Show Final validation
                                f.write(f"\n    Final validation ({attempts} Redo Attempts)\n")
                                crit_pct = live_critical_pct if live_critical_pct is not None else result.get('critical_pct')
                                crit_count = live_critical_count if live_critical_count is not None else result.get('critical_count', 0)
                                if crit_pct is not None:
                                    f.write(f"    Critical:         {crit_pct}% ({crit_count} structures)\n")
                                skip_pct = live_skipped_pct if live_skipped_pct is not None else result.get('skipped_pct')
                                skip_count = live_skipped_count if live_skipped_count is not None else result.get('skipped_count', 0)
                                if skip_pct is not None:
                                    f.write(f"    Skipped:          {skip_pct}% ({skip_count} structures)\n")

                                if threshold_value is not None:
                                    actual_final = result.get(f'{threshold_type}_pct', 'N/A')
                                    f.write(f"\n    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual_final}%\n")

                                if threshold_met:
                                    f.write(f"\n    Validation: Step [{step_num-1}] passed ✓\n")
                                else:
                                    f.write(f"\n    Max redo attempts ({attempts}) reached\n")
                            else:
                                # No redo - show single validation
                                f.write("\n")
                                crit_pct = live_critical_pct if live_critical_pct is not None else result.get('critical_pct')
                                crit_count = live_critical_count if live_critical_count is not None else result.get('critical_count', 0)
                                if crit_pct is not None:
                                    f.write(f"    Critical:         {crit_pct}% ({crit_count} structures)\n")
                                skip_pct = live_skipped_pct if live_skipped_pct is not None else result.get('skipped_pct')
                                skip_count = live_skipped_count if live_skipped_count is not None else result.get('skipped_count', 0)
                                if skip_pct is not None:
                                    f.write(f"    Skipped:          {skip_pct}% ({skip_count} structures)\n")

                                # Threshold validation
                                if threshold_value is not None:
                                    actual = result.get(f'{threshold_type}_pct', 'N/A')
                                    if threshold_met:
                                        f.write(f"\n    Validation: Step [{step_num-1}] passed ✓\n")
                                        f.write(f"    {threshold_type.capitalize()} ≤ {threshold_value}%\n")
                                        f.write(f"    Redo: 0 attempts (threshold met on first run)\n")
                                    else:
                                        f.write(f"\n    Validation: Step [{step_num-1}] threshold exceeded!\n")
                                        f.write(f"    Target: {threshold_type} ≤ {threshold_value}% | Actual: {actual}%\n")
                    
                    elif stage_type in ('Calculation', 'Optimization'):
                        if 'xyz_source' in result and result['xyz_source']:
                            xyz_source = result['xyz_source'] if result['xyz_source'] else "annealing"
                            f.write(f"    Inputs from:      {xyz_source}\n")
                        if 'completed' in result and 'total' in result:
                            f.write(f"    Completed:        {result['completed']}/{result['total']} optimizations\n")
                        if 'concurrent_jobs' in result:
                            f.write(f"    Concurrent:       {result['concurrent_jobs']} jobs\n")
                        # Mean exec time: use total CPU time / completed (accurate with concurrency).
                        # Try cache first, then live parse from orca_summary.txt as fallback.
                        _total_cpu = result.get('total_cpu_time')
                        if not _total_cpu and 'working_dir' in result:
                            _sum_file = os.path.join(result['working_dir'], 'orca_summary.txt')
                            _total_cpu = _extract_time_from_orca_summary(_sum_file)
                        if _total_cpu and 'completed' in result and result['completed'] > 0:
                            mean_time = _total_cpu / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        elif wall_time and 'completed' in result and result['completed'] > 0:
                            mean_time = wall_time / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        if 'cosmic_folder' in result and result['cosmic_folder']:
                            f.write(f"    Outputs to:       {format_concise_path(result['cosmic_folder'])}\n")

                    elif stage_type in ('Refinement', 'Energy_refinement'):
                        if 'motifs_source' in result and result['motifs_source']:
                            f.write(f"    Inputs from:      {format_concise_path(result['motifs_source'])}\n")
                        if 'completed' in result and 'total' in result:
                            label = "calculations" if stage_type == 'Energy_refinement' else "refinements"
                            f.write(f"    Completed:        {result['completed']}/{result['total']} {label}\n")
                        if 'concurrent_jobs' in result:
                            f.write(f"    Concurrent:       {result['concurrent_jobs']} jobs\n")
                        # Mean exec time: prefer total_cpu_time/completed (accurate with concurrency).
                        # Try cache first, then live parse from orca_summary.txt as fallback.
                        _total_cpu = result.get('total_cpu_time')
                        if not _total_cpu and 'working_dir' in result:
                            _sum_file = os.path.join(result['working_dir'], 'orca_summary.txt')
                            _total_cpu = _extract_time_from_orca_summary(_sum_file)
                        if _total_cpu and 'completed' in result and result['completed'] > 0:
                            mean_time = _total_cpu / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        elif wall_time and 'completed' in result and result['completed'] > 0:
                            mean_time = wall_time / result['completed']
                            f.write(f"    Mean exec time:   {format_wall_time_timing(mean_time)}\n")
                        if 'cosmic_folder' in result and result['cosmic_folder']:
                            f.write(f"    Outputs to:       {format_concise_path(result['cosmic_folder'])}\n")

                    # Wall time for non-cosmic stages
                    if wall_time and stage_type.lower() != 'cosmic':
                        f.write(f"    Wall time:        {format_wall_time_timing(wall_time)}\n")
                    
                    f.write("\n")
                    step_num += 1
            
            # ══════════════════════════════════════════════════════════════════════
            # FOOTER
            # ══════════════════════════════════════════════════════════════════════
            f.write("=" * 75 + "\n")
            _footer_msg = "Workflow in progress..." if _any_in_progress else "Workflow completed successfully"
            f.write(center_text(_footer_msg) + "\n")
            f.write("=" * 75 + "\n")

        # Only print confirmation message when the workflow is fully done (avoid noise during run)
        if not _any_in_progress:
            # ASCII-only: cp1252 (default Windows console) cannot encode '✓'.
            print(f"Protocol summary saved to {output_file}")
        
    except Exception as e:
        print(f"Warning: Failed to generate protocol summary: {e}")


def create_qm_input_file(config_data: Dict, template_content: str, output_path: str, qm_program: str) -> bool:
    """
    Creates a QM input file from configuration data and template.
    
    Args:
        config_data (Dict): Configuration data with atoms, energy, etc.
        template_content (str): Template file content
        output_path (str): Path where to save the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create coordinates section (preserve original precision with proper alignment)
        coords_section = ""
        for atom in config_data['atoms']:
            if len(atom) == 7:  # New format with string coordinates
                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                # Right-align coordinates to maintain column alignment while preserving original precision
                coords_section += f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n"
            else:  # Old format compatibility
                symbol, x, y, z = atom
                coords_section += f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n"
        
        # Replace name placeholders with the configuration comment (only if placeholders exist)
        content = template_content
        if qm_program == 'orca':
            # ORCA uses # for placeholders
            if "#name" in content:
                content = content.replace("#name", f"# {config_data['comment']}")
        elif qm_program == 'gaussian':
            # Gaussian uses ! for placeholders
            if "!name" in content:
                content = content.replace("!name", f"! {config_data['comment']}")
        
        if qm_program == 'orca':
            # For ORCA, replace the coordinate section between * xyz and *
            # Look for * xyz pattern (case-insensitive) with charge and multiplicity
            import re
            
            lines = content.split('\n')
            new_lines = []
            in_coords = False
            xyz_pattern = re.compile(r'^\s*\*\s+xyz\s+[-\d]+\s+\d+\s*$', re.IGNORECASE)
            
            for line in lines:
                if xyz_pattern.match(line.strip()):
                    new_lines.append(line)
                    in_coords = True
                elif in_coords and line.strip() == "*":
                    new_lines.append(line)
                    in_coords = False
                elif in_coords and line.strip() == "#":
                    # Replace the # placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                elif in_coords:
                    # Skip other coordinate lines (they will be replaced by coords_section when we hit #)
                    continue
                elif not in_coords:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        elif qm_program == 'gaussian':
            # For Gaussian, replace ! placeholder with coordinates
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.strip() == "!":
                    # Replace the ! placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the input file
        with open(output_path, 'w') as f:
            f.write(content)
        
        return True
        
    except IOError as e:
        print(f"Error creating QM input file '{output_path}': {e}")
        return False


def create_xyz_input_file(config_data: Dict, output_path: str) -> bool:
    """
    Creates an XYZ file directly from configuration data.

    This is used by standalone xTB, which reads Cartesian coordinates
    from XYZ files directly instead of a separate QM input template.
    """
    try:
        atoms = config_data.get('atoms', [])
        comment = config_data.get('comment', 'Generated by ASCEC')

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom in atoms:
                if len(atom) == 7:
                    symbol, x_str, y_str, z_str, _, _, _ = atom
                    f.write(f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n")
                else:
                    symbol, x, y, z = atom
                    f.write(f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n")

        return True
    except IOError as e:
        print(f"Error creating XYZ input file '{output_path}': {e}")
        return False


def get_box_size_recommendation(input_file_path: str, packing_percent: float = 20.0) -> Optional[float]:
    """
    Runs box analysis and computes the recommended box size for any packing percentage.

    Args:
        input_file_path (str): Path to the input file
        packing_percent (float): Desired packing percentage (default: 20.0)

    Returns:
        Optional[float]: Recommended box size in Angstroms, or None if analysis failed
    """
    state = SystemState()
    state.verbosity_level = 0  # Suppress verbose output

    try:
        input_file_path_full = os.path.abspath(input_file_path)
        read_input_file(state, input_file_path_full)

        # Calculate directly for the requested packing fraction
        packing_fraction = packing_percent / 100.0
        if packing_fraction <= 0 or packing_fraction > 1:
            return None

        results = calculate_optimal_box_length(state, target_packing_fractions=[packing_fraction])
        if 'error' in results:
            return None

        key = f"{packing_fraction:.1%}"
        if key in results['box_length_recommendations']:
            return results['box_length_recommendations'][key].get('box_length_A')

        return None

    except Exception as e:
        print(f"Warning: Could not determine box size recommendation: {e}")
        return None


def check_orca_terminated_normally_opi(output_path: str) -> bool:
    """
    Check if ORCA calculation terminated normally using OPI for ORCA 6.1+.
    Falls back to text search for ORCA 5.x or when OPI is not available.
    
    Args:
        output_path: Path to ORCA output file (.out)
        
    Returns:
        True if terminated normally, False otherwise
    """
    if not os.path.exists(output_path):
        return False
    
    # Detect ORCA version
    version = detect_orca_version(output_path)
    
    # For ORCA 6.1+, try to use OPI
    if version and version >= (6, 1) and OPI_AVAILABLE and OPIOutput is not None:
        try:
            file_path = Path(output_path)
            basename = file_path.stem
            working_dir = file_path.parent
            
            opi_output = OPIOutput(
                basename=basename,
                working_dir=working_dir,
                version_check=False
            )
            
            return opi_output.terminated_normally()
        except Exception:
            # Fall back to text search if OPI fails
            pass
    
    # Fall back to text search (works for all ORCA versions)
    try:
        with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return '****ORCA TERMINATED NORMALLY****' in content
    except Exception:
        return False


def detect_convergence_status(logfile_path: str) -> dict:
    """
    Detect the convergence status of an optimization output file.

    Supports ORCA, Gaussian, and standalone xTB output formats.

    Args:
        logfile_path: Path to output file (.out for ORCA/xTB, .log for Gaussian)

    Returns:
        dict with keys:
            'status': 'converged', 'not_converged', 'error', or 'unknown'
            'terminated_normally': bool
            'max_iterations_reached': bool
            'error_message': str or None
    """
    result = {
        'status': 'unknown',
        'terminated_normally': False,
        'max_iterations_reached': False,
        'error_message': None
    }

    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        content_lower = content.lower()

        # Detect if this is standalone xTB output
        is_xtb = (
            'x]  t  b' in content_lower
            or 'xtb version' in content_lower
            or 'normal termination of xtb' in content_lower
            or ('total energy' in content_lower and 'gfn' in content_lower)
        )

        if is_xtb:
            # --- Standalone xTB convergence detection ---
            if 'normal termination of xtb' in content_lower:
                result['terminated_normally'] = True
                result['status'] = 'converged'

            # xTB: explicit convergence message
            if 'geometry optimization converged' in content_lower:
                result['status'] = 'converged'
                result['terminated_normally'] = True

            # xTB: optimization did not converge
            xtb_not_converged = [
                'geometry optimization did not converge',
                'exceeded the maximum number of optimization cycles',
                'optimization did not converge',
            ]
            for pattern in xtb_not_converged:
                if pattern in content_lower:
                    result['max_iterations_reached'] = True
                    result['status'] = 'not_converged'
                    result['error_message'] = 'xTB optimization reached maximum cycles without converging'
                    break

            # xTB: SCC convergence failure
            if 'scc not converged' in content_lower or 'scf not converged' in content_lower:
                result['status'] = 'error'
                result['error_message'] = 'xTB SCC/SCF did not converge'

            # xTB: runtime/setup error
            if 'error termination' in content_lower or 'abnormal termination' in content_lower:
                if result['status'] != 'not_converged':
                    result['status'] = 'error'
                    result['error_message'] = 'xTB error termination'

            return result

        # --- ORCA convergence detection ---
        # Check for normal termination
        if 'ORCA TERMINATED NORMALLY' in content:
            result['terminated_normally'] = True
            result['status'] = 'converged'

        # Check for optimization not converged error (max iterations reached)
        not_converged_patterns = [
            'The optimization did not converge but reached the maximum',
            'maximum number of optimization cycles',
            'ORCA will abort at this point',
            'Please restart the calculation with the lowest energy geometry',
            'larger maxiter for the geometry optimization'
        ]

        for pattern in not_converged_patterns:
            if pattern in content:
                result['max_iterations_reached'] = True
                result['status'] = 'not_converged'
                result['error_message'] = 'Optimization reached maximum cycles without converging'
                break

        # Also check for explicit convergence success
        if '*** THE OPTIMIZATION HAS CONVERGED ***' in content:
            result['status'] = 'converged'
            result['max_iterations_reached'] = False

        # If terminated normally but no explicit convergence indicator, check for geometry opt
        if result['terminated_normally'] and '*** OPTIMIZATION RUN DONE ***' in content:
            if result['status'] == 'unknown':
                result['status'] = 'converged'

        # Check for SCF convergence issues
        if 'SCF NOT CONVERGED' in content or 'The wavefunction could not be converged' in content:
            result['status'] = 'error'
            result['error_message'] = 'SCF did not converge'

    except Exception as e:
        result['error_message'] = str(e)

    return result


def parse_rescue_method(input_content: str, orca_path: Optional[str] = None,
                         launcher_content: Optional[str] = None,
                         qm_program: Optional[str] = None) -> Tuple[str, bool]:
    """
    Parse the #rescue(Method) directive from a QM input/template file, or detect
    the rescue method from the template's method line.

    Logic:
    1. If #rescue(method) is specified, use that method
    2. If #rescue(method,num) is specified, use that method with NumFreq
    3. If template uses an xTB method, reuse that method with NumFreq
       (converting to native for ORCA 6.1+ or non-native for 5.x)
    4. Default: HF-3c with Freq

    For standalone xTB (qm_program='xtb'), ORCA version conversion is skipped
    and xTB methods default to analytical Freq (use_numfreq=False).

    Example input:
        #rescue(Native-GFN2-xTB)       -> Uses Native-GFN2-xTB with NumFreq (xTB)
        #rescue(b97-c)                 -> Uses b97-c with Freq
        #rescue(b97-c,num)             -> Uses b97-c with NumFreq
        #rescue(GFN2-xTB/freq)         -> Uses GFN2-xTB with analytical Freq (standalone xTB)
        Template: ! Native-GFN2-xTB Opt -> Uses Native-GFN2-xTB with NumFreq
        Template: ! B98-3c Opt         -> Uses HF-3c with Freq (default)

    Args:
        input_content: Content of the QM input file
        orca_path: Optional path to ORCA executable (for version detection)
        launcher_content: Optional launcher script content (for version detection from ORCA611_ROOT etc.)
        qm_program: Optional QM program identifier ('orca', 'xtb', 'gaussian')

    Returns:
        Tuple of (rescue_method, use_numfreq) where:
        - rescue_method: Method string (e.g., "Native-GFN2-xTB", "HF-3c", "GFN2-xTB")
        - use_numfreq: True if NumFreq should be used, False for Freq
    """
    is_standalone_xtb = (qm_program == 'xtb')

    def _parse_rescue_spec(spec: str) -> Tuple[str, Optional[bool]]:
        """Parse rescue specifiers like method,num or method/num into method + freq mode."""
        raw = spec.strip()
        if not raw:
            return ('', None)

        # Accept both comma and slash qualifiers, e.g.:
        #   Native-GFN2-xTB/num, HF-3c/freq, b97-c,num
        method = raw
        freq_mode: Optional[bool] = None

        if '/' in raw:
            base, suffix = raw.split('/', 1)
            method = base.strip()
            suffix_token = suffix.strip().lower()
            if suffix_token in ('num', 'numfreq'):
                freq_mode = True
            elif suffix_token in ('freq',):
                freq_mode = False
        elif ',' in raw:
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            method = parts[0]
            for token in parts[1:]:
                token_l = token.lower()
                if token_l in ('num', 'numfreq'):
                    freq_mode = True
                elif token_l in ('freq',):
                    freq_mode = False

        return (method, freq_mode)

    # Look for #rescue(method) / #rescue(method,num) / #rescue(method/num) pattern
    match = re.search(r'#rescue\(([^)]+)\)', input_content, re.IGNORECASE)
    if match:
        method, explicit_numfreq = _parse_rescue_spec(match.group(1))
        if not method:
            if is_standalone_xtb:
                return ('GFN2-xTB', False)
            return ('HF-3c', False)

        if is_xtb_method(method):
            if is_standalone_xtb:
                # Standalone xTB: strip Native- prefix, use analytical Freq by default
                method = re.sub(r'^(Native-)+', '', method, flags=re.IGNORECASE)
                if explicit_numfreq is not None:
                    return (method, explicit_numfreq)
                return (method, False)  # xTB has analytical Hessian
            else:
                # ORCA: convert for ORCA version and default to NumFreq
                method = convert_xtb_for_orca_version(method, orca_path, launcher_content)
                if explicit_numfreq is not None:
                    return (method, explicit_numfreq)
                return (method, True)

        # Non-xTB methods default to Freq unless explicitly overridden.
        if explicit_numfreq is not None:
            return (method, explicit_numfreq)
        return (method, False)

    # No #rescue directive - check if template uses xTB method
    xtb_method = detect_xtb_in_template(input_content)
    if xtb_method:
        if is_standalone_xtb:
            method = re.sub(r'^(Native-)+', '', xtb_method, flags=re.IGNORECASE)
            return (method, False)  # xTB analytical Hessian
        # Reuse the template's xTB method, converted for ORCA version
        converted_method = convert_xtb_for_orca_version(xtb_method, orca_path, launcher_content)
        return (converted_method, True)  # xTB methods use NumFreq in ORCA

    # Default
    if is_standalone_xtb:
        return ('GFN2-xTB', False)
    return ('HF-3c', False)


def generate_rescue_hessian_input(template_content: str, rescue_method: str, xyz_coords: str, 
                                   charge: int = 0, multiplicity: int = 1, 
                                   nprocs: int = 8, use_numfreq: Optional[bool] = None) -> str:
    """
    Generate an ORCA input file for calculating the Hessian with the rescue method.
    
    Args:
        template_content: Original input file content (for extracting settings)
        rescue_method: Method to use for Hessian calculation (e.g., "Native-GFN2-xTB")
        xyz_coords: XYZ coordinates block (without first two lines)
        charge: Molecular charge
        multiplicity: Spin multiplicity
        nprocs: Number of processors to use
        use_numfreq: If True, use NumFreq; if False, use Freq; if None, auto-detect
        
    Returns:
        ORCA input file content for Hessian calculation
    """
    # Check if it's a semiempirical/xTB method (no basis set needed)
    rescue_upper = rescue_method.upper()
    is_xtb = any(kw in rescue_upper for kw in ['GFN', 'XTB'])
    is_semiempirical = any(kw in rescue_upper for kw in ['PM3', 'PM6', 'PM7', 'AM1', 'MNDO'])
    
    # Determine whether to use NumFreq or Freq
    # If use_numfreq is explicitly set, use that; otherwise auto-detect
    if use_numfreq is None:
        use_numfreq = is_xtb  # xTB methods default to NumFreq
    
    lines = []
    lines.append(f"# Rescue Hessian calculation with {rescue_method}")
    
    # Build the method line
    freq_keyword = "NumFreq" if use_numfreq else "Freq"
    if is_xtb or is_semiempirical:
        lines.append(f"! {rescue_method} {freq_keyword}")
    else:
        lines.append(f"! {rescue_method} {freq_keyword} TightSCF")
    
    lines.append("")
    
    # %pal block - use the same nprocs as the template
    lines.append(f"%pal")
    lines.append(f"  nprocs {nprocs}")
    lines.append(f"end")
    
    lines.append("")
    lines.append(f"* xyz {charge} {multiplicity}")
    lines.append(xyz_coords.strip())
    lines.append("*")
    lines.append("")
    
    return "\n".join(lines)


def enable_hessian_restart(input_path: str, hessian_path: str) -> bool:
    """
    Modify an ORCA input file to use a pre-calculated Hessian for the optimization.
    
    This function automatically adds inhess read and inhessname directives to the
    %geom block. If no %geom block exists, one is created. If the user included
    #inhess read and #inhessname placeholders, they are uncommented.
    
    Args:
        input_path: Path to the ORCA input file to modify
        hessian_path: Path to the .hess file to read
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Get basename of hess file
        hess_basename = os.path.basename(hessian_path)
        
        # First, check if placeholders exist and uncomment them
        has_inhess_placeholder = re.search(r'#\s*inhess\s+read', content, re.IGNORECASE)
        has_inhessname_placeholder = re.search(r'#\s*inhessname\s+["\'][^"\']*["\']', content, re.IGNORECASE)
        
        if has_inhess_placeholder or has_inhessname_placeholder:
            # User provided placeholders - uncomment them
            content = re.sub(r'#\s*inhess\s+read', 'inhess read', content, flags=re.IGNORECASE)
            content = re.sub(
                r'#\s*inhessname\s+["\'][^"\']*["\']',
                f'inhessname "{hess_basename}"',
                content,
                flags=re.IGNORECASE
            )
            # If inhessname wasn't in the file, add it after inhess read
            if 'inhess read' in content.lower() and 'inhessname' not in content.lower():
                content = re.sub(
                    r'(inhess\s+read)',
                    f'\\1\n  inhessname "{hess_basename}"',
                    content,
                    flags=re.IGNORECASE
                )
        else:
            # No placeholders - auto-add to %geom block
            # Check if %geom block already exists
            geom_match = re.search(r'(%geom\b.*?)(end)', content, re.IGNORECASE | re.DOTALL)
            
            if geom_match:
                # %geom block exists - add inhess directives at the end
                geom_block = geom_match.group(1)
                # Check if inhess already exists (shouldn't, but be safe)
                if 'inhess' not in geom_block.lower():
                    new_geom_block = geom_block.rstrip() + f'\n  inhess read\n  inhessname "{hess_basename}"\n'
                    content = content[:geom_match.start()] + new_geom_block + 'end' + content[geom_match.end():]
            else:
                # No %geom block - create one
                # Insert before the coordinate block (* xyz or * xyzfile)
                coord_match = re.search(r'^\s*\*\s*(xyz|xyzfile)', content, re.IGNORECASE | re.MULTILINE)
                if coord_match:
                    geom_block = f'\n%geom\n  inhess read\n  inhessname "{hess_basename}"\nend\n\n'
                    content = content[:coord_match.start()] + geom_block + content[coord_match.start():]
                else:
                    # Fallback: append to end (before last * if it exists)
                    # This is a less common case
                    geom_block = f'\n%geom\n  inhess read\n  inhessname "{hess_basename}"\nend\n'
                    # Insert before the last section
                    content += geom_block
        
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"  Warning: Could not enable Hessian restart: {e}")
        return False


def _run_xtb_rescue_hessian(xyz_file: str, rescue_method: str, working_dir: str,
                            basename: str, charge: int, multiplicity: int,
                            _rescue_log, verbose: bool) -> Optional[str]:
    """
    Run a standalone xTB rescue Hessian calculation.

    xTB has analytical Hessian via --hess. This runs:
        xtb <xyz> --gfn <N> --hess --namespace <basename>_rescue > <basename>_rescue.out

    The Hessian is written to <namespace>.hessian (xTB format).
    We convert it to ORCA .hess format for compatibility with enable_hessian_restart().

    Returns:
        Path to the generated hessian file, or None if failed.
    """
    rescue_namespace = f"{basename}_rescue"
    rescue_out_path = os.path.join(working_dir, f"{rescue_namespace}.out")
    # xTB writes hessian as <namespace>.hessian
    rescue_hessian_xtb = os.path.join(working_dir, f"{rescue_namespace}.hessian")
    # Also produce a .hess path for the caller (we'll use the xTB hessian file directly)
    rescue_hess_path = os.path.join(working_dir, f"{basename}_rescue.hess")

    # Determine GFN level from rescue method
    method_upper = (rescue_method or 'GFN2-xTB').upper()
    if 'GFN-FF' in method_upper or 'GFNFF' in method_upper:
        gfn_flags = '--gfnff'
    elif 'GFN0' in method_upper:
        gfn_flags = '--gfn 0'
    elif 'GFN1' in method_upper:
        gfn_flags = '--gfn 1'
    else:
        gfn_flags = '--gfn 2'

    # Build xTB command
    xyz_basename = os.path.basename(xyz_file)
    charge_flag = f'--chrg {charge}' if charge != 0 else ''
    uhf_flag = f'--uhf {multiplicity - 1}' if multiplicity > 1 else ''
    cmd_parts = ['xtb', xyz_basename, gfn_flags, '--hess',
                 '--namespace', rescue_namespace]
    if charge_flag:
        cmd_parts.append(charge_flag)
    if uhf_flag:
        cmd_parts.append(uhf_flag)

    xtb_cmd = ' '.join(cmd_parts)

    # Create run script
    temp_script = os.path.join(working_dir, f'_run_rescue_{basename}.sh')
    with open(temp_script, 'w') as f:
        f.write("#!/bin/bash\nset -e\n\n")
        f.write(f"cd \"{working_dir}\"\n")
        f.write(f"{xtb_cmd} > \"{os.path.basename(rescue_out_path)}\" 2>&1\n")

    os.chmod(temp_script, 0o755)

    # Run calculation
    try:
        if sys.platform == 'win32':
            # No usable POSIX bash on Windows — run xtb directly, redirecting
            # to the rescue .out exactly as the launcher script would. POSIX
            # keeps the bash path unchanged.
            with open(rescue_out_path, 'wb') as _out_fh:
                result = subprocess.run(
                    shlex.split(xtb_cmd),
                    cwd=working_dir,
                    stdout=_out_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_PDEATHSIG_PREEXEC,
                )
        else:
            result = subprocess.run(
                ['bash', temp_script],
                cwd=working_dir,
                capture_output=True,
                text=True
            )
    except Exception as e:
        _rescue_log(f"  ✗ xTB rescue Hessian calculation error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None

    # Cleanup script
    if os.path.exists(temp_script):
        os.remove(temp_script)

    # Check if xTB hessian was generated
    if os.path.exists(rescue_hessian_xtb):
        # xTB writes its own format. Copy/symlink as .hess for the caller.
        try:
            shutil.copy2(rescue_hessian_xtb, rescue_hess_path)
        except Exception:
            rescue_hess_path = rescue_hessian_xtb
        return rescue_hess_path

    # Fallback: check if plain 'hessian' file was written (no namespace)
    plain_hessian = os.path.join(working_dir, 'hessian')
    if os.path.exists(plain_hessian):
        try:
            shutil.copy2(plain_hessian, rescue_hess_path)
        except Exception:
            rescue_hess_path = plain_hessian
        return rescue_hess_path

    # Failed
    error_msg = "  ✗ xTB rescue Hessian calculation failed"
    if os.path.exists(rescue_out_path):
        try:
            with open(rescue_out_path, 'r', encoding='utf-8', errors='replace') as f:
                out_content = f.read()
            if 'error' in out_content.lower():
                error_lines = [l for l in out_content.split('\n') if 'error' in l.lower()]
                if error_lines:
                    error_msg += f"\n    Error: {error_lines[0][:100]}"
        except Exception:
            pass
    else:
        error_msg += " (No output file generated - xtb may not be in PATH)"

    _rescue_log(error_msg)
    return None


def run_rescue_hessian_calculation(xyz_file: str, rescue_method: str, launcher_path: str,
                                     charge: int = 0, multiplicity: int = 1,
                                     nprocs: int = 8, verbose: bool = False,
                                     output_basename: Optional[str] = None,
                                     use_numfreq: Optional[bool] = None,
                                     qm_program: Optional[str] = None) -> Optional[str]:
    """
    Run a rescue Hessian calculation for a structure that failed to converge.

    For ORCA: generates .inp file with rescue method and runs ORCA.
    For standalone xTB: runs `xtb xyz --hess` (analytical Hessian).

    Args:
        xyz_file: Path to XYZ file with the geometry
        rescue_method: Method to use (e.g., "Native-GFN2-xTB", "GFN2-xTB")
        launcher_path: Path to launcher script
        charge: Molecular charge
        multiplicity: Spin multiplicity
        nprocs: Number of processors
        verbose: If True, print detailed error information
        output_basename: Base name for output files (e.g., "opt_conf_8" -> "opt_conf_8_rescue.hess")
                        If None, derives from xyz_file name
        use_numfreq: If True, use NumFreq; if False, use Freq; if None, auto-detect
        qm_program: QM program ('orca', 'xtb', 'gaussian'). If None, defaults to 'orca'.

    Returns:
        Path to the generated Hessian file, or None if failed
    """
    workflow_ctx = getattr(sys, '_current_workflow_context', None)
    workflow_concise = bool(
        workflow_ctx
        and getattr(workflow_ctx, 'is_workflow', False)
        and getattr(workflow_ctx, 'workflow_verbose_level', 0) < 1
        and not verbose
    )

    def _rescue_log(message: str) -> None:
        if not workflow_concise:
            print(message)

    # Get working directory and basename
    working_dir = os.path.dirname(xyz_file)
    if not working_dir:
        working_dir = "."
    # Make working_dir absolute for reliable path handling
    working_dir = os.path.abspath(working_dir)

    # Use provided basename or derive from xyz_file
    if output_basename:
        basename = output_basename
    else:
        basename = os.path.splitext(os.path.basename(xyz_file))[0]

    # --- Standalone xTB rescue path ---
    if qm_program == 'xtb':
        return _run_xtb_rescue_hessian(
            xyz_file, rescue_method, working_dir, basename,
            charge, multiplicity, _rescue_log, verbose
        )

    # --- ORCA rescue path (default) ---
    # Read XYZ coordinates
    try:
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        # Skip first two lines (atom count and comment)
        xyz_coords = "".join(lines[2:])
    except Exception as e:
        _rescue_log(f"  Error reading XYZ file: {e}")
        return None

    # Generate rescue input (pass use_numfreq for explicit control)
    rescue_input = generate_rescue_hessian_input(
        "", rescue_method, xyz_coords, charge, multiplicity, nprocs, use_numfreq
    )

    # Write rescue input file
    rescue_inp_path = os.path.join(working_dir, f"{basename}_rescue.inp")
    rescue_out_path = os.path.join(working_dir, f"{basename}_rescue.out")
    rescue_hess_path = os.path.join(working_dir, f"{basename}_rescue.hess")

    with open(rescue_inp_path, 'w') as f:
        f.write(rescue_input)

    # Read launcher for environment setup
    try:
        with open(launcher_path, 'r') as f:
            launcher_content = f.read()
    except Exception as e:
        _rescue_log(f"  Error reading launcher: {e}")
        return None

    # Extract environment setup from launcher (before ### separator if present)
    env_setup = launcher_content.split('###')[0]

    # Ensure shebang is present
    if not env_setup.strip().startswith('#!'):
        env_setup = "#!/bin/bash\nset -e\n\n" + env_setup

    # Find the ORCA root variable name from the launcher (e.g., ORCA611_ROOT, ORCA_ROOT, etc.)
    # Look for patterns like: export ORCA*_ROOT=... or ORCA*_ROOT=...
    orca_root_var = None
    for line in env_setup.split('\n'):
        # Match patterns like ORCA611_ROOT, ORCA_ROOT, ORCA5_ROOT, etc.
        match = re.search(r'\b(ORCA\w*_ROOT)\s*=', line)
        if match:
            orca_root_var = match.group(1)
            # Prefer more specific names (e.g., ORCA611_ROOT over ORCA_ROOT)
            if 'ORCA_ROOT' not in orca_root_var or orca_root_var != 'ORCA_ROOT':
                break  # Use the specific one

    # Create run script with full paths
    temp_script = os.path.join(working_dir, f'_run_rescue_{basename}.sh')
    with open(temp_script, 'w') as f:
        f.write(env_setup)
        f.write("\n\n")
        f.write(f"cd \"{working_dir}\"\n")
        # Use the ORCA root variable from the launcher if found, otherwise fall back to PATH
        if orca_root_var:
            f.write(f"\"${orca_root_var}/orca\" \"{os.path.basename(rescue_inp_path)}\" > \"{os.path.basename(rescue_out_path)}\" 2>&1\n")
        else:
            # Fallback: use orca from PATH
            f.write(f"orca \"{os.path.basename(rescue_inp_path)}\" > \"{os.path.basename(rescue_out_path)}\" 2>&1\n")
    
    os.chmod(temp_script, 0o755)
    
    # Run calculation with timeout
    try:
        if sys.platform == 'win32':
            # No usable POSIX bash on Windows — resolve the ORCA executable from
            # the launcher's *_ROOT export and run it directly, redirecting to
            # the rescue .out as the launcher script would. POSIX unchanged.
            _env = {**os.environ, **_parse_launcher_env(env_setup)}
            _root = _env.get(orca_root_var) if orca_root_var else None
            _orca_exe = os.path.join(_root, 'orca') if _root else 'orca'
            with open(rescue_out_path, 'wb') as _out_fh:
                result = subprocess.run(
                    [_orca_exe, os.path.basename(rescue_inp_path)],
                    cwd=working_dir,
                    stdout=_out_fh,
                    stderr=subprocess.STDOUT,
                    env=_env,
                    preexec_fn=_PDEATHSIG_PREEXEC,
                )
        else:
            result = subprocess.run(
                ['bash', temp_script],
                cwd=working_dir,
                capture_output=True,
                text=True
            )
    except Exception as e:
        _rescue_log(f"  ✗ Rescue Hessian calculation error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None
    
    # Cleanup script
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    # Check if Hessian was generated
    if os.path.exists(rescue_hess_path):
        # Success - return path silently (calling code will print status)
        return rescue_hess_path
    else:
        # Detailed error reporting
        error_msg = "  ✗ Rescue Hessian calculation failed"
        
        # Check if output file exists and look for errors
        if os.path.exists(rescue_out_path):
            try:
                with open(rescue_out_path, 'r', encoding='utf-8', errors='replace') as f:
                    out_content = f.read()
                
                # Check for common ORCA errors
                if "ORCA TERMINATED NORMALLY" in out_content:
                    # Calculation finished but no .hess file - check if freq was actually requested
                    if ".hess" not in out_content and "VIBRATIONAL FREQUENCIES" not in out_content:
                        error_msg += " (Freq calculation may not be supported for this method)"
                elif "ERROR" in out_content.upper():
                    # Find the error message
                    error_lines = [line for line in out_content.split('\n') 
                                   if 'ERROR' in line.upper() or 'ABORTING' in line.upper()]
                    if error_lines:
                        error_msg += f"\n    Error: {error_lines[0][:100]}"
                elif "ORCA finished by error termination" in out_content:
                    error_msg += " (ORCA error termination)"
                elif len(out_content) < 100:
                    error_msg += " (Output file is nearly empty - ORCA may not have started)"
                    if verbose:
                        error_msg += f"\n    Launcher check: {result.stderr[:200] if result.stderr else 'No stderr'}"
            except Exception:
                pass
        else:
            error_msg += " (No output file generated - ORCA may not have been found)"
            if verbose and result.stderr:
                error_msg += f"\n    stderr: {result.stderr[:200]}"
        
        _rescue_log(error_msg)
        return None


def execute_optimization_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute optimization stage with automatic retry logic."""
    # Store context globally for access in helper functions
    sys._current_workflow_context = context  # type: ignore[attr-defined]

    args = stage['args']
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1

    # Parse flags - defaults for when flags are not explicitly provided
    max_critical = 0  # Default: 0% critical structures allowed (retry all)
    max_skipped = None    # Not set by default (only --critical is used unless --skipped specified)
    max_stage_redos = 3   # default: 3 stage redos (--redo: redo entire optimization+cosmic)
    concurrent_jobs = 1   # default: 1 concurrent QM job for optimization
    _critical_set = False
    _skipped_set = False
    _concurrent_given = False
    # Note: --retry removed; launch failures auto-retry up to 10 times (hardcoded)
    auto_select = 'combined'  # Workflow mode defaults to combining files (like -c flag)
    template_file = None
    launcher_file = None
    unknown_template_token = None

    i = 0
    while i < len(args):
        arg = args[i]

        if arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
            _critical_set = True
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
            _skipped_set = True
        elif arg.startswith('--redo='):
            max_stage_redos = int(arg.split('=')[1])
        elif arg.startswith('--concurrent='):
            concurrent_jobs = max(1, int(arg.split('=')[1]))
            _concurrent_given = True
        elif arg.startswith('--auto-select='):
            auto_select = arg.split('=')[1]
        elif arg == '-a':
            # Auto-select all result_*.xyz files (excludes combined files)
            # Example: If you have result_443189.xyz, result_130389.xyz, result_536046.xyz
            # All 3 files will be processed separately
            auto_select = 'all'
        elif arg == '-c':
            # Auto-combine all result_*.xyz files first, then process the combined file
            # Example: If you have 3 result files, they'll be combined into combined_r3.xyz
            # Then only combined_r3.xyz will be processed
            auto_select = 'combined'
        elif arg.endswith(('.inp', '.com', '.gjf', '.xtb')) and template_file is None:
            template_file = arg
        elif arg.endswith('.sh') and launcher_file is None:
            launcher_file = arg
        elif not arg.startswith('-') and unknown_template_token is None:
            # Keep first unrecognized positional token for clearer diagnostics.
            unknown_template_token = arg
        
        i += 1

    # If --skipped given without explicit --critical, clear the default critical=0
    if _skipped_set and not _critical_set:
        max_critical = None

    # Allow template labels from embedded blocks (e.g., "opt input1").
    if not template_file and unknown_template_token:
        resolved_template = resolve_template_reference(context, unknown_template_token)
        if resolved_template:
            template_file = resolved_template

    if not template_file:
        if unknown_template_token:
            print(
                f"Error: No template file specified for optimization stage. "
                f"Got '{unknown_template_token}', expected .inp/.com/.gjf/.xtb or an embedded template label"
            )
        else:
            print("Error: No template file specified for optimization stage (.inp/.com/.gjf/.xtb or embedded label)")
        return 1

    # No --concurrent= in stage args ⇒ silent default of 1.

    context.max_tries = max_stage_redos  # For compatibility with existing code

    # Pin optimization output to a deterministic cosmic base folder for this cycle.
    # This prevents redo/resume runs from drifting to cosmic_3, cosmic_4, etc.
    optimization_cycle = getattr(context, 'optimization_stage_number', 1)
    if optimization_cycle <= 1:
        fixed_opt_cosmic_base = "cosmic"
    else:
        fixed_opt_cosmic_base = f"cosmic_{optimization_cycle}"

    context.cosmic_dir = fixed_opt_cosmic_base
    context.pending_cosmic_folder = fixed_opt_cosmic_base
    
    # Process redo structures at the START of the stage (if need_recalculation exists)
    # This ensures that when the workflow restarts this stage after a cosmic failure,
    # we immediately regenerate inputs and delete old outputs before checking completion
    optimization_dir_path = getattr(context, 'optimization_stage_dir', 'geometry_optimization')
    if not optimization_dir_path:  # Handle empty string
        optimization_dir_path = 'geometry_optimization'
    if os.path.exists(optimization_dir_path):
        redo_result = process_redo_structures(context, optimization_dir_path, template_file, concurrent_jobs=concurrent_jobs)

    
    # Check if we're resuming and optimization directory already exists
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    
    # Get the current stage key from context (e.g., "optimization_2" for first opt at position 2)
    stage_key = getattr(context, 'current_stage_key', '')
    
    # Get completed calculations from cache
    completed_calcs = cache.get('stages', {}).get(stage_key, {}).get('result', {}).get('completed_files', [])
    opt_dir_exists = os.path.exists("geometry_optimization") or os.path.exists("Geom Optimization")
    
    # If resuming (cache exists + optimization stage was started before), reuse existing directory  
    # This works even if no runs completed yet (e.g., interrupted during first opt)
    stage_was_started = stage_key in cache.get('stages', {})
    
    # Determine if this is a fresh start or resume
    optimization_dir_path = getattr(context, 'optimization_stage_dir', None)
    if opt_dir_exists and stage_was_started:
        # Use absolute path from context if available, otherwise use relative path
        if not optimization_dir_path:
            if os.path.exists("geometry_optimization"):
                optimization_dir_path = "geometry_optimization"
            elif os.path.exists("Geom Optimization"):
                optimization_dir_path = "Geom Optimization"
            else:
                optimization_dir_path = "geometry_optimization"
        
        # Check if redo structures exist (files scheduled for recalculation)
        redo_files = set()
        if hasattr(context, 'recalculated_files') and context.recalculated_files:
            redo_files = set(context.recalculated_files)
        
        # Scan ALL subdirectories for completed calculations (check for .out files, NOT .out.backup)
        # Files with only .out.backup are being redone and should NOT be counted as completed
        actual_completed = []
        
        # 1. Check optimization directory subfolders
        if os.path.exists(optimization_dir_path):
            for item in os.listdir(optimization_dir_path):
                item_path = os.path.join(optimization_dir_path, item)
                if os.path.isdir(item_path) and not item.startswith('_'):  # Skip _run_ directories
                    # Skip if this file is marked for redo
                    # Check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    if any(variant in redo_files for variant in redo_variants):
                        continue
                        
                    # Check if this subdirectory has a completed calculation (.out file, not just .backup)
                    out_file = os.path.join(item_path, f"{item}.out")
                    if os.path.exists(out_file):
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if check_qm_output_completed('auto', out_file):
                            actual_completed.append(item)
            
            # 1b. Check for flat files in optimization directory (e.g. opt_conf_1.out)
            # This handles the case where calculations are not in subdirectories
            for item in os.listdir(optimization_dir_path):
                if item.endswith('.out') or item.endswith('.log'):
                    basename = os.path.splitext(item)[0]
                    # Skip if already found in subdir or marked for redo
                    if basename not in actual_completed and basename not in redo_files:
                        out_file = os.path.join(optimization_dir_path, item)
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if item.endswith('.out'):
                            if check_qm_output_completed('auto', out_file):
                                actual_completed.append(basename)
                        else:  # Gaussian .log files
                            if check_qm_output_completed('gaussian', out_file):
                                actual_completed.append(basename)
        
        # 2. Check cosmic/orca_out_* folders (files moved there after sorting)
        # This is CRITICAL for showing correct counts (e.g. 11/11) when resuming
        cosmic_dir = getattr(context, 'cosmic_dir', 'cosmic')
        if os.path.exists(cosmic_dir) or os.path.exists('COSMIC'):  # Try fallbacks
            if not os.path.exists(cosmic_dir) and os.path.exists('COSMIC'):
                cosmic_dir = 'COSMIC'
            for item in os.listdir(cosmic_dir):
                if item.startswith('orca_out_') or item.startswith('gaussian_out_') or item.startswith('calc_out_'):
                    out_dir = os.path.join(cosmic_dir, item)
                    if os.path.isdir(out_dir):
                        files_in_subdir = os.listdir(out_dir)
                        for f in files_in_subdir:
                            if f.endswith('.out') or f.endswith('.log'):
                                basename = os.path.splitext(f)[0]
                                if basename not in actual_completed and basename not in redo_files:
                                    # Verify completion (OPI-aware for ORCA 6.1+)
                                    out_file = os.path.join(out_dir, f)
                                    if f.endswith('.out'):
                                        if check_qm_output_completed('auto', out_file):
                                            actual_completed.append(basename)
                                    else:  # Gaussian .log files
                                        if check_qm_output_completed('gaussian', out_file):
                                            actual_completed.append(basename)
        
        # Also check for cosmic_2, cosmic_3, etc.
        # This is needed if we have multiple cosmic stages
        parent_dir = os.getcwd()
        for item in os.listdir(parent_dir):
            if (item.startswith('cosmic_') or item.startswith('COSMIC_')) and os.path.isdir(item):
                cosmic_dir_n = item
                for subitem in os.listdir(cosmic_dir_n):
                    if subitem.startswith('orca_out_') or subitem.startswith('gaussian_out_') or subitem.startswith('calc_out_'):
                        out_dir = os.path.join(cosmic_dir_n, subitem)
                        if os.path.isdir(out_dir):
                            for f in os.listdir(out_dir):
                                if f.endswith('.out') or f.endswith('.log'):
                                    basename = os.path.splitext(f)[0]
                                    if basename not in actual_completed and basename not in redo_files:
                                        actual_completed.append(basename)
        
        # Update completed_calcs to match reality
        completed_calcs = actual_completed
        
        # CRITICAL: If we found NO completed files but input_files is very small,
        # this might be a redo scenario where all other files are already done
        # Count ALL subdirectories in calculation/ as potential completed files
        if len(completed_calcs) == 0 and opt_dir_exists:
            for item in os.listdir(optimization_dir_path):
                item_path = os.path.join(optimization_dir_path, item)
                if os.path.isdir(item_path) and not item.startswith('_'):
                    # Even if no .out file, if the subdirectory exists, a run was attempted
                    # Check for .out.backup (means it's being redone)
                    # Check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    is_redo = any(variant in redo_files for variant in redo_variants)
                    if not is_redo and item not in actual_completed:
                        # Count any subdirectory as a potential completed run for total count
                        backup_file = os.path.join(item_path, f"{item}.out.backup")
                        # Only count if it has either .out or .out.backup
                        out_file = os.path.join(item_path, f"{item}.out")
                        if os.path.exists(backup_file) or os.path.exists(out_file):
                            actual_completed.append(item)
            completed_calcs = actual_completed

        
        if not workflow_concise:
            if completed_calcs:
                print(f"\nResuming: Using existing optimization directory ({len(completed_calcs)} files already completed)\n")
            else:
                print(f"\nResuming: Using existing optimization directory\n")
        
        # Clean up ORCA intermediate files at root level (they should not be there)
        excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.']
        for item in os.listdir(optimization_dir_path):
            item_path = os.path.join(optimization_dir_path, item)
            if os.path.isfile(item_path) and any(pattern in item for pattern in excluded_patterns):
                try:
                    os.remove(item_path)
                except:
                    pass
        
        # Clean up any old input files at the root level (they're now in subdirectories after calculations run)
        old_root_inputs = (
            glob.glob(os.path.join(optimization_dir_path, "*.inp")) +
            glob.glob(os.path.join(optimization_dir_path, "*.com")) +
            glob.glob(os.path.join(optimization_dir_path, "*.gjf")) +
            glob.glob(os.path.join(optimization_dir_path, "*.xyz"))
        )
        for old_inp in old_root_inputs:
            # Skip ORCA intermediate files (already handled above)
            if any(pattern in old_inp for pattern in excluded_patterns):
                continue
                
            # Only remove if a corresponding subdirectory exists (calculation was run)
            basename = os.path.splitext(os.path.basename(old_inp))[0]
            
            # CRITICAL: Do NOT remove if this file was just regenerated by redo logic!
            if basename in redo_files:
                continue
                
            subdir = os.path.join(optimization_dir_path, basename)
            if os.path.isdir(subdir):
                os.remove(old_inp)
    else:
        # Handle launcher: if not provided, auto-detect ORCA and create launcher
        if launcher_file:
            if not os.path.exists(launcher_file):
                print(f"Warning: Launcher script not found: {launcher_file}")
                print("  Attempting to auto-detect ORCA installation...")
                launcher_file = None
        
        # Track existing optimization directories before running calculate_input_files
        existing_opt_dirs = set(d for d in os.listdir('.') if (d.startswith('geometry_optimization') or d.startswith('Geom Optimization')) and os.path.isdir(d))
        
        # Get QM alias from context (read from input.in line 9)
        qm_alias = getattr(context, 'qm_alias', 'orca')
        
        # Run calculation system creation with auto_select (in workflow mode)
        status = calculate_input_files(
            template_file,
            launcher_file,
            auto_select=auto_select,
            stage_type="optimization",
            workflow_mode=True,
            qm_alias=qm_alias,
            qm_nproc=getattr(context, 'qm_nproc', None),
            xtb_cycles=getattr(context, 'xtb_cycles', None),
            charge=getattr(context, 'charge', None),
            multiplicity=getattr(context, 'multiplicity', None),
        )
        # Check if calculate_input_files succeeded (returns string message)
        # Successfully created files contain "Created" in the message
        # All other returns are failures (Error:, No result_*.xyz, No XYZ files, etc.)
        if isinstance(status, str) and "Created" not in status:
            print(f"\n{status}")
            return 1
        
        # Find the optimization directory that was just created (may be optimization, optimization_2, etc.)
        current_opt_dirs = set(d for d in os.listdir('.') if (d.startswith('geometry_optimization') or d.startswith('Geom Optimization')) and os.path.isdir(d))
        new_opt_dirs = current_opt_dirs - existing_opt_dirs
        if new_opt_dirs:
            # Use the newly created directory
            optimization_dir_path = sorted(new_opt_dirs)[-1]  # Get the highest numbered one
        elif "geometry_optimization" in current_opt_dirs:
            optimization_dir_path = "geometry_optimization"
        elif "Geom Optimization" in current_opt_dirs:
            optimization_dir_path = "Geom Optimization"
        else:
            optimization_dir_path = None
        
    
    if optimization_dir_path and os.path.exists(optimization_dir_path):
        context.optimization_stage_dir = optimization_dir_path
        
        # Find and execute the launcher script
        launcher_script = os.path.join(optimization_dir_path, "launcher_orca.sh")
        if not os.path.exists(launcher_script):
            launcher_script = os.path.join(optimization_dir_path, "launcher_gaussian.sh")
        if not os.path.exists(launcher_script):
            launcher_script = os.path.join(optimization_dir_path, "launcher_xtb.sh")
        if not os.path.exists(launcher_script):
            launcher_script = os.path.join(optimization_dir_path, "launcher.sh")
        
        if os.path.exists(launcher_script):
            if not workflow_concise:
                print(f"\nExecuting calculations...\n")
            
            # Helper function to filter out ORCA intermediate files and rescue inputs
            def is_valid_input_file(filename):
                """Check if file is a valid input file (not an ORCA intermediate or rescue file)"""
                if not filename.endswith(('.inp', '.com', '.gjf', '.xyz')):
                    return False
                # Exclude ORCA intermediate files and rescue inputs (already processed)
                excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.', '_rescue.', 'combined_']
                return not any(pattern in filename for pattern in excluded_patterns)
            
            # Get list of input files to process
            # Check if files are at root level or in subdirectories (after sort command)
            input_files = sorted([f for f in os.listdir(optimization_dir_path) if is_valid_input_file(f)], key=natural_sort_key)
            
            if not input_files:
                # No input files at root - check if they're in subdirectories (sorted)
                # Look for subdirectories with input files
                for item in os.listdir(optimization_dir_path):
                    item_path = os.path.join(optimization_dir_path, item)
                    if os.path.isdir(item_path):
                        subdir_files = [f for f in os.listdir(item_path) if is_valid_input_file(f)]
                        if subdir_files:
                            # Found input files in subdirectory - add them with relative path
                            for f in subdir_files:
                                input_files.append(os.path.join(item, f))
                input_files = sorted(input_files, key=natural_sort_key)
            
            # Determine QM program from input files (check first file)
            if input_files:
                first_file = os.path.basename(input_files[0])
                if first_file.endswith('.inp'):
                    qm_program = 'orca'
                elif first_file.endswith(('.com', '.gjf')):
                    qm_program = 'gaussian'
                else:
                    qm_program = 'xtb'
            else:
                qm_program = 'orca'  # Default

            # Rebuild optimization launcher using the same style as refinement launcher.
            try:
                launcher_env_setup = ""
                existing_launcher = ""
                if os.path.exists(launcher_script):
                    with open(launcher_script, 'r') as lf:
                        existing_launcher = lf.read()
                    if '###' in existing_launcher:
                        launcher_env_setup = existing_launcher.split('###')[0].rstrip()
                    else:
                        # No ### separator - extract only the header/env lines (shebang, set, export),
                        # NOT the actual QM calculation commands.  If we took the whole file the
                        # batch commands (+ set -e) would be prepended to every individual job
                        # temp-script, causing all jobs to abort when the first one fails.
                        _env_lines = []
                        for _ln in existing_launcher.splitlines():
                            _s = _ln.strip()
                            if _s and (
                                _s.startswith('xtb ') or
                                _s.startswith('orca ') or
                                _s.startswith('g16 ') or
                                _s.startswith('$GAUSS') or
                                _s.startswith('ASCEC_XTB_RUNTIME') or
                                ' xtb ' in _s or
                                ' orca ' in _s
                            ):
                                break
                            _env_lines.append(_ln)
                        while _env_lines and not _env_lines[-1].strip():
                            _env_lines.pop()
                        launcher_env_setup = '\n'.join(_env_lines)

                launcher_inputs: List[str] = []
                seen_basenames = set()
                for root, _, files in os.walk(optimization_dir_path):
                    for file_name in files:
                        if not is_valid_input_file(file_name):
                            continue
                        if file_name not in seen_basenames:
                            seen_basenames.add(file_name)
                            launcher_inputs.append(file_name)
                launcher_inputs = sorted(launcher_inputs, key=natural_sort_key)

                if launcher_inputs:
                    # Resolve full ORCA path for the launcher
                    orca_exe_for_launcher = "orca"
                    xtb_options_for_launcher = parse_xtb_options_from_launcher(existing_launcher)
                    if qm_program == 'xtb':
                        xtb_options_for_launcher = build_xtb_runtime_options(
                            xtb_options_for_launcher,
                            getattr(context, 'qm_nproc', None),
                            getattr(context, 'xtb_cycles', None),
                        )
                    if qm_program == 'orca' and launcher_env_setup:
                        orca_exe_for_launcher = resolve_orca_executable_from_launcher(
                            launcher_env_setup, getattr(context, 'qm_alias', 'orca'))

                    with open(launcher_script, 'w') as lf:
                        if launcher_env_setup:
                            lf.write(launcher_env_setup + "\n\n")
                        if qm_program == 'xtb':
                            lf.write(f"export {_xtb_thread_env_prefix()}\n\n")
                        lf.write("###\n\n")
                        for i, inp_name in enumerate(launcher_inputs):
                            inp_base = os.path.splitext(inp_name)[0]
                            if qm_program == 'orca':
                                cmd = f"{orca_exe_for_launcher} {inp_base}.inp > {inp_base}.out"
                            elif qm_program == 'xtb':
                                cmd = f"{_xtb_thread_env_prefix()} xtb {inp_base}.xyz {xtb_options_for_launcher} --namespace {inp_base} > {inp_base}.out 2>&1"
                            else:
                                cmd = f"g16 {inp_base}.com"
                            if i < len(launcher_inputs) - 1:
                                lf.write(f"{cmd} ; \\\n")
                            else:
                                lf.write(f"{cmd}\n")
                    os.chmod(launcher_script, 0o755)
            except Exception:
                pass
            
            # Get exclusions from cache (already loaded earlier)
            # Use appropriate key based on which optimization stage this is
            optimization_stage_num = getattr(context, 'optimization_stage_number', 1)
            if optimization_stage_num == 1:
                excluded_numbers = cache.get('excluded_optimizations', [])
            else:
                excluded_numbers = cache.get('excluded_optimizations_2', [])
            
            # Apply exclusion filtering, then collapse every scan spelling to one
            # canonical structure key so a calculation discovered under both
            # 'opt_conf_1' and 'opt_conf_1.inp' (or its redo/rescue variant) is
            # only counted once and the total never grows across redo passes.
            completed_calcs = [f for f in completed_calcs if not match_exclusion(f, excluded_numbers)]
            completed_calcs = list(dict.fromkeys(_canonical_calc_key(f) for f in completed_calcs))

            # Count total inputs (including those already completed and cached)
            # This should be the TOTAL expected calculations, not just new ones,
            # keyed by canonical structure id.
            all_input_basenames = set()

            # Add keys from input files (pending calculations)
            for f in input_files:
                if not match_exclusion(f, excluded_numbers):
                    # Handle both simple filenames and paths like "opt_conf_1/opt_conf_1.inp"
                    all_input_basenames.add(_canonical_calc_key(f))

            # Add keys from already completed files (CRITICAL for correct count)
            for f in completed_calcs:
                all_input_basenames.add(_canonical_calc_key(f))

            num_inputs = len(all_input_basenames)

            # Lock the maximum across redo attempts so the live N/M display does
            # not grow when redos rerun a subset of structures. Redo should drop
            # the completed count back and rebuild up to the same maximum.
            _opt_stage_key = getattr(context, 'current_stage_key', '')
            if _opt_stage_key:
                _locked_totals = getattr(context, '_opt_locked_totals', None)
                if _locked_totals is None:
                    _locked_totals = {}
                    context._opt_locked_totals = _locked_totals
                _prev_locked = _locked_totals.get(_opt_stage_key, 0)
                if _prev_locked > num_inputs:
                    num_inputs = _prev_locked
                else:
                    _locked_totals[_opt_stage_key] = num_inputs

            # Store initial completed count (before loop may modify completed_calcs)
            initial_completed_count = len(completed_calcs)
            if initial_completed_count > num_inputs:
                initial_completed_count = num_inputs

            progress_cb = context.update_progress
            if workflow_concise and callable(progress_cb):
                progress_cb(f"{initial_completed_count}/{num_inputs} ...")
            
            if completed_calcs and not workflow_concise:
                print(f"Resuming: {len(completed_calcs)}/{num_inputs} calculations already completed")
            
            if excluded_numbers and not workflow_concise:
                print(f"Exclusions active: {excluded_numbers}")
                print()
            
            # Read launcher script to get environment setup
            with open(launcher_script, 'r') as f:
                launcher_content = f.read()
            
            try:
                # Make script executable (for manual use later)
                os.chmod(launcher_script, 0o755)

                # Resolve ORCA executable path for temp scripts
                orca_exe = 'orca'
                if qm_program == 'orca':
                    orca_exe = resolve_orca_executable_from_launcher(
                        launcher_content, getattr(context, 'qm_alias', 'orca'))

                # Build list of pending jobs (skip excluded and already completed)
                max_launch_retries = 10
                launch_failure_threshold = 5.0
                pending_jobs = []

                for input_file in input_files:
                    if match_exclusion(input_file, excluded_numbers):
                        if not workflow_concise:
                            print(f"  Skipping: {input_file} (excluded)")
                        continue

                    basename = os.path.splitext(input_file)[0]
                    output_file = basename + ('.out' if qm_program in ('orca', 'xtb') else '.log')
                    output_path = os.path.join(optimization_dir_path, output_file)

                    # Skip if output already exists AND is successfully completed
                    if os.path.exists(output_path):
                        try:
                            is_complete = check_qm_output_completed(qm_program, output_path)
                            if is_complete:
                                continue
                        except Exception:
                            pass

                    pending_jobs.append({
                        'input_file': input_file,
                        'base_dir': optimization_dir_path,
                        'launcher_content': launcher_content,
                        'qm_program': qm_program,
                        'max_launch_retries': max_launch_retries,
                        'launch_failure_threshold': launch_failure_threshold,
                        'orca_exe': orca_exe,
                        'xtb_options': build_xtb_runtime_options(
                            parse_xtb_options_from_launcher(launcher_content),
                            getattr(context, 'qm_nproc', None),
                            getattr(context, 'xtb_cycles', None),
                        ) if qm_program == 'xtb' else None,
                    })

                # Run calculations with concurrency
                _opt_wall_start = time.time()
                num_completed, num_failed, failed_calculations = _run_qm_calculations_with_concurrency(
                    pending_jobs=pending_jobs,
                    concurrent_jobs=concurrent_jobs,
                    workflow_concise=workflow_concise,
                    context=context,
                    initial_completed_count=initial_completed_count,
                    num_inputs=num_inputs,
                    completed_list=completed_calcs,
                    all_input_basenames=all_input_basenames,
                    cache_file=cache_file,
                    stage_key_prefix='calculation',
                )
                # Accumulate active wall across resume sessions so the summary's
                # "Total wall time" survives interruptions (never < longest job).
                context.optimization_job_wall_time = accumulate_stage_wall_time(
                    cache_file, _opt_stage_key, time.time() - _opt_wall_start)
                
                # Print status (not "results" - that's redundant)
                # Recalculate num_inputs from the updated set to reflect newly completed calculations
                num_inputs = len(all_input_basenames)
                # Keep the redo-locked maximum so subsequent redo attempts do not inflate the total.
                if _opt_stage_key:
                    _locked_totals = getattr(context, '_opt_locked_totals', {})
                    _prev_locked = _locked_totals.get(_opt_stage_key, 0)
                    if _prev_locked > num_inputs:
                        num_inputs = _prev_locked
                    else:
                        _locked_totals[_opt_stage_key] = num_inputs
                # Total completed = initial completed + newly completed in this run
                total_completed = initial_completed_count + num_completed
                # Ensure we don't show more completed than total inputs (can happen if files are removed)
                if total_completed > num_inputs:
                    total_completed = num_inputs
                if not workflow_concise:
                    print(f"\nStatus: {total_completed}/{num_inputs} calculations completed")

                # Store for protocol summary (use total)
                context.optimization_completed = total_completed
                context.optimization_total = num_inputs
                
                # Handle failed calculations
                if failed_calculations:
                    if not workflow_concise:
                        print(f"Failed calculations: {len(failed_calculations)}/{num_inputs}")
                
                # Use total_completed for status checks (includes resumed + new)
                if total_completed == 0:
                    print("Error: No calculations completed successfully")
                    return 1
                # All calculations completed
                if total_completed == num_inputs:
                    if not workflow_concise:
                        print(f"All calculations completed successfully")
                elif total_completed < num_inputs:
                    # Continue - don't stop workflow, cosmic will handle quality control
                    pass # Status line already printed above
                
                # Sort output files by energy (only in workflow mode)
                # Run organize step if any calculations completed (new or already done)
                if total_completed > 0 and context.is_workflow:
                    saved_cwd = os.getcwd()
                    try:
                        # Pin this stage to a deterministic cosmic base folder so redo attempts
                        # and resumes do not create extra folders (cosmic_3, cosmic_4, ...).
                        # cosmic folder lives inside the project directory (cwd), not its parent.
                        project_dir = os.getcwd()

                        target_cosmic_base = fixed_opt_cosmic_base

                        already_organized = os.path.exists(os.path.join(project_dir, target_cosmic_base))

                        if already_organized:
                            if not workflow_concise:
                                print("\n✓ Files already organized (resuming from cache)")
                            # Reuse this optimization cycle's fixed cosmic folder.
                            context.optimization_cosmic_folder = target_cosmic_base
                            context.pending_cosmic_folder = target_cosmic_base
                        else:
                            # Merge good structures back if they exist (from retry)
                            if os.path.exists("good_structures"):
                                for folder in os.listdir("good_structures"):
                                    src_folder = os.path.join("good_structures", folder)
                                    if os.path.isdir(src_folder):
                                        dest_folder = os.path.join(optimization_dir_path, folder)
                                        if not os.path.exists(dest_folder):
                                            shutil.copytree(src_folder, dest_folder)
                                # Clean up good_structures folder
                                shutil.rmtree("good_structures")
                            
                            os.chdir(optimization_dir_path)

                            # Group files by base names into subfolders (silent in workflow)
                            import io
                            import contextlib
                            f = io.StringIO()
                            _actual_wall = getattr(context, 'optimization_job_wall_time', None)
                            with contextlib.redirect_stdout(f):
                                group_files_by_base_with_tracking(".")
                                combine_xyz_files()
                                create_combined_mol()
                                create_summary_with_tracking(".", actual_wall_time=_actual_wall)

                                # Reuse stage folder for redo/resume and write into fixed target folder.
                                is_redo = hasattr(context, 'recalculated_files') and bool(context.recalculated_files)
                                reuse_folder = is_redo or os.path.exists(os.path.join(project_dir, target_cosmic_base))
                                cosmic_folder = collect_out_files_with_tracking(
                                    reuse_existing=reuse_folder,
                                    target_cosmic_folder=target_cosmic_base
                                )

                            # Extract key info from output
                            output = f.getvalue()
                            # Surface any errors that were captured during redirect
                            if 'Error' in output or 'Traceback' in output:
                                import sys as _sys
                                for _line in output.split('\n'):
                                    if 'Error' in _line or 'Traceback' in _line or _line.startswith('  File '):
                                        print(_line, file=_sys.stderr)
                            if context.workflow_verbose_level >= 1:
                                if 'Summary written to' in output:
                                    print("\nSummary file(s) generated")
                            if cosmic_folder:
                                context.optimization_cosmic_folder = cosmic_folder
                                cosmic_base = _cosmic_base_name(cosmic_folder)
                                context.pending_cosmic_folder = cosmic_base
                            # Look for cosmic folder reference
                            if 'Copied' in output and ('cosmic' in output.lower() or 'COSMIC' in output):
                                for line in output.split('\n'):
                                    if 'Copied' in line and '.out files to' in line:
                                        if context.workflow_verbose_level >= 1:
                                            print(line)
                                        # Extract cosmic folder name (e.g., "cosmic/orca_out_3" or "COSMIC/orca_out_3")
                                        import re
                                        match = re.search(r'to\s+([cC][oO][sS][mM][iI][cC][^\s]*)', line)
                                        if match:
                                            cosmic_folder = match.group(1)
                                            context.optimization_cosmic_folder = cosmic_folder
                                            # Also set as pending for next cosmic stage
                                            cosmic_base = _cosmic_base_name(cosmic_folder)
                                            context.pending_cosmic_folder = cosmic_base
                                        break

                            if context.workflow_verbose_level >= 1:
                                print(f"\n✓ Files organized and sorted")
                            
                    except Exception as e:
                        print(f"⚠ Warning: Could not organize files: {e}")
                    finally:
                        os.chdir(saved_cwd)

                    # Parse total CPU time from orca_summary.txt for protocol summary mean exec time.
                    # Use saved_cwd to build an absolute path so cwd changes don't break resolution.
                    _orca_sum_path = os.path.join(saved_cwd, optimization_dir_path, "orca_summary.txt")
                    if not os.path.exists(_orca_sum_path):
                        # Fallback: try relative to current cwd
                        _orca_sum_path = os.path.join(optimization_dir_path, "orca_summary.txt")
                    if os.path.exists(_orca_sum_path):
                        try:
                            with open(_orca_sum_path, 'r') as _sf:
                                _sc = _sf.read()
                            _tm = __import__('re').search(r'Total execution time:\s+(\d+):(\d+):(\d+\.\d+)', _sc)
                            if _tm:
                                context.optimization_total_cpu_time = (
                                    int(_tm.group(1)) * 3600 + int(_tm.group(2)) * 60 + float(_tm.group(3))
                                )
                        except Exception:
                            pass

            except Exception as e:
                print(f"✗ Error running calculations: {e}")
                return 1
        else:
            print("✗ Error: No launcher script found, cannot execute calculations")
            return 1
            
    else:
        print(f"Warning: geometry_optimization directory not found")
        return 1
    
    # Clean up retry_input folder if it exists
    if os.path.exists("retry_input"):
        shutil.rmtree("retry_input")
    
    return 0


def execute_cosmic_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """
    Execute cosmic analysis stage.
    Runs from the cosmic/ folder which contains orca_out_N/ subfolders.
    Supports both cosmic/ (first run) and cosmic_2/ (after optimization).
    """
    def _count_latest_cosmic_representatives(cosmic_dir: str) -> Tuple[Optional[int], Optional[int]]:
        """Return counts for latest motifs_* and umotifs_* representative xyz files."""
        try:
            motif_dirs = sorted(glob.glob(os.path.join(cosmic_dir, "motifs_*")), key=natural_sort_key)
            umotif_dirs = sorted(glob.glob(os.path.join(cosmic_dir, "umotifs_*")), key=natural_sort_key)

            motif_count: Optional[int] = None
            umotif_count: Optional[int] = None

            if motif_dirs:
                latest_motif_dir = motif_dirs[-1]
                motif_count = len(glob.glob(os.path.join(latest_motif_dir, "motif_*.xyz")))

            if umotif_dirs:
                latest_umotif_dir = umotif_dirs[-1]
                umotif_count = len(glob.glob(os.path.join(latest_umotif_dir, "umotif_*.xyz")))

            return motif_count, umotif_count
        except Exception:
            return None, None

    update_list_file: Optional[str] = None

    # Determine which cosmic folder to use dynamically based on the most recent stage
    # Priority: use the most recently set cosmic folder (from optimization/refinement that just ran)
    cosmic_base = None
    
    # Check if there's a pending_cosmic_folder (set by optimization/refinement organize step)
    if hasattr(context, 'pending_cosmic_folder') and context.pending_cosmic_folder:
        pending_cosmic = context.pending_cosmic_folder
        pending_base = _cosmic_base_name(pending_cosmic)
        if os.path.isdir(pending_base):
            cosmic_base = pending_base
        elif getattr(context, 'workflow_verbose_level', 0) >= 1:
            print(f"Warning: Pending cosmic folder '{pending_base}' not found. Using fallback selection.")
        # Clear it after checking so stale values don't affect later stages
        context.pending_cosmic_folder = None

    # Fallback: check what optimization_cosmic_folder, refinement_cosmic_folder, or eref_cosmic_folder were set
    if not cosmic_base:
        # If eref_cosmic_folder is set (from energy refinement stage), prefer it
        if hasattr(context, 'eref_cosmic_folder') and context.eref_cosmic_folder:
            eref_base = _cosmic_base_name(context.eref_cosmic_folder)
            if os.path.exists(eref_base):
                cosmic_base = eref_base

        # If refinement_cosmic_folder is more recent (set after optimization_cosmic_folder), prefer it
        if not cosmic_base and hasattr(context, 'refinement_cosmic_folder') and context.refinement_cosmic_folder:
            opt_base = _cosmic_base_name(context.refinement_cosmic_folder)
            if os.path.exists(opt_base):
                cosmic_base = opt_base

        # Otherwise use optimization_cosmic_folder
        if not cosmic_base and hasattr(context, 'optimization_cosmic_folder') and context.optimization_cosmic_folder:
            calc_base = _cosmic_base_name(context.optimization_cosmic_folder)
            if os.path.exists(calc_base):
                cosmic_base = calc_base
    
    # Final fallback: pick the latest cosmic folder if present.
    if not cosmic_base:
        cosmic_candidates = [
            d for d in os.listdir('.')
            if (d.lower().startswith('cosmic') and d[0] in 'cC') and os.path.isdir(d)
        ]
        if cosmic_candidates:
            def _cosmic_sort_key(name: str) -> int:
                if name.lower() == 'cosmic':
                    return 1
                match = re.search(r'^[cC][oO][sS][mM][iI][cC]_(\d+)$', name)
                return int(match.group(1)) if match else 0

            cosmic_base = sorted(cosmic_candidates, key=_cosmic_sort_key)[-1]
        else:
            print("Warning: cosmic folder not found")
            if getattr(context, 'is_workflow', False):
                return 1
            return 0

    # Final guard in case a stale/non-existent folder name slipped through selection logic.
    if not os.path.isdir(cosmic_base):
        cosmic_candidates = [
            d for d in os.listdir('.')
            if (d.lower().startswith('cosmic') and d[0] in 'cC') and os.path.isdir(d)
        ]
        if cosmic_candidates:
            def _cosmic_sort_key(name: str) -> int:
                if name.lower() == 'cosmic':
                    return 1
                match = re.search(r'^[cC][oO][sS][mM][iI][cC]_(\d+)$', name)
                return int(match.group(1)) if match else 0

            cosmic_base = sorted(cosmic_candidates, key=_cosmic_sort_key)[-1]
        else:
            print(f"Warning: COSMIC folder not found ({cosmic_base})")
            if getattr(context, 'is_workflow', False):
                return 1
            return 0
    
    # Store cosmic folder for protocol summary
    context.cosmic_folder = cosmic_base
    # CRITICAL: Also update cosmic_dir so redo logic uses correct folder
    context.cosmic_dir = cosmic_base
    
    # Clean old cosmic results before re-running (but keep orca_out and cache)
    # This is CRITICAL to prevent reusing stale skipped_structures from previous runs
    # Check if this is a re-run by looking for existing clustering results
    has_old_results = (
        os.path.exists(os.path.join(cosmic_base, "clustering_summary.txt")) or
        os.path.exists(os.path.join(cosmic_base, "skipped_structures"))
    )
    
    if has_old_results:
        # Clean everything except orca_out_* folders and cache
        items_to_remove = [
            'dendrogram_images', 'extracted_clusters', 'extracted_data',
            'skipped_structures', 'clustering_summary.txt', 'boltzmann_distribution.txt',
            'resolved_threshold.txt',
        ]
        # Also remove motifs and umotifs folders
        for item in os.listdir(cosmic_base):
            if item.startswith('motifs_') or item.startswith('umotifs_'):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            item_path = os.path.join(cosmic_base, item)
            if os.path.exists(item_path):
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
    
    # Verify output subfolder exists and count input structures.
    out_candidates = []
    for item in sorted(os.listdir(cosmic_base)):
        if item.startswith("orca_out_") or item.startswith("opt_out_") or item.startswith("gaussian_out_") or item.startswith("calc_out_") or item.startswith("xtb_out_"):
            out_candidates.append(item)

    out_folder_found = len(out_candidates) > 0
    selected_out_name: Optional[str] = None
    cosmic_input_count = 0
    if out_folder_found:
        def _extract_out_count(folder_name: str) -> Optional[int]:
            match = re.search(r'_(\d+)$', folder_name)
            return int(match.group(1)) if match else None

        def _out_type_rank(folder_name: str) -> int:
            lower = folder_name.lower()
            if lower.startswith('calc_out_'):
                return 0
            if lower.startswith('orca_out_'):
                return 1
            if lower.startswith('gaussian_out_'):
                return 2
            if lower.startswith('xtb_out_'):
                return 3
            if lower.startswith('opt_out_'):
                return 4
            return 9

        # Prefer the explicit canonical leaf folder set by the most recent
        # optimization/refinement/eref sort step. This avoids count-matching
        # ambiguity when stale sibling folders exist.
        explicit_leaf: Optional[str] = None
        for attr_name in ('eref_cosmic_folder', 'refinement_cosmic_folder', 'optimization_cosmic_folder'):
            attr_val = getattr(context, attr_name, None)
            if isinstance(attr_val, str) and attr_val:
                leaf = os.path.basename(attr_val.rstrip('/'))
                if leaf in out_candidates:
                    explicit_leaf = leaf
                    break

        selected_out = out_candidates[0]
        if explicit_leaf is not None:
            selected_out = explicit_leaf
        elif len(out_candidates) > 1:
            expected_count: Optional[int] = None
            for attr_name in ('eref_completed', 'refinement_completed', 'optimization_completed'):
                attr_val = getattr(context, attr_name, None)
                if isinstance(attr_val, int) and attr_val > 0:
                    expected_count = attr_val
                    break

            if expected_count is not None:
                exact_matches = [
                    c for c in out_candidates
                    if _extract_out_count(c) == expected_count
                ]
                if exact_matches:
                    selected_out = sorted(exact_matches, key=lambda c: (_out_type_rank(c), c))[0]
                else:
                    selected_out = sorted(
                        out_candidates,
                        key=lambda c: (
                            abs((_extract_out_count(c) or 10**9) - expected_count),
                            _out_type_rank(c),
                            -((_extract_out_count(c) or -1)),
                            c,
                        ),
                    )[0]
            else:
                selected_out = sorted(
                    out_candidates,
                    key=lambda c: (
                        -((_extract_out_count(c) or -1)),
                        _out_type_rank(c),
                        c,
                    ),
                )[0]

            if getattr(context, 'workflow_verbose_level', 0) >= 1:
                print(
                    f"Warning: Multiple output folders found in {cosmic_base}/: {', '.join(out_candidates)}. "
                    f"Using '{selected_out}' deterministically."
                )

        out_dir = os.path.join(cosmic_base, selected_out)
        selected_out_name = selected_out
        cosmic_input_count = len([
            f for f in glob.glob(os.path.join(out_dir, "*.out"))
                     + glob.glob(os.path.join(out_dir, "*.log"))
            if not is_qm_byproduct(f)
        ])
    
    if not out_folder_found:
        print(f"Warning: No orca_out_*, gaussian_out_*, calc_out_*, xtb_out_*, or opt_out_* folder found in {cosmic_base}/")
        if getattr(context, 'is_workflow', False):
            return 1
        return 0
    
    args = stage['args']
    
    # Find cosmic script
    cosmic_script = None
    other_args = []
    
    for arg in args:
        if arg.endswith('.py') and 'cosmic' in arg.lower():
            cosmic_script = arg
        else:
            other_args.append(arg)
    
    if not cosmic_script:
        cosmic_script = find_cosmic_script()
        if not cosmic_script:
            print("Warning: cosmic-v01.py script not found")
            if getattr(context, 'is_workflow', False):
                return 1
            return 0  # Not an error
    
    # Default: concise output (no extra printing)
    # In workflow mode, always concise. Could add --verbose flag to stage args later if needed.
    verbose = False
    
    if verbose:
        print(f"\nRunning cosmic analysis...")
        print(f"Using cosmic script: {os.path.basename(cosmic_script)}")
    
    # Build command. When a deterministic output subfolder is known, pass it explicitly
    # to bypass cosmic-v01 interactive folder selection.
    if selected_out_name:
        cmd = [sys.executable, cosmic_script, selected_out_name] + other_args
    else:
        cmd = [sys.executable, cosmic_script] + other_args
    
    # Add --cores if not already specified and user explicitly set ascec_parallel_cores in input file
    # (ascec_parallel_cores > 0 means it was explicitly set)
    has_cores_arg = any(arg.startswith('--cores') or arg.startswith('-j') for arg in other_args)
    if not has_cores_arg and hasattr(context, 'ascec_parallel_cores') and context.ascec_parallel_cores > 0:
        cmd.extend(['--cores', str(context.ascec_parallel_cores)])

    # If user provides --th/--threshold, pass it through; otherwise
    # cosmic uses statistical consensus cutting (no threshold needed).
    
    # No need to specify motif prefix - cosmic script auto-detects from filenames:
    # conf_* files → creates motifs_*/ folder (after calculation)
    # motif_* files → creates motifs_*/ folder (first cosmic)
    # umotif_* files → creates umotifs_*/ folder (after optimization)
    
    # If this is a redo and we have a list of recalculated files, pass them for incremental update
    if hasattr(context, 'recalculated_files') and context.recalculated_files:
        # Create temp file with list of files to update
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            update_list_file = f.name
            for basename in context.recalculated_files:
                f.write(f"{basename}\n")
        cmd.extend(['--update-cache', update_list_file])
    
    # If previous stage was eref, pass --prev-out-dir for composite energy calculation.
    # eref_motifs_source is the COSMIC base folder that eref took motifs from; it also
    # contains the orca_out_*/ with the ref-level G and E_elec needed for thermal corrections.
    eref_source = getattr(context, 'eref_motifs_source', None)
    if eref_source:
        # Strip to cosmic base only (in case eref_motifs_source includes subpath like cosmic_2/orca_out_5)
        eref_source = _cosmic_base_name(eref_source)
        if os.path.isdir(eref_source):
            cmd.extend(['--prev-out-dir', os.path.abspath(eref_source)])
        context.eref_motifs_source = None  # consume after use

    if verbose:
        print(f"{' '.join(cmd)}\n")
        print(f"Working directory: {cosmic_base}\n")
    try:
        # Stream output to avoid pipe buffer deadlock on large outputs.
        # Keep stdin open only for fallback interactive mode when no explicit folder was passed.
        use_stdin = selected_out_name is None
        proc = subprocess.Popen(cmd,
                               stdin=(subprocess.PIPE if use_stdin else subprocess.DEVNULL),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               bufsize=1,
                               cwd=cosmic_base,
                               universal_newlines=True,
                               preexec_fn=_PDEATHSIG_PREEXEC)

        # Fallback auto-selection for interactive mode only.
        if use_stdin and proc.stdin:
            proc.stdin.write('1\n')
            proc.stdin.close()
        
        # Filter patterns for output
        skip_lines = [
            'Found the following folder(s) containing quantum chemistry',
            'Enter the number of the folder to process',
            'Only .log files found',
            'Only .out files found',
            'Processing ',
            'folder(s) for files matching',
            'Created motifs dendrogram',
            'COSMIC_OPT_ONLY_MODE'
        ]
        
        # Lines that should have blank line BEFORE them
        add_blank_before = [
            'H-bond group',
            'Processed ',
            'Creating ',
            ',   Critical:'  # Summary line "Redundant: X,   Critical: Y"
        ]
        
        # Lines that should have blank line AFTER them
        add_blank_after = [
            'Data extraction complete. Proceeding to clustering.',
            'Motifs created:',
            ',   Critical:',  # Summary line "Redundant: X,   Critical: Y"
            'Clustering summary saved to'
        ]
        
        # Stream output line by line in real-time and collect for post-processing
        # Only print in verbose mode
        if verbose:
            print()
        stdout_lines = []
        prev_line = ""
        
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line = line.rstrip('\n')
                stdout_lines.append(line)
                
                # Capture processing folder line
                if 'Processing folder:' in line:
                    match = re.search(r'Processing folder:\s+(\S+)', line)
                    if match:
                        folder_name = match.group(1)
                        context.cosmic_folder = f"{cosmic_base}/{folder_name}"

                # Skip printing in non-verbose mode (but still collect lines)
                if not verbose:
                    continue
                # Skip interactive prompts and folder listing
                if any(skip in line for skip in skip_lines):
                    continue
                
                # Skip folder listing lines
                if re.match(r'\s*\[\d+\]\s+\S+', line):
                    continue
                
                # Skip blank lines
                if not line.strip():
                    continue
                
                # Add blank line before certain sections
                if any(phrase in line for phrase in add_blank_before):
                    if prev_line.strip():  # Only if previous wasn't blank
                        print()
                
                print(line)
                
                # Add blank line after certain sections
                if any(phrase in line for phrase in add_blank_after):
                    print()
                
                prev_line = line
            
            proc.stdout.close()
        
        proc.wait()
        
        stdout = '\n'.join(stdout_lines)
        
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, '')
        
        # Parse collected output for motifs count (already printed in real-time above)
        for line in stdout_lines:
            # Capture motifs created count for protocol summary
            if 'Motifs created:' in line and 'representatives' in line:
                match = re.search(r'(\d+)\s+representatives', line)
                if match:
                    context.cosmic_motifs_created = int(match.group(1))

        # Persist latest motif/umotif counts for compact workflow progress display.
        motif_count, umotif_count = _count_latest_cosmic_representatives(cosmic_base)
        context.last_cosmic_motif_count = motif_count
        context.last_cosmic_umotif_count = umotif_count
        context.last_cosmic_input_count = cosmic_input_count if cosmic_input_count > 0 else None
        stage_total = (motif_count or 0) + (umotif_count or 0)
        if stage_total > 0:
            # Keep summary count stage-local to avoid carrying over previous cosmic values.
            context.cosmic_motifs_created = stage_total
        def _record_cosmic_input_count(stage_num: int, raw_count: int) -> None:
            """Persist the cosmic input-structure count under a redo-locked ceiling.

            The first (full) cosmic run for a stage establishes the maximum input
            population. Redo attempts re-cluster a reduced subset and re-count the
            output folder, which can momentarily report fewer OR (when stray/duplicate
            outputs linger) inflate the total above the genuine population. We lock the
            first value as a hard maximum so redo can never inflate it and the display
            always returns to that max once the redo subset is merged back in.
            """
            if raw_count <= 0:
                return
            locked = context.cosmic_locked_input_counts.get(stage_num)
            if locked is None:
                # On a resumed session the in-memory lock is empty even though a
                # genuine full-run count was already persisted to the cache. Seed
                # the lock from the cached value so a redo-only resume cannot
                # re-establish (and inflate) the maximum.
                cached_max = _cached_cosmic_input_count(context, stage_num)
                locked = cached_max if cached_max and cached_max > 0 else raw_count
                context.cosmic_locked_input_counts[stage_num] = locked
                context.cosmic_stage_input_counts[stage_num] = locked
            else:
                # Redo run: keep the established maximum (ignore inflated/reduced recount).
                context.cosmic_stage_input_counts[stage_num] = locked

        stage_key = getattr(context, 'current_stage_key', '')
        match = re.search(r'^cosmic_(\d+)$', stage_key)
        if match:
            stage_num = int(match.group(1))
            context.cosmic_stage_counts[stage_num] = stage_total
            _record_cosmic_input_count(stage_num, cosmic_input_count)
        else:
            # Combined mode runs cosmic while current_stage_key is still optimization/refinement.
            prev_match = re.search(r'^(optimization|refinement)_(\d+)$', stage_key)
            if prev_match:
                stage_num = int(prev_match.group(2)) + 1
                context.cosmic_stage_counts[stage_num] = stage_total
                _record_cosmic_input_count(stage_num, cosmic_input_count)
        
        if verbose:
            print("\n✓ cosmic analysis completed")
        
        # Check if files were saved to need_recalculation directory
        # This happens when structures with imaginary frequencies need to be recalculated
        need_recalc_dir = os.path.join(cosmic_base, "skipped_structures", "need_recalculation")
        clustered_with_minima_dir = os.path.join(cosmic_base, "skipped_structures", "clustered_with_minima")
        critical_non_conv_dir = os.path.join(cosmic_base, "skipped_structures", "critical_non_converged")
        
        recalc_basenames = []
        
        # Always include critical structures (need_recalculation - imaginary frequencies)
        if os.path.exists(need_recalc_dir):
            xyz_files = glob.glob(os.path.join(need_recalc_dir, "*.xyz"))
            if xyz_files:
                # Extract basenames (without .xyz extension)
                recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        # Always include critical non-converged structures (need rescue hessian)
        if os.path.exists(critical_non_conv_dir):
            xyz_files = glob.glob(os.path.join(critical_non_conv_dir, "*.xyz"))
            if xyz_files:
                recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        # If user wants all structures to be true minima (--skipped=0), also include clustered_with_minima
        # Check if this is optimization stage with --skipped=0 threshold
        if os.path.exists(clustered_with_minima_dir):
            # Check if we're in an optimization context with skipped threshold = 0
            # This would be stored in context from the stage arguments
            include_clustered = False
            
            # For optimization stage, check if skipped threshold is 0
            current_stage = getattr(context, 'current_stage', None)
            if current_stage:
                if current_stage.get('type') == 'optimization':  # type: ignore[union-attr]
                    args = current_stage.get('args', [])  # type: ignore[union-attr]
                    for arg in args:
                        if arg.startswith('--skipped='):
                            skipped_val = float(arg.split('=')[1])
                            # User wants 0% skipped (all structures to be true minima)
                            if skipped_val <= 0.0:
                                include_clustered = True
                                break
            
            if include_clustered:
                xyz_files = glob.glob(os.path.join(clustered_with_minima_dir, "*.xyz"))
                if xyz_files:
                    recalc_basenames.extend([os.path.splitext(os.path.basename(f))[0] for f in xyz_files])
        
        if recalc_basenames:
            # Store in context for the optimization stage to use
            context.recalculated_files = recalc_basenames
        
        # Store cosmic directory for context
        context.cosmic_dir = cosmic_base
        
        # Cleanup temp file if created (but DON'T clear recalculated_files - it's needed for the redo loop!)
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
                # NOTE: Do NOT clear context.recalculated_files here!
                # The optimization stage redo needs this list to know which files to regenerate.
            except Exception:
                pass
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error: cosmic analysis failed with code {e.returncode}")
        # stderr was merged into stdout (subprocess.STDOUT), so the captured
        # cosmic output — including any argparse "usage:/error:" lines that
        # caused an exit-2 — lives on e.output. Print the tail so the failure
        # reason is visible even when the workflow ran cosmic in non-verbose
        # mode.
        captured = e.output or e.stderr
        if captured:
            tail = captured.splitlines()[-30:]
            print("--- cosmic output (tail) ---")
            for tail_line in tail:
                print(tail_line)
            print("--- end cosmic output ---")
        # Cleanup temp file if it exists
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
            except Exception:
                pass
        return e.returncode
    except Exception as e:
        if update_list_file and os.path.exists(update_list_file):
            try:
                os.remove(update_list_file)
            except Exception:
                pass
        print(f"Error running cosmic analysis: {e}")
        return 1


def execute_refinement_stage(context: WorkflowContext, stage: Dict[str, Any], _stage_kind: str = 'refinement') -> int:
    """Execute refinement stage (for motifs from cosmic clustering).

    Args:
        _stage_kind: 'refinement' (geometry_refinement dir, context.refinement_*) or
                     'energy_refinement' (energy_refinement dir, context.eref_*)
    """
    # Very similar to optimization stage, but uses motifs from cosmic analysis

    # Store context globally for helper functions
    sys._current_workflow_context = context  # type: ignore[attr-defined]

    # --- Stage-kind specific configuration ---
    if _stage_kind == 'energy_refinement':
        _opt_dir_name = 'energy_refinement'
        _attr_stage_dir = 'energy_refinement_stage_dir'
        _attr_cosmic_folder = 'eref_cosmic_folder'
        _attr_motifs_source = 'eref_motifs_source'
        _attr_completed = 'eref_completed'
        _attr_total = 'eref_total'
        _attr_job_wall_time = 'eref_job_wall_time'
        _attr_total_cpu_time = 'eref_total_cpu_time'
        _stage_label = 'Energy refinement'
    else:
        _opt_dir_name = 'geometry_refinement'
        _attr_stage_dir = 'refinement_stage_dir'
        _attr_cosmic_folder = 'refinement_cosmic_folder'
        _attr_motifs_source = 'refinement_motifs_source'
        _attr_completed = 'refinement_completed'
        _attr_total = 'refinement_total'
        _attr_job_wall_time = 'refinement_job_wall_time'
        _attr_total_cpu_time = 'refinement_total_cpu_time'
        _stage_label = 'Refinement'
    
    # Parse arguments - defaults for when flags are not explicitly provided
    max_stage_redos = 3   # --redo: redo entire opt+cosmic
    max_critical = 0  # Default: 0% critical structures allowed (retry all)
    max_skipped = None
    concurrent_jobs = 1   # default: 1 concurrent job for refinement (serial)
    _critical_set = False
    _skipped_set = False
    _concurrent_given = False
    # Note: --retry removed; launch failures auto-retry up to 10 times (hardcoded)

    args = stage.get('args', [])
    workflow_concise = getattr(context, 'is_workflow', False) and getattr(context, 'workflow_verbose_level', 0) < 1
    template_inp = stage.get('template_inp')
    launcher_sh = stage.get('launcher_sh')

    for arg in args:
        if arg.startswith('--redo='):
            max_stage_redos = int(arg.split('=')[1])
        elif arg.startswith('--critical='):
            max_critical = float(arg.split('=')[1])
            _critical_set = True
        elif arg.startswith('--skipped='):
            max_skipped = float(arg.split('=')[1])
            _skipped_set = True
        elif arg.startswith('--concurrent='):
            concurrent_jobs = max(1, int(arg.split('=')[1]))
            _concurrent_given = True

    # If --skipped given without explicit --critical, clear the default critical=0
    if _skipped_set and not _critical_set:
        max_critical = None

    # No --concurrent= in stage args ⇒ silent default of 1.

    # Store threshold mode in context for redo logic
    # --critical: only retry structures with imaginary freqs (need_recalculation)
    # --skipped: retry all skipped structures (need_recalculation + clustered_with_minima)
    context.use_skipped_threshold = (max_skipped is not None)
    
    if not template_inp:
        print(f"Error: {_stage_label} requires template input file or embedded template label")
        return 1

    # Resolve path or embedded label from the workflow input file.
    resolved_template = resolve_template_reference(context, template_inp)
    if not resolved_template:
        print(f"Error: Template file/label not found: {template_inp}")
        return 1
    template_inp = resolved_template
    
    # Handle launcher: if not provided, auto-detect ORCA and create launcher
    if launcher_sh:
        launcher_sh = os.path.abspath(launcher_sh)
        if not os.path.exists(launcher_sh):
            print(f"Warning: Launcher script not found: {launcher_sh}")
            if not workflow_concise:
                print("  Attempting to auto-detect ORCA installation...")
            launcher_sh = None
    
    # The line-9 alias from input.in describes the *annealing* QM program only.
    # Refinement templates (#orca/#xtb/#gaussian) declare their own program, so
    # the ORCA auto-launcher must always look for the standard "orca" binary.
    qm_alias = getattr(context, 'qm_alias', 'orca')

    if not launcher_sh:
        # Try to auto-create launcher
        ref_launcher_dir = _opt_dir_name
        if not os.path.exists(ref_launcher_dir):
            os.makedirs(ref_launcher_dir, exist_ok=True)
        auto_launcher = create_auto_launcher(ref_launcher_dir, "orca", "orca", quiet=workflow_concise)
        if auto_launcher:
            launcher_sh = auto_launcher
            if not workflow_concise:
                print(f"  Created auto-generated launcher: {os.path.basename(auto_launcher)}")
        else:
            print("Error: No launcher script provided and could not auto-detect ORCA")
            return 1
    
    # CRITICAL: Refinement/Energy_refinement needs to READ motifs from the previous cosmic stage
    # For refinement: reads from cosmic_1 (optimization output), writes to cosmic_2
    # For energy_refinement: reads from cosmic_2 or cosmic_3 (refinement output), writes to cosmic_3 or cosmic_4

    # Step 1: Find where the MOTIFS are
    motifs_source_folder = None

    # For energy_refinement, use refinement_cosmic_folder; otherwise use optimization_cosmic_folder
    if _stage_kind == 'energy_refinement':
        if hasattr(context, 'refinement_cosmic_folder') and context.refinement_cosmic_folder:
            # Extract base folder (e.g., "cosmic_3" from "cosmic_3/orca_out_10")
            calc_base = _cosmic_base_name(context.refinement_cosmic_folder)
            motifs_source_folder = calc_base
    else:
        # For regular refinement, use optimization_cosmic_folder
        if hasattr(context, 'optimization_cosmic_folder') and context.optimization_cosmic_folder:
            # Extract base folder (e.g., "cosmic" from "cosmic/orca_out_10")
            calc_base = _cosmic_base_name(context.optimization_cosmic_folder)
            motifs_source_folder = calc_base

    if not motifs_source_folder:
        # Cosmic folders are created at the parent level of stage directories.
        search_root = os.path.dirname(os.getcwd())
        existing_sims = []
        for item in os.listdir(search_root):
            item_path = os.path.join(search_root, item)
            if (item.lower().startswith('cosmic') and item[0] in 'cC') and os.path.isdir(item_path):
                existing_sims.append(item)

        if existing_sims:
            # Sort numerically
            existing_sims.sort(key=lambda x: (int(m.group(1)) if (m := re.search(r'_(\d+)', x)) else 0))
            # Find the first one with motifs or umotifs
            for cosmic_folder in existing_sims:
                if glob.glob(os.path.join(cosmic_folder, "motifs_*/")) or glob.glob(os.path.join(cosmic_folder, "umotifs_*/")):
                    motifs_source_folder = cosmic_folder
                    break

    if not motifs_source_folder:
        motifs_source_folder = "cosmic"  # Default

    # Step 2: Find motifs or umotifs in the source folder (prefer umotifs if both exist)
    umotif_dirs = glob.glob(os.path.join(motifs_source_folder, "umotifs_*/"))
    motif_dirs = glob.glob(os.path.join(motifs_source_folder, "motifs_*/"))

    # Prefer umotifs over motifs (more refined clustering)
    if umotif_dirs:
        motif_dirs = umotif_dirs
    elif not motif_dirs:
        print(f"Warning: No motif/umotif directories found in {motifs_source_folder}/")
        print("  Skipping optimization stage")
        return 0

    # Use the most recent motifs directory (natural order: motifs_1 < motifs_2 < ... < motifs_10)
    motif_dirs.sort(key=natural_sort_key)
    motif_dir = motif_dirs[-1]

    # Step 3: Determine where optimization OUTPUTS will go
    # This is typically the next cosmic folder (cosmic_2, cosmic_3, etc.)
    # Always prefer the expected folder from motifs_source_folder and only reuse cached
    # refinement_cosmic_folder if it matches; this prevents stale folder drift.
    if motifs_source_folder.lower() == "cosmic":
        expected_cosmic_folder: str = "cosmic_2"
    else:
        # Extract number and increment
        match = re.search(r'[cC][oO][sS][mM][iI][cC]_(\d+)', motifs_source_folder)
        if match:
            next_num = int(match.group(1)) + 1
            expected_cosmic_folder = f"cosmic_{next_num}"
        else:
            expected_cosmic_folder = "cosmic_2"

    existing_ref_sim_raw = getattr(context, _attr_cosmic_folder, None)
    existing_ref_sim: Optional[str] = existing_ref_sim_raw if isinstance(existing_ref_sim_raw, str) else None
    existing_ref_base: Optional[str] = _cosmic_base_name(existing_ref_sim) if existing_ref_sim else existing_ref_sim
    if isinstance(existing_ref_base, str) and existing_ref_base == expected_cosmic_folder:
        used_cosmic_folder: str = str(existing_ref_base)
    else:
        used_cosmic_folder = str(expected_cosmic_folder)

    # Store cosmic folder in stage-specific context attribute
    setattr(context, _attr_cosmic_folder, used_cosmic_folder)
    # Also update cosmic_dir so the cosmic stage knows where to look
    context.cosmic_dir = used_cosmic_folder
    context.pending_cosmic_folder = used_cosmic_folder

    # Store motifs source in context
    setattr(context, _attr_motifs_source, motif_dir)
    
    # CRITICAL: When resuming optimization stage (with -i flag), clean the OUTPUT cosmic folder
    # This must happen BEFORE process_optimization_redo() checks for skipped_structures
    # This ensures we don't reuse stale skipped_structures from previous cosmic runs
    # Only delete the OUTPUT folder (used_cosmic_folder), NOT the INPUT folder (motifs_source_folder)
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    stage_key = getattr(context, 'current_stage_key', '')
    stage_was_started = stage_key in cache.get('stages', {})
    
    if stage_was_started:
        output_cosmic_folder: str = str(used_cosmic_folder)
        if os.path.exists(output_cosmic_folder):
            # Verify this is NOT the motifs source folder (don't delete our input!)
            if output_cosmic_folder != motifs_source_folder:
                # CRITICAL: Check if this cosmic folder has skipped_structures for redo
                # If skipped_structures exists, we're in a redo scenario - do NOT delete!
                skipped_dir = os.path.join(output_cosmic_folder, "skipped_structures")
                if os.path.exists(skipped_dir):
                    # Redo mode - preserve the cosmic folder with skipped structures
                    pass
                else:
                    # No skipped structures - safe to delete and rebuild
                    try:
                        shutil.rmtree(output_cosmic_folder)
                    except Exception:
                        pass
    
    # Process redo structures at the START of the stage (if need_recalculation exists)
    # This ensures that when the workflow restarts this stage after a cosmic failure,
    # we immediately regenerate inputs and delete old outputs before checking completion
    opt_dir = getattr(context, _attr_stage_dir, _opt_dir_name)
    if not opt_dir:  # Handle empty string
        opt_dir = _opt_dir_name

    # Only process redo if optimization directory exists
    # process_optimization_redo will check for skipped_structures internally and return False if none
    if os.path.exists(opt_dir):
        redo_result = process_optimization_redo(context, opt_dir, template_inp, concurrent_jobs=concurrent_jobs)
        
        # If no redo structures found, clear recalculated_files
        if not redo_result and hasattr(context, 'recalculated_files'):
            context.recalculated_files = None
        
        # Clean up any old input files at the root level (they're now in subdirectories after calculations run)
        # This matches execute_optimization_stage logic to prevent duplicates
        # Note: Do NOT clean redo files here - they need to be run first!
        # Cleanup happens AFTER sorting when files are moved to subdirectories
        if not redo_result:
            # Not a redo - safe to clean orphaned root files
            old_root_inputs = (
                glob.glob(os.path.join(opt_dir, "*.inp")) +
                glob.glob(os.path.join(opt_dir, "*.com")) +
                glob.glob(os.path.join(opt_dir, "*.gjf")) +
                glob.glob(os.path.join(opt_dir, "*.xyz"))
            )
            for old_inp in old_root_inputs:
                basename = os.path.splitext(os.path.basename(old_inp))[0]
                short_name = basename.replace('_opt', '').replace('_calc', '')
                subdir = os.path.join(opt_dir, short_name)
                if os.path.isdir(subdir):
                    try:
                        os.remove(old_inp)
                    except:
                        pass
    
    # Only print motifs message if this is NOT a redo attempt
    # (redo attempts already printed "Processing redo structures..." message)
    if not (hasattr(context, 'recalculated_files') and context.recalculated_files):
        if not workflow_concise:
            print(f"Using motifs from: {motif_dir}")
    
    # Get motif/umotif XYZ files from the motifs directory
    # Look for both motif_*.xyz and umotif_*.xyz patterns
    motif_files = glob.glob(os.path.join(motif_dir, "motif_*.xyz"))
    umotif_files = glob.glob(os.path.join(motif_dir, "umotif_*.xyz"))
    motif_files.extend(umotif_files)  # Combine both patterns
    combined_file = glob.glob(os.path.join(motif_dir, "*combined*.xyz"))
    
    if not motif_files and not combined_file:
        print("Warning: No motif/umotif files found in motifs directory")
        if getattr(context, 'is_workflow', False):
            return 1
        return 0
    
    # Create refinement directory (or reuse if resuming)
    opt_dir = _opt_dir_name
    
    # Check if we're resuming - if so, reuse existing directory
    cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
    cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
    stage_key = getattr(context, 'current_stage_key', '')
    stage_was_started = stage_key in cache.get('stages', {})
    
    # Check if this is a redo scenario (process_redo_structures was called and found files)
    is_redo = hasattr(context, 'recalculated_files') and bool(context.recalculated_files)
    
    # Check if directory has meaningful content (input or output files)
    has_content = False
    if os.path.exists(opt_dir):
        for item in os.listdir(opt_dir):
            if item.endswith(('.inp', '.com', '.gjf', '.xyz', '.out', '.log')):
                has_content = True
                break
            # Also check subdirectories
            item_path = os.path.join(opt_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.endswith(('.inp', '.com', '.gjf', '.xyz', '.out', '.log')):
                        has_content = True
                        break
                if has_content:
                    break
    
    if os.path.exists(opt_dir) and has_content and (stage_was_started or is_redo):
        # Resuming or redo - reuse existing directory
        if is_redo:
            if not workflow_concise:
                print("Resuming: Using existing refinement directory (Redo Mode)\n")
        else:
            if not workflow_concise:
                print("Resuming: Using existing refinement directory\n")
    else:
        # Not resuming or empty directory - create fresh directory
        if os.path.exists(opt_dir):
            # Remove old directory if it exists
            shutil.rmtree(opt_dir)
        os.makedirs(opt_dir)
        # CRITICAL: If we recreated the directory, this is NOT a redo scenario
        is_redo = False

    # Persist the refinement working directory for downstream stages.
    setattr(context, _attr_stage_dir, opt_dir)
    
    # Choose files to process:
    # - If only 1 motif file and a combined file exists: use only the single motif file
    # - If multiple motif files: use the combined file (if it exists), otherwise use individual files
    if len(motif_files) == 1 and combined_file:
        # Single motif - use the individual file, ignore combined
        all_xyz_files = motif_files
    elif combined_file:
        # Multiple motifs - prefer combined file
        all_xyz_files = combined_file
    else:
        # No combined file - use individual motifs
        all_xyz_files = motif_files
    
    # Determine QM program and input extension from template
    if template_inp.endswith(".inp"):
        qm_program = "orca"
        input_ext = ".inp"
    elif template_inp.endswith((".com", ".gjf")):
        qm_program = "gaussian"
        input_ext = ".com" if template_inp.endswith(".com") else ".gjf"
    else:
        qm_program = "xtb"
        input_ext = ".xyz"
    
    # Read template content
    with open(template_inp, 'r') as f:
        template_content = f.read()
    
    # Determine if we should skip input file creation:
    # - is_redo: redo mode, use existing files
    # - stage_was_started AND has_content: resume mode, use existing files
    should_skip_input_creation = is_redo or (stage_was_started and has_content)
    
    # Only generate new input files if NOT in redo/resume mode
    if not should_skip_input_creation:
        # Process each XYZ file and create input files
        all_input_files = []
        for xyz_file in all_xyz_files:
            # Call _process_xyz_file_for_opt with the correct parameters
            xyz_file_data = (xyz_file, template_content, opt_dir, qm_program, input_ext)
            result_files, message = _process_xyz_file_for_opt(xyz_file_data)
            all_input_files.extend(result_files)
        
        if not all_input_files:
            print("Error: No input files created")
            return 1
        
        if not workflow_concise:
            print(f"Created {len(all_input_files)} input file(s)")
    else:
        # Redo/Resume mode: reuse existing input files
        # process_redo_structures handles updating failed ones in redo mode
        # Get list of input files from optimization directory
        all_input_files = []
        
        # Prefer subdirectory files over root files (they may have been organized)
        subdirs = [d for d in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, d))]
        for subdir in sorted(subdirs, key=natural_sort_key):
            subdir_path = os.path.join(opt_dir, subdir)
            for f in os.listdir(subdir_path):
                if f.endswith(('.inp', '.com', '.gjf', '.xyz')):
                    all_input_files.append(os.path.join(subdir, f))
        
        # Only check root if no subdirectory files found
        if not all_input_files:
            root_inputs = [f for f in os.listdir(opt_dir) if f.endswith(('.inp', '.com', '.gjf', '.xyz'))]
            all_input_files.extend(root_inputs)
        
        if not all_input_files:
            print("Error: No existing input files found in optimization directory")
            return 1
    
    # Create launcher script if provided
    # If auto-generated launcher was created before directory reset, recreate it now.
    # Use "orca" alias literally — line 9's alias is for annealing only; the
    # template header (#orca) determines the program for this stage.
    if launcher_sh and not os.path.exists(launcher_sh):
        auto_launcher = create_auto_launcher(opt_dir, "orca", "orca", quiet=workflow_concise)
        if auto_launcher and os.path.exists(auto_launcher):
            launcher_sh = auto_launcher
        else:
            print("Error: Launcher template not found and auto-launcher regeneration failed")
            return 1

    if launcher_sh:
        launcher_path = os.path.join(opt_dir, f"launcher_{qm_program}.sh")
        
        # Read launcher template to get environment setup
        with open(launcher_sh, 'r') as f:
            launcher_template = f.read()
        
        # Extract the environment setup part (everything before ###)
        env_setup = ""
        if "###" in launcher_template:
            env_setup = launcher_template.split("###")[0].strip()
        else:
            # If no ### marker, use the whole template as environment setup
            env_setup = launcher_template.strip()
        
        # For launcher, only use base filenames (not subdirectory paths)
        # The launcher is for user convenience with flat structure
        launcher_input_files = []
        for inp_file in all_input_files:
            # Extract just the filename, not subdirectory paths
            basename_with_ext = os.path.basename(inp_file)
            if basename_with_ext not in launcher_input_files:
                launcher_input_files.append(basename_with_ext)
        
        # Resolve full ORCA path for the launcher
        orca_exe_for_launcher = "orca"
        if qm_program == "orca" and env_setup:
            resolved_orca = resolve_orca_executable_from_launcher(env_setup)
            if resolved_orca:
                orca_exe_for_launcher = resolved_orca

        with open(launcher_path, 'w') as f:
            # Write environment setup from template
            f.write(env_setup)
            f.write("\n\n###\n\n")

            # Write execution commands for each input file (flat structure)
            for i, inp_file in enumerate(sorted(launcher_input_files, key=natural_sort_key)):
                basename = os.path.splitext(inp_file)[0]
                if qm_program == "orca":
                    # Execute ORCA with full path
                    f.write(f"{orca_exe_for_launcher} {basename}.inp > {basename}.out")
                elif qm_program == "xtb":
                    xtb_opts = build_xtb_runtime_options(
                        parse_xtb_options_from_template(template_content),
                        getattr(context, 'qm_nproc', None),
                        getattr(context, 'xtb_cycles', None),
                    )
                    f.write(f"export {_xtb_thread_env_prefix()}\n")
                    f.write(f"{_xtb_thread_env_prefix()} xtb {basename}.xyz {xtb_opts} --namespace {basename} > {basename}.out 2>&1")
                else:
                    # For Gaussian, use g16 or g09
                    f.write(f"g16 {basename}.com")

                # Add continuation for all but last command
                if i < len(launcher_input_files) - 1:
                    f.write(" ; \\\n")
                else:
                    f.write("\n")
        
        os.chmod(launcher_path, 0o755)
        if not workflow_concise:
            print(f"Created launcher script: {launcher_path}")
    
    # Execute the refinement calculations
    if (os.path.exists(os.path.join(opt_dir, "launcher_orca.sh")) or
        os.path.exists(os.path.join(opt_dir, "launcher_gaussian.sh")) or
        os.path.exists(os.path.join(opt_dir, "launcher_xtb.sh"))):
        if not workflow_concise:
            print(f"\nExecuting refinement calculations...")
        
        # Determine launcher name and QM program
        if os.path.exists(os.path.join(opt_dir, "launcher_orca.sh")):
            launcher_path = os.path.join(opt_dir, "launcher_orca.sh")
            qm_program = 'orca'
        elif os.path.exists(os.path.join(opt_dir, "launcher_xtb.sh")):
            launcher_path = os.path.join(opt_dir, "launcher_xtb.sh")
            qm_program = 'xtb'
        else:
            launcher_path = os.path.join(opt_dir, "launcher_gaussian.sh")
            qm_program = 'gaussian'
        
        # Get list of input files to process
        # First check at root level
        # Filter out ORCA intermediate files (.scfgrad.inp, .scfp.inp, etc.) and rescue inputs
        def is_valid_input_file(filename):
            """Check if file is a valid input file (not an ORCA intermediate or rescue file)"""
            if not filename.endswith(('.inp', '.com', '.gjf', '.xyz')):
                return False
            # Exclude ORCA intermediate files, rescue inputs, and aggregate combined files
            excluded_patterns = ['.scfgrad.', '.scfp.', '.tmp.', '.densities.', '.scfhess.', '_rescue.', 'combined_']
            return not any(pattern in filename for pattern in excluded_patterns)

        input_files = sorted([f for f in os.listdir(opt_dir) if is_valid_input_file(f)], key=natural_sort_key)
        
        # If no files at root and sort command was used, check subdirectories
        if not input_files:
            subdirs = [d for d in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, d))]
            for subdir in sorted(subdirs, key=natural_sort_key):
                subdir_path = os.path.join(opt_dir, subdir)
                subdir_files = [os.path.join(subdir, f) for f in os.listdir(subdir_path) if is_valid_input_file(f)]
                input_files.extend(sorted(subdir_files, key=natural_sort_key))
        
        # Load cache and exclusions BEFORE using them
        cache_file = getattr(context, 'cache_file', 'protocol_cache.pkl')
        cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
        
        # Get the current stage key from context (e.g., "optimization_4")
        stage_key = getattr(context, 'current_stage_key', '')
        
        # Check if redo structures exist (files scheduled for recalculation)
        redo_files = set()
        if hasattr(context, 'recalculated_files') and context.recalculated_files:
            redo_files = set(context.recalculated_files)
        
        # Scan ALL subdirectories for completed calculations (check for .out files)
        # Files with only .out.backup are being redone and should NOT be counted as completed
        actual_completed = []
        opt_dir = getattr(context, _attr_stage_dir, None) or _opt_dir_name

        # 1. Check optimization directory subfolders
        if os.path.exists(opt_dir):
            for item in os.listdir(opt_dir):
                item_path = os.path.join(opt_dir, item)
                if os.path.isdir(item_path):
                    # Skip if this file is marked for redo
                    # CRITICAL: redo_files contains basenames with _opt suffix (e.g., "umotif_01_opt")
                    # but item is the directory name (e.g., "umotif_01")
                    # So we need to check both "item" AND "item_opt"/"item_calc" against redo_files
                    redo_variants = [item, f"{item}_opt", f"{item}_calc"]
                    if any(variant in redo_files for variant in redo_variants):
                        continue
                    
                    # Check if this subdirectory has a completed calculation
                    # Try different naming patterns: umotif_01/umotif_01.out or umotif_01/umotif_01_opt.out
                    possible_names = [f"{item}.out", f"{item}_opt.out", f"{item}_calc.out"]
                    out_file = None
                    for name in possible_names:
                        test_path = os.path.join(item_path, name)
                        if os.path.exists(test_path):
                            out_file = test_path
                            break
                    
                    if out_file:
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if check_qm_output_completed('auto', out_file):
                            # Store with _opt suffix for consistency
                            if '_opt' not in item and '_calc' not in item:
                                actual_completed.append(f"{item}_opt")
                            else:
                                actual_completed.append(item)
            
            # 1b. Check for flat files in optimization directory
            for item in os.listdir(opt_dir):
                # Skip rescue artifacts (e.g. motif_24_opt_rescue.out): they are
                # intermediate redo outputs for a parent motif, not a distinct
                # refinement. Counting them double-inflates completed/total (and
                # the inflated value then gets locked across redo attempts).
                if '_rescue' in item:
                    continue
                if item.endswith('.out') or item.endswith('.log'):
                    basename = os.path.splitext(item)[0]
                    # Skip if already found in subdir or marked for redo
                    if basename not in actual_completed and basename not in redo_files:
                        out_file = os.path.join(opt_dir, item)
                        # Verify completion (OPI-aware for ORCA 6.1+)
                        if item.endswith('.out'):
                            if check_qm_output_completed('auto', out_file):
                                actual_completed.append(basename)
                        else:  # Gaussian .log files
                            if check_qm_output_completed('gaussian', out_file):
                                actual_completed.append(basename)
        
        # 2. Check cosmic_X/orca_out_* folders (files moved there after sorting)
        # This is CRITICAL for showing correct counts when resuming
        # IMPORTANT: Only check the optimization's OWN cosmic folder, not all cosmic folders
        # (to avoid counting calculation results from cosmic/ as optimization results)
        parent_dir = os.getcwd()
        
        # Determine which cosmic folder belongs to THIS optimization stage
        refinement_cosmic_folder = getattr(context, _attr_cosmic_folder, None)

        # CRITICAL: Strip orca_out_X suffix if present
        # organize step may set the folder to "cosmic_2/orca_out_5"
        # but we need just "cosmic_2" to scan for orca_out subdirectories
        if refinement_cosmic_folder and '/' in refinement_cosmic_folder:
            refinement_cosmic_folder = _cosmic_base_name(refinement_cosmic_folder)

        if not refinement_cosmic_folder:
            # Calculate the expected folder based on motifs source
            if hasattr(context, _attr_motifs_source):
                motifs_source = getattr(context, _attr_motifs_source)
                # Extract base folder (e.g., "cosmic" from "cosmic/motifs_03/")
                calc_base = _cosmic_base_name(motifs_source)

                # Optimization outputs go to the NEXT cosmic folder
                if calc_base.lower() == "cosmic":
                    refinement_cosmic_folder = "cosmic_2"
                else:
                    # Extract number and increment
                    match = re.search(r'[cC][oO][sS][mM][iI][cC]_(\d+)', calc_base)
                    if match:
                        next_num = int(match.group(1)) + 1
                        refinement_cosmic_folder = f"cosmic_{next_num}"
                    else:
                        refinement_cosmic_folder = "cosmic_2"
        
        # Only check the optimization's designated cosmic folder
        if refinement_cosmic_folder and os.path.isdir(refinement_cosmic_folder):
            for subitem in os.listdir(refinement_cosmic_folder):
                if subitem.startswith('orca_out_') or subitem.startswith('gaussian_out_') or subitem.startswith('calc_out_'):
                    out_dir = os.path.join(refinement_cosmic_folder, subitem)
                    if os.path.isdir(out_dir):
                        for f in os.listdir(out_dir):
                            # Skip rescue artifacts — intermediate redo outputs,
                            # not distinct refinements (see flat-file scan above).
                            if '_rescue' in f:
                                continue
                            if f.endswith('.out') or f.endswith('.log'):
                                basename = os.path.splitext(f)[0]
                                if basename not in actual_completed and basename not in redo_files:
                                    # Verify completion (OPI-aware for ORCA 6.1+)
                                    out_file = os.path.join(out_dir, f)
                                    if f.endswith('.out'):
                                        if check_qm_output_completed('auto', out_file):
                                            actual_completed.append(basename)
                                    else:  # Gaussian .log files
                                        if check_qm_output_completed('gaussian', out_file):
                                            actual_completed.append(basename)
        
        # Update completed_opts to match reality
        completed_opts = actual_completed
        
        excluded_numbers = cache.get('excluded_refinements', [])
        
        # Apply exclusion filtering to completed_opts, then collapse every scan
        # spelling to one canonical structure key so a motif discovered under
        # both 'motif_24_opt' and 'motif_24_opt.inp' (or its redo/rescue variant)
        # is only counted once.
        completed_opts = [f for f in completed_opts if not match_exclusion(f, excluded_numbers)]
        completed_opts = list(dict.fromkeys(_canonical_calc_key(f) for f in completed_opts))

        # Count total inputs (including those already completed and cached)
        # This should be the TOTAL expected optimizations, not just new ones.
        # Keyed by canonical structure id so the denominator stays anchored to
        # the real motif count and never grows across redo passes.
        all_input_basenames = set()

        # Add keys from input files (pending optimizations)
        for f in input_files:
            if not match_exclusion(f, excluded_numbers):
                # Handle both simple filenames and paths like "motif_01/motif_01_opt.inp"
                all_input_basenames.add(_canonical_calc_key(f))

        # Add keys from already completed files (CRITICAL for correct count)
        for f in completed_opts:
            all_input_basenames.add(_canonical_calc_key(f))

        num_inputs = len(all_input_basenames)

        # Lock the maximum total across redo attempts so the live N/M display does
        # not grow when redos rerun a subset. Redos drop completed back, then
        # rebuild up to the locked maximum.
        _ref_stage_key = getattr(context, 'current_stage_key', '')
        if _ref_stage_key:
            _locked_totals = getattr(context, '_opt_locked_totals', None)
            if _locked_totals is None:
                _locked_totals = {}
                context._opt_locked_totals = _locked_totals
            _prev_locked = _locked_totals.get(_ref_stage_key, 0)
            if _prev_locked > num_inputs:
                num_inputs = _prev_locked
            else:
                _locked_totals[_ref_stage_key] = num_inputs

        # Store initial completed count (before loop may modify completed_opts)
        initial_completed_count = len(completed_opts)
        if initial_completed_count > num_inputs:
            initial_completed_count = num_inputs

        progress_cb = context.update_progress
        if workflow_concise and callable(progress_cb):
            progress_cb(f"{initial_completed_count}/{num_inputs} ...")
        
        if completed_opts and not workflow_concise:
            print(f"Resuming: {len(completed_opts)}/{num_inputs} optimizations already completed")
        
        if excluded_numbers and not workflow_concise:
            print(f"Exclusions active: {excluded_numbers}")
            print()
        
        # Read launcher script to get environment setup
        with open(launcher_path, 'r') as f:
            launcher_content = f.read()
        
        # Resolve ORCA executable path
        orca_exe = 'orca'
        if qm_program == 'orca':
            orca_exe = resolve_orca_executable_from_launcher(launcher_content, qm_alias)

        # Build list of pending jobs (skip excluded and already completed)
        max_launch_retries = 10
        launch_failure_threshold = 5.0
        pending_jobs = []

        for input_file in input_files:
            basename = os.path.splitext(input_file)[0]

            # Skip excluded optimizations UNLESS this file is being redone
            if match_exclusion(input_file, excluded_numbers):
                is_redo_file = (is_redo and hasattr(context, 'recalculated_files') and
                                context.recalculated_files is not None and
                                basename in context.recalculated_files)
                if not is_redo_file:
                    if not is_redo:
                        if not workflow_concise:
                            print(f"  Skipping: {input_file} (excluded)")
                    continue

            output_file = basename + ('.out' if qm_program in ('orca', 'xtb') else '.log')
            output_path = os.path.join(opt_dir, output_file)

            # Skip if output already exists AND is successfully completed
            output_exists = False
            if os.path.exists(output_path):
                output_exists = True
            else:
                subdir_path = os.path.join(opt_dir, basename, output_file)
                if os.path.exists(subdir_path):
                    output_path = subdir_path
                    output_exists = True
                else:
                    short_basename = basename
                    if '_opt' in basename:
                        short_basename = basename.replace('_opt', '')
                    elif '_calc' in basename:
                        short_basename = basename.replace('_calc', '')
                    subdir_path = os.path.join(opt_dir, short_basename, output_file)
                    if os.path.exists(subdir_path):
                        output_path = subdir_path
                        output_exists = True

            if output_exists:
                try:
                    is_complete = check_qm_output_completed(qm_program, output_path)
                except Exception:
                    is_complete = False
                if is_complete:
                    continue

            pending_jobs.append({
                'input_file': input_file,
                'base_dir': opt_dir,
                'launcher_content': launcher_content,
                'qm_program': qm_program,
                'max_launch_retries': max_launch_retries,
                'launch_failure_threshold': launch_failure_threshold,
                'orca_exe': orca_exe,
                'xtb_options': build_xtb_runtime_options(
                    parse_xtb_options_from_launcher(launcher_content),
                    getattr(context, 'qm_nproc', None),
                    getattr(context, 'xtb_cycles', None),
                ) if qm_program == 'xtb' else None,
            })

        # Run calculations with concurrency
        _ref_wall_start = time.time()
        num_completed, num_failed, failed_optimizations = _run_qm_calculations_with_concurrency(
            pending_jobs=pending_jobs,
            concurrent_jobs=concurrent_jobs,
            workflow_concise=workflow_concise,
            context=context,
            initial_completed_count=initial_completed_count,
            num_inputs=num_inputs,
            completed_list=completed_opts,
            all_input_basenames=all_input_basenames,
            cache_file=cache_file,
            stage_key_prefix=_stage_kind,
        )
        # Accumulate active wall across resume sessions so the summary's
        # "Total wall time" survives interruptions (never < longest job).
        setattr(context, _attr_job_wall_time, accumulate_stage_wall_time(
            cache_file, _ref_stage_key, time.time() - _ref_wall_start))

        # Print status
        # Recalculate num_inputs from the updated set to reflect newly completed optimizations
        num_inputs = len(all_input_basenames)
        # Keep the redo-locked maximum so subsequent redo attempts do not inflate the total.
        if _ref_stage_key:
            _locked_totals = getattr(context, '_opt_locked_totals', {})
            _prev_locked = _locked_totals.get(_ref_stage_key, 0)
            if _prev_locked > num_inputs:
                num_inputs = _prev_locked
            else:
                _locked_totals[_ref_stage_key] = num_inputs
        # Total completed = initial completed + newly completed in this run
        total_completed = initial_completed_count + num_completed
        # Ensure we don't show more completed than total inputs (can happen if files are removed)
        if total_completed > num_inputs:
            total_completed = num_inputs

        # Fallback pass: detect successfully terminated outputs recursively.
        # This covers edge cases where folder/file naming differs from expected _opt/_calc patterns.
        if total_completed == 0 and num_inputs > 0 and os.path.isdir(opt_dir):
            fallback_completed = set()
            for root, _, files in os.walk(opt_dir):
                for filename in files:
                    if filename.endswith('.backup'):
                        continue
                    # Rescue artifacts are intermediate redo outputs, not distinct refinements.
                    if '_rescue' in filename:
                        continue
                    if not (filename.endswith('.out') or filename.endswith('.log')):
                        continue

                    out_path = os.path.join(root, filename)
                    qm_kind = 'gaussian' if filename.endswith('.log') else 'auto'
                    if not check_qm_output_completed(qm_kind, out_path):
                        continue

                    base = os.path.splitext(filename)[0]
                    normalized = base
                    if normalized not in all_input_basenames:
                        for candidate in (f"{base}_opt", f"{base}_calc"):
                            if candidate in all_input_basenames:
                                normalized = candidate
                                break
                    if normalized in all_input_basenames:
                        fallback_completed.add(normalized)

            if fallback_completed:
                completed_opts = sorted(fallback_completed, key=natural_sort_key)
                total_completed = min(len(fallback_completed), num_inputs)

        _kind_noun = 'energy refinements' if _stage_kind == 'energy_refinement' else 'refinements'
        _kind_title = 'Energy refinement' if _stage_kind == 'energy_refinement' else 'Refinement'

        # Collapse redo/rescue reruns to their parent-motif identity before
        # recording the result. Across redo attempts the live counter and its
        # locked maximum can grow past the real target count (each rerun of a
        # non-converged motif, or a rescue variant like ``motif_24_opt_rescue``,
        # otherwise inflates total/completed). extract_base() strips the
        # _opt/_calc/_rescue suffixes so motif_03_opt and motif_03_opt_rescue both
        # map to motif_03 — the distinct-parent count is the true number of
        # refinement targets (e.g. 27, not the redo-inflated 37).
        if all_input_basenames:
            _distinct_targets = len({extract_base(b) for b in all_input_basenames})
            if _distinct_targets > 0:
                _distinct_completed = len({extract_base(b) for b in completed_opts})
                num_inputs = min(num_inputs, _distinct_targets)
                total_completed = min(total_completed, _distinct_completed, num_inputs)

        if not workflow_concise:
            print(f"\nStatus: {total_completed}/{num_inputs} {_kind_noun} completed")

        # Store for protocol summary (use total)
        setattr(context, _attr_completed, total_completed)
        setattr(context, _attr_total, num_inputs)

        # Clean up old failed files if they exist and all succeeded
        if not failed_optimizations:
            for old_file in [os.path.join(opt_dir, "failed_opt.txt"), os.path.join(opt_dir, "launcher_failed.sh")]:
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                    except Exception:
                        pass

        # Handle failed optimizations
        if failed_optimizations:
            if not workflow_concise:
                print(f"Failed {_kind_noun}: {len(failed_optimizations)}/{num_inputs}")

            # Write failed_opt.txt
            failed_list_file = os.path.join(opt_dir, "failed_opt.txt")
            with open(failed_list_file, 'w') as f:
                f.write(f"# Failed {_kind_noun}: {len(failed_optimizations)}/{num_inputs}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for failed_file in sorted(failed_optimizations, key=natural_sort_key):
                    f.write(f"{failed_file}\n")

            if not workflow_concise:
                print(f"Failed {_kind_noun} list written to: {failed_list_file}")

        # Organize files if any refinements are completed (new or already done)
        if total_completed > 0:
            if num_completed > 0 and not workflow_concise:
                print(f"\n\n{_kind_title} calculations completed")

            # Check if this is a redo (recalculated files exist)
            if not workflow_concise:
                print(f"All {_kind_noun} completed successfully")
            
            # Organize results - run full sort/organize for both redo and normal mode
            saved_cwd = os.getcwd()
            try:
                # Determine cosmic folder - use the stage-specific cosmic folder attribute
                _ctx_cosmic = getattr(context, _attr_cosmic_folder, None)
                if _ctx_cosmic:
                    # Use the folder determined at the start of this function
                    cosmic_base = _cosmic_base_name(_ctx_cosmic)
                else:
                    # Calculate next cosmic folder
                    root_dir = os.getcwd()
                    base_name = "cosmic"
                    counter = 2
                    cosmic_base = base_name

                    if os.path.exists(os.path.join(root_dir, cosmic_base)):
                        while True:
                            cosmic_base = f"{base_name}_{counter}"
                            if not os.path.exists(os.path.join(root_dir, cosmic_base)):
                                break
                            counter += 1
                    
                    context.cosmic_dir = cosmic_base
                
                os.chdir(opt_dir)
                
                # Get exclusions from cache to filter output files
                cache = load_protocol_cache(cache_file) if os.path.exists(cache_file) else {}
                excluded_numbers = cache.get('excluded_refinements', [])
                
                # Group files by base names into subfolders (silent in workflow)
                import io
                import contextlib
                f = io.StringIO()
                _ref_actual_wall = getattr(context, _attr_job_wall_time, None)
                with contextlib.redirect_stdout(f):
                    group_files_by_base_with_tracking(".")
                    combine_xyz_files()
                    create_combined_mol()
                    create_summary_with_tracking(".", actual_wall_time=_ref_actual_wall)

                    # Collect output files - reuse existing cosmic folder in redo mode
                    # But first, we need to temporarily filter excluded files
                    # Save original find_out_files function
                    original_find_out_files = find_out_files
                    
                    def filtered_find_out_files(root_dir, include_orca=True, include_gaussian=True):
                        """Find .out files but exclude those matching exclusion patterns."""
                        all_files = original_find_out_files(
                            root_dir,
                            include_orca=include_orca,
                            include_gaussian=include_gaussian
                        )
                        filtered_files = []
                        for file_path in all_files:
                            basename = os.path.splitext(os.path.basename(file_path))[0]
                            # Check if this file matches any exclusion
                            if not match_exclusion(basename, excluded_numbers):
                                filtered_files.append(file_path)
                        return filtered_files
                    
                    # Temporarily replace function in globals
                    globals()['find_out_files'] = filtered_find_out_files
                    
                    try:
                        # Check if we should reuse existing folder:
                        # - True if redo mode (recalculated_files exist)
                        # - Also True if resuming and cosmic folder already has orca_out folder
                        reuse_folder = is_redo
                        if not reuse_folder and os.path.exists(os.path.join(os.path.dirname(os.getcwd()), cosmic_base)):
                            # Check if orca_out folder already exists in cosmic
                            cosmic_full_path = os.path.join(os.path.dirname(os.getcwd()), cosmic_base)
                            existing_orca = glob.glob(os.path.join(cosmic_full_path, "orca_out_*"))
                            if existing_orca:
                                reuse_folder = True
                        
                        cosmic_folder = collect_out_files_with_tracking(
                            reuse_existing=reuse_folder,
                            target_cosmic_folder=cosmic_base
                        )
                    finally:
                        # Restore original function
                        globals()['find_out_files'] = original_find_out_files
                
                # Extract key info from output
                output = f.getvalue()
                if context.workflow_verbose_level >= 1:
                    if 'Summary written to' in output:
                        print("\nSummary file(s) generated")
                if cosmic_folder:
                    setattr(context, _attr_cosmic_folder, cosmic_folder)
                    cosmic_base = _cosmic_base_name(cosmic_folder)
                    context.pending_cosmic_folder = cosmic_base
                if 'Copied' in output and ('cosmic' in output.lower() or 'COSMIC' in output):
                    # Extract the copy message and cosmic folder
                    for line in output.split('\n'):
                        if 'Copied' in line and '.out files to' in line:
                            if context.workflow_verbose_level >= 1:
                                print(line)
                            # Extract cosmic folder name
                            match = re.search(r'to\s+([cC][oO][sS][mM][iI][cC][^\s]*)', line)
                            if match:
                                cosmic_folder = match.group(1)
                                setattr(context, _attr_cosmic_folder, cosmic_folder)
                                # Also set as pending for next cosmic stage
                                cosmic_base = _cosmic_base_name(cosmic_folder)
                                context.pending_cosmic_folder = cosmic_base
                            break
                
                # Note: File update messages are now handled within collect_out_files_with_tracking
                # No need to print them again here (unlike optimization stage where we copy manually)
                
                # Clean up orphaned root input files after sorting
                # Files should now be in subdirectories, so root copies are duplicates
                old_root_inputs = (
                    glob.glob(os.path.join(".", "*.inp")) +
                    glob.glob(os.path.join(".", "*.com")) +
                    glob.glob(os.path.join(".", "*.gjf")) +
                    glob.glob(os.path.join(".", "*.xyz"))
                )
                for old_inp in old_root_inputs:
                    inp_basename = os.path.splitext(os.path.basename(old_inp))[0]
                    short_name = inp_basename.replace('_opt', '').replace('_calc', '')
                    subdir = os.path.join(".", short_name)
                    if os.path.isdir(subdir):
                        # Subfolder exists - safe to remove root file
                        try:
                            os.remove(old_inp)
                        except:
                            pass
            
                if context.workflow_verbose_level >= 1:
                    print(f"\n✓ Files organized and sorted")
            except Exception as e:
                print(f"⚠ Warning: Could not organize files: {e}")
            finally:
                    os.chdir(saved_cwd)

            # Parse total CPU time from orca_summary.txt for protocol summary mean exec time.
            # Use saved_cwd to build an absolute path so cwd changes don't break resolution.
            _ref_sum_path = os.path.join(saved_cwd, opt_dir, "orca_summary.txt")
            if not os.path.exists(_ref_sum_path):
                _ref_sum_path = os.path.join(opt_dir, "orca_summary.txt")
            if os.path.exists(_ref_sum_path):
                try:
                    with open(_ref_sum_path, 'r') as _sf:
                        _sc = _sf.read()
                    _tm = __import__('re').search(r'Total execution time:\s+(\d+):(\d+):(\d+\.\d+)', _sc)
                    if _tm:
                        setattr(context, _attr_total_cpu_time,
                                int(_tm.group(1)) * 3600 + int(_tm.group(2)) * 60 + float(_tm.group(3)))
                except Exception:
                    pass

        else:
            if failed_optimizations:
                print(f"✗ All {num_inputs} {_kind_noun} failed "
                      f"(0 normal terminations) — see {os.path.join(opt_dir, 'failed_opt.txt')}")
            else:
                print("✗ No output files found")
            return 1
    else:
        print(f"Warning: No launcher script found in {opt_dir}/")
        return 1

    return 0


def execute_energy_refinement_stage(context: WorkflowContext, stage: Dict[str, Any]) -> int:
    """Execute energy refinement stage (single-point calculations on motifs from COSMIC).

    Runs high-level single-point energy calculations on motifs from the previous COSMIC
    stage and stores results in energy_refinement/.  Stores context.eref_motifs_source
    pointing to the source COSMIC folder so the following cosmic stage can compute
    composite Gibbs energies: G = E_eref + (G_prev - E_prev).
    """
    return execute_refinement_stage(context, stage, _stage_kind='energy_refinement')


# =========================================================================== #
# R7 verbatim ports — the CLI helper closure.                                  #
#                                                                             #
# v04's ``main_ascec_integrated`` dispatcher (R7 — see                         #
# ``cosmic_ascec/command_line/ascec.py``) calls a small army of top-level     #
# helpers: ``print_version_banner`` (and the ``ASCEC_VERSION`` constant it     #
# echoes), the protocol-extraction pair                                       #
# ``extract_protocol_from_input`` / ``consume_protocol_maxprint_flag``, the   #
# ``parse_exclusion_pattern`` parser used by ``ascec <file> exclude`` (R7     #
# branch 4), ``provide_box_length_advice`` (used by every bare-input-file     #
# single-run *and* by the standalone ``box`` subcommand), and the seven       #
# ``execute_*`` subcommand handlers — ``execute_sort_command``,               #
# ``execute_summary_only``, ``execute_box_analysis``,                         #
# ``execute_cosmic_analysis``, ``execute_diagram_generation``,                #
# ``execute_merge_command`` (+ its ``execute_merge_result_command`` wrapper), #
# plus the geometry-optimization / refinement input-generators                #
# ``create_simple_optimization_system`` / ``create_refinement_system`` and    #
# ``update_existing_input_files``. The interactive ``ascec status`` viewer     #
# ``show_ascec_status`` and its 14 nested helpers (``_attach_view`` /          #
# ``_show_progress_screen`` / …) are self-contained, so they round-trip       #
# verbatim too.                                                                #
#                                                                             #
# Every constant + function below is a byte-identical extract of              #
# ``ascec-v04.py`` (**D-039**: faithful decomposition — no redesign).         #
# Extracted programmatically by ``scripts/extract_r7_helpers.py``, not        #
# retyped. v04 source order is preserved by ``--- label  (lines N-M) ---``    #
# banners between definitions.                                                #
#                                                                             #
# A handful of these helpers reference further v04 module-level helpers       #
# (``merge_xyz_files``, ``calculate_optimal_box_length``,                     #
# ``calculate_input_files``, ``create_qm_input_file``,                        #
# ``plot_annealing_diagrams`` / ``plot_combined_replicas_diagram``, the       #
# ``SystemState`` class, ``read_input_file``, ``_print_verbose``,             #
# ``summarize_calculations``, ``detect_output_file_type``,                    #
# ``collect_out_files_with_tracking``, ``combine_xyz_files`` /                #
# ``create_combined_mol``, ``group_files_by_base_with_tracking``,             #
# ``create_summary_with_tracking``) — all already verbatim-ported above       #
# (R6/R6b/R6c/R6d). Python resolves bare names at call time, so the R7 ports  #
# defined below pick them up correctly.                                       #
# =========================================================================== #


# --- const ASCEC_VERSION  (ascec-v04.py 64-64) ---
ASCEC_VERSION = "* ASCEC-v04: Feb 2026 *"  # ASCEC version string for display


# --- const B2  (ascec-v04.py 68-68) ---
B2 = 3.166811563e-6   # Boltzmann constant in Hartree/K (approx. 3.166811563 × 10^-6 Hartree/K)


# --- const create_box_xyz_copy  (ascec-v04.py 81-81) ---
create_box_xyz_copy = True 


# --- def print_version_banner  (ascec-v04.py 119-153) ---
def print_version_banner(script_name="ASCEC"):
    """Print the ASCII art banner with UdeA logo and version information."""
    banner = """
===========================================================================

                           *********************                           
                           *     A S C E C     *                           
                           *********************                           

                             √≈≠==≈                                  
   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       
     ÷++÷       ÷++÷           =++=                     ÷×××××=      
     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      
     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     
     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    
     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    
     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   
      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   
       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- 
          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ 


               Universidad de Antioquia - Medellín - Colombia              


                  Annealing Simulado Con Energía Cuántica                  

                           {version}                           

                        Química Física Teórica - QFT                       


===========================================================================
""".format(version=ASCEC_VERSION)
    print(banner)


# --- def get_molecular_formula  (ascec-v04.py 1122-1164) ---
def get_molecular_formula(mol_def) -> str:
    """Generate molecular formula from molecule definition."""
    if not mol_def.atoms_coords:
        return "Unknown"
    
    # Count atoms by element
    element_counts = {}
    for atom_data in mol_def.atoms_coords:
        # Handle both 4-field and 5+ field formats (atomic_num, x, y, z, [extra...])
        # mol_def.atoms_coords is already parsed, so we just need the atomic_num
        if len(atom_data) >= 1:
            atomic_num = atom_data[0]
        else:
            continue  # Skip malformed entries
        element = get_element_symbol(atomic_num)
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Sort elements: C, H, then by electronegativity
    sorted_elements = []
    if 'C' in element_counts:
        sorted_elements.append('C')
    if 'H' in element_counts:
        sorted_elements.append('H')
    
    # Get remaining elements
    remaining_elements = [e for e in element_counts.keys() if e not in ['C', 'H']]
    
    # Sort remaining elements by electronegativity (ascending)
    # Elements not in the dictionary get a high value to be at the end
    remaining_elements.sort(key=lambda e: electronegativity_values.get(e, 1000.0))
    
    sorted_elements.extend(remaining_elements)
    
    # Build formula string
    formula = ""
    for element in sorted_elements:
        count = element_counts[element]
        if count == 1:
            formula += element
        else:
            formula += f"{element}{count}"
    
    return formula if formula else "Unknown"


# --- def extract_protocol_from_input  (ascec-v04.py 1746-1786) ---
def extract_protocol_from_input(input_file: str) -> Optional[str]:
    """
    Extract protocol line(s) from input file if present.
    Looks for lines starting with '.asc,' marker, supports multi-line format.
    
    Returns:
        Protocol command string or None if not found
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()

        protocol_lines: List[str] = []
        in_protocol_section = False

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if '#' in stripped:
                stripped = stripped.split('#')[0].strip()
                if not stripped:
                    continue
            if is_protocol_marker_line(stripped):
                in_protocol_section = True
                protocol_lines.append(stripped)
                continue
            if in_protocol_section:
                protocol_lines.append(stripped)
                if not stripped.endswith(',') and not stripped.endswith('.'):
                    break

        if protocol_lines:
            protocol = ' '.join(protocol_lines)
            protocol = ' '.join(protocol.split())
            return protocol

    except Exception:
        pass
    
    return None


# --- def consume_protocol_maxprint_flag  (ascec-v04.py 1789-1818) ---
def consume_protocol_maxprint_flag(protocol_text: str) -> Tuple[str, bool]:
    """Strip embedded --maxprint token from protocol text and report if present.

    Supports both inline and split-line styles, e.g.:
      .asc --maxprint,
      .asc,
      --maxprint,
    """
    if not protocol_text:
        return protocol_text, False

    found_maxprint = False

    def _strip_flag(match: re.Match) -> str:
        nonlocal found_maxprint
        found_maxprint = True
        return match.group(1) or ''

    cleaned = re.sub(
        r'(?i)(^|[\s,])--maxprint(?=\s|,|$)\s*,?',
        _strip_flag,
        protocol_text,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r',\s+', ', ', cleaned)
    while ',,' in cleaned:
        cleaned = cleaned.replace(',,', ',')

    return cleaned, found_maxprint


# --- def parse_exclusion_pattern  (ascec-v04.py 1950-1981) ---
def parse_exclusion_pattern(pattern: str) -> List[int]:
    """
    Parse exclusion pattern into list of numbers.
    
    Supported formats:
        "03" -> [3]
        "03,04" -> [3, 4]
        "03-15" -> [3, 4, 5, ..., 15]
        "03,05-15" -> [3, 5, 6, 7, ..., 15]
        "01,03-05,10" -> [1, 3, 4, 5, 10]
    
    Args:
        pattern: Exclusion pattern string
        
    Returns:
        List of numbers to exclude
    """
    excluded = []
    parts = pattern.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            # Range: 03-15
            start, end = part.split('-')
            start_num = int(start)
            end_num = int(end)
            excluded.extend(range(start_num, end_num + 1))
        else:
            # Single number: 03
            excluded.append(int(part))
    
    return sorted(list(set(excluded)))  # Remove duplicates and sort


# --- def provide_box_length_advice  (ascec-v04.py 4599-4727) ---
def provide_box_length_advice(state: SystemState):
    """
    Provides comprehensive advice on appropriate box lengths.
    Both methods use L = (V_eff / phi)^(1/3):
      Method A (HB):     V_eff = V_mol + V_HB
      Method B (non-HB): V_eff = L_diag³, where L_diag = sum_of_extents / sqrt(3)
    """
    if not state.all_molecule_definitions:
        _print_verbose("Cannot provide box length advice: No molecule definitions found.", 0, state)
        return

    _print_verbose("\n" + "="*78, 1, state)
    _print_verbose("Box length analysis", 1, state)
    _print_verbose("="*78, 1, state)

    # Calculate optimal box lengths using unified volume-based approach
    results = calculate_optimal_box_length(state)

    if 'error' in results:
        _print_verbose(f"Error in volume analysis: {results['error']}", 0, state)
        return

    system_has_hbonds = results.get('has_primary_hbonds', False)
    num_molecules = results['num_molecules']

    _print_verbose(f"Successfully parsed {state.natom} atoms", 1, state)
    _print_verbose("", 1, state)

    # Section 1: Molecular volume analysis
    total_molecular_volume = results['total_molecular_volume']
    total_hb_volume = results['total_hb_network_volume']
    total_effective_volume = results['total_effective_volume']

    _print_verbose("1. Molecular volume analysis:", 1, state)
    _print_verbose("-" * 50, 1, state)
    _print_verbose(f"Number of molecules to place: {num_molecules}", 1, state)
    _print_verbose(f"  Total molecular hull volume: {total_molecular_volume:.2f} Å³", 1, state)
    if system_has_hbonds:
        _print_verbose(f"  Total H-bond network volume: {total_hb_volume:.2f} Å³", 1, state)
        _print_verbose(f"  Effective volume: V_mol + V_HB = {total_effective_volume:.2f} Å³", 1, state)
    else:
        _print_verbose(f"  Sum of extents (ΣE): {results['diagonal_sum_extents']:.2f} Å", 1, state)
        _print_verbose(f"  L_diag = ΣE / √3 = {results['diagonal_box_length']:.2f} Å", 1, state)
        _print_verbose(f"  Effective volume: L_diag³ = {total_effective_volume:.2f} Å³", 1, state)

    _print_verbose("\nIndividual molecule analysis:", 1, state)
    for i, mol_info in enumerate(results['individual_molecular_volumes']):
        if i < len(state.all_molecule_definitions):
            mol_def = state.all_molecule_definitions[i]
            molecular_formula = get_molecular_formula(mol_def)
            extent_str = f", extent={mol_info['extent_A']:.2f} Å" if 'extent_A' in mol_info else ""
            _print_verbose(f"  • {mol_info['molecule_label']}: {molecular_formula} {mol_info['volume_A3']:.2f} Å³{extent_str}", 1, state)
        else:
            _print_verbose(f"  • {mol_info['molecule_label']}: {mol_info['volume_A3']:.2f} Å³", 1, state)

    # Section 2: Box length suggestions table (5% to 50%)
    recommendations = results['box_length_recommendations']
    _print_verbose(f"\n2. Box length suggestions:", 1, state)
    _print_verbose("-" * 66, 1, state)
    _print_verbose("Packing (%)    Box Length (Å)     Box Volume (Å³)       Free (%)", 1, state)
    _print_verbose("-" * 66, 1, state)
    for key, rec in recommendations.items():
        pf = rec['packing_fraction']
        bl = rec['box_length_A']
        bv = rec['box_volume_A3']
        free_pct = rec['free_volume_fraction'] * 100
        _print_verbose(f"    {pf*100:4.1f}          {bl:6.2f}             {bv:6.0f}               {free_pct:4.1f}", 1, state)

    # Section 3: Current box analysis
    if 'current_box_analysis' in results:
        _print_verbose("\n3. Current box analysis:", 1, state)
        _print_verbose("-" * 26, 1, state)
        current = results['current_box_analysis']
        _print_verbose(f"Cube's length = {current['current_box_length_A']:.2f} Å", 1, state)
        _print_verbose(f"  Current effective packing: {current['current_packing_fraction']:.1%}", 1, state)
        _print_verbose(f"    └ Molecular: {current['molecular_packing_fraction']:.1%}, H-bond network: {current['hb_network_fraction']:.1%}", 1, state)
        _print_verbose(f"  Current free volume: {current['current_free_volume_A3']:.0f} Å³ "
                      f"({current['current_free_volume_fraction']:.1%})", 1, state)
        _print_verbose(f"  Largest molecular extent: {results['max_molecular_extent_A']:.2f} Å", 1, state)

        pf = current['current_packing_fraction']
        if pf < 0.05:
            assessment = "Very dilute - good for isolated cluster studies"
        elif pf < 0.15:
            assessment = "Dilute - appropriate for cluster formation"
        elif pf < 0.25:
            assessment = "Moderate - suitable for network formation studies"
        elif pf < 0.35:
            assessment = "Dense - good for condensed phase simulations"
        elif pf < 0.45:
            assessment = "Very dense - may constrain molecular flexibility"
        else:
            assessment = "Extremely dense - may cause steric clashes"

        _print_verbose(f"\n  {assessment}", 1, state)

    # Store results in state
    max_extent = results['max_molecular_extent_A']
    state.max_molecular_extent = max_extent
    state.volume_based_recommendations = recommendations

    # Section 4: Recommendation (15/20/25)
    rec_15 = recommendations.get('15.0%', {}).get('box_length_A', 0)
    rec_20 = recommendations.get('20.0%', {}).get('box_length_A', 0)
    rec_25 = recommendations.get('25.0%', {}).get('box_length_A', 0)

    _print_verbose("\n4. Recommendation:", 1, state)
    _print_verbose("-" * 48, 1, state)

    if rec_20 > 0:
        _print_verbose(f">>> Default suggestion: {rec_20:.1f} Å (20% packing) <<<", 1, state)
        _print_verbose("", 1, state)
    if rec_15 > 0 and rec_25 > 0:
        _print_verbose(f"  • For isolated clusters: {rec_15:.1f} Å (15% packing)", 1, state)
        _print_verbose(f"  • For cluster formation: {rec_20:.1f} Å (20% packing)", 1, state)
        _print_verbose(f"  • For network studies:   {rec_25:.1f} Å (25% packing)", 1, state)
    _print_verbose("", 1, state)
    _print_verbose(f"  Use --box<P> to use a specific packing %", 1, state)
    _print_verbose(f"  (e.g., --box10) for a 10%", 1, state)

    _print_verbose("\n" + "="*78, 1, state)
    if system_has_hbonds:
        _print_verbose("Note: This analysis accounts for hydrogen bonding networks in molecular clusters.", 1, state)
        _print_verbose("H-bond volume estimated using 2.5 Å average bond length and 1.2 Å interaction radius.", 1, state)
    else:
        _print_verbose("Note: This analysis uses molecular extents to estimate the box size.", 1, state)
        _print_verbose("No primary hydrogen bonding detected; diagonal method (Method B) applied.", 1, state)
    _print_verbose("Run the full simulation to validate these recommendations.", 1, state)
    _print_verbose("="*78, 1, state)


# --- def interactive_directory_selection_with_pattern  (ascec-v04.py 5233-5381) ---
def interactive_directory_selection_with_pattern(input_ext: str, pattern: str = "") -> List[str]:
    """
    Provides interactive directory selection for updating input files with optional pattern filtering.
    Shows only directories that contain matching files and lets user choose.
    
    Args:
        input_ext (str): File extension to search for
        pattern (str): Pattern to filter files (empty string means no filtering)
    
    Returns:
        List[str]: List of selected input file paths
    """
    verbose = True  # Interactive mode always shows detailed output
    
    print("\n" + "=" * 60)
    print("Directory selection".center(60))
    if pattern and pattern.strip():
        print(f"Filtering files containing: '{pattern}'".center(60))
    print("=" * 60)
    
    def filter_files(files, pattern):
        """Filter files by pattern if pattern is provided"""
        if not pattern or not pattern.strip():
            return files
        return [f for f in files if pattern in os.path.basename(f)]
    
    # Scan for directories with matching files, but group similar paths
    directories_with_files = {}
    all_files = []
    
    # Find all matching files recursively
    for root, dirs, files in os.walk("."):
        matching_files_in_dir = []
        for file in files:
            if file.endswith(input_ext):
                full_path = os.path.join(root, file)
                if not pattern or not pattern.strip() or pattern in file:
                    matching_files_in_dir.append(full_path)
                    all_files.append(full_path)
        
        # Only include directories that have matching files
        if matching_files_in_dir:
            # Normalize and group directory paths
            if root == ".":
                dir_display = "Current working directory"
            else:
                dir_display = root
                if dir_display.startswith("./"):
                    dir_display = dir_display[2:]
                dir_display += "/"
            
            # Group similar directories (e.g., merge individual semiempiric/calculation/opt1_conf_XX/ into one)
            if "semiempiric/calculation/" in dir_display and dir_display.count("/") > 2:
                # Group all semiempiric individual calculation directories
                parent_dir = "semiempiric/calculation/ (individual directories)"
                if parent_dir not in directories_with_files:
                    directories_with_files[parent_dir] = []
                directories_with_files[parent_dir].extend(matching_files_in_dir)
            else:
                directories_with_files[dir_display] = matching_files_in_dir
    
    if not directories_with_files:
        if pattern and pattern.strip():
            print(f"No files found containing pattern '{pattern}'.")
        else:
            print("No input files found in any directory.")
        return []
    
    # Display options - only directories with files
    print("\nDirectories with matching files:")
    print("-" * 40 + "\n")
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # Sort directories by name for consistent ordering
    sorted_dirs = sorted(directories_with_files.items(), key=lambda x: (len(x[1]), x[0]))
    
    for dir_name, files in sorted_dirs:
        options[str(option_num)] = (dir_name, files)
        print(f"{option_num}. {dir_name}: {len(files)} files")
        
        # Show first few files as examples
        if len(files) <= 5:
            # Show all files if 5 or fewer
            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"   - {filename}")
        else:
            # Show first 3 files if more than 5
            examples = files[:3]
            for example in examples:
                filename = os.path.basename(example)
                print(f"   - {filename}")
            print(f"   ... and {len(files) - 3} more")
        if verbose:
            print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_files) > 1:
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {len(all_files)} files total")
        if verbose:
            print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            if len(valid_options) > 6:  # If too many options, simplify the prompt
                choice = input(f"Select option (1-{len(valid_options)-2}, 'a' for all, or 'q' to quit): ").strip().lower()
            else:
                # Build a cleaner prompt with explicit 'a' for all description
                option_parts = []
                for opt in valid_options[:-1]:  # Exclude 'q'
                    if opt == 'a':
                        option_parts.append("'a' for all")
                    else:
                        option_parts.append(opt)
                choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Update cancelled.")
                return []
            
            if choice in options:
                selected_files = options[choice][1]
                dir_name = options[choice][0]
                
                print(f"\nSelected: {dir_name}")
                print(f"Files to update: {len(selected_files)}")
                if pattern and pattern.strip():
                    print(f"Pattern filter: '{pattern}'")
                
                return selected_files
            else:
                if len(valid_options) > 6:
                    print(f"Invalid option. Please choose 1-{len(valid_options)-2}, 'a', or 'q' to quit.")
                else:
                    print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nUpdate cancelled by user.")
            return []
        except EOFError:
            print("\nUpdate cancelled.")
            return []


# --- def update_existing_input_files  (ascec-v04.py 5414-5567) ---
def update_existing_input_files(template_file: str, target_pattern: str = "") -> str:
    """
    Updates existing QM input files with a new template, preserving the coordinates 
    and configuration information from the original files.
    
    Args:
        template_file (str): New template file (e.g., new_template.inp)
        target_pattern (str): Search pattern - empty string for all locations, or specific pattern for filtered search
    
    Returns:
        str: Status message
    """
    # Determine QM program from template file extension
    if template_file.endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
    elif template_file.endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
    elif template_file.endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template file exists
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Always show interactive directory selection, but filter by pattern if provided
    input_files = interactive_directory_selection_with_pattern(input_ext, target_pattern)
    if not input_files:
        return "No files selected for update."
    
    # Exclude the template file from the update list
    template_file_abs = os.path.abspath(template_file)
    input_files = [f for f in input_files if os.path.abspath(f) != template_file_abs]
    
    if not input_files:
        return "No input files found for update (excluding template file)."
    
    updated_count = 0
    skipped_count = 0
    backup_files = []  # Track backup files for potential revert
    
    print(f"\nStarting update of {len(input_files)} files...")
    
    for input_file in input_files:
        try:
            # Read existing input file
            with open(input_file, 'r') as f:
                existing_content = f.read()
            
            # Create backup file
            backup_file = input_file + '.backup_temp'
            with open(backup_file, 'w') as f:
                f.write(existing_content)
            backup_files.append((input_file, backup_file))
            
            # Extract configuration information and coordinates from existing file
            config_info = extract_config_from_input_file(existing_content, qm_program)
            if not config_info:
                print(f"Warning: Could not extract configuration from {os.path.basename(input_file)}, skipping.")
                skipped_count += 1
                continue
            
            # Create updated content using the new template
            if create_qm_input_file(config_info, template_content, input_file, qm_program):
                print(f"  Updated: {os.path.basename(input_file)}")
                updated_count += 1
            else:
                print(f"  Failed to update: {os.path.basename(input_file)}")
                skipped_count += 1
                
        except IOError as e:
            print(f"Error processing {os.path.basename(input_file)}: {e}")
            skipped_count += 1
    
    # Show final result and offer revert option
    print(f"\nUpdate completed: {updated_count} files updated, {skipped_count} files skipped.")
    
    if updated_count > 0:
        print("\nPress ENTER to finish and keep changes, or type 'r' and ENTER to revert all changes:")
        try:
            user_choice = input().strip().lower()
            if user_choice == 'r':
                # Revert all changes
                reverted_count = 0
                for original_file, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            # Restore original content
                            with open(backup_file, 'r') as f:
                                original_content = f.read()
                            with open(original_file, 'w') as f:
                                f.write(original_content)
                            reverted_count += 1
                    except IOError as e:
                        print(f"Error reverting {os.path.basename(original_file)}: {e}")
                
                # Clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                
                return f"Reverted {reverted_count} files to original state."
            else:
                # Keep changes, clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except KeyboardInterrupt:
            print("\nKeeping changes (Ctrl+C pressed).")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except EOFError:
            print("\nKeeping changes.")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
    else:
        # No files were updated, just clean up any backup files
        for _, backup_file in backup_files:
            try:
                if os.path.exists(backup_file):
                    os.remove(backup_file)
            except OSError:
                pass
        return f"Update completed: {updated_count} files updated, {skipped_count} files skipped."


# --- def extract_config_from_input_file  (ascec-v04.py 5570-5647) ---
def extract_config_from_input_file(content: str, qm_program: str) -> Optional[Dict]:
    """
    Extracts configuration information from an existing QM input file.
    
    Args:
        content (str): Content of the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        Dict: Configuration data or None if extraction fails
    """
    try:
        lines = content.split('\n')
        
        # Extract comment line (configuration info)
        comment = ""
        for line in lines:
            if line.strip().startswith('#') and ('Configuration:' in line or 'E =' in line):
                comment = line.strip()[1:].strip()  # Remove # and extra spaces
                break
        
        # Extract coordinates
        atoms = []
        
        if qm_program == 'orca':
            # For ORCA, find coordinates between "* xyz 0 1" and "*"
            in_coords = False
            for line in lines:
                if line.strip() == "* xyz 0 1":
                    in_coords = True
                    continue
                elif line.strip() == "*" and in_coords:
                    break
                elif in_coords and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
        
        elif qm_program == 'gaussian':
            # For Gaussian, find coordinates after charge/multiplicity line
            found_charge_mult = False
            for line in lines:
                if not found_charge_mult and line.strip() and len(line.strip().split()) == 2:
                    try:
                        int(line.strip().split()[0])  # charge
                        int(line.strip().split()[1])  # multiplicity
                        found_charge_mult = True
                        continue
                    except ValueError:
                        pass
                elif found_charge_mult and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
                elif found_charge_mult and not line.strip():
                    # Empty line might indicate end of coordinates
                    if atoms:  # Only break if we have found some atoms
                        break
        
        if not atoms:
            return None
        
        return {
            'comment': comment,
            'atoms': atoms
        }
        
    except Exception as e:
        print(f"Error extracting configuration: {e}")
        return None


# --- def create_simple_optimization_system  (ascec-v04.py 6527-6538) ---
def create_simple_optimization_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates an optimization system.
    
    Args:
        template_file (str): Path to the QM input template file
        launcher_template (str): Path to the launcher script template
    
    Returns:
        str: Status message indicating success or failure
    """
    return calculate_input_files(template_file, launcher_template, stage_type="optimization")


# --- def create_refinement_system  (ascec-v04.py 6541-6553) ---
def create_refinement_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates a refinement system by looking for files with 'combined' in their name 
    (like all_motifs_combined.xyz) and motif_*.xyz files.
    
    Args:
        template_file (str): Path to the QM input template file
        launcher_template (str): Path to the launcher script template
    
    Returns:
        str: Status message indicating success or failure
    """
    return calculate_input_files(template_file, launcher_template, stage_type="refinement")


# --- def execute_merge_command  (ascec-v04.py 6666-6820) ---
def execute_merge_command(result_files_only=False):
    """
    Execute the merge command functionality.
    Shows directories with XYZ files and allows user to merge them.
    
    Args:
        result_files_only (bool): If True, only show result_*.xyz files. 
                                   If False, show all XYZ files except _trj, combined_results, and combined_r
    """
    verbose = True  # Interactive mode always shows detailed output
    
    # Set title and file filter based on mode
    if result_files_only:
        title = "Result XYZ Files Merge System"
        file_descriptor = "result XYZ"
    else:
        title = "XYZ Files Merge System"
        file_descriptor = "XYZ"
    
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    
    # Scan for directories with XYZ files based on filter
    directories_with_xyz = {}
    
    # Check current directory
    current_xyz = []
    for file in os.listdir("."):
        if result_files_only:
            if file.startswith("result_") and file.endswith(".xyz"):
                current_xyz.append(file)
        else:
            if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
                not file.startswith("combined_results") and not file.startswith("combined_r")):
                current_xyz.append(file)
    
    if current_xyz:
        directories_with_xyz["."] = current_xyz
    
    # Check subdirectories
    for root, dirs, files in os.walk("."):
        if root == ".":
            continue
            
        xyz_files = []
        for file in files:
            if result_files_only:
                if file.startswith("result_") and file.endswith(".xyz"):
                    xyz_files.append(os.path.join(root, file))
            else:
                if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
                    not file.startswith("combined_results") and not file.startswith("combined_r")):
                    xyz_files.append(os.path.join(root, file))
        
        if xyz_files:
            directories_with_xyz[root] = xyz_files
    
    if not directories_with_xyz:
        if result_files_only:
            print("No result_*.xyz files found in current directory or subdirectories.")
        else:
            print("No .xyz files found in current directory or subdirectories (excluding _trj.xyz, combined_results, and combined_r files).")
        return

    # Display options
    print(f"\nDirectories with {file_descriptor} files:")
    print("-" * 40)
    
    options = {}
    option_num = 1
    
    total_files = 0
    for dir_path, xyz_files in directories_with_xyz.items():
        dir_display = "Current directory" if dir_path == "." else dir_path
        options[str(option_num)] = (dir_path, xyz_files)
        file_count = len(xyz_files)
        total_files += file_count
        
        print(f"{option_num}. {dir_display}: {file_count} files")
        
        # Show first few files as examples
        examples = xyz_files[:3]
        for example in examples:
            filename = os.path.basename(example)
            print(f"   - {filename}")
        if len(xyz_files) > 3:
            print(f"   ... and {len(xyz_files) - 3} more")
        if verbose:
            print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_xyz) > 1:
        all_files = []
        for files in directories_with_xyz.values():
            all_files.extend(files)
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {total_files} files total")
        if verbose:
            print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            choice = input(f"Select option (1-{len(directories_with_xyz)}, 'a' for all, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Merge operation cancelled.")
                return
            
            if choice in options:
                dir_path, files_to_merge = options[choice]
                
                if choice == "a":
                    output_file = "combined_results.xyz"
                    merge_msg = f"result files" if result_files_only else "files"
                    print(f"\nMerging {len(files_to_merge)} {merge_msg} from all directories...")
                else:
                    if dir_path == ".":
                        output_file = "combined_results.xyz"
                        merge_msg = f"result files" if result_files_only else "files"
                        print(f"\nMerging {len(files_to_merge)} {merge_msg} from current directory...")
                    else:
                        dir_name = os.path.basename(dir_path)
                        output_file = f"combined_results_{dir_name}.xyz"
                        merge_msg = f"result files" if result_files_only else "files"
                        print(f"\nMerging {len(files_to_merge)} {merge_msg} from {dir_path}...")
                
                # Perform the merge
                success = merge_xyz_files(files_to_merge, output_file)
                
                if success:
                    print(f"✓ Successfully created {output_file}")
                    file_type = "result XYZ files" if result_files_only else "XYZ files"
                    print(f"  Combined {len(files_to_merge)} {file_type}")
                    
                    # Check if .mol file was also created
                    mol_file = output_file.replace('.xyz', '.mol')
                    if os.path.exists(mol_file):
                        print(f"  Also created {mol_file}")
                else:
                    print(f"✗ Failed to create {output_file}")
                
                return
            else:
                print(f"Invalid option. Please select 1-{len(directories_with_xyz)}, 'a', or 'q'.")
                
        except KeyboardInterrupt:
            print("\nMerge operation cancelled by user.")
            return
        except EOFError:
            print("\nMerge operation cancelled.")
            return


# --- def execute_merge_result_command  (ascec-v04.py 6823-6828) ---
def execute_merge_result_command():
    """
    Execute the merge result command (wrapper for backward compatibility).
    Shows directories with result_*.xyz files and allows user to merge them.
    """
    execute_merge_command(result_files_only=True)


# --- def capture_current_state  (ascec-v04.py 9254-9268) ---
def capture_current_state(directory):
    """Capture the current state of files and folders for potential revert."""
    state = {
        'files': {},  # filename -> full_path
        'folders': set()
    }
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            state['files'][item] = item_path
        elif os.path.isdir(item_path):
            state['folders'].add(item_path)
    
    return state


# --- def execute_summary_only  (ascec-v04.py 9686-9748) ---
def execute_summary_only():
    """Execute only summary creation without sorting files."""
    print("=" * 50)
    print("ASCEC Summary Creation")
    print("=" * 50)
    
    # Check for ORCA files (.out), xTB files (.out), and Gaussian files (.log)
    orca_files = []
    xtb_files = []
    gaussian_files = []

    for root, _, files in os.walk("."):
        for filename in files:
            if filename.endswith(".out"):
                filepath = os.path.join(root, filename)
                detected = detect_output_file_type(filepath)
                if detected == 'xtb':
                    xtb_files.append(filepath)
                else:
                    orca_files.append(filepath)
            elif filename.endswith(".log"):
                gaussian_files.append(os.path.join(root, filename))

    created_summaries = []
    file_types_to_process = []

    if orca_files:
        print(f"\nFound {len(orca_files)} ORCA output files.")
        file_types_to_process.append('orca')

    if xtb_files:
        print(f"\nFound {len(xtb_files)} xTB output files.")
        file_types_to_process.append('xtb')

    if gaussian_files:
        print(f"\nFound {len(gaussian_files)} Gaussian output files.")
        file_types_to_process.append('gaussian')

    if not orca_files and not xtb_files and not gaussian_files:
        print("\nNo ORCA (.out), xTB (.out), or Gaussian (.log) output files found in the current directory or its subfolders.")
        return

    # Create summaries for found file types
    if file_types_to_process:
        print("Creating summaries...")
        num_summaries = summarize_calculations(".", file_types_to_process)

        # Check which summary files were created
        if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
            created_summaries.append("orca_summary.txt")
        if 'xtb' in file_types_to_process and os.path.exists("xtb_summary.txt"):
            created_summaries.append("xtb_summary.txt")
        if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
            created_summaries.append("gaussian_summary.txt")
    
    print("\n" + "=" * 50)
    print("Summary Creation Completed")
    print("=" * 50)
    
    if created_summaries:
        print(f"\nCreated summary files: {', '.join(created_summaries)}")
    else:
        print("\nNo summary files were created (no valid calculation data found).")


# --- def execute_sort_command  (ascec-v04.py 9751-9813) ---
def execute_sort_command(include_summary=True, target_cosmic_folder=None, reuse_existing=False):
    """Execute the complete sort process with option to revert."""
    print("=" * 50)
    print("ASCEC Sort Process Started")
    print("=" * 50)
    
    # Store original state for potential revert
    original_state = capture_current_state(".")
    created_files = []
    created_folders = []
    
    try:
        # Step 1: Sort files by base names
        print("\n1. Sorting files by base names...")
        moved_files = group_files_by_base_with_tracking(".")
        created_folders.extend(moved_files.get('folders', []))
        
        # Step 2: Merge XYZ files
        print("\n2. Merging XYZ files...")
        combined_file = "combined_results.xyz"
        if combine_xyz_files():
            created_files.append(combined_file)
            # Step 3: Create MOL file
            print("\n3. Creating MOL file...")
            mol_file = "combined_results.mol"
            if create_combined_mol():
                created_files.append(mol_file)
        
        # Step 4: Create summary (if requested)
        if include_summary:
            print("\n4. Creating calculation summary...")
            summary_files = create_summary_with_tracking(".")
            created_files.extend(summary_files)
        else:
            print("\n4. Skipping summary creation (--nosum flag used)")
        
        # Step 5: Collect .out files
        print("\n5. Collecting .out files...")
        cosmic_folder = collect_out_files_with_tracking(reuse_existing=reuse_existing, target_cosmic_folder=target_cosmic_folder)
        if cosmic_folder:
            created_folders.append(cosmic_folder)
        
        print("\n" + "=" * 50)
        print("ASCEC Sort Process Completed")
        print("=" * 50)
        
        # Suggest cosmic analysis if .out files were collected
        # Check if any cosmic folder exists at the parent level (accept legacy uppercase too)
        parent_dir = os.path.dirname(os.getcwd())
        cosmic_dirs = []
        for item in os.listdir(parent_dir):
            if item.lower() == "cosmic" or item.lower().startswith("cosmic_"):
                cosmic_path = os.path.join(parent_dir, item)
                if os.path.isdir(cosmic_path):
                    cosmic_dirs.append(item)
            
        if cosmic_dirs:
            print("\nSuggested next step:")
            print("  python cosmic")
            print("  Run cosmic analysis on collected output files")

    except Exception as e:
        print(f"\nError during sort process: {e}")


# --- def execute_cosmic_analysis  (ascec-v04.py 9815-9850) ---
def execute_cosmic_analysis(*args):
    """Execute cosmic analysis by calling the cosmic script."""
    import subprocess
    import sys
    
    # Locate cosmic-v01.py via the shared finder: __file__ now lives three
    # levels deep in cosmic_ascec/workflow/, so a dirname(__file__) lookup
    # misses the repo root where cosmic-v01.py actually sits.
    cosmic_script = find_cosmic_script()

    if not cosmic_script:
        print("Error: cosmic-v01.py not found.")
        print("Make sure cosmic-v01.py is in the same directory as ascec-v04.py")
        return
    
    # Build command
    cmd = [sys.executable, cosmic_script] + list(args)
    # If user provides --th/--threshold, pass it through; otherwise
    # cosmic uses statistical consensus cutting (no threshold needed).

    print("=" * 50)
    print("ASCEC COSMIC Analysis")
    print("=" * 50)
    print(f"Executing: {' '.join(cmd[1:])}")  # Don't show python path
    print()
    
    try:
        # Execute the cosmic script with all arguments
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("COSMIC analysis completed successfully.")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing cosmic analysis: {e}")
        print("Check the cosmic script arguments and try again.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


# --- def execute_diagram_generation  (ascec-v04.py 9853-9934) ---
def execute_diagram_generation(scaled: bool = False):
    """
    Generate or regenerate all annealing diagrams from tvse_*.dat files.
    Searches for tvse_*.dat files in current directory and subdirectories,
    creates individual diagrams, and generates combined replica diagrams.
    
    Args:
        scaled: If True, apply intelligent y-axis scaling to remove initial high-energy clutter
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        print("Error: matplotlib is not available")
        print("Install with: pip install matplotlib")
        return
    
    print("=" * 60)
    print("ASCEC Diagram Generation")
    if scaled:
        print("(Scaled mode: intelligent y-axis scaling enabled)")
    print("=" * 60)
    
    # Find all tvse_*.dat files
    tvse_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('tvse_') and file.endswith('.dat'):
                tvse_files.append(os.path.join(root, file))
    
    if not tvse_files:
        print("No tvse_*.dat files found in current directory or subdirectories")
        return
    
    print(f"Found {len(tvse_files)} tvse file(s)")
    print()
    
    # Generate individual diagrams
    diagrams_generated = 0
    for tvse_file in tvse_files:
        output_dir = os.path.dirname(tvse_file)
        filename = os.path.basename(tvse_file)
        seed = filename.replace('tvse_', '').replace('.dat', '')
        
        print(f"  {filename}...", end=" ", flush=True)
        
        if plot_annealing_diagrams(tvse_file, output_dir, scaled=scaled):
            print(f"✓")
            diagrams_generated += 1
        else:
            print(f"✗ Failed")
    
    print(f"\nGenerated {diagrams_generated} individual diagram(s)")
    
    # Check for replica groups (multiple tvse files in same parent directory)
    # Group by parent directory
    parent_dirs = {}
    for tvse_file in tvse_files:
        parent = os.path.dirname(os.path.dirname(tvse_file))
        if parent not in parent_dirs:
            parent_dirs[parent] = []
        parent_dirs[parent].append(tvse_file)
    
    # Generate combined diagrams for parent directories with multiple replicas
    combined_generated = 0
    for parent_dir, files in parent_dirs.items():
        if len(files) > 1:
            # This looks like a replica group
            num_replicas = len(files)
            output_file = os.path.join(parent_dir, f"tvse_r{num_replicas}.png")
            
            print(f"\nGenerating combined diagram for {num_replicas} replica(s)...")
            
            if plot_combined_replicas_diagram(files, output_file, num_replicas):
                print(f"  ✓ Created: {os.path.basename(output_file)}")
                combined_generated += 1
            else:
                print(f"  ✗ Failed to create combined diagram")
    
    if combined_generated > 0:
        print(f"\nGenerated {combined_generated} combined diagram(s)")
    
    print("\n" + "=" * 60)
    print("Diagram generation completed")
    print("=" * 60)


# --- def execute_box_analysis  (ascec-v04.py 9937-9965) ---
def execute_box_analysis(input_file: str):
    """
    Analyze an input file and provide box length recommendations without running the simulation.
    This reads the input file, parses the molecular structure, and outputs the volume-based 
    box length analysis to help users choose appropriate box sizes.
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Create a minimal SystemState for parsing
        state = SystemState()
        state.verbosity_level = 1  # Enable verbose output for box analysis
        
        # Parse the input file
        read_input_file(state, input_file)
        state.input_file_path = input_file

        # Provide box length analysis using the existing function
        provide_box_length_advice(state)
        
        # Exit successfully after box analysis
        return
        
    except Exception as e:
        print(f"Error during box analysis: {e}")
        sys.exit(1)


# --- def show_ascec_status  (ascec-v04.py 18676-19252) ---
def show_ascec_status() -> None:
    """Interactive ASCEC job status viewer."""
    import signal as _sig

    def _collect_descendant_pids(root_pid: int) -> List[int]:
        """Collect root PID and all descendants using /proc parent links (Linux)."""
        if root_pid <= 0:
            return []
        children: Dict[int, List[int]] = {}
        try:
            for entry in os.listdir('/proc'):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                try:
                    with open(f"/proc/{pid}/stat", 'r') as sf:
                        stat_line = sf.read().strip()
                    # Format: pid (comm) state ppid ... ; split after final ') '
                    tail = stat_line.rsplit(') ', 1)[1].split()
                    ppid = int(tail[1])
                    children.setdefault(ppid, []).append(pid)
                except Exception:
                    continue
        except Exception:
            return [root_pid]

        out = []
        seen = set()
        queue = [root_pid]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            queue.extend(children.get(cur, []))
        return out

    def _is_non_zombie_pid_alive(pid: int) -> bool:
        """Return True only when PID exists and is not a zombie."""
        if not _is_pid_alive(pid):
            return False
        try:
            with open(f"/proc/{pid}/stat", 'r') as sf:
                stat_line = sf.read().strip()
            tail = stat_line.rsplit(') ', 1)[1].split()
            state = tail[0]
            return state != 'Z'
        except Exception:
            return _is_pid_alive(pid)

    def _collect_qm_related_pids(working_dir: str, input_file: str) -> List[int]:
        """Find likely ORCA/QM worker processes for this job (Linux /proc).

        Match by executable/cmdline keywords plus cwd rooted at job working dir.
        """
        if not working_dir:
            return []
        try:
            wd_real = os.path.realpath(working_dir)
        except Exception:
            wd_real = working_dir

        input_base = os.path.splitext(os.path.basename(input_file or ""))[0].lower()
        qm_terms = (
            'orca', 'xtb', 'crest', 'g16', 'gaussian', 'qchem', 'nwchem',
            'psi4', 'cp2k', 'mopac', 'molpro', 'turbomole'
        )

        found: List[int] = []
        try:
            for entry in os.listdir('/proc'):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                try:
                    cwd = os.path.realpath(os.readlink(f'/proc/{pid}/cwd'))
                except Exception:
                    continue

                if not (cwd == wd_real or cwd.startswith(wd_real + os.sep)):
                    continue

                try:
                    with open(f'/proc/{pid}/cmdline', 'rb') as cf:
                        raw = cf.read().replace(b'\x00', b' ').strip().lower()
                    cmd = raw.decode('utf-8', errors='ignore')
                except Exception:
                    cmd = ""

                if any(t in cmd for t in qm_terms) or (input_base and input_base in cmd):
                    found.append(pid)
                    continue

                # Fallback to executable name when cmdline is sparse.
                try:
                    exe = os.path.basename(os.readlink(f'/proc/{pid}/exe')).lower()
                except Exception:
                    exe = ""
                if any(t in exe for t in qm_terms):
                    found.append(pid)
        except Exception:
            return []

        return sorted(set(found))

    def _prepare_ui_jobs(raw_jobs: list) -> list:
        """Return jobs ordered for display with user-facing IDs.

        Running jobs are listed first, then history. UI IDs are reassigned from 1.
        Original database row ID is preserved in 'db_id'.
        """
        _live = ('running', 'holding')
        running = [j for j in raw_jobs if j.get('status') in _live]
        done = [j for j in raw_jobs if j.get('status') not in _live]
        running.sort(key=lambda j: j.get('id', 0))
        # Show history newest-first.
        done.sort(key=lambda j: j.get('id', 0), reverse=True)
        ordered = running + done

        ui_jobs = []
        for idx, j in enumerate(ordered, start=1):
            item = dict(j)
            item['db_id'] = j.get('id', 0)
            item['id'] = idx
            ui_jobs.append(item)
        return ui_jobs

    def _render_bar(pct: float, width: int = 30) -> str:
        filled = int(min(pct / 100.0, 1.0) * width)
        return "█" * filled + "░" * (width - filled)

    def _tail_lines(path: str, n: int = 10) -> List[str]:
        try:
            with open(path) as f:
                return f.readlines()[-n:]
        except Exception:
            return []

    def _resolve_run_realpath(job: dict) -> str:
        """Return best-effort realpath for the run input file."""
        raw_input = str(job.get('input_file') or '').strip()
        if not raw_input:
            return "(unknown)"
        try:
            return os.path.realpath(raw_input)
        except Exception:
            return os.path.abspath(raw_input)

    def _display_menu(jobs: list, show_paths: bool = False) -> None:
        os.system('clear')
        now_str = time.strftime('%Y-%m-%d %H:%M:%S')

        def _fit(text: str, width: int) -> str:
            if width <= 0:
                return ""
            if len(text) <= width:
                return text
            if width <= 3:
                return text[:width]
            return text[:width - 3] + "..."

        try:
            term_w = os.get_terminal_size().columns
        except OSError:
            term_w = 100
        sep_w = max(78, min(term_w, 140))
        dash_sep = "  " + "-" * (sep_w - 2)

        run_input_w = max(18, min(40, sep_w - 40))
        hist_input_w = max(18, min(40, sep_w - 34))

        print(f"{'─'*sep_w}")
        print(f"  ASCEC STATUS  ({now_str})")
        print(f"{'─'*sep_w}")
        _live = ('running', 'holding')
        running = [j for j in jobs if j['status'] in _live]
        done = [j for j in jobs if j['status'] not in _live]
        if running:
            print(f"\n  Running:")
            print(f"    {'ID':>3}  {'PID':>7}  {'INPUT FILE':<{run_input_w}}  STARTED")
            print(f"    {'─'*3}  {'─'*7}  {'─'*run_input_w}  {'─'*19}")
            for j in running:
                fname = os.path.basename(j['input_file'])
                print(f"    {j['id']:>3}  {j['pid']:>7}  {_fit(fname, run_input_w):<{run_input_w}}  {j['started_at']}")
                if j['status'] == 'holding':
                    _wait_pid = j.get('after_pid') or 0
                    _wait_txt = f" PID {_wait_pid}" if _wait_pid else ""
                    print(f"         ⏸  HOLDING — waiting for{_wait_txt} to finish")
                if show_paths:
                    run_realpath = _resolve_run_realpath(j)
                    print(f"         path: {run_realpath}")
        else:
            print("\n  No running jobs.")

        # Clear visual separation between live jobs and history.
        print(f"\n{dash_sep}")

        if done:
            print(f"\n  History runs:")
            print()
            print(f"    {'ID':>3}  {'STATUS':<10}  {'INPUT FILE':<{hist_input_w}}  UPDATED")
            print(f"    {'─'*3}  {'─'*10}  {'─'*hist_input_w}  {'─'*19}")
            for j in done:
                fname = os.path.basename(j['input_file'])
                print(f"    {j['id']:>3}  {j['status']:<10}  {_fit(fname, hist_input_w):<{hist_input_w}}  {j['updated_at']}")
                if show_paths:
                    run_realpath = _resolve_run_realpath(j)
                    print(f"         path: {run_realpath}")

        print(f"\n{dash_sep}")
        print(f"{'─'*sep_w}")
        info_state = "ON" if show_paths else "OFF"
        print(f"  [V <id>] Attach/View   [K <id>] Kill   [I] Info Paths ({info_state})   [R] Refresh   [Q] Quit")
        print(f"{'─'*sep_w}")

    def _show_progress_screen(job: dict, data: dict) -> None:
        def _hydrate_live_stage_lines(_job: dict, _data: dict) -> Tuple[List[str], Optional[float]]:
            """Hydrate active stage n/N from protocol cache when available."""
            lines_in = list(_data.get('stage_lines', []))
            try:
                cache_file = str(_job.get('cache_file') or '').strip()
                if not cache_file or not os.path.exists(cache_file):
                    return lines_in, None

                cache = load_protocol_cache(cache_file)
                if not isinstance(cache, dict):
                    return lines_in, None

                stages_data = cache.get('stages', {})
                if not isinstance(stages_data, dict):
                    return lines_in, None

                stage_num = int(_data.get('current_stage_num', 0) or 0)
                stages_total = int(_data.get('stages_total', 0) or 0)
                completed_stages = int(_data.get('stages_completed', 0) or 0)
                if stage_num <= 0 or stages_total <= 0:
                    return lines_in, None

                stage_key = None
                stage_info = None
                for k, v in stages_data.items():
                    if isinstance(k, str) and k.endswith(f"_{stage_num}"):
                        stage_key = k
                        stage_info = v if isinstance(v, dict) else None
                        break
                if not stage_key or not isinstance(stage_info, dict):
                    return lines_in, None
                if stage_info.get('status') != 'in_progress':
                    return lines_in, None

                result = stage_info.get('result', {})
                if not isinstance(result, dict):
                    return lines_in, None

                done = result.get('num_completed', result.get('completed'))
                total = result.get('total_files', result.get('total'))
                if done is None or total is None:
                    return lines_in, None

                done_i = int(done)
                total_i = int(total)
                if total_i <= 0:
                    return lines_in, None

                stage_type = re.sub(r'_\d+$', '', stage_key)
                stage_name_map = {
                    'replication': 'annealing',
                    'optimization': 'geometry_optimization',
                    'cosmic': 'cosmic',
                    'refinement': 'geometry_refinement',
                    'energy_refinement': 'energy_refinement',
                }
                stage_name = stage_name_map.get(stage_type, stage_type)
                # Surface an active redo (e.g. "(Redo 1/3)") when the stage is being
                # re-run. The cache carries current_redo/redo_max only for cached
                # runs; the progress file always does, so fall back to it.
                _redo_n = result.get('current_redo')
                _redo_max = result.get('redo_max')
                if not _redo_max and int(_data.get('redo_stage_num', 0) or 0) == stage_num:
                    _redo_n = _data.get('current_redo')
                    _redo_max = _data.get('redo_max')
                # redo_max drives the tag, not redo_n: attempt 0 must still render
                # "(Redo 0/N)". A stage without --redo= has no redo_max and no tag.
                _redo_tag = f" (Redo {int(_redo_n or 0)}/{_redo_max})" if _redo_max else ""
                replacement = f"[{stage_num}/{stages_total}] {stage_name} {done_i}/{total_i} ...{_redo_tag}"

                prefix = f"[{stage_num}/{stages_total}] "
                replaced = False
                for idx, line in enumerate(lines_in):
                    if line.startswith(prefix):
                        lines_in[idx] = replacement
                        replaced = True
                        break
                if not replaced:
                    lines_in.append(replacement)

                ratio = min(max(done_i / total_i, 0.0), 1.0)
                pct = ((completed_stages + ratio) / stages_total) * 100.0
                return lines_in, pct
            except Exception:
                return lines_in, None

        os.system('clear')
        jid = job['id']
        run_realpath = _resolve_run_realpath(job)
        hydrated_lines, hydrated_pct = _hydrate_live_stage_lines(job, data)
        pct = hydrated_pct if hydrated_pct is not None else data.get('pct', 0.0)
        bar = _render_bar(pct, width=30)
        upd = data.get('updated', '')
        print("")
        print("=== COSMIC ASCEC ===")
        print("-" * 60)
        print(f"Progress [{bar}] {pct:.1f}%")
        print("-" * 60)
        for line in hydrated_lines:
            print(line)
        print("")
        if upd:
            # The panel is only rewritten when a calculation finishes, so on a
            # long QM job 'updated' can legitimately be hours old. Show the age
            # so a stalled-looking screen is not mistaken for a stalled run.
            _age = ""
            try:
                _delta = time.time() - time.mktime(time.strptime(upd, '%Y-%m-%d %H:%M:%S'))
                if _delta >= 60:
                    _h, _m = int(_delta // 3600), int((_delta % 3600) // 60)
                    _age = f" ({_h}h {_m}m ago)" if _h else f" ({_m}m ago)"
            except Exception:
                pass
            print(f"Attached: job {jid} ({os.path.basename(job['input_file'])})   updated: {upd}{_age}")
        else:
            print(f"Attached: job {jid} ({os.path.basename(job['input_file'])})")
        print("Ctrl+C or Ctrl+D to detach (job keeps running)")
        print(f"Input realpath: {run_realpath}")

    def _attach_view(job: dict) -> None:
        prog_file = job['progress_file']
        last_data: Dict[str, Any] = {}
        last_render_key: Optional[tuple] = None  # (pct, stage_lines_tuple, updated)

        # Try to set up raw terminal input so single-keypress 'D' works
        _old_settings = None
        _has_raw = False
        _select = None
        try:
            import termios as _termios, tty as _tty, select as _select_mod
            _select = _select_mod
            if sys.stdin.isatty():
                _old_settings = _termios.tcgetattr(sys.stdin)
                _tty.setcbreak(sys.stdin.fileno())
                _has_raw = True
        except (ImportError, OSError):
            _has_raw = False

        def _show_connecting() -> None:
            os.system('clear')
            print("")
            print("=== COSMIC ASCEC ===")
            print("-" * 60)
            print(f"Connecting to job {job['id']} ({os.path.basename(job['input_file'])})...")
            print("-" * 60)
            print("  Waiting for progress data — job may still be initializing.")
            print("")
            print("D or Ctrl+C to detach (job keeps running)")

        _connecting_shown = False

        try:
            while True:
                if not _is_pid_alive(job['pid']):
                    os.system('clear')
                    print(f"\n  Job {job['id']} has finished.")
                    input("  Press Enter to return to menu...")
                    return

                # Always attempt to read the file; compare content to avoid redundant redraws.
                # This avoids the mtime-stuck bug where last_mtime is set before a successful
                # json.load, causing the viewer to miss updates when the file is briefly empty
                # during truncate+write (especially on filesystems with coarse mtime resolution).
                try:
                    with open(prog_file) as _pf:
                        new_data = json.load(_pf)
                    render_key = (
                        new_data.get('pct'),
                        tuple(new_data.get('stage_lines', [])),
                        new_data.get('updated'),
                    )
                    if render_key != last_render_key:
                        last_data = new_data
                        last_render_key = render_key
                        _connecting_shown = False
                        _show_progress_screen(job, last_data)
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    if not _connecting_shown:
                        _show_connecting()
                        _connecting_shown = True

                # Non-blocking check for 'D' keypress
                if _has_raw and _select is not None:
                    try:
                        if _select.select([sys.stdin], [], [], 0.5)[0]:
                            ch = sys.stdin.read(1)
                            if ch in ('', '\x04') or ch.lower() == 'd':
                                raise KeyboardInterrupt
                    except (IOError, OSError):
                        time.sleep(0.5)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings
            if _old_settings is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(sys.stdin, _termios.TCSADRAIN, _old_settings)
                except Exception:
                    pass

        print(f"\n  Detached — job {job['id']} running in background.")
        print(f"  Safe to close terminal (SIGHUP is ignored by the running process).")
        time.sleep(0.8)

    def _view_completed(job: dict) -> None:
        os.system('clear')
        run_realpath = _resolve_run_realpath(job)
        print(f"{'─'*62}")
        print(f"  Job {job['id']}  —  {job['status']}  ({os.path.basename(job['input_file'])})")
        print(f"{'─'*62}\n")

        lines = _tail_lines(job['log_file'], 200)
        if lines:
            clean_lines = [re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', ln.rstrip()) for ln in lines]

            # Show only the final progress snapshot, not the full stream of redraws.
            start_idx = None
            for idx in range(len(clean_lines) - 1, -1, -1):
                if clean_lines[idx].strip() == '=== COSMIC ASCEC ===':
                    start_idx = idx
                    break

            if start_idx is not None:
                snapshot = clean_lines[start_idx:]
                for ln in snapshot:
                    txt = ln.strip()
                    if not txt:
                        print()
                        continue
                    print(f"  {txt}")
            else:
                # Fallback: show only the last few meaningful lines.
                fallback = [ln.strip() for ln in clean_lines if ln.strip()][-12:]
                for ln in fallback:
                    print(f"  {ln}")
        else:
            print("  (no log available)")
        print(f"  Input realpath: {run_realpath}")
        print(f"\n{'─'*62}")
        input("  Press Enter to return to menu...")

    show_info_paths = False
    while True:
        jobs = _prepare_ui_jobs(_get_recent_jobs())
        _display_menu(jobs, show_paths=show_info_paths)
        try:
            raw = input("  Choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not raw:
            continue
        if raw.lower() == 'q':
            break
        if raw.lower() == 'r':
            continue
        if raw.lower() == 'i':
            show_info_paths = not show_info_paths
            continue
        m = re.match(r'^\s*([VKvk])\s*(\d+)\s*$', raw)
        if not m:
            continue
        action = m.group(1).upper()
        target_id = int(m.group(2))
        job = next((j for j in jobs if j['id'] == target_id), None)
        if not job:
            print(f"  No job with id {target_id}")
            time.sleep(1)
            continue
        if action == 'V':
            if job['status'] == 'holding':
                _wait_pid = job.get('after_pid') or 0
                print(f"  Job {job['id']} is HOLDING"
                      + (f" — waiting for PID {_wait_pid} to finish." if _wait_pid else "."))
                print("  It will start automatically once the other run ends. [K] to cancel.")
                time.sleep(2)
            elif job['status'] == 'running':
                _attach_view(job)
            else:
                _view_completed(job)
        elif action == 'K':
            if job['status'] in ('running', 'holding') and _is_pid_alive(job['pid']):
                try:
                    _self_pid = os.getpid()
                    _self_pgid = os.getpgrp()
                    _root_pid = int(job['pid'])
                    _tree = [p for p in _collect_descendant_pids(_root_pid) if p != _self_pid]

                    if job['status'] == 'holding':
                        # A HOLDING job is only a waiting placeholder process; it
                        # has no QM workers of its own yet. Any ORCA/QM processes
                        # sharing this input_file/working_dir belong to the job it
                        # is queued behind (``after_pid``). Discovering them here
                        # would kill that still-running job — so skip QM discovery
                        # and, as a safety belt, exclude the held job's PID and its
                        # descendants from the kill set.
                        _after_pid = int(job.get('after_pid') or 0)
                        _protected = set()
                        if _after_pid > 0:
                            _protected.update(_collect_descendant_pids(_after_pid))
                            _protected.update(_collect_qm_related_pids(
                                str(job.get('working_dir', '') or ''),
                                str(job.get('input_file', '') or ''),
                            ))
                        _kill_set = (set([_root_pid] + _tree) - _protected)
                        _kill_set.add(_root_pid)  # never drop the placeholder itself
                        _tree = sorted(_kill_set)
                    else:
                        # Include likely ORCA/QM workers that may have detached from parent tree.
                        _qm_pids = [
                            p for p in _collect_qm_related_pids(
                                str(job.get('working_dir', '') or ''),
                                str(job.get('input_file', '') or ''),
                            )
                            if p != _self_pid
                        ]
                        _tree = sorted(set([_root_pid] + _tree + _qm_pids))

                    # Also target process groups of discovered descendants.
                    _pgids = set()

                    # Always include the main job process group when safe.
                    try:
                        _root_pgid = os.getpgid(_root_pid)
                        if _root_pgid > 0 and _root_pgid != _self_pgid:
                            _pgids.add(_root_pgid)
                    except OSError:
                        pass

                    for _p in _tree:
                        try:
                            _pg = os.getpgid(_p)
                            if _pg > 0 and _pg != _self_pgid:
                                _pgids.add(_pg)
                        except OSError:
                            pass

                    # Graceful termination first.
                    for _pg in sorted(_pgids):
                        try:
                            os.killpg(_pg, _sig.SIGTERM)
                        except OSError:
                            pass

                    # Always signal root PID directly as anchor.
                    try:
                        os.kill(_root_pid, _sig.SIGTERM)
                    except OSError:
                        pass

                    for _p in _tree:
                        try:
                            os.kill(_p, _sig.SIGTERM)
                        except OSError:
                            pass
                    print(f"  Sent SIGTERM to {len(_tree)} process(es) across {len(_pgids)} group(s).")

                    # Wait for shutdown.
                    for _ in range(20):
                        if not any(_is_non_zombie_pid_alive(p) for p in _tree):
                            break
                        time.sleep(0.1)

                    # Escalate remaining to SIGKILL.
                    _alive = [p for p in _tree if _is_non_zombie_pid_alive(p)]
                    if _alive:
                        _alive_pgids = set()
                        for _p in _alive:
                            try:
                                _pg = os.getpgid(_p)
                                if _pg > 0 and _pg != _self_pgid:
                                    _alive_pgids.add(_pg)
                            except OSError:
                                pass
                        for _pg in sorted(_alive_pgids):
                            try:
                                os.killpg(_pg, _sig.SIGKILL)
                            except OSError:
                                pass

                        # Always hard-kill root PID too if still around.
                        try:
                            if _is_non_zombie_pid_alive(_root_pid):
                                os.kill(_root_pid, _sig.SIGKILL)
                        except OSError:
                            pass

                        for _p in _alive:
                            try:
                                os.kill(_p, _sig.SIGKILL)
                            except OSError:
                                pass
                        print(f"  Escalated SIGKILL to {len(_alive)} remaining process(es).")

                        for _ in range(20):
                            if not any(_is_non_zombie_pid_alive(p) for p in _alive):
                                break
                            time.sleep(0.1)

                    _remaining = [p for p in _tree if _is_non_zombie_pid_alive(p)]
                    if _remaining:
                        print(f"  Warning: {len(_remaining)} process(es) still alive: {_remaining[:6]}")
                        print("  Try running kill again or terminate with elevated privileges.")
                    else:
                        _remove_progress_artifacts(job.get('progress_file', ''))
                        _update_ascec_job(job.get('db_id', 0), 'killed')
                        print(f"  Job {job['id']} marked as killed.")
                except OSError as e:
                    print(f"  Could not kill PID {job['pid']}: {e}")
            else:
                print(f"  Job {target_id} is not running.")
            time.sleep(1)

