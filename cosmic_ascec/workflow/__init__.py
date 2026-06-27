"""Workflow orchestration — the COSMIC ASCEC protocol from a single ``.asc``.

The ``.asc`` input can carry an embedded "protocol" line (e.g.
``.asc, r1 --box30, opt --concurrent=8, cosmic -j4``) that describes a
multi-stage pipeline: annealing → replication → optimization → clustering →
energy refinement, with checkpointing between stages. This package owns the
parsing, sequencing, persistence, and cross-stage helpers for that pipeline.

Submodule responsibilities:

* :mod:`~cosmic_ascec.workflow.protocol` — parse the embedded protocol line
  into a list of stage descriptors.
* :mod:`~cosmic_ascec.workflow.context` — the shared mutable
  :class:`~cosmic_ascec.workflow.context.WorkflowContext` threaded through
  every stage.
* :mod:`~cosmic_ascec.workflow.protocol_cache` — the ``protocol_*.pkl``
  resume cache that lets ``ascec <file> protocol`` pick up where a previous
  run left off.
* :mod:`~cosmic_ascec.workflow.job_registry` — the SQLite ``jobs.db``
  machine-wide job registry powering ``ascec status``.
* :mod:`~cosmic_ascec.workflow.rescue` — imaginary-frequency detection and
  normal-mode displacement for the Hessian-rescue protocol.
* :mod:`~cosmic_ascec.workflow.stages` — the per-stage runners, the
  orchestrator ``execute_workflow_stages``, and the large body of
  cross-stage helpers (file discovery, QM input generation, plotting, ...).
* :mod:`~cosmic_ascec.workflow.replicas` — replicated-annealing creation,
  launcher merging, and the concurrent replica aggregator.
"""

from cosmic_ascec.workflow.context import WorkflowContext
from cosmic_ascec.workflow.job_registry import (
    _adopt_ascec_job,
    _ascec_db_path,
    _ascec_state_dir,
    _atomic_claim_ascec_job,
    _cleanup_stale_jobs,
    _get_recent_jobs,
    _init_ascec_db,
    _is_pid_alive,
    _kill_orphaned_job_processes,
    _pdeathsig_preexec,
    _register_ascec_job,
    _remove_progress_artifacts,
    _set_job_holding,
    _update_ascec_job,
)
from cosmic_ascec.workflow.protocol import (
    contains_workflow_separator,
    finalize_stage,
    parse_workflow_stages,
)
from cosmic_ascec.workflow.protocol_cache import (
    invalidate_stage_cache,
    load_protocol_cache,
    save_protocol_cache,
    update_protocol_cache,
)
from cosmic_ascec.workflow.replicas import (
    PROTOCOL_MARKER_RE,
    create_launcher_script,
    create_replicated_runs,
    is_protocol_marker_line,
    merge_launcher_scripts,
    strip_protocol_from_content,
    update_box_size_in_input,
)
from cosmic_ascec.workflow.rescue import (
    count_imaginary_frequencies,
    displace_along_imaginary_mode,
    displace_gaussian_imaginary_mode,
    displace_orca_imaginary_mode,
    extract_displaced_frame,
    extract_final_geometry,
    format_ordinal,
    handle_imaginary_frequencies,
)
from cosmic_ascec.workflow.stages import (
    ASCEC_VERSION,
    B2,
    capture_current_state,
    check_orca_terminated_normally_opi,
    check_qm_output_completed,
    check_workflow_pause,
    consume_protocol_maxprint_flag,
    create_box_xyz_copy,
    create_qm_input_file,
    create_refinement_system,
    create_simple_optimization_system,
    create_xyz_input_file,
    detect_convergence_status,
    enable_hessian_restart,
    execute_box_analysis,
    execute_cosmic_analysis,
    execute_cosmic_stage,
    execute_diagram_generation,
    execute_energy_refinement_stage,
    execute_merge_command,
    execute_merge_result_command,
    execute_optimization_stage,
    execute_refinement_stage,
    execute_replication_stage,
    execute_sort_command,
    execute_summary_only,
    execute_workflow_stages,
    extract_config_from_input_file,
    extract_protocol_from_input,
    find_cosmic_script,
    find_out_file_in_subdirs,
    generate_protocol_summary,
    generate_rescue_hessian_input,
    get_box_size_recommendation,
    get_critical_count,
    get_critical_files_list,
    get_molecular_formula,
    interactive_directory_selection_with_pattern,
    natural_sort_key,
    parse_cosmic_output,
    parse_cosmic_percentages,
    parse_cosmic_summary,
    parse_exclusion_pattern,
    parse_rescue_method,
    parse_verbosity_level,
    plot_annealing_diagrams,
    plot_combined_replicas_diagram,
    print_version_banner,
    process_optimization_redo,
    process_redo_structures,
    provide_box_length_advice,
    run_rescue_hessian_calculation,
    show_ascec_status,
    update_existing_input_files,
    validate_cached_optimization_cosmic,
    validate_cached_refinement_cosmic,
)

__all__ = [
    # context
    "WorkflowContext",
    # protocol grammar
    "contains_workflow_separator",
    "parse_workflow_stages",
    "finalize_stage",
    # protocol_*.pkl cache
    "load_protocol_cache",
    "save_protocol_cache",
    "update_protocol_cache",
    "invalidate_stage_cache",
    # SQLite jobs.db registry
    "_ascec_state_dir",
    "_ascec_db_path",
    "_init_ascec_db",
    "_is_pid_alive",
    "_pdeathsig_preexec",
    "_register_ascec_job",
    "_update_ascec_job",
    "_set_job_holding",
    "_remove_progress_artifacts",
    "_adopt_ascec_job",
    "_atomic_claim_ascec_job",
    "_kill_orphaned_job_processes",
    "_cleanup_stale_jobs",
    "_get_recent_jobs",
    # rescue helpers
    "handle_imaginary_frequencies",
    "count_imaginary_frequencies",
    "displace_along_imaginary_mode",
    "displace_orca_imaginary_mode",
    "displace_gaussian_imaginary_mode",
    "extract_displaced_frame",
    "format_ordinal",
    "extract_final_geometry",
    # cross-stage helpers
    "find_cosmic_script",
    "parse_cosmic_percentages",
    "parse_cosmic_summary",
    "parse_cosmic_output",
    "get_critical_count",
    "get_critical_files_list",
    "check_workflow_pause",
    "validate_cached_optimization_cosmic",
    "validate_cached_refinement_cosmic",
    # workflow orchestrator + QM-concurrency machinery
    "execute_workflow_stages",
    "execute_replication_stage",
    "find_out_file_in_subdirs",
    "process_redo_structures",
    "process_optimization_redo",
    "check_qm_output_completed",
    "natural_sort_key",
    "parse_verbosity_level",
    # stage runners
    "execute_optimization_stage",
    "execute_cosmic_stage",
    "execute_refinement_stage",
    "execute_energy_refinement_stage",
    # Hessian-rescue protocol
    "parse_rescue_method",
    "generate_rescue_hessian_input",
    "enable_hessian_restart",
    "run_rescue_hessian_calculation",
    # cross-stage helpers
    "generate_protocol_summary",
    "get_box_size_recommendation",
    "detect_convergence_status",
    "check_orca_terminated_normally_opi",
    "create_qm_input_file",
    "create_xyz_input_file",
    "plot_annealing_diagrams",
    "plot_combined_replicas_diagram",
    # replicas
    "PROTOCOL_MARKER_RE",
    "is_protocol_marker_line",
    "strip_protocol_from_content",
    "update_box_size_in_input",
    "create_launcher_script",
    "merge_launcher_scripts",
    "create_replicated_runs",
    # CLI helper surface
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
