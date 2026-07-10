"""WorkflowContext — the mutable state object threaded through the stages.

A multi-stage protocol run is a long sequence of independent stages
(annealing, replication, optimization, clustering, refinement) that need to
share state — temperatures, accepted-config counters, file lists, verbosity,
cosmic stage counts, exclusion patterns, etc. Rather than passing ~60
parameters between functions, every stage receives a single
:class:`WorkflowContext` and reads/writes the fields it cares about.

The object is intentionally mutable and the field set is intentionally wide.
Stages cooperate by leaving information in the context for later stages to
find; the three accessor methods at the bottom delegate to the
``protocol_*.pkl`` resume cache so a resumed run can recover output
directories from earlier stages.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional

from cosmic_ascec.workflow.protocol_cache import load_protocol_cache

__all__ = ["WorkflowContext"]


@dataclasses.dataclass
class WorkflowContext:
    """Context object to pass information between workflow stages."""
    input_file: str = ""
    num_replicas: int = 0
    annealing_dirs: List[str] = dataclasses.field(default_factory=list)
    optimization_stage_dir: str = ""
    cosmic_dir: str = ""
    refinement_stage_dir: str = ""
    critical_count: int = 0
    skipped_count: int = 0
    total_structures: int = 0
    current_try: int = 1
    max_tries: int = 3
    optimization_stage_number: int = 0  # Track which optimization/refinement cycle stage (1 or 2)
    cache_file: str = ""  # Protocol-specific cache filename
    current_stage_key: str = ""  # Current stage key (e.g., "optimization_2")
    cosmic_args: List[str] = dataclasses.field(default_factory=list)  # Store all cosmic args
    is_workflow: bool = False  # True when running in workflow mode (with , or then separators)
    workflow_verbose: bool = False  # True only when workflow should print detailed stage logs
    workflow_verbose_level: int = 0  # 0: silent, 1: -v, 2: -v2, 3: -v3
    max_launch_retries: int = 10  # Hardcoded retry attempts for launch failures only (instant crashes)
    ascec_parallel_cores: int = 0  # Number of cores for parallel processing (0 = auto-detect, capped at 12)
    cosmic_opt_only: bool = False
    optimization_job_wall_time: Optional[float] = None
    optimization_total_cpu_time: Optional[float] = None
    refinement_job_wall_time: Optional[float] = None
    refinement_total_cpu_time: Optional[float] = None
    # Exclude patterns for optimization/refinement stages
    optimization_exclude_patterns: Dict[str, List[int]] = dataclasses.field(default_factory=dict)  # e.g., {'opt1': [2, 5, 6, 7, 8, 9]}
    refinement_exclude_patterns: Dict[str, List[int]] = dataclasses.field(default_factory=dict)  # e.g., {'ref1': [2, 5, 6, 7, 8, 9]}
    # QM program alias from input.in line 9 (e.g., "orca", "g16")
    qm_alias: str = "orca"  # Default to "orca" for ORCA installations
    qm_nproc: Optional[int] = None  # QM nprocs parsed from index line 11
    xtb_cycles: Optional[int] = 200  # Default xTB geometry cycle cap (overridden by index line 6)
    # Data capture attributes for protocol summary
    annealing_box_size: Optional[float] = None
    annealing_packing: Optional[float] = None
    optimization_xyz_source: Optional[str] = None
    optimization_completed: Optional[int] = None
    optimization_total: Optional[int] = None
    optimization_cosmic_folder: Optional[str] = None
    cosmic_folder: Optional[str] = None
    cosmic_motifs_created: Optional[int] = None
    last_cosmic_input_count: Optional[int] = None
    last_cosmic_motif_count: Optional[int] = None
    last_cosmic_umotif_count: Optional[int] = None
    cosmic_stage_counts: Dict[int, int] = dataclasses.field(default_factory=dict)  # stage index -> representative count
    cosmic_stage_input_counts: Dict[int, int] = dataclasses.field(default_factory=dict)  # stage index -> input structure count
    cosmic_locked_input_counts: Dict[int, int] = dataclasses.field(default_factory=dict)  # stage index -> redo-locked max input count (first full run wins; redo cannot inflate it)
    refinement_motifs_source: Optional[str] = None
    refinement_completed: Optional[int] = None
    refinement_total: Optional[int] = None
    refinement_cosmic_folder: Optional[str] = None
    eref_cosmic_folder: Optional[str] = None  # Folder for energy refinement cosmic stage
    eref_motifs_source: Optional[str] = None  # Motifs source for energy refinement
    recalculated_files: Optional[List[str]] = None  # List of basenames for files being recalculated in redo
    pending_cosmic_folder: Optional[str] = None  # Folder set by optimization/refinement for the next cosmic stage
    use_skipped_threshold: bool = False  # True if --skipped flag is used, False if --critical (default)
    current_stage: Optional[Dict[str, Any]] = None  # Active workflow stage (for stage-aware helpers)
    update_progress: Optional[Callable[[str], None]] = None  # Compact workflow progress callback
    completed_stage_count: int = 0  # Number of finished workflow stages for progress rendering
    # Live redo attempt of the stage currently executing (0 = initial run, not a redo).
    # Read by the progress renderer so the panel and 'ascec status' can show "(Redo n/N)".
    active_redo_stage_num: int = 0
    active_redo_attempt: int = 0
    active_redo_max: int = 0
    generated_template_files: List[str] = dataclasses.field(default_factory=list)  # Temp files extracted from embedded template labels
    maxprint: bool = False  # If True, keep all intermediate files (legacy behavior). Default: miniprint (clean up at end)
    _concurrent_prompted: Optional[int] = None  # Cached optimization concurrency selected interactively

    def get_previous_stage_output_dir(self, stage_type: str) -> Optional[str]:
        """
        Get output directory from the most recent completed stage of given type.

        Args:
            stage_type: Stage type prefix (e.g., 'r', 'optimization', 'cosmic', 'refinement')

        Returns:
            Output directory path or None if not found

        Example:
            context.get_previous_stage_output_dir('cosmic')  # Returns 'cosmic/motifs'
            context.get_previous_stage_output_dir('optimization')  # Returns 'calculation'
        """
        if not self.cache_file:
            return None

        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})

        # Search backwards through stages for matching type
        for stage_key in reversed(list(stages.keys())):
            if stage_key.startswith(stage_type):
                stage_data = stages[stage_key]
                if stage_data.get('status') == 'completed':
                    result = stage_data.get('result', {})
                    return result.get('output_dir')

        return None

    def get_stage_working_dir(self, stage_key: str) -> Optional[str]:
        """
        Get working directory for a specific stage by its key.

        Args:
            stage_key: Exact stage key (e.g., 'optimization_1', 'cosmic_2')

        Returns:
            Working directory path or None if not found

        Example:
            context.get_stage_working_dir('optimization_1')  # Returns 'calculation'
        """
        if not self.cache_file:
            return None

        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})

        if stage_key in stages:
            result = stages[stage_key].get('result', {})
            return result.get('working_dir')

        return None

    def get_previous_stage_input_dir(self, stage_type: str) -> Optional[str]:
        """
        Get input directory from the most recent completed stage of given type.

        Args:
            stage_type: Stage type prefix (e.g., 'r', 'optimization', 'cosmic', 'refinement')

        Returns:
            Input directory path or None if not found
        """
        if not self.cache_file:
            return None

        cache = load_protocol_cache(self.cache_file)
        stages = cache.get('stages', {})

        # Search backwards through stages for matching type
        for stage_key in reversed(list(stages.keys())):
            if stage_key.startswith(stage_type):
                stage_data = stages[stage_key]
                if stage_data.get('status') == 'completed':
                    result = stage_data.get('result', {})
                    return result.get('input_dir')

        return None
