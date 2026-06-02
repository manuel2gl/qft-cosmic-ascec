"""Protocol cache — v04's ``protocol_*.pkl`` resumability store.

R6 reverts **D-034** (the JSON job-registry rewrite). v04 keeps two distinct
persistence mechanisms; this module is the first of them, ported **verbatim**
from ``ascec-v04.py`` lines 2011-2122:

* ``load_protocol_cache`` / ``save_protocol_cache`` — read/write a per-protocol
  ``pickle`` blob keyed by stage.
* ``update_protocol_cache`` — record a stage's ``in_progress`` /
  ``completed`` / ``failed`` status, its wall time, and a stage-specific
  ``result`` dict.
* ``invalidate_stage_cache`` — drop stage entries from the on-disk cache
  (round-tripping through disk so a caller's stale in-memory ``cache`` dict
  never clobbers an intermediate update).

The cache is what makes a paused or interrupted workflow resumable: every
stage records itself here, and :func:`~cosmic_ascec.workflow.context.WorkflowContext`'s
``get_previous_stage_*`` accessors read it back. v04's *logic* — the
``in_progress`` merge, the ``start_time`` carry-over, the wall-time arithmetic
— is reproduced exactly (**D-039**: faithful decomposition, no redesign).

The SQLite machine-wide job registry is the *other* mechanism — see
:mod:`cosmic_ascec.workflow.job_registry`.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional

__all__ = [
    "load_protocol_cache",
    "save_protocol_cache",
    "update_protocol_cache",
    "accumulate_stage_wall_time",
    "invalidate_stage_cache",
]


def load_protocol_cache(cache_file: str = "protocol_cache.pkl") -> Dict[str, Any]:
    """
    Load protocol cache from pickle file.

    Returns:
        Dictionary with cache data or empty dict if file doesn't exist
    """
    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache file: {e}")
        return {}


def save_protocol_cache(cache_data: Dict[str, Any], cache_file: str = "protocol_cache.pkl"):
    """Save protocol cache to pickle file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache file: {e}")


def accumulate_stage_wall_time(cache_file: str, stage_key: str,
                               session_seconds: Optional[float]) -> float:
    """Add this session's *active* wall seconds to a stage's running total.

    The per-stage ``wall_time`` recorded by :func:`update_protocol_cache` is a
    *calendar* span (completion minus first start) and so includes idle gaps
    between resume sessions. For the per-stage QM batch we instead want the
    *active* wall time — the sum of each session's concurrent-run duration — so
    that a resumed run reports a "Total wall time" that is never shorter than
    its longest single job. The counter lives under the stage entry's
    ``accumulated_wall_time`` key so it survives across resumes.

    Returns the cumulative active wall time (prior total + this session).
    """
    session = max(0.0, session_seconds or 0.0)
    if not cache_file or not stage_key:
        return session
    try:
        cache = load_protocol_cache(cache_file)
    except Exception:
        return session
    stages = cache.setdefault('stages', {})
    entry = stages.setdefault(stage_key, {})
    total = (entry.get('accumulated_wall_time') or 0.0) + session
    entry['accumulated_wall_time'] = total
    cache['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        save_protocol_cache(cache, cache_file)
    except Exception:
        pass
    return total


def update_protocol_cache(stage_name: str, status: str, result: Optional[Dict[str, Any]] = None,
                          cache_file: str = "protocol_cache.pkl"):
    """
    Update protocol cache with stage completion info.

    Args:
        stage_name: Name of the stage (e.g., 'r1', 'opt', 'cosmic')
        status: Status ('in_progress', 'completed', 'failed')
        result: Optional result dictionary with stage-specific data
        cache_file: Path to cache file
    """
    cache = load_protocol_cache(cache_file)

    # Initialize structure if needed
    if 'stages' not in cache:
        cache['stages'] = {}
    if 'start_time' not in cache:
        cache['start_time'] = time.time()
        cache['start_time_str'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if 'protocol_file' not in cache:
        cache['protocol_file'] = cache_file  # Store which cache file this is

    # For in_progress: record start time or update existing entry
    if status == 'in_progress':
        # Get existing stage data if it exists
        existing_stage = cache['stages'].get(stage_name, {})
        existing_result = existing_stage.get('result', {})

        # Merge new result with existing result
        merged_result = existing_result.copy()
        if result:
            merged_result.update(result)

        new_entry = {
            'status': status,
            'start_time': existing_stage.get('start_time', time.time()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'result': merged_result
        }
        # Preserve the cross-session active-wall counter (see
        # accumulate_stage_wall_time); rebuilding the entry must not drop it.
        if 'accumulated_wall_time' in existing_stage:
            new_entry['accumulated_wall_time'] = existing_stage['accumulated_wall_time']
        cache['stages'][stage_name] = new_entry
    else:
        # For completed/failed: calculate wall time
        stage_data = cache['stages'].get(stage_name, {})
        start_time = stage_data.get('start_time', time.time())
        wall_time = time.time() - start_time

        new_entry = {
            'status': status,
            'start_time': start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wall_time': wall_time,
            'result': result or {}
        }
        if 'accumulated_wall_time' in stage_data:
            new_entry['accumulated_wall_time'] = stage_data['accumulated_wall_time']
        cache['stages'][stage_name] = new_entry

    cache['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')

    save_protocol_cache(cache, cache_file)


def invalidate_stage_cache(cache_file: str, *stage_keys: str) -> None:
    """Remove stage entries from the on-disk cache.

    Callers in execute_workflow_stages keep their own local `cache` dict that
    goes stale the moment update_protocol_cache writes to disk. Round-tripping
    through disk here ensures we never overwrite intermediate updates when
    invalidating a failed stage.
    """
    if not cache_file:
        return
    try:
        disk_cache = load_protocol_cache(cache_file)
    except Exception:
        return
    if not disk_cache or 'stages' not in disk_cache:
        return
    modified = False
    for key in stage_keys:
        if key and key in disk_cache['stages']:
            del disk_cache['stages'][key]
            modified = True
    if modified:
        try:
            save_protocol_cache(disk_cache, cache_file)
        except Exception:
            pass
