"""The annealing engine — main Monte Carlo loop.

A run is:

1. Build :class:`~cosmic_ascec.geometry.molecule.Molecule` templates from
   the parsed ``.asc`` config and place them in the cube with
   :func:`~cosmic_ascec.geometry.placement.initialize_cluster`.
2. **Initial-QM retry loop.** Up to
   :data:`~cosmic_ascec.monte_carlo.moves.INITIAL_QM_RETRIES` attempts,
   each redrawing a fresh placement, until the initial QM evaluation
   succeeds. Each attempt — success or failure — counts as a QM call.
3. Build the per-molecule rotatable-bond cache; if no molecule has any
   rotatable bonds, the conformational branch is auto-disabled.
4. Walk the temperature schedule. At each temperature, run Monte Carlo
   cycles via :func:`cosmic_ascec.monte_carlo.trial.trial` until either a
   lower-energy ("LwE") move is accepted — which ends the temperature
   early — or the per-temperature QM budget ``maxstep`` is exhausted.
   Every cycle spends exactly one QM call; clashing geometries go straight
   to QM (no overlap gate inside the MC loop).
5. After every temperature the budget shrinks by 10%:
   ``maxstep = max(floor, int(maxstep * 0.9))``.

**Observability is via callbacks, not return values.** The engine never
opens a file. It emits events (``RunStart``, ``TemperatureStart``,
``ConfigAccepted``, ``TemperatureComplete``, ``RunFinish``) to a list of
:class:`AnnealingCallback` objects; the writers in
:mod:`cosmic_ascec.file_formats` are the concrete handlers, and the
``.ascec_step`` progress writer in
:mod:`cosmic_ascec.command_line.ascec` is what lets the workflow progress
bar update during a long run.

All randomness flows through one explicit ``numpy.random.RandomState``
threaded into the engine. A given seed reproduces a given trajectory.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from cosmic_ascec.annealing.schedule import build_temperature_schedule
from cosmic_ascec.exceptions import QMError, WorkflowError
from cosmic_ascec.file_formats.asc_schema import AscConfig
from cosmic_ascec.geometry.molecule import Cluster, Molecule
from cosmic_ascec.geometry.placement import PlacementSettings, initialize_cluster
from cosmic_ascec.monte_carlo import (
    INITIAL_QM_RETRIES,
    EnergyFn,
    MoveParams,
    initialize_rotatable_bond_cache,
    trial,
)

logger = logging.getLogger(__name__)

# v04 shrinks the per-temperature QM budget by 10% after each temperature
# (ascec-v04.py line 20804) and never lets it fall below ``max_cycle_floor``.
MAXSTEP_REDUCTION_FACTOR: float = 0.90

# v04's default floor when the ``.asc`` omits the line-6 floor value.
DEFAULT_CYCLE_FLOOR: int = 10


# --------------------------------------------------------------------------- #
# Event payloads                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunStart:
    """Emitted once, before the first temperature, after the initial QM call."""

    config: AscConfig
    initial_cluster: Cluster
    initial_energy: float
    initial_temperature: float
    temperatures: tuple[float, ...]


@dataclass(frozen=True)
class ConfigAccepted:
    """Emitted every time a configuration is accepted (including the initial one).

    ``history_n_eval`` is the v04 ``.out``-history column: QM calls since the
    previous history line. ``cumulative_n_eval`` is the v04 ``tvse`` column:
    total QM calls so far. ``criterion`` is one of ``"Intv"`` (initial),
    ``"LwE"``, ``"Mpol"``, ``"EMpol"``.
    """

    index: int  # 1-based accepted-configuration counter
    cluster: Cluster
    energy: float
    temperature: float
    criterion: str
    history_n_eval: int
    cumulative_n_eval: int
    is_initial: bool
    is_new_lowest: bool


@dataclass(frozen=True)
class TemperatureStart:
    """Emitted at the top of every temperature in the annealing schedule.

    Fires once per temperature, *before* any MC cycles run at that
    temperature. Carries the 1-based temperature index and the schedule
    length so observers (progress writers, status displays) can report
    "step N/M" the same way the monolith does — v04 writes the same
    "{step+1}/{total}" line into ``run_dir/.ascec_step`` (ascec-v04.py
    line 20640), which the replication aggregator polls every 2 s to
    drive the workflow progress bar.
    """

    temperature: float
    step_index: int   # 1-based: 1 .. total_steps
    total_steps: int


@dataclass(frozen=True)
class TemperatureComplete:
    """Emitted when a temperature ends without a final LwE acceptance.

    Mirrors v04's "N/A" history line (ascec-v04.py lines 20787-20797): it
    accounts for QM calls made at this temperature that were not already
    written to the history (rejected moves, or moves after the last Mpol).
    ``history_n_eval`` is always > 0 when this event fires.
    """

    temperature: float
    energy: float
    history_n_eval: int


@dataclass(frozen=True)
class AnnealingResult:
    """The full outcome of an :func:`anneal` run, also handed to ``on_run_finish``."""

    initial_cluster: Cluster
    final_cluster: Cluster
    lowest_cluster: Cluster
    initial_energy: float
    final_energy: float
    lowest_energy: float
    lowest_config_index: int
    total_accepted: int
    total_qm_calls: int
    boltzmann_accepted: int  # v04 ``iboltz`` — Mpol + EMpol acceptances
    lower_energy_accepted: int  # v04 ``lower_energy_configs`` — initial + LwE
    temperatures: tuple[float, ...]
    wall_time_seconds: float


@dataclass(frozen=True)
class RunFinish:
    """Emitted once, after the last temperature."""

    result: AnnealingResult


# --------------------------------------------------------------------------- #
# Callback contract                                                           #
# --------------------------------------------------------------------------- #


class AnnealingCallback:
    """Base class for engine observers; every hook defaults to a no-op.

    The writers in :mod:`cosmic_ascec.file_formats` subclass this and override
    only the hooks they care about. Callbacks are invoked synchronously, in
    registration order, on the thread running :func:`anneal`.
    """

    def on_run_start(self, event: RunStart) -> None:  # noqa: D401
        """Called once before the first temperature."""

    def on_temperature_start(self, event: TemperatureStart) -> None:
        """Called once at the top of every temperature, before any MC cycles."""

    def on_config_accepted(self, event: ConfigAccepted) -> None:
        """Called for every accepted configuration, initial config included."""

    def on_temperature_complete(self, event: TemperatureComplete) -> None:
        """Called when a temperature ends with unlogged QM calls (v04 'N/A')."""

    def on_run_finish(self, event: RunFinish) -> None:
        """Called once after the last temperature."""


def _emit(callbacks: Sequence[AnnealingCallback], hook: str, event: object) -> None:
    """Dispatch one event to every callback, isolating handler failures.

    A misbehaving writer must not abort a multi-hour annealing run, so its
    exception is logged and swallowed (the science already happened — only the
    bookkeeping output is at risk).
    """
    for callback in callbacks:
        try:
            getattr(callback, hook)(event)
        except Exception:  # noqa: BLE001 - a writer crash must not kill the run
            logger.exception("annealing callback %r failed in %s", callback, hook)


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def anneal(
    config: AscConfig,
    energy_fn: EnergyFn,
    *,
    rng: np.random.RandomState,
    callbacks: Sequence[AnnealingCallback] = (),
    placement_settings: Optional[PlacementSettings] = None,
    run_logger: Optional[logging.Logger] = None,
    use_standard_metropolis: bool = False,
) -> AnnealingResult:
    """Run one full simulated-annealing trajectory — verbatim v04 main loop.

    Args:
        config: Parsed ``.asc`` configuration.
        energy_fn: ``Cluster -> float`` energy provider. Raising
            :class:`QMError` from inside a move is treated as a rejected trial
            (v04 ``jo_status == 0``); raising it on every one of the initial
            placements is fatal.
        rng: The run's single ``numpy.random.RandomState`` — placement, move
            proposal, and the standard-Metropolis draw all pull from it
            (D-038).
        callbacks: Observers notified as the run progresses (see
            :class:`AnnealingCallback`).
        placement_settings: Optional override for initial-placement tuning.
        run_logger: Optional logger threaded into placement and the move cache.
        use_standard_metropolis: ``True`` selects v04's standard Metropolis
            (the ``--standard`` CLI flag); the default ``False`` is v04's
            modified criterion.

    Returns:
        :class:`AnnealingResult` describing the trajectory and its minimum.

    Raises:
        WorkflowError: invalid schedule, empty system, or every initial QM
            evaluation failing.
    """
    start_time = time.monotonic()

    if not config.molecules:
        raise WorkflowError("anneal: configuration has no molecules to place")

    move_params = MoveParams.from_config(
        config, use_standard_metropolis=use_standard_metropolis
    )
    temperatures = build_temperature_schedule(config.schedule)
    cycle_floor = (
        config.cycles.floor_value
        if config.cycles.floor_value is not None
        else DEFAULT_CYCLE_FLOOR
    )
    maxstep = config.cycles.max_cycles_per_temperature
    molecules = [Molecule.from_spec(spec) for spec in config.molecules]

    # ----- Initial placement + QM, with v04's retry loop -------------------- #
    # v04 redraws a fresh placement on every failed initial QM call and counts
    # each attempt as a QM call (ascec-v04.py lines 20462-20557). Initial
    # geometry uses the .asc template dihedrals as-is — the live MC loop is
    # responsible for exploring the conformational space from there.
    #
    # The rotatable-bond cache is built once from the placement of the first
    # attempt; atom indices are stable across replacements, so the same cache
    # is reused for the live MC loop after the run starts.
    placement_seed = initialize_cluster(
        molecules,
        box_length=config.box.cube_length_angstrom,
        rng=rng,
        settings=placement_settings,
        logger=run_logger,
    )
    cache, effective_prob = initialize_rotatable_bond_cache(
        placement_seed.cluster,
        conformational_move_prob=move_params.conformational_move_prob,
        max_dihedral_angle_rad=move_params.max_dihedral_angle_rad,
        logger=run_logger,
    )
    # Conformational sampling was requested but the system has no rotatable
    # bonds to sample (e.g. a lone ring like cyclohexane — ring bonds are not
    # rotatable, see cosmic_ascec.geometry.bonds). Rather than silently fall
    # back to rigid-body-only moves, stop and tell the user.
    if move_params.conformational_move_prob > 0.0 and cache.total_rotatable_bonds == 0:
        message = (
            "no rotatable bonds found in any molecule, so conformational "
            "sampling has nothing to explore (ring bonds are excluded because "
            "rotating them would tear the ring). Set the conformational "
            "sampling probability to 0 to run rigid-body annealing instead."
        )
        if run_logger is not None:
            run_logger.error(message)
        raise WorkflowError(message)
    if effective_prob != move_params.conformational_move_prob:
        move_params = replace(move_params, conformational_move_prob=effective_prob)

    initial_cluster: Optional[Cluster] = None
    initial_energy = 0.0
    qm_call_count = 0
    last_initial_exc: Optional[QMError] = None
    pending_placement: Optional[Cluster] = placement_seed.cluster
    for attempt in range(INITIAL_QM_RETRIES):
        if pending_placement is None:
            pending_placement = initialize_cluster(
                molecules,
                box_length=config.box.cube_length_angstrom,
                rng=rng,
                settings=placement_settings,
                logger=run_logger,
            ).cluster
        candidate = pending_placement
        pending_placement = None
        qm_call_count += 1
        try:
            initial_energy = float(energy_fn(candidate))
        except QMError as exc:
            last_initial_exc = exc
            if run_logger is not None:
                run_logger.warning(
                    "initial QM attempt %d/%d failed: %s — redrawing placement",
                    attempt + 1, INITIAL_QM_RETRIES, exc,
                )
            continue
        initial_cluster = candidate
        break

    if initial_cluster is None:
        raise WorkflowError(
            f"initial QM evaluation failed after {INITIAL_QM_RETRIES} attempts; "
            f"cannot start annealing: {last_initial_exc}"
        )

    # v04 lines 20506-20513 — surface conformational-sampling status to the
    # terminal so the user can confirm the feature is active. v04 prints this
    # via _print_verbose level 0 (always to stderr); we mirror that here.
    if effective_prob > 0.0:
        max_deg = float(np.degrees(move_params.max_dihedral_angle_rad))
        print(
            f"Conformational sampling enabled: {effective_prob * 100:.1f}% probability",
            file=sys.stderr,
        )
        print(f"  Maximum dihedral rotation: ±{max_deg:.1f}°", file=sys.stderr)
        print(
            f"  Move types: {effective_prob * 100:.1f}% conformational (dihedral rotations), "
            f"{(1 - effective_prob) * 100:.1f}% rigid-body (translation+rotation)",
            file=sys.stderr,
        )
        print(
            "  Note: Conformational moves with atom overlaps automatically "
            "fall back to rigid-body moves",
            file=sys.stderr,
        )
    else:
        print(
            "Using rigid-body moves only (translation + rotation)",
            file=sys.stderr,
        )

    last_history_qm_call = qm_call_count

    current_cluster = initial_cluster
    current_energy = initial_energy
    lowest_cluster = initial_cluster
    lowest_energy = initial_energy
    lowest_config_index = 1

    # v04 counts the initial configuration as the first lower-energy accept
    # (ascec-v04.py line 20495) and the first written configuration.
    total_accepted = 1
    lower_energy_configs = 1
    boltzmann_accepted = 0

    _emit(callbacks, "on_run_start", RunStart(
        config=config,
        initial_cluster=initial_cluster,
        initial_energy=initial_energy,
        initial_temperature=temperatures[0],
        temperatures=temperatures,
    ))
    _emit(callbacks, "on_config_accepted", ConfigAccepted(
        index=1,
        cluster=initial_cluster,
        energy=initial_energy,
        temperature=temperatures[0],
        criterion="Intv",
        history_n_eval=0,
        cumulative_n_eval=qm_call_count,
        is_initial=True,
        is_new_lowest=True,
    ))

    # ----- Temperature walk ------------------------------------------------- #
    total_temperature_steps = len(temperatures)
    for temperature_index, temperature in enumerate(temperatures, start=1):
        # Fire the per-temperature start event so progress writers can update
        # the workflow progress bar (mirrors the v04 inline ``.ascec_step``
        # write at ascec-v04.py line 20640).
        _emit(callbacks, "on_temperature_start", TemperatureStart(
            temperature=temperature,
            step_index=temperature_index,
            total_steps=total_temperature_steps,
        ))
        qm_calls_this_temp = 0
        moved_to_next_temperature = False

        # v04: run MC cycles until maxstep QM evaluations are spent, or an LwE
        # acceptance breaks early. Every cycle spends exactly one QM call —
        # there is no overlap retry (v04 has no overlap gate in this loop).
        while qm_calls_this_temp < maxstep:
            qm_calls_this_temp += 1
            try:
                result = trial(
                    current_cluster,
                    current_energy=current_energy,
                    energy_fn=energy_fn,
                    rng=rng,
                    temperature_kelvin=temperature,
                    params=move_params,
                    cache=cache,
                )
            except QMError as exc:
                # v04 ``jo_status == 0``: the QM call was spent, the move is
                # rejected, the loop continues (ascec-v04.py lines 20681-20683).
                qm_call_count += 1
                if run_logger is not None:
                    run_logger.warning(
                        "QM evaluation failed (call %d): %s — rejecting move",
                        qm_call_count, exc,
                    )
                continue

            qm_call_count += 1
            if not result.accepted:
                continue

            # ----- Accepted ---------------------------------------------- #
            current_cluster = result.cluster
            current_energy = result.energy
            total_accepted += 1

            is_new_lowest = current_energy < lowest_energy
            if is_new_lowest:
                lowest_energy = current_energy
                lowest_cluster = current_cluster
                lowest_config_index = total_accepted

            if result.criterion == "LwE":
                lower_energy_configs += 1
            else:  # "Mpol" / "EMpol"
                boltzmann_accepted += 1

            history_n_eval = qm_call_count - last_history_qm_call
            _emit(callbacks, "on_config_accepted", ConfigAccepted(
                index=total_accepted,
                cluster=current_cluster,
                energy=current_energy,
                temperature=temperature,
                criterion=result.criterion,
                history_n_eval=history_n_eval,
                cumulative_n_eval=qm_call_count,
                is_initial=False,
                is_new_lowest=is_new_lowest,
            ))
            last_history_qm_call = qm_call_count

            if result.criterion == "LwE":
                # v04: a lower-energy acceptance ends the temperature.
                moved_to_next_temperature = True
                break

        # v04's "N/A" line: account for QM calls since the last history entry
        # when the temperature ended without a final LwE acceptance.
        if not moved_to_next_temperature and qm_call_count > last_history_qm_call:
            _emit(callbacks, "on_temperature_complete", TemperatureComplete(
                temperature=temperature,
                energy=current_energy,
                history_n_eval=qm_call_count - last_history_qm_call,
            ))
            last_history_qm_call = qm_call_count

        maxstep = max(cycle_floor, int(maxstep * MAXSTEP_REDUCTION_FACTOR))

    result = AnnealingResult(
        initial_cluster=initial_cluster,
        final_cluster=current_cluster,
        lowest_cluster=lowest_cluster,
        initial_energy=initial_energy,
        final_energy=current_energy,
        lowest_energy=lowest_energy,
        lowest_config_index=lowest_config_index,
        total_accepted=total_accepted,
        total_qm_calls=qm_call_count,
        boltzmann_accepted=boltzmann_accepted,
        lower_energy_accepted=lower_energy_configs,
        temperatures=temperatures,
        wall_time_seconds=time.monotonic() - start_time,
    )
    _emit(callbacks, "on_run_finish", RunFinish(result=result))
    return result


__all__ = [
    "AnnealingCallback",
    "AnnealingResult",
    "ConfigAccepted",
    "DEFAULT_CYCLE_FLOOR",
    "MAXSTEP_REDUCTION_FACTOR",
    "RunFinish",
    "RunStart",
    "TemperatureComplete",
    "TemperatureStart",
    "anneal",
]
