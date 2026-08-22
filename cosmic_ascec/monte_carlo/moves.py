"""Monte Carlo move proposal — :func:`propose_unified_move`.

The annealing loop spends one QM call per MC cycle and proposes a move via
this single function. It moves *every* molecule in the system simultaneously
("unified"). For each molecule, independently:

* If the molecule has at least one rotatable bond and a per-move probability
  draw succeeds, propose a single dihedral rotation about a randomly chosen
  bond. The new geometry is checked for intramolecular overlap; if it
  clashes, the rigid branch runs instead.
* Otherwise, propose a rigid-body translation + rotation of the whole
  molecule. No overlap check on this branch — a clashing geometry is sent
  straight to the QM program (deliberate: the QM cost is the search budget).

The "send the clash to QM" rule holds for molecular clusters, where a clash
costs one high-but-finite energy that Metropolis then rejects. It breaks down
for a run with a ``*$`` frozen substrate: nothing stops a molecule from being
translated *into* the pinned block, and a molecule embedded in, say, a gold
slab does not produce a high energy — it produces an SCF that never converges,
so the QM call returns no energy and the run dies. Substrate runs therefore
enable a contact gate: a molecule whose proposed position violates the van der
Waals floor (:func:`~cosmic_ascec.geometry.overlap.has_contact_clash`) against
any other molecule, frozen or mobile, stays where it was for this cycle. That
is a hard-wall rejection — the standard Metropolis treatment of a hard-core
potential, and symmetric, so detailed balance is preserved.

The gate is active **only** when the cluster contains a frozen molecule. ``*$``
is a construct v04 cannot express, so every input without it keeps the verbatim
v04 behaviour, bit-identical, and consumes exactly the same random numbers —
the gate itself never draws.

The RNG-draw order per molecule is load-bearing for replay; if you reorder
draws you will not reproduce historical trajectories at the same seed:

1. ``rng.rand()`` — the conformational-attempt test. Drawn only when
   ``conformational_move_prob > 0`` *and* the molecule has a rotatable bond
   (short-circuit; no draw is consumed if either is false).
2. If a conformational move is attempted: ``rng.randint(n_bonds)`` (pick a
   bond), then ``rng.rand()`` (pick a dihedral angle).
3. The rigid branch (reached if step 2 was skipped or clashed):
   ``rng.rand(3)`` (translation vector) then three scalar ``rng.rand()``
   (Euler angles α, β, γ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from cosmic_ascec.elements.data import ATOMIC_WEIGHTS
from cosmic_ascec.geometry.bonds import find_rotatable_bonds
from cosmic_ascec.geometry.inertia import calculate_mass_center
from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.geometry.overlap import (
    DEFAULT_SUBSTRATE_CONTACT_SCALE,
    check_intramolecular_overlap,
    has_contact_clash,
    vdw_contact_radii,
)

_AXIS_ROTATION_MIN_NORM = 1e-6
"""Axis-length floor for :func:`rotate_around_bond`.

Below this length, the rotation is a no-op — dividing by such a tiny norm
would amplify floating-point noise.
"""

# How many times the annealing engine redraws the initial placement until
# the initial QM call succeeds.
INITIAL_QM_RETRIES = 100

# --------------------------------------------------------------------------- #
# Mixture-proposal band fractions (used only when MoveParams.flip_probability  #
# > 0 — the default 0 path preserves the verbatim v04 uniform draw).           #
# --------------------------------------------------------------------------- #

FLIP_BAND_FRACTION = 0.85
"""Lower edge of the mixture's flip band, as a fraction of ``max_dihedral``.

When the mixture toggle picks the flip band the angle magnitude is drawn
uniformly in ``[FLIP_BAND_FRACTION * max, max]``, with a random sign. At
``max = 180°`` this is ``[153°, 180°]`` — exactly the "near full rotation"
band a cis ↔ trans flip needs.
"""

SMALL_BAND_FRACTION = 1.0 / 3.0
"""Half-width of the mixture's small band, as a fraction of ``max_dihedral``.

When the mixture toggle picks the small band the angle is drawn uniformly
in ``[-SMALL_BAND_FRACTION * max, +SMALL_BAND_FRACTION * max]``. Concentrates
"refinement nudges" near zero while leaving the flip band to do barrier
crossings. The mixture skips the energetically-expensive middle band
(``max/3 < |angle| < 0.85 * max``).
"""

# A rotatable bond: (axis atom 1, axis atom 2, indices of the atoms that move).
RotatableBond = Tuple[int, int, List[int]]


# --------------------------------------------------------------------------- #
# Move parameters                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MoveParams:
    """The Monte Carlo move/acceptance knobs, straight from the ``.asc`` file.

    v04 keeps these on ``SystemState``; v05 threads them explicitly. Fields map
    one-to-one onto v04 attributes:

    * ``max_displacement``        — ``state.max_displacement_a`` (``.asc`` line 7).
    * ``max_rotation_angle_rad``  — ``state.max_rotation_angle_rad`` (line 7).
    * ``max_dihedral_angle_rad``  — ``state.max_dihedral_angle_rad`` (line 8).
    * ``conformational_move_prob``— ``state.conformational_move_prob`` (line 8).
    * ``use_standard_metropolis`` — the ``--standard`` CLI flag (default
      ``False`` → v04's *modified* Metropolis criterion).
    * ``flip_probability``        — opt-in mixture knob (default ``0.0`` →
      verbatim v04 uniform draw). When ``> 0``, the dihedral angle is drawn
      from a symmetric mixture: with probability ``flip_probability`` from
      the *flip band* ``±[FLIP_BAND_FRACTION * max, max]``, otherwise from the
      *small band* ``[-SMALL_BAND_FRACTION * max, +SMALL_BAND_FRACTION * max]``.
      Both bands are symmetric in sign, so detailed balance still holds (no
      Metropolis-Hastings correction). The mixture is the right answer when
      the uniform ``[-max, +max]`` distribution wastes most draws on the
      energetically-unfavorable middle band (e.g., cis-formic-acid OH at
      ~90° is the worst place to be — small nudges or full flips help; 90°
      itself does not).
    * ``substrate_contact_scale`` — VDW scale factor for the contact gate that
      runs only on ``*$`` substrate inputs (see the module docstring). Has no
      effect on a run without a frozen block.
    """

    max_displacement: float
    max_rotation_angle_rad: float
    max_dihedral_angle_rad: float
    conformational_move_prob: float
    use_standard_metropolis: bool = False
    flip_probability: float = 0.0
    substrate_contact_scale: float = DEFAULT_SUBSTRATE_CONTACT_SCALE

    @classmethod
    def from_config(cls, config, *, use_standard_metropolis: bool = False) -> "MoveParams":
        """Build from a parsed :class:`~cosmic_ascec.file_formats.asc_schema.AscConfig`."""
        moves = config.moves
        return cls(
            max_displacement=moves.max_displacement_angstrom,
            max_rotation_angle_rad=moves.max_rotation_radian,
            max_dihedral_angle_rad=moves.max_dihedral_radian,
            conformational_move_prob=moves.conformational_probability,
            use_standard_metropolis=use_standard_metropolis,
        )


def _draw_dihedral_angle(
    rng: np.random.RandomState,
    max_rad: float,
    flip_probability: float,
) -> float:
    """Pick one dihedral rotation angle.

    ``flip_probability <= 0.0`` (the default for all .asc-driven callers) is
    the verbatim v04 draw: uniform on ``[-max_rad, +max_rad]``. Exactly one
    ``rng.rand()`` is consumed — the load-bearing RNG sequence for replay
    is unchanged from v04 in that branch.

    ``flip_probability > 0.0`` uses the symmetric two-band mixture described
    on :class:`MoveParams`. The branch consumes additional ``rng.rand()``
    draws (toggle, then sign+magnitude or just magnitude), so a run with a
    non-zero flip probability follows a different RNG trajectory than the
    same seed at zero — by design, since the proposal distribution itself
    is what changed.
    """
    if flip_probability <= 0.0:
        return (rng.rand() - 0.5) * 2.0 * max_rad

    if rng.rand() < flip_probability:
        # Flip band: ±[FLIP_BAND_FRACTION * max, max], sign drawn uniformly.
        sign = 1.0 if rng.rand() < 0.5 else -1.0
        u = rng.rand()
        return sign * max_rad * (FLIP_BAND_FRACTION + u * (1.0 - FLIP_BAND_FRACTION))

    # Small band: uniform on [-SMALL_BAND_FRACTION * max, +SMALL_BAND_FRACTION * max].
    return (rng.rand() - 0.5) * 2.0 * SMALL_BAND_FRACTION * max_rad


@dataclass
class RotatableBondCache:
    """Per-molecule rotatable-bond cache — v04's ``state.rotatable_bonds_by_molecule``
    / ``state.molecules_with_rotatable_bonds`` pair (ascec-v04.py lines 3733-3749).

    Mutable: :func:`propose_unified_move` re-fills an empty entry in place,
    exactly as v04 does at lines 3824-3828.
    """

    rotatable_bonds_by_molecule: List[List[RotatableBond]]
    molecules_with_rotatable_bonds: List[bool]
    total_rotatable_bonds: int


def _masses_for(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Per-atom mass vector. Unknown Z falls back to 1.0 amu, mirroring v04
    (``state.atomic_number_to_mass.get(anum, 1.0)``)."""
    return np.array(
        [ATOMIC_WEIGHTS.get(int(z), 1.0) for z in atomic_numbers],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Dihedral primitive (Rodrigues) — v04 rotate_around_bond, line 3569           #
# --------------------------------------------------------------------------- #


def rotate_around_bond(
    coords: np.ndarray,
    atom1_idx: int,
    atom2_idx: int,
    moving_atoms: Sequence[int],
    angle_rad: float,
) -> np.ndarray:
    """Rotate ``moving_atoms`` around the ``atom1 → atom2`` bond by ``angle_rad``.

    Verbatim port of v04 ``rotate_around_bond`` (ascec-v04.py lines 3569-3618):
    Rodrigues' formula with ``np.cos``/``np.sin`` (numpy, as v04 — not
    ``math``), atom1 as the rotation centre, and the axis-defining atoms left
    untouched even when listed in ``moving_atoms`` (v04 lines 3605-3606).

    Returns a *new* ``(N, 3)`` ``float64`` array; the input is never mutated.
    """
    new_coords = np.copy(coords)

    axis_vector = coords[atom2_idx] - coords[atom1_idx]
    axis_length = np.linalg.norm(axis_vector)
    if axis_length < _AXIS_ROTATION_MIN_NORM:
        return new_coords

    axis_unit = axis_vector / axis_length
    rotation_center = coords[atom1_idx]
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    for atom_idx in moving_atoms:
        if atom_idx == atom1_idx or atom_idx == atom2_idx:
            continue
        point_vector = coords[atom_idx] - rotation_center
        rotated_vector = (
            point_vector * cos_angle
            + np.cross(axis_unit, point_vector) * sin_angle
            + axis_unit * np.dot(axis_unit, point_vector) * (1.0 - cos_angle)
        )
        new_coords[atom_idx] = rotation_center + rotated_vector

    return new_coords


# --------------------------------------------------------------------------- #
# Rotatable-bond cache — v04 initialize_rotatable_bond_cache, line 3727        #
# --------------------------------------------------------------------------- #


def initialize_rotatable_bond_cache(
    cluster: Cluster,
    *,
    conformational_move_prob: float,
    max_dihedral_angle_rad: float,
    logger=None,
) -> Tuple[RotatableBondCache, float]:
    """Build the per-molecule rotatable-bond cache and decide whether
    conformational sampling stays enabled.

    Verbatim port of v04 ``initialize_rotatable_bond_cache``
    (ascec-v04.py lines 3727-3765): conformational sampling is auto-disabled
    (the returned probability becomes ``0.0``) when the system has *no*
    rotatable bonds, or when the maximum dihedral angle is zero.

    Returns ``(cache, effective_conformational_move_prob)``.
    """
    rotatable_bonds_by_molecule: List[List[RotatableBond]] = []
    molecules_with_rotatable_bonds: List[bool] = []
    total_rotatable_bonds = 0

    for molecule_idx in range(cluster.num_molecules):
        start = int(cluster.molecule_offsets[molecule_idx])
        end = int(cluster.molecule_offsets[molecule_idx + 1])
        mol_coords = cluster.coords[start:end, :]
        mol_atomic_numbers = [int(cluster.atomic_numbers[i]) for i in range(start, end)]

        rotatable_bonds = find_rotatable_bonds(mol_coords, mol_atomic_numbers)
        rotatable_bonds_by_molecule.append(rotatable_bonds)
        molecules_with_rotatable_bonds.append(len(rotatable_bonds) > 0)
        total_rotatable_bonds += len(rotatable_bonds)

    effective_prob = conformational_move_prob

    if effective_prob > 0.0 and total_rotatable_bonds == 0:
        effective_prob = 0.0
        if logger is not None:
            logger.info(
                "Conformational sampling auto-disabled: no rotatable bonds "
                "found in any molecule."
            )

    if effective_prob > 0.0 and max_dihedral_angle_rad == 0.0:
        effective_prob = 0.0
        if logger is not None:
            logger.info(
                "Conformational sampling auto-disabled: maximum dihedral "
                "angle is 0 degrees."
            )

    cache = RotatableBondCache(
        rotatable_bonds_by_molecule=rotatable_bonds_by_molecule,
        molecules_with_rotatable_bonds=molecules_with_rotatable_bonds,
        total_rotatable_bonds=total_rotatable_bonds,
    )
    return cache, effective_prob


# --------------------------------------------------------------------------- #
# The live move — v04 propose_unified_move, line 3778                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UnifiedMoveResult:
    """Outcome of one :func:`propose_unified_move` call.

    ``cluster`` is the proposed full-system geometry (every molecule moved).
    ``last_moved_molecule`` and ``move_type`` mirror v04's return tuple — the
    annealing loop only consumes ``cluster``, the rest is kept for parity
    inspection. ``move_type`` is ``"conformational"`` if any molecule's last
    applied move was a dihedral rotation, else ``"translate_rotate"``.
    """

    cluster: Cluster
    last_moved_molecule: int
    move_type: str


def propose_unified_move(
    cluster: Cluster,
    *,
    rng: np.random.RandomState,
    params: MoveParams,
    cache: RotatableBondCache,
) -> UnifiedMoveResult:
    """Propose a new full-system configuration — verbatim v04 ``propose_unified_move``.

    Every molecule that is not frozen is moved. For each, with probability
    ``params.conformational_move_prob`` (and a rotatable bond available) a
    dihedral rotation is attempted; if it is skipped or produces an
    intramolecular clash, a rigid-body translation+rotation is applied instead.

    The returned :class:`UnifiedMoveResult` carries a brand-new
    :class:`Cluster`; the input cluster is never mutated (D-004). The run RNG
    is threaded explicitly (D-038); see the module docstring for the
    load-bearing draw order.
    """
    # v04 line 3794: proposed_rp_full_system = np.copy(current_rp)
    proposed = np.array(cluster.coords, dtype=np.float64, copy=True)
    offsets = cluster.molecule_offsets
    last_moved_mol_idx = -1
    move_type_used = "translate_rotate"

    # v04 line 3865 used one half-edge for all three axes. ``half_box`` is the
    # per-axis generalisation and expands a cube to three equal halves, so the
    # cubic case is arithmetically identical. Nothing here draws a random
    # number, so the box shape cannot perturb the RNG stream.
    half_box = cluster.half_box
    with_rb = cache.molecules_with_rotatable_bonds

    # Contact gate — substrate runs only, so v04-expressible inputs are
    # untouched (module docstring). Nothing below draws a random number, so the
    # gate cannot perturb the RNG stream even when it is active.
    gate_active = any(
        getattr(mol, "frozen", False) for mol in cluster.molecules
    )
    contact_radii = (
        vdw_contact_radii(cluster.atomic_numbers) if gate_active else None
    )

    def contact_clash(start: int, end: int, candidate: np.ndarray) -> bool:
        """True if ``candidate`` violates the VDW floor against any other atom.

        ``proposed`` is mutated in place as the sweep walks the molecules, so
        this sees molecules below ``molecule_idx`` at their new positions and
        those above at their current ones — an ordinary sequential Metropolis
        sweep against a hard-core potential.
        """
        others = np.concatenate((proposed[:start, :], proposed[end:, :]), axis=0)
        other_radii = np.concatenate(
            (contact_radii[:start], contact_radii[end:]), axis=0
        )
        return has_contact_clash(
            candidate,
            contact_radii[start:end],
            others,
            other_radii,
            scale=params.substrate_contact_scale,
        )

    for molecule_idx in range(cluster.num_molecules):
        # A frozen molecule (``*$`` block in the .asc file) is never proposed
        # for a move — neither rigid-body nor conformational — so it keeps the
        # exact coordinates it was given. Skipping shifts the RNG draw order
        # relative to v04, but only for inputs that contain ``*$``, a construct
        # v04 cannot express; runs without it are bit-identical to before.
        if getattr(cluster.molecules[molecule_idx], "frozen", False):
            continue

        start = int(offsets[molecule_idx])
        end = int(offsets[molecule_idx + 1])

        # Where this molecule sits before the proposal, so the contact gate can
        # put it back if the proposal turns out to be inside something.
        pre_move_coords = (
            np.copy(proposed[start:end, :]) if gate_active else None
        )

        # v04 lines 3802-3806.
        molecule_has_rotatable_bond = (
            bool(with_rb)
            and molecule_idx < len(with_rb)
            and with_rb[molecule_idx]
        )

        # v04 lines 3809-3811 — the rng.rand() is short-circuited away unless
        # conformational sampling is enabled *and* a rotatable bond exists.
        attempt_conformational = (
            params.conformational_move_prob > 0
            and molecule_has_rotatable_bond
            and rng.rand() < params.conformational_move_prob
        )

        conformational_success = False
        if attempt_conformational:
            mol_coords = proposed[start:end, :]
            mol_atomic_numbers = [
                int(cluster.atomic_numbers[i]) for i in range(start, end)
            ]

            # v04 lines 3820-3828 — cache lookup with a re-find fallback.
            rotatable_bonds: List[RotatableBond] = []
            if cache.rotatable_bonds_by_molecule and molecule_idx < len(
                cache.rotatable_bonds_by_molecule
            ):
                rotatable_bonds = cache.rotatable_bonds_by_molecule[molecule_idx]

            if not rotatable_bonds:
                rotatable_bonds = find_rotatable_bonds(mol_coords, mol_atomic_numbers)
                if cache.rotatable_bonds_by_molecule and molecule_idx < len(
                    cache.rotatable_bonds_by_molecule
                ):
                    cache.rotatable_bonds_by_molecule[molecule_idx] = rotatable_bonds
                    cache.molecules_with_rotatable_bonds[molecule_idx] = bool(
                        rotatable_bonds
                    )

            if rotatable_bonds:
                # v04 line 3832 — pick a rotatable bond uniformly.
                bond_atom1, bond_atom2, moving_atoms = rotatable_bonds[
                    rng.randint(len(rotatable_bonds))
                ]
                # Angle draw. When ``params.flip_probability == 0`` this is
                # the verbatim v04 uniform draw on [-max, +max]; otherwise
                # it's the symmetric two-band mixture (see :class:`MoveParams`
                # and :func:`_draw_dihedral_angle`). Both are symmetric, so
                # detailed balance holds without a Hastings correction.
                rotation_angle = _draw_dihedral_angle(
                    rng,
                    params.max_dihedral_angle_rad,
                    params.flip_probability,
                )

                new_mol_coords = rotate_around_bond(
                    mol_coords, bond_atom1, bond_atom2, moving_atoms, rotation_angle
                )

                # v04 line 3843 — accept only if no intramolecular clash. On a
                # substrate run the dihedral must also clear the contact gate;
                # a rotation that swings a branch into the frozen block falls
                # through to the rigid branch, exactly as an intramolecular
                # clash already does.
                if not check_intramolecular_overlap(
                    new_mol_coords, mol_atomic_numbers
                ) and not (
                    gate_active and contact_clash(start, end, new_mol_coords)
                ):
                    proposed[start:end, :] = new_mol_coords
                    last_moved_mol_idx = molecule_idx
                    move_type_used = "conformational"
                    conformational_success = True

        # v04 lines 3850-3851 — a successful conformational move ends this
        # molecule; otherwise fall through to the rigid-body move.
        if conformational_success:
            continue

        mol_coords_current = proposed[start:end, :]
        mol_atomic_numbers = [
            int(cluster.atomic_numbers[i]) for i in range(start, end)
        ]
        mol_masses = _masses_for(mol_atomic_numbers)
        current_rcm = calculate_mass_center(mol_coords_current, mol_masses)

        # --- Translation, with v04's per-axis box bounce (lines 3864-3876) -- #
        random_displacement_vector = (
            (rng.rand(3) - 0.5) * 2.0 * params.max_displacement
        )
        new_rcm_after_translation = np.copy(current_rcm)
        actual_atom_displacement = np.copy(random_displacement_vector)

        for dim in range(3):
            new_rcm_after_translation[dim] += random_displacement_vector[dim]
            if np.abs(new_rcm_after_translation[dim]) > half_box[dim]:
                new_rcm_after_translation[dim] -= 2.0 * random_displacement_vector[dim]
                actual_atom_displacement[dim] = -random_displacement_vector[dim]

        proposed[start:end, :] += actual_atom_displacement

        # --- Rotation, v04's intrinsic-ZYX matrix (lines 3880-3908) --------- #
        alpha_rot = (rng.rand() - 0.5) * 2.0 * params.max_rotation_angle_rad
        beta_rot = (rng.rand() - 0.5) * 2.0 * params.max_rotation_angle_rad
        gamma_rot = (rng.rand() - 0.5) * 2.0 * params.max_rotation_angle_rad

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma_rot), -np.sin(gamma_rot)],
            [0, np.sin(gamma_rot), np.cos(gamma_rot)],
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(beta_rot), 0, np.sin(beta_rot)],
            [0, 1, 0],
            [-np.sin(beta_rot), 0, np.cos(beta_rot)],
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(alpha_rot), -np.sin(alpha_rot), 0],
            [np.sin(alpha_rot), np.cos(alpha_rot), 0],
            [0, 0, 1],
        ], dtype=np.float64)

        rotation_matrix = Rz @ Ry @ Rx

        mol_coords_after_translation = proposed[start:end, :]
        mol_coords_relative_to_cm = (
            mol_coords_after_translation - new_rcm_after_translation
        )
        rotated_relative_coords = (rotation_matrix @ mol_coords_relative_to_cm.T).T
        proposed[start:end, :] = (
            rotated_relative_coords + new_rcm_after_translation
        )

        # Hard-wall rejection: a rigid move that lands the molecule inside the
        # frozen substrate — or inside another molecule — is undone, and the
        # molecule simply holds still this cycle.
        if gate_active and contact_clash(start, end, proposed[start:end, :]):
            proposed[start:end, :] = pre_move_coords
            continue

        last_moved_mol_idx = molecule_idx

    proposed_cluster = Cluster(
        coords=proposed,
        atomic_numbers=cluster.atomic_numbers,
        molecule_offsets=np.array(cluster.molecule_offsets, copy=True),
        molecules=cluster.molecules,
        box_length=cluster.box_length,
        box_lengths=cluster.box_lengths,
    )
    return UnifiedMoveResult(
        cluster=proposed_cluster,
        last_moved_molecule=last_moved_mol_idx,
        move_type=move_type_used,
    )


__all__ = [
    "FLIP_BAND_FRACTION",
    "INITIAL_QM_RETRIES",
    "MoveParams",
    "RotatableBond",
    "RotatableBondCache",
    "SMALL_BAND_FRACTION",
    "UnifiedMoveResult",
    "initialize_rotatable_bond_cache",
    "propose_unified_move",
    "rotate_around_bond",
]
