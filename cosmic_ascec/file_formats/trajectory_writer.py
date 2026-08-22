"""Trajectory writers — the accepted-configuration geometry outputs.

:class:`TrajectoryWriter` is an :class:`~cosmic_ascec.annealing.AnnealingCallback`
that reproduces three v04 geometry files:

* ``result_<seed>.xyz``    — one XYZ frame per accepted configuration
  (v04 ``write_accepted_xyz`` / ``write_single_xyz_configuration``,
  ascec-v04.py lines 3517-3566).
* ``resultbox_<seed>.xyz`` — the same frames plus eight ``X`` markers at the
  cube corners, so a viewer can show the simulation box.
* ``rless_<seed>.out``     — the single lowest-energy configuration, grouped
  by molecule (v04 ``write_lowest_energy_config_file``, lines 4742-4770),
  written once at the end of the run.

The ``<seed>`` suffix matches v04's filename convention (accepted decision for
Session 5): the run's RNG seed doubles as the output identifier.
"""

from __future__ import annotations

from pathlib import Path

from cosmic_ascec.annealing.engine import (
    AnnealingCallback,
    ConfigAccepted,
    RunFinish,
    RunStart,
)
from cosmic_ascec.elements import Z_TO_SYMBOL
from cosmic_ascec.geometry.molecule import Cluster

# Symbol used for the eight cube-corner markers in resultbox_<seed>.xyz
# (v04 ``include_dummy_atoms`` path).
_BOX_MARKER_SYMBOL = "X"


def _restore_dummy_atom_symbol(mol_path: Path) -> None:
    # obabel writes the non-standard 'X' element as '*' in the MOL atom block;
    # restore it so viewers don't choke on the placeholder.
    try:
        text = mol_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return
    fixed_lines = []
    for line in text.splitlines(keepends=True):
        if len(line) >= 34 and line[31] == "*" and line[32] == " ":
            line = line[:31] + _BOX_MARKER_SYMBOL + line[32:]
        fixed_lines.append(line)
    try:
        mol_path.write_text("".join(fixed_lines), encoding='utf-8')
    except OSError:
        pass


def _atom_line(symbol: str, x: float, y: float, z: float) -> str:
    """One XYZ coordinate row in v04's ``result.xyz`` spacing."""
    return f"{symbol:<3}{x: 13.6f}{y: 13.6f}{z: 13.6f}"


def box_corner_coords(box_lengths) -> list[tuple[float, float, float]]:
    """Eight box corners at +/-L_i/2, x fastest then y then z (v04 order).

    Accepts a single edge (a cube, what v04 could express) or ``(Lx, Ly, Lz)``
    for a rectangular prism. Shared with the ``ascec <file> box`` command so
    the two box files cannot drift apart.
    """
    if isinstance(box_lengths, (int, float)):
        hx = hy = hz = float(box_lengths) / 2.0
    else:
        hx, hy, hz = (float(v) / 2.0 for v in box_lengths)
    return [
        (sx * hx, sy * hy, sz * hz)
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    ]


def box_label(box_lengths) -> str:
    """``BoxL=`` text for an XYZ comment line: one number for a cube, three
    for a prism."""
    if isinstance(box_lengths, (int, float)):
        return f"BoxL={float(box_lengths):.1f} A"
    lx, ly, lz = (float(v) for v in box_lengths)
    if lx == ly == lz:
        return f"BoxL={lx:.1f} A"
    return f"BoxL={lx:.1f}x{ly:.1f}x{lz:.1f} A"


# Backwards-compatible private alias (this module's original spelling).
_box_corner_coords = box_corner_coords


class TrajectoryWriter(AnnealingCallback):
    """Append accepted XYZ frames and write the final lowest-energy structure."""

    def __init__(self, run_dir: Path, seed: int) -> None:
        run_dir = Path(run_dir)
        self.result_path = run_dir / f"result_{seed}.xyz"
        self.resultbox_path = run_dir / f"resultbox_{seed}.xyz"
        self.rless_path = run_dir / f"rless_{seed}.out"

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def on_run_start(self, event: RunStart) -> None:  # noqa: ARG002
        """Truncate the trajectory files so a re-run never appends to stale data."""
        self.result_path.write_text("", encoding='utf-8')
        self.resultbox_path.write_text("", encoding='utf-8')

    def on_config_accepted(self, event: ConfigAccepted) -> None:
        """Append one frame to ``result`` and one (box-marked) to ``resultbox``."""
        cluster = event.cluster
        header = (
            f"Configuration: {event.index} | E = {event.energy:.8f} a.u. "
            f"| T = {event.temperature:.1f} K"
        )
        atom_lines = [
            _atom_line(Z_TO_SYMBOL.get(z, "X"), x, y, zc)
            for z, (x, y, zc) in zip(cluster.atomic_numbers, cluster.coords)
        ]
        with self.result_path.open("a", encoding='utf-8') as fh:
            fh.write(f"{cluster.num_atoms}\n{header}\n")
            fh.write("\n".join(atom_lines) + "\n")

        box_header = (
            f"Configuration: {event.index} | E = {event.energy:.8f} a.u. "
            f"| {box_label(cluster.box_lengths or cluster.box_length)} (X box markers)"
        )
        corner_lines = [
            _atom_line(_BOX_MARKER_SYMBOL, x, y, z)
            for x, y, z in box_corner_coords(cluster.box_lengths or cluster.box_length)
        ]
        with self.resultbox_path.open("a", encoding='utf-8') as fh:
            fh.write(f"{cluster.num_atoms + len(corner_lines)}\n{box_header}\n")
            fh.write("\n".join(atom_lines + corner_lines) + "\n")

    def on_run_finish(self, event: RunFinish) -> None:
        """Write the lowest-energy configuration, generate .mol siblings, clean xtb scratch."""
        result = event.result
        self._write_rless(result.lowest_cluster, result.lowest_energy,
                          result.lowest_config_index)
        self._convert_xyz_results_to_mol()
        self._clean_xtb_scratch()

    def _convert_xyz_results_to_mol(self) -> None:
        """Generate result_<seed>.mol and resultbox_<seed>.mol via obabel, matching mono."""
        import shutil as _shutil
        import subprocess as _subprocess
        # Invoke the resolved path, not the bare name. conda-forge ships
        # obabel on Windows as a .bat shim, which CreateProcess will not
        # find from "obabel" alone (PATHEXT is consulted by shutil.which,
        # not by subprocess).
        _obabel = _shutil.which("obabel")
        if not _obabel:
            return
        for xyz_path in (self.result_path, self.resultbox_path):
            if not xyz_path.exists() or xyz_path.stat().st_size == 0:
                continue
            mol_path = xyz_path.with_suffix(".mol")
            try:
                _subprocess.run(
                    [_obabel, "-ixyz", str(xyz_path), "-omol", "-O", str(mol_path)],
                    capture_output=True, check=False, timeout=60,
                )
            except (OSError, _subprocess.SubprocessError):
                continue
            if mol_path.exists():
                _restore_dummy_atom_symbol(mol_path)

    def _clean_xtb_scratch(self) -> None:
        """Remove xtb scratch artifacts (charges/wbo/xtbtopo.mol/xtbrestart) — mono parity."""
        run_dir = self.result_path.parent
        for name in ("charges", "wbo", "xtbtopo.mol", "xtbrestart"):
            scratch = run_dir / name
            try:
                if scratch.exists():
                    scratch.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _write_rless(self, cluster: Cluster, energy: float, config_index: int) -> None:
        lines = [f"# Configuration: {config_index} | Energy = {energy:.8f} u.a."]
        for mol_idx, molecule in enumerate(cluster.molecules):
            start = int(cluster.molecule_offsets[mol_idx])
            end = int(cluster.molecule_offsets[mol_idx + 1])
            lines.append(str(molecule.num_atoms))
            lines.append(molecule.label)
            for z, (x, y, zc) in zip(
                cluster.atomic_numbers[start:end], cluster.coords[start:end]
            ):
                symbol = Z_TO_SYMBOL.get(z, "X")
                lines.append(f"{symbol:<3} {x: 12.6f} {y: 12.6f} {zc: 12.6f}")
        self.rless_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


__all__ = ["TrajectoryWriter", "box_corner_coords", "box_label"]
