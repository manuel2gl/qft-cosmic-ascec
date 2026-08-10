"""Summary writer — the human-readable ``<stem>.out`` run report.

:class:`SummaryWriter` is an
:class:`~cosmic_ascec.annealing.AnnealingCallback` that produces the main
output file for a single annealing run. The layout is:

1. ASCII-art banner with the ASCEC version.
2. System / QM-method / schedule preamble (deterministic — same for any
   run of the same input).
3. ``History:`` block — one line per accepted configuration, plus an "N/A"
   line at the end of each temperature that ended without an LwE
   acceptance (accounting for QM calls not already reflected in history).
4. Closing statistics — totals (accepted, QM calls), lowest energy plus
   its configuration index, wall time.

The column widths and headings are a long-standing contract: downstream
plotting scripts and the parity harness match on specific columns.
Preserve them when editing.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path

from cosmic_ascec.annealing.engine import (
    AnnealingCallback,
    ConfigAccepted,
    RunFinish,
    RunStart,
    TemperatureComplete,
)
from cosmic_ascec.elements import Z_TO_SYMBOL, molecular_formula
from cosmic_ascec.file_formats.asc_schema import (
    AscConfig,
    QMProgram,
    QuenchingRoute,
)
from cosmic_ascec.geometry.box import calculate_optimal_box_length
from cosmic_ascec.geometry.molecule import Molecule

# v04 centres banner text in a 75-column field (``center_text``, line 1023).
_BANNER_WIDTH = 75
_RULE = "=" * _BANNER_WIDTH
_INNER_DIVIDER = "=" * 60  # v04's annealing-history separator (lines 1097 / 20819).

# v04's version line is ``* ASCEC-v04: Feb 2026 *`` (ASCEC_VERSION, line 64).
# v05 reports its own decomposition build — the one honest divergence in an
# otherwise byte-identical banner.
ASCEC_VERSION = "* COSMIC ASCEC v05 *"

# v04's ASCII-art logo — copied verbatim from ``write_simulation_summary``
# (ascec-v04.py lines 1033-1043), trailing spaces included.
_LOGO_ART = [
    "                             √≈≠==≈                                  ",
    "   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ",
    "     ÷++÷       ÷++÷           =++=                     ÷×××××=      ",
    "     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ",
    "     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ",
    "     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ",
    "     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ",
    "     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ",
    "      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ",
    "       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ",
    "          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ",
]

_PROGRAM_NAME = {
    QMProgram.GAUSSIAN: "Gaussian",
    QMProgram.ORCA: "Orca",
    QMProgram.XTB: "Xtb",
}


def _center(text: str) -> str:
    """v04 ``center_text`` — ``str.center`` in a 75-column field (line 1023)."""
    return text.center(_BANNER_WIDTH)


def _banner() -> list[str]:
    """The 30-line ASCII-art header — verbatim v04 ``write_simulation_summary``
    lines 1027-1056."""
    lines = [
        _RULE,
        "",
        _center("*********************"),
        _center("*     A S C E C     *"),
        _center("*********************"),
        "",
    ]
    lines.extend(_LOGO_ART)
    lines.extend([
        "",
        "",
        _center("Universidad de Antioquia - Medellín - Colombia"),
        "",
        "",
        _center("Annealing Simulado Con Energía Cuántica"),
        "",
        _center(ASCEC_VERSION),
        "",
        _center("Química Física Teórica - QFT"),
        "",
        "",
        _RULE,
    ])
    return lines


def _preamble(config: AscConfig, seed: int) -> list[str]:
    """The system / QM / schedule block — verbatim v04 lines 1057-1099."""
    # v04 line 1057 — a blank line between the banner rule and the preamble.
    lines: list[str] = [""]

    # --- Elemental composition (v04 lines 967-973, 1058-1061) -------------- #
    z_counts: Counter[int] = Counter()
    for molecule in config.molecules:
        for atom in molecule.atoms:
            z_counts[atom[0]] += 1
    max_symbol_len = max(
        (len(Z_TO_SYMBOL.get(z, f"Unk({z})")) for z in z_counts), default=1
    )
    lines.append("Elemental composition of the system:")
    for z in sorted(z_counts):
        symbol = Z_TO_SYMBOL.get(z, f"Unk({z})")
        lines.append(f"   {symbol:<{max_symbol_len}} {z_counts[z]:>3}")
    lines.append(f"There are a total of {config.total_atoms:>2} nuclei")

    # --- Box length + box-volume advice (v04 lines 1063-1069) -------------- #
    lines.append("")
    lines.append(f"Cube's length = {config.box.cube_length_angstrom:.2f} A")

    molecules = [Molecule.from_spec(spec) for spec in config.molecules]
    report = calculate_optimal_box_length(molecules)
    recs = report.recommendations
    b15 = recs["15.0%"].box_length_angstrom if "15.0%" in recs else 0.0
    b20 = recs["20.0%"].box_length_angstrom if "20.0%" in recs else 0.0
    b25 = recs["25.0%"].box_length_angstrom if "25.0%" in recs else 0.0
    method_label = "Method A" if report.has_primary_hbonds else "Method B"
    if b20 > 0:
        lines.append(
            f"Box suggestion ({method_label}): "
            f"{b15:.1f} / {b20:.1f} / {b25:.1f} A (15/20/25%)"
        )
    if report.max_molecular_extent > 0:
        lines.append(
            f"Largest molecular extent: {report.max_molecular_extent:.2f} A"
        )

    # --- Molecules (v04 lines 1071-1074) ----------------------------------- #
    lines.append("")
    lines.append(f"Number of molecules: {config.num_molecules}")
    lines.append("")
    lines.append("Molecular composition")
    max_label_len = max(
        (len(m.label) for m in config.molecules), default=1
    )
    for molecule in config.molecules:
        zs = tuple(atom[0] for atom in molecule.atoms)
        lines.append(f"  {molecule.label:<{max_label_len}} {molecular_formula(zs)}")

    # --- Move caps (v04 lines 1077-1078) ----------------------------------- #
    lines.append("")
    lines.append(
        f"Maximum displacement of each mass center = "
        f"{config.moves.max_displacement_angstrom:.2f} A"
    )
    lines.append(
        f"Maximum rotation angle = {config.moves.max_rotation_radian:.2f} radians"
    )

    # Conformational-sampling block — what v04 only prints to stderr
    # (ascec-v04.py lines 20507-20513); we also persist it to the .out so
    # the user can confirm dihedral moves were configured after the run.
    conf_prob = config.moves.conformational_probability
    if conf_prob > 0.0:
        import math
        max_dihedral_deg = math.degrees(config.moves.max_dihedral_radian)
        lines.append(
            f"Conformational sampling: {conf_prob * 100:.1f}% probability, "
            f"max dihedral rotation = ±{max_dihedral_deg:.1f}°"
        )
    else:
        lines.append("Conformational sampling: disabled (rigid-body moves only)")

    # --- QM block (v04 lines 1081-1085) ------------------------------------ #
    lines.append("")
    program = _PROGRAM_NAME.get(config.qm.program, "Unknown")
    lines.append(f"Energy calculated with {program}")
    lines.append(f" Hamiltonian: {config.qm.method or 'Not specified'}")
    lines.append(f" Basis set: {config.qm.basis_set or 'Not specified'}")
    lines.append(
        f" Charge = {config.qm.charge}   Multiplicity = {config.qm.multiplicity}"
    )

    # --- Seed + energy-eval message (v04 lines 1087-1090) ------------------ #
    lines.append("")
    lines.append(f"Seed = {seed:>6}")
    lines.append("")
    lines.append("** Energy will be evaluated **")
    lines.append("")

    # --- Quenching route (v04 lines 1014-1018) ----------------------------- #
    if config.schedule.route is QuenchingRoute.LINEAR:
        sched = config.schedule.linear
        lines.append("Linear quenching route.")
        lines.append(
            f"  To = {sched.initial_temperature:.1f} K    "
            f"dT = {sched.delta_temperature:.1f}     "
            f"nT = {sched.num_steps} steps"
        )
    else:
        sched = config.schedule.geometric
        percent_decrease = (1.0 - sched.factor) * 100.0
        lines.append("Geometrical quenching route.")
        lines.append(
            f"  To = {sched.initial_temperature:.1f} K  "
            f"%dism = {percent_decrease:.1f} %  "
            f"nT = {sched.num_steps} steps"
        )

    # --- History header (v04 lines 1097-1099) ------------------------------ #
    lines.append("")
    lines.append(_INNER_DIVIDER)
    lines.append("")
    lines.append("History: [T(K), E(u.a.), n-eval, Criterion]")
    lines.append("")
    return lines


def _format_wall_time(seconds: float) -> str:
    delta = timedelta(seconds=int(seconds))
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    secs = delta.seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{delta.days} days {hours} h {minutes} min {secs} s {millis} ms"


class SummaryWriter(AnnealingCallback):
    """Write the banner, the running history block, and the closing statistics."""

    def __init__(self, run_dir: Path, seed: int, stem: str) -> None:
        run_dir = Path(run_dir)
        self.seed = seed
        self.out_path = run_dir / f"{stem}.out"

    def on_run_start(self, event: RunStart) -> None:
        """Write the banner and the system / QM / schedule preamble."""
        lines = _banner() + _preamble(event.config, self.seed)
        self.out_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    def on_config_accepted(self, event: ConfigAccepted) -> None:
        """Append one history line for this accepted configuration."""
        self._append_history(
            event.temperature, event.energy, event.history_n_eval, event.criterion
        )

    def on_temperature_complete(self, event: TemperatureComplete) -> None:
        """Append v04's 'N/A' history line accounting for unlogged QM calls."""
        self._append_history(
            event.temperature, event.energy, event.history_n_eval, "N/A"
        )

    def on_run_finish(self, event: RunFinish) -> None:
        """Append the closing statistics block (v04 lines 20817-20844)."""
        result = event.result
        lines = [
            "",
            _INNER_DIVIDER,
            "",
            "  ** Normal annealing termination **",
            f"  Total Wall time: {_format_wall_time(result.wall_time_seconds)}",
            f"  Energy was evaluated {result.total_qm_calls} times",
            "",
            f"Energy evolution in tvse_{self.seed}.dat",
            f"Configurations accepted by Max.-Boltz. statistics = "
            f"{result.boltzmann_accepted}",
            f"Accepted lower energy configurations = {result.lower_energy_accepted}",
            f"Accepted configurations in result_{self.seed}.xyz = "
            f"{result.total_accepted}",
            f"Lowest energy configuration in rless_{self.seed}.out",
            f"Lowest energy = {result.lowest_energy:.8f} u.a. "
            f"(Config. {result.lowest_config_index})",
            "",
            _INNER_DIVIDER,
        ]
        with self.out_path.open("a", encoding='utf-8') as fh:
            fh.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _append_history(
        self, temperature: float, energy: float, n_eval: int, criterion: str
    ) -> None:
        with self.out_path.open("a", encoding='utf-8') as fh:
            fh.write(
                f"  {temperature:>8.2f} {energy:>12.6f} "
                f"{n_eval:>7} {criterion:>8}\n"
            )


__all__ = ["SummaryWriter"]
