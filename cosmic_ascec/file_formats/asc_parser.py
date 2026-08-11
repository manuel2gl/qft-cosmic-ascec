"""Line-by-line parser for ``.asc`` input files.

The ``.asc`` format is a hand-written, positional plain-text file: each line
carries one piece of configuration (line 1 = mode + #configs, line 2 = cube
length, line 3 = quenching route, ...), molecule definitions in
``*``-delimited blocks at the bottom, and an optional embedded protocol /
QM template appended at the end.

This parser turns one ``.asc`` file into a typed
:class:`~cosmic_ascec.file_formats.asc_schema.AscConfig`. Units are
normalized along the way (degrees → radians, percentages → fractions),
numeric fields are validated, and the embedded protocol/template (if
present) are split out into their own sections of the returned object.

A second reader inside :mod:`cosmic_ascec.workflow.stages` accepts the same
files for the historical workflow path. The parity test suite pins both
readers against each other on every bundled example so they cannot drift.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from cosmic_ascec.elements.data import SYMBOL_TO_Z
from cosmic_ascec.exceptions import AscParseError
from cosmic_ascec.file_formats.asc_schema import (
    AscConfig,
    AtomRow,
    BoxSpec,
    CycleSpec,
    EmbeddedTemplate,
    GeometricSchedule,
    LinearSchedule,
    MoleculeSpec,
    MoveSpec,
    ProtocolSpec,
    QMProgram,
    QMSpec,
    QuenchingRoute,
    ScheduleSpec,
    SimulationMode,
)


# Matches v04's PROTOCOL_MARKER_RE (line 99): a bare ".asc," or an explicit
# "<name>.asc," marker that opens the protocol block.
_PROTOCOL_MARKER_RE = re.compile(
    r"^\s*(?:\.asc|[^,\s#]+\.asc)\s*,",
    re.IGNORECASE,
)


# Matches v04's embedded-template header regex (line 1842).
_TEMPLATE_HEADER_RE = re.compile(
    r"^\s*#\s*(orca|gaussian|xtb)\s+(\S+)\s*$",
    re.IGNORECASE,
)


_TEMPLATE_EXTENSION = {
    "orca": ".inp",
    "gaussian": ".com",
    "xtb": ".xtb",
}


_TEMPLATE_PROGRAM = {
    "orca": QMProgram.ORCA,
    "gaussian": QMProgram.GAUSSIAN,
    "xtb": QMProgram.XTB,
}


def parse_asc(source: str | os.PathLike[str]) -> AscConfig:
    """Parse a ``.asc`` file into a typed :class:`AscConfig`.

    Raises:
        AscParseError: on any malformed line. The exception carries the source
            path and the offending 1-based line number.
    """
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AscParseError(f"could not read .asc file: {exc}", path=str(path)) from exc

    raw_lines = text.splitlines(keepends=True)
    str_path = str(path)

    header = _parse_header(raw_lines, str_path)
    molecules, header_end_idx = _parse_molecules(raw_lines, header.expected_molecules,
                                                 first_idx=header.consumed_through + 1,
                                                 path=str_path)
    protocol = _parse_protocol(raw_lines, start_idx=header_end_idx)
    templates = _parse_embedded_templates(raw_lines)

    return AscConfig(
        mode=header.mode,
        num_configurations=header.num_configurations,
        box=header.box,
        schedule=header.schedule,
        cycles=header.cycles,
        moves=header.moves,
        qm=header.qm,
        molecules=tuple(molecules),
        protocol=protocol,
        embedded_templates=tuple(templates),
        source_path=str_path,
    )


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


class _HeaderResult:
    """Bundle of parsed-header values plus bookkeeping for downstream phases."""

    __slots__ = (
        "mode",
        "num_configurations",
        "box",
        "schedule",
        "cycles",
        "moves",
        "qm",
        "expected_molecules",
        "consumed_through",
    )

    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _clean(line: str) -> str:
    """Strip ``#``/``!`` comments and surrounding whitespace, mirroring v04."""
    cleaned = line
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0]
    if "!" in cleaned:
        cleaned = cleaned.split("!", 1)[0]
    return cleaned.replace("\xa0", " ").strip()


def _require_parts(parts: List[str], count: int, line_no: int, path: str, what: str) -> None:
    if len(parts) < count:
        raise AscParseError(
            f"expected at least {count} value(s) for {what}, got {len(parts)}: "
            f"'{' '.join(parts)}'",
            line=line_no,
            path=path,
        )


def _to_int(value: str, line_no: int, path: str, what: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise AscParseError(f"expected integer for {what}, got '{value}'",
                            line=line_no, path=path) from exc


def _to_float(value: str, line_no: int, path: str, what: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise AscParseError(f"expected float for {what}, got '{value}'",
                            line=line_no, path=path) from exc


def _parse_header(raw_lines: List[str], path: str) -> _HeaderResult:
    """Parse the 13-line fixed configuration block (v04 phase 1)."""
    config_lines_parsed = 0
    last_consumed_idx = -1

    mode: Optional[SimulationMode] = None
    num_configurations = 0
    box: Optional[BoxSpec] = None
    route: Optional[QuenchingRoute] = None
    linear: Optional[LinearSchedule] = None
    geometric: Optional[GeometricSchedule] = None
    cycles: Optional[CycleSpec] = None
    max_disp = max_rot = 0.0
    conformational_prob = max_dihedral_rad = 0.0
    qm_program: Optional[QMProgram] = None
    qm_alias = qm_method = ""
    qm_basis: Optional[str] = None
    qm_nprocs = 1
    ascec_nprocs: Optional[int] = None
    charge = 0
    multiplicity = 1
    expected_molecules = 0

    for idx, raw_line in enumerate(raw_lines):
        if config_lines_parsed >= 13:
            break
        line = _clean(raw_line)
        if not line:
            continue
        parts = line.split()
        line_no = idx + 1

        match config_lines_parsed:
            case 0:  # mode + number of configurations
                _require_parts(parts, 2, line_no, path, "simulation mode + count")
                raw_mode = _to_int(parts[0], line_no, path, "simulation mode")
                if raw_mode not in (0, 1):
                    raise AscParseError(
                        f"simulation mode must be 0 or 1, got {raw_mode}",
                        line=line_no, path=path,
                    )
                mode = SimulationMode(raw_mode)
                num_configurations = _to_int(parts[1], line_no, path, "number of configurations")
            case 1:  # cube length
                _require_parts(parts, 1, line_no, path, "cube length")
                box = BoxSpec(cube_length_angstrom=_to_float(parts[0], line_no, path, "cube length"))
            case 2:  # quenching route
                _require_parts(parts, 1, line_no, path, "quenching route")
                raw_route = _to_int(parts[0], line_no, path, "quenching route")
                if raw_route not in (1, 2):
                    raise AscParseError(
                        f"quenching route must be 1 (linear) or 2 (geometric), got {raw_route}",
                        line=line_no, path=path,
                    )
                route = QuenchingRoute(raw_route)
            case 3:  # linear schedule
                _require_parts(parts, 3, line_no, path, "linear quenching params")
                linear = LinearSchedule(
                    initial_temperature=_to_float(parts[0], line_no, path, "linear Ti"),
                    delta_temperature=_to_float(parts[1], line_no, path, "linear ΔT"),
                    num_steps=_to_int(parts[2], line_no, path, "linear num_steps"),
                )
            case 4:  # geometric schedule (note v04's % → factor conversion)
                _require_parts(parts, 3, line_no, path, "geometric quenching params")
                temp_init = _to_float(parts[0], line_no, path, "geometric Ti")
                raw_factor = _to_float(parts[1], line_no, path, "geometric factor")
                if route is QuenchingRoute.GEOMETRIC:
                    if not (0.0 < raw_factor < 100.0):
                        raise AscParseError(
                            f"geometric factor must be a percentage between 0 and 100, "
                            f"got '{parts[1]}'",
                            line=line_no, path=path,
                        )
                    factor = 1.0 - raw_factor / 100.0
                else:
                    factor = raw_factor
                geometric = GeometricSchedule(
                    initial_temperature=temp_init,
                    factor=factor,
                    num_steps=_to_int(parts[2], line_no, path, "geometric num_steps"),
                )
            case 5:  # max MC cycles + optional floor
                _require_parts(parts, 1, line_no, path, "max MC cycles")
                max_cycles = _to_int(parts[0], line_no, path, "max MC cycles")
                floor = _to_int(parts[1], line_no, path, "floor value") if len(parts) >= 2 else None
                cycles = CycleSpec(max_cycles_per_temperature=max_cycles, floor_value=floor)
            case 6:  # max displacement + rotation (radians on disk)
                _require_parts(parts, 2, line_no, path, "max displacement + rotation")
                max_disp = _to_float(parts[0], line_no, path, "max displacement")
                max_rot = _to_float(parts[1], line_no, path, "max rotation")
            case 7:  # conformational %, dihedral cap in degrees
                _require_parts(parts, 2, line_no, path, "conformational sampling")
                pct = _to_float(parts[0], line_no, path, "conformational %")
                if not 0.0 <= pct <= 100.0:
                    raise AscParseError(
                        f"conformational percent must be in [0, 100], got '{parts[0]}'",
                        line=line_no, path=path,
                    )
                conformational_prob = pct / 100.0
                deg = _to_float(parts[1], line_no, path, "max dihedral angle (deg)")
                if not 0.0 <= deg <= 180.0:
                    raise AscParseError(
                        f"max dihedral angle must be in [0, 180] degrees, got '{parts[1]}'",
                        line=line_no, path=path,
                    )
                max_dihedral_rad = math.radians(deg)
            case 8:  # QM program index + alias
                _require_parts(parts, 2, line_no, path, "QM program selector")
                raw_program = _to_int(parts[0], line_no, path, "QM program index")
                if raw_program not in (1, 2, 3):
                    raise AscParseError(
                        f"QM program index must be 1 (Gaussian), 2 (ORCA), or 3 (xTB); "
                        f"got {raw_program}",
                        line=line_no, path=path,
                    )
                qm_program = QMProgram(raw_program)
                qm_alias = parts[1]
            case 9:  # method + optional basis
                _require_parts(parts, 1, line_no, path, "QM method")
                qm_method = parts[0]
                qm_basis = parts[1] if len(parts) >= 2 else None
            case 10:  # nprocs (QM + optional ASCEC)
                _require_parts(parts, 1, line_no, path, "QM nprocs")
                qm_nprocs = _to_int(parts[0], line_no, path, "QM nprocs")
                ascec_nprocs = _to_int(parts[1], line_no, path, "ASCEC nprocs") if len(parts) >= 2 else None
            case 11:  # charge + multiplicity
                _require_parts(parts, 2, line_no, path, "charge + multiplicity")
                charge = _to_int(parts[0], line_no, path, "charge")
                multiplicity = _to_int(parts[1], line_no, path, "multiplicity")
            case 12:  # number of molecules
                _require_parts(parts, 1, line_no, path, "number of molecules")
                expected_molecules = _to_int(parts[0], line_no, path, "number of molecules")
                if expected_molecules < 1:
                    raise AscParseError(
                        f"number of molecules must be >= 1, got {expected_molecules}",
                        line=line_no, path=path,
                    )

        config_lines_parsed += 1
        last_consumed_idx = idx

    if config_lines_parsed < 13:
        raise AscParseError(
            f"expected 13 header lines, found only {config_lines_parsed}",
            path=path,
        )

    assert mode is not None and box is not None and route is not None
    assert linear is not None and geometric is not None and cycles is not None
    assert qm_program is not None

    return _HeaderResult(
        mode=mode,
        num_configurations=num_configurations,
        box=box,
        schedule=ScheduleSpec(route=route, linear=linear, geometric=geometric),
        cycles=cycles,
        moves=MoveSpec(
            max_displacement_angstrom=max_disp,
            max_rotation_radian=max_rot,
            conformational_probability=conformational_prob,
            max_dihedral_radian=max_dihedral_rad,
        ),
        qm=QMSpec(
            program=qm_program,
            alias=qm_alias,
            method=qm_method,
            basis_set=qm_basis,
            qm_nprocs=qm_nprocs,
            ascec_nprocs=ascec_nprocs,
            charge=charge,
            multiplicity=multiplicity,
        ),
        expected_molecules=expected_molecules,
        consumed_through=last_consumed_idx,
    )


def _parse_molecules(
    raw_lines: List[str],
    expected: int,
    *,
    first_idx: int,
    path: str,
) -> Tuple[List[MoleculeSpec], int]:
    """Parse the ``*``-delimited molecule blocks (v04 phase 2).

    Two delimiters are recognised. ``*`` is the v04 separator. ``*$`` marks a
    *frozen* boundary: a block is frozen when the delimiter above it **and**
    the delimiter below it are both ``*$``. That rule keeps the shared-delimiter
    chain unambiguous — a mobile block next to a frozen one still has a plain
    ``*`` on its far side, so it is not mistaken for frozen::

        *      -> opens "water"          (water: *, *$   -> mobile)
        *$     -> closes "water", opens "site"
        *$     -> closes "site"          (site:  *$, *$  -> FROZEN)
        *      -> closes "water2"        (water2: *$, *  -> mobile)

    Returns the molecules plus the index of the line where parsing stopped
    (the protocol-marker line, or ``len(raw_lines)`` if none was found).
    """
    molecules: List[MoleculeSpec] = []
    reading = False
    expected_atoms = 0
    label = ""
    atoms: List[AtomRow] = []
    end_idx = len(raw_lines)
    # The delimiter that opened the block currently being read ("*" or "*$").
    open_delim = ""

    for idx in range(first_idx, len(raw_lines)):
        raw_line = raw_lines[idx]
        line_no = idx + 1

        # v04 break condition: hit the protocol block.
        if "# Protocol" in raw_line or "# protocol" in raw_line.lower() \
                or _is_protocol_marker(raw_line):
            end_idx = idx
            break

        line = _clean(raw_line)
        if not line:
            continue
        parts = line.split()
        if parts[0] in ("*", "*$"):
            delim = parts[0]
            if reading:
                if expected_atoms != len(atoms):
                    raise AscParseError(
                        f"molecule '{label}' declared {expected_atoms} atoms but "
                        f"got {len(atoms)}",
                        line=line_no, path=path,
                    )
                molecules.append(MoleculeSpec(
                    label=label,
                    atoms=tuple(atoms),
                    frozen=(open_delim == "*$" and delim == "*$"),
                ))
                expected_atoms = 0
                label = ""
                atoms = []
            reading = True
            open_delim = delim
            continue

        if not reading:
            # v04 emits a warning here and silently skips. We do the same to
            # stay forgiving on hand-edited inputs.
            continue

        if expected_atoms == 0:
            expected_atoms = _to_int(parts[0], line_no, path, "atom count")
        elif not label:
            label = parts[0]
        else:
            if len(atoms) >= expected_atoms:
                raise AscParseError(
                    f"molecule '{label}' has more atom rows than its declared count "
                    f"({expected_atoms}); missing '*' delimiter?",
                    line=line_no, path=path,
                )
            if len(parts) < 4:
                raise AscParseError(
                    f"atom row needs symbol + 3 coordinates, got '{line}'",
                    line=line_no, path=path,
                )
            symbol = parts[0]
            atomic_num = SYMBOL_TO_Z.get(symbol)
            if atomic_num is None:
                raise AscParseError(
                    f"unknown element symbol '{symbol}'",
                    line=line_no, path=path,
                )
            x = _to_float(parts[1], line_no, path, "x coordinate")
            y = _to_float(parts[2], line_no, path, "y coordinate")
            z = _to_float(parts[3], line_no, path, "z coordinate")
            atoms.append((atomic_num, x, y, z))

    # Trailing molecule without a closing ``*`` (v04 also handles this). It has
    # no closing delimiter, so by the both-sides rule it is never frozen — an
    # unterminated ``*$`` block is a typo we refuse rather than guess at.
    if reading and expected_atoms > 0:
        if expected_atoms != len(atoms):
            raise AscParseError(
                f"final molecule '{label}' declared {expected_atoms} atoms but got "
                f"{len(atoms)}; block not properly closed",
                path=path,
            )
        if open_delim == "*$":
            raise AscParseError(
                f"molecule '{label}' opens with '*$' but is never closed; a frozen "
                f"block needs '*$' on both sides",
                path=path,
            )
        molecules.append(MoleculeSpec(label=label, atoms=tuple(atoms)))

    if len(molecules) != expected:
        raise AscParseError(
            f"header declared {expected} molecules but found {len(molecules)}",
            path=path,
        )

    return molecules, end_idx


def _is_protocol_marker(raw_line: str) -> bool:
    """Reproduce v04's :func:`is_protocol_marker_line` (line 105)."""
    if not raw_line:
        return False
    stripped = raw_line.strip().lstrip("﻿")
    if not stripped:
        return False
    if "#" in stripped:
        stripped = stripped.split("#", 1)[0].strip()
    if not stripped:
        return False
    return bool(_PROTOCOL_MARKER_RE.match(stripped))


def _parse_protocol(raw_lines: List[str], *, start_idx: int) -> Optional[ProtocolSpec]:
    """Collect the ``# Protocol`` block (v04: ``extract_protocol_from_input``)."""
    protocol_lines: List[str] = []
    in_section = False

    for raw_line in raw_lines[start_idx:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
            if not stripped:
                continue
        if _is_protocol_marker(stripped):
            in_section = True
            protocol_lines.append(stripped)
            continue
        if in_section:
            protocol_lines.append(stripped)
            if not stripped.endswith(",") and not stripped.endswith("."):
                break

    if not protocol_lines:
        return None

    raw_text = " ".join(" ".join(line.split()) for line in protocol_lines)
    raw_text = " ".join(raw_text.split())
    steps = tuple(seg.strip() for seg in raw_text.split(",") if seg.strip())
    return ProtocolSpec(raw_text=raw_text, steps=steps)


def _parse_embedded_templates(raw_lines: Iterable[str]) -> List[EmbeddedTemplate]:
    """Extract every ``#orca/#gaussian/#xtb <label>`` block in file order.

    Mirrors v04's :func:`extract_embedded_qm_template` (line 1821), but returns
    *all* templates at once so the workflow stage can pick by label later.
    """
    templates: List[EmbeddedTemplate] = []
    current_program: Optional[str] = None
    current_label: Optional[str] = None
    collected: List[str] = []

    def flush() -> None:
        if current_program is None or current_label is None:
            return
        content = "".join(collected).strip() + "\n"
        if not content.strip():
            return
        templates.append(EmbeddedTemplate(
            program=_TEMPLATE_PROGRAM[current_program],
            label=current_label,
            content=content,
            extension=_TEMPLATE_EXTENSION[current_program],
        ))

    for raw_line in raw_lines:
        match = _TEMPLATE_HEADER_RE.match(raw_line)
        if match:
            flush()
            current_program = match.group(1).lower()
            current_label = match.group(2).strip()
            collected = []
            continue
        if current_program is not None:
            collected.append(raw_line)

    flush()
    return templates


__all__ = ["parse_asc"]
