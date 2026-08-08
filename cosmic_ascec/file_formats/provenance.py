"""``mapping.dat`` — what each clustered structure was, back in the dynamics.

A motif carved out of an MD trajectory is useless if you cannot say where it
came from. ``motifs_46/motif_01_shell_004.xyz`` names its source frame, but
``shell_004`` is the fourth *extracted* frame, which with ``--stride=7`` is
trajectory frame 21 — not something a reader can guess. And the simulation time
that frame sits at, and which solvent molecules were in its shell, are gone
entirely unless they are written down.

This module writes them down. ``mapping.dat`` lands in the work directory and
joins the three things that are otherwise never connected:

* the trajectory frame, its ``t=`` and its ``step=``, known only during
  extraction;
* the per-frame filename the clustering pipeline will use, which is assigned
  later by :func:`~cosmic_ascec.clustering.features.xyz_input.explode_multiframe_xyz`;
* the cluster and motif each structure ends up in, known only after clustering.

So it is written in two passes: :func:`write_mapping` at extraction time, and
:func:`update_mapping` once the motifs exist.

The file is also the switch that turns MD provenance on in the rest of COSMIC.
:func:`load_mapping` returns ``None`` when it is absent, and it is only ever
present in a trajectory work directory — so an ordinary ``cosmic xyz_dir`` run
finds nothing, and every comment line it writes is built exactly as before.
There is no flag to set and no mode to remember.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Written into, and looked for in, the run's output directory.
MAPPING_FILENAME = "mapping.dat"

#: Printed where a value is genuinely unknown. Never a zero or an empty string,
#: so "the trajectory had no step counter" cannot be misread as "step 0".
MISSING = "-"

_HEADER = (
    "# frame  file                traj        t(ps)         step   atoms  "
    "cluster  motif"
)
_SOLVENT_HEADER = "# solvent kept per frame (resname:resid, nearest first)"


@dataclass
class FrameRecord:
    """One extracted frame, and where it came from."""

    ordinal: int                    # 1-based position within the extracted file
    stem: str                       # per-frame filename stem, e.g. 'shell_004'
    traj_frame: int                 # 0-based index in the source trajectory
    atoms: int
    time_ps: Optional[float] = None
    step: Optional[int] = None
    solvent: Sequence[str] = ()     # 'MEN:44', nearest first
    cluster: Optional[int] = None
    motif: Optional[int] = None

    def describe(self) -> str:
        """One-line provenance for an XYZ comment, e.g. for a motif file."""
        parts = [f"traj frame {self.traj_frame}"]
        if self.time_ps is not None:
            parts.append(f"t= {self.time_ps:g} ps")
        if self.step is not None:
            parts.append(f"step= {self.step}")
        return "  ".join(parts)


def _format_row(record: FrameRecord) -> str:
    time_text = f"{record.time_ps:.3f}" if record.time_ps is not None else MISSING
    step_text = str(record.step) if record.step is not None else MISSING
    cluster_text = str(record.cluster) if record.cluster is not None else MISSING
    motif_text = str(record.motif) if record.motif is not None else MISSING
    return (f"{record.ordinal:7d}  {record.stem:<16s}  {record.traj_frame:>6d}  "
            f"{time_text:>11s}  {step_text:>11s}  {record.atoms:>5d}  "
            f"{cluster_text:>7s}  {motif_text:>5s}")


def write_mapping(path: str, records: Sequence[FrameRecord], *,
                  source: str, command: str = "") -> None:
    """Write the extraction half of ``mapping.dat``.

    Cluster and motif columns are filled with :data:`MISSING`; they are only
    knowable after clustering, and :func:`update_mapping` fills them in then.
    Writing them now rather than leaving them out keeps the file readable after
    ``--extract-only``, where clustering never runs at all.
    """
    lines = [
        f"# COSMIC MD frame mapping",
        f"# source: {source}",
    ]
    if command:
        lines.append(f"# command: {command}")
    lines += [
        "#",
        "# 'traj' is the 0-based frame index in the source trajectory; 'frame' and",
        "# 'file' are the position and name after extraction. They differ whenever",
        "# --stride, --first or --last is used.",
        "#",
        _HEADER,
    ]
    lines += [_format_row(record) for record in records]

    if any(record.solvent for record in records):
        lines += ["", _SOLVENT_HEADER]
        lines += [f"  {record.stem:<16s}  {' '.join(record.solvent)}"
                  for record in records if record.solvent]

    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_mapping(output_base_dir: str) -> Optional[Dict[str, FrameRecord]]:
    """Read ``mapping.dat`` from *output_base_dir*, keyed by filename stem.

    Returns ``None`` when the file is not there. Callers use that to mean "this
    is not a trajectory-derived run" and leave their output untouched, which is
    what keeps MD provenance out of ordinary COSMIC runs.
    """
    path = os.path.join(output_base_dir, MAPPING_FILENAME)
    if not os.path.isfile(path):
        return None

    records: Dict[str, FrameRecord] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split()
                if len(fields) < 8 or not fields[0].isdigit():
                    # The solvent section and anything unexpected: not a row.
                    continue
                records[fields[1]] = FrameRecord(
                    ordinal=int(fields[0]),
                    stem=fields[1],
                    traj_frame=int(fields[2]),
                    time_ps=None if fields[3] == MISSING else float(fields[3]),
                    step=None if fields[4] == MISSING else int(fields[4]),
                    atoms=int(fields[5]),
                    cluster=None if fields[6] == MISSING else int(fields[6]),
                    motif=None if fields[7] == MISSING else int(fields[7]),
                )
    except (OSError, ValueError):
        # A malformed mapping is not worth failing a completed clustering run
        # over; the output simply carries no provenance.
        return None
    return records or None


def describe(mapping: Optional[Mapping[str, FrameRecord]], filename: str) -> str:
    """Provenance text for *filename*, or ``""`` when there is none.

    Takes the filename as the pipeline knows it (``shell_004.xyz``) and tolerates
    a missing entry, so callers can stay a single unconditional expression.
    """
    if not mapping:
        return ""
    record = mapping.get(os.path.splitext(os.path.basename(filename))[0])
    return record.describe() if record else ""


def update_mapping(output_base_dir: str, cluster_of: Mapping[str, int],
                   motif_of: Optional[Mapping[str, int]] = None) -> bool:
    """Fill in the cluster and motif columns once clustering has finished.

    *cluster_of* maps each structure's filename to its cluster id, *motif_of*
    maps the representative filenames to their motif numbers. Both are keyed by
    filename rather than by position, because a motif representative is the
    cluster's lowest-energy or most central member, not its first.

    Returns False, having changed nothing, when there is no mapping to update —
    which is every run that did not come from a trajectory.
    """
    records = load_mapping(output_base_dir)
    if records is None:
        return False

    def _stem(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    for filename, cluster_id in cluster_of.items():
        record = records.get(_stem(filename))
        if record is not None:
            record.cluster = cluster_id

    for filename, motif_number in (motif_of or {}).items():
        record = records.get(_stem(filename))
        if record is not None:
            record.motif = motif_number

    _rewrite_columns(os.path.join(output_base_dir, MAPPING_FILENAME), records)
    return True


def _rewrite_columns(path: str, records: Mapping[str, FrameRecord]) -> None:
    """Rewrite the table rows in place, leaving comments and solvent untouched.

    Editing the existing file rather than regenerating it keeps the header,
    the recorded command and the solvent section exactly as extraction wrote
    them — this pass knows nothing about any of that.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return

    out: List[str] = []
    for line in lines:
        fields = line.split()
        if (not line.startswith("#") and len(fields) >= 8
                and fields[0].isdigit() and fields[1] in records):
            out.append(_format_row(records[fields[1]]))
        else:
            out.append(line)

    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
