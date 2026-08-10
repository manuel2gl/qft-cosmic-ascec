"""One run, one system — the composition check every clustering run starts with.

COSMIC clusters *configurations of one system*, and every descriptor says so.
The nuclear repulsion, the rotational constants and the Cartesian RMSD are only
defined between structures built from the same atoms, and Z-standardising a
column across a pool that mixes C15H16O2 with C15H16O turns a chemical
difference into a distance without ever saying which it was. What comes out is
a dendrogram that separates molecules rather than conformers.

Nothing downstream catches it. :func:`~cosmic_ascec.clustering.rmsd.calculate_rmsd`
returns ``None`` for a pair of different atom counts and the caller records
``"N/A"`` and treats the structure as an outlier, so the mismatch is absorbed as
a clustering *result*; the feature vector never looks at composition at all. The
run finishes, writes motifs, and is wrong. Hence a check up front, and a stop.

The exception is a trajectory shell. ``--shell R`` keeps every solvent molecule
inside a radius, so the count — and with it the atom count and the formula —
varies from frame to frame by design; ``--nearest N`` is exempted alongside it,
since a named mixed solvent varies there too. The orchestrator recognises those
runs and never calls in here.

This module decides and describes: it prints nothing and raises nothing, so it
can be exercised on plain dicts without dragging in the clustering stack. The
orchestrator turns a failing report into a
:class:`~cosmic_ascec.exceptions.ClusteringError`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cosmic_ascec.elements import Z_TO_SYMBOL, element_sort_key, formula_from_counts

__all__ = [
    "Composition",
    "MAX_LISTED",
    "Offender",
    "StoichiometryReport",
    "check_stoichiometry",
    "composition_of",
    "format_report",
]

#: A pool's identity: ``(atomic number, count)`` pairs, ascending Z. A tuple so
#: it is hashable and can key the grouping dict.
Composition = tuple[tuple[int, int], ...]

#: Offenders named in full before the tail is summarised. Enough to see a
#: pattern, few enough that the explanation above them is not scrolled away.
MAX_LISTED = 15


def composition_of(atomic_numbers: Iterable[int]) -> Composition:
    """Which elements, how many of each — order-independent."""
    return tuple(sorted(Counter(int(z) for z in atomic_numbers).items()))


@dataclass(frozen=True)
class Offender:
    """One structure whose composition differs from the reference."""

    filename: str
    composition: Composition
    formula: str
    natoms: int
    delta: str          # 'H -2, O -1' — this structure relative to the reference


@dataclass(frozen=True)
class StoichiometryReport:
    """What :func:`check_stoichiometry` found. Empty ``offenders`` means pass."""

    reference: Composition
    reference_formula: str
    reference_natoms: int
    reference_files: int            # how many inputs carry the reference
    total_checked: int
    offenders: tuple[Offender, ...]
    unchecked: tuple[str, ...]      # no geometry parsed — not a disagreement

    @property
    def ok(self) -> bool:
        return not self.offenders

    @property
    def n_compositions(self) -> int:
        """Distinct compositions in the pool, the reference included."""
        if not self.total_checked:
            return 0
        return len({offender.composition for offender in self.offenders}) + 1


def check_stoichiometry(records: Sequence[Mapping[str, Any]]) -> StoichiometryReport:
    """Group *records* by composition and name the ones that disagree.

    *records* are the orchestrator's extracted-property dicts; only
    ``filename`` and ``final_geometry_atomnos`` are read.
    """
    groups: dict[Composition, list[str]] = {}
    unchecked: list[str] = []

    for record in records:
        name = str(record.get('filename', '<unnamed>'))
        atomnos = record.get('final_geometry_atomnos')
        # A record whose geometry never parsed says nothing about composition.
        # The orchestrator's own essential-feature filter reports and drops it a
        # few lines further on, so failing the whole run over it here would
        # blame the wrong file. `len` rather than `.size` so plain lists work.
        if atomnos is None or len(atomnos) == 0:
            unchecked.append(name)
            continue
        groups.setdefault(composition_of(atomnos), []).append(name)

    if not groups:
        return StoichiometryReport((), "", 0, 0, 0, (), tuple(unchecked))

    # Majority vote, earliest-seen breaking a tie (dicts keep insertion order,
    # so this is deterministic). The composition most inputs agree on is the
    # system under study; taking "whatever the first file was" instead would let
    # one stray structure turn the other 199 into offenders and bury the real
    # culprit in the tail of the list. For a two-file --compare the tie-break
    # degenerates to the first file, which is the intuitive answer there.
    first_seen = {composition: i for i, composition in enumerate(groups)}
    reference = min(groups, key=lambda c: (-len(groups[c]), first_seen[c]))
    reference_counts = dict(reference)

    offenders = tuple(sorted(
        (
            Offender(
                filename=name,
                composition=composition,
                formula=formula_from_counts(dict(composition)),
                natoms=sum(n for _, n in composition),
                delta=_delta(reference_counts, dict(composition)),
            )
            for composition, names in groups.items() if composition != reference
            for name in names
        ),
        key=lambda offender: offender.filename,
    ))

    return StoichiometryReport(
        reference=reference,
        reference_formula=formula_from_counts(reference_counts),
        reference_natoms=sum(reference_counts.values()),
        reference_files=len(groups[reference]),
        total_checked=sum(len(names) for names in groups.values()),
        offenders=offenders,
        unchecked=tuple(unchecked),
    )


def _delta(reference: Mapping[int, int], other: Mapping[int, int]) -> str:
    """'H -2, O -1' — what *other* has that the reference does not."""
    differences = [
        (Z_TO_SYMBOL.get(z, "X"), other.get(z, 0) - reference.get(z, 0))
        for z in set(reference) | set(other)
    ]
    differences = [(symbol, d) for symbol, d in differences if d]
    differences.sort(key=lambda pair: element_sort_key(pair[0]))
    return ", ".join(f"{symbol} {d:+d}" for symbol, d in differences)


def format_report(report: StoichiometryReport, *, max_listed: int = MAX_LISTED) -> str:
    """The user-facing message for a failing *report*. Never called when ok."""
    listed = report.offenders[:max_listed]
    name_width = max((len(offender.filename) for offender in listed), default=0)
    formula_width = max((len(offender.formula) for offender in listed), default=0)

    lines = [
        "the input structures do not all have the same composition.",
        "",
        "  COSMIC clusters configurations of one system: the feature vector and",
        "  the RMSD are only defined between structures built from the same atoms.",
        "",
        f"  Reference composition: {report.reference_formula}, "
        f"{report.reference_natoms} atoms "
        f"({report.reference_files} of {report.total_checked} files)",
        "",
        f"  {len(report.offenders)} file(s) differ:",
    ]
    for offender in listed:
        lines.append(
            f"    {offender.filename:<{name_width}}  {offender.formula:<{formula_width}}  "
            f"{offender.natoms:>4} atoms   {offender.delta}"
        )

    remaining = len(report.offenders) - len(listed)
    if remaining:
        # The count of distinct compositions is the diagnostic number when the
        # pool is a mess: "412 more, in 37 compositions" says "these are
        # unrelated systems", not "one file is broken".
        lines.append(f"    ... and {remaining} more, in "
                     f"{report.n_compositions - 1} distinct composition(s) overall.")

    lines += [
        "",
        "  Correct or remove them and run again. Composition is not checked for",
        "  --shell / --nearest runs, where the number of solvent molecules varies",
        "  by design.",
    ]
    return "\n".join(lines)
