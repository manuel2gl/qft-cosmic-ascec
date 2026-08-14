"""Refinement levels — what a representative is called, and at which stage.

A COSMIC protocol clusters three times, once after each computational stage, and
each pass writes out one representative per cluster. Those representatives used
to be named by sniffing the *input* filenames: ``motif_*`` in meant ``umotif_*``
out, and ``umotif_*`` in meant ``umotif_*`` out again. The ladder therefore
saturated after two rungs, and the last two stages both emitted ``umotif_NN``.

That collision is not cosmetic. In the w6 ultimate run, ``umotif_23`` is
``motif_41_opt`` in the geometry-refinement pass and ``motif_42_opt`` in the
energy-refinement pass — two different molecules wearing one label, with nothing
in the name to say which stage produced it. Tracing a final structure back to
its annealing geometry meant knowing which directory you were standing in.

So the level is now named explicitly, one distinct prefix per stage:

===================  ==============  ==============  =========================
Produced after       Representative  Folder          Its QM output
===================  ==============  ==============  =========================
geometry opt         ``candidate``   ``candidates``  ``candidate_NN_opt.out``
geometry refinement  ``motif``       ``motifs``      ``motif_NN_opt.out``
energy refinement    ``u_motif``     ``u_motifs``    ``u_motif_NN_opt.out``
===================  ==============  ==============  =========================

The protocol runner knows which stage precedes each ``cosmic`` call and passes
``--level`` accordingly (see :func:`stage_to_level`). Filename sniffing survives
only as the fallback for a bare ``cosmic <dir>`` run outside a protocol, and it
is a real three-rung ladder now — see
:func:`~cosmic_ascec.clustering.motifs.detect_motif_input_level`.

``u_motif`` still contains ``motif`` as a substring, so any pattern matching the
middle rung has to exclude the top one explicitly. :data:`MOTIF_ONLY_RE` is that
pattern; use it rather than hand-rolling the lookbehind, and never reach for
``str.startswith('motif')`` on a name that could be a ``u_motif``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Final, Optional, Tuple


@dataclass(frozen=True)
class Level:
    """One rung: how it is named on disk and how it is spoken about."""

    key: str        # canonical id, also the --level value
    label: str      # per-structure prefix -> candidate_01.xyz
    folder: str     # folder stem, suffixed with a count -> candidates_29/
    display: str    # human phrasing for progress lines
    display_one: str


CANDIDATE: Final = Level('candidate', 'candidate', 'candidates',
                         'candidates', 'candidate')
MOTIF: Final = Level('motif', 'motif', 'motifs', 'motifs', 'motif')
U_MOTIF: Final = Level('u_motif', 'u_motif', 'u_motifs',
                       'unique motifs', 'unique motif')

#: In ladder order. Index is the rung.
LEVELS: Final[Tuple[Level, ...]] = (CANDIDATE, MOTIF, U_MOTIF)

BY_KEY: Final[Dict[str, Level]] = {lv.key: lv for lv in LEVELS}

#: Protocol stage type -> the level its following ``cosmic`` pass should emit.
#: Keys are the stage ``type`` values produced by
#: :func:`~cosmic_ascec.workflow.protocol.parse_workflow_stages`.
STAGE_TO_LEVEL: Final[Dict[str, str]] = {
    'optimization': CANDIDATE.key,
    'refinement': MOTIF.key,
    'energy_refinement': U_MOTIF.key,
}

#: Superseded prefixes still present in runs on disk. Readers must accept these;
#: writers never emit them. ``umotif`` was the old name for both of the upper two
#: rungs, which is exactly the ambiguity this module exists to remove — when it
#: is seen on input it can only be resolved by which directory it came from.
LEGACY_LABELS: Final[Tuple[str, ...]] = ('umotif',)

#: Every per-structure prefix a reader may encounter, new and legacy.
ALL_LABELS: Final[Tuple[str, ...]] = tuple(lv.label for lv in LEVELS) + LEGACY_LABELS

#: Every folder stem a reader may encounter. Ordered most-refined first, so a
#: caller taking the first hit gets the latest stage present.
ALL_FOLDER_GLOBS: Final[Tuple[str, ...]] = (
    'u_motifs_*', 'umotifs_*', 'motifs_*', 'candidates_*',
)

#: Matches ``motif_12`` but not the ``motif_12`` inside ``u_motif_12`` or the one
#: inside the legacy ``umotif_12``. Both spellings must be excluded: ``u_`` for
#: the new top rung, bare ``u`` for the old one.
MOTIF_ONLY_RE: Final[re.Pattern] = re.compile(r'(?<!u)(?<!u_)(motif_\d+)', re.IGNORECASE)

#: Matches the top rung in either spelling, new or legacy.
U_MOTIF_RE: Final[re.Pattern] = re.compile(r'(u_?motif_\d+)', re.IGNORECASE)

#: Matches any rung, capturing (label, number). Ordered longest-first so
#: ``u_motif_07`` cannot be mis-read as ``motif_07``.
ANY_LABEL_RE: Final[re.Pattern] = re.compile(
    r'\b(u_motif|umotif|motif|candidate)_(\d+)', re.IGNORECASE)


def stage_to_level(stage_type: Optional[str]) -> Optional[str]:
    """Level key a ``cosmic`` pass should emit after ``stage_type``.

    ``None`` when the stage does not feed a clustering pass, or is unknown — the
    caller then leaves ``--level`` off and filename sniffing decides.
    """
    if not stage_type:
        return None
    return STAGE_TO_LEVEL.get(str(stage_type).strip().lower())


def resolve(key: Optional[str]) -> Optional[Level]:
    """:class:`Level` for a ``--level`` value, or ``None`` if unrecognised.

    Accepts the legacy ``umotif`` spelling so an old command line keeps working;
    it resolves to the top rung, which is what it always meant when it appeared
    on the *output* side.
    """
    if not key:
        return None
    norm = str(key).strip().lower().replace('-', '_')
    if norm in ('umotif', 'umotifs'):
        return U_MOTIF
    return BY_KEY.get(norm.rstrip('s') if norm not in BY_KEY else norm)


def next_level(current: Optional[str]) -> Level:
    """The rung above ``current``; the top rung is its own successor.

    Used by the sniffing fallback: seeing ``candidate_*`` on input means this
    pass produces ``motif_*``. Unknown or absent input means the bottom rung.
    """
    lv = resolve(current)
    if lv is None:
        return CANDIDATE
    idx = LEVELS.index(lv)
    return LEVELS[min(idx + 1, len(LEVELS) - 1)]


def label_of(name: str) -> Optional[str]:
    """Level label a filename carries, or ``None``.

    Longest-first matching, so ``u_motif_07_opt`` reports ``u_motif`` rather than
    the ``motif`` sitting inside it.
    """
    m = ANY_LABEL_RE.search(str(name))
    if not m:
        return None
    found = m.group(1).lower()
    return 'u_motif' if found in ('u_motif', 'umotif') else found


__all__ = [
    'Level', 'CANDIDATE', 'MOTIF', 'U_MOTIF', 'LEVELS', 'BY_KEY',
    'STAGE_TO_LEVEL', 'LEGACY_LABELS', 'ALL_LABELS', 'ALL_FOLDER_GLOBS',
    'MOTIF_ONLY_RE', 'U_MOTIF_RE', 'ANY_LABEL_RE',
    'stage_to_level', 'resolve', 'next_level', 'label_of',
]
