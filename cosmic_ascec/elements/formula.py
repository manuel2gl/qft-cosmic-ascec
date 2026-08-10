"""Molecular formulas from atomic numbers — v04's ordering rule.

Carbon first, hydrogen second, then everything else by ascending
electronegativity; a count of one is left off. That is what ``ascec-v04.py``
used (``get_molecular_formula_string``, lines 858-890) and the run reports are
read against it, so it is preserved exactly rather than replaced with strict
Hill notation.

Lives in ``elements`` because ``Z_TO_SYMBOL`` and ``ELECTRONEGATIVITY`` are its
only dependencies and this package is the leaf that owns them — so the ASCEC
summary writer and the COSMIC clustering side can both reach it without either
importing the other.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from cosmic_ascec.elements.data import ELECTRONEGATIVITY, Z_TO_SYMBOL

__all__ = ["element_sort_key", "formula_from_counts", "molecular_formula"]

#: Stands in for an atomic number outside the table — including the ``0`` an
#: unrecognised XYZ element symbol parses to
#: (:func:`~cosmic_ascec.clustering.features.xyz_input._parse_atom_line`).
UNKNOWN_SYMBOL = "X"


def element_sort_key(symbol: str) -> tuple[int, float]:
    """C, then H, then ascending electronegativity; unknowns last."""
    if symbol == "C":
        return (-2, 0.0)
    if symbol == "H":
        return (-1, 0.0)
    return (0, ELECTRONEGATIVITY.get(symbol, float("inf")))


def formula_from_counts(counts: Mapping[int, int]) -> str:
    """Formula for a ``{atomic number: count}`` map."""
    symbols: Counter[str] = Counter()
    for z, n in counts.items():
        symbols[Z_TO_SYMBOL.get(int(z), UNKNOWN_SYMBOL)] += n

    parts: list[str] = []
    for symbol in sorted(symbols, key=element_sort_key):
        parts.append(symbol)
        if symbols[symbol] > 1:
            parts.append(str(symbols[symbol]))
    return "".join(parts)


def molecular_formula(atomic_numbers: Iterable[int]) -> str:
    """Formula for a sequence of atomic numbers."""
    return formula_from_counts(Counter(int(z) for z in atomic_numbers))
