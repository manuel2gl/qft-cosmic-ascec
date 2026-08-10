"""TVSE writer — the temperature-vs-energy ledger.

:class:`TvseWriter` is an :class:`~cosmic_ascec.annealing.AnnealingCallback`
that reproduces v04's ``tvse_<seed>.dat`` file (v04 ``write_tvse_file``,
ascec-v04.py lines 4772-4793). One row is appended for every accepted
configuration — the initial one included — with the *cumulative* QM-call
count, the temperature, and the energy. v04's "N/A" temperature-summary lines
go only to the ``.out`` history, never here, so :class:`TvseWriter` ignores
``on_temperature_complete``.

The column layout (header, blank line, separator, then ``%14d %10.2f
%15.6f`` rows) is copied verbatim from v04 so existing plotting scripts and
the parity harness keep working.
"""

from __future__ import annotations

from pathlib import Path

from cosmic_ascec.annealing.engine import AnnealingCallback, ConfigAccepted, RunStart

_HEADER = f"{'# n-eval (Cuml)':>16} {'T(K)':>10} {'E(u.a.)':>15}\n"
_SEPARATOR = "#----------------------------------\n"


class TvseWriter(AnnealingCallback):
    """Append a ``(n-eval, T, E)`` row per accepted configuration."""

    def __init__(self, run_dir: Path, seed: int) -> None:
        self.tvse_path = Path(run_dir) / f"tvse_{seed}.dat"

    def on_run_start(self, event: RunStart) -> None:  # noqa: ARG002
        """Write the header block, truncating any stale file from a prior run."""
        self.tvse_path.write_text(_HEADER + "\n" + _SEPARATOR, encoding='utf-8')

    def on_config_accepted(self, event: ConfigAccepted) -> None:
        """Append the cumulative-QM-call ledger row for this accepted config."""
        with self.tvse_path.open("a", encoding='utf-8') as fh:
            fh.write(
                f"  {event.cumulative_n_eval:>14} "
                f"{event.temperature:>10.2f} {event.energy:>15.6f}\n"
            )


__all__ = ["TvseWriter"]
