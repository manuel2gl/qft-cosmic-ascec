"""Exception hierarchy for COSMIC ASCEC.

All exceptions raised by package code inherit from `CosmicAscecError` so callers can
catch the entire family with one `except`. Boundary layers (CLI, parser) translate
internal exceptions into user-friendly messages.
"""

from __future__ import annotations


class CosmicAscecError(Exception):
    """Base class for all COSMIC ASCEC exceptions."""


class AscParseError(CosmicAscecError):
    """Raised when a `.asc` input file fails to parse."""

    def __init__(self, message: str, *, line: int | None = None, path: str | None = None):
        self.line = line
        self.path = path
        location = f" (line {line})" if line is not None else ""
        prefix = f"{path}{location}: " if path else f"{'line ' + str(line) + ': ' if line else ''}"
        super().__init__(f"{prefix}{message}")


class GeometryError(CosmicAscecError):
    """Raised on invalid molecular geometry."""


class OverlapError(GeometryError):
    """Atoms placed too close together."""


class BoxTooSmallError(GeometryError):
    """Requested cluster does not fit in the simulation box."""


class TrajectoryError(CosmicAscecError):
    """Raised when an MD trajectory cannot be read or the shell selection is invalid."""


class QMError(CosmicAscecError):
    """Raised on quantum chemistry package failures."""


class QMExecutableNotFound(QMError):
    """The configured QM binary cannot be located on PATH."""


class QMConvergenceError(QMError):
    """The QM single-point did not converge."""


class QMTimeoutError(QMError):
    """The QM subprocess exceeded its allotted wall time."""


class ClusteringError(CosmicAscecError):
    """Raised on clustering-pipeline failures."""


class WorkflowError(CosmicAscecError):
    """Raised on workflow orchestration failures."""
