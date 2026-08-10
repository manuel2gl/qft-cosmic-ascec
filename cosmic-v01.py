#!/usr/bin/env python3
"""``cosmic`` root entry point — what the ``cosmic`` shell alias runs.

Thin wrapper around :func:`cosmic_ascec.command_line.cosmic.main`. Sits at
the repo root next to ``ascec-v04.py`` / ``index.html`` / ``install.sh`` so
both CLIs are reachable from a bare ``git clone`` without needing
``pip install`` — the shim puts the repo root on ``sys.path`` then imports
the ``cosmic_ascec`` package.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 safe on Windows.

    Same reasoning as the matching helper in ``ascec-v04.py``: the banner and
    progress output contain non-ASCII, and a redirected stream on Windows falls
    back to the ANSI code page (cp1252), where writing them raises
    UnicodeEncodeError before any work is done.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_force_utf8_console()


def main() -> int:
    from cosmic_ascec.command_line.cosmic import main as _cosmic_main
    return _cosmic_main()


if __name__ == "__main__":
    raise SystemExit(main())
