#!/usr/bin/env python3
"""``ascec`` root entry point — what the ``ascec`` shell alias runs.

This is a thin wrapper around :func:`cosmic_ascec.command_line.ascec.main`.
It exists at the repo root so that:

1. A bare ``git clone`` exposes a runnable ``ascec`` (no ``pip install``
   needed — the shim puts the repo root on ``sys.path`` then imports the
   ``cosmic_ascec`` package).
2. ``ascec.py`` sits next to ``cosmic-v01.py``, ``install.sh`` and
   ``index.html`` so all the user-facing entry points are in one place.
3. The workflow's replication stage can re-launch itself by spawning
   ``[sys.executable, sys.argv[0], <input>]`` — that resolves back to
   this file inside each replica subprocess.

Three "auxiliary" subcommands are intercepted before the main dispatcher
because they don't need the full annealing machinery:

* ``ascec analyze_box <xyz> [nmol]`` — print box-size recommendations for
  a system from an existing .xyz file.
* ``ascec test_box`` — run the diagnostic self-test for the box-length
  calculator.
* ``ascec input`` — open the hosted WebGUI that builds .asc files in the
  browser; ``ascec input local [port]`` serves it from a local ``http.server``.

Everything else is handed to ``cosmic_ascec.command_line.ascec.main``.
"""

from __future__ import annotations

import os
import sys

# Drop the repo root onto sys.path so ``import cosmic_ascec`` works from a
# bare ``git clone`` without needing ``pip install``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _route_aux_modes() -> int | None:
    """Intercept the auxiliary subcommands that don't need the main dispatcher."""
    if len(sys.argv) <= 1:
        return None
    cmd = sys.argv[1]
    if cmd == "analyze_box" and len(sys.argv) >= 3:
        from cosmic_ascec.diagnostics import analyze_box_length_from_xyz
        xyz_file = sys.argv[2]
        num_molecules = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        analyze_box_length_from_xyz(xyz_file, num_molecules)
        return 0
    if cmd == "test_box":
        from cosmic_ascec.diagnostics import test_box_length_analysis
        test_box_length_analysis()
        return 0
    if cmd == "input":
        rest = sys.argv[2:]
        if rest and rest[0] == "local":
            from cosmic_ascec.webgui import run_input_server
            port = int(rest[1]) if len(rest) > 1 else 8080
            return run_input_server(port=port, script_dir=_HERE)
        from cosmic_ascec.webgui import open_hosted_input
        return open_hosted_input()
    return None


def main() -> int:
    rc = _route_aux_modes()
    if rc is not None:
        return rc
    from cosmic_ascec.command_line.ascec import main as _ascec_main
    return _ascec_main()


if __name__ == "__main__":
    raise SystemExit(main())
