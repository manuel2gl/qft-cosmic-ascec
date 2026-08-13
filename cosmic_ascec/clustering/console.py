"""Shared verbose-print helpers used across the whole clustering pipeline.

Every clustering submodule needs to emit progress text, and they all need to
respect the same ``-v`` / verbose toggle. Three helpers live here:

* :data:`VERBOSE` — the single global flag. Flip ``console.VERBOSE = True``
  before invoking the pipeline and every submodule shows its details.
* :func:`vprint` — print only when :data:`VERBOSE` is set. Reads the flag
  *at call time*, so toggling it mid-run is honoured immediately.
* :func:`print_step` — always print, used for section/step banners.

Plus the version string and the ASCII banner shown at startup.

The clustering stage's stdout is a contract — downstream parsers and the
``ascec status`` viewer match on specific lines — so do not refactor the
print-calls into ``logging`` without checking what reads them.
"""

from __future__ import annotations

# Version string embedded in the ASCII-art clustering reports.
version = "* COSMIC-v01: Feb-2026 *"

# cosmic-v01.py line 86 — verbosity flag. Set by the CLI; read by vprint.
VERBOSE = False


# cosmic-v01.py lines 51-82 — the banner body, verbatim including the trailing
# whitespace that pads each line. Kept as an explicit line list (rather than a
# triple-quoted blob) so the trailing spaces are visible and cannot be lost to
# a whitespace-trimming edit; ``print_version_banner`` rejoins them exactly as
# cosmic-v01's triple-quoted literal.
_VERSION_BANNER_LINES = (
    "===========================================================================",
    "",
    "                        ***************************                        ",
    "                        *       C O S M I C       *                          ",
    "                        ***************************                        ",
    "",
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
    "",
    "",
    "               Universidad de Antioquia - Medellín - Colombia              ",
    "",
    "",
    "           Clustering Analysis for Quantum Chemistry Calculations          ",
    "",
    "                          {version}                       ",
    "",
    "                        Química Física Teórica - QFT                       ",
    "",
    "",
    "===========================================================================",
)


def print_version_banner():
    """Print the ASCII art banner with UDEA logo and version information.

    Verbatim port of cosmic-v01's ``print_version_banner`` (lines 49-83) — the
    triple-quoted ``banner`` literal is rebuilt from :data:`_VERSION_BANNER_LINES`
    (a leading and trailing newline frame the body, exactly as the literal did).
    """
    banner = "\n" + "\n".join(_VERSION_BANNER_LINES).format(version=version) + "\n"
    print(banner)


def vprint(message, **kwargs):
    """Print *message* only when verbose mode is on — cosmic-v01.py 171-174."""
    if VERBOSE:
        print(message, **kwargs)


def print_step(message, **kwargs):
    """Print concise step information (always shown) — cosmic-v01.py 176-178."""
    print(message, **kwargs)


__all__ = [
    "VERBOSE",
    "print_step",
    "print_version_banner",
    "version",
    "vprint",
]
