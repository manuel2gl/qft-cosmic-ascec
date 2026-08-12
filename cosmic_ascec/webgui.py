"""``ascec input`` — web-input-generator launcher.

v04 ships ``index.html`` (the GUI that produces ``.asc`` inputs) and served it
from the script's own directory via ``http.server`` (ascec-v04.py lines
21181-21216). v05 changes the default: ``ascec input`` now opens the hosted
GitHub Pages deployment (:func:`open_hosted_input`) so no local server is
needed, while ``ascec input local [port]`` preserves v04's local-server path
(:func:`run_input_server`) for offline use or serving an edited ``index.html``.

The implementation is a verbatim port of v04's inline server: the static-file
handler serves the directory containing the root shim (``index.html`` lives
there per the v04 release layout), an optional port argument shifts off the
default 8080, the browser is opened to ``http://localhost:<port>/index.html``,
and ``serve_forever`` blocks until Ctrl-C.

The decomposition is **mechanical only** (D-039): the port-argument
default-fallback, the ``index.html`` discovery via ``script_dir``, the
``OSError("Address already in use")`` message, and the ``KeyboardInterrupt``
shutdown message all match v04 byte-for-byte.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import sys
import webbrowser
from functools import partial


#: Public GitHub Pages deployment of ``index.html``. ``ascec input`` opens this
#: directly so no local server is required; ``ascec input local`` falls back to
#: :func:`run_input_server` for offline / development use.
HOSTED_INPUT_URL = "https://manuel2gl.github.io/qft-cosmic-ascec/"


def open_hosted_input(url: str = HOSTED_INPUT_URL) -> int:
    """Open the hosted WebGUI in the user's browser — no local server needed.

    This is the default for ``ascec input``. The page is served from GitHub
    Pages, so it works without binding a port or shipping ``index.html``
    locally. Use ``ascec input local [port]`` (:func:`run_input_server`) when
    offline or when serving an edited local copy.
    """
    print(f"\n  ASCEC Input Generator")
    print(f"  ─────────────────────")
    print(f"  Opening: {url}")
    print(f"  (run 'ascec input local' to serve it from localhost instead)\n")
    webbrowser.open(url)
    return 0


def run_input_server(port: int = 8080, script_dir: str | None = None) -> int:
    """Verbatim port of v04 ``ascec-v04.py`` lines 21181-21216.

    Parameters
    ----------
    port:
        TCP port to bind. v04's default is ``8080``; ``ascec input <port>``
        passes through to here.
    script_dir:
        Directory served by the static-file handler. The v04 default is
        ``os.path.dirname(os.path.abspath(__file__))`` — the directory the
        root script (``ascec-v04.py`` in v04, ``ascec.py`` in v05) lives in,
        which is also where ``index.html`` is shipped. v05's root shim passes
        its own directory in here.
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    web_dir = script_dir
    input_page = "index.html"

    if not os.path.exists(os.path.join(web_dir, input_page)):
        print("Error: Web input generator not found.")
        print(f"Expected at: {web_dir}/{input_page}")
        return 1

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=web_dir)

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}/{input_page}"
            print(f"\n  ASCEC Input Generator")
            print(f"  ─────────────────────")
            print(f"  Opening: {url}")
            print(f"  Press Ctrl+C to stop\n")
            webbrowser.open(url)
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {port} is already in use.")
            print(f"Try: python ascec.py input {port + 1}")
            return 1
        raise
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    return 0


__all__ = ["open_hosted_input", "run_input_server", "HOSTED_INPUT_URL"]
