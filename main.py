#!/usr/bin/env python3
"""
relational-reality
==================
Measuring the spectral dimension d_s of graphs grown from a configurable
Hamiltonian, and searching for where that geometry looks 4-dimensional.

Just run it — no arguments:

    python main.py

That opens the live dashboard in your browser and immediately starts
sweeping the grid defined in config.toml, resuming wherever a previous run
left off (already-computed cells are skipped). The shape analysis refreshes
automatically as new data arrives.

A few optional flags (everything else is config.toml):

    python main.py --port 8001      # bind a different port (default 8000)
    python main.py --host 0.0.0.0   # expose on your network (read-only, untrusted)
    python main.py --no-browser     # don't auto-open a tab

To stop the workers, press Ctrl-C in this terminal. The web page only
reports incoming data — it does not start/stop anything.

What to edit:
  • config.toml                the sweep grid (k, T, lb, N, seed)
  • src/core/project_constants.py  internal physics/probe constants (rarely)
"""

import argparse
import os
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
OUTPUT = os.path.join(ROOT, "output")

# Make both the source packages (core, metrics, ds4_search, shape_analysis)
# importable from anywhere.
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_output():
    os.makedirs(os.path.join(OUTPUT, "flow"), exist_ok=True)


def _grid_args():
    """Translate the sweep grid (from config.toml, via core.project_constants) into
    CLI args for the sweep backend that the live app spawns."""
    from core import project_constants as cfg

    def csv(xs):
        return ",".join(str(x) for x in xs)
    return [
        "--k", csv(cfg.K_ALL),
        "--T", csv(cfg.T_ALL),
        "--lb", csv(cfg.LB_ALL),
        "--N", csv(cfg.N_ALL),
        "--seeds", csv(cfg.SEEDS),
    ]


def _run_sweep_worker(rest):
    """Internal entry: the live app (and the sweep's own hot-reload) launch
    the batch worker as `python main.py __sweep__ …`. Not part of the public
    interface — users never call this directly."""
    _ensure_output()
    os.chdir(OUTPUT)              # all sweep output lands under output/
    from ds4_search import sweep_runner
    sweep_runner.main(rest)


def _run_dashboard_and_sweep(host="127.0.0.1", port=8000, open_browser=True):
    """The default: serve the live dashboard and auto-start the sweep."""
    from ds4_search import live_app

    port = str(port)
    url = f"http://{host}:{port}/"
    _ensure_output()

    argv = [
        "--dir", OUTPUT,
        "--host", host,
        "--port", port,
        "--auto-start",          # begin sweeping immediately
    ] + _grid_args()

    print(f"  Opening the live dashboard at {url}")
    print(f"  Sweeping the grid from config.toml — press Ctrl-C here to stop.\n")
    # Pop the browser once the server has had a moment to bind.
    if open_browser:
        threading.Timer(1.5, lambda: _open(url)).start()
    live_app.main(argv)          # blocks until Ctrl-C


def _open(url):
    print(f"  → {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    # Hidden worker route used by the app to spawn the batch sweep.
    # Must stay first and bypass argparse entirely: everything after the
    # marker is the sweep_runner's own CLI, not ours.
    if len(sys.argv) > 1 and sys.argv[1] == "__sweep__":
        _run_sweep_worker(sys.argv[2:])
        return

    # Everything else (including no args) → the one thing this project does:
    # serve the dashboard + auto-start the sweep. A few optional flags let
    # you move it off the default port/host without editing code.
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Serve the live d_s dashboard and auto-start the sweep "
                    "(grid comes from config.toml). Run with no arguments "
                    "for the defaults.",
    )
    ap.add_argument("--port", type=int, default=8000,
                    help="HTTP port to bind (default 8000; 0 = let the OS "
                         "pick a free port).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default 127.0.0.1, localhost-only). "
                         "Use 0.0.0.0 to expose on your network — the page "
                         "is read-only but unauthenticated, so only do this "
                         "on networks you trust.")
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open a browser tab (the URL is still "
                         "printed).")
    args = ap.parse_args()

    _run_dashboard_and_sweep(host=args.host, port=args.port,
                             open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
