#!/usr/bin/env python3
"""
relational-reality
==================
Measuring the spectral dimension d_s of graphs grown from a configurable
Hamiltonian, and searching for where that geometry looks 4-dimensional.

Just run it — no arguments:

    python main.py

That opens the live dashboard in your browser and immediately starts
sweeping the grid defined in main.toml, resuming wherever a previous run
left off (already-computed cells are skipped). The shape analysis refreshes
automatically as new data arrives.

The science needs no command-line options — everything is configured in
main.toml (the sweep grid under [grid]; optional [server], [dashboard],
[compute], [calibration] and [engine] tables for runtime knobs). To stop the
workers, press Ctrl-C in this terminal. The web page only reports incoming
data — it does not start/stop anything.

A handful of optional diagnostic sub-commands live here too (they operate on
the engines or on existing output/, and aren't part of a sweep):

    python main.py engines        # list graph-growth engines + their parameters
    python main.py figures        # regenerate the heatmaps/charts from output/
    python main.py uniformity ... # the standalone field-isotropy deep-dive probe
    python main.py bench  ...      # the cache->DRAM memory benchmark (reads bench.toml)
    python main.py selftest        # regression checks (figures + pipeline + ETA model)

Anything after a sub-command is passed through to it, so each tool's own
--help works (e.g. `python main.py bench --help`).

What to edit:
  • main.toml                the sweep grid (k, T, lb, N, seed) + runtime knobs
  • src/core/project_constants.py  internal physics/probe constants (rarely)
"""

import os
import subprocess
import sys
import threading
import webbrowser

# Pin BLAS / OpenMP / Numba to ONE thread per process *before* anything imports
# numpy. The sweep runs one worker process per core and relies on each worker
# being single-threaded. If numpy/BLAS gets imported (even transitively, e.g.
# via matplotlib) before these are set, OpenBLAS grabs a thread pool sized to
# the whole machine; the pool is then inherited by every forked worker, so
# N workers x N BLAS threads thrash over N cores and the spectral probe appears
# to hang (no error, just stuck). setdefault so an explicit override still wins.
for _thr in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_thr, "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
OUTPUT = os.path.join(ROOT, "output")

# Make both the source packages (core, metrics, ds4_search, shape_analysis)
# importable from anywhere.
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _check_plotting_dep():
    """Loudly warn if matplotlib is missing. Without it the sweep still runs and
    writes every flow CSV, but the periodic figure refresh silently produces NO
    heatmap/chart PNGs — which is easy to miss for a long time. Non-fatal."""
    import importlib.util
    if importlib.util.find_spec("matplotlib") is not None:
        return True
    if True:
        bar = "!" * 74
        print("\n" + bar +
              "\n  matplotlib is NOT installed in this environment."
              "\n  The sweep will run and write data, but it will NOT generate any heatmap/"
              "\n  chart PNGs (shape_heatmap.png, flow_map.png, ...). To get them, install:"
              "\n"
              "\n      pip install -r requirements.txt        (or: pip install matplotlib)"
              "\n"
              "\n  Then the figures regenerate automatically on the configured interval"
              "\n  ([dashboard].redraw_interval_s in main.toml). No manual step needed."
              "\n" + bar + "\n", flush=True)
        return False


def _ensure_output():
    os.makedirs(os.path.join(OUTPUT, "flow"), exist_ok=True)


def _run_sweep_worker(rest):
    """Internal entry: the live app (and the sweep's own hot-reload) launch
    the batch worker as `python main.py __sweep__ …`. Not part of the public
    interface — users never call this directly."""
    _ensure_output()
    _check_plotting_dep()
    os.chdir(OUTPUT)              # all sweep output lands under output/
    from ds4_search import sweep_runner
    sweep_runner.main(rest)


def _run_dashboard_and_sweep():
    """The default (and only) action: serve the live dashboard and auto-start
    the sweep. Bind address / port / browser come from main.toml [server];
    the grid comes from main.toml [grid] and is read directly by the sweep
    backend (core.project_constants) — main does not marshal it."""
    from core import project_constants as cfg
    from ds4_search import live_app

    host = cfg.SERVER_HOST
    port = str(cfg.SERVER_PORT)
    url = f"http://{host}:{port}/"
    _ensure_output()
    _check_plotting_dep()

    argv = [
        "--dir", OUTPUT,
        "--host", host,
        "--port", port,
        "--auto-start",          # begin sweeping immediately
    ]

    print(f"  Opening the live dashboard at {url}")
    print("  Sweeping the grid from main.toml — press Ctrl-C here to stop.\n")
    # Pop the browser once the server has had a moment to bind.
    if cfg.SERVER_OPEN_BROWSER:
        threading.Timer(1.5, lambda: _open(url)).start()
    live_app.main(argv)          # blocks until Ctrl-C


def _open(url):
    print(f"  → {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ── diagnostic sub-commands (formerly tools.py) ─────────────────────────────
# These operate on the engines or on existing output/, not on a sweep. The
# default `python main.py` (no sub-command) runs the science; these are the
# occasional workshop tools.

def _cmd_engines(_argv):
    """List registered graph-growth engines, whether each is runnable now, and
    its Hamiltonian parameters — plus which backend is configured in main.toml
    and which one a sweep would ACTUALLY use right now."""
    import engines
    from core.project_constants import ENGINE_BACKEND
    avail = engines.available_engines()
    configured = (ENGINE_BACKEND or "auto").lower()
    want = "cpp" if configured == "auto" else configured
    effective = want if want in avail else "numba"
    print("Graph-growth engines (src/engines/)\n")
    for backend, cls in engines.all_engines().items():
        mark = "available" if backend in avail else "unavailable"
        extra = "" if backend in avail else "  (needs compiler / toolchain)"
        tag = "  ← would be used" if backend == effective else ""
        print(f"  [{mark:11}] {backend:6} — {cls.name}{extra}{tag}")
        for p in cls.parameters():
            rng = ""
            if p.low is not None or p.high is not None:
                rng = f"  [{p.low if p.low is not None else '-'}, {p.high if p.high is not None else '-'}]"
            print(f"                  · {p.name} = {p.default} ({p.kind}){rng}  {p.help}")
        print()
    print(f"configured: [engine].backend = \"{configured}\" in main.toml"
          + ("  (auto = prefer cpp, fall back to numba)"
             if configured == "auto" else ""))
    print(f"effective:  a sweep started now would run on the "
          f"'{effective}' engine.")
    print("\nTo select a different engine, edit main.toml:\n"
          "    [engine]\n"
          "    backend = \"numba\"     # or \"cpp\", \"rust\", \"auto\"\n"
          "and re-run `python main.py` — there is deliberately no CLI flag for\n"
          "this (main.toml is the single source of configuration, so the\n"
          "engine that grew the data is always recorded in one auditable\n"
          "place; each cached graph also records its engine in its sidecar).\n"
          "Native engines auto-build on first use (g++/clang++ for cpp, rustc "
          "for rust).")
    return 0


def _cmd_figures(argv):
    """Regenerate the shape/flow heatmap PNGs from data ALREADY in output/ — no
    re-sweep. Mirrors the sweep's periodic refresh (shape_analysis + flow_modes
    + flow_convergence + flow_charts + the isotropy heatmap)."""
    import argparse
    ap = argparse.ArgumentParser(prog="main.py figures",
                                 description=_cmd_figures.__doc__)
    ap.add_argument("--dir", default=OUTPUT,
                    help="output directory holding flow/ and flow_*.csv (default: ./output)")
    args, _ = ap.parse_known_args(argv)
    d = os.path.abspath(args.dir)
    if not os.path.isdir(d):
        print(f"no output directory at {d}", file=sys.stderr)
        return 1
    before = {f for f in os.listdir(d) if f.endswith(".png")}
    cwd = os.getcwd()
    os.chdir(d)   # the generators write bare-filename PNGs into the cwd (as the sweep does)
    ran, skipped = [], []
    try:
        jobs = [("shape_analysis", ["--dir", "."]),
                ("flow_modes", ["--dir", "."]),
                ("flow_convergence", ["--dir", "."]),
                ("flow_charts", ["--csv", "flow_convergence.csv"])]
        for mod, margs in jobs:
            try:
                __import__(mod).main(margs)
                ran.append(mod)
            except SystemExit:
                ran.append(mod)
            except Exception as e:
                skipped.append((mod, f"{type(e).__name__}: {e}"))
        # the isotropy heatmap reads the per-cell meta sidecars
        try:
            from physics_tests import isotropy as _iso
            if _iso.render_isotropy_heatmap(".", "isotropy_heatmap.png"):
                ran.append("isotropy_heatmap")
        except Exception as e:
            skipped.append(("isotropy_heatmap", f"{type(e).__name__}: {e}"))
    finally:
        os.chdir(cwd)
    after = {f for f in os.listdir(d) if f.endswith(".png")}
    new = sorted(after - before)
    print(f"\nfigures regenerated in {d}")
    print(f"  ran:     {', '.join(ran) if ran else '(none)'}")
    for mod, why in skipped:
        print(f"  SKIPPED  {mod}: {why}")
    print(f"  PNGs now present ({len(after)}): {', '.join(sorted(after)) or '(none)'}")
    if new:
        print(f"  newly written: {', '.join(new)}")
    if skipped and not after:
        print("\n  No PNGs were produced. The most common cause is a missing plotting\n"
              "  dependency — check that matplotlib is installed in THIS environment\n"
              "  (pip install -r requirements.txt). The per-module reason is shown above.")
    return 0


def _cmd_selftest(_argv):
    """Regression checks: the heatmap/chart PNGs still generate, the sweep's
    per-cell path (build -> probe -> write flow CSV) still works with BLAS/Numba
    pinned to one thread, and the ETA model still satisfies its stability
    property."""
    from tests import test_figures, test_pipeline, test_eta_stability
    print("── figures ──")
    ok_fig, _pngs, _detail = test_figures.check(verbose=True)
    print("\n── pipeline ──")
    ok_pipe, _ = test_pipeline.check(verbose=True)
    print("\n── eta model ──")
    try:
        test_eta_stability.main()
        ok_eta = True
    except Exception as e:
        ok_eta = False
        print(f"  ETA model self-test FAILED: {type(e).__name__}: {e}")
    ok = ok_fig and ok_pipe and ok_eta
    print("\n" + ("OK — all self-tests passed." if ok else "FAIL — see above."))
    return 0 if ok else 1


def _cmd_bench(argv):
    """Run the cache->DRAM time-to-equilibrium memory benchmark
    (src/benchmarks/memory_benchmark.py). Reads bench.toml by default. This is a
    one-off hardware characterization, not part of a sweep."""
    from benchmarks import memory_benchmark as mb
    sys.argv = ["bench"] + list(argv)
    mb.main()
    return 0


def _cmd_reverify(argv):
    """Re-judge every cell's equilibration verdict from the thermalisation
    traces already stored in the meta sidecars — seconds, no recompute. Use
    after upgrading the verification statistics so existing data is judged
    by the same test as new data."""
    import reverify
    return reverify.main(list(argv) or ["--dir", OUTPUT])


def _cmd_uniformity(argv):
    """Run the standalone field-isotropy (uniformity) deep-dive on one cell. The
    sweep already stores a cheap isotropy summary per cell and renders
    isotropy_heatmap.png automatically; this is the closer look."""
    script = os.path.join(SRC, "physics_tests", "isotropy.py")
    if not os.path.exists(script):
        print(f"uniformity probe not found at {script}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, script] + list(argv))


_COMMANDS = {
    "engines": _cmd_engines,
    "figures": _cmd_figures, "heatmaps": _cmd_figures,
    "uniformity": _cmd_uniformity, "isotropy": _cmd_uniformity,
    "bench": _cmd_bench, "mem-bench": _cmd_bench, "memory": _cmd_bench,
    "selftest": _cmd_selftest, "test": _cmd_selftest,
    "reverify": _cmd_reverify,
}

# One-line description per mode, shown on EVERY parse miss — the contract is
# that using main.py wrongly fully documents using it rightly, so nobody has
# to guess strings or reach for --help.
_USAGE = """\
relational-reality — spectral-dimension sweep + live dashboard

  python main.py                 (no arguments) THE default action: open the
                                 live dashboard and start sweeping the grid in
                                 main.toml, resuming where a previous run left
                                 off. Ctrl-C in this terminal stops everything.

diagnostic sub-commands (operate on the engines or on existing output/):

  python main.py engines         list graph-growth engines (numba / cpp /
                                 rust), availability, their Hamiltonian
                                 parameters, and which one is configured +
                                 which would actually run. Selection itself
                                 lives in main.toml ([engine].backend).
  python main.py figures         regenerate every heatmap/chart PNG from the
                                 data already in output/ — no re-sweep.
                                 (alias: heatmaps; option: --dir DIR)
  python main.py uniformity ...  standalone field-isotropy deep-dive on one
                                 cell — full per-shell/per-time curves vs a
                                 reference torus. (alias: isotropy; see
                                 `python main.py uniformity --help`)
  python main.py bench ...       cache->DRAM memory benchmark, a one-off
                                 hardware characterization that prints the
                                 recommended [compute].workers. Reads
                                 bench.toml. (aliases: mem-bench, memory)
  python main.py selftest        regression checks: figures + cell pipeline +
                                 ETA model. (alias: test)
  python main.py reverify        re-judge equilibration verdicts from the
                                 traces stored in existing meta sidecars —
                                 seconds, no recompute. Run once after
                                 upgrading; --dry-run previews the table.

Anything after a sub-command is passed through to it, so each tool's own
--help works too (e.g. `python main.py bench --help`).

There are NO other command-line flags on main.py itself — every knob (sweep
grid, server host/port, workers, memory guard, calibration, engine backend)
lives in main.toml. Edit it and re-run; already-computed cells are skipped.
"""

_HELP_TOKENS = {"-h", "--help", "-help", "help", "--h", "/?", "-?"}


def _usage(to=sys.stdout):
    print(_USAGE, file=to)


def main():
    argv = sys.argv[1:]

    # Hidden worker route used by the live app to spawn the batch sweep. Must
    # stay first and bypass everything else: arguments after the marker are the
    # sweep_runner's own internal CLI, not a user interface.
    if argv and argv[0] == "__sweep__":
        _run_sweep_worker(argv[1:])
        return

    # Help, in any of the spellings people actually type.
    if argv and argv[0].lower() in _HELP_TOKENS:
        _usage()
        raise SystemExit(0)

    # A diagnostic sub-command (engines/figures/uniformity/bench/selftest)?
    if argv and argv[0] in _COMMANDS:
        raise SystemExit(_COMMANDS[argv[0]](argv[1:]))

    # ANYTHING else — a typo'd sub-command, a stray flag, whatever — gets the
    # full usage reference, never a silently-launched multi-process sweep.
    # (An earlier version only caught non-dash tokens, so `-help` fell through
    # and started the sweep; now only an empty argv reaches the default.)
    if argv:
        print(f"unknown argument {argv[0]!r}.\n", file=sys.stderr)
        _usage(to=sys.stderr)
        raise SystemExit(2)

    # No arguments → the one thing this project does: serve the dashboard and
    # auto-start the sweep. All configuration lives in main.toml.
    _run_dashboard_and_sweep()


if __name__ == "__main__":
    main()
