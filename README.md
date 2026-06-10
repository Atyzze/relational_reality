# relational-reality

Measuring the **spectral dimension** `d_s` of graphs grown from a configurable
Hamiltonian, and searching for where that geometry looks **4-dimensional**.

---

## Status & scope

This is **exploratory research code, not a finished result.** It *measures* the
scale-dependent spectral dimension `d_s(t)` of graphs grown from a relational
Hamiltonian and maps where that geometry looks 4-dimensional. A `d_s ≈ 4`
region — *if it survives finite-size scaling* — is evidence that 4D-like
geometry can emerge from a coordinate-free relational structure. That is the
claim the code can support. It is **not** a spacetime metric, a dynamics, or a
theory of physics, and nothing here should be read as one. Treat the outputs as
measurements to be checked — especially the trend at large `N` — not as

---

## Quickstart

### macOS / Linux

```bash
./setup.sh
```

(Permission error? `chmod +x setup.sh` first, or run `bash setup.sh`.)

### Windows

```bat
setup.bat
```

Double-click `setup.bat` in Explorer, or run it from PowerShell / Command
Prompt. (If SmartScreen warns about an unrecognized script, that's the standard
prompt for any unsigned `.bat` — choose *More info → Run anyway*.)

### What it does (either platform)

The script creates a local virtual environment (`.venv`), installs the
dependencies, and then **launches the project automatically** — no second step.
It opens the live dashboard in your browser and immediately starts sweeping the
grid defined in `main.toml`, resuming wherever a previous run left off
(already-computed cells are skipped). The shape analysis (histogram + heatmap)
and the flow-mode maps refresh automatically as new data arrives.

**To stop**, press `Ctrl-C` in the terminal. That tears down both the web
server and the sweep workers. The web page is purely a *reporting* surface — it
shows incoming data and does not start or stop anything.

Later runs are the same — the script reuses the venv and relaunches.

### Manual setup (if the script won't run)

Any platform, no helper script:

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
.venv\Scripts\activate              # Windows (PowerShell / CMD)
pip install -r requirements.txt
python main.py
```

**Requires Python 3.11+** (the config loader uses the stdlib `tomllib`). On
Windows, install from [python.org](https://www.python.org/downloads/) and tick
*"Add python.exe to PATH"*; the `py` launcher it ships with is what `setup.bat`
uses. Note that `numba`/`llvmlite` can lag the newest CPython by a release or
two — if install fails on a brand-new Python, use the most recent version that
has a matching wheel.

---

## What you change: `main.toml`

`main.toml` holds the two things you tune by hand: the sweep **grid** (which
parameter combinations get measured) and a few **runtime** knobs (plot-redraw
cadence, worker count, memory headroom). Everything else — physics constants,
probe settings, file layout — is internal and lives in
`src/core/project_constants.py`.

### `[grid]` — what gets swept

```toml
[grid]
k  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]      # mean target degree
T  = [0, 0.001, 0.004, 0.016, 0.064]           # growth temperature
lb = [0, 0.5, 0.90, 0.95, 0.99, 0.995, 0.999]  # locality bias — the main d_s→4 dial
N  = [1000, 4000, 16000, 64000, 256000,        # graph sizes (the finite-size ladder;
      1024000, 4096000]                        #   the large-N cells are expensive)
seed = 42                                      # base RNG seed for graph growth
n_seeds = 5                                    # graphs per cell: runs seeds 42..46 so
                                               #   you see the seed-to-seed spread
```

The full sweep is the cross-product of the four lists, run once per seed
(`n_seeds` of them). Each cell→seed writes its own `flow_*.csv`, so a multi-seed
run resumes per-seed and the heatmap can both average d_s and report its
spread. More seeds = more confidence per cell but linearly more compute; to go
faster, narrow the grid (fewer k / ℓ / T / N) rather than dropping seeds, since
the small-N cells are cheap and that is where the spread is largest. Set
`n_seeds = 1` for the old single-seed behaviour. Edit, rerun `./setup.sh` (or
`.venv/bin/python main.py`), and new cells fill in alongside the old ones.

> **Mind the top of the N ladder.** Cost and memory grow steeply with N
> (`node_neighbors` alone is `N · MAX_DEG · 4` bytes *per worker*). The very
> large rungs (N ≥ ~10⁸) need tens to hundreds of GB per worker and are
> infeasible on typical hardware. Trim `N` to what your machine can actually
> hold rather than leaving rungs in that will OOM mid-sweep — the shipped grid
> lists rungs well past what most machines can run, and you are expected to cut
> it down.

### `[dashboard]`, `[compute]`, `[server]`, and `[calibration]` — runtime knobs (optional)

All three tables are optional; omit any one and the code uses the defaults
shown. These do **not** affect *what* is measured, only how the run behaves.
(There are no command-line flags — `main.toml` is the single place anything is
configured, and `python main.py` takes no arguments.)

```toml
[dashboard]
redraw_interval_s       = 60   # force heatmaps/plots to redraw at least this
                               #   often even when no new cell finished
                               #   (0 = off; default 10)
flow_refresh_interval_s = 60   # min seconds between the heavier N→∞ convergence
                               #   pass + its leaderboard charts (0 = every cell;
                               #   negative = disable that pass; default 60)

[compute]
workers     = 0   # parallel worker processes; 0 = all logical cores. Each
                  #   worker is single-threaded (BLAS + numba pinned to 1), so
                  #   this number IS the parallelism. --workers overrides it.
                  #   Tune it with src/physics_tests/optimal_workers.py.
min_free_gb = 4   # memory headroom: the sweep pauses before starting a cell
                  #   that would drop free RAM below this, throttling itself
                  #   instead of being OOM-killed (0 = disable the guard).

[server]
host         = "127.0.0.1"  # dashboard bind address; "0.0.0.0" exposes it on
                            #   your network (read-only but unauthenticated)
port         = 8000         # dashboard HTTP port (0 = OS picks a free one)
open_browser = true         # auto-open a browser tab on launch (false = headless)

[calibration]
n_cal           = 4000  # graph size for the μ binary search (small = cheap)
drift_tol       = 0.02  # relative |k̂−k| beyond which a μ-vs-N correction is derived
drift_min_seeds = 2     # seeds required at an N before its drift is trusted
n_correction    = true  # false = freeze the table; audit still warns, never corrects

[engine]
backend = "auto"  # graph-growth kernel: "auto" (default) prefers the native C++
                  #   engine, compiling it on first use and falling back to numba
                  #   if there's no compiler; "cpp"/"rust" force a native engine
                  #   (still falling back to numba); "numba" is the always-works
                  #   reference. All produce statistically equivalent graphs.
```

(Internal physics/probe constants live in `src/core/project_constants.py`; you
rarely touch those.)

---

## The mental model

This is **one** physics codebase. The same backend (`src/core` + `src/metrics`)
both grows the graphs and measures `d_s(t)` on them. On top of it sit two views
of the same data:

- **Search for d_s → 4** — sweep the locality-bias dial `lb` and watch the
  running spectral dimension, looking for the region where geometry sits at
  `d_s ≈ 4`. This is the live dashboard.
- **Compare curve shapes against reference lattices** — classify each
  `d_s(t)` curve's shape and compare the population against known tori. This is
  the shape analysis (the histogram + heatmap), refreshed automatically as the
  sweep runs.

The reference lattices (2D/3D/… tori) are **always shown** on the live plot as
dashed grey curves — comparing against them is the whole point.

---

## What's where

The top level is deliberately small: the launcher, its config, dependency
files, and two folders — **source** vs **generated**.

```
relational-reality/
├── main.py             ← the only thing you run (no arguments)
├── main.toml           ← the sweep grid you edit (k, T, lb, N, seed, n_seeds)
├── requirements.txt    ← Python dependencies
├── setup.sh            ← one-shot: venv + install + launch (macOS / Linux)
├── setup.bat           ← one-shot: venv + install + launch (Windows)
├── README.md
├── src/                ← source code (never written to at runtime)
│   ├── core/               shared backend
│   │   ├── physics_engine.py     graph growth (numba reference; the C++/Rust
│   │   │                         back-ends in src/engines/ are selectable via [engine])
│   │   ├── project_constants.py  physics/probe constants; loads the grid from main.toml
│   │   ├── disk_io.py            CSV / JSON / log helpers, μ-table load/save
│   │   ├── graph_builder.py      graph build + μ-calibration  (library; no CLI)
│   │   ├── cell_tests.py         build one cell (basin or torus reference) + tests
│   │   └── flow_probe.py         the d_s(t) "flow" probe — the core measurement
│   ├── metrics/            spectral-dimension library:
│   │   ├── stochastic_lanczos.py   SLQ (Lanczos) heat-kernel d_s probe
│   │   ├── numba_kernels.py        njit kernels (LCC extraction, CSR matvec, Laplacian build)
│   │   ├── size_extrapolation.py   finite-size extrapolation to large N
│   │   ├── reference_lattices.py   reference-lattice builders + registry
│   │   ├── graph_topology.py       structural graph metrics
│   │   └── graph_container.py      Graph container
│   ├── ds4_search/         the d_s→4 search
│   │   ├── sweep_runner.py     headless grid worker: grows cells, classifies,
│   │   │                       extrapolates the ETA, refreshes shape analysis
│   │   ├── live_app.py         the live reporting web server
│   │   ├── live_app_page.py    the page it serves (HTML/CSS/JS; presentation only)
│   │   ├── static_dashboard.py renders a standalone offline HTML dashboard
│   │   └── dashboard_page.py   its page (presentation only)
│   ├── shape_analysis.py   curve-shape classification + value/flatness/seed-std heatmaps
│   ├── flow_modes.py       residual-plane + mode-score "shape" view vs a 4D torus
│   ├── flow_convergence.py per-cell plateau + N^-β extrapolation to d_s_∞ (run manually)
│   ├── flow_charts.py      leaderboards from flow_convergence.csv (auto + manual)
│   ├── bench_workers.py    worker-count throughput benchmark library + CLI
│   └── physics_tests/      validation experiments on developed graphs:
│       ├── isotropy.py         does a field spread as a uniform sphere (no highway)?
│       └── optimal_workers.py  fastest worker count (prints [compute].workers)
└── output/             ← everything generated lands here
    ├── flow/               per measured cell:
    │   ├── flow_*.csv          the raw d_s(t) curve
    │   ├── meta_*.json         topology + provenance + equilibration trace,
    │   │                       plus wall_s, workers, rss_mb (for ETA + mem guard)
    │   └── quad_*.npz          per-probe SLQ Ritz quadrature (Tier-2): lets you
    │                           re-derive d_s(t) on any t-grid with NO SLQ rerun —
    │                           see reconstruct_Z() in src/core/flow_probe.py
    ├── mu_table.json       calibrated μ per (k, T, lb)
    ├── lb_sweep_status.json live sweep status (progress, ETA) the page polls
    ├── lb_sweep_classifications.csv  shape tag per cell
    ├── shape_summary.csv   per-cell shape summary (d_s_local, dips, peaks)
    ├── shape_histogram.png distribution of local spectral dimension
    ├── shape_heatmap.png   scalar-summary d_s over (k, ℓ); rows = T, columns = N
    ├── shape_heatmap_flatness.png  in-window flatness (shape companion; same layout)
    ├── shape_heatmap_seed_std.png  seed-to-seed spread of d_s (n_seeds > 1)
    ├── flow_modes.csv / _plane.png / _maps.png  dual-mode (flat/flowing-4D) view
    ├── isotropy_heatmap.png   per-cell field anisotropy over (k, ℓ); rows/panels per (T, N)
    ├── flow_convergence.csv   per-cell N→∞ plateau extrapolation (auto-refreshed)
    └── flow_map.png / flow_scatter.png  convergence leaderboards (auto-refreshed)
```

Standalone helpers under `src/`, run from the project root. The analysis ones
(`flow_convergence.py`, `flow_charts.py`) now also refresh automatically during
a sweep, but can be run by hand: `python src/flow_convergence.py --dir output`.
The `physics_tests/` folder holds validation experiments that reuse the engine
(see its README): `python src/physics_tests/isotropy.py` is the full field-
isotropy deep-dive (does a field spread as a uniform sphere, with no highway?).
A cheap version of this now runs automatically on every cell the sweep builds —
the per-cell anisotropy is stored in each `meta_*.json` sidecar and rendered to
`isotropy_heatmap.png` alongside the other plots — so the standalone is only
needed for a closer look at one cell. `python src/physics_tests/optimal_workers.py`
benchmarks the worker count and prints the recommended `[compute].workers`
value for you to set (it does not edit the config).

### Diagnostic sub-commands

`main.py` also carries a few optional diagnostics (they act on the engines or on
existing `output/`, not on a sweep). The bare `python main.py` still just runs
the science; these are the workshop drawer:

```bash
python main.py engines        # list graph-growth engines + their parameters
python main.py figures        # regenerate the heatmaps/charts from output/
python main.py uniformity ... # the standalone field-isotropy deep-dive on one cell
python main.py bench  ...      # the cache→DRAM memory benchmark (reads bench.toml)
python main.py selftest        # regression checks (figures + pipeline + ETA model)
```

Anything after the sub-command is forwarded to it, so e.g. `python main.py bench
--dry-run` previews the benchmark schedule and `python main.py bench --help`
shows its options. The memory benchmark is deliberately a sub-command, not a
config switch — it's a one-off hardware characterization you run once on a new
machine to read the recommended `[compute].workers`, not something a normal
launch should ever trigger.

`main.py` runs the sweep with `output/` as the working directory, which is how
generated files stay out of `src/`.

---

## Notes

- **Two kinds of module under `src/`.** The *backend packages* —  everything in
  `core/` and `metrics/`, plus `ds4_search/sweep_runner.py` and
  `ds4_search/live_app.py` — import each other as packages
  (`from core.flow_probe import …`, `from metrics import …`) and rely on
  `main.py` to put `src/` and the project root on `sys.path`; don't run those
  directly. The *standalone scripts* — `src/flow_convergence.py`,
  `src/flow_charts.py`, `src/flow_modes.py`, `src/shape_analysis.py`,
  `src/bench_workers.py`, and everything under `src/physics_tests/` — set up
  their own path and **are** meant to be run from the project root (e.g.
  `python src/flow_convergence.py --dir output`). The sweep also imports several
  of them to auto-refresh their outputs.
- **Equilibration is verified, not assumed.** The drift+noise criterion on
  mean degree decides when thermalisation *looks* done; a τ-aware drift test
  then has to CONFIRM stationarity on both `k_avg` and `Σd²/N` (the
  structural observable, which carries the slowest modes) — see
  `core.graph_builder._drift_verdict`. Both observables are sampled every
  sweep over a continuation the length of phase-1; the drift SLOPE is tested
  with errors inflated by the MEASURED integrated autocorrelation time
  (never an assumed one — a fixed-τ window comparison false-fails 70–80% of
  the time on perfectly stationary series with τ ~ hundreds of sweeps,
  verified by simulation), and a detected drift only fails the cell if it
  is also IMPACTFUL: projected over another phase-1-length horizon it would
  move the observable beyond its own fluctuation band ("running it as long
  again would give a different graph"). The check is hard-capped at ~2–3×
  the phase-1 cost; the verdict is three-valued — `true`, `false` (drift
  detected and impactful), or `null` (unresolved: τ too long for the budget
  to decide, recorded as such). `flow_modes`, `flow_convergence`, and the
  `flow_map` green boxes **reject only explicit `false`** — exactly like an
  LCC failure; unresolved cells stay usable but flagged. Verdicts and their
  evidence (τ, z, impact) live in the meta sidecar and the graph cache, and
  `python main.py reverify` re-judges EXISTING data from the stored traces
  in seconds whenever the test improves — no recompute (live full-resolution
  verdicts are authoritative and skipped unless `--force`).
- **μ tracks N.** μ is calibrated once at small N (`[calibration].n_cal`), but
  the realised mean degree can drift as N grows. Every cell's meta sidecar
  already records the μ it used and the `k_avg` it realised, so the sweep
  audits the drift for free and — where it exceeds `drift_tol` — stores a
  first-order corrected μ under an N-qualified key (`"k_T_lb@N"`,
  `μ′ = μ·k̂/k` from the empirical `k·μ ≈ const` relation, clipped to
  [0.5μ, 2μ], derived once per N and then pinned). On top of the measured
  corrections sits a **prediction layer**: with ≥2 measured rungs the ideal
  μ\*(N) = μ·k̂/k is fit against ln N and extrapolated to every larger grid
  rung that has no data yet, so the expensive million-node rungs START with
  an accurate μ instead of waiting to measure their own drift — there is
  never a recalibration build at large N (nothing in the whole audit grows a
  graph; it only reads sidecars). Predictions are conservative (clipped to
  [0.5, 2]× the largest measured ideal), tracked in `mu_predictions.json`,
  and refreshed on every audit as more rungs complete; a measured correction
  always overrides a prediction. Corrections and predictions apply **only to
  rungs with no data yet**: a cell that already has seeds on disk reuses the
  μ recorded in its own sidecar, so seeds within one cell can never mix μ
  values. Per-cell drift beyond the tolerance is also warned about at build
  time (`⚠ realised k̂ … off target`).
- **The ETA is extrapolated, not naïve.** Cells run smallest-N first and
  per-cell cost grows steeply with N, so a simple "remaining ÷ average rate"
  badly underestimates the expensive tail. The sweep instead fits the observed
  cost-vs-N trend and predicts each remaining cell from its own N.
- **`d_s` everywhere means the spectral dimension** (the shape summary's
  `d_s_local` is the local spectral dimension read off at the curve's dip).
  This project is about the spectral dimension only. Other notions of
  dimension (Hausdorff/box-counting, random-walk return probability) are not
  measured here — for these graphs they aren't expected to stabilise, so they
  are deliberately out of scope; anyone who wants one can add a probe that
  reuses the existing BFS-distance helper in `core.cell_tests`.
- **Reference lattices** are built by `core.cell_tests.build_torus_cell`;
  analytically-known reference builders live in `metrics/reference_lattices.py`.

---

## License

Released into the **public domain** under
[The Unlicense](https://unlicense.org) (see `LICENSE`). You may copy, modify,
publish, use, compile, sell, or distribute this software — in source or binary
form, for any purpose, with or without attribution — and no permission is
needed. The software is provided "as is", without warranty of any kind.
*(Not legal advice.)*
