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
conclusions. (See `TIER12_NOTES.md` for the capture/analysis details.)

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
grid defined in `config.toml`, resuming wherever a previous run left off
(already-computed cells are skipped). The shape analysis (histogram + heatmap)
and the flow-mode maps refresh automatically as new data arrives.

**To stop**, press `Ctrl-C` in the terminal. That tears down both the web
server and the sweep workers. The web page is purely a *reporting* surface — it
shows incoming data and does not start or stop anything.

Later runs are the same — the script reuses the venv and relaunches. To set up
(or refresh dependencies) **without** launching, pass `--no-run`:

```bash
./setup.sh --no-run        # macOS / Linux
setup.bat  --no-run        # Windows
```

…then start it yourself:

```bash
.venv/bin/python main.py            # macOS / Linux
.venv\Scripts\python.exe main.py    # Windows
```

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

## What you change: `config.toml`

One file defines *what* gets swept — the parameter ranges:

```toml
[grid]
k  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    # mean target degree
T  = [0, 0.005, 0.015, 0.03, 0.05]           # growth temperature
lb = [0.90, 0.95, 0.99, 0.995, 0.999]        # locality bias — the main d_s→4 dial
N  = [1000, 4000, 16000, 64000, 256000,      # graph sizes (the finite-size ladder;
      1024000, 4096000]                      #   the large-N cells are expensive)
seed = 42                                    # base RNG seed for graph growth
n_seeds = 5                                  # graphs per cell: runs seeds 42..46 so
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
├── config.toml         ← the sweep grid you edit (k, T, lb, N, seed, n_seeds)
├── requirements.txt    ← Python dependencies
├── setup.sh            ← one-shot: venv + install + launch (macOS / Linux)
├── setup.bat           ← one-shot: venv + install + launch (Windows)
├── README.md
├── TIER12_NOTES.md     ← what the capture sidecars are (meta/quad) + how to use them
├── src/                ← source code (never written to at runtime)
│   ├── core/               shared backend
│   │   ├── physics_engine.py     graph growth from the Hamiltonian (numba kernels)
│   │   ├── project_constants.py  physics/probe constants; loads the grid from config.toml
│   │   ├── disk_io.py            CSV / JSON / log helpers, μ-table load/save
│   │   ├── graph_builder.py      graph build + μ-calibration  (library; no CLI)
│   │   ├── cell_tests.py         build one cell (basin or torus reference) + tests
│   │   └── flow_probe.py         the d_s(t) "flow" probe — the core measurement
│   ├── metrics/            spectral-dimension library:
│   │   ├── random_walk.py          random-walk return-probability d_s probe
│   │   ├── stochastic_lanczos.py   SLQ (Lanczos) d_s probe
│   │   ├── hausdorff_dimension.py  Hausdorff dimension probes
│   │   ├── size_extrapolation.py   finite-size extrapolation to large N
│   │   ├── reference_lattices.py   reference-lattice builders + registry
│   │   ├── graph_topology.py       structural graph metrics
│   │   ├── graph_container.py      Graph container + archive
│   │   └── (random_walk_fits, bfs_kernels, numba_kernels — internal helpers)
│   ├── ds4_search/         the d_s→4 search
│   │   ├── sweep_runner.py     headless grid worker: grows cells, classifies,
│   │   │                       extrapolates the ETA, refreshes shape analysis
│   │   ├── live_app.py         the live reporting web server
│   │   ├── live_app_page.py    the page it serves (HTML/CSS/JS; presentation only)
│   │   ├── static_dashboard.py renders a standalone offline HTML dashboard
│   │   └── dashboard_page.py   its page (presentation only)
│   └── shape_analysis.py   curve-shape classification + histogram/heatmap
└── output/             ← everything generated lands here
    ├── flow/               per measured cell:
    │   ├── flow_*.csv          the raw d_s(t) curve
    │   ├── meta_*.json         topology + provenance + equilibration trace (Tier-1)
    │   └── quad_*.npz          per-probe SLQ Ritz quadrature (Tier-2): lets you
    │                           re-derive d_s(t) on any t-grid with NO SLQ rerun —
    │                           see reconstruct_Z() in src/core/flow_probe.py
    ├── mu_table.json       calibrated μ per (k, T, lb)
    ├── lb_sweep_status.json live sweep status (progress, ETA) the page polls
    ├── lb_sweep_classifications.csv  shape tag per cell
    ├── shape_summary.csv   per-cell shape summary (d_s_local, dips, peaks)
    ├── shape_histogram.png distribution of local spectral dimension
    ├── shape_heatmap.png   mean d_s over (k, ℓ); rows = T, columns = N
    ├── shape_heatmap_seed_std.png  seed-to-seed spread of d_s (same layout;
    │                       written only when n_seeds > 1)
    ├── flow_modes.csv      dual-mode (flat-4D vs flowing-4D) score per cell
    ├── flow_modes_plane.png   residual-vs-4D-torus plane (flat-4D at origin)
    └── flow_modes_maps.png    (k × lb) score maps, one column per T
```

Two standalone analysis helpers live at the top level and read `output/`
directly: `flow_convergence.py` (per-cell plateau + `N^{-β}` extrapolation) and
`flow_charts.py` (leaderboards from `flow_modes.csv`).

`main.py` runs the sweep with `output/` as the working directory, which is how
generated files stay out of `src/`.

---

## Notes

- **Single entry point.** Modules under `src/` import each other as packages
  (`from core.flow_probe import …`, `from metrics import …`) and are driven
  through `main.py`, which puts `src/` and the project root on the path. They
  aren't meant to be run directly.
- **The ETA is extrapolated, not naïve.** Cells run smallest-N first and
  per-cell cost grows steeply with N, so a simple "remaining ÷ average rate"
  badly underestimates the expensive tail. The sweep instead fits the observed
  cost-vs-N trend and predicts each remaining cell from its own N.
- **`d_s` everywhere means the spectral dimension** (the shape summary's
  `d_s_local` is the local spectral dimension read off at the curve's dip).
  Hausdorff dimension, where computed, lives in `metrics/` and is named
  separately.
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
