"""
core/project_constants.py — physics + I/O constants (internal)
==========================================================
Pure data, no logic that imports any other project module. Sits at the
bottom of the dependency graph so nothing here can produce a cycle.

Owns:
  • Sweep grid (K_ALL, T_ALL, LB_ALL, N_ALL, SEEDS)
  • Physics constants (EC, MAX_DEG)
  • μ-calibration parameters
  • Thermalisation parameters
  • File paths (output dir, log, μ-table)

Reference-graph builders with analytically-known d_s live in
metrics/reference_lattices.py; the torus references used by the sweep are built
directly by core.cell_tests.build_torus_cell.
"""

# ═══════════════════════════════════════════════════════════════════
#  Sweep grid — loaded from the user-facing config.toml at the project root
# ═══════════════════════════════════════════════════════════════════
# The grid (what to sweep) is the one thing users tune, so it lives in
# config.toml next to main.py, not buried here. We parse it with the stdlib
# tomllib (Python 3.11+) and expose it under the internal *_ALL names the rest
# of the code already uses.
import os as _os
import tomllib as _tomllib

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_CONFIG_PATH = _os.path.join(_ROOT, "config.toml")
with open(_CONFIG_PATH, "rb") as _f:
    _cfg = _tomllib.load(_f)
_grid = _cfg["grid"]
# Optional [dashboard] table: runtime/plotting knobs kept out of [grid] so the
# grid stays purely about *what* is swept. Absent table → all defaults, so a
# config that predates these keys still loads unchanged.
_dash = _cfg.get("dashboard", {})

K_ALL   = list(_grid["k"])
T_ALL   = list(_grid["T"])
LB_ALL  = list(_grid["lb"])
N_ALL   = list(_grid["N"])
# SEEDS: run seeds  seed, seed+1, …, seed+n_seeds-1  for every cell so the
# sweep samples the graph-to-graph spread, not just one graph. n_seeds is
# the single user knob (config.toml); it defaults to 1 here only as a
# back-compat safety net for an older config that predates the knob.
_SEED0   = int(_grid["seed"])
_N_SEEDS = max(1, int(_grid.get("n_seeds", 1)))
SEEDS    = [_SEED0 + _i for _i in range(_N_SEEDS)]

# REDRAW_INTERVAL_S: force the heatmaps / shape plots to redraw at least this
# often (in seconds) while the sweep is running, even when no new cell has
# finished. This is what lets config/marker changes and newly-flagged cells
# appear without waiting for the next (possibly very slow, high-N) cell to
# land. 0 disables the periodic redraw — plots then refresh only when a cell
# completes, as before. Set in config.toml under [dashboard].
REDRAW_INTERVAL_S = max(0.0, float(_dash.get("redraw_interval_s", 10.0)))


# ═══════════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════════
EC      = -1.0
# Adjacency cap. Allocates N · MAX_DEG · 4 bytes/worker, so it's kept as
# tight as the physics allows: 2× the largest target degree in the grid.
# Our runs never realise a degree above ~max(K_ALL), so 2× leaves a full
# margin of headroom while halving the old fixed allocation. If a build
# ever pushes a node to MAX_DEG the cap clips and the chain stops sampling
# H — that event is surfaced as k_top == MAX_DEG (see PhysicsEngine.iterate),
# which is the signal to widen the grid or raise this multiplier.
MAX_DEG = 2 * max(K_ALL)


# ═══════════════════════════════════════════════════════════════════
#  μ calibration
# ═══════════════════════════════════════════════════════════════════
MU_N_CAL    = 4000
MU_LO       = 0.001
MU_HI       = 2.0
MU_TOL      = 0.0005
MU_MAX_ITER = 25


# ═══════════════════════════════════════════════════════════════════
#  Thermalisation (adaptive; lb-scaled)
# ═══════════════════════════════════════════════════════════════════
THERM_WINDOW    = 100
THERM_DRIFT_TOL = 0.0005
THERM_MIN_BASE  = 50
THERM_MAX_BASE  = 1000
THERM_CONFIRM   = 2
PROD_SWEEPS     = 3        # extra sweeps after thermalisation


# ═══════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════
# Generated data lives under the project's output/ directory. main.py runs
# every tool with output/ as the working directory, so these relative paths
# resolve inside it. (DATA_DIR is the output root itself, "." for that reason.)
DATA_DIR = "."
LOG_FILE = "sweep.log"
MU_JSON  = "mu_table.json"
