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
#  Sweep grid — loaded from the user-facing main.toml at the project root
# ═══════════════════════════════════════════════════════════════════
# The grid (what to sweep) is the one thing users tune, so it lives in
# main.toml next to main.py, not buried here. We parse it with the stdlib
# tomllib (Python 3.11+) and expose it under the internal *_ALL names the rest
# of the code already uses.
import os as _os
import tomllib as _tomllib

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_CONFIG_PATH = _os.path.join(_ROOT, "main.toml")
with open(_CONFIG_PATH, "rb") as _f:
    _cfg = _tomllib.load(_f)
_grid = _cfg["grid"]
# Optional [dashboard] table: runtime/plotting knobs kept out of [grid] so the
# grid stays purely about *what* is swept. Absent table → all defaults, so a
# config that predates these keys still loads unchanged.
_dash = _cfg.get("dashboard", {})
# Optional [compute] table: resource limits (worker/process count).
_compute = _cfg.get("compute", {})
_calib = _cfg.get("calibration", {})
# Optional [server] table: where the live dashboard binds and whether it opens
# a browser tab. Kept here (not as CLI flags) so `python main.py` takes no
# arguments — main.toml is the single place anything is configured. Absent
# table → the localhost defaults below.
_server = _cfg.get("server", {})
# Optional [engine] table: which graph-growth back-end the flow pipeline uses.
_engine = _cfg.get("engine", {})

K_ALL   = list(_grid["k"])
T_ALL   = list(_grid["T"])
LB_ALL  = list(_grid["lb"])
N_ALL   = list(_grid["N"])
# SEEDS: run seeds  seed, seed+1, …, seed+n_seeds-1  for every cell so the
# sweep samples the graph-to-graph spread, not just one graph. n_seeds is
# the single user knob (main.toml); it defaults to 1 here only as a
# back-compat safety net for an older config that predates the knob.
_SEED0   = int(_grid["seed"])
_N_SEEDS = max(1, int(_grid.get("n_seeds", 1)))
SEEDS    = [_SEED0 + _i for _i in range(_N_SEEDS)]

# REDRAW_INTERVAL_S: force the heatmaps / shape plots to redraw at least this
# often (in seconds) while the sweep is running, even when no new cell has
# finished. This is what lets config/marker changes and newly-flagged cells
# appear without waiting for the next (possibly very slow, high-N) cell to
# land. 0 disables the periodic redraw — plots then refresh only when a cell
# completes, as before. Set in main.toml under [dashboard].
REDRAW_INTERVAL_S = max(0.0, float(_dash.get("redraw_interval_s", 10.0)))

# WORKERS: how many parallel worker processes the sweep runs. 0 = auto (all
# logical cores granted to the process). Lower it to cap CPU/RAM use — on a
# memory- or cache-bound load, the physical core count (or fewer) often runs
# as fast as all logical cores while drawing less power. The --workers CLI
# flag overrides this. Set in main.toml under [compute].
WORKERS = max(0, int(_compute.get("workers", 0)))

# FLOW_REFRESH_INTERVAL_S: minimum seconds between automatic refreshes of the
# heavier flow-convergence (N→∞ plateau extrapolation) pass + its leaderboard
# charts. The cheap per-cell heatmaps/maps still refresh on every cell; the
# convergence fit is only meaningful once cells have several N and costs more,
# so it runs on this slower cadence. 0 = refresh it every time (heaviest);
# <0 = disable the automatic convergence/charts pass. Set under [dashboard].
FLOW_REFRESH_INTERVAL_S = float(_dash.get("flow_refresh_interval_s", 60.0))

# MIN_FREE_GB: the memory safety margin the sweep keeps free. Before starting
# another cell it estimates that cell's memory need (from the measured RSS of
# earlier cells vs N) and pauses — letting running cells finish first — rather
# than start one that would drop free memory below this. This stops the OS
# OOM-killer from killing the run (and losing the big in-flight cells). Set
# under [compute]; 0 disables the guard. Default 4 GB.
MIN_FREE_GB = float(_compute.get("min_free_gb", 4.0))


# ═══════════════════════════════════════════════════════════════════
#  Dashboard server (where the live page binds; not science)
# ═══════════════════════════════════════════════════════════════════
# SERVER_HOST: bind address. "127.0.0.1" (default) is localhost-only. Set to
#   "0.0.0.0" in main.toml [server] to expose on your network — the page is
#   read-only but unauthenticated, so only do that on networks you trust.
# SERVER_PORT: HTTP port (default 8000; set 0 to let the OS pick a free port).
# SERVER_OPEN_BROWSER: auto-open a browser tab on launch (default true).
SERVER_HOST = str(_server.get("host", "127.0.0.1"))
SERVER_PORT = int(_server.get("port", 8000))
SERVER_OPEN_BROWSER = bool(_server.get("open_browser", True))


# ═══════════════════════════════════════════════════════════════════
#  Graph-growth engine (which back-end evolves the graphs)
# ═══════════════════════════════════════════════════════════════════
# ENGINE_BACKEND selects the Metropolis kernel the flow pipeline runs on:
#   "auto"  (default) — prefer the native C++ engine (compiled on first use);
#                       fall back to numba if no compiler / build fails.
#   "cpp"             — C++ engine; falls back to numba if unavailable.
#   "rust"            — Rust engine; falls back to numba if unavailable.
#   "numba"           — the pure-Python/numba reference kernel (always works).
# All back-ends implement the same Hamiltonian and produce statistically
# equivalent graphs; numba is the guaranteed fallback so a sweep never fails
# for lack of a compiler.
ENGINE_BACKEND = str(_engine.get("backend", "auto")).lower()


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
MU_N_CAL    = int(_calib.get("n_cal", 4000))   # [calibration].n_cal
MU_LO       = 0.001
MU_HI       = 2.0
MU_TOL      = 0.0005
MU_MAX_ITER = 25

# μ-vs-N drift correction (see disk_io.mu_lookup for the mechanism).
# The sweep audits realised k_avg against the target k from the meta sidecars
# it has already written, and — where the relative drift at some N exceeds
# MU_DRIFT_TOL with at least MU_DRIFT_MIN_SEEDS seeds measured — stores a
# first-order corrected μ under an N-qualified key, used only by rungs that
# have produced no data yet. All three knobs live in [calibration]:
#     drift_tol = 0.02, drift_min_seeds = 2, n_correction = true
MU_DRIFT_TOL       = float(_calib.get("drift_tol", 0.02))
MU_DRIFT_MIN_SEEDS = int(_calib.get("drift_min_seeds", 2))
MU_N_CORRECTION    = bool(_calib.get("n_correction", True))


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
MU_JSON  = "mu_table.json"
# Registry of N-qualified μ entries that are PREDICTED (extrapolated from the
# measured μ-vs-N trend) rather than measured. Predicted entries refresh on
# every drift audit until their rung produces real data; measured corrections
# are pinned. The split lives in this sidecar so mu_table.json itself stays a
# plain {key: μ} map every existing reader already understands.
MU_PRED_JSON = "mu_predictions.json"
FLOW_DIR = "flow"          # per-cell flow_*.csv / meta_*.json / quad_*.npz
