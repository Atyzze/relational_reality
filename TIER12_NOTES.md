# Tier-1 + Tier-2 capture upgrade — notes

Goal of this change: **never have to re-run a scan to recover something we
forgot to save.** Capture now persists the cheap *primitive* one step
upstream of the derived quantities, so future analysis changes are
post-processing, not recompute. All changes are **additive and forward-only**:
existing `flow_*.csv` stay valid; new cells gain the extra sidecars.

## What now gets written, per cell (in `output/flow/`)

Alongside the existing `flow_<tag>.csv` (unchanged), each cell now also writes:

- **`meta_<tag>.json`** (Tier 1) — agnostic graph properties + full provenance:
  - topology: `lcc_pct`, `transitivity`, `tri_per_edge`, `assortativity`,
    `k_avg/k_min/k_max`, `edges`, `triangles`, `k_top`, `N_eff`
  - `degree_hist` (full-graph degree histogram) and `sum_deg_sq` (Σd²),
    giving the exact Hamiltonian energy `energy_H = ec·E + μ·Σd²`
  - `therm_trace`: ~120 points of `(sweep, k_avg, sum_deg_sq)` across
    thermalisation — lets you **audit equilibration post-hoc on the
    structural observable (Σd² / energy)**, not just mean degree, without
    re-running the cell
  - provenance: `mu`, `ec`, `max_degree`, `code_hash`, SLQ params
    (`n_probes`, `lanczos_m`, `half_window`), t-grid (`t_lo/t_hi/n_t`),
    `therm_sweeps`, `prod_sweeps`, `wall_s` — every cell is self-describing,
    so you can always tell whether old cells are comparable after a code change.

- **`quad_<tag>.npz`** (Tier 2) — the per-probe SLQ **Ritz quadrature**
  `(theta, omega)`, the primitive SLQ produces before it is projected onto a
  t-grid. With it on disk, `Z(t) = N_eff · mean_p Σ_i ω[p,i]·exp(−t·θ[p,i])`,
  so the **t-grid, UV reach, window bounds, half-window smoothing, and the
  jackknife all become post-processing decisions you can change forever
  without re-running SLQ.** ~150–220 KB/cell vs the minutes-to-hours SLQ costs.
  θ kept float64 (small near-zero eigenvalues govern the IR), ω float32.

Both sidecars are written atomically (tmp + `os.replace`), same as the flow
CSV — a killed worker leaves at most a `.tmp`.

## Re-deriving d_s(t) on a new grid, for free

```python
from core.flow_probe import reconstruct_Z, compute_ds_flow
import numpy as np

# rebuild Z(t) on ANY grid from the saved quadrature (no SLQ rerun):
t, Z = reconstruct_Z("output/flow/quad_k8_T0.0_lb0.995_N1024000_s42.npz")
# e.g. push the UV a decade lower / densify, then re-extract d_s(t):
t2 = np.logspace(np.log10(t[0]/10), np.log10(t[-1]), 600)
_, Z2 = reconstruct_Z("output/flow/quad_...npz", t_grid=t2)
```

(Verified: reconstructed Z matches the original to ~1e-8.)

## Files changed (all additive)

- `src/core/flow_probe.py` — `slq_per_probe_Z(..., return_quad=True)` returns
  `(theta, omega)`; `FlowResult.quad`; new `write_quad_npz()` and
  `reconstruct_Z()`. Existing callers/return shapes unchanged when
  `return_quad` is omitted.
- `src/core/graph_builder.py` — `_thermalise(..., trace_out=None)` records the
  equilibration trace (no return/signature change for existing callers);
  `build_graph` stashes it on `eng.therm_trace`.
- `src/core/cell_tests.py` — `CellState` gains `deg_hist / sum_deg_sq /
  therm_trace` (defaulted); `build_cell` populates them; `run_flow_test`
  captures the quadrature into `FlowResult.quad`.
- `src/ds4_search/sweep_runner.py` — writes `meta_*.json` + `quad_*.npz` after
  each flow CSV; auto-refreshes `flow_modes` next to `shape_analysis`.
- `src/flow_modes.py` — NEW dual-mode (flat-4D vs flowing-4D) analyzer
  (LCC≥90% gate, residual-vs-4D-torus anchoring). Auto-picked up by the sweep.
- `flow_convergence.py`, `flow_charts.py` — standalone analysis helpers (run
  from the project root against `output/`).

## Restart sequence

1. (already in this zip) modules in place.
2. Add the new T value to `config.toml` if desired (e.g. `T = [0, 0.001,
   0.005, 0.02, 0.05]`) — new `(k, 0.02, lb)` tuples auto-calibrate μ forward.
3. Restart the sweep. Existing cells skip (cached); new cells write the
   sidecars. Confirm `↻ flow-mode analysis refreshed` in the log and that
   `meta_*.json` + `quad_*.npz` appear in `output/flow/`.

Note: existing cells keep their `from_Z` LCC estimate and have no quadrature/
transitivity (they predate this); only cells grown after the restart get the
full sidecars. That is expected — capture is future-proofed forward, not
back-filled.

## Dashboard fixes (live page)

`src/ds4_search/live_app_page.py` only — the live dashboard `main.py` serves.
(`dashboard_page.py`/`static_dashboard.py` are the separate offline export and
were not touched.)

- **Chart blank until next datapoint, on (re)open.** `renderAll()` began with
  `if (!STATE) return;`, so the plot/leaderboards/table — which need only the
  curve data (`DATA`), not the sweep status (`STATE`) — were blocked whenever
  `/api/state` lagged the first poll. Now the data panels render independently
  of `STATE`; existing curves draw on first load. The banner still waits for
  `STATE` (it's the only thing that needs it).
- **Chart freezes after toggling buttons (chips stay responsive, F5 recovers).**
  `poll()` rebuilt the full SVG every 4 s unconditionally, with no re-entrancy
  guard — as the dataset grew these routine rebuilds stacked and fought user
  toggles for the main thread. Now: `poll()` has a re-entrancy guard (one poll
  at a time), and only re-renders the heavy panels when the dataset actually
  changed (`data_hash`); user interactions still re-render immediately. Each
  panel is also wrapped so one panel's throw can't abort the others or bubble
  out of the poll loop.
- **Log colour scale.** When the colour axis spans ≥1000× (e.g. N = 1e3…1e9)
  the curve colouring now maps in log space (legend shows "(log)"); small-range
  axes (k, T, lb) stay linear.

### Correction — the actual freeze root cause (from a captured console error)

The "freeze after toggles" item above attributed the hang to polls stacking /
re-render competition. A captured `RangeError: Maximum call stack size
exceeded` in `renderPlot` proved that was the wrong diagnosis. The real cause:
`renderPlot` computed axis ranges with `Math.min(...allT_full)` /
`Math.max(...allD)`, spreading the per-point accumulator arrays (~240 points ×
every visible curve) as function arguments. JS spreads hit the engine's
argument-count limit (~65k in V8); at ~267 cells × 240 ≈ 64k points the call
threw, which is exactly why the freeze appeared only once the sweep had grown
and why toggling *more* cells on triggered it (over the limit) while toggling
cells off recovered it (under the limit).

Fix: loop-based `arrMin` / `arrMax` that never spread. (Verified: `Math.min(...
arr)` throws on a 200k array; `arrMin` returns correctly.) The earlier
re-entrancy guard, render-on-change, and per-panel `safe()` isolation remain as
genuine robustness — and are why the error got *logged* instead of silently
freezing the page — but the crash itself was the spread, now removed.

## Figure annotations (counts + glyph legend)

So every saved PNG is self-dating and the glyphs are explained:
- **shape_heatmap.png** — bottom caption: `N cell-records · S seed(s) present ·
  updated <timestamp>`. (Its `·` / `×` glyph meanings were already in the
  subtitle.)
- **flow_modes_plane.png** — title now carries `N cells · M pass LCC≥X% · K
  excluded (gray ×) · updated <time>`, plus a real legend: green = flat-4D
  leaning (LCC≥gate), cyan = flowing-4D leaning (LCC≥gate), gray × = LCC below
  gate (excluded from the mode call), ★ = ideal target (marker size ∝ score).
- **flow_modes_maps.png** — suptitle now carries the same cell/LCC counts +
  timestamp.

The gray × was never a bug — it means "LCC below the gate, so excluded from the
4D call" (and in the heatmap, `×` separately means "no data file yet"). The
counts give an at-a-glance check that a refresh actually happened.
