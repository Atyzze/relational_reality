# physics_tests — validation experiments on developed graphs

These are **physics validations**, not unit tests. Each one builds (or rebuilds)
a graph with the *same* engine the sweep uses and runs an extra probe on it, to
check that the emergent geometry behaves the way real space should.

They live here — a subfolder that imports the engine — rather than in a separate
project, because every probe reuses `core/` (graph build), `metrics/` (kernels),
and the reference-lattice machinery. Forking would duplicate all of that. A new
probe is just a new file here.

## Running

From the project root:

```bash
python src/physics_tests/isotropy.py --k 8 --T 0 --lb 0.99 --N 16000
```

Each script is self-contained, takes the cell coordinates on the command line,
loads `mu_table.json` (calibrating that one cell if needed), and writes its
report + plot into the output directory.

## The tests

- **isotropy.py** — propagates a heat field from random source nodes and checks
  that it spreads as a uniform expanding sphere: the field should be uniform
  across each graph-distance shell (no preferred "direction", no hidden
  fast-path / "highway"), and the ball volume should grow like r^d. It runs the
  same probe on a matched reference torus (which is isotropic by construction)
  so you have a baseline to compare against — the question is whether the
  disordered graph is *as uniform as* a lattice of the same size, not whether
  it is perfectly uniform.

- **optimal_workers.py** — a CPU stress test (not a physics one, but it belongs
  here for the same reason: it should run alone, with its own clean console).
  It builds one fixed 256k cell over and over at a 1, 4, 8, 16, … all-cores
  ladder, prints a per-core-efficiency table (per-core speed falls as cores
  contend for memory; total throughput still climbs), and prints the
  recommended `workers` value. It does **not** edit `config.toml` — set the
  value there yourself.

## Adding a test

Copy `isotropy.py`'s header (the `sys.path` setup that puts `src/` on the path),
import what you need from `core` / `metrics`, build a cell with `build_cell`,
and write your probe. Keep it runnable on its own.
