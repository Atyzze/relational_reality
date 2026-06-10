"""
Regression test: the sweep's heatmap / chart PNGs must keep being generated.

Synthesizes a tiny but REAL dataset and runs the actual figure generators,
then asserts the figures appear. Two tiers:

  STRICT  — must produce these exact PNGs (the headline heatmaps + flow charts):
            shape_analysis -> shape_heatmap.png, shape_heatmap_flatness.png,
                              shape_heatmap_seed_std.png, shape_histogram.png
            flow_charts    -> flow_map.png, flow_scatter.png
  NO-CRASH — flow_modes / flow_convergence must import and run without raising
            (they legitimately need 4D-torus reference curves to emit their own
            maps, which synthetic data can't fake — but a crash/import break is
            still a real regression and fails the test).

The most common breakage this catches: matplotlib missing/broken in the env, an
import moved by a refactor, or a generator throwing. Run:

    python main.py selftest          # friendly runner
    pytest src/tests/test_figures.py  # CI
"""
import importlib
import math
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

STRICT = {
    "shape_analysis": ["shape_heatmap.png", "shape_heatmap_flatness.png",
                       "shape_heatmap_seed_std.png", "shape_histogram.png"],
    "flow_charts": ["flow_map.png", "flow_scatter.png"],
}
NO_CRASH = ["flow_modes", "flow_convergence"]

_KS, _LBS, _NS, _SEEDS = (6, 8, 10), (0.90, 0.99), (1000, 2000, 4000), (0, 1)


def _write_cell(path, k, n, seed):
    base = 3.7 + 0.25 * ((k - 8) / 4.0) + 0.05 * seed - 0.1 * (n > 2000)
    with open(path, "w") as f:
        f.write("t,d_s_mean,in_window\n")
        for i in range(48):
            t = 0.08 * (i + 1)
            d = base * (1.0 - math.exp(-t * 1.6)) + 0.02 * math.sin(i * 0.7)
            f.write(f"{t:.4f},{d:.5f},{1 if i >= 22 else 0}\n")


def _write_convergence(path):
    """flow_convergence.csv in the exact column set flow_charts.load reads."""
    cols = ["k", "T", "lb", "d_inf", "d_inf_ci", "value_at_maxN", "dev4_at_maxN",
            "N_max", "seedstd_dir", "verdict"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in _KS:
            for lb in _LBS:
                val = 4.0 + 0.15 * (k - 8) / 2.0
                f.write(f"{k},0.0,{lb},{val:.4f},0.18,{val:.4f},0.12,"
                        f"{max(_NS)},,flat-4D\n")


def make_dataset(d):
    flow = os.path.join(d, "flow")
    os.makedirs(flow, exist_ok=True)
    for k in _KS:
        for lb in _LBS:
            for n in _NS:
                for s in _SEEDS:
                    _write_cell(os.path.join(flow, f"flow_k{k}_T0.0_lb{lb}_N{n}_s{s}.csv"), k, n, s)
    _write_convergence(os.path.join(d, "flow_convergence.csv"))


def run_generators(d):
    """Run each generator with `d` as cwd (mirrors the sweep). Returns
    (pngs:set, crashed:dict mod->error)."""
    cwd = os.getcwd()
    os.chdir(d)
    crashed = {}
    try:
        plan = [("shape_analysis", ["--dir", "."]),
                ("flow_charts", ["--csv", "flow_convergence.csv"]),
                ("flow_modes", ["--dir", "."]),
                ("flow_convergence", ["--dir", "."])]
        for mod, margs in plan:
            try:
                importlib.import_module(mod).main(margs)
            except SystemExit:
                pass
            except Exception as e:
                crashed[mod] = f"{type(e).__name__}: {e}"
        return {f for f in os.listdir(d) if f.endswith(".png")}, crashed
    finally:
        os.chdir(cwd)


def check(verbose=True):
    with tempfile.TemporaryDirectory() as d:
        make_dataset(d)
        pngs, crashed = run_generators(d)
    missing = {}
    for mod, files in STRICT.items():
        miss = [f for f in files if f not in pngs]
        if miss:
            missing[mod] = miss
    crash_in_nocrash = {m: e for m, e in crashed.items()}
    ok = (not missing) and (not crash_in_nocrash)
    if verbose:
        print(f"PNGs produced ({len(pngs)}): {', '.join(sorted(pngs)) or '(none)'}")
        for mod, miss in missing.items():
            print(f"  MISSING (strict) from {mod}: {', '.join(miss)}")
        for mod, err in crash_in_nocrash.items():
            print(f"  CRASHED {mod}: {err}")
        print("OK — figure pipeline healthy." if ok else "\nFigure pipeline regression detected (above).")
    return ok, pngs, {"missing": missing, "crashed": crash_in_nocrash}


def test_figures_generated():
    ok, pngs, detail = check(verbose=False)
    assert ok, f"figure regression: {detail} (produced {sorted(pngs)})"


if __name__ == "__main__":
    ok, _, _ = check()
    raise SystemExit(0 if ok else 1)
