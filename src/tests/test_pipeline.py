"""
test_pipeline — guards the sweep's per-cell pipeline end to end.

Two regressions this catches that the figure test (test_figures.py) cannot,
because that test *synthesises* flow CSVs and only exercises the plotting:

  1. thread pinning — importing the entry point (or its plotting-dependency
     check) must NOT pull in numpy/BLAS before the per-process thread caps are
     set. If it does, OpenBLAS sizes its pool to the whole machine, every forked
     worker inherits it, and N workers x N BLAS threads thrash to a standstill:
     the spectral probe prints "[flow] SLQ:" and then hangs, with no error and
     no flow CSV. (That is exactly the bug introduced when the matplotlib
     startup check did `import matplotlib` — which imports numpy — ahead of the
     sweep's `*_NUM_THREADS=1` block.)

  2. cell pipeline — build_cell -> run_flow_test -> write the per-cell flow CSV.
     Running ONE tiny real cell and asserting a flow_*.csv lands in flow/ with
     the expected columns is what catches "cells run but no flow files appear",
     whatever the cause (a moved import, a crash in the probe, a write going to
     the wrong place). A hang fails via SIGALRM; a crash fails via the exception.

Run via `python main.py selftest` (alongside the figure test) or with pytest.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../src/tests
_SRC = os.path.dirname(_HERE)                          # .../src
_ROOT = os.path.dirname(_SRC)                          # project root

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")


def check_thread_pinning(verbose=True):
    """In a clean interpreter with the thread env vars UNSET, require that:
      (a) importing `main` sets OMP/OPENBLAS_NUM_THREADS to '1', and
      (b) neither importing `main` nor its plotting-dependency check imports
          numpy or matplotlib (which would load BLAS before the caps apply).
    Run in a subprocess so the environment and sys.modules are pristine."""
    code = (
        "import sys, os, json\n"
        "import main\n"
        "r = {\n"
        "  'omp': os.environ.get('OMP_NUM_THREADS'),\n"
        "  'openblas': os.environ.get('OPENBLAS_NUM_THREADS'),\n"
        "  'numba': os.environ.get('NUMBA_NUM_THREADS'),\n"
        "  'numpy_after_import_main': 'numpy' in sys.modules,\n"
        "}\n"
        "main._check_plotting_dep()\n"
        "r['numpy_after_depcheck'] = 'numpy' in sys.modules\n"
        "r['matplotlib_after_depcheck'] = 'matplotlib' in sys.modules\n"
        "print('RESULT' + json.dumps(r))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in _THREAD_VARS}
    problems = []
    try:
        out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                             capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        problems.append("subprocess importing `main` timed out (180s)")
        out = None

    if out is not None:
        line = next((ln for ln in out.stdout.splitlines()
                     if ln.startswith("RESULT")), None)
        if line is None:
            problems.append("subprocess produced no RESULT line "
                            f"(stderr tail: {out.stderr[-400:]!r})")
        else:
            r = json.loads(line[len("RESULT"):])
            if r.get("omp") != "1":
                problems.append(f"OMP_NUM_THREADS is {r.get('omp')!r}, expected '1' — "
                                "threads are not pinned before numpy can load")
            if r.get("openblas") != "1":
                problems.append(f"OPENBLAS_NUM_THREADS is {r.get('openblas')!r}, expected '1'")
            if r.get("numpy_after_depcheck"):
                problems.append("the plotting-dependency check imported numpy — it must "
                                "only probe availability (importlib.find_spec), otherwise "
                                "it loads BLAS before *_NUM_THREADS=1 takes effect and the "
                                "multi-worker sweep oversubscribes BLAS and hangs")
            if r.get("matplotlib_after_depcheck"):
                problems.append("the plotting-dependency check imported matplotlib "
                                "(which imports numpy) — use importlib.find_spec instead")

    ok = not problems
    if verbose:
        print("[thread-pinning] " + ("OK — BLAS/Numba pinned to 1 thread before any "
              "numpy import" if ok else "FAIL"))
        for p in problems:
            print(f"   - {p}")
    return ok, problems


def check_cell_pipeline(verbose=True):
    """Run ONE tiny real cell through the production path and assert a flow CSV
    with the contract columns is written. SIGALRM turns a hang into a failure."""
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from ds4_search import sweep_runner as sr
    from core.disk_io import mu_key

    k, T, lb, N, seed = 6, 0.0, 0.9, 300, 42
    mu_table = {mu_key(k, T, lb): 0.05}      # any sane mu; we test the *pipeline*
    params = {"n_probes": 12, "lanczos_m": 60, "half_window": 5}
    work_item = ("cell", k, T, lb, N, seed)

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="cell_pipeline_test_")
    have_alarm = hasattr(signal, "SIGALRM")
    problems = []
    try:
        os.makedirs(os.path.join(tmp, sr.FLOW_DIR), exist_ok=True)
        os.chdir(tmp)
        if have_alarm:
            def _on_timeout(signum, frame):
                raise TimeoutError("cell pipeline exceeded 150s — likely hung "
                                   "(check BLAS thread oversubscription)")
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(150)

        res = sr._run_one(work_item, params, mu_table)

        if have_alarm:
            signal.alarm(0)

        if not (isinstance(res, dict) and res.get("ok")):
            problems.append(f"_run_one did not report success: {res!r}")
        csvs = glob.glob(os.path.join(sr.FLOW_DIR, "flow_k*.csv"))
        if not csvs:
            problems.append("a successful cell wrote NO flow_k*.csv "
                            f"(dir contents: {os.listdir(sr.FLOW_DIR)})")
        else:
            with open(csvs[0]) as fh:
                header = fh.readline().strip()
            for col in ("t", "d_s_mean", "in_window"):
                if col not in header.split(","):
                    problems.append(f"flow CSV missing column '{col}' (header: {header})")
    except Exception as e:
        problems.append(f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    ok = not problems
    if verbose:
        print("[cell-pipeline]  " + ("OK — a tiny real cell produced a flow_*.csv"
              if ok else "FAIL"))
        for p in problems:
            print(f"   - {p}")
    return ok, problems


def check(verbose=True):
    """Run both pipeline guards; return (ok, detail)."""
    ok_pin, pin = check_thread_pinning(verbose=verbose)
    ok_cell, cell = check_cell_pipeline(verbose=verbose)
    ok = ok_pin and ok_cell
    return ok, {"thread_pinning": pin, "cell_pipeline": cell}


# ── pytest entry points ────────────────────────────────────────────────────
def test_thread_pinning():
    ok, problems = check_thread_pinning(verbose=False)
    assert ok, "thread-pinning regression: " + "; ".join(problems)


def test_cell_pipeline_writes_csv():
    ok, problems = check_cell_pipeline(verbose=False)
    assert ok, "cell pipeline did not produce a flow CSV: " + "; ".join(problems)


if __name__ == "__main__":
    good, _ = check(verbose=True)
    sys.exit(0 if good else 1)
