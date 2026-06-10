#!/usr/bin/env python3
"""bench_workers.py — find the worker count that maximises cells/min.

This stands apart from the sweep. It builds ONE representative cell over and
over at several pool sizes and reports throughput (cells/min), so you can see
directly whether all cores, half, or fewer is fastest on *your* machine — and,
wrapped in `taskset`, whether pinning to the V-cache CCD wins.

Why a separate script and not just timing the sweep: the sweep resumes from
cache, so a "finished" cell is a no-op and the timing is meaningless. This
calls build_cell + run_flow_test directly — every measured cell is real work.

The decision rule is simple: **peak cells/min is your optimum.** If it peaks
at your physical-core count and stays flat to 2× that, SMT isn't helping. If
it peaks *below* the core count, you're cache/bandwidth-bound and fewer
workers genuinely win — exactly the regime the 9950X3D's asymmetric L3 (one
CCD with 3D V-Cache, one without) can create.

USAGE  (run from the project root)
  # defaults: cell k8 T0 lb0.995 at N=64000, sweep {4,8,12,16,24,32}
  .venv/bin/python src/bench_workers.py

  # pick the cell, N, worker list, and reps-per-config
  .venv/bin/python src/bench_workers.py --k 8 --T 0 --lb 0.995 --N 256000 \
        --workers 4 8 16 32 --reps 24

  # pin the WHOLE run to one CCD to isolate the V-cache effect.
  # first find the V-cache cores (they report a much larger L3):
  #   for c in /sys/devices/system/cpu/cpu[0-9]*; do \
  #     printf "%s L3=%s\n" "$(basename $c)" \
  #       "$(cat $c/cache/index3/size 2>/dev/null)"; done | sort -V
  # the ~96M cores are the V-cache CCD (say 0-7); then:
  taskset -c 0-7  .venv/bin/python src/bench_workers.py --workers 4 8
  taskset -c 8-15 .venv/bin/python src/bench_workers.py --workers 4 8  # compare

Every math backend (BLAS + numba) is pinned to one thread per process BEFORE
numpy/numba import — same as the sweep — so a pool of W workers uses W cores,
not W×threads. Without this a "16-worker" run silently spawns hundreds of
threads and the numbers are noise.
"""
import os

# --- pin threads BEFORE numpy/numba are imported anywhere ------------------
# setdefault, so an explicit env from the caller still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
# This file lives in src/, so src/ itself is the import root (core.*, metrics.*).
_SRC = _HERE
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.cell_tests import build_cell, run_flow_test          # noqa: E402
from core.disk_io import load_mu_table, mu_key                 # noqa: E402
from core.graph_builder import calibrate_mu                    # noqa: E402


def _init_worker():
    """Belt-and-suspenders: hard-cap numba threads inside each worker, in
    case the env pin above didn't reach it (some builds read it lazily)."""
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _one_cell(job):
    """Top-level (picklable) job: build one cell + run the flow probe; return
    its wall seconds. Per-cell stdout (the build/therm logs) is silenced so
    the benchmark table stays readable."""
    (k, T, lb, N, seed, mu_table,
     n_probes, lanczos_m, half_window) = job
    t0 = time.perf_counter()
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        cell = build_cell(k, T, lb, N, seed, mu_table)
        run_flow_test(cell, n_probes=n_probes, lanczos_m=lanczos_m,
                      half_window=half_window)
    return time.perf_counter() - t0


def run_benchmark(k, T, lb, N, workers, reps=0, reps_per_core=0,
                  n_probes=60, lanczos_m=300, half_window=10, ec=-1.0,
                  warmup=False, mu_table=None, log=lambda *a: None):
    """Measure cells/min at each worker count for one fixed cell, bypassing
    the sweep cache. Returns [(W, cells_per_min), ...] in the given order.

    `reps_per_core` (if > 0) sets the cells per config to per_core × W, so each
    core runs the same number regardless of W (apples-to-apples). Otherwise the
    fixed `reps` is used for every W. The cell's μ is taken from the μ-table
    (calibrated once if absent). `log` receives progress strings (default
    silent) so both the CLI and the sweep's auto-tune can route output.
    """
    if mu_table is None:
        mu_table = load_mu_table()
    key = mu_key(k, T, lb)
    if key not in mu_table:
        log(f"{key} not calibrated — calibrating once (small-N) …")
        mu, err, k_act = calibrate_mu(k, T, lb, ec, verbose=False)
        mu_table[key] = mu
        log(f"calibrated μ={mu:.5f} (k_actual≈{k_act:.2f}, err {err:.3f})")
    aff = (len(os.sched_getaffinity(0))
           if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
    fixed_reps = reps or max(12, min(96, 3 * max(workers)))
    if reps_per_core > 0:
        log(f"cell k={k} T={T:g} lb={lb:g} N={N:,} μ={mu_table[key]:.5f}; "
            f"{reps_per_core} cells/core/cycle; {aff} CPUs in mask")
    else:
        log(f"cell k={k} T={T:g} lb={lb:g} N={N:,} μ={mu_table[key]:.5f}; "
            f"reps/config={fixed_reps}; {aff} CPUs in mask")
    if warmup:
        log("warmup cell (JIT + page warm) …")
        _one_cell((k, T, lb, N, 0, mu_table, n_probes, lanczos_m, half_window))
    results = []
    for W in workers:
        Weff = min(W, aff)                       # never oversubscribe the mask
        reps_W = reps_per_core * Weff if reps_per_core > 0 else fixed_reps
        jobs = [(k, T, lb, N, s, mu_table, n_probes, lanczos_m, half_window)
                for s in range(reps_W)]
        per_cell = []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=Weff,
                                 initializer=_init_worker) as ex:
            for f in as_completed([ex.submit(_one_cell, j) for j in jobs]):
                per_cell.append(f.result())
        wall = time.perf_counter() - t0
        cpm = reps_W / wall * 60.0
        results.append({
            "workers": W, "workers_eff": Weff, "cells": reps_W,
            "wall_s": wall, "cells_per_min": cpm,
            "s_per_cell": statistics.median(per_cell) if per_cell else wall,
        })
        wlabel = f"{W}" if Weff == W else f"{W}→{Weff}"
        log(f"W={wlabel:>5}: {cpm:7.1f} cells/min  "
            f"({reps_W} cells, s/cell med "
            f"{statistics.median(per_cell) if per_cell else float('nan'):.2f})")
    return results


def pick_best(results):
    """Worker count with the highest cells/min (0 if there are no results)."""
    return max(results, key=lambda r: r["cells_per_min"])["workers"] \
        if results else 0


def format_table(results):
    """A full per-core-efficiency table from run_benchmark results. Per-core
    efficiency = (1-worker s/cell) / (this s/cell): 100% when a core is as fast
    as it is solo, lower when more cores contend for memory/cache. Speedup is
    aggregate cells/min vs the 1-worker baseline — so you can see efficiency
    fall while total throughput still climbs."""
    if not results:
        return "  (no benchmark results)"
    base_sc = next((r["s_per_cell"] for r in results if r["workers"] == 1),
                   results[0]["s_per_cell"])
    base_cpm = next((r["cells_per_min"] for r in results if r["workers"] == 1),
                    results[0]["cells_per_min"]) or 1e-9
    best_w = pick_best(results)
    lines = [
        f"  {'workers':>7}  {'cells':>6}  {'s/cell':>7}  {'core-eff':>8}  "
        f"{'cells/min':>9}  {'speedup':>8}",
        "  " + "-" * 58,
    ]
    for r in results:
        wl = (str(r["workers"]) if r["workers"] == r["workers_eff"]
              else f"{r['workers']}→{r['workers_eff']}")
        eff = (base_sc / r["s_per_cell"] * 100.0) if r["s_per_cell"] else 0.0
        spd = r["cells_per_min"] / base_cpm
        mark = "  ← fastest" if r["workers"] == best_w else ""
        lines.append(
            f"  {wl:>7}  {r['cells']:>6}  {r['s_per_cell']:>7.2f}  "
            f"{eff:>7.0f}%  {r['cells_per_min']:>9.1f}  {spd:>7.2f}x{mark}")
    lines.append("  " + "-" * 58)
    lines.append("  core-eff = per-core speed vs 1 worker (drops as cores "
                 "contend); speedup = total throughput vs 1 worker.")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  CPU topology (Linux sysfs) — feeds the per-core / V-Cache report
# ════════════════════════════════════════════════════════════════════
def _read_first_line(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _size_to_bytes(s):
    """'32768K' / '96M' / '1024' → bytes."""
    if not s:
        return None
    s = s.strip()
    mult = 1
    if s and s[-1] in "KkMmGg":
        mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[s[-1].upper()]
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return None


def _parse_cpu_list(s):
    """'0-7,16-23' → [0..7, 16..23]."""
    out = []
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _percentile(vals, q):
    """Linear-interpolated percentile q∈[0,1]. stdlib-only companion to
    statistics.median so we don't pull numpy into the worker."""
    s = sorted(vals)
    if not s:
        return float("nan")
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    fr = pos - lo
    return float(s[lo] * (1 - fr) + s[hi] * fr)


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def detect_topology(base="/sys/devices/system/cpu"):
    """Read per-CPU cache + topology from Linux sysfs and group logical CPUs
    into CCDs by shared L3. Returns the dict bench_report expects. Falls back to
    a flat single-group topology off os.cpu_count() when sysfs is unreadable
    (non-Linux, or a restricted container).

    A CCD = the set of logical CPUs that share one L3. The V-Cache CCD on an
    asymmetric part (e.g. 9950X3D) is detected as the group with the markedly
    larger L3 (> 1.5× the smallest), so no hard-coded core lists are needed.
    """
    import glob
    import re
    n_os = os.cpu_count() or 1
    fallback = {
        "source": "fallback", "model": _cpu_model(),
        "n_logical": n_os, "n_physical": n_os, "smt": False, "cpus": {},
        "ccds": [{"id": 0, "cpus": list(range(n_os)), "l3_bytes": None,
                  "is_vcache": False}],
        "vcache_ccd_ids": [],
    }
    try:
        cpu_dirs = glob.glob(os.path.join(base, "cpu[0-9]*"))
    except OSError:
        return fallback
    cpus = {}
    for d in cpu_dirs:
        m = re.search(r"cpu(\d+)$", d)
        if not m or not os.path.isdir(os.path.join(d, "cache")):
            continue
        cid = int(m.group(1))
        ci = _read_first_line(os.path.join(d, "topology", "core_id"))
        pk = _read_first_line(os.path.join(d, "topology",
                                           "physical_package_id"))
        info = {"l1d": None, "l2": None, "l3": None, "l3_group": None,
                "core_id": int(ci) if ci and ci.isdigit() else None,
                "package": int(pk) if pk and pk.isdigit() else None}
        for idx in glob.glob(os.path.join(d, "cache", "index*")):
            lvl = _read_first_line(os.path.join(idx, "level"))
            typ = _read_first_line(os.path.join(idx, "type"))
            sz = _size_to_bytes(_read_first_line(os.path.join(idx, "size")))
            shr = _read_first_line(os.path.join(idx, "shared_cpu_list"))
            if lvl == "1" and typ == "Data":
                info["l1d"] = sz
            elif lvl == "2":
                info["l2"] = sz
            elif lvl == "3":
                info["l3"] = sz
                info["l3_group"] = tuple(sorted(_parse_cpu_list(shr)))
        cpus[cid] = info
    if not cpus:
        return fallback
    groups = {}
    for cid, info in cpus.items():
        groups.setdefault(info["l3_group"] or (cid,), []).append(cid)
    ccds = []
    for i, (key, members) in enumerate(sorted(groups.items())):
        l3 = next((cpus[c]["l3"] for c in members if cpus[c]["l3"]), None)
        ccds.append({"id": i, "cpus": sorted(members), "l3_bytes": l3,
                     "is_vcache": False})
    l3s = [c["l3_bytes"] for c in ccds if c["l3_bytes"]]
    vcache_ids = []
    if l3s and max(l3s) > min(l3s) * 1.5:        # asymmetric L3 → V-Cache part
        big = max(l3s)
        for c in ccds:
            if c["l3_bytes"] and c["l3_bytes"] >= big * 0.99:
                c["is_vcache"] = True
                vcache_ids.append(c["id"])
    phys = len({(i["package"], i["core_id"]) for i in cpus.values()
                if i["core_id"] is not None}) or len(cpus)
    return {"source": "sysfs", "model": _cpu_model(),
            "n_logical": len(cpus), "n_physical": phys,
            "smt": phys < len(cpus), "cpus": cpus, "ccds": ccds,
            "vcache_ccd_ids": vcache_ids}


def _cpu_of_self():
    """CPU the process's MAIN thread last executed on, from /proc/self/stat
    field 39 (parsed robustly past the parenthesised comm). Reading the proc
    file means a *sampler* thread can poll where the *compute* thread is — which
    os.sched_getcpu() can't do, since it reports its own caller. Falls back to
    sched_getcpu when /proc is unavailable."""
    try:
        with open("/proc/self/stat") as f:
            data = f.read()
        after = data[data.rfind(")") + 2:].split()
        return int(after[36])     # field 39 → index 36 after the comm field
    except Exception:
        try:
            return os.sched_getcpu()
        except Exception:
            return None


def _seeds(n, random_seeds, rng_seed=12345):
    """n seeds for a run: distinct random draws (so every cell is genuinely
    different work — the 'keep picking at random') or a plain 0..n-1 range."""
    if not random_seeds:
        return list(range(n))
    import random
    return random.Random(rng_seed).sample(range(1, 10_000_000), n)


def _read_rss_bytes():
    """Resident set size in bytes from /proc/self/status VmRSS (None off-Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _one_cell_detailed(job):
    """build_cell + run_flow_test, returning a per-cell record. Times the graph
    BUILD and the d_s(t) PROBE separately (the probe is the repeated-scan,
    cache-sensitive part), samples the CPU it ran on and its peak RSS every
    20 ms, and — if `pin` is set — pins this worker to that one logical CPU for
    the run (used by the per-core profile and the cache map). Per-cell
    build/therm stdout is silenced.

    The dominant CPU is the modal sample, with `cpu_stability` = the fraction
    of samples on it (1.0 = never migrated; lower = the scheduler moved it
    around mid-cell, so attributing its time to one core is approximate)."""
    import threading
    from collections import Counter
    (k, T, lb, N, seed, mu_table,
     n_probes, lanczos_m, half_window, pin) = job
    if pin is not None and hasattr(os, "sched_setaffinity"):
        try:
            if pin in os.sched_getaffinity(0):
                os.sched_setaffinity(0, {pin})
        except OSError:
            pass
    hist = Counter()
    rss_peak = [0]
    stop = threading.Event()

    def _sampler():
        while not stop.wait(0.02):
            c = _cpu_of_self()
            if c is not None:
                hist[c] += 1
            r = _read_rss_bytes()
            if r and r > rss_peak[0]:
                rss_peak[0] = r

    th = threading.Thread(target=_sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        tb = time.perf_counter()
        cell = build_cell(k, T, lb, N, seed, mu_table)
        build_s = time.perf_counter() - tb
        tp = time.perf_counter()
        run_flow_test(cell, n_probes=n_probes, lanczos_m=lanczos_m,
                      half_window=half_window)
        probe_s = time.perf_counter() - tp
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=0.5)
    end_cpu = _cpu_of_self()
    dom = hist.most_common(1)[0][0] if hist else end_cpu
    stab = (hist[dom] / sum(hist.values())) if hist else 1.0
    return {"N": N, "seed": seed, "wall_s": wall, "build_s": build_s,
            "probe_s": probe_s, "cpu": dom, "cpu_end": end_cpu,
            "pinned_to": pin, "cpu_stability": round(stab, 3),
            "rss_peak_bytes": rss_peak[0] or None,
            "samples": sum(hist.values())}


def run_throughput_detailed(k, T, lb, N, workers, reps_per_core=3, reps=0,
                            n_probes=60, lanczos_m=300, half_window=10, ec=-1.0,
                            warmup=True, random_seeds=True, mu_table=None,
                            topology=None, log=lambda *a: None):
    """Throughput sweep (cells/min vs worker count) that ALSO records, per
    cell, its wall time and the CPU it ran on. Returns (summary, per_cell):
    `summary` is the same shape run_benchmark returns (so format_table /
    pick_best apply); `per_cell` is a flat list of records tagged with the
    worker count they ran under and the CCD their dominant CPU belongs to."""
    if mu_table is None:
        mu_table = load_mu_table()
    key = mu_key(k, T, lb)
    if key not in mu_table:
        log(f"{key} not calibrated — calibrating once (small-N) …")
        mu, err, k_act = calibrate_mu(k, T, lb, ec, verbose=False)
        mu_table[key] = mu
        log(f"calibrated μ={mu:.5f} (k_actual≈{k_act:.2f}, err {err:.3f})")
    if topology is None:
        topology = detect_topology()
    aff = (len(os.sched_getaffinity(0))
           if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
    fixed_reps = reps or max(12, min(96, 3 * max(workers)))
    if warmup:
        log("warmup cell (JIT + page warm) …")
        _one_cell_detailed((k, T, lb, N, 0, mu_table, n_probes, lanczos_m,
                            half_window, None))

    def _ccd(cpu):
        for c in topology.get("ccds", []):
            if cpu in c["cpus"]:
                return c["id"]
        return None

    summary, per_cell = [], []
    for W in workers:
        Weff = min(W, aff)
        reps_W = reps_per_core * Weff if reps_per_core > 0 else fixed_reps
        seeds = _seeds(reps_W, random_seeds, rng_seed=10_000 + W)
        jobs = [(k, T, lb, N, s, mu_table, n_probes, lanczos_m, half_window,
                 None) for s in seeds]
        recs = []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=Weff,
                                 initializer=_init_worker) as ex:
            for f in as_completed([ex.submit(_one_cell_detailed, j)
                                   for j in jobs]):
                r = f.result()
                r["workers"] = W
                r["ccd"] = _ccd(r.get("cpu"))
                recs.append(r)
        wall = time.perf_counter() - t0
        walls = sorted(x["wall_s"] for x in recs)
        cpm = reps_W / wall * 60.0
        summary.append({
            "workers": W, "workers_eff": Weff, "cells": reps_W,
            "wall_s": wall, "cells_per_min": cpm,
            "s_per_cell": statistics.median(walls) if walls else wall,
            "p10_s": _percentile(walls, 0.1), "p90_s": _percentile(walls, 0.9),
        })
        per_cell.extend(recs)
        wl = f"{W}" if Weff == W else f"{W}→{Weff}"
        log(f"W={wl:>5}: {cpm:7.1f} cells/min  ({reps_W} cells, s/cell med "
            f"{statistics.median(walls) if walls else float('nan'):.2f})")
    return summary, per_cell


def profile_per_core(k, T, lb, N, cores=None, reps_per_core=3,
                     n_probes=60, lanczos_m=300, half_window=10, ec=-1.0,
                     random_seeds=True, mu_table=None, topology=None,
                     warmup=True, log=lambda *a: None):
    """Run cells ONE AT A TIME, each pinned to a specific logical CPU,
    round-robin across `cores` (default: every CPU in the affinity mask),
    `reps_per_core` cells each. With no concurrency, a core's median time
    isolates its own cache/memory path, so an asymmetric-L3 (V-Cache) CCD shows
    up as a systematically faster median. Returns (percore, per_cell):
    percore[cpu] = {median_s, p10_s, p90_s, n, samples}; per_cell is the raw
    records with pinned_to set.

    This is the deliberate counterpart to the random-scheduled throughput
    sweep: random scheduling rarely visits every core evenly (the kernel keeps
    a process on one core for cache affinity), so to actually *compare* cores
    we pin. Cost is reps_per_core × (cores) cells run serially — keep
    reps_per_core small, or pass a `cores` subset (e.g. one core per CCD)."""
    if not hasattr(os, "sched_setaffinity"):
        log("per-core profile needs os.sched_setaffinity (Linux) — skipping.")
        return {}, []
    if mu_table is None:
        mu_table = load_mu_table()
    key = mu_key(k, T, lb)
    if key not in mu_table:
        mu, _, _ = calibrate_mu(k, T, lb, ec, verbose=False)
        mu_table[key] = mu
    if topology is None:
        topology = detect_topology()
    mask = sorted(os.sched_getaffinity(0))
    cores = [c for c in (cores if cores is not None else mask) if c in mask]
    if not cores:
        log("no cores in affinity mask to profile — skipping.")
        return {}, []
    if warmup:
        log("warmup cell (JIT + page warm) …")
        _one_cell_detailed((k, T, lb, N, 0, mu_table, n_probes, lanczos_m,
                            half_window, cores[0]))
    n_jobs = len(cores) * reps_per_core
    seeds = _seeds(n_jobs, random_seeds, rng_seed=777)
    jobs, si = [], 0
    for _rep in range(reps_per_core):
        for c in cores:
            jobs.append((k, T, lb, N, seeds[si], mu_table,
                         n_probes, lanczos_m, half_window, c))
            si += 1
    # Randomise visit order: each core is then sampled across the whole run
    # rather than in one block, so turbo/thermal drift averages out instead of
    # systematically favouring whichever core went first.
    import random
    random.Random(2024).shuffle(jobs)
    log(f"per-core profile: {len(cores)} core(s) × {reps_per_core} "
        f"= {n_jobs} cells, one at a time (pinned, randomised order).")
    per_cell, done = [], 0
    # max_workers=1 → strictly one cell at a time; each job pins itself.
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker) as ex:
        for f in as_completed([ex.submit(_one_cell_detailed, j) for j in jobs]):
            per_cell.append(f.result())
            done += 1
            if done % max(1, len(cores)) == 0:
                log(f"  …{done}/{n_jobs} cells done")
    by_core = {}
    for r in per_cell:
        by_core.setdefault(r["pinned_to"], []).append(r["wall_s"])
    percore = {}
    for c, walls in by_core.items():
        ws = sorted(walls)
        percore[c] = {"median_s": statistics.median(ws),
                      "p10_s": _percentile(ws, 0.1),
                      "p90_s": _percentile(ws, 0.9),
                      "n": len(ws), "samples": ws}
    return percore, per_cell


def _representative_cores(topology):
    """One logical CPU per CCD (the first in each), V-Cache CCD(s) first, so the
    cache map compares each cache class once instead of every core."""
    ccds = topology.get("ccds", [])
    if not ccds:
        return [0]
    vids = set(topology.get("vcache_ccd_ids", []))
    ordered = sorted(ccds, key=lambda c: (c["id"] not in vids, c["id"]))
    out = [c["cpus"][0] for c in ordered if c["cpus"]]
    return out or [0]


def cache_working_set_bytes(N, max_deg):
    """Model of the dominant repeatedly-scanned footprint of an N-node cell:
    node_neighbors (N·max_deg·int32) + node_degrees (N·int32) + a few N-length
    float64 Lanczos vectors. This is the 'working set' the cache must hold to
    keep the d_s(t) probe's matvec fast — the x-axis the cache map is read
    against."""
    return N * (max_deg * 4 + 4 + 3 * 8)


def map_cache_hierarchy(k, T, lb, n_grid, cores=None, reps_per_n=8,
                        n_probes=60, lanczos_m=300, half_window=10, ec=-1.0,
                        max_deg=None, random_seeds=True, mu_table=None,
                        topology=None, warmup=True, log=lambda *a: None):
    """Pin to a fixed core and sweep the graph size N (the working set) to map
    the *effective* memory hierarchy this workload sees — independent of the
    datasheet. For each core in `cores` (default: one per CCD, so a V-Cache CCD
    is compared against a standard one) and each N in `n_grid`, run `reps_per_n`
    cells PINNED to that core and record per-cell build/probe time + peak RSS.

    Returns (cache_data, per_cell):
        cache_data[cpu] = [ {N, working_set_bytes, rss_bytes, build_s, probe_s,
                             wall_s, per_node_probe_s, n}, … ]  sorted by N.

    Reading it: when the working set outgrows a cache level (L2, then L3, then
    spills to DRAM) the per-node probe time steps UP; the step locations
    approximate that level's effective size AS THIS WORKLOAD EXPERIENCES IT.
    These are soft transitions, not the razor edges of a pointer-chase
    microbenchmark — the access pattern is an irregular graph traversal, the
    prefetchers help, and the smallest graph usually already exceeds L1 — so the
    map is most meaningful for the L2 → L3 → DRAM regimes (and on a 9950X3D the
    L3 step lands at a larger working set on the V-Cache CCD)."""
    if not hasattr(os, "sched_setaffinity"):
        log("cache mapping needs os.sched_setaffinity (Linux) — skipping.")
        return {}, []
    if mu_table is None:
        mu_table = load_mu_table()
    key = mu_key(k, T, lb)
    if key not in mu_table:
        mu, _, _ = calibrate_mu(k, T, lb, ec, verbose=False)
        mu_table[key] = mu
    if topology is None:
        topology = detect_topology()
    if max_deg is None:
        try:
            from core.project_constants import MAX_DEG as _MD
            max_deg = _MD
        except Exception:
            max_deg = 26
    mask = sorted(os.sched_getaffinity(0))
    cores = [c for c in (cores if cores is not None
                         else _representative_cores(topology)) if c in mask]
    if not cores:
        cores = [mask[0]] if mask else [0]
    n_grid = sorted({int(n) for n in n_grid if n and n > 0})
    if not n_grid:
        return {}, []
    if warmup:
        log("warmup cell (JIT + page warm) …")
        _one_cell_detailed((k, T, lb, n_grid[0], 0, mu_table, n_probes,
                            lanczos_m, half_window, cores[0]))
    seeds = _seeds(len(cores) * len(n_grid) * reps_per_n, random_seeds,
                   rng_seed=4242)
    jobs, si = [], 0
    for c in cores:
        for N in n_grid:
            for _r in range(reps_per_n):
                jobs.append((k, T, lb, N, seeds[si], mu_table,
                             n_probes, lanczos_m, half_window, c))
                si += 1
    # Shuffle so each (core, N) is sampled across the whole run — turbo/thermal
    # drift then averages out instead of biasing one core or one size.
    import random
    random.Random(99).shuffle(jobs)
    log(f"cache map: {len(cores)} core(s) × {len(n_grid)} sizes × {reps_per_n} "
        f"= {len(jobs)} cells, pinned, one at a time (randomised order).")
    per_cell, done = [], 0
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker) as ex:
        for f in as_completed([ex.submit(_one_cell_detailed, j) for j in jobs]):
            per_cell.append(f.result())
            done += 1
            if done % max(1, reps_per_n) == 0:
                log(f"  …{done}/{len(jobs)} cells")
    from collections import defaultdict
    groups = defaultdict(lambda: {"build": [], "probe": [], "wall": [],
                                  "rss": []})
    for r in per_cell:
        g = groups[(r["pinned_to"], r["N"])]
        g["build"].append(r["build_s"])
        g["probe"].append(r["probe_s"])
        g["wall"].append(r["wall_s"])
        if r.get("rss_peak_bytes"):
            g["rss"].append(r["rss_peak_bytes"])
    cache_data = defaultdict(list)
    for (cpu, N), g in groups.items():
        probe_med = statistics.median(g["probe"]) if g["probe"] else float("nan")
        cache_data[cpu].append({
            "N": N,
            "working_set_bytes": cache_working_set_bytes(N, max_deg),
            "rss_bytes": statistics.median(g["rss"]) if g["rss"] else None,
            "build_s": statistics.median(g["build"]) if g["build"] else float("nan"),
            "probe_s": probe_med,
            "wall_s": statistics.median(g["wall"]) if g["wall"] else float("nan"),
            "per_node_probe_s": (probe_med / N) if N else float("nan"),
            "n": len(g["probe"]),
        })
    for cpu in cache_data:
        cache_data[cpu].sort(key=lambda d: d["N"])
    return dict(cache_data), per_cell


def _mem_total_bytes():
    """Total RAM in bytes (/proc/meminfo MemTotal; psutil fallback)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _mem_available_bytes():
    """Allocatable RAM in bytes (/proc/meminfo MemAvailable; psutil fallback).
    None if undeterminable — callers then skip the memory gate. Same source the
    sweep's guard uses."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _mem_admit(n_inflight, next_need, avail, total, reserve, max_workers):
    """Memory-aware admission decision for the next cell. Returns:
      'stop'   — nothing in flight AND a single cell of this size can't fit in
                 (total − reserve): we've climbed as far as this machine allows.
      'wait'   — the pool is full, or launching now would drop free RAM below
                 the reserve (and something is in flight that will free it).
      'launch' — otherwise.
    Pure (no I/O) so the throttling logic is unit-testable with synthetic
    memory readings."""
    if n_inflight == 0:
        # always make progress with at least one worker; only refuse if a
        # single cell physically cannot fit under the reserve.
        if total is not None and next_need > max(0, total - reserve):
            return "stop"
        return "launch"
    if n_inflight >= max_workers:
        return "wait"
    if avail is not None and (avail - next_need) < reserve:
        return "wait"
    return "launch"


def _predict_cell_seconds(batches, N):
    """Predict median per-cell wall seconds at size N from completed batches
    [(N_i, cell_median_s_i), …] via a log-log least-squares power-law fit
    c ≈ a·N^b. Falls back to ∝N from a single point. Pure / unit-testable."""
    import math as _m
    pts = [(float(n), float(c)) for (n, c) in batches
           if n and n > 0 and c and c == c and c > 0]
    if not pts:
        return None
    if len(pts) == 1:
        n0, c0 = pts[0]
        return c0 * (N / n0)            # assume ∝ N with one anchor
    xs = [_m.log(n) for n, _ in pts]
    ys = [_m.log(c) for _, c in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = (sxy / sxx) if sxx > 0 else 1.0
    a = my - b * mx
    return _m.exp(a + b * _m.log(N))


def live_memory_benchmark(k, T, lb, n_grid, reserve_bytes=None, reps_per_n=0,
                          max_workers=None, bytes_per_node=None,
                          n_probes=60, lanczos_m=300, half_window=10, ec=-1.0,
                          random_seeds=True, mu_table=None, topology=None,
                          max_deg=None, pin_cores=None, on_update=None,
                          update_interval=4.0, warmup=True, log=lambda *a: None):
    """Live memory-pressure benchmark. Climbs the graph size N in ascending
    **batches** (one N at a time). Within a batch it launches `reps_per_n` cells
    (default: one per logical core), each pinned round-robin to a distinct core,
    running as many concurrently as fit while keeping `reserve_bytes` (default
    config min_free_gb, else 4 GiB) of RAM free — so the machine fills up at
    small N and the batch collapses to a single worker as N grows, finally
    stopping when even one cell no longer fits in (total − reserve).

    Because every cell is pinned, per-core / per-CCD timing is captured at every
    N — the V-Cache vs standard-CCD gap is visible from the first batch and
    re-measured as N grows. After each completed batch the per-cell time trend
    feeds an ETA for the rest of the current batch and the next one.

    Emits the full live dataset via `on_update(mem_bench_dict)` at most every
    `update_interval` seconds, where mem_bench_dict = {points, status, by_ccd,
    vcache_delta}. `status` carries the ETA fields. Returns the final dict.
    Ctrl-C stops the climb cleanly and keeps the partial map (running cells are
    drained, not killed)."""
    import math
    from collections import defaultdict
    from concurrent.futures import wait as _fwait, FIRST_COMPLETED
    if mu_table is None:
        mu_table = load_mu_table()
    key = mu_key(k, T, lb)
    if key not in mu_table:
        mu, _, _ = calibrate_mu(k, T, lb, ec, verbose=False)
        mu_table[key] = mu
    if topology is None:
        topology = detect_topology()
    if max_deg is None:
        try:
            from core.project_constants import MAX_DEG as _MD
            max_deg = _MD
        except Exception:
            max_deg = 26
    if reserve_bytes is None:
        try:
            from core.project_constants import MIN_FREE_GB as _MF
            reserve_bytes = int(float(_MF) * 1024 ** 3)
        except Exception:
            reserve_bytes = 4 * 1024 ** 3
    if max_workers is None:
        max_workers = (len(os.sched_getaffinity(0))
                       if hasattr(os, "sched_getaffinity")
                       else (os.cpu_count() or 1))
    if pin_cores is None:
        pin_cores = (sorted(os.sched_getaffinity(0))
                     if hasattr(os, "sched_getaffinity")
                     else list(range(os.cpu_count() or 1)))
    if not pin_cores:
        pin_cores = [0]
    batch_size = reps_per_n if reps_per_n and reps_per_n > 0 else len(pin_cores)
    safety = 1.30
    vids = set(topology.get("vcache_ccd_ids", []))

    def need_bytes(N):
        if bytes_per_node:
            return int(N * bytes_per_node * safety)
        return int(cache_working_set_bytes(N, max_deg) * safety)

    def ccd_of(cpu):
        for c in topology.get("ccds", []):
            if cpu in c["cpus"]:
                return c["id"]
        return None

    def _ts():
        return time.strftime("%H:%M:%S")

    n_grid = sorted({int(n) for n in n_grid if n and n > 0})
    seed_seq = iter(_seeds(batch_size * max(1, len(n_grid)), random_seeds,
                           rng_seed=20240601))
    total_mem = _mem_total_bytes()
    if warmup and n_grid:
        log("warmup cell (JIT + page warm) …")
        _one_cell_detailed((k, T, lb, n_grid[0], 0, mu_table, n_probes,
                            lanczos_m, half_window, pin_cores[0]))

    agg = {}                 # N -> aggregates incl. by_cpu (probe times)
    batches_done = []        # (N, cell_median_wall_s) for the ETA fit
    state = {"peak_used": 0, "max_conc": 0, "cells_done": 0, "stop_reason": None,
             "cur_N": None, "cur_done": 0, "cur_total": 0, "cur_conc": 1,
             "cur_cell_med": None, "cur_eta_s": None, "next_N": None,
             "next_cell_s": None, "next_eta_s": None}
    t_start = time.perf_counter()

    def _vcache_delta_at(N):
        a = agg.get(N)
        if not a:
            return None
        v, o = [], []
        for cpu, probes in a["by_cpu"].items():
            (v if ccd_of(cpu) in vids else o).extend(probes)
        if not v or not o:
            return None
        vm, om = statistics.median(v), statistics.median(o)
        return {"v": vm, "o": om, "speedup": (om / vm) if vm > 0 else float("nan"),
                "pct": ((om / vm) - 1) * 100 if vm > 0 else float("nan")}

    def _vcache_delta_series():
        out = []
        for N in sorted(agg):
            d = _vcache_delta_at(N)
            if d and d["speedup"] == d["speedup"]:
                out.append({"N": N,
                            "working_set_bytes": cache_working_set_bytes(N, max_deg),
                            "speedup": d["speedup"]})
        return out

    def _per_ccd_points():
        repr_cpu = {c["id"]: (c["cpus"][0] if c["cpus"] else None)
                    for c in topology.get("ccds", [])}
        out = defaultdict(list)
        for N in sorted(agg):
            byccd = defaultdict(list)
            for cpu, probes in agg[N]["by_cpu"].items():
                byccd[ccd_of(cpu)].extend(probes)
            for cid, probes in byccd.items():
                rc = repr_cpu.get(cid)
                if not probes or rc is None:
                    continue
                med = statistics.median(probes)
                out[rc].append({"N": N,
                                "working_set_bytes": cache_working_set_bytes(N, max_deg),
                                "per_node_probe_s": med / N, "probe_s": med})
        return dict(out)

    def _points():
        pts = []
        for N in sorted(agg):
            a = agg[N]
            if not a["probe"]:
                continue
            pm = statistics.median(a["probe"])
            pts.append({
                "N": N,
                "working_set_bytes": cache_working_set_bytes(N, max_deg),
                "rss_bytes": statistics.median(a["rss"]) if a["rss"] else None,
                "probe_s": pm,
                "wall_s": statistics.median(a["wall"]) if a["wall"] else float("nan"),
                "build_s": statistics.median(a["build"]) if a["build"] else float("nan"),
                "per_node_probe_s": (pm / N) if N else float("nan"),
                "concurrency": int(statistics.median(a["conc"])) if a["conc"] else 1,
                "n": len(a["probe"])})
        return pts

    def _update_eta(N):
        cc = max(1, state["cur_conc"])
        cell = state["cur_cell_med"] or _predict_cell_seconds(batches_done, N)
        remaining = max(0, state["cur_total"] - state["cur_done"])
        state["cur_eta_s"] = (math.ceil(remaining / cc) * cell) if cell else None
        idx = n_grid.index(N) if N in n_grid else -1
        nxt = n_grid[idx + 1] if 0 <= idx < len(n_grid) - 1 else None
        state["next_N"] = nxt
        if nxt is not None:
            hist = batches_done + ([(N, cell)] if cell else [])
            cs = _predict_cell_seconds(hist, nxt)
            state["next_cell_s"] = cs
            if cs and total_mem:
                kfit = max(1, min(max_workers,
                                  int((total_mem - reserve_bytes)
                                      // max(1, need_bytes(nxt)))))
                state["next_eta_s"] = math.ceil(batch_size / kfit) * cs
            else:
                state["next_eta_s"] = None
        else:
            state["next_cell_s"] = state["next_eta_s"] = None

    def _status(running):
        return {"running": running, "elapsed_s": time.perf_counter() - t_start,
                "cells_done": state["cells_done"],
                "peak_used_bytes": state["peak_used"],
                "mem_total_bytes": total_mem,
                "mem_available_bytes": _mem_available_bytes(),
                "reserve_bytes": reserve_bytes, "max_N": max(agg) if agg else 0,
                "max_concurrency": state["max_conc"], "workers_cap": max_workers,
                "batch_size": batch_size, "stop_reason": state["stop_reason"],
                "cur_N": state["cur_N"], "cur_done": state["cur_done"],
                "cur_total": state["cur_total"], "cur_eta_s": state["cur_eta_s"],
                "next_N": state["next_N"], "next_cell_s": state["next_cell_s"],
                "next_eta_s": state["next_eta_s"]}

    def _emit(running):
        if on_update:
            on_update({"points": _points(), "status": _status(running),
                       "by_ccd": _per_ccd_points(),
                       "vcache_delta": _vcache_delta_series()})

    ex = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker)
    last_update = [0.0]

    def _maybe_emit(force=False):
        now = time.perf_counter()
        if force or (now - last_update[0]) >= update_interval:
            _emit(True)
            last_update[0] = now

    _emit(True)
    try:
        for N in n_grid:
            need = need_bytes(N)
            if total_mem is not None and need > max(0, total_mem - reserve_bytes):
                state["stop_reason"] = (
                    f"a single N={N:,} cell needs ~{need/1024**3:.1f} GiB > "
                    f"RAM−reserve ({reserve_bytes/1024**3:.0f} GiB free kept); "
                    f"mapped up to N={(max(agg) if agg else 0):,} on this machine.")
                log(_ts() + " " + state["stop_reason"])
                break
            a = agg.setdefault(N, {"probe": [], "wall": [], "build": [],
                                   "rss": [], "conc": [],
                                   "by_cpu": defaultdict(list)})
            state.update(cur_N=N, cur_done=0, cur_total=batch_size,
                         cur_cell_med=None, cur_conc=1)
            _update_eta(N)
            ws = cache_working_set_bytes(N, max_deg)
            log(f"{_ts()} N={N:,} batch: {batch_size} cells, ws≈"
                f"{ws/1024**2:.1f} MiB/cell"
                + (f", next batch est ~{state['next_eta_s']:.0f}s"
                   if state['next_eta_s'] else ""))
            inflight, conc_at, submitted = {}, {}, 0
            while submitted < batch_size or inflight:
                while submitted < batch_size:
                    dec = _mem_admit(len(inflight), need, _mem_available_bytes(),
                                     total_mem, reserve_bytes, max_workers)
                    if dec != "launch":
                        break
                    core = pin_cores[submitted % len(pin_cores)]
                    fut = ex.submit(_one_cell_detailed,
                                    (k, T, lb, N, next(seed_seq), mu_table,
                                     n_probes, lanczos_m, half_window, core))
                    inflight[fut] = core
                    conc_at[fut] = len(inflight)
                    submitted += 1
                    state["max_conc"] = max(state["max_conc"], len(inflight))
                av = _mem_available_bytes()
                if total_mem and av is not None:
                    state["peak_used"] = max(state["peak_used"], total_mem - av)
                if not inflight:
                    if submitted >= batch_size:
                        break
                    continue
                state["cur_conc"] = len(inflight)
                done, _ = _fwait(list(inflight), timeout=1.0,
                                 return_when=FIRST_COMPLETED)
                for fut in done:
                    core = inflight.pop(fut)
                    try:
                        rec = fut.result()
                    except Exception as e:
                        log(f"{_ts()} cell N={N:,} cpu~{core} FAILED: {e}")
                        conc_at.pop(fut, None)
                        continue
                    cpu = rec.get("pinned_to", core)
                    a["probe"].append(rec["probe_s"])
                    a["wall"].append(rec["wall_s"])
                    a["build"].append(rec["build_s"])
                    if rec.get("rss_peak_bytes"):
                        a["rss"].append(rec["rss_peak_bytes"])
                    a["conc"].append(conc_at.pop(fut, 1))
                    a["by_cpu"][cpu].append(rec["probe_s"])
                    state["cells_done"] += 1
                    state["cur_done"] += 1
                    log(f"{_ts()} cell N={N:,} cpu={cpu} ccd={ccd_of(cpu)} "
                        f"build={rec['build_s']:.3f}s probe={rec['probe_s']:.3f}s "
                        f"wall={rec['wall_s']:.2f}s "
                        f"rss={(rec.get('rss_peak_bytes') or 0)/1024**2:.0f}MiB "
                        f"[{state['cur_done']}/{batch_size}]")
                if a["wall"]:
                    state["cur_cell_med"] = statistics.median(a["wall"])
                _update_eta(N)
                _maybe_emit()
            if a["wall"]:
                batches_done.append((N, statistics.median(a["wall"])))
            d = _vcache_delta_at(N)
            if d:
                log(f"{_ts()} N={N:,} per-CCD probe: V-Cache {d['v']:.3f}s vs "
                    f"others {d['o']:.3f}s → {d['speedup']:.2f}× "
                    f"({d['pct']:+.0f}%)")
            _maybe_emit(force=True)
    except KeyboardInterrupt:
        state["stop_reason"] = "interrupted by user (Ctrl-C) — partial map kept."
        log(_ts() + " " + state["stop_reason"])
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    _emit(False)
    return {"points": _points(), "status": _status(False),
            "by_ccd": _per_ccd_points(), "vcache_delta": _vcache_delta_series()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Throughput (cells/min) vs worker count, on one "
                    "representative cell, bypassing the sweep cache.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--lb", type=float, default=0.995)
    ap.add_argument("--N", type=int, default=64000,
                    help="cell size to benchmark. Bigger N leans more on "
                         "memory bandwidth / cache — try 256000 to surface a "
                         "bandwidth knee. Default 64000.")
    ap.add_argument("--workers", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32],
                    help="pool sizes to test, space-separated. Include 1 for "
                         "the per-core-efficiency baseline.")
    ap.add_argument("--reps", type=int, default=0,
                    help="cells per worker-count config (same work each time, "
                         "so rates compare). 0 = auto (3× the largest worker "
                         "count, clamped 12..96). Ignored if --reps-per-core "
                         "is set.")
    ap.add_argument("--reps-per-core", type=int, default=0,
                    help="cells per CORE per config: a W-worker test runs this "
                         "× W cells, so each core does the same number "
                         "regardless of W. 0 = use --reps instead.")
    ap.add_argument("--n-probes", type=int, default=60)
    ap.add_argument("--lanczos-m", type=int, default=300)
    ap.add_argument("--half-window", type=int, default=10)
    ap.add_argument("--ec", type=float, default=-1.0)
    ap.add_argument("--warmup", action="store_true",
                    help="run one throwaway cell first (numba JIT + page "
                         "warm) so the first timed config isn't penalised.")
    args = ap.parse_args(argv)

    ncpu = os.cpu_count() or 1
    aff = (len(os.sched_getaffinity(0))
           if hasattr(os, "sched_getaffinity") else ncpu)
    print(f"[bench] host  {ncpu} logical CPUs, {aff} in this affinity mask"
          + ("  (taskset-pinned — good for isolating a CCD)"
             if aff < ncpu else ""))
    print(f"[bench] probe n_probes={args.n_probes} lanczos_m={args.lanczos_m}"
          f" half_window={args.half_window}")

    results = run_benchmark(
        args.k, args.T, args.lb, args.N, args.workers, reps=args.reps,
        reps_per_core=args.reps_per_core,
        n_probes=args.n_probes, lanczos_m=args.lanczos_m,
        half_window=args.half_window, ec=args.ec, warmup=args.warmup,
        log=lambda m: print(f"[bench] {m}", flush=True))

    print()
    print(format_table(results))
    if results:
        best = max(results, key=lambda r: r["cells_per_min"])
        print(f"[bench] peak: {best['cells_per_min']:.1f} cells/min at "
              f"{best['workers']} workers = your optimum for this cell/N.")
        if best["workers"] < (aff // 2 + 1):
            print("[bench] peak is well below the core count → cache/"
                  "bandwidth-bound; fewer workers win. Try the V-cache CCD "
                  "via taskset.")
    print("[bench] note: optimum can shift with N — a cell that fits in L3 "
          "scales differently than one that spills to DRAM.")


if __name__ == "__main__":
    main()
