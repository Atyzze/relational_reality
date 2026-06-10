#!/usr/bin/env python3
"""
memory_benchmark.py
===================
Benchmarks TIME TO EQUILIBRIUM of the graph-growth engine across the cache->DRAM
hierarchy, comparing three back-ends that all implement the SAME Hamiltonian,
SAME Metropolis moves, and produce the SAME array construct (N x max_degree int32
neighbours + int32 degrees):

  py   - Numba engine  (engines/numba/)            the reference
  cpp  - C++ engine    (engines/native/engine_core.cpp)
  rust - Rust engine   (engines/native/engine_core.rs; identical RNG to cpp)

Each cell grows ONE graph from empty to its final equilibrium and times the whole
run - no warm/measure split and no mid-run polling (that would dilute the timing).
The only reporting cadence is per finished cell (the live HTML refreshes ~every 4s).

Sweep order: for each N, expand parallelism - every core individually (only for the
first full_percore_rounds sizes; after that just the fastest & slowest core), then
all disjoint pairs, 4s, 8s, ... up to all cores - running all back-ends at each step;
THEN grow N (x n_factor) and repeat. The concurrency ladder auto-truncates as the
per-worker footprint grows, down to one worker, stopping at the largest N that fits
in RAM-reserve (one worker, no swap). A running ETA projects time to that terminal N.

    python memory_benchmark.py
    python memory_benchmark.py --dry-run [--simulate-cores 32] [--simulate-ram-gb 92]
"""

import argparse
import json
import os
import statistics
import sys
import time
import datetime
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))   # .../src/benchmarks
SRC = os.path.dirname(HERE)                          # .../src
ROOT = os.path.dirname(SRC)                          # project root
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engines import get_engine, available_engines, ensure_built   # noqa: E402
from engines import build as _engbuild                            # noqa: E402

# benchmark track label -> engine backend, and the Hamiltonian knobs we forward
_BACKEND = {"py": "numba", "cpp": "cpp", "rust": "rust"}
_CFGKEY = {"py": "run_python", "cpp": "run_cpp", "rust": "run_rust"}
_HAM_KEYS = ("temperature", "degree_penalty", "edge_cost", "locality_bias")


def _params(eng):
    return {k: eng[k] for k in _HAM_KEYS if k in eng}


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
# config
DEFAULTS = {
    "engine": dict(max_degree=32, degree_penalty=0.05, temperature=0.004,
                   edge_cost=-1.0, locality_bias=0.99, cpp_mode=1, mean_degree=8),
    "work": dict(equilibrium_sweeps=0, reps_per_cell=1),
    "phases": dict(n_start=8, budget_s=0.5, warm_cap_s=45, solo_n_max=0,
                   smt_threshold=0.30, l3_threshold=0.10),
    "cpu": dict(physical_only=False, restrict_cores=[], vcache_first=True, pin=True),
    "memory": dict(reserve_gb=4.0, safety=1.30),
    "tracks": dict(run_python=True, run_cpp=True, run_rust=True, validate=True),
    "output": dict(dir="output/mem_bench", csv="memory_benchmark.csv",
                   json="memory_benchmark.json", html="memory_benchmark.html",
                   title="Graph-growth engine - time-to-equilibrium benchmark (Numba vs C++ vs Rust)"),
}


def load_config(path):
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            user = tomllib.load(f)
        for sect, vals in user.items():
            cfg.setdefault(sect, {}).update(vals)
    return cfg


# topology
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _parse_cpu_list(s):
    out = []
    for part in (s or "").split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def detect_topology():
    base = "/sys/devices/system/cpu"
    mask = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or 1))
    model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if "model name" in line:
            model = line.split(":", 1)[1].strip(); break
    cpus = {}
    for c in mask:
        d = f"{base}/cpu{c}"
        l3 = _read(f"{d}/cache/index3/size")
        l3g = _parse_cpu_list(_read(f"{d}/cache/index3/shared_cpu_list")) or [c]
        core_id = _read(f"{d}/topology/core_id")
        cpus[c] = dict(l3=l3, l3_group=sorted(g for g in l3g if g in mask) or [c],
                       core_id=int(core_id) if core_id is not None else c)
    groups, seen = [], {}
    for c in mask:
        key = tuple(cpus[c]["l3_group"])
        if key not in seen:
            seen[key] = len(groups); groups.append(dict(id=len(groups), cpus=list(key)))

    def l3_bytes(g):
        s = cpus[g["cpus"][0]]["l3"]
        if not s:
            return 0
        s = s.strip(); mult = 1
        if s[-1] in "KkMmGg":
            mult = {"K": 1024, "M": 1024**2, "G": 1024**3}[s[-1].upper()]; s = s[:-1]
        try:
            return int(float(s) * mult)
        except Exception:
            return 0
    for g in groups:
        g["l3_bytes"] = l3_bytes(g)
    maxl3 = max((g["l3_bytes"] for g in groups), default=0)
    for g in groups:
        g["is_vcache"] = bool(maxl3 and g["l3_bytes"] == maxl3 and len(groups) > 1)
    phys = {}
    for c in mask:
        phys.setdefault(cpus[c]["core_id"], c)
    return dict(model=model, mask=mask, groups=groups,
                physical=sorted(phys.values()), n_logical=len(mask))


def ordered_cores(topo, cfg):
    pool = topo["physical"] if cfg["cpu"]["physical_only"] else topo["mask"]
    rc = cfg["cpu"]["restrict_cores"]
    if rc:
        pool = [c for c in pool if c in set(rc)]
    if cfg["cpu"]["vcache_first"] and topo["groups"]:
        seq = []
        for g in sorted(topo["groups"], key=lambda g: (not g["is_vcache"], g["id"])):
            seq.extend([c for c in g["cpus"] if c in pool])
        seq.extend([c for c in pool if c not in seq])
        return seq
    return list(pool)


def ccd_of(topo, cpu):
    for g in topo["groups"]:
        if cpu in g["cpus"]:
            return g["id"]
    return None


# memory
def _meminfo(key):
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith(key):
            return int(line.split()[1]) * 1024
    return 0


def mem_total():
    return _meminfo("MemTotal:")


def mem_available():
    return _meminfo("MemAvailable:")


def per_worker_bytes(N, md, base_rss, safety):
    """Per-worker memory ceiling used to size the concurrency ladder. The data
    arrays (N x max_degree int32 neighbours + N int32 degrees) are PRIVATE to each
    worker and carry the safety margin; base_rss (interpreter + JIT image) is largely
    copy-on-write *shared* across the fork pool, so it is added once per worker
    WITHOUT the array safety factor. Counting it per-worker at all is the conservative
    choice — it slightly overstates the marginal cost on Linux, so the taper point is
    predicted a touch early rather than late (i.e. errs away from OOM)."""
    arrays = N * (md * 4 + 4)
    return int(arrays * safety) + base_rss


# ─────────────────── engine-backed calibration / equilibrium ───────────────────
# These delegate to engines.build, so the benchmark and the probes share ONE
# implementation through the engine interface. Signatures kept stable so run()
# is unchanged.
def calibrate_degree_penalty(eng, target_k, prefer, sweeps, log=print):
    cls = get_engine(_BACKEND.get(prefer, "numba"))
    return _engbuild.calibrate_degree_penalty(cls, target_k, _params(eng),
                                              n=30000, sweeps=sweeps,
                                              max_degree=eng["max_degree"], log=log)


def detect_equilibrium_sweeps(eng, prefer, log=print):
    cls = get_engine(_BACKEND.get(prefer, "numba"))
    return _engbuild.detect_equilibrium_sweeps(cls, _params(eng),
                                               max_degree=eng["max_degree"], log=log)
# (the schedule lives in benchmarks.topo_bench now: phase-structured —
#  solo per-core map → MEASURED grouping → packed/spread scaling →
#  language compare — instead of the old flat N×group×track ladder)

# workers
def _pin(core):
    """Pin the CURRENT process to one logical CPU. Attempted
    unconditionally: pool workers are REUSED across cells, so after the
    first pin the process's affinity mask has shrunk to {prev_core} — the
    old `if core in sched_getaffinity(0)` guard then silently no-opped
    every later pin and EVERY solo cell ran on the first core (one busy
    core in btop, identical rows in the per-core map). Pinning to a core
    outside the cgroup/cpuset raises OSError, which is caught — that was
    the only thing the guard protected against."""
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
        except OSError:
            pass


def _rss():
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _cpu_now():
    try:
        return os.sched_getcpu()
    except Exception:
        return None


def worker(job):
    """Grow ONE graph from empty to equilibrium and time it as a single block,
    through the engine interface (no warm/measure split, no mid-run sampling)."""
    (track, N, steps, seed, core, eng, mode, barrier) = job
    _pin(core)
    cls = get_engine(_BACKEND[track])
    params = _params(eng)
    rss0 = _rss()
    try:
        e = cls(N, max_degree=eng["max_degree"], seed=seed, mode=mode, **params)
    except Exception as ex:
        return dict(track=track, N=N, core=core, error=f"{type(ex).__name__}: {ex}")
    if barrier is not None:
        try:
            barrier.wait(timeout=600)
        except Exception:
            pass
    t0 = time.perf_counter()
    try:
        e.step(steps)
    except RuntimeError:
        e.close()
        return dict(track=track, N=N, core=core, error="capped")
    dt = time.perf_counter() - t0
    rss1 = _rss()
    st = e.stats()
    e.close()
    edges, peak = st["edges"], st["peak_degree"]
    return dict(track=track, N=N, core=core, cpu=_cpu_now(), seconds=dt, steps=steps,
                ns_step=dt * 1e9 / steps, m_steps_s=steps / dt / 1e6,
                edges=edges, peak=peak, avg_deg=2 * edges / N,
                rss_bytes=max(rss0, rss1, _rss()))


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
# verification
def verify_tracks(cfg, eng, eq_sweeps, log=print):
    """Confirm the back-ends agree through the interface: cpp & rust bit-identical,
    both match Numba's degree distribution, and the Graph construct is identical."""
    import numpy as np
    N = 4000; sweeps = max(50, eq_sweeps); params = _params(eng)
    log("[verify] same parameters for every track: max_degree=%d %s"
        % (eng["max_degree"], " ".join(f"{k}={params[k]}" for k in _HAM_KEYS)))
    avail = available_engines()
    snaps = {}
    for label, backend in _BACKEND.items():
        if cfg["tracks"].get(_CFGKEY[label]) and backend in avail:
            e = get_engine(backend)(N, max_degree=eng["max_degree"], seed=12345,
                                    mode=eng["cpp_mode"], **params)
            e.sweep(sweeps)
            snaps[label] = (e.snapshot(), e.stats())
            e.close()
    if "cpp" in snaps and "rust" in snaps:
        a, b = snaps["cpp"][0], snaps["rust"][0]
        same = np.array_equal(a.neighbors, b.neighbors) and np.array_equal(a.degrees, b.degrees)
        log(f"[verify] cpp vs rust (same seed): {'BIT-IDENTICAL graph' if same else 'DIFFER'}")
    ref = "cpp" if "cpp" in snaps else ("rust" if "rust" in snaps else None)
    if "py" in snaps and ref:
        pg, ps = snaps["py"]; rg, rs = snaps[ref]
        log(f"[verify] py vs {ref}: avg deg {ps['k_avg']:.2f} vs {rs['k_avg']:.2f} "
            f"(delta {abs(ps['k_avg']-rs['k_avg']):.2f}); array {pg.neighbors.shape}/"
            f"{pg.neighbors.dtype} matches {rg.neighbors.shape}/{rg.neighbors.dtype}")
    pick = snaps.get(ref) or snaps.get("py")
    if pick:
        g = pick[0]; nb, dg = g.neighbors, g.degrees; ok_sym = ok_deg = True
        for u in range(0, N, max(1, N // 500)):
            d = dg[u]
            if int((nb[u] != -1).sum()) != d:
                ok_deg = False
            for i in range(d):
                v = nb[u, i]
                if v < 0 or u not in nb[v][:dg[v]]:
                    ok_sym = False; break
        log(f"[verify] graph is a valid simple graph: symmetric={ok_sym}, "
            f"degrees match rows={ok_deg}")
# formatting
def fmt_bytes(b):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024 or u == "TiB":
            return f"{b:.0f} {u}" if u == "B" else f"{b:.1f} {u}"
        b /= 1024


def fmt_int(n):
    return f"{n:,}"


def fmt_dur(s):
    s = int(max(0, s))
    if s < 90:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 90:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


# engines: ensure built / track selection / baseline RSS
def ensure_natives(cfg):
    want = [b for label, b in _BACKEND.items() if cfg["tracks"].get(_CFGKEY[label])]
    status = ensure_built(want)
    for label, backend in _BACKEND.items():
        if cfg["tracks"].get(_CFGKEY[label]) and not status.get(backend, False):
            print(f"[warn] {label} engine ({backend}) unavailable - disabling that track.")
            cfg["tracks"][_CFGKEY[label]] = False


def tracks_for(cfg):
    return [label for label in ("py", "cpp", "rust") if cfg["tracks"].get(_CFGKEY[label])]


def warm_base_rss(cfg, eng):
    bases = []
    for label in tracks_for(cfg):
        r = worker((label, 2000, 6000, 1, None, eng, eng["cpp_mode"], None))
        bases.append(r.get("rss_bytes", 200 * 1024**2))
    return max(bases) if bases else 64 * 1024**2
# run
def _ts():
    """Local ISO-8601 timestamp, seconds precision, with UTC offset — e.g.
    2026-06-09T01:14:31+02:00 — prefixed to every progress line."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _cell_key(N, conc, track, group):
    """Stable identity of a benchmark cell, for resume de-duplication."""
    if isinstance(group, (list, tuple)):
        g = tuple(int(x) for x in group)
    elif group is None:
        g = ()
    else:
        g = (group,)
    return (int(N), int(conc), str(track), g)


def _load_prior_results(cfg):
    """Resume support: read the records from a previous run's JSON so this run
    can skip cells already on disk and append to them instead of overwriting.
    Returns [] when there is nothing to resume. Deleting the output directory
    is the intended way to reset."""
    path = os.path.join(ROOT, cfg["output"]["dir"], cfg["output"]["json"])
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        recs = data.get("results", []) if isinstance(data, dict) else []
        for r in recs:
            g = r.get("group")
            if isinstance(g, list):
                try:
                    r["group"] = [int(x) for x in g]
                except Exception:
                    pass
        return recs
    except Exception as ex:
        print(f"[warn] could not read prior results ({ex}); starting fresh.")
        return []


# CPU package power via RAPL (best-effort; energy_uj is often root-only since PLATYPUS)
_RAPL_PATHS = None


def _rapl_paths():
    global _RAPL_PATHS
    if _RAPL_PATHS is not None:
        return _RAPL_PATHS
    import glob
    found = []
    for d in sorted(glob.glob("/sys/class/powercap/*")):
        nm_f = os.path.join(d, "name"); ej = os.path.join(d, "energy_uj")
        if not (os.path.exists(nm_f) and os.path.exists(ej)):
            continue
        try:
            nm = open(nm_f).read().strip()
        except Exception:
            continue
        if nm.startswith("package") or nm == "psys":
            try:
                mr = int(open(os.path.join(d, "max_energy_range_uj")).read().strip())
            except Exception:
                mr = 0
            found.append((ej, mr))
    _RAPL_PATHS = found
    return found


def _rapl_read():
    paths = _rapl_paths()
    if not paths:
        return None
    out = []
    for ej, _mr in paths:
        try:
            out.append(float(open(ej).read().strip()))
        except Exception:
            return None
    return out


def _watts(r0, r1, secs):
    """Average package watts over [r0, r1] readings spanning `secs`, handling
    per-domain counter wraparound. None if RAPL was unreadable."""
    if not r0 or not r1 or secs <= 0 or len(r0) != len(r1):
        return None
    total_uj = 0.0
    for (ej, mr), a, b in zip(_rapl_paths(), r0, r1):
        d = b - a
        if d < 0 and mr > 0:
            d += mr
        if d < 0:
            return None
        total_uj += d
    return total_uj / 1e6 / secs


def run(cfg, dry=False, sim_cores=0, sim_ram_gb=0.0, open_browser=True):
    """Entry point kept stable for main.py bench; the phase-structured run
    itself lives in benchmarks.topo_bench (solo per-core map → measured
    grouping → packed/spread scaling → language compare)."""
    from benchmarks import topo_bench
    return topo_bench.run(cfg, dry=dry, sim_cores=sim_cores,
                          sim_ram_gb=sim_ram_gb, open_browser=open_browser)


# reporting
def write_outputs(cfg, meta, results, quiet=False):
    outdir = os.path.join(ROOT, cfg["output"]["dir"])
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, cfg["output"]["csv"])
    cols = ["N", "working_set", "conc", "track", "group", "ccds", "seconds_med", "ns_step_med",
            "ns_step_min", "ns_step_max", "per_worker_m_steps_s", "aggregate_m_steps_s",
            "aggregate_peak_m_steps_s",
            "avg_deg", "peak", "rss_per_worker", "watts"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")).replace(",", " ") for c in cols) + "\n")
    with open(os.path.join(outdir, cfg["output"]["json"]), "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=1, default=str)
    html_path = os.path.join(outdir, cfg["output"]["html"])
    tmp = html_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(render_html(cfg, meta, results))
    os.replace(tmp, html_path)
    if not quiet:
        print(f"\nwrote:\n  {csv_path}\n  {html_path}")


def _ccd_of_core(meta, core):
    for g in meta.get("groups", []):
        if core in g.get("cpus", []):
            return g["id"], bool(g.get("is_vcache"))
    return 0, False


TCOL = {"py": "#e0633a", "cpp": "#4aa3df", "rust": "#c6923b"}


def _svg_curve(results, meta):
    import math as _m
    agg, isv = {}, {}
    for r in results:
        if r["conc"] != 1 or not r.get("group"):
            continue
        cid, v = _ccd_of_core(meta, r["group"][0]); isv[cid] = v
        agg.setdefault((r["track"], cid), {}).setdefault(r["working_set"], []).append(r["ns_step_med"])
    series = {k: sorted((ws, statistics.median(vs)) for ws, vs in m.items()) for k, m in agg.items()}
    if not series:
        return "<p style='color:#8a93a0'>no concurrency-1 points yet...</p>"
    xs = [x for s in series.values() for x, _ in s]; ys = [y for s in series.values() for _, y in s]
    xlo, xhi = min(xs), max(xs); ylo, yhi = min(ys) * 0.8, max(ys) * 1.15
    W, H, pad = 780, 340, 54
    lx = lambda x: pad + (_m.log10(x) - _m.log10(xlo)) / max(1e-9, _m.log10(xhi) - _m.log10(xlo)) * (W - 2 * pad)
    ly = lambda y: H - pad - (_m.log10(y) - _m.log10(ylo)) / max(1e-9, _m.log10(yhi) - _m.log10(ylo)) * (H - 2 * pad)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:820px">']
    gx = xlo
    while gx <= xhi * 1.001:
        x = lx(gx)
        out.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{H-pad}" stroke="#222a33"/>')
        out.append(f'<text x="{x:.1f}" y="{H-pad+15}" fill="#8a93a0" font-size="10" text-anchor="middle" font-family="monospace">{fmt_bytes(gx)}</text>')
        gx *= 8
    for fy in (ylo, (ylo * yhi) ** 0.5, yhi):
        y = ly(fy)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{W-pad}" y2="{y:.1f}" stroke="#222a33"/>')
        out.append(f'<text x="{pad-6}" y="{y+3:.1f}" fill="#8a93a0" font-size="10" text-anchor="end" font-family="monospace">{fy:.0f}</text>')
    for (trk, cid), pts in sorted(series.items()):
        col = TCOL.get(trk, "#9aa"); dash = "" if isv.get(cid) else ' stroke-dasharray="5 4"'
        d = " ".join(f"{lx(x):.1f},{ly(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for x, y in pts:
            fill = col if isv.get(cid) else "#0b0e13"
            out.append(f'<circle cx="{lx(x):.1f}" cy="{ly(y):.1f}" r="3" fill="{fill}" stroke="{col}"/>')
    out.append(f'<text x="{W/2:.0f}" y="{H-3}" fill="#8a93a0" font-size="11" text-anchor="middle" font-family="monospace">working set (log) - ns/step (log), 1 worker</text>')
    lx0 = pad + 8; yy = pad + 6
    for trk in ("py", "cpp", "rust"):
        if any(t == trk for t, _ in series):
            out.append(f'<line x1="{lx0}" y1="{yy}" x2="{lx0+18}" y2="{yy}" stroke="{TCOL[trk]}" stroke-width="2"/>')
            out.append(f'<text x="{lx0+24}" y="{yy+3}" fill="#cfd6df" font-size="11" font-family="monospace">{trk}</text>'); yy += 15
    if len({cid for _, cid in series}) > 1:
        out.append(f'<line x1="{lx0}" y1="{yy}" x2="{lx0+18}" y2="{yy}" stroke="#8a93a0" stroke-width="2"/><text x="{lx0+24}" y="{yy+3}" fill="#8a93a0" font-size="11" font-family="monospace">- V-Cache</text>'); yy += 15
        out.append(f'<line x1="{lx0}" y1="{yy}" x2="{lx0+18}" y2="{yy}" stroke="#8a93a0" stroke-width="2" stroke-dasharray="5 4"/><text x="{lx0+24}" y="{yy+3}" fill="#8a93a0" font-size="11" font-family="monospace">-- other CCD</text>')
    out.append("</svg>")
    return "".join(out)



_BYTES_PER_STEP_EST = 320   # ~5 cache lines touched per Metropolis step at large N
                            # (random access); rough, only used for a GB/s estimate.


def _gbps(m_steps_s):
    """Effective-bandwidth ESTIMATE from throughput: (M steps/s) × bytes/step.

    Each accepted/rejected Metropolis move chases a handful of random pointers
    (neighbour lists, degree counters), so it touches ~_BYTES_PER_STEP_EST bytes
    of mostly-uncached memory. Multiplying throughput by that gives a
    bandwidth-equivalent figure that is directly comparable to a DIMM spec —
    which is what makes it more legible than raw M steps/s. It is NOT measured
    DRAM traffic: the per-step byte count is a fixed guess, and random access is
    latency-bound, so the true streaming bandwidth of the parts is higher. Treat
    it as an order-of-magnitude "how hard is this leaning on memory" number, and
    keep M steps/s as the ground truth it is derived from.
    """
    return (m_steps_s or 0.0) * 1e6 * _BYTES_PER_STEP_EST / 1e9



def _open_report(cfg):
    # Auto-open the live HTML report in a browser (best-effort; no-op when headless).
    try:
        html = os.path.abspath(os.path.join(ROOT, cfg["output"]["dir"], cfg["output"]["html"]))
        webbrowser.open("file://" + html)
        print(f"[browser] opened {html}")
    except Exception:
        pass


def _primary_track(results):
    for t in ("cpp", "rust", "py"):
        if any(r.get("track") == t for r in results):
            return t
    return None


def _agg_ladder(results, track):
    # {N: {concurrency: median TRUE total throughput (M steps/s) with that many workers}}
    # (true = c x steps / slowest-worker time; see the cell record. Not median x c.)
    tmp = {}
    for r in results:
        if r.get("track") != track:
            continue
        tmp.setdefault(r["N"], {}).setdefault(r["conc"], []).append(r["aggregate_m_steps_s"])
    return {N: {c: statistics.median(v) for c, v in cs.items()} for N, cs in tmp.items()}


def _scaling(results, meta):
    track = _primary_track(results)
    if not track:
        return ""
    lad = _agg_ladder(results, track)
    Ns = sorted(lad)
    cset = sorted({c for cs in lad.values() for c in cs})
    if len(cset) < 2:
        return ""                       # single-core machine: nothing to scale
    pairs = list(zip(cset, cset[1:]))
    top, prev = cset[-1], cset[-2]
    ws_by_N = {r["N"]: r["working_set"] for r in results}

    bigN = Ns[-1]; big = lad[bigN]
    bestc = max(big, key=big.get); agg_best = big[bestc]
    smt = (big[top] / big[prev] - 1.0) if (top in big and prev in big) else None
    gbps = _gbps(agg_best)
    head = (f'<p style="color:#cfd6df;font-size:13px">Track <b>{track}</b>. At the largest size '
            f'N={fmt_int(bigN)} (DRAM-resident), best TOTAL throughput is at <b>{bestc} workers</b> '
            f'= {agg_best:.0f} M steps/s')
    if smt is not None:
        head += (f'. The final doubling {prev}\u2192{top} is the SMT step if {top} exceeds your '
                 f'physical core count, so hyper-threading {"adds" if smt > 0 else "costs"} '
                 f'<b>{smt*100:+.1f}%</b> total throughput here')
    head += (f'. Effective DRAM bandwidth \u2248 <b>{gbps:.0f} GB/s</b> (est. at ~{_BYTES_PER_STEP_EST} '
             f'B/step random access) \u2014 well under the DIMMs\' streaming peak (~64\u201396 GB/s for '
             f'dual-channel DDR5), because random pointer-chasing is latency-bound, not bandwidth-bound.</p>')

    hdr = "".join(f"<th>{a}\u2192{b}{' (SMT)' if b == top else ''}</th>" for a, b in pairs)
    rows = [f'<table><thead><tr><th>N</th><th>working set</th>{hdr}'
            f'<th>best c</th><th>speedup@best</th></tr></thead><tbody>']
    for N in Ns:
        d = lad[N]; cells = ""
        for a, b in pairs:
            if a in d and b in d and d[a] > 0:
                g = d[b] / d[a] - 1.0
                col = "#3fb950" if g > 0.02 else ("#e5534b" if g < -0.02 else "#8a93a0")
                cells += f'<td style="color:{col}">{g*100:+.0f}%</td>'
            else:
                cells += "<td>-</td>"
        bc = max(d, key=d.get)
        sp = (d[bc] / d[cset[0]]) if (cset[0] in d and d[cset[0]] > 0) else float("nan")
        rows.append(f'<tr><td>{fmt_int(N)}</td><td>{fmt_bytes(ws_by_N.get(N, 0))}</td>{cells}'
                    f'<td>{bc}</td><td>{sp:.1f}x</td></tr>')
    rows.append("</tbody></table>")
    return ('<h2 style="color:#e6edf3;font-size:16px">Concurrency scaling \u2014 total throughput gain '
            'per core-doubling</h2>' + head + "".join(rows) +
            '<div class="note">Each cell is the % change in TOTAL throughput when doubling the worker count '
            'at that N (green = real gain, red = slower from contention past the memory ceiling). Near-linear '
            'at small N (cache-resident, no shared bottleneck); it flattens as N grows and the shared DDR5 '
            'channels saturate. The final column is the SMT step (threads beyond physical cores), so a positive '
            'value there means hyper-threading raises total throughput by hiding memory latency.</div>')


def _scaling_console(results, meta):
    track = _primary_track(results)
    if not track:
        return
    lad = _agg_ladder(results, track)
    Ns = sorted(lad)
    cset = sorted({c for cs in lad.values() for c in cs})
    if not Ns or len(cset) < 2:
        return
    print(f"\nConcurrency scaling ({track} track) \u2014 total M steps/s and per-doubling gain:")
    picks = sorted(set([Ns[0], Ns[len(Ns) // 2], Ns[-1]]))
    for N in picks:
        d = lad[N]
        seq = " ".join(
            f"{c}c={d[c]:.0f}" + (f"({(d[c]/d[p]-1)*100:+.0f}%)" if (p is not None and p in d and d[p] > 0) else "")
            for p, c in zip([None] + cset, cset) if c in d)
        print(f"  N={fmt_int(N):>12}: {seq}   best={max(d, key=d.get)}c")
    big = lad[Ns[-1]]
    if cset[-1] in big and cset[-2] in big:
        smt = big[cset[-1]] / big[cset[-2]] - 1
        print(f"  SMT step ({cset[-2]}\u2192{cset[-1]} threads): {smt*100:+.1f}% total throughput "
              f"({'SMT helps' if smt > 0 else 'SMT hurts'})")


def render_html(cfg, meta, results):
    title = cfg["output"]["title"]
    trks = meta["tracks"]
    by_N = {}
    for r in results:
        by_N.setdefault(r["N"], []).append(r)

    def med_c1(rs, track):
        v = [x["ns_step_med"] for x in rs if x["conc"] == 1 and x["track"] == track]
        return statistics.median(v) if v else None

    def cell(v, f="{}"):
        return f.format(v) if v is not None else "-"

    head = "".join(f"<th>{t} ns/step</th>" for t in trks)
    spd = "".join(f"<th>{t} x</th>" for t in trks if t != "py")
    rows = [f'<table><thead><tr><th>N</th><th>working set</th>{head}{spd}'
            f'<th>max conc</th><th>best aggregate (M steps/s)</th><th>GB/s (est)</th><th>peak W</th><th>avg deg</th></tr></thead><tbody>']
    for N in sorted(by_N):
        rs = by_N[N]; ws = rs[0]["working_set"]
        nsv = {t: med_c1(rs, t) for t in trks}
        py = nsv.get("py")
        tds = "".join(f"<td>{cell(nsv[t], '{:.1f}')}</td>" for t in trks)
        sds = "".join(f"<td>{cell((py / nsv[t]) if (py and nsv.get(t)) else None, '{:.2f}x')}</td>"
                      for t in trks if t != "py")
        maxc = max(x["conc"] for x in rs)
        agg = max((x["aggregate_m_steps_s"] for x in rs), default=0)
        avgd = statistics.median([x["avg_deg"] for x in rs])
        wN = [x["watts"] for x in rs if x.get("watts")]
        wcell = f"{max(wN):.0f} W" if wN else "-"
        rows.append(f"<tr><td>{fmt_int(N)}</td><td>{fmt_bytes(ws)}</td>{tds}{sds}"
                    f"<td>{maxc}</td><td>{agg:.1f}</td><td>{_gbps(agg):.1f}</td><td>{wcell}</td><td>{avgd:.1f}</td></tr>")
    rows.append("</tbody></table>")

    ccd_ids = [g["id"] for g in meta["groups"]]
    vset = {g["id"] for g in meta["groups"] if g.get("is_vcache")}
    ptable = ""
    if len(ccd_ids) > 1:
        hdr = "".join(f"<th>{t} CCD{cid}{'(V$)' if cid in vset else ''}</th>" for t in trks for cid in ccd_ids)
        gain = "".join(f"<th>{t} V-Cache gain</th>" for t in trks)
        pr = [f'<table><thead><tr><th>N</th><th>working set</th>{hdr}{gain}</tr></thead><tbody>']
        for N in sorted(by_N):
            rs = [x for x in by_N[N] if x["conc"] == 1 and x.get("group")]
            ws = by_N[N][0]["working_set"]; tds = ""; gds = ""
            for t in trks:
                permed = {}
                for cid in ccd_ids:
                    vals = [x["ns_step_med"] for x in rs if x["track"] == t and _ccd_of_core(meta, x["group"][0])[0] == cid]
                    permed[cid] = statistics.median(vals) if vals else None
                    tds += f"<td>{cell(permed[cid], '{:.1f}')}</td>"
                vc = [permed[c] for c in ccd_ids if c in vset and permed[c]]
                ot = [permed[c] for c in ccd_ids if c not in vset and permed[c]]
                gds += f"<td>{cell((max(ot)/min(vc)) if vc and ot else None, '{:.2f}x')}</td>"
            pr.append(f"<tr><td>{fmt_int(N)}</td><td>{fmt_bytes(ws)}</td>{tds}{gds}</tr>")
        pr.append("</tbody></table>")
        ptable = ('<h2 style="color:#e6edf3;font-size:16px">Per-cache-class, 1 worker '
                  '(V-Cache CCD vs the rest - the bigger L3 should pull ahead as N grows)</h2>'
                  + "".join(pr))

    groups = ", ".join(f"CCD{g['id']}{'(V-Cache)' if g['is_vcache'] else ''}:{fmt_bytes(g['l3_bytes'])}/{len(g['cpus'])}c"
                       for g in meta["groups"])
    p = meta.get("params", {})
    running = meta.get("running", False); done = meta.get("cells_done", len(results))
    tot = meta.get("cells_total", len(results)); pct = (100.0 * done / tot) if tot else 100.0
    eta = meta.get("eta_s")
    refresh = '<meta http-equiv="refresh" content="4">' if running else ''
    if running:
        status = (f'<span style="color:#e3b341">running</span> - {done}/{tot} cells ({pct:.0f}%) - '
                  f'elapsed {fmt_dur(meta["elapsed_s"])}' + (f' - ETA {fmt_dur(eta)}' if eta else '') + ' - auto-refreshing')
    else:
        status = f'<span style="color:#3fb950">complete</span> - {done}/{tot} cells - {fmt_dur(meta["elapsed_s"])}'
    bar = (f'<div style="height:6px;background:#222a33;border-radius:3px;margin:10px 0 18px;overflow:hidden">'
           f'<div style="height:100%;width:{pct:.1f}%;background:{"#e3b341" if running else "#3fb950"}"></div></div>')
    pdp = p.get("degree_penalty")
    pdp = f"{pdp:.4f}" if isinstance(pdp, (int, float)) else "-"
    _mile = meta.get("milestones") or {}
    mileline = ""
    if _mile:
        _band = _mile.get("t_total_hi", _mile["t_total"]) - _mile["t_total"]
        _b = [f"remaining ~{fmt_dur(_mile['t_total'])} (\u00b1{fmt_dur(_band)})"]
        if _mile.get("drop_N") is None:
            _b.append(f"max {_mile.get('max_conc', '?')} workers fit throughout")
        elif _mile.get("t_to_drop"):
            _b.append(f"workers taper below {_mile.get('max_conc', '?')} to {_mile.get('fill_c', '?')} at "
                      f"N={fmt_int(_mile['drop_N'])} (~{fmt_dur(_mile['t_to_drop'])})")
        else:
            _b.append("workers tapering (RAM-limited)")
        _b.append(f"final N={fmt_int(_mile['final_N'])} cell ~{fmt_dur(_mile['final_secs'])}")
        mileline = f'<div class="status" style="color:#8a93a0">{" \u00b7 ".join(_b)}</div>'
    return f"""<!doctype html><meta charset="utf-8">{refresh}<title>{title}</title>
<style>
 body{{background:#0e1116;color:#cfd6df;font-family:ui-monospace,Menlo,Consolas,monospace;margin:24px;line-height:1.5}}
 h1{{color:#e6edf3;font-size:20px;font-weight:600}} h2{{margin-top:26px}}
 .sub{{color:#8a93a0;font-size:13px;margin-bottom:6px}} .status{{font-size:13px;margin:2px 0}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0 20px}}
 .card{{background:#161b22;border:1px solid #222a33;border-radius:8px;padding:12px 16px;min-width:140px}}
 .card .v{{font-size:22px;color:#e6edf3}} .card .l{{font-size:12px;color:#8a93a0}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:right;padding:6px 9px;border-bottom:1px solid #222a33}}
 th{{color:#8a93a0;font-weight:500}} td:first-child,th:first-child{{text-align:left}}
 svg{{background:#0b0e13;border:1px solid #222a33;border-radius:8px;margin:8px 0}}
 .note{{color:#8a93a0;font-size:12px;margin-top:18px}}
</style>
<h1>{title}</h1>
<div class="sub">{meta.get('model')} - {meta['n_logical']} logical - {groups}<br>
RAM total {fmt_bytes(meta['total_ram'])} - available {fmt_bytes(meta['available'])} - reserve {meta['reserve_gb']} GiB - tracks {', '.join(trks)}<br>
params: k={p.get('mean_degree')} - mu={pdp} - T={p.get('temperature')} - lb={p.get('locality_bias')} - max_degree={p.get('max_degree')} - equilibrium {p.get('equilibrium_sweeps')} sweeps/cell</div>
<div class="status">{status}</div>
{mileline}
{bar}
<div class="cards">
 <div class="card"><div class="v">{fmt_int(max(by_N) if by_N else 0)}</div><div class="l">largest N so far</div></div>
 <div class="card"><div class="v">{max((x['conc'] for x in results), default=0)}</div><div class="l">max concurrency</div></div>
 <div class="card"><div class="v">{_gbps(max((x['aggregate_m_steps_s'] for x in results), default=0)):.1f} GB/s</div><div class="l">peak bandwidth (est, ~{_BYTES_PER_STEP_EST}B/step) &middot; {max((x['aggregate_m_steps_s'] for x in results), default=0):.0f} M steps/s</div></div>
 <div class="card"><div class="v">{len(results)}</div><div class="l">cells recorded</div></div>
</div>
<h2 style="color:#e6edf3;font-size:16px">Memory-pressure curve - time per Metropolis step vs working set (1 worker)</h2>
{_svg_curve(results, meta)}
<h2 style="color:#e6edf3;font-size:16px">Per-size summary (per-core speed at 1 worker, separate from aggregate)</h2>
{''.join(rows)}
{_scaling(results, meta)}
{ptable}
<div class="note">ns/step is per single Metropolis move (lower is faster); each cell times growth from empty to equilibrium.
Per-core columns are concurrency=1 (median across cores); "best aggregate" is the highest summed throughput across the
concurrency ladder at that N. <b>GB/s (est)</b> = aggregate M steps/s &times; ~{_BYTES_PER_STEP_EST}B/step (the random-access
bytes a move touches) &mdash; a bandwidth-equivalent figure for comparing against DIMM specs, not measured DRAM traffic;
M steps/s is the ground truth it is derived from. cpp and rust use an identical RNG so produce bit-identical graphs; both
match the Numba reference's degree distribution. Full per-group / per-concurrency rows are in the CSV/JSON.</div>
"""


# cli
def main():
    ap = argparse.ArgumentParser(description="Time-to-equilibrium memory benchmark: Numba vs C++ vs Rust.")
    ap.add_argument("--config", default=os.path.join(ROOT, "bench.toml"))
    ap.add_argument("--dry-run", action="store_true", help="print the schedule + rough ETA, run nothing")
    ap.add_argument("--simulate-cores", type=int, default=0, help="preview the schedule for a hypothetical core count")
    ap.add_argument("--simulate-ram-gb", type=float, default=0.0,
                    help="preview against this much RAM (GiB); default ~90 with --simulate-cores")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the HTML report in a browser")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, dry=args.dry_run, sim_cores=args.simulate_cores, sim_ram_gb=args.simulate_ram_gb,
        open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
