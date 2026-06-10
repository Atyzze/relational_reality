#!/usr/bin/env python3
"""
topo_bench — phase-structured engine benchmark with MEASURED core topology
==========================================================================
Replaces the old flat (N × every-disjoint-group × every-track) ladder, which
re-asked hardware questions at every N for every language (3000+ cells, days)
and grouped cores by INDEX ORDER — mixing SMT siblings and crossing CCD
boundaries blindly, which is exactly what produced 100%+ within-group spreads.

Four phases, each answering one question once:

  S  SOLO map        every physical core alone, N = 8, 16, 32, … doubling to
                     past the DRAM cliff. One curve per core: per-core quality
                     (clocks) AND cache cliffs (V-Cache cores keep speed to
                     ~3× larger N). Fastest native track only — the hardware
                     doesn't care what language asks.
  G  GROUPING        search-based, MEASURED: co-run pairs and group cores by
                     interference. Level 1 at cache-resident N finds SMT
                     siblings (two workers on one physical core collide even
                     with zero L3 pressure); level 2 at working set ≈ 0.6×L3
                     finds shared-L3 (CCD) groups: same-L3 pairs thrash, cross-
                     L3 pairs don't. Adaptive union-find: each new core is
                     tested against one representative per known group, so the
                     cost is n_cores × n_groups co-runs, not n². The measured
                     map is cross-checked against /sys and disagreements are
                     reported — /sys is treated as a claim, not truth.
  C  SCALING         with the DISCOVERED groups: for c = 2, 4, …, packed
                     (fill one L3 domain first) vs spread (round-robin across
                     domains), at a compact N set around the cache edges, then
                     the spread/all config alone keeps doubling N to the RAM
                     cap — the high-N answer the sweep actually needs.
  L  LANGUAGES       py vs cpp vs rust: solo on the best core across the full
                     N ladder, plus all-core spread at the largest sizes.

Measurement protocol (per cell): pin → construct (first-touch on the pinned
core) → WARM by growing until avg degree reaches the k target (wall-capped;
the achieved degree is recorded, so an under-warmed huge-N probe is visible,
never silent) → TIMED chunks until the wall budget is met. Time-boxed, so the
total runtime is known in advance and the ETA is near-exact. Effective MHz is
sampled each chunk so "slow because clocks" separates from "slow because
cache" in the per-core map.
"""

import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engines import get_engine                              # noqa: E402
from benchmarks import memory_benchmark as mb               # noqa: E402
from benchmarks.eta_model import EtaSmoother, plan_eta      # noqa: E402

BYTES_PER_NODE = None   # filled from max_degree at plan time: md*4 + 4


# ───────────────────────── measurement worker ─────────────────────────
def _mhz(core):
    v = mb._read(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
    try:
        return int(v) / 1000.0
    except (TypeError, ValueError):
        return None


def measure_worker(job):
    """One time-boxed measurement: pin, build, warm to the degree target,
    then timed chunks until budget_s of steady-state is accumulated."""
    track = job["track"]; N = job["N"]; core = job["core"]
    eng = job["eng"]; barrier = job.get("barrier")
    budget = job["budget_s"]; warm_cap = job["warm_cap_s"]
    k_target = job["warm_k_target"]; warm_tol = job.get("warm_tol", 0.05)
    mb._pin(core)
    cls = get_engine(mb._BACKEND[track])
    try:
        e = cls(N, max_degree=eng["max_degree"], seed=job["seed"],
                mode=eng["cpp_mode"], **mb._params(eng))
    except Exception as ex:
        return dict(track=track, N=N, core=core,
                    error=f"{type(ex).__name__}: {ex}")
    if barrier is not None:
        try:
            barrier.wait(timeout=600)
        except Exception:
            pass
    try:
        # chunk autotune: one sweep timed, then size chunks to ~0.15 s
        t0 = time.perf_counter()
        e.step(N)
        dt1 = max(time.perf_counter() - t0, 1e-9)
        chunk = max(1, int(0.15 / dt1))          # sweeps per chunk
        # WARM: grow until avg degree ≈ target — where "target" respects
        # the STRUCTURAL cap min(k, N−1): a simple graph on N=8 nodes can
        # never exceed degree 7, so waiting for k̂ ≥ 7.6 just burns the
        # whole warm_cap (≈45 s per tiny cell, the exact failure this
        # replaces). A plateau early-exit additionally stops the warm as
        # soon as k̂ stabilises (3 consecutive chunk readings within 1%),
        # whichever comes first; the wall cap stays as the last resort and
        # warm_deg is recorded either way, so nothing is ever silent.
        k_eff = min(k_target, N - 1) * (1 - warm_tol)
        warm_t0 = time.perf_counter(); warm_deg = e.stats()["k_avg"]
        recent = []
        while (warm_deg < k_eff
               and time.perf_counter() - warm_t0 < warm_cap):
            e.step(chunk * N)
            warm_deg = e.stats()["k_avg"]
            recent.append(warm_deg)
            if len(recent) >= 3:
                lo, hi = min(recent[-3:]), max(recent[-3:])
                if hi - lo <= max(0.01 * k_target, 0.02):
                    break                      # equilibrated below target
        warm_s = time.perf_counter() - warm_t0
        # TIMED: steady-state chunks until the budget is met
        steps = 0; secs = 0.0; mhz = []
        while secs < budget:
            m0 = _mhz(core)
            t0 = time.perf_counter()
            e.step(chunk * N)
            secs += time.perf_counter() - t0
            steps += chunk * N
            if m0 is not None:
                mhz.append(m0)
        st = e.stats()
    except RuntimeError:
        e.close()
        return dict(track=track, N=N, core=core, error="capped")
    e.close()
    return dict(track=track, N=N, core=core, cpu=mb._cpu_now(),
                seconds=secs, steps=steps, ns_step=secs * 1e9 / steps,
                m_steps_s=steps / secs / 1e6,
                warm_s=round(warm_s, 3), warm_deg=round(float(warm_deg), 3),
                edges=st["edges"], peak=st["peak_degree"],
                avg_deg=2 * st["edges"] / N,
                mhz=(round(statistics.median(mhz)) if mhz else None),
                rss_bytes=mb._rss())


# ───────────────────────── simulation (for tests / --simulate) ─────────
class SimMachine:
    """Deterministic fake machine: n physical cores × SMT, two L3 domains,
    domain 0 with V-Cache (3× L3) and slightly lower clocks (like an X3D
    part). measure() reproduces solo curves, sibling collisions and shared-L3
    thrash, so the whole pipeline — including topology DISCOVERY — runs and
    can be asserted against the planted ground truth in seconds."""

    def __init__(self, n_logical=16, bytes_per_node=132):
        self.nl = n_logical
        self.bpn = bytes_per_node
        nphys = n_logical // 2
        self.sib = {}
        for p in range(nphys):                       # cpu p ↔ cpu p+nphys
            self.sib[p] = p + nphys; self.sib[p + nphys] = p
        half = nphys // 2
        self.dom = {c: (0 if (c % nphys) < half else 1) for c in range(n_logical)}
        self.l3 = {0: 96 * 2**20, 1: 32 * 2**20}
        self.clock = {c: (0.92 if self.dom[c] == 0 else 1.0) for c in range(n_logical)}

    def topo(self):
        nphys = self.nl // 2
        groups = [dict(id=d, cpus=sorted(c for c in range(self.nl)
                                         if self.dom[c] == d),
                       l3_bytes=self.l3[d], is_vcache=(d == 0))
                  for d in (0, 1)]
        return dict(model=f"(simulated {self.nl} logical, V-Cache CCD0)",
                    mask=list(range(self.nl)), n_logical=self.nl,
                    physical=list(range(nphys)), groups=groups)

    def ns_step(self, core, N, co_resident=()):
        ws = N * self.bpn
        l3 = self.l3[self.dom[core]]
        same_l3_ws = ws * (1 + sum(1 for o in co_resident
                                   if self.dom[o] == self.dom[core]
                                   and self.sib[o] != core and o != core))
        base = 8.0 / self.clock[core]                       # cache-resident ns
        if same_l3_ws > l3:
            frac = min(1.0, math.log2(same_l3_ws / l3) / 3.0)
            base *= 1.0 + 4.0 * frac                        # → ~40 ns at DRAM
        if any(self.sib[core] == o for o in co_resident):
            base *= 1.9                                     # SMT collision
        return base

    def measure(self, job, co=()):
        ns = self.ns_step(job["core"], job["N"], co)
        return dict(track=job["track"], N=job["N"], core=job["core"],
                    cpu=job["core"], seconds=job["budget_s"],
                    steps=int(job["budget_s"] / ns * 1e9),
                    ns_step=ns, m_steps_s=1e3 / ns,
                    warm_s=0.0, warm_deg=7.9, edges=4 * job["N"],
                    peak=16, avg_deg=7.9, mhz=4500 * self.clock[job["core"]],
                    rss_bytes=job["N"] * self.bpn)


# ───────────────────────── topology discovery ─────────────────────────
def cache_edge_N(curve):
    """Measured cache edge from a solo curve, robust to GRADUAL ramps.

    Real hardware does not plateau-then-cliff: on a 9950X3D the solo curve
    creeps 12.6→14→16→19 ns through the L1→L2→L3 region long before the
    DRAM cliff, so "last N within 15% of fastest" lands at N≈4k for BOTH
    CCDs and the L3 probe would test far below either cache (zero
    interference, discovery fails). Instead: take the geometric MIDPOINT
    between the fastest speed and the slow (DRAM-side) plateau, find the
    last ladder N still below it and the first above it, and return
    (below_N, probe_N) with probe_N their geometric mean — a working set
    just inside the cache being measured, so a same-cache PAIR is ~2×
    over it (strong signal) while the solo baseline at the same N stays
    (nearly) resident. Returns (None, None) when the curve never slows.
    """
    if len(curve) < 4:
        return None, None
    Ns = sorted(curve)
    fast = min(curve.values())
    slow = max(curve[n] for n in Ns[-3:])
    if slow <= fast * 1.6:
        return None, None
    mid = math.sqrt(fast * slow)
    below = None
    above = None
    for n in Ns:
        if curve[n] <= mid:
            below = n
        elif above is None and below is not None:
            above = n
            break
    if below is None:
        return None, None
    if above is None:
        above = Ns[-1]
    return below, int(math.sqrt(below * above))


def discover_groups(cores, co_measure, solo_ns, n_small, solo_curves,
                    sib_thr=0.30, l3_thr=0.10, fallback_l3=32 * 2**20,
                    bpn=132, log=print):
    """MEASURED grouping by interference, adaptive union-find.

    co_measure(a, b, N) → (ns_a, ns_b) for a simultaneous co-run.
    solo_ns(c, N) → that core's solo ns/step at N (measured on demand).
    solo_curves[core] → {N: ns} from phase S.

    Level 1 (N = n_small, working set ≪ any cache): pairs that slow each
    other > sib_thr share a PHYSICAL core (SMT siblings) — cache-resident
    workers on different cores barely interact, siblings collide on
    execution ports.

    Level 2 is CACHE-EDGE driven, with no prior on cache sizes: each
    physical rep's own phase-S curve yields its cache edge (cache_edge_N);
    reps are binned by edge class, and a union-find runs WITHIN each class
    at a probe of 0.9× that class's edge — a solo worker still fits its
    cache, a same-cache pair is ~1.8× over it (a strong signal regardless
    of how gentle the penalty ramp is), while a different-domain pair of
    the same class stays resident. Different edge classes have different
    caches by measurement, hence different domains — never merged. This is
    what lets one probe size find the 32 MB CCD while another finds the
    96 MB V-Cache CCD, without being told either exists.
    """
    evidence = []

    def _interf(a, b, N):
        ns_a, ns_b = co_measure(a, b, N)
        sa = ns_a / solo_ns(a, N) - 1.0
        sb = ns_b / solo_ns(b, N) - 1.0
        s = max(sa, sb)
        evidence.append(dict(a=a, b=b, N=N, slow_a=round(sa, 3),
                             slow_b=round(sb, 3)))
        return s

    def _unionfind(pool, N, thr, tag):
        groups = []
        for c in pool:
            placed = False
            for g in groups:
                if _interf(c, g[0], N) > thr:
                    g.append(c); placed = True
                    break
            if not placed:
                groups.append([c])
            log(f"  [grp:{tag}] core {c:>3} → "
                f"{['|'.join(map(str, g)) for g in groups]}")
        return [sorted(g) for g in groups]

    sib_groups = _unionfind(list(cores), n_small, sib_thr, "smt")
    phys_reps = [g[0] for g in sib_groups]      # one logical per physical core
    edges = {r: cache_edge_N(solo_curves.get(r, {})) for r in phys_reps}
    classes = {}
    for r in phys_reps:
        classes.setdefault(edges[r][0], []).append(r)
    l3_groups = []
    for below, members in sorted(classes.items(),
                                 key=lambda kv: (kv[0] is None, kv[0])):
        if below is not None:
            probe = max(n_small * 2, edges[members[0]][1])
        else:
            probe = max(n_small * 2, int(0.75 * fallback_l3 / bpn))
        log(f"  [grp:l3] edge class "
            f"{'cache≈N=' + str(below) if below else 'undetected (fallback)'}"
            f" → probing {len(members)} rep(s) at N={probe} "
            f"(ws {probe * bpn / 2**20:.0f} MiB)")
        l3_groups += _unionfind(members, probe, l3_thr, f"l3@{probe}")
    # expand L3 groups back to all logicals via their sibling sets
    sib_of = {}
    for g in sib_groups:
        for c in g:
            sib_of[c] = g
    l3_full = [sorted({c for rep in g for c in sib_of[rep]})
               for g in l3_groups]
    return sib_groups, l3_full, evidence


# ───────────────────────── planning ─────────────────────────
def _ladder(n_start, n_max):
    out, n = [], int(n_start)
    while n <= n_max:
        out.append(n); n *= 2
    return out


def _snap(ladder, target):
    return min(ladder, key=lambda n: abs(math.log(max(n, 1) / max(target, 1))))


def make_plan(cfg, topo, avail_bytes, base_rss):
    """All phases as explicit cell lists with per-cell second estimates, so
    the ETA is a sum, not a model."""
    eng = cfg["engine"]; md = eng["max_degree"]
    bpn = md * 4 + 4
    ph = cfg["phases"]
    budget = float(ph["budget_s"]); reps = int(cfg["work"]["reps_per_cell"])
    # Reserve never eats more than a quarter of what is actually free, so a
    # small machine (or container) still gets a usable ladder instead of a
    # negative RAM cap; the floor keeps at least a 64× n_start ladder alive.
    reserve = min(int(cfg["memory"]["reserve_gb"] * 2**30),
                  int(0.25 * avail_bytes))
    safety = float(cfg["memory"]["safety"])
    ram_cap_N = int((avail_bytes - reserve) / (bpn * safety) - base_rss / bpn)
    ram_cap_N = max(ram_cap_N, int(ph["n_start"]) * 64)

    l3s = sorted({g.get("l3_bytes") or 0 for g in topo["groups"]} - {0})
    l3_min = l3s[0] if l3s else 32 * 2**20
    l3_max = l3s[-1] if l3s else l3_min
    solo_cap = min(int(ph["solo_n_max"]) or int(4 * l3_max / bpn), ram_cap_N)
    ladder = _ladder(ph["n_start"], max(solo_cap, 4 * l3_max // bpn))
    ladder = [n for n in ladder if n <= ram_cap_N]
    # Probe sizes are EXACT, not snapped to the ladder (solo baselines at a
    # probe N are measured on demand). n_star sits at 0.75× the smaller L3:
    # a solo worker still fits its cache, a same-L3 PAIR is 1.5× over it —
    # deep in the thrash regime, a strong unambiguous signal — while a
    # V-Cache pair (2×0.75×32M = 48M < 96M) stays comfortably resident.
    # (At 0.6×L3 the pair overflows by only ~3%, a ~6% slowdown that hides
    # under measurement noise; that mistake is what a fixed-threshold test
    # must avoid.)
    n_small = max(64, 256 * 2**10 // bpn)                # fits L2: SMT probe
    n_star = max(n_small * 2, int(0.75 * l3_min / bpn))  # L3-sharing probe

    phys = topo["physical"]
    pool = phys if cfg["cpu"]["physical_only"] else topo["mask"]
    rc = cfg["cpu"]["restrict_cores"]
    if rc:
        pool = [c for c in pool if c in set(rc)]
        phys = [c for c in phys if c in set(rc)]

    prefer = "cpp"            # fastest native; run() downgrades if unavailable
    plan = dict(ladder=ladder, n_small=n_small, n_star=n_star,
                solo_cap=solo_cap, ram_cap_N=ram_cap_N, pool=pool, phys=phys,
                budget_s=budget, reps=reps, bpn=bpn)

    def cell(phase, track, N, group, comp=""):
        return dict(phase=phase, track=track, N=N, conc=len(group),
                    group=list(group), comp=comp,
                    est_s=budget + min(float(ph["warm_cap_s"]),
                                       0.3 + N / 4e7))

    # S: every physical core solo, native track, full ladder up to solo_cap
    plan["S"] = [cell("S", prefer, N, (c,))
                 for N in ladder if N <= solo_cap for c in phys]
    # G: adaptive — bounded estimate only (cells materialise at runtime)
    n_groups_guess = max(2, len(topo["groups"]))
    plan["G_budget"] = ((len(pool) * 2) + (len(phys) * n_groups_guess)) \
        * (budget * 2 + 1.0)
    # C: filled at runtime from DISCOVERED groups (placeholders for the ETA)
    c_ladder = []
    c_ = 2
    while c_ <= len(phys):
        c_ladder.append(c_); c_ *= 2
    if not cfg["cpu"]["physical_only"] and len(pool) > len(phys):
        c_ladder.append(len(pool))
    n_set = sorted({n_small, n_star,
                    _snap(ladder, int(2.5 * l3_max / bpn)),
                    _snap(ladder, int(8 * l3_max / bpn))})
    plan["C_concs"] = c_ladder
    plan["C_Ns"] = [n for n in n_set if n <= ram_cap_N]
    n_all = max(len(pool), 1)
    high = [n for n in _ladder(max(plan["C_Ns"], default=ph["n_start"]),
                               ram_cap_N // n_all)][1:]   # EACH of the n_all
    plan["C_high_Ns"] = high                              # workers owns a
    # full graph, so the per-worker cap is ram_cap/n_all — without this the
    # spread-all doubling marches straight into the OOM killer
    plan["C_budget"] = (len(c_ladder) * 2 * len(plan["C_Ns"]) + len(high)) \
        * (budget + 4.0)
    # L: all tracks, solo ladder on best core + all-core spread at 3 largest N
    plan["L"] = ([cell("L", t, N, ("best",))
                  for t in ("py", "cpp", "rust") for N in ladder
                  if N <= solo_cap]
                 + [cell("L", t, N, tuple(pool), comp="spread")
                    for t in ("py", "cpp", "rust")
                    for N in (high[-3:] if high
                              else [min(plan["C_Ns"][-1],
                                        ram_cap_N // n_all)])])
    return plan


# ───────────────────────── orchestration ─────────────────────────
def _key(c):
    return (c["phase"], c["track"], int(c["N"]), int(c["conc"]),
            tuple(int(x) for x in c["group"] if isinstance(x, int)
                  or str(x).isdigit()), c.get("comp", ""))


def _rec_key(r):
    g = r.get("group") or []
    return (r.get("phase", "?"), r.get("track"), int(r.get("N", 0)),
            int(r.get("conc", 0)),
            tuple(int(x) for x in g if isinstance(x, int)
                  or str(x).isdigit()), r.get("comp", ""))


class Runner:
    """Executes cells either on the real machine (pinned processes with a
    start barrier) or against a SimMachine (instant, deterministic)."""

    def __init__(self, cfg, eng, sim=None):
        self.cfg, self.eng, self.sim = cfg, eng, sim
        self.pool = self.mgr = None

    def __enter__(self):
        if self.sim is None:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import Manager
            n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 8
            self.pool = ProcessPoolExecutor(max_workers=max(2, n),
                                            initializer=mb._init_worker)
            self.mgr = Manager()
        return self

    def __exit__(self, *a):
        if self.pool:
            self.pool.shutdown(wait=True)

    def co_run(self, cores, N, track, budget_s, warm_k, warm_cap_s, seed0=1):
        """Run len(cores) workers simultaneously (barrier-released), one per
        pinned core. Returns the per-worker records in core order."""
        jobs = [dict(track=track, N=N, core=c, seed=seed0 + 7 * i,
                     eng=self.eng, budget_s=budget_s, warm_cap_s=warm_cap_s,
                     warm_k_target=warm_k)
                for i, c in enumerate(cores)]
        if self.sim is not None:
            return [self.sim.measure(j, co=[c for c in cores if c != j["core"]])
                    for j in jobs]
        barrier = self.mgr.Barrier(len(cores)) if len(cores) > 1 else None
        for j in jobs:
            j["barrier"] = barrier
        return list(self.pool.map(measure_worker, jobs))


def _gname(groups, c):
    for i, g in enumerate(groups):
        if c in g:
            return i
    return None


def run(cfg, dry=False, sim_cores=0, sim_ram_gb=0.0, open_browser=True):
    sim = SimMachine(sim_cores) if sim_cores else None
    topo = sim.topo() if sim else mb.detect_topology()
    avail = int((sim_ram_gb or 90) * 2**30) if (sim or sim_ram_gb) \
        else (mb.mem_available() or mb.mem_total())

    eng = dict(cfg["engine"])
    if sim is None:
        mb.ensure_natives(cfg)
    trks = mb.tracks_for(cfg) or ["py"]
    prefer = "cpp" if "cpp" in trks else ("rust" if "rust" in trks else "py")
    if sim is None and not dry:
        if eng.get("mean_degree", 0) > 0:
            eng["degree_penalty"] = mb.calibrate_degree_penalty(
                eng, eng["mean_degree"], prefer, sweeps=400)
        if cfg["tracks"]["validate"]:
            mb.verify_tracks(cfg, eng, 300)
        base_rss = mb.warm_base_rss(cfg, eng)
    else:
        base_rss = 200 * 2**20

    plan = make_plan(cfg, topo, avail, base_rss)
    warm_k = (eng.get("mean_degree") or 8) * 1.0
    budget = plan["budget_s"]; warm_cap = float(cfg["phases"]["warm_cap_s"])

    print(f"\nCPU: {topo.get('model')} — {topo['n_logical']} logical, "
          f"{len(plan['phys'])} physical in pool, "
          f"{len(topo['groups'])} L3 group(s) per /sys")
    print(f"ladder: N {plan['ladder'][0]} → {plan['ladder'][-1]} (×2); "
          f"solo cap {mb.fmt_int(plan['solo_cap'])}; RAM cap N "
          f"{mb.fmt_int(plan['ram_cap_N'])}; probe sizes: SMT@N="
          f"{mb.fmt_int(plan['n_small'])}, L3@N={mb.fmt_int(plan['n_star'])}")
    eta0 = (sum(c["est_s"] for c in plan["S"]) + plan["G_budget"]
            + plan["C_budget"] + sum(c["est_s"] for c in plan["L"]))
    print(f"phases: S={len(plan['S'])} solo cells · G≈interference search · "
          f"C={plan['C_concs']}×(packed,spread) · L={len(plan['L'])} cells "
          f"— planned ≈ {mb.fmt_dur(eta0)} total "
          f"(time-boxed {budget:.2g}s/measurement)\n")
    if dry:
        return None

    outdir = os.path.join(ROOT, cfg["output"]["dir"])
    os.makedirs(outdir, exist_ok=True)
    results = []
    done = set()
    for r in mb._load_prior_results(cfg):
        if "phase" in r:
            results.append(r); done.add(_rec_key(r))
    if done:
        print(f"resume: {len(done)} cells on disk will be skipped "
              f"(delete {outdir}/ to reset)\n")

    meta = dict(model=topo.get("model"), n_logical=topo["n_logical"],
                groups=topo["groups"], cores=plan["pool"], tracks=trks,
                total_ram=avail, available=avail, base_rss=base_rss,
                reserve_gb=cfg["memory"]["reserve_gb"],
                params=dict(mean_degree=eng.get("mean_degree"),
                            degree_penalty=eng.get("degree_penalty"),
                            temperature=eng["temperature"],
                            locality_bias=eng["locality_bias"],
                            max_degree=eng["max_degree"]),
                cells_total=len(plan["S"]) + len(plan["L"]),
                cells_done=0, running=True, elapsed_s=0.0, eta_s=eta0)
    smoother = EtaSmoother()
    t_start = time.time()
    state = dict(done_est=0.0, total_est=eta0)
    rate_ns = {}            # track -> {N: measured ns/step}, fed by record()

    def est_cell(track, N, conc=1):
        """Per-cell wall estimate from measured rates: the timed budget plus
        the WARM cost (~300 sweeps × N × ns/step), which dominates at big N
        and is exactly what a fixed formula cannot know in advance."""
        tab = rate_ns.get(track) or rate_ns.get("cpp") or {}
        if tab:
            below = [n for n in tab if n <= N]
            n0 = max(below) if below else min(tab)
            ns = tab[n0] * (1.6 if N > 4 * n0 else (1.25 if N > n0 else 1.0))
            if track == "py" and "py" not in rate_ns:
                ns *= 4.0
        else:
            ns = 20.0
        warm = min(max(float(cfg["phases"]["warm_cap_s"]),
                       240.0 if N > plan["solo_cap"] else 0.0),
                   300.0 * N * ns * 1e-9)
        return budget + warm + 0.25

    def remaining_est():
        rem = sum(est_cell(c["track"], c["N"]) for c in plan["S"]
                  if _key(c) not in state["done_keys"])
        rem += sum(est_cell(c["track"], c["N"], len(c["group"]))
                   for c in plan["L"] if _key(c) not in state["done_keys"])
        rem += state["phase_budgets"]
        return rem
    state["done_keys"] = set(done)
    state["phase_budgets"] = plan["G_budget"] + plan["C_budget"]

    def flush(force=False):
        meta["elapsed_s"] = time.time() - t_start
        meta["cells_done"] = len(results)
        state["total_est"] = state["done_est"] + remaining_est()
        meta["eta_s"] = plan_eta(state["total_est"], state["done_est"],
                                 meta["elapsed_s"], smoother)
        try:
            mb.write_outputs(cfg, meta, results, quiet=True)
        except Exception as ex:
            print(f"[warn] report write failed: {ex}")

    def record(phase, track, N, group, comp, recs, est_s):
        ok = [x for x in recs if "error" not in x]
        state["done_est"] += est_s
        if not ok:
            print(f"{mb._ts()}  [{phase}] N={mb.fmt_int(N):>12} "
                  f"grp{list(group)}: SKIP ({recs[0].get('error', '?')})")
            return None
        nsv = [x["ns_step"] for x in ok]
        rec = dict(phase=phase, comp=comp, N=N, conc=len(group),
                   group=list(group), track=track,
                   working_set=N * plan["bpn"],
                   ns_step_med=statistics.median(nsv),
                   ns_step_min=min(nsv), ns_step_max=max(nsv),
                   per_worker_m_steps_s=statistics.median(
                       [x["m_steps_s"] for x in ok]),
                   aggregate_m_steps_s=len(group) * 1e3 / max(nsv),
                   aggregate_peak_m_steps_s=statistics.median(
                       [x["m_steps_s"] for x in ok]) * len(group),
                   seconds_med=statistics.median([x["seconds"] for x in ok]),
                   avg_deg=statistics.median([x["avg_deg"] for x in ok]),
                   warm_deg=min(x.get("warm_deg", 0) for x in ok),
                   mhz=[x.get("mhz") for x in ok],
                   peak=max(x["peak"] for x in ok),
                   rss_per_worker=statistics.median(
                       [x["rss_bytes"] for x in ok]),
                   ccds=sorted({_gname(meta.get("l3_groups", []), c)
                                for c in group} - {None}),
                   watts=None)
        results.append(rec)
        rate_ns.setdefault(track, {})[N] = rec["ns_step_med"]
        state["done_keys"].add((phase, track, N, len(group),
                                tuple(int(x) for x in group), comp))
        spread = (max(nsv) - min(nsv)) / rec["ns_step_med"] * 100
        print(f"{mb._ts()}  [{phase}{':' + comp if comp else ''}] "
              f"N={mb.fmt_int(N):>12} c={len(group):>2} {track:>4} "
              f"grp{str(list(group)):<18} {rec['ns_step_med']:7.1f} ns/step  "
              f"per-wkr {rec['per_worker_m_steps_s']:7.2f} M/s  "
              f"agg {rec['aggregate_m_steps_s']:8.2f} M/s  "
              f"spread {spread:5.1f}%  warm k̂={rec['warm_deg']:.2f}  "
              f"ETA {mb.fmt_dur(meta.get('eta_s') or 0)}")
        flush()
        return rec

    with Runner(cfg, eng, sim) as rn:
        # ── S: solo per-core map ──────────────────────────────────
        solo = {}                                   # (core, N) -> ns_step
        for c in plan["S"]:
            kk = _key(c)
            prior = next((r for r in results if _rec_key(r) == kk), None)
            if prior:
                solo[(prior["group"][0], prior["N"])] = prior["ns_step_med"]
                state["done_est"] += c["est_s"]
                continue
            recs = rn.co_run(c["group"], c["N"], c["track"], budget,
                             warm_k, warm_cap)
            r = record("S", c["track"], c["N"], c["group"], "", recs,
                       c["est_s"])
            if r:
                solo[(c["group"][0], c["N"])] = r["ns_step_med"]

        # ── G: measured grouping ─────────────────────────────────
        def co_measure(a, b, N):
            recs = rn.co_run((a, b), N, prefer, budget, warm_k, warm_cap)
            state["done_est"] += budget * 2 + 1.0
            by = {r["core"]: r for r in recs if "error" not in r}
            return (by.get(a, {}).get("ns_step", float("inf")),
                    by.get(b, {}).get("ns_step", float("inf")))

        def solo_ns(c, N):
            if (c, N) not in solo:           # SMT sibling not in phys set:
                recs = rn.co_run((c,), N, prefer, budget, warm_k, warm_cap)
                solo[(c, N)] = recs[0].get("ns_step", float("inf"))
                state["done_est"] += budget + 1.0
            return solo[(c, N)]

        ph = cfg["phases"]
        solo_curves = {}
        for (cc, N), ns in solo.items():
            solo_curves.setdefault(cc, {})[N] = ns
        l3s_sys = [g.get("l3_bytes") or 0 for g in topo["groups"]]
        sibs, l3g, evidence = discover_groups(
            plan["pool"], co_measure, solo_ns, plan["n_small"], solo_curves,
            sib_thr=float(ph["smt_threshold"]),
            l3_thr=float(ph["l3_threshold"]),
            fallback_l3=(min([x for x in l3s_sys if x] or [32 * 2**20])),
            bpn=plan["bpn"])
        state["phase_budgets"] = plan["C_budget"]     # G is done
        meta["sib_groups"] = sibs
        meta["l3_groups"] = l3g
        meta["grouping_evidence"] = evidence
        # cache edge + V-Cache verdict per measured L3 group: the group(s)
        # whose measured cache edge is ≥2× the smallest edge hold the bigger
        # L3 — on an X3D part, that IS the V-Cache CCD.
        cliffs = {}
        for gi, g in enumerate(l3g):
            reps = [c for c in g if c in plan["phys"]] or g
            cliffs[gi] = cache_edge_N(solo_curves.get(reps[0], {}))[0]
        meta["cliff_N"] = cliffs
        emin = min((v for v in cliffs.values() if v), default=None)
        meta["vcache_groups_measured"] = [
            gi for gi, v in cliffs.items()
            if v and emin and v >= 2 * emin and len(cliffs) > 1]
        # cross-check vs /sys
        sys_groups = [sorted(g["cpus"]) for g in topo["groups"]]
        agree = sorted(map(tuple, l3g)) == sorted(map(tuple, sys_groups))
        print(f"\n[G] measured SMT pairs: {sibs}")
        print(f"[G] measured L3 groups: {l3g}  "
              f"(cliff N per group: { {k: mb.fmt_int(v) if v else '-' for k, v in cliffs.items()} })")
        print(f"[G] V-Cache by MEASUREMENT: group(s) "
              f"{meta['vcache_groups_measured'] or 'none distinguishable'}")
        print(f"[G] /sys agreement: {'MATCHES /sys topology' if agree else 'DIFFERS from /sys — trust the measurement, report both'}\n")
        flush(True)

        # ── D: deep-RAM anchors per measured cache class ─────────
        # For each L3 group's fastest rep, solo points where the RAM-share
        # of the working set is pinned at 10/50/90/99% of ITS measured
        # cache: ws = cache/(1−r) ⇒ N = edge_N/(1−r). The drop from the
        # cache plateau through these anchors is the curve the needed
        # production N is read from; 99% is the "nothing fits" reference.
        # (Solo phase deliberately stops at ~4× the largest L3 — running
        # all 16 cores to 100× cache would cost hours of pure warm time;
        # one rep per cache class carries the deep-RAM answer.)
        for gi, g in enumerate(l3g):
            reps = [c for c in g if c in plan["phys"]] or g
            rep = min(reps, key=lambda c: solo.get((c, plan["n_small"]), 1e18))
            below = cliffs.get(gi)
            if not below:
                continue
            for r_share in (0.10, 0.50, 0.90, 0.99):
                Nd = min(int(below / (1.0 - r_share)), plan["ram_cap_N"])
                if (rep, Nd) in solo or Nd <= 0:
                    continue
                comp = f"ram{int(r_share * 100)}"
                kk = ("D", prefer, Nd, 1, (rep,), comp)
                if kk in done:
                    continue
                recs = rn.co_run((rep,), Nd, prefer, budget, warm_k,
                                 max(warm_cap, 240.0))
                rec = record("D", prefer, Nd, (rep,), comp, recs,
                             budget + 20.0)
                if rec:
                    solo[(rep, Nd)] = rec["ns_step_med"]

        # ── C: scaling with the discovered groups ────────────────
        order_packed = [c for gi in sorted(
            cliffs, key=lambda g: -(cliffs[g] or 0)) for c in l3g[gi]
            if c in plan["phys"]]
        def spread_pick(c_):
            seq, idx = [], {gi: 0 for gi in range(len(l3g))}
            gi = 0
            phys_by_g = {g: [c for c in l3g[g] if c in plan["phys"]]
                         for g in range(len(l3g))}
            while len(seq) < c_:
                g = gi % len(l3g)
                pool_g = phys_by_g[g]
                if idx[g] < len(pool_g):
                    seq.append(pool_g[idx[g]]); idx[g] += 1
                gi += 1
                if gi > 4 * c_:                     # SMT fill
                    rest = [x for x in plan["pool"] if x not in seq]
                    seq.extend(rest[:c_ - len(seq)]); break
            return tuple(seq[:c_])

        for c_ in plan["C_concs"]:
            comps = {"packed": tuple(order_packed[:c_]),
                     "spread": spread_pick(c_)}
            if c_ > len(plan["phys"]):
                comps = {"smt-all": tuple(plan["pool"][:c_])}
            for comp, grp in comps.items():
                if len(set(grp)) < c_:
                    continue
                for N in plan["C_Ns"]:
                    kk = ("C", prefer, N, c_, tuple(grp), comp)
                    if kk in done:
                        continue
                    recs = rn.co_run(grp, N, prefer, budget, warm_k, warm_cap)
                    record("C", prefer, N, grp, comp, recs, budget + 4.0)
        # High-N doubling to the RAM cap: spread across every domain when a
        # multi-core ladder exists, SOLO on the single physical core when it
        # doesn't (a 1-core machine still deserves the high-N answer — and
        # must not crash the run, as an earlier draft did via C_concs[-1]).
        grp_all = (spread_pick(plan["C_concs"][-1]) if plan["C_concs"]
                   else (plan["phys"][0],))
        high_comp = "spread-high" if plan["C_concs"] else "solo-high"
        for N in plan["C_high_Ns"]:
            kk = ("C", prefer, N, len(grp_all), tuple(grp_all), high_comp)
            if kk in done:
                continue
            recs = rn.co_run(grp_all, N, prefer, budget, warm_k,
                             max(warm_cap, 120.0))
            record("C", prefer, N, grp_all, high_comp, recs, budget + 10)

        state["phase_budgets"] = 0.0                  # C is done
        # ── L: language compare ──────────────────────────────────
        best = min(plan["phys"],
                   key=lambda c: solo.get((c, plan["n_small"]), 1e18))
        for c in plan["L"]:
            grp = ((best,) if c["group"] == ["best"]
                   else tuple(int(x) for x in c["group"]))
            track = c["track"]
            if track not in trks:
                continue
            kk = ("L", track, c["N"], len(grp), grp, c.get("comp", ""))
            if kk in done:
                continue
            recs = rn.co_run(grp, c["N"], track, budget, warm_k, warm_cap)
            record("L", track, c["N"], grp, c.get("comp", ""), recs,
                   c["est_s"])

    # ── recommendation + per-core map ────────────────────────────
    meta["recommendation"] = _recommend(results, meta, plan)
    meta["running"] = False
    flush(True)
    _percore_png(results, meta, plan,
                 os.path.join(outdir, "percore_map.png"))
    _print_summary(meta)
    print(f"\nwrote report to {outdir}/ (csv, json, html, percore_map.png) — "
          f"elapsed {mb.fmt_dur(time.time() - t_start)}")
    if open_browser and sim is None:
        mb._open_report(cfg)
    return results


def _recommend(results, meta, plan):
    crecs = [r for r in results if r.get("phase") == "C"]
    if not crecs:
        # single-core machine or C skipped: recommend the best solo config
        srecs = [r for r in results if r.get("phase") in ("S", "L")
                 and r.get("conc") == 1]
        if not srecs:
            return None
        bigN = max(r["N"] for r in srecs)
        best = max((r for r in srecs if r["N"] == bigN),
                   key=lambda r: r["aggregate_m_steps_s"])
        return dict(workers=1, pinning=best["group"], comp="solo",
                    at_N=bigN,
                    aggregate_m_steps_s=round(best["aggregate_m_steps_s"], 1),
                    note="no multi-core scaling data (1 physical core in "
                         "pool) — solo recommendation",
                    toml=(f"[compute]\nworkers = 1   "
                          f"# pin: {best['group']}"))
    bigN = max(r["N"] for r in crecs)
    at_big = [r for r in crecs if r["N"] == bigN]
    best = max(at_big, key=lambda r: r["aggregate_m_steps_s"])
    mid = [r for r in crecs if r["N"] == plan["n_star"]]
    note = ""
    if mid:
        packed = {r["conc"]: r for r in mid if r["comp"] == "packed"}
        spread = {r["conc"]: r for r in mid if r["comp"] == "spread"}
        common = sorted(set(packed) & set(spread))
        if common:
            c0 = common[-1]
            gain = (spread[c0]["aggregate_m_steps_s"]
                    / max(packed[c0]["aggregate_m_steps_s"], 1e-9) - 1)
            note = (f"at N={mb.fmt_int(plan['n_star'])} spreading across L3 "
                    f"domains is {gain * 100:+.0f}% vs packing one domain")
    return dict(workers=best["conc"], pinning=best["group"],
                comp=best["comp"], at_N=bigN,
                aggregate_m_steps_s=round(best["aggregate_m_steps_s"], 1),
                note=note,
                toml=(f"[compute]\nworkers = {best['conc']}   "
                      f"# pin order: {best['group']} ({best['comp']})"))


def _print_summary(meta):
    r = meta.get("recommendation")
    print("\n──────── measured topology ────────")
    print(f"  SMT sibling sets : {meta.get('sib_groups')}")
    print(f"  L3 groups        : {meta.get('l3_groups')}")
    print(f"  cache-cliff N    : {meta.get('cliff_N')}")
    print(f"  V-Cache (meas.)  : group(s) {meta.get('vcache_groups_measured')}")
    if r:
        print("──────── recommendation ────────")
        print(f"  best total throughput at N={mb.fmt_int(r['at_N'])}: "
              f"{r['workers']} workers ({r['comp']}) = "
              f"{r['aggregate_m_steps_s']} M steps/s")
        if r["note"]:
            print(f"  {r['note']}")
        print("  " + r["toml"].replace("\n", "\n  "))


def _percore_png(results, meta, plan, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    srecs = [r for r in results if r.get("phase") == "S"]
    if not srecs:
        return
    cores = sorted({r["group"][0] for r in srecs})
    Ns = sorted({r["N"] for r in srecs})
    M = np.full((len(cores), len(Ns)), np.nan)
    for r in srecs:
        M[cores.index(r["group"][0]), Ns.index(r["N"])] = \
            r["per_worker_m_steps_s"]
    fig, ax = plt.subplots(figsize=(1.1 + 0.5 * len(Ns),
                                    1.4 + 0.32 * len(cores)))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([mb.fmt_int(n) for n in Ns], rotation=60, fontsize=7)
    glab = {c: _gname(meta.get("l3_groups", []), c) for c in cores}
    vg = set(meta.get("vcache_groups_measured") or [])
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels([f"cpu{c}  [L3g{glab[c]}{' V$' if glab[c] in vg else ''}]"
                        for c in cores], fontsize=7)
    fig.colorbar(im, ax=ax, label="solo M steps/s")
    ax.set_title("per-core solo throughput across N — rows grouped by "
                 "MEASURED L3 domain (V$ = measured V-Cache)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
