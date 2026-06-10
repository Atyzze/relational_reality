#!/usr/bin/env python3
"""optimal_workers.py — find the worker count with the most cells/min.

A stress test, run on its own (like isotropy.py) so its output isn't tangled
with the sweep's live plotting loop, and so the machine is doing nothing but
this benchmark while it runs — a clean sample.

It builds ONE fixed cell over and over at a 1, 4, 8, 16, … all-cores ladder
(1 core = the per-core-efficiency baseline), runs the same number of cells per
core at each step, prints a full per-core-efficiency table, and reports the
fastest count. It does NOT touch main.toml — the config is yours to edit; it
just prints the `workers = N` line to paste in if you want it.

The cell is fixed and deliberately large (N=256k by default): a small cell
fits entirely in L3, so memory contention never shows and more cores always
"wins"; 256k spills to DRAM, so the optimum reflects real bandwidth/cache
pressure on this machine.

Run (from the project root):
    python src/physics_tests/optimal_workers.py
    python src/physics_tests/optimal_workers.py --N 256000 --reps-per-core 10
    taskset -c 0-7 python src/physics_tests/optimal_workers.py      # one CCD

By default it also writes a self-contained HTML report (worker_tuning_report.html)
with the throughput-vs-workers curve, a per-core latency profile that isolates
each core (so a 3D-V-Cache CCD's lower memory latency is visible), and a
per-cell scatter. Useful flags:
    --report PATH        where to write the HTML ('none' to skip)
    --no-per-core        skip the pinned per-core profile (throughput only)
    --per-core-reps N    cells per core in that profile (serial; keep small)
    --demo               render a synthetic-data report and exit (no compute) —
                         preview the layout in seconds
    --map-cache          also map the EFFECTIVE cache hierarchy by sweeping the
                         graph size on a fixed core (one per CCD): the per-node
                         time knees approximate the L2/L3/DRAM boundaries this
                         workload actually sees. A longer pinned pass.
    --mem-bench          run the LIVE memory-pressure benchmark instead of the
                         tuner: climb N from small to ~1e9 in per-core-pinned
                         batches, eating RAM but always leaving a reserve free,
                         throttling workers down as cells grow, writing a
                         self-refreshing HTML with current/next-batch ETAs and a
                         V-Cache-vs-standard-CCD curve at every N.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)                       # …/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def doubling_ladder(cores, start=4):
    """1 (baseline) + start, 2·start, … up to and including the core count."""
    w, out = start, []
    while w < cores:
        out.append(w)
        w *= 2
    out.append(cores)
    return sorted({1, *(x for x in out if 1 <= x <= cores)})


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--lb", type=float, default=0.99)
    ap.add_argument("--N", type=int, default=256000,
                    help="cell size (256k stresses memory; the point of the "
                         "test). Smaller fits in cache and over-picks cores.")
    ap.add_argument("--reps-per-core", type=int, default=10,
                    help="throughput sweep: cells per core per worker-count "
                         "(sample size).")
    ap.add_argument("--workers", type=int, nargs="+", default=None,
                    help="explicit ladder; default is 1,4,8,… up to all cores.")
    ap.add_argument("--n-probes", type=int, default=60)
    ap.add_argument("--lanczos-m", type=int, default=300)
    ap.add_argument("--half-window", type=int, default=10)
    ap.add_argument("--ec", type=float, default=-1.0)
    # ── report / per-core profile ──
    ap.add_argument("--report", default="worker_tuning_report.html",
                    help="path for the self-contained HTML report "
                         "('none' to skip). Default worker_tuning_report.html.")
    ap.add_argument("--per-core", dest="per_core", action="store_true",
                    default=True,
                    help="also run the affinity-pinned per-core profile that "
                         "isolates each core's speed (reveals the V-Cache CCD). "
                         "On by default.")
    ap.add_argument("--no-per-core", dest="per_core", action="store_false",
                    help="skip the per-core profile (throughput sweep only).")
    ap.add_argument("--per-core-reps", type=int, default=3,
                    help="cells per core in the per-core profile (run serially, "
                         "so keep small). Default 3.")
    ap.add_argument("--per-core-cores", type=int, nargs="+", default=None,
                    help="explicit logical-CPU ids to profile per-core "
                         "(default: every core in the affinity mask).")
    ap.add_argument("--fixed-seeds", dest="random_seeds", action="store_false",
                    default=True,
                    help="reuse seeds 0..n instead of random distinct seeds per "
                         "cell (default: random, so every cell is real work).")
    ap.add_argument("--demo", action="store_true",
                    help="write a synthetic-data report and exit (no compute) — "
                         "preview the report layout in seconds.")
    # ── cache-hierarchy mapping (longer run; off by default) ──
    ap.add_argument("--map-cache", dest="map_cache", action="store_true",
                    help="also sweep the graph size on a fixed core (one per "
                         "CCD) to map the EFFECTIVE cache hierarchy the workload "
                         "sees — the per-node-time knees approximate L2/L3/DRAM "
                         "boundaries. Adds a long pinned pass; off by default.")
    ap.add_argument("--cache-n-min", type=int, default=2000,
                    help="smallest N in the cache-map sweep (default 2000).")
    ap.add_argument("--cache-n-max", type=int, default=4_000_000,
                    help="largest N in the cache-map sweep (default 4,000,000 — "
                         "raise to push past a big V-Cache L3 into DRAM).")
    ap.add_argument("--cache-points", type=int, default=12,
                    help="number of N values (log-spaced) in the sweep.")
    ap.add_argument("--cache-reps", type=int, default=6,
                    help="cells per (core, N) in the sweep (median). Default 6.")
    ap.add_argument("--cache-cores", type=int, nargs="+", default=None,
                    help="logical-CPU ids to map (default: one per CCD).")
    # ── live memory-pressure benchmark ──
    ap.add_argument("--mem-bench", dest="mem_bench", action="store_true",
                    help="run the LIVE memory-pressure benchmark instead of the "
                         "worker tuner: climb N from small to huge, eating RAM "
                         "but always leaving the reserve free, throttling workers "
                         "down as cells grow, writing a self-refreshing HTML.")
    ap.add_argument("--mem-n-min", type=int, default=1000,
                    help="smallest N (start small so points stream in fast). "
                         "Default 1000.")
    ap.add_argument("--mem-n-max", type=int, default=1_000_000_000,
                    help="largest N to attempt (default 1,000,000,000 = 1e9; "
                         "the run stops earlier if a single cell won't fit).")
    ap.add_argument("--mem-points", type=int, default=26,
                    help="number of N values (log-spaced) from min to max.")
    ap.add_argument("--mem-reps", type=int, default=1,
                    help="cells per N (median). 1 keeps the climb moving.")
    ap.add_argument("--reserve-gb", type=float, default=None,
                    help="GiB of RAM to always keep free (default: config "
                         "[compute].min_free_gb, else 4).")
    ap.add_argument("--live-interval", type=float, default=4.0,
                    help="seconds between live HTML rewrites / auto-refresh. "
                         "Default 4.")
    ap.add_argument("--mem-bytes-per-node", type=float, default=None,
                    help="override the per-node footprint estimate (bytes) used "
                         "for the memory gate; default models node_neighbors.")
    args = ap.parse_args(argv)

    import bench_report  # pure (no numpy/numba) — safe even for --demo

    # Demo: render a report from fabricated data and stop (no compute).
    if args.demo:
        out = (args.report if args.report.lower() != "none"
               else "worker_tuning_report.html")
        if args.mem_bench:
            topo, params, mb = bench_report.synthetic_mem_demo()
            bench_report.write_html_report(
                out, topo, [], [], {}, params,
                meta={"elapsed_s": mb["status"].get("elapsed_s")},
                refresh_s=args.live_interval, mem_bench=mb)
            print(f"[mem] wrote SYNTHETIC live-memory demo → "
                  f"{os.path.abspath(out)}")
        else:
            bench_report.write_html_report(out, *bench_report.synthetic_demo())
            print(f"[tune] wrote SYNTHETIC demo report → {os.path.abspath(out)}")
        print("[*] no benchmark ran; drop --demo for real numbers.")
        return

    import bench_workers

    # ── live memory-pressure benchmark mode ──
    if args.mem_bench:
        topo = bench_workers.detect_topology()
        lo, hi = args.mem_n_min, args.mem_n_max
        npts = max(2, args.mem_points)
        ratio = (hi / lo) ** (1.0 / (npts - 1))
        n_grid = sorted({int(round(lo * ratio ** i)) for i in range(npts)})
        reserve = int(args.reserve_gb * 1024 ** 3) if args.reserve_gb else None
        report_path = (args.report if args.report.lower() != "none"
                       else "worker_tuning_report.html")
        params = {"k": args.k, "T": args.T, "lb": args.lb, "N": n_grid[-1],
                  "n_probes": args.n_probes, "lanczos_m": args.lanczos_m,
                  "half_window": args.half_window}
        rgb = args.reserve_gb if args.reserve_gb else "config min_free_gb"
        print(f"[mem] live memory benchmark: N {n_grid[0]:,} … {n_grid[-1]:,} "
              f"({len(n_grid)} sizes); keeping {rgb} GiB free.")
        print(f"[mem] OPEN {os.path.abspath(report_path)} in a browser — it "
              f"self-refreshes every {args.live_interval:g}s.")
        print("[mem] Ctrl-C stops the climb; partial results are kept.\n")

        def on_update(mb):
            running = mb.get("status", {}).get("running", True)
            bench_report.write_html_report(
                report_path, topo, [], [], {}, params,
                meta={"elapsed_s": mb["status"].get("elapsed_s")},
                refresh_s=(args.live_interval if running else None),
                mem_bench=mb)

        final = bench_workers.live_memory_benchmark(
            args.k, args.T, args.lb, n_grid, reserve_bytes=reserve,
            reps_per_n=args.mem_reps, bytes_per_node=args.mem_bytes_per_node,
            n_probes=args.n_probes, lanczos_m=args.lanczos_m,
            half_window=args.half_window, ec=args.ec,
            random_seeds=args.random_seeds, topology=topo,
            on_update=on_update, update_interval=args.live_interval,
            log=lambda m: print(f"[mem] {m}", flush=True))
        stt = final["status"]
        print(f"\n[mem] done — cells={stt['cells_done']}, "
              f"max N={stt['max_N']:,}, peak mem="
              f"{(stt['peak_used_bytes'] or 0)/1024**3:.0f} GiB. "
              f"report → {os.path.abspath(report_path)}")
        if stt.get("stop_reason"):
            print(f"[mem] {stt['stop_reason']}")
        return

    topo = bench_workers.detect_topology()
    aff = (len(os.sched_getaffinity(0))
           if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
    ladder = sorted(set(args.workers)) if args.workers else doubling_ladder(aff)
    vtag = ""
    if topo.get("vcache_ccd_ids"):
        vset = set(topo["vcache_ccd_ids"])
        vc = [c for c in topo["ccds"] if c["id"] in vset]
        vtag = ("; V-Cache CCD: "
                + ", ".join(f"{len(c['cpus'])} cores @ "
                            f"{bench_report.fmt_bytes(c['l3_bytes'])} L3"
                            for c in vc))
    print(f"[tune] machine: {topo.get('model') or '?'} — "
          f"{os.cpu_count()} logical CPUs, {aff} in mask{vtag}")
    print(f"[tune] fixed cell k={args.k} T={args.T:g} lb={args.lb:g} "
          f"N={args.N:,}; ladder {ladder}; "
          f"{args.reps_per_core} cells/core each.")
    print("[tune] nothing else should be running — this wants the machine to "
          "itself for a clean sample.\n")

    t_start = time.time()
    summary, per_cell = bench_workers.run_throughput_detailed(
        args.k, args.T, args.lb, args.N, workers=ladder,
        reps_per_core=args.reps_per_core, n_probes=args.n_probes,
        lanczos_m=args.lanczos_m, half_window=args.half_window, ec=args.ec,
        warmup=True, random_seeds=args.random_seeds, topology=topo,
        log=lambda m: print(f"[tune] {m}", flush=True))

    print("\n" + bench_workers.format_table(summary))
    best = bench_workers.pick_best(summary)
    if best:
        best_cpm = max(r["cells_per_min"] for r in summary)
        print(f"\n[tune] fastest: {best} workers ({best_cpm:.1f} cells/min).")
        # Report only — the config is yours to edit. Print the line to paste.
        print(f"[tune] to use it, set this in main.toml under [compute]:\n"
              f"           workers = {best}")
    else:
        print("[tune] no usable throughput result.")

    # Per-core profile (pinned, one cell at a time) — the V-Cache revealer.
    percore = {}
    if args.per_core:
        print("\n[tune] per-core profile (pinned, serial — isolates each "
              "core) …", flush=True)
        percore, _pc = bench_workers.profile_per_core(
            args.k, args.T, args.lb, args.N,
            cores=args.per_core_cores, reps_per_core=args.per_core_reps,
            n_probes=args.n_probes, lanczos_m=args.lanczos_m,
            half_window=args.half_window, ec=args.ec,
            random_seeds=args.random_seeds, topology=topo, warmup=False,
            log=lambda m: print(f"[tune] {m}", flush=True))
        verdict = bench_report.vcache_verdict(percore, topo)
        if verdict:
            print(f"[tune] V-Cache CCD median {verdict['vcache_median_s']:.2f}"
                  f"s/cell vs {verdict['other_median_s']:.2f}s on the rest "
                  f"→ {verdict['speedup']:.2f}× ({verdict['pct_faster']:+.0f}%).")

    # Cache-hierarchy map (pinned size sweep) — discover the effective levels.
    cache_data = None
    if args.map_cache:
        lo, hi = args.cache_n_min, args.cache_n_max
        npts = max(2, args.cache_points)
        ratio = (hi / lo) ** (1.0 / (npts - 1))
        n_grid = sorted({int(round(lo * ratio ** i)) for i in range(npts)})
        print(f"\n[tune] cache map: sweeping N {n_grid[0]:,}…{n_grid[-1]:,} "
              f"({len(n_grid)} sizes) on a fixed core per CCD …", flush=True)
        cache_data, _cc = bench_workers.map_cache_hierarchy(
            args.k, args.T, args.lb, n_grid, cores=args.cache_cores,
            reps_per_n=args.cache_reps, n_probes=args.n_probes,
            lanczos_m=args.lanczos_m, half_window=args.half_window, ec=args.ec,
            random_seeds=args.random_seeds, topology=topo,
            log=lambda m: print(f"[tune] {m}", flush=True))
        for cpu in sorted(cache_data or {}):
            knees = bench_report.detect_cache_levels(cache_data[cpu])
            ktxt = ", ".join(f"≈{bench_report.fmt_bytes(k['boundary_bytes'])}"
                             for k in knees) or "none detected"
            print(f"[tune]   cpu {cpu}: effective knees at {ktxt}")

    # Write the HTML report.
    if args.report and args.report.lower() != "none":
        params = {"k": args.k, "T": args.T, "lb": args.lb, "N": args.N,
                  "n_probes": args.n_probes, "lanczos_m": args.lanczos_m,
                  "half_window": args.half_window,
                  "reps_per_core": args.reps_per_core}
        meta = {"elapsed_s": time.time() - t_start}
        bench_report.write_html_report(args.report, topo, summary, per_cell,
                                       percore, params, meta,
                                       cache_data=cache_data)
        print(f"\n[tune] report → {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
