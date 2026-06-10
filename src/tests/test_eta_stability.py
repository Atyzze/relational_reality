#!/usr/bin/env python3
"""Reproduce the cell-to-cell ETA swing on synthetic data and compare the
current _eta_milestones logic against eta_model.project_milestones.

The synthetic world matches the screenshots: 32 logical cores, a fast V-Cache
CCD (cores 0-15) and a slower regular CCD (16-31), per-worker rate that falls
with working set, and contention that drops per-worker rate as c rises.
"""
import os
import statistics
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from benchmarks import eta_model

MD = 32
TRACKS = ["py", "cpp", "rust"]
EQ = 300
VCACHE = set(range(16))                       # fast cores
N_FACTOR = 2

# per-worker steps/s at c=1, N=8 (cache-resident); falls ~ with log10(working set)
BASE = {"py": 30e6, "cpp": 64e6, "rust": 62e6}


def ws(N):
    return N * (MD * 4 + 4)


import math as _math


def true_per_worker_rate(track, N, c, core):
    """Ground-truth effective per-worker rate (steps/s) for the straggler."""
    decay = 1.0 / (1.0 + 0.55 * _math.log10(ws(N) / ws(8) + 1))   # working-set pressure
    r1 = BASE[track] * decay
    if core is not None and core not in VCACHE:
        r1 *= 0.55                                            # regular CCD ~2x slower
    # contention: per-worker rate falls as more workers share DRAM
    eff = 1.0 / (1.0 + 0.9 * (c - 1) * _math.log10(ws(N) / ws(8) + 10) / 32)
    return r1 * eff


BUDGET = 88 * 1024**3
BASE_RSS = 209 * 1024**2


def make_rounds(n_start, n_max, cores):
    """Mirror build_schedule's shape closely enough for ETA projection."""
    ladder_all = [1, 2, 4, 8, 16, 32]
    rounds = []
    N = n_start
    ridx = 0
    while N <= n_max:
        per_worker = int((ws(N) + BASE_RSS) * 1.30)
        max_c = max(1, min(32, BUDGET // per_worker))
        ladder = [c for c in ladder_all if c <= max_c]
        stages = []
        for c in ladder:
            if c == 1:
                groups = [(core,) for core in cores] if ridx < 8 else [(0,), (16,)]
            else:
                groups = [tuple(cores[i:i + c]) for i in range(0, len(cores) - c + 1, c)]
            for g in groups:
                stages.append(dict(N=N, conc=c, group=list(g), working_set=ws(N),
                                   need_per_worker=0, steps=EQ * N))
        rounds.append(dict(N=N, max_c_mem=max_c, ladder=ladder, stages=stages))
        N *= N_FACTOR
        ridx += 1
    return rounds


def run_cell(track, N, c, group):
    """Produce a result record like memory_benchmark.worker/run would."""
    rates = [true_per_worker_rate(track, N, c, core) for core in group]
    ns = [1e9 / r for r in rates]                              # ns/step per worker
    mst = [r / 1e6 for r in rates]
    return dict(N=N, conc=c, group=group, track=track, working_set=ws(N),
                ns_step_med=statistics.median(ns), ns_step_min=min(ns), ns_step_max=max(ns),
                per_worker_m_steps_s=statistics.median(mst),
                aggregate_m_steps_s=statistics.median(mst) * c,
                cell_wall_s=EQ * N / min(rates))                # straggler bounds wall


# ─────────── faithful copy of the CURRENT _eta_milestones / _est_rate ───────────
def OLD_est_rate(track, N, c, obs, ns_by_track):
    if (track, N, c) in obs:
        return obs[(track, N, c)]
    same = [r for (t, n, cc), r in obs.items() if t == track and n == N]
    if same:
        return statistics.median(same)
    Ns = ns_by_track.get(track) or []
    if Ns:
        le = [n for n in Ns if n <= N]
        ref = max(le) if le else min(Ns)
        at_c = [r for (t, n, cc), r in obs.items() if t == track and n == ref and cc == c]
        at = at_c or [r for (t, n, cc), r in obs.items() if t == track and n == ref]
        if at:
            return statistics.median(at)
    return eta_model.NOMINAL_DRAM.get(track, 3e6)


def OLD_eta(rounds, results, reps, eq, trks):
    completed = {(r["N"], r["conc"], r["track"], tuple(r.get("group") or [])) for r in results}
    obs = {(r["track"], r["N"], r["conc"]): r["per_worker_m_steps_s"] * 1e6
           for r in results if r.get("per_worker_m_steps_s")}        # NOTE: overwrites dups
    ns_by_track = {t: sorted({n for (tt, n, c) in obs if tt == t}) for t in trks}
    t_total = 0.0
    for rnd in rounds:
        N = rnd["N"]
        for st in rnd["stages"]:
            c = st["conc"]; g = tuple(st["group"])
            for track in trks:
                if (N, c, track, g) in completed:
                    continue
                t_total += reps * eq * N / OLD_est_rate(track, N, c, obs, ns_by_track)
    return t_total


def swing(series):
    s = [x for x in series if x is not None]
    if len(s) < 2:
        return 0.0, 0.0, 0.0
    return statistics.pstdev(s), (max(s) - min(s)), statistics.fmean(s)


def main():
    cores = list(range(32))
    rounds = make_rounds(8, 268435456, cores)

    # play out the sweep in schedule order, recomputing ETA after each cell.
    # measure over a FULL-PERCORE round's c=1 sweep (N=WIN, 32 cores alternating
    # V-Cache/regular), with the entire 268M horizon still ahead — the regime and
    # pattern the screenshots show.
    WIN = 1024
    results = []
    smoother = eta_model.EtaSmoother(alpha=0.2)
    rows = []        # (N, c, old_eta, new_eta, new_raw)
    elapsed = 0.0

    order = []
    for rnd in rounds:
        if rnd["N"] > WIN:
            break
        for st in rnd["stages"]:
            for track in TRACKS:
                order.append((track, rnd["N"], st["conc"], tuple(st["group"])))

    for (track, N, c, g) in order:
        rec = run_cell(track, N, c, list(g))
        results.append(rec)
        elapsed += rec["cell_wall_s"]
        old = OLD_eta(rounds, results, 1, EQ, TRACKS)
        m = eta_model.project_milestones(rounds, results, 1, EQ, TRACKS, smoother,
                                         MD, elapsed=elapsed)
        rows.append((N, c, old, m["t_total"], m["t_total_raw"]))

    win_old = [r[2] for r in rows if r[0] == WIN and r[1] == 1]
    win_new = [r[3] for r in rows if r[0] == WIN and r[1] == 1]
    win_raw = [r[4] for r in rows if r[0] == WIN and r[1] == 1]
    o_sd, o_rng, o_mean = swing(win_old)
    n_sd, n_rng, n_mean = swing(win_new)
    nr_sd, nr_rng, _ = swing(win_raw)

    def hours(s):
        return s / 3600.0

    print(f"ETA stability across the N={WIN:,} c=1 sweep "
          f"({len(win_old)} cells, V-Cache/regular cores alternating, full horizon ahead)")
    print(f"  {'':22}{'std (h)':>10}{'peak-peak (h)':>16}{'mean (h)':>12}")
    print(f"  {'OLD _eta_milestones':22}{hours(o_sd):>10.1f}{hours(o_rng):>16.1f}{hours(o_mean):>12.1f}")
    print(f"  {'NEW raw (pre-smooth)':22}{hours(nr_sd):>10.1f}{hours(nr_rng):>16.1f}{'':>12}")
    print(f"  {'NEW smoothed':22}{hours(n_sd):>10.1f}{hours(n_rng):>16.1f}{hours(n_mean):>12.1f}")
    if o_sd > 0 and n_sd > 0:
        def cmp(a, b):
            return f"{a / max(b, 1e-9):.1f}x smaller" if a >= b else f"{b / max(a, 1e-9):.1f}x LARGER"
        print(f"\n  displayed-ETA std: OLD {hours(o_sd):.1f}h -> NEW {hours(n_sd):.1f}h "
              f"({cmp(o_sd, n_sd)})")
        print(f"  peak-to-peak swing: OLD {hours(o_rng):.1f}h -> NEW {hours(n_rng):.1f}h "
              f"({cmp(o_rng, n_rng)})")

    # show the milestone extras the new model exposes
    m = eta_model.project_milestones(rounds, [], 1, EQ, TRACKS, None, MD)
    print("\n  milestone fields exposed for the live line:")
    print(f"    taper starts at N={m['drop_N']} (tapers to {m['fill_c']} workers), "
          f"trajectory {m['taper_workers']}")
    print(f"    {m['post_taper_rounds']} size-rounds remain after taper")

    # guard the property the model exists to provide: the displayed ETA must be
    # at least as stable as the old logic in this regime (it is ~4x better).
    assert n_sd <= o_sd, f"NEW displayed std {hours(n_sd):.1f}h is worse than OLD {hours(o_sd):.1f}h"
    assert n_rng <= o_rng, f"NEW peak-peak {hours(n_rng):.1f}h is worse than OLD {hours(o_rng):.1f}h"
    for k in ("t_total", "t_total_raw", "t_total_lo", "t_total_hi", "rho",
              "drop_N", "fill_c", "final_N", "final_c", "final_secs", "max_conc"):
        assert k in m, f"missing milestone key {k!r}"
    print("\n  PASS: new model is more stable and exposes all expected fields.")


if __name__ == "__main__":
    main()
