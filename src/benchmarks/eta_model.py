#!/usr/bin/env python3
"""
eta_model.py — a stable, working-set- and concurrency-aware ETA model for the
memory benchmark. Pure stdlib; no engine/hardware deps, so it is unit-testable
without a multi-day run (that testability is the whole point — the original
_eta_milestones could only be exercised by actually running the sweep, which is
why its jitter went unnoticed). See test_eta_stability.py.

Drop-in usage (memory_benchmark.py does exactly this):

    from eta_model import project_milestones, EtaSmoother, NOMINAL_DRAM
    eta_smoother = EtaSmoother()          # one per run
    def milestones_now():
        return project_milestones(rounds, results, reps, eq_sweeps, trks,
                                  eta_smoother, md, elapsed=time.time() - t_start)

Why this is stable where the original snapped by hours
------------------------------------------------------
The total run time is tail-dominated: per-cell time grows ~ N / rate(N), so the
largest few N dwarf everything else. The original projected every unmeasured
large-N cell from the SINGLE most-recent same-(track,N,c) sample — and because a
c=1 key is shared by every core, that sample flip-flopped between a fast V-Cache
core and a slow regular core. Multiplied across the tail, the displayed ETA
lurched by hours each round.

This model instead fits a SMOOTH SURFACE from ALL cells measured so far and
projects the remaining schedule off the fit:

1. RATE BASIS = SLOWEST WORKER. A cell ends only when its straggler does, so the
   effective per-step cost is ns_step_max (or a recorded cell_wall_s), not the
   median per-worker rate. Summing these gives an honest wall-clock estimate.

2. MEDIAN ACROSS GROUPS/REPS per (track, N, c) — no last-writer-wins overwrite.

3. MONOTONE FIT. A 1-worker latency curve ns1(log2 working_set) is forced
   non-decreasing by pool-adjacent-violators (cache-cheap -> DRAM-slow is
   physically monotone), plus a contention slope alpha(log2 ws) giving the
   multiplier g = 1 + alpha*(c-1). Isotonic regression removes sample wobble
   without assuming a parametric shape; one noisy new cell barely moves a curve
   pinned by dozens of points. Extrapolation past the measured range continues
   the last segment's slope, capped near the DRAM plateau so it can't explode.

4. CONFIDENCE-WEIGHTED CALIBRATION. The same model costs every DONE cell too;
   comparing that to real elapsed time gives an observed scale factor. We blend
   it in by how much of the run is actually measured (conf = done/total model
   cost), so early on we trust the model and late on we trust reality.

5. DEADBAND + DAMPED DISPLAY. Even a good estimate jitters as cells land. The
   shown total ignores changes inside a 5% deadband and otherwise steps only a
   fraction of the way, with a hard reset on >4x discontinuities (resume /
   schedule rebuild). Raw is preserved in the result dict.

A relative-residual +/- band (rho) is returned so the UI can show honest
uncertainty; it shrinks as measured cells span more of the working-set range.
"""

import bisect
import math
import statistics


NOMINAL_DRAM = {"py": 12.0e6, "cpp": 24.0e6, "rust": 24.0e6}
# Per-worker steps/s, rough DRAM-resident floor, used ONLY before anything is
# measured (dry-run, first cells). The previous {2.5e6, 4e6} values were ~8x too
# pessimistic vs. observed DRAM-resident rates (~20-40 M steps/s), which made
# dry-run ETAs and early projections misleading.


# ───────────────────────── monotone surface fit ─────────────────────────
def _isotonic(ys):
    """Pool-adjacent-violators: nearest non-decreasing sequence to ys (L2 sense)."""
    stack = []                                   # each entry: [mean, weight, count]
    for y in ys:
        v, w, n = float(y), 1.0, 1
        while stack and stack[-1][0] > v:
            pv, pw, pn = stack.pop()
            v = (pv * pw + v * w) / (pw + w); w += pw; n += pn
        stack.append([v, w, n])
    out = []
    for v, _w, n in stack:
        out.extend([v] * n)
    return out


def _eff_ns(rec, eq_sweeps):
    """Effective ns/step for an already-run cell = the SLOWEST worker (what bounds
    the cell). Prefer a recorded cell_wall_s, else ns_step_max, else reconstruct
    from the median per-worker rate."""
    if rec.get("cell_wall_s") and rec.get("N"):
        steps = eq_sweeps * rec["N"]
        return (float(rec["cell_wall_s"]) * 1e9 / steps) if steps else None
    if rec.get("ns_step_max"):
        return rec["ns_step_max"]
    if rec.get("ns_step_med"):
        return rec["ns_step_med"]
    r = rec.get("per_worker_m_steps_s")
    return (1e3 / r) if r else None


class RateModel:
    """Effective ns/step for any (track, working_set, concurrency), fit from the
    cells measured so far: a monotone 1-worker latency curve ns1(log2 W) and a
    contention slope alpha(log2 W). Refit cheaply on every call."""

    def __init__(self):
        self.curve = {}     # track -> (xs=log2 W, ns1)
        self.alpha = {}     # track -> (xs=log2 W, slope)

    def update(self, results, trks, eq_sweeps):
        self.curve.clear(); self.alpha.clear()
        for t in trks:
            rows = [(r, _eff_ns(r, eq_sweeps)) for r in results if r.get("track") == t]
            rows = [(r, ns) for r, ns in rows if ns]
            if not rows:
                continue
            c1 = {}
            for r, ns in rows:
                if r["conc"] == 1:
                    c1.setdefault(math.log2(max(2, r["working_set"])), []).append(ns)
            if c1:
                xs = sorted(c1)
                self.curve[t] = (xs, _isotonic([statistics.median(c1[x]) for x in xs]))
            byN, ws_of = {}, {}
            for r, ns in rows:
                byN.setdefault(r["N"], {}).setdefault(r["conc"], []).append(ns)
                ws_of[r["N"]] = r["working_set"]
            slopes = []
            for N, d in byN.items():
                base = statistics.median(d[1]) if d.get(1) else None
                if not base:
                    continue
                sl = [(statistics.median(d[c]) / base - 1.0) / (c - 1) for c in d if c > 1 and d[c]]
                if sl:
                    slopes.append((math.log2(max(2, ws_of[N])), max(0.0, statistics.median(sl))))
            if slopes:
                slopes.sort()
                self.alpha[t] = ([x for x, _ in slopes], _isotonic([a for _, a in slopes]))

    @staticmethod
    def _interp(xs, ys, x, cap_right=1.4):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:                          # right tail: continue last slope, capped near plateau
            if len(xs) >= 2:
                slope = (ys[-1] - ys[-2]) / max(1e-9, xs[-1] - xs[-2])
                return min(ys[-1] + max(0.0, slope) * (x - xs[-1]), ys[-1] * cap_right)
            return ys[-1]
        i = bisect.bisect_right(xs, x)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1

    def predict_ns(self, track, W, c):
        lW = math.log2(max(2, W))
        cur = self.curve.get(track)
        if cur and len(cur[0]) >= 2:
            ns1 = self._interp(cur[0], cur[1], lW)
        elif cur:
            ns1 = cur[1][0]
        else:
            ns1 = 1e9 / NOMINAL_DRAM.get(track, 3e6)
        al = self.alpha.get(track)
        alpha = self._interp(al[0], al[1], lW, cap_right=2.0) if al else 0.0
        return ns1 * (1.0 + alpha * max(0, c - 1))

    def residual(self, results, trks, eq_sweeps):
        """Relative RMS of (predicted-measured)/measured over measured cells — the
        basis for the +/- band. Shrinks as cells span more of the W/c range."""
        errs = []
        for r in results:
            if r.get("track") not in trks:
                continue
            act = _eff_ns(r, eq_sweeps)
            if not act:
                continue
            pred = self.predict_ns(r["track"], r["working_set"], r["conc"])
            errs.append((pred - act) / act)
        if len(errs) < 3:
            return 0.30
        return min(0.60, (sum(e * e for e in errs) / len(errs)) ** 0.5)


# ───────────────────────── display smoother ─────────────────────────
class EtaSmoother:
    """Damped display value for the ETA. Ignores changes inside a deadband and
    otherwise steps only `alpha` of the way toward the new estimate; resets on a
    >4x discontinuity (resume / schedule rebuild) so it never lags a real regime
    change. (`alpha` is the step gain; kept as the kwarg name for back-compat.)"""
    def __init__(self, alpha=0.35, deadband=0.05):
        self.gain = alpha; self.db = deadband
        self.value = None

    def update(self, raw):
        if raw is None:
            return self.value
        if self.value is None or raw <= 0 or raw > 4 * self.value or raw < 0.25 * self.value:
            self.value = raw
        else:
            rel = abs(raw - self.value) / max(1e-9, self.value)
            if rel > self.db:
                self.value += self.gain * (raw - self.value)
        return self.value


# ───────────────────────── milestones ─────────────────────────
def project_milestones(rounds, results, reps, eq_sweeps, trks, smoother=None,
                       md=32, elapsed=None):
    """Project the known remaining schedule into wall-clock milestones off the
    fitted rate surface. Sequential across (round, stage, track, reps), matching
    run(). Every N-coordinate is a deterministic schedule fact; only the
    wall-clock mappings depend on the (fitted, stable) rate."""
    if not rounds or not trks:
        return None

    completed = {(r["N"], r["conc"], r["track"], tuple(r.get("group") or []))
                 for r in results}
    model = RateModel()
    model.update(results, trks, eq_sweeps)

    max_conc = max((c for rnd in rounds for c in rnd["ladder"]), default=1)
    drop_rnd = next((rnd for rnd in rounds
                     if rnd["ladder"] and rnd["ladder"][-1] < max_conc), None)
    drop_N = drop_rnd["N"] if drop_rnd else None
    fill_c = drop_rnd["ladder"][-1] if drop_rnd else max_conc     # workers it first tapers TO

    final_N = rounds[-1]["N"]
    last_stage = rounds[-1]["stages"][-1] if rounds[-1]["stages"] else None
    final_c = last_stage["conc"] if last_stage else 1            # last executed cell (c=1 at true terminal)
    final_W = last_stage["working_set"] if last_stage else final_N * (md * 4 + 4)
    final_secs = reps * eq_sweeps * final_N * model.predict_ns(trks[-1], final_W, final_c) / 1e9

    cost_remaining = 0.0
    cost_done = 0.0
    t_to_drop = 0.0
    before_drop = (drop_N is not None)
    post_taper_steps = 0
    post_taper_round_Ns = []
    for rnd in rounds:
        N = rnd["N"]
        if drop_N is not None and N >= drop_N:
            before_drop = False
            if N not in post_taper_round_Ns:
                post_taper_round_Ns.append(N)
        for st in rnd["stages"]:
            c = st["conc"]; g = tuple(st["group"]); W = st["working_set"]
            for track in trks:
                secs = reps * eq_sweeps * N * model.predict_ns(track, W, c) / 1e9
                if (N, c, track, g) in completed:
                    cost_done += secs
                    continue
                cost_remaining += secs
                if before_drop:
                    t_to_drop += secs
                else:
                    post_taper_steps += reps * eq_sweeps * N

    # confidence-weighted calibration against real elapsed time
    scale = 1.0
    if elapsed is not None and cost_done > 0:
        obs_scale = elapsed / cost_done
        conf = cost_done / (cost_done + cost_remaining) if (cost_done + cost_remaining) else 0.0
        scale = (1 - conf) * 1.0 + conf * obs_scale
    raw_total = scale * cost_remaining
    if drop_N is not None and t_to_drop > 0:
        t_to_drop *= scale
    final_secs *= scale

    taper_workers = [(rnd["N"], rnd["ladder"][-1]) for rnd in rounds
                     if drop_N is not None and rnd["N"] >= drop_N and rnd["ladder"]]

    shown_total = smoother.update(raw_total) if smoother is not None else raw_total
    rho = model.residual(results, trks, eq_sweeps)
    return dict(
        t_total=shown_total, t_total_raw=raw_total,
        t_total_lo=shown_total * (1 - rho), t_total_hi=shown_total * (1 + rho), rho=rho,
        drop_N=drop_N, fill_c=fill_c,
        t_to_drop=(t_to_drop if (drop_N is not None and t_to_drop > 0) else None),
        final_N=final_N, final_c=final_c, final_secs=final_secs,
        t_to_final=max(0.0, raw_total - final_secs), max_conc=max_conc,
        taper_workers=taper_workers,
        post_taper_rounds=len(post_taper_round_Ns),
        post_taper_steps=post_taper_steps, calib_scale=scale,
    )
