#!/usr/bin/env python3
"""Stability properties of the TIME-BOXED benchmark ETA (eta_model.plan_eta
+ EtaSmoother). The old rate-surface projection (project_milestones) died
with the flat schedule it modelled: a time-boxed plan makes the ETA
arithmetic — remaining estimated seconds × a measured overhead correction —
so what is left to test is exactly that arithmetic and its display
smoothing:

  1. determinism + sanity: never negative, zero when the plan is done,
     equals the remaining estimate before any correction data exists;
  2. correction: when reality runs k× slower than the per-cell estimates,
     the ETA converges to k× the naive remainder once >30 s of estimated
     work has completed (and the correction is clamped to [0.3, 5]);
  3. smoothing: cell-to-cell display swing through a noisy run stays small
     and monotone-decreasing in trend (no sawtooth), and the smoother
     resets on >4× discontinuities (resume) instead of slewing for ages.
"""
import os
import random
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from benchmarks.eta_model import EtaSmoother, plan_eta


def _run(total_est, per_cell_est, slowdown, jitter, smoother):
    """Simulate completing a plan where reality = est × slowdown × noise.
    Returns the sequence of displayed ETAs."""
    rng = random.Random(7)
    done_est = 0.0
    elapsed = 0.0
    out = []
    n = int(total_est / per_cell_est)
    for _ in range(n):
        done_est += per_cell_est
        elapsed += per_cell_est * slowdown * rng.uniform(1 - jitter,
                                                         1 + jitter)
        out.append(plan_eta(total_est, done_est, elapsed, smoother))
    return out


def main():
    # 1) sanity
    assert plan_eta(100.0, 0.0, 0.0) == 100.0
    assert plan_eta(100.0, 100.0, 250.0) == 0.0
    assert plan_eta(100.0, 120.0, 10.0) == 0.0          # over-complete clamps
    # before 30 s of estimated work is done, no correction is applied
    assert plan_eta(100.0, 10.0, 90.0) == 90.0

    # 2) correction converges to the true slowdown (1.8×), clamped range
    etas = _run(600.0, 2.0, slowdown=1.8, jitter=0.0, smoother=None)
    midpoint = etas[len(etas) // 2]
    naive_mid = 600.0 / 2                                # half the plan left
    assert abs(midpoint / (naive_mid * 1.8) - 1) < 0.02, midpoint
    assert plan_eta(100.0, 50.0, 50.0 * 50, None) == 50.0 * 5   # clamp hi
    assert plan_eta(100.0, 50.0, 1.0, None) == 50.0 * 0.3       # clamp lo

    # 3) smoothing: noisy run, small cell-to-cell swing, trending down
    sm = EtaSmoother(alpha=0.25)
    etas = _run(600.0, 2.0, slowdown=1.5, jitter=0.35, smoother=sm)
    # judge swing only while a meaningful amount of work remains — at the
    # endgame the ETA legitimately snaps toward 0 and any relative metric
    # explodes on a denominator a few seconds wide
    body = [(a, b) for a, b in zip(etas, etas[1:]) if a > 0.05 * 600.0]
    swings = [abs(b - a) / a for a, b in body]
    assert max(swings[5:]) < 0.20, max(swings[5:])
    assert etas[-1] < etas[len(etas) // 4], "ETA should trend down"
    # reset on a resume-style discontinuity
    sm2 = EtaSmoother()
    sm2.update(1000.0)
    assert sm2.update(50.0) == 50.0     # >4× drop snaps, doesn't slew

    print("  PASS: time-boxed plan ETA is sane, converges to the measured "
          "overhead, and the displayed value is stable (no sawtooth).")


if __name__ == "__main__":
    main()
