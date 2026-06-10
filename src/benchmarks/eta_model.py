#!/usr/bin/env python3
"""
eta_model — ETA for the TIME-BOXED benchmark plan
=================================================
Every benchmark cell is wall-boxed (warm cap + fixed timed budget), so the
plan's total runtime is a SUM of known per-cell estimates, not a model. What
remains is bookkeeping: plan_eta scales the remaining estimated seconds by a
measured overhead correction (elapsed / estimated-done, clamped), and
EtaSmoother damps the displayed value. Unit-tested in
src/tests/test_eta_stability.py without running any benchmark.
"""


NOMINAL_DRAM = {"py": 12.0e6, "cpp": 24.0e6, "rust": 24.0e6}
# Per-worker steps/s, rough DRAM-resident floor, used ONLY before anything is
# measured (dry-run, first cells). The previous {2.5e6, 4e6} values were ~8x too
# pessimistic vs. observed DRAM-resident rates (~20-40 M steps/s), which made
# dry-run ETAs and early projections misleading.


# ───────────────────────── monotone surface fit ─────────────────────────
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


def plan_eta(total_est_s, done_est_s, elapsed_s, smoother=None):
    """ETA for a TIME-BOXED plan: every cell has a known second estimate, so
    the remaining time is (total_est − done_est) × correction, where the
    correction is simply measured: elapsed / done_est once enough of the plan
    has run (warm-up, JIT, pool overheads all land in it automatically).
    Smoothed for display. This replaces the rate-surface projection that the
    old fixed-step schedule needed — with wall-boxed measurements the ETA is
    arithmetic, not a model."""
    remaining = max(0.0, total_est_s - done_est_s)
    if remaining == 0.0:
        return smoother.update(0.0) if smoother is not None else 0.0
    corr = 1.0
    if done_est_s > 30.0 and elapsed_s > 0:
        corr = min(5.0, max(0.3, elapsed_s / done_est_s))
    raw = remaining * corr
    return smoother.update(raw) if smoother is not None else raw
