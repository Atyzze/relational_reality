"""
metrics.size_extrapolation — Finite-size extrapolation to d_s_∞
==============================================================
extrapolate_asymptote fits d_s(N) = d_s_∞ + α · N^(-β) by weighted
nonlinear least squares with three regimes (flat-data, 3-param,
2-param-fixed-β) for robustness on noisy real data.
"""

import math
import numpy as np

def extrapolate_asymptote(N_values, ds_values, ci95_values,
                          hi_N_boost: float = 0.0,
                          fix_beta: float = None,
                          d_s_guess: float = None):
    """Fit d_s(N) = d_s_∞ + α × N^(-β) with weighted nonlinear LS.

    This is the standard finite-size-scaling extrapolation: measurements
    at a range of N are fit to remove the leading finite-size bias,
    producing a more accurate estimate of d_s(N → ∞) than any single-N
    point.

    Three-regime logic:
      1. FLAT DATA — if all measurements agree within their CIs (χ² of
         a constant fit is ≤ 2·dof), there is no N-trend to extrapolate.
         We return the inverse-variance-weighted mean with its SE as
         the CI. This is the correct answer: d_s(N) is already
         constant, so d_∞ equals that constant.

      2. TRENDED DATA — 3-parameter fit of d_∞ + α·N^(-β). If β lands
         at a physical bound (< 0.05 or > 3.0) or the fit covariance
         explodes (|ci95_inf| > 0.5·|d_∞|), the fit is ill-conditioned
         — fall back to regime 3.

      3. ILL-CONDITIONED — fix β at 2/d_s_guess (the theoretical lattice
         finite-size scaling exponent), re-fit with only 2 free params
         (d_∞ and α). d_s_guess defaults to the largest-N measurement.

    Parameters
    ----------
    N_values, ds_values, ci95_values : array-like
        Graph sizes, d_s estimates, and ±95% CI half-widths.
    hi_N_boost : float, default 0
        Extra weight toward higher N: multiply 1/σ² weights by
        (N / N_max)^hi_N_boost. 0 = pure precision weighting,
        0.5-1.0 = progressively favour the highest-N points.
    fix_beta : float or None
        If given, skip regime selection and run 2-parameter fit with
        β fixed to this value (used internally for fallback).
    d_s_guess : float or None
        Used to compute the natural lattice β = 2/d_s_guess in regime 3.
        Defaults to largest-N measurement.

    Returns
    -------
    dict with keys:
      d_inf, ci95_inf         — extrapolated d_s_∞ and ±95% CI
      alpha, beta             — correction coefficients (NaN if flat)
      d_inf_se                — 1-σ SE on d_s_∞
      n_points                — number of valid (N, d_s) points used
      chi2, chi2_per_dof      — goodness-of-fit
      regime                  — which branch produced the answer:
                                'flat_weighted_mean' | 'fit_3param' |
                                'fit_2param_fixbeta'
      reason                  — failure reason if d_inf is nan
    """
    try:
        import scipy.optimize
    except ImportError:
        return {'d_inf': math.nan, 'ci95_inf': math.nan,
                'reason': 'scipy.optimize not available'}

    N = np.asarray(N_values, dtype=np.float64)
    ds = np.asarray(ds_values, dtype=np.float64)
    ci = np.asarray(ci95_values, dtype=np.float64)

    mask = (np.isfinite(ds) & np.isfinite(ci) & (ci > 0) &
            np.isfinite(N) & (N > 0))
    N, ds, ci = N[mask], ds[mask], ci[mask]

    if len(N) < 3:
        return {'d_inf': math.nan, 'ci95_inf': math.nan,
                'reason': f'need ≥3 valid points, have {len(N)}',
                'n_points': len(N)}

    order = np.argsort(N)
    N, ds, ci = N[order], ds[order], ci[order]
    sigma = ci / 1.96

    # Effective weights (optionally boost high-N)
    if hi_N_boost > 0:
        boost = (N / N.max()) ** (hi_N_boost / 2.0)
        sigma_fit = sigma / boost
    else:
        sigma_fit = sigma
    weights = 1.0 / (sigma_fit ** 2)

    # ═══════════════════════════════════════════════════════════════
    # Regime 1: FLAT DATA check
    # ═══════════════════════════════════════════════════════════════
    # Weighted mean and its standard error
    wmean = float((weights * ds).sum() / weights.sum())
    wmean_se = float(1.0 / math.sqrt(weights.sum()))

    # χ² of constant-model fit
    chi2_const = float(np.sum(weights * (ds - wmean) ** 2))
    dof_const = max(len(N) - 1, 1)
    chi2_const_per_dof = chi2_const / dof_const

    # MONOTONIC-TREND OVERRIDE:
    # Even when χ²_const is statistically "fine" (≤ 2·dof), a clear
    # monotonic trend in d_s vs N is the signature of physical finite-
    # size scaling — NOT noise. We should trust the trend and run the
    # full N→∞ fit instead of averaging. This is the physics consider-
    # ation that χ² alone misses: lattice corrections are *deterministic*
    # functions of N, not random fluctuations.
    #
    # Detection: Spearman rank correlation between log(N) and d_s.
    # This is robust to a single noisy point that breaks strict
    # monotonicity (which is what realistic data always has — a single
    # noise blip or anomalous small-N point shouldn't disqualify the
    # whole sweep). |rho| ≥ 0.7 with ≥4 points is a strong trend
    # signal. Combined with span > 1.5·σ_min, this catches lattice
    # finite-size scaling without false positives on noise.
    log_N = np.log(N)
    rank_logN = np.argsort(np.argsort(log_N)).astype(np.float64)
    rank_ds = np.argsort(np.argsort(ds)).astype(np.float64)
    n_pts = len(N)
    if n_pts >= 2:
        # Pearson correlation of ranks = Spearman correlation
        rmean = (n_pts - 1) / 2.0
        cov_r = np.mean((rank_logN - rmean) * (rank_ds - rmean))
        var_r = np.mean((rank_logN - rmean) ** 2)
        spearman_rho = cov_r / var_r if var_r > 0 else 0.0
    else:
        spearman_rho = 0.0
    span = float(ds.max() - ds.min())
    has_trend_signal = (abs(spearman_rho) >= 0.7
                        and span > float(sigma.min()) * 1.5
                        and n_pts >= 4)

    # Data is flat if the measurements are consistent with a single
    # value within their CIs AND there's no monotonic trend signal.
    if (fix_beta is None and chi2_const_per_dof <= 2.0
            and not has_trend_signal):
        # Inflate CI by max(sigma/1.96, wmean_se) to not claim better
        # than any individual measurement
        ci95_inf = max(1.96 * wmean_se, float(ci.min()))
        return {
            'd_inf': wmean,
            'ci95_inf': ci95_inf,
            'alpha': 0.0, 'beta': math.nan,
            'd_inf_se': wmean_se,
            'n_points': int(len(N)),
            'chi2': chi2_const,
            'chi2_per_dof': chi2_const_per_dof,
            'regime': 'flat_weighted_mean',
            'note': 'data flat across N (no trend), using weighted mean',
        }

    # ═══════════════════════════════════════════════════════════════
    # Regime 2: TRENDED DATA — 3-parameter fit
    # ═══════════════════════════════════════════════════════════════
    d_inf_0 = float(ds[-1])           # largest-N point as starting guess
    alpha_0 = float(ds[0] - d_inf_0)  # bias at smallest N
    beta_0 = 0.5

    if fix_beta is not None:
        def model(N, d_inf, alpha):
            return d_inf + alpha * N ** (-float(fix_beta))
        p0 = [d_inf_0, alpha_0]
        # Wider α bounds (was ±30) — when β is fixed and the trend is
        # strong, α can legitimately need to be larger to track the
        # data at small N. Capping α at 30 was forcing fits to settle
        # on suboptimal d_∞ estimates.
        bounds = ([0.05, -1000], [15.0, 1000])
        n_params = 2
    else:
        def model(N, d_inf, alpha, beta):
            return d_inf + alpha * N ** (-beta)
        p0 = [d_inf_0, alpha_0, beta_0]
        # PHYSICAL β bounds: [0.05, 3.0] — the original [0.01, 5.0] let
        # scipy settle β→0 for flat data (giving degenerate fits with
        # nonsensical d_∞). β < 0.05 is unphysical (correction decays
        # slower than 1/N^0.05 means essentially constant over any
        # reasonable N range); β > 3 is unphysical (correction vanishes
        # before any lattice effect could matter).
        # Wider α bounds (was ±30) for the same reason as the 2-param
        # case: strong corrections at small N can need large α.
        bounds = ([0.05, -1000, 0.05], [15.0, 1000, 3.0])
        n_params = 3

    try:
        popt, pcov = scipy.optimize.curve_fit(
            model, N, ds,
            p0=p0, sigma=sigma_fit, absolute_sigma=True,
            bounds=bounds, maxfev=10000,
        )
    except Exception as e:
        if fix_beta is None:
            # Fallback to regime 3 with β = 2/d_s_guess
            guess = d_s_guess if d_s_guess is not None else d_inf_0
            fallback_beta = 2.0 / max(abs(guess), 0.5)
            fb = extrapolate_asymptote(N_values, ds_values, ci95_values,
                                       hi_N_boost=hi_N_boost,
                                       fix_beta=fallback_beta,
                                       d_s_guess=d_s_guess)
            if not math.isnan(fb['d_inf']):
                fb['regime'] = 'fit_2param_fixbeta'
                fb['note'] = (fb.get('note', '') +
                              f' (3-param fit raised: {str(e)[:60]})')
                return fb
        return {'d_inf': math.nan, 'ci95_inf': math.nan,
                'reason': f'curve_fit failed: {str(e)[:120]}',
                'n_points': int(len(N))}

    if fix_beta is not None:
        d_inf, alpha = popt
        beta = float(fix_beta)
        d_inf_se = float(math.sqrt(pcov[0, 0])) if pcov is not None else math.nan
        regime = 'fit_2param_fixbeta'
    else:
        d_inf, alpha, beta = popt
        d_inf_se = float(math.sqrt(pcov[0, 0])) if pcov is not None else math.nan

        # Degeneracy check: β at a bound, or d_∞ CI blown up (>50% of
        # |d_∞|). Both signal that the 3-param fit is ill-conditioned
        # — switch to fixed-β fallback.
        beta_at_bound = (beta <= 0.06) or (beta >= 2.95)
        ci_blown = (math.isfinite(d_inf_se) and
                    d_inf_se * 1.96 > 0.5 * abs(d_inf))
        if beta_at_bound or ci_blown:
            guess = d_s_guess if d_s_guess is not None else d_inf_0
            fallback_beta = 2.0 / max(abs(guess), 0.5)
            fb = extrapolate_asymptote(N_values, ds_values, ci95_values,
                                       hi_N_boost=hi_N_boost,
                                       fix_beta=fallback_beta,
                                       d_s_guess=d_s_guess)
            if not math.isnan(fb['d_inf']):
                reason = ("β hit bound" if beta_at_bound
                          else "d_∞ CI > 50% of |d_∞|")
                fb['note'] = (f'fell back from 3-param ({reason}, '
                              f'β was {beta:.3f}); using β=2/d_s_guess'
                              f'={fallback_beta:.3f}')
                return fb
        regime = 'fit_3param'

    # χ² of the actual fit
    resid = ds - model(N, *popt)
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = max(len(N) - n_params, 1)
    chi2_per_dof = chi2 / dof

    # BIRGE-RATIO CI INFLATION:
    # When χ²/dof > 1, the residuals are larger than the per-point
    # CIs predict — meaning either the model is wrong or the input CIs
    # are underestimated. The standard metrology fix (Birge ratio) is
    # to scale the reported uncertainty by √(χ²/dof). This makes the
    # reported CI honest about model misfit instead of falsely claiming
    # high precision when the model can't track the data.
    #
    # We only inflate (never deflate): if χ²/dof < 1, the per-point CIs
    # were probably conservative and we keep the formal CI.
    birge_factor = math.sqrt(max(1.0, chi2_per_dof))
    d_inf_se_inflated = d_inf_se * birge_factor

    return {
        'd_inf': float(d_inf),
        'ci95_inf': float(1.96 * d_inf_se_inflated) if math.isfinite(d_inf_se) else math.nan,
        'alpha': float(alpha),
        'beta': float(beta),
        'd_inf_se': d_inf_se_inflated,
        'n_points': int(len(N)),
        'chi2': chi2,
        'chi2_per_dof': chi2_per_dof,
        'regime': regime,
        'note': (f'CI inflated by √χ²/dof = {birge_factor:.2f}'
                 if birge_factor > 1.05 else ''),
    }
