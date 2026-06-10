"""
ds4_search/shape_classify.py — d_s(t) curve-shape classifier (single source)
============================================================================
One heuristic that tags a d_s(t) curve by shape. Lives here, on its own, so
the live/headless sweep (ds4_search.sweep_runner) and the offline export
(ds4_search.static_dashboard) share *one* implementation instead of keeping
two hand-synced copies that silently drift apart.

Pure: numpy only, no project imports, no side effects. Import it as
`from ds4_search.shape_classify import classify_shape`.
"""

import numpy as np


def classify_shape(d_s_mean, in_window):
    """Heuristic shape tag for a d_s(t) curve.

    Operates on the in-window region only. Counts local maxima/minima
    after a small smoothing pass to suppress per-point noise. Returns
    one of:
       'flat'              — d_s varies by less than 0.5 across the window
                             (or featureless with std < 0.7)
       'monotone_rising'   — d_s increases >= 0.8 with no significant dip
       'monotone_falling'  — d_s decreases >= 0.8 with no significant rise
       'wobble'            — featureless but non-trivial variation (no sharp
                             extrema, but std >= 0.7 — meandering, not at one
                             dimension)
       'single_peak'       — one local maximum (optionally with one trough)
       'double_peak'       — two local maxima
       'bumpy'             — three or more local extrema
       'undetermined'      — too few in-window points
    """
    valid = in_window & np.isfinite(d_s_mean)
    if valid.sum() < 8:
        return "undetermined"
    y = d_s_mean[valid].copy()
    # Smooth by a 3-point box average to suppress one-point wiggles.
    if len(y) >= 5:
        y_smooth = np.convolve(y, np.ones(3) / 3, mode="valid")
    else:
        y_smooth = y
    # Find local maxima/minima (interior points only).
    n = len(y_smooth)
    peaks_idx = []
    troughs_idx = []
    for i in range(1, n - 1):
        if y_smooth[i] > y_smooth[i - 1] and y_smooth[i] > y_smooth[i + 1]:
            peaks_idx.append(i)
        if y_smooth[i] < y_smooth[i - 1] and y_smooth[i] < y_smooth[i + 1]:
            troughs_idx.append(i)

    span = float(y_smooth.max() - y_smooth.min())
    std_y = float(np.std(y))
    # Flat requires BOTH small total range AND small std — otherwise
    # a smoothly meandering curve passes the span check but is not
    # at one dimension.
    if span < 0.5 and std_y < 0.4:
        return "flat"

    # Ignore extremely shallow extrema (< 0.15 amplitude relative to
    # nearest neighbour). These are noise-level wiggles, not features.
    def amplitude(idx):
        # Compare to closest left and right neighbour values
        left = max(idx - 1, 0)
        right = min(idx + 1, n - 1)
        return min(abs(y_smooth[idx] - y_smooth[left]),
                   abs(y_smooth[idx] - y_smooth[right]))
    peaks = [i for i in peaks_idx if amplitude(i) >= 0.15]
    troughs = [i for i in troughs_idx if amplitude(i) >= 0.15]

    n_peaks = len(peaks)
    n_troughs = len(troughs)

    # Endpoint-based classification: did the curve net rise or fall?
    net = float(y_smooth[-1] - y_smooth[0])

    if n_peaks == 0 and n_troughs == 0:
        if net >= 0.8:
            return "monotone_rising"
        elif net <= -0.8:
            return "monotone_falling"
        elif std_y < 0.7:
            return "flat"
        else:
            # Featureless but non-trivial variation — meandering
            # without strong trend. Distinct from "flat" so the
            # dashboard's leaderboard isn't misled by a tag that
            # implies a single-dimension regime when there isn't one.
            return "wobble"
    if n_peaks == 1 and n_troughs == 0:
        return "single_peak"
    if n_peaks == 1 and n_troughs == 1:
        return "single_peak"
    if n_peaks == 2:
        return "double_peak"
    return "bumpy"
