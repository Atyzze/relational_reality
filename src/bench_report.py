#!/usr/bin/env python3
"""
bench_report.py — HTML report for the worker-count / CPU-memory benchmark
=========================================================================
Pure presentation + light stats. **No numpy / numba / scipy** so it imports
instantly and can render a report (or a synthetic demo) without the heavy
compute stack — `bench_workers.py` does the measuring and hands the numbers
here.

It turns three data products into one self-contained dark-theme HTML file
(inline SVG, no external assets):

  1. throughput sweep   — cells/min ("universes/min") vs worker count, with the
     peak marked and an ideal-linear-scaling reference, so the optimal worker
     count and the memory-bandwidth knee are visible at a glance.
  2. per-core profile   — median time-per-cell for each logical CPU, grouped and
     coloured by CCD. On a part with asymmetric L3 (e.g. a 9950X3D: one CCD
     with 3D V-Cache, one without) the larger-cache cores finish a
     memory-bound cell systematically faster, and that gap is the headline.
  3. per-cell scatter   — every measured cell (seed + wall time + the CPU it ran
     on), coloured by CCD, so seed-to-seed spread and any two-CCD bimodality
     show up directly.

The data contracts (plain dicts/lists, all JSON-safe) are documented on
`write_html_report`. `synthetic_demo()` fabricates a plausible dual-CCD dataset
for tests and for `optimal_workers.py --demo`.
"""

import html as _html
import math
import os
import time

# ── Dark theme, matched to the project's other self-contained SVG/HTML
#    (flow_probe._render_svg, the dashboards): same bg, mono font, palette. ──
BG = "#0a0a0e"
PANEL = "#15151b"
GRID = "#2a2a35"
FRAME = "#444"
FG = "#ddd"
MUTE = "#888"
ACCENT = "#7ec9ff"
GOOD = "#7ec96e"
WARN = "#e5b06b"
# CCD hues: V-Cache CCD vs the rest. Extra entries cycle for >2 CCDs.
CCD_COLORS = ["#4ec9b0", "#c586c0", "#dcdcaa", "#9cdcfe", "#ce9178", "#b5cea8"]
VCACHE_COLOR = "#7ec96e"   # always green for the V-Cache CCD


# ════════════════════════════════════════════════════════════════════
#  Small stats + formatting helpers (stdlib only)
# ════════════════════════════════════════════════════════════════════
def _pct(sorted_vals, q):
    """Linear-interpolated percentile q∈[0,1] of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0)


def fmt_bytes(b):
    """Bytes → '96 MiB' / '32 MiB' / '1.0 MiB' / '512 KiB'."""
    if b is None:
        return "?"
    b = float(b)
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if b >= scale:
            v = b / scale
            return f"{v:.0f} {unit}" if v >= 10 or v == int(v) else f"{v:.1f} {unit}"
    return f"{int(b)} B"


def fmt_hms(seconds):
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _esc(s):
    return _html.escape(str(s))


# ════════════════════════════════════════════════════════════════════
#  SVG chart builders
# ════════════════════════════════════════════════════════════════════
_PW, _PH = 940, 360
_ML, _MR, _MT, _MB = 70, 120, 36, 52


def _svg_head(width=_PW, height=_PH):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="font-family: ui-monospace, monospace; background:{BG}; '
            f'width:100%; height:auto; max-width:{width}px;">')


def _frame(x, y, w, h):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PANEL}" '
            f'stroke="{FRAME}" stroke-width="1"/>')


def throughput_chart_svg(summary):
    """Line chart: cells/min vs worker count (log2 x), the peak ringed, plus a
    dashed 'ideal linear scaling' reference (baseline × W). The gap between the
    actual curve and the ideal line IS the contention / bandwidth loss."""
    if not summary:
        return "<p style='color:#888'>no throughput data</p>"
    rows = sorted(summary, key=lambda r: r["workers"])
    ws = [r["workers"] for r in rows]
    cpms = [r["cells_per_min"] for r in rows]
    base = next((r["cells_per_min"] for r in rows if r["workers"] == 1), cpms[0]) or 1e-9
    base_w = next((r["workers"] for r in rows if r["workers"] == 1), ws[0])
    ideal = [base * (w / base_w) for w in ws]
    peak_i = max(range(len(rows)), key=lambda i: cpms[i])

    plot_w = _PW - _ML - _MR
    plot_h = _PH - _MT - _MB
    lxmin = math.log2(min(ws)) if min(ws) > 0 else 0.0
    lxmax = math.log2(max(ws)) if max(ws) > 0 else 1.0
    if lxmax <= lxmin:
        lxmax = lxmin + 1.0
    ymax = max(max(cpms), max(ideal)) * 1.08
    ymax = ymax if ymax > 0 else 1.0

    def X(w):
        return _ML + (math.log2(w) - lxmin) / (lxmax - lxmin) * plot_w
    def Y(v):
        return _MT + (1 - v / ymax) * plot_h

    s = [_svg_head(), _frame(_ML, _MT, plot_w, plot_h)]
    s.append(f'<text x="{_PW/2}" y="20" fill="{FG}" font-size="13" '
             f'text-anchor="middle" font-weight="bold">Throughput vs worker '
             f'count — cells/min ("universes/min")</text>')
    # y gridlines
    for f in range(0, 6):
        yv = ymax * f / 5.0
        yy = Y(yv)
        s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yy:.1f}" y2="{yy:.1f}" '
                 f'stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{_ML-8}" y="{yy+4:.1f}" fill="{MUTE}" font-size="10" '
                 f'text-anchor="end">{yv:.0f}</text>')
    # x ticks at each measured worker count
    for w in ws:
        xx = X(w)
        s.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{_MT}" y2="{_MT+plot_h}" '
                 f'stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{xx:.1f}" y="{_MT+plot_h+15}" fill="{MUTE}" '
                 f'font-size="10" text-anchor="middle">{w}</text>')
    s.append(f'<text x="{_ML+plot_w/2}" y="{_MT+plot_h+38}" fill="{MUTE}" '
             f'font-size="11" text-anchor="middle">worker processes (log₂)</text>')
    s.append(f'<text x="18" y="{_MT+plot_h/2}" fill="{MUTE}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 18 {_MT+plot_h/2})">'
             f'cells / min</text>')
    # ideal line (dashed)
    idl = " ".join(f"{'M' if i==0 else 'L'}{X(ws[i]):.1f},{Y(ideal[i]):.1f}"
                   for i in range(len(ws)))
    s.append(f'<path d="{idl}" fill="none" stroke="{MUTE}" stroke-width="1" '
             f'stroke-dasharray="5,4" opacity="0.7"/>')
    # actual line
    act = " ".join(f"{'M' if i==0 else 'L'}{X(ws[i]):.1f},{Y(cpms[i]):.1f}"
                   for i in range(len(ws)))
    s.append(f'<path d="{act}" fill="none" stroke="{ACCENT}" stroke-width="2.2"/>')
    for i, w in enumerate(ws):
        r = 5 if i == peak_i else 3
        col = GOOD if i == peak_i else ACCENT
        s.append(f'<circle cx="{X(w):.1f}" cy="{Y(cpms[i]):.1f}" r="{r}" '
                 f'fill="{col}" stroke="{BG}" stroke-width="1"/>')
        if i == peak_i:
            s.append(f'<circle cx="{X(w):.1f}" cy="{Y(cpms[i]):.1f}" r="9" '
                     f'fill="none" stroke="{GOOD}" stroke-width="1.5"/>')
            s.append(f'<text x="{X(w):.1f}" y="{Y(cpms[i])-14:.1f}" fill="{GOOD}" '
                     f'font-size="11" text-anchor="middle" font-weight="bold">'
                     f'peak {cpms[i]:.0f} @ {w}w</text>')
    # legend
    lx = _ML + plot_w + 14
    s.append(f'<line x1="{lx}" x2="{lx+18}" y1="{_MT+14}" y2="{_MT+14}" '
             f'stroke="{ACCENT}" stroke-width="2.2"/>')
    s.append(f'<text x="{lx+24}" y="{_MT+18}" fill="{FG}" font-size="10">measured</text>')
    s.append(f'<line x1="{lx}" x2="{lx+18}" y1="{_MT+32}" y2="{_MT+32}" '
             f'stroke="{MUTE}" stroke-width="1" stroke-dasharray="5,4"/>')
    s.append(f'<text x="{lx+24}" y="{_MT+36}" fill="{FG}" font-size="10">ideal ∝W</text>')
    s.append(f'<text x="{lx}" y="{_MT+62}" fill="{MUTE}" font-size="9.5">'
             f'gap = cache /</text>')
    s.append(f'<text x="{lx}" y="{_MT+75}" fill="{MUTE}" font-size="9.5">'
             f'bandwidth loss</text>')
    s.append("</svg>")
    return "\n".join(s)


def percore_chart_svg(percore, topology):
    """Bar per logical CPU: median seconds/cell (lower = faster), ordered by CPU
    id, coloured by CCD, V-Cache CCD in green. Dashed line at the overall median.
    This is the view where a V-Cache CCD's lower memory latency shows up."""
    if not percore:
        return "<p style='color:#888'>no per-core data (run with --per-core)</p>"
    cores = sorted(percore.keys())
    meds = [percore[c]["median_s"] for c in cores]
    p10s = [percore[c].get("p10_s", percore[c]["median_s"]) for c in cores]
    p90s = [percore[c].get("p90_s", percore[c]["median_s"]) for c in cores]
    vcache_ids = set(topology.get("vcache_ccd_ids", []))

    def ccd_of(cpu):
        for c in topology.get("ccds", []):
            if cpu in c["cpus"]:
                return c["id"]
        return None

    plot_w = _PW - _ML - _MR
    plot_h = _PH - _MT - _MB
    ymax = (max(p90s) if p90s else max(meds)) * 1.10 or 1.0
    n = len(cores)
    bw = plot_w / max(n, 1)

    def Y(v):
        return _MT + (1 - v / ymax) * plot_h

    s = [_svg_head(), _frame(_ML, _MT, plot_w, plot_h)]
    s.append(f'<text x="{_PW/2}" y="20" fill="{FG}" font-size="13" '
             f'text-anchor="middle" font-weight="bold">Per-core latency — median '
             f'seconds per cell (lower is faster)</text>')
    for f in range(0, 6):
        yv = ymax * f / 5.0
        yy = Y(yv)
        s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yy:.1f}" y2="{yy:.1f}" '
                 f'stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{_ML-8}" y="{yy+4:.1f}" fill="{MUTE}" font-size="10" '
                 f'text-anchor="end">{yv:.1f}</text>')
    overall_med = _median(meds)
    yo = Y(overall_med)
    s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yo:.1f}" y2="{yo:.1f}" '
             f'stroke="{WARN}" stroke-width="1" stroke-dasharray="6,4" opacity="0.8"/>')
    s.append(f'<text x="{_ML+plot_w-4}" y="{yo-4:.1f}" fill="{WARN}" font-size="10" '
             f'text-anchor="end">overall median {overall_med:.2f}s</text>')
    for i, c in enumerate(cores):
        x = _ML + i * bw
        cid = ccd_of(c)
        is_v = cid in vcache_ids
        col = VCACHE_COLOR if is_v else CCD_COLORS[(cid or 0) % len(CCD_COLORS)]
        h = (plot_h) * (meds[i] / ymax)
        s.append(f'<rect x="{x+bw*0.12:.1f}" y="{Y(meds[i]):.1f}" '
                 f'width="{bw*0.76:.1f}" height="{h:.1f}" fill="{col}" '
                 f'opacity="0.85"/>')
        # p10–p90 whisker
        s.append(f'<line x1="{x+bw*0.5:.1f}" x2="{x+bw*0.5:.1f}" '
                 f'y1="{Y(p90s[i]):.1f}" y2="{Y(p10s[i]):.1f}" '
                 f'stroke="{FG}" stroke-width="1" opacity="0.5"/>')
        if n <= 40:
            s.append(f'<text x="{x+bw*0.5:.1f}" y="{_MT+plot_h+13}" fill="{MUTE}" '
                     f'font-size="8.5" text-anchor="middle">{c}</text>')
    s.append(f'<text x="{_ML+plot_w/2}" y="{_MT+plot_h+38}" fill="{MUTE}" '
             f'font-size="11" text-anchor="middle">logical CPU id</text>')
    s.append(f'<text x="18" y="{_MT+plot_h/2}" fill="{MUTE}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 18 {_MT+plot_h/2})">'
             f'median s / cell</text>')
    # legend: one swatch per CCD
    ly = _MT + 12
    lx = _ML + plot_w + 14
    for c in topology.get("ccds", []):
        is_v = c["id"] in vcache_ids
        col = VCACHE_COLOR if is_v else CCD_COLORS[c["id"] % len(CCD_COLORS)]
        s.append(f'<rect x="{lx}" y="{ly-8}" width="12" height="10" fill="{col}"/>')
        tag = "V-Cache" if is_v else f"CCD{c['id']}"
        s.append(f'<text x="{lx+16}" y="{ly+1}" fill="{FG}" font-size="9.5">'
                 f'{tag} · {fmt_bytes(c.get("l3_bytes"))} L3</text>')
        ly += 16
    s.append("</svg>")
    return "\n".join(s)


def scatter_svg(per_cell, topology):
    """Every cell from the throughput phase: x = completion order, y = wall_s,
    point coloured by the CCD it ran on. Two CCDs with different cache → two
    horizontal bands."""
    pts = [r for r in per_cell if r.get("wall_s")]
    if not pts:
        return "<p style='color:#888'>no per-cell data</p>"
    vcache_ids = set(topology.get("vcache_ccd_ids", []))

    def ccd_of(cpu):
        for c in topology.get("ccds", []):
            if cpu is not None and cpu in c["cpus"]:
                return c["id"]
        return None

    walls = [r["wall_s"] for r in pts]
    ymax = max(walls) * 1.10 or 1.0
    ymin = min(walls) * 0.92
    plot_w = _PW - _ML - _MR
    plot_h = _PH - _MT - _MB
    n = len(pts)

    def X(i):
        return _ML + (i / max(n - 1, 1)) * plot_w
    def Y(v):
        return _MT + (1 - (v - ymin) / (ymax - ymin)) * plot_h

    s = [_svg_head(), _frame(_ML, _MT, plot_w, plot_h)]
    s.append(f'<text x="{_PW/2}" y="20" fill="{FG}" font-size="13" '
             f'text-anchor="middle" font-weight="bold">Per-cell wall time — '
             f'every measured cell, coloured by CCD</text>')
    for f in range(0, 6):
        yv = ymin + (ymax - ymin) * f / 5.0
        yy = Y(yv)
        s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yy:.1f}" y2="{yy:.1f}" '
                 f'stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{_ML-8}" y="{yy+4:.1f}" fill="{MUTE}" font-size="10" '
                 f'text-anchor="end">{yv:.1f}</text>')
    for i, r in enumerate(pts):
        cid = ccd_of(r.get("cpu"))
        is_v = cid in vcache_ids
        col = VCACHE_COLOR if is_v else CCD_COLORS[(cid or 0) % len(CCD_COLORS)]
        s.append(f'<circle cx="{X(i):.1f}" cy="{Y(r["wall_s"]):.1f}" r="2.6" '
                 f'fill="{col}" opacity="0.75"/>')
    s.append(f'<text x="{_ML+plot_w/2}" y="{_MT+plot_h+38}" fill="{MUTE}" '
             f'font-size="11" text-anchor="middle">cell completion order</text>')
    s.append(f'<text x="18" y="{_MT+plot_h/2}" fill="{MUTE}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 18 {_MT+plot_h/2})">'
             f'wall s / cell</text>')
    ly = _MT + 12
    lx = _ML + plot_w + 14
    seen = sorted({ccd_of(r.get("cpu")) for r in pts}, key=lambda x: (x is None, x))
    for cid in seen:
        is_v = cid in vcache_ids
        col = VCACHE_COLOR if is_v else CCD_COLORS[(cid or 0) % len(CCD_COLORS)]
        s.append(f'<circle cx="{lx+5}" cy="{ly-3}" r="4" fill="{col}"/>')
        tag = "V-Cache" if is_v else (f"CCD{cid}" if cid is not None else "?")
        s.append(f'<text x="{lx+14}" y="{ly}" fill="{FG}" font-size="9.5">{tag}</text>')
        ly += 16
    s.append("</svg>")
    return "\n".join(s)


# ════════════════════════════════════════════════════════════════════
#  V-Cache comparison: median speedup of the big-L3 CCD vs the rest
# ════════════════════════════════════════════════════════════════════
def vcache_verdict(percore, topology):
    """From the per-core medians, compare the V-Cache CCD(s) against the
    non-V-Cache CCD(s). Returns a dict (or None if there's nothing to compare).
    """
    if not percore:
        return None
    vids = set(topology.get("vcache_ccd_ids", []))
    if not vids:
        return None

    def ccd_of(cpu):
        for c in topology.get("ccds", []):
            if cpu in c["cpus"]:
                return c["id"]
        return None

    v_meds, o_meds = [], []
    for cpu, stat in percore.items():
        (v_meds if ccd_of(cpu) in vids else o_meds).append(stat["median_s"])
    if not v_meds or not o_meds:
        return None
    v_med = _median(v_meds)
    o_med = _median(o_meds)
    # lower s/cell is faster → speedup of V-Cache over the rest
    speedup = (o_med / v_med) if v_med > 0 else float("nan")
    return {"vcache_median_s": v_med, "other_median_s": o_med,
            "speedup": speedup, "pct_faster": (speedup - 1.0) * 100.0,
            "n_vcache_cores": len(v_meds), "n_other_cores": len(o_meds)}


# ════════════════════════════════════════════════════════════════════
#  The report
# ════════════════════════════════════════════════════════════════════
def detect_cache_levels(points, rise=0.18):
    """From one core's per-N points (each with working_set_bytes and
    per_node_probe_s, any order), find the working-set sizes where per-node
    probe time steps UP — the effective cache-level boundaries this workload
    sees. Returns knees: {boundary_bytes, ratio, from_per_node, to_per_node}.

    `boundary_bytes` is the largest working set that still fit the faster level
    ≈ that level's effective capacity. `rise` is the minimum fractional jump
    over the level's best that counts as a knee. Deliberately conservative:
    the transitions are soft (irregular access + prefetchers), so these are
    approximate, workload-relative boundaries — not exact datasheet sizes."""
    pts = sorted((p for p in points
                  if p.get("per_node_probe_s")
                  and p["per_node_probe_s"] == p["per_node_probe_s"]
                  and p["per_node_probe_s"] > 0),
                 key=lambda p: p["working_set_bytes"])
    if len(pts) < 3:
        return []
    knees = []
    base = pts[0]["per_node_probe_s"]
    for i in range(1, len(pts)):
        t = pts[i]["per_node_probe_s"]
        if t < base:                       # faster: still amortising overhead
            base = t
            continue
        if t >= base * (1.0 + rise):       # sustained step up → cache boundary
            knees.append({"boundary_bytes": pts[i - 1]["working_set_bytes"],
                          "ratio": t / base, "from_per_node": base,
                          "to_per_node": t})
            base = t
    return knees


def cache_map_chart_svg(cache_data, topology):
    """Per-node probe time (µs/node) vs working-set size on a log x-axis, one
    line per profiled core, coloured by CCD. Detected knees (dashed, in the
    core's colour) and the datasheet L2/L3 sizes (faint) are overlaid so the
    measured effective hierarchy sits right next to the spec."""
    series = [(c, cache_data[c]) for c in sorted(cache_data) if cache_data[c]]
    if not series:
        return ("<p style='color:#888'>no cache-map data "
                "(run with --map-cache)</p>")
    vids = set(topology.get("vcache_ccd_ids", []))
    cpus_info = topology.get("cpus", {}) or {}

    def ccd_of(cpu):
        for c in topology.get("ccds", []):
            if cpu in c["cpus"]:
                return c["id"]
        return None

    def color_of(cpu):
        cid = ccd_of(cpu)
        return (VCACHE_COLOR if cid in vids
                else CCD_COLORS[(cid or 0) % len(CCD_COLORS)])

    all_ws = [p["working_set_bytes"] for _, pts in series for p in pts]
    all_y = [p["per_node_probe_s"] * 1e6 for _, pts in series for p in pts
             if p.get("per_node_probe_s")]
    spec_ws = [c["l3_bytes"] for c in topology.get("ccds", []) if c.get("l3_bytes")]
    for cpu, _pts in series:
        info = cpus_info.get(cpu) or cpus_info.get(str(cpu))
        if info and info.get("l2"):
            spec_ws.append(info["l2"])
    wsmin = min(all_ws)
    wsmax = max(all_ws + spec_ws) if spec_ws else max(all_ws)
    lxmin, lxmax = math.log10(wsmin), math.log10(wsmax)
    if lxmax <= lxmin:
        lxmax = lxmin + 1.0
    ymax = (max(all_y) * 1.12) if all_y else 1.0
    plot_w = _PW - _ML - _MR
    plot_h = _PH - _MT - _MB

    def X(ws):
        return _ML + (math.log10(ws) - lxmin) / (lxmax - lxmin) * plot_w

    def Y(v):
        return _MT + (1 - v / ymax) * plot_h

    s = [_svg_head(), _frame(_ML, _MT, plot_w, plot_h)]
    s.append(f'<text x="{_PW/2}" y="20" fill="{FG}" font-size="13" '
             f'text-anchor="middle" font-weight="bold">Effective cache '
             f'hierarchy — per-node probe time vs working set</text>')
    for f in range(0, 6):
        yv = ymax * f / 5.0
        yy = Y(yv)
        s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"'
                 f' stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{_ML-8}" y="{yy+4:.1f}" fill="{MUTE}" font-size="10"'
                 f' text-anchor="end">{yv:.2f}</text>')
    j0 = int(math.floor(math.log2(wsmin)))
    j1 = int(math.ceil(math.log2(wsmax)))
    for j in range(j0, j1 + 1):
        ws = 2.0 ** j
        if ws < wsmin or ws > wsmax:
            continue
        xx = X(ws)
        s.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{_MT}" y2="{_MT+plot_h}"'
                 f' stroke="{GRID}" stroke-width="0.5"/>')
        if j % 2 == 0:
            s.append(f'<text x="{xx:.1f}" y="{_MT+plot_h+15}" fill="{MUTE}" '
                     f'font-size="9" text-anchor="middle">{fmt_bytes(ws)}</text>')
    s.append(f'<text x="{_ML+plot_w/2}" y="{_MT+plot_h+38}" fill="{MUTE}" '
             f'font-size="11" text-anchor="middle">working set (log scale)'
             f'</text>')
    s.append(f'<text x="16" y="{_MT+plot_h/2}" fill="{MUTE}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 16 {_MT+plot_h/2})">'
             f'µs / node (probe)</text>')
    # datasheet overlays (faint vertical lines)
    drawn = set()
    for cpu, _pts in series:
        cid = ccd_of(cpu)
        col = color_of(cpu)
        l3 = next((c["l3_bytes"] for c in topology.get("ccds", [])
                   if c["id"] == cid), None)
        if l3 and ("L3", cid) not in drawn and wsmin <= l3 <= wsmax:
            drawn.add(("L3", cid))
            x = X(l3)
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{_MT}" '
                     f'y2="{_MT+plot_h}" stroke="{col}" stroke-width="1" '
                     f'stroke-dasharray="2,3" opacity="0.45"/>')
            s.append(f'<text x="{x+3:.1f}" y="{_MT+12}" fill="{col}" '
                     f'font-size="9" opacity="0.8">L3 {fmt_bytes(l3)} spec</text>')
        info = cpus_info.get(cpu) or cpus_info.get(str(cpu))
        l2 = info.get("l2") if info else None
        if l2 and ("L2", l2) not in drawn and wsmin <= l2 <= wsmax:
            drawn.add(("L2", l2))
            x = X(l2)
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{_MT}" '
                     f'y2="{_MT+plot_h}" stroke="{MUTE}" stroke-width="1" '
                     f'stroke-dasharray="2,3" opacity="0.4"/>')
            s.append(f'<text x="{x+3:.1f}" y="{_MT+plot_h-6}" fill="{MUTE}" '
                     f'font-size="9">L2 {fmt_bytes(l2)} spec</text>')
    # series + detected knees
    flip = 0
    for cpu, pts in series:
        col = color_of(cpu)
        pth = " ".join(
            f"{'M' if i == 0 else 'L'}{X(p['working_set_bytes']):.1f},"
            f"{Y(p['per_node_probe_s']*1e6):.1f}"
            for i, p in enumerate(pts) if p.get("per_node_probe_s"))
        s.append(f'<path d="{pth}" fill="none" stroke="{col}" '
                 f'stroke-width="2.1"/>')
        for p in pts:
            if p.get("per_node_probe_s"):
                s.append(f'<circle cx="{X(p["working_set_bytes"]):.1f}" '
                         f'cy="{Y(p["per_node_probe_s"]*1e6):.1f}" r="2.6" '
                         f'fill="{col}"/>')
        for kn in detect_cache_levels(pts):
            x = X(kn["boundary_bytes"])
            yoff = _MT + 30 + (flip % 2) * 14
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{_MT}" '
                     f'y2="{_MT+plot_h}" stroke="{col}" stroke-width="1.4" '
                     f'stroke-dasharray="6,3"/>')
            s.append(f'<text x="{x-3:.1f}" y="{yoff:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="end" font-weight="bold">'
                     f'knee ≈{fmt_bytes(kn["boundary_bytes"])} '
                     f'(×{kn["ratio"]:.2f})</text>')
            flip += 1
    # legend
    ly = _MT + 12
    lx = _ML + plot_w + 14
    for cpu, _pts in series:
        col = color_of(cpu)
        is_v = ccd_of(cpu) in vids
        s.append(f'<line x1="{lx}" x2="{lx+16}" y1="{ly-3}" y2="{ly-3}" '
                 f'stroke="{col}" stroke-width="2.1"/>')
        s.append(f'<text x="{lx+20}" y="{ly}" fill="{FG}" font-size="9.5">'
                 f'cpu {cpu}{" (V$)" if is_v else ""}</text>')
        ly += 16
    s.append("</svg>")
    return "\n".join(s)


def _card(label, value, sub=""):
    return (f'<div class="card"><div class="cv">{_esc(value)}</div>'
            f'<div class="cl">{_esc(label)}</div>'
            + (f'<div class="cs">{_esc(sub)}</div>' if sub else "") + "</div>")


def _lerp_hex(a, b, t):
    """Linear blend between two #rrggbb colors, t∈[0,1]."""
    t = max(0.0, min(1.0, t))
    a, b = a.lstrip("#"), b.lstrip("#")
    out = []
    for i in (0, 2, 4):
        ca, cb = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        out.append(round(ca + (cb - ca) * t))
    return f"#{out[0]:02x}{out[1]:02x}{out[2]:02x}"


def membench_chart_svg(points, status=None, topology=None):
    """Per-node probe time (µs/node) vs working set (log x) for the live memory
    benchmark. Point colour encodes the concurrency that was sustained at that
    size — green when many workers ran in parallel (small graphs, RAM to spare),
    shading to red as memory pressure throttles down to a single worker. The
    working set where concurrency first hits 1 is marked: that's where the run
    became memory-bound."""
    pts = sorted((p for p in points if p.get("per_node_probe_s")),
                 key=lambda p: p["working_set_bytes"])
    if not pts:
        return "<p style='color:#888'>no data points yet…</p>"
    ws = [p["working_set_bytes"] for p in pts]
    ys = [p["per_node_probe_s"] * 1e6 for p in pts]
    concs = [max(1, int(p.get("concurrency", 1) or 1)) for p in pts]
    cmax = max(concs)
    l3 = 0
    if topology:
        l3 = max([c.get("l3_bytes") or 0 for c in topology.get("ccds", [])]
                 or [0])
    wsmin = min(ws)
    wsmax = max(max(ws), l3 or 0)
    lxmin = math.log10(wsmin)
    lxmax = math.log10(max(wsmax, wsmin * 10))
    if lxmax <= lxmin:
        lxmax = lxmin + 1.0
    ymax = (max(ys) * 1.12) if ys else 1.0
    plot_w = _PW - _ML - _MR
    plot_h = _PH - _MT - _MB

    def X(w):
        return _ML + (math.log10(w) - lxmin) / (lxmax - lxmin) * plot_w

    def Y(v):
        return _MT + (1 - v / ymax) * plot_h

    def conc_color(c):
        return _lerp_hex("#e06c6c", VCACHE_COLOR, (c - 1) / max(1, cmax - 1))

    s = [_svg_head(), _frame(_ML, _MT, plot_w, plot_h)]
    s.append(f'<text x="{_PW/2}" y="20" fill="{FG}" font-size="13" '
             f'text-anchor="middle" font-weight="bold">Memory-pressure curve — '
             f'per-node time vs working set (colour = concurrency)</text>')
    for f in range(0, 6):
        yv = ymax * f / 5.0
        yy = Y(yv)
        s.append(f'<line x1="{_ML}" x2="{_ML+plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"'
                 f' stroke="{GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{_ML-8}" y="{yy+4:.1f}" fill="{MUTE}" font-size="10"'
                 f' text-anchor="end">{yv:.2f}</text>')
    j0 = int(math.floor(math.log2(wsmin)))
    j1 = int(math.ceil(math.log2(wsmax)))
    for j in range(j0, j1 + 1):
        w = 2.0 ** j
        if w < wsmin or w > wsmax:
            continue
        xx = X(w)
        s.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{_MT}" y2="{_MT+plot_h}"'
                 f' stroke="{GRID}" stroke-width="0.5"/>')
        if j % 2 == 0:
            s.append(f'<text x="{xx:.1f}" y="{_MT+plot_h+15}" fill="{MUTE}" '
                     f'font-size="9" text-anchor="middle">{fmt_bytes(w)}</text>')
    s.append(f'<text x="{_ML+plot_w/2}" y="{_MT+plot_h+38}" fill="{MUTE}" '
             f'font-size="11" text-anchor="middle">working set (log scale)</text>')
    s.append(f'<text x="16" y="{_MT+plot_h/2}" fill="{MUTE}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 16 {_MT+plot_h/2})">'
             f'µs / node (probe)</text>')
    if l3 and wsmin <= l3 <= wsmax:
        x = X(l3)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{_MT}" y2="{_MT+plot_h}" '
                 f'stroke="{MUTE}" stroke-width="1" stroke-dasharray="2,3" '
                 f'opacity="0.5"/>')
        s.append(f'<text x="{x+3:.1f}" y="{_MT+12}" fill="{MUTE}" font-size="9">'
                 f'L3 {fmt_bytes(l3)}</text>')
    one_i = next((i for i, c in enumerate(concs) if c <= 1), None)
    if one_i is not None:
        x = X(ws[one_i])
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{_MT}" y2="{_MT+plot_h}" '
                 f'stroke="#e06c6c" stroke-width="1.4" stroke-dasharray="6,3"/>')
        s.append(f'<text x="{x-4:.1f}" y="{_MT+plot_h-8}" fill="#e06c6c" '
                 f'font-size="10" text-anchor="end" font-weight="bold">1 worker '
                 f'— memory-bound</text>')
    pth = " ".join(f"{'M' if i == 0 else 'L'}{X(ws[i]):.1f},{Y(ys[i]):.1f}"
                   for i in range(len(pts)))
    s.append(f'<path d="{pth}" fill="none" stroke="{ACCENT}" stroke-width="1.6" '
             f'opacity="0.55"/>')
    for i in range(len(pts)):
        s.append(f'<circle cx="{X(ws[i]):.1f}" cy="{Y(ys[i]):.1f}" r="3.4" '
                 f'fill="{conc_color(concs[i])}"/>')
    lx = _ML + plot_w + 14
    ly = _MT + 14
    s.append(f'<text x="{lx}" y="{ly-2}" fill="{FG}" font-size="10">concurrency'
             f'</text>')
    for kk, (lab, c) in enumerate([(f"{cmax}w", cmax),
                                   ("mid", max(1, cmax // 2)), ("1w", 1)]):
        yy = ly + 12 + kk * 15
        s.append(f'<circle cx="{lx+5}" cy="{yy-3}" r="4" '
                 f'fill="{conc_color(c)}"/>')
        s.append(f'<text x="{lx+14}" y="{yy}" fill="{FG}" font-size="9.5">{lab}'
                 f'</text>')
    s.append("</svg>")
    return "\n".join(s)


def _membench_render(mem_bench, topology):
    """Build (cards, section_html) for the live memory-pressure benchmark."""
    points = (mem_bench or {}).get("points", []) or []
    st = (mem_bench or {}).get("status", {}) or {}
    maxN = st.get("max_N") or max((p["N"] for p in points), default=0)
    cards = [_card("max N reached", f"{maxN:,}", "nodes in one graph")]
    used, tot = st.get("peak_used_bytes"), st.get("mem_total_bytes")
    if used and tot:
        cards.append(_card("peak memory",
                           f"{used/1024**3:.0f}/{tot/1024**3:.0f} GiB",
                           f"≥{(st.get('reserve_bytes') or 0)/1024**3:.0f} GiB "
                           f"kept free"))
    cards.append(_card("max concurrency", f"{st.get('max_concurrency', '?')}",
                       f"of {st.get('workers_cap', '?')} workers"))
    cards.append(_card("cells simulated", f"{st.get('cells_done', len(points))}",
                       "real universes"))
    if st.get("running"):
        if st.get("cur_eta_s") is not None:
            cards.append(_card("current batch", "~" + fmt_hms(st["cur_eta_s"]),
                               f"N={st.get('cur_N', 0):,} · "
                               f"{st.get('cur_done', 0)}/{st.get('cur_total', 0)}"
                               f" cells done"))
        if st.get("next_eta_s") is not None:
            cards.append(_card("next batch (est)", "~" + fmt_hms(st["next_eta_s"]),
                               f"N={st.get('next_N', 0):,} · "
                               f"~{(st.get('next_cell_s') or 0):.1f}s/cell "
                               f"projected"))
    rows = []
    for p in sorted(points, key=lambda d: d["N"]):
        rows.append(
            f"<tr><td>{p['N']:,}</td>"
            f"<td>{fmt_bytes(p['working_set_bytes'])}</td>"
            f"<td>{(p.get('per_node_probe_s') or 0)*1e6:.2f}</td>"
            f"<td>{p.get('concurrency', '')}</td>"
            f"<td>{fmt_bytes(p['rss_bytes']) if p.get('rss_bytes') else '—'}</td>"
            f"<td>{p.get('n', '')}</td></tr>")
    stop = st.get("stop_reason")
    stop_html = f'<p class="verdict">⏹ {_esc(stop)}</p>' if stop else ""

    # per-core / CCD comparison as N grows (reuse the cache-map chart, keyed by
    # a representative cpu per CCD so it colours the V-Cache curve green)
    by_ccd = (mem_bench or {}).get("by_ccd") or {}
    vdelta = (mem_bench or {}).get("vcache_delta") or []
    percore_block = ""
    if len([c for c in by_ccd.values() if c]) >= 2:
        note = ""
        if vdelta:
            pk = max(vdelta, key=lambda d: d.get("speedup", 0))
            note = (f'<p>The V-Cache advantage on the probe peaks near a '
                    f'<b>{fmt_bytes(pk["working_set_bytes"])}</b> working set '
                    f'(<b style="color:{VCACHE_COLOR}">{pk["speedup"]:.2f}×</b>) — '
                    f"where the graph has outgrown the standard CCD's L3 but "
                    f"still fits the V-Cache CCD's, then narrows once both spill "
                    f"to DRAM.</p>")
        percore_block = f"""
<h2>★ Per-core difference as the graph grows — V-Cache vs standard CCD</h2>
<p>Every cell is pinned to a specific core, so splitting the per-node probe time
by CCD shows the two cache tiers as separate curves, re-measured at every N as
the working set climbs.</p>
{cache_map_chart_svg(by_ccd, topology)}
{note}"""

    section = f"""
<h2>★ Memory-pressure benchmark — small graphs to RAM-filling</h2>
<p>Climbing the graph size from small upward, the benchmark runs as many cells
in parallel as fit while keeping the reserve free, then throttles concurrency
down to a single worker and finally stops when even one cell no longer fits in
RAM minus the reserve. The per-node time traces the memory hierarchy as the
working set grows past each cache into DRAM; the colour shows how parallel it
stayed at each size.</p>
{membench_chart_svg(points, st, topology)}
{stop_html}
<table><tr><th>N</th><th>working set</th><th>µs/node</th><th>concurrency</th>
<th>peak RSS/cell</th><th>cells</th></tr>
{''.join(rows) if rows else '<tr><td colspan=6>warming up…</td></tr>'}</table>
{percore_block}"""
    return cards, section


def write_html_report(path, topology, summary, per_cell, percore,
                      params, meta=None, cache_data=None,
                      refresh_s=None, mem_bench=None):
    """Write the self-contained HTML benchmark report.

    Data contracts
    --------------
    topology : dict from bench_workers.detect_topology(). Keys used:
        model, source, n_logical, n_physical, smt, ccds (list of
        {id, cpus, l3_bytes, is_vcache}), vcache_ccd_ids.
    summary  : list of per-worker-count dicts (exactly what
        bench_workers.run_benchmark/run_throughput_detailed return):
        {workers, workers_eff, cells, wall_s, cells_per_min, s_per_cell, ...}.
    per_cell : flat list of per-cell records from the throughput phase:
        {seed, wall_s, cpu, cpu_end, workers, ...}.  May be [].
    percore  : dict cpu_id -> {median_s, p10_s, p90_s, n, ...} from the
        affinity-pinned per-core profile.  May be {} (then that section is
        omitted with a hint to re-run with --per-core).
    params   : dict of the cell/probe settings shown in the header
        (k, T, lb, N, n_probes, lanczos_m, half_window, reps...).
    meta     : optional dict (elapsed_s, generated_at, notes...).
    cache_data : optional dict cpu_id -> [ {N, working_set_bytes, probe_s,
        per_node_probe_s, ...}, ... ] from bench_workers.map_cache_hierarchy.
        When present, adds the "effective cache hierarchy" section; omitted
        otherwise.
    refresh_s : if set, add <meta http-equiv="refresh"> for `refresh_s` seconds
        and a live status line — used by the memory benchmark to self-update.
    mem_bench : optional {"points": [...], "status": {...}} from
        bench_workers.live_memory_benchmark. When present, leads with the
        memory-pressure section.

    Sections render only for the data actually supplied, so a memory-benchmark
    run and a full tuner run both use this one function. The write is atomic
    (tmp + os.replace) so a browser auto-refreshing never reads a half-written
    file.
    """
    meta = meta or {}
    peak = max(summary, key=lambda r: r["cells_per_min"]) if summary else None
    verdict = vcache_verdict(percore, topology)
    gen_at = meta.get("generated_at") or time.strftime("%Y-%m-%d %H:%M:%S")

    # ── headline numbers ──
    model = topology.get("model") or "unknown CPU"
    n_log = topology.get("n_logical", os.cpu_count() or 0)
    n_phys = topology.get("n_physical", n_log)
    ccds = topology.get("ccds", [])
    vids = set(topology.get("vcache_ccd_ids", []))

    def card(label, value, sub=""):
        return (f'<div class="card"><div class="cv">{_esc(value)}</div>'
                f'<div class="cl">{_esc(label)}</div>'
                + (f'<div class="cs">{_esc(sub)}</div>' if sub else "")
                + "</div>")

    cards = []
    if peak:
        cards.append(card("peak throughput", f'{peak["cells_per_min"]:.0f}',
                          "cells / min ('universes/min')"))
        cards.append(card("optimal workers", f'{peak["workers"]}',
                          f'of {n_log} logical / {n_phys} physical'))
        frac = peak["workers"] / n_log * 100 if n_log else 0
        cards.append(card("optimum vs logical", f'{frac:.0f}%',
                          "fewer ⇒ more memory-bound"))
    if verdict:
        cards.append(card("V-Cache speedup", f'{verdict["speedup"]:.2f}×',
                          f'{verdict["pct_faster"]:+.0f}% vs non-V-Cache CCD'))

    # ── CCD topology table ──
    ccd_rows = []
    for c in ccds:
        is_v = c["id"] in vids
        ccd_rows.append(
            f"<tr><td>{'V-Cache' if is_v else 'CCD ' + str(c['id'])}</td>"
            f"<td>{fmt_bytes(c.get('l3_bytes'))}</td>"
            f"<td>{len(c['cpus'])}</td>"
            f"<td class='mono'>{_esc(_compact_ranges(c['cpus']))}</td></tr>")

    # ── per-worker table ──
    base_cpm = (next((r["cells_per_min"] for r in summary if r["workers"] == 1),
                     summary[0]["cells_per_min"]) if summary else 1.0) or 1e-9
    base_sc = (next((r["s_per_cell"] for r in summary if r["workers"] == 1),
                    summary[0]["s_per_cell"]) if summary else 1.0)
    w_rows = []
    for r in sorted(summary, key=lambda x: x["workers"]):
        eff = (base_sc / r["s_per_cell"] * 100.0) if r.get("s_per_cell") else 0.0
        spd = r["cells_per_min"] / base_cpm
        star = " ★" if peak and r["workers"] == peak["workers"] else ""
        w_rows.append(
            f"<tr><td>{r['workers']}{star}</td><td>{r.get('cells','')}</td>"
            f"<td>{r.get('s_per_cell', float('nan')):.2f}</td>"
            f"<td>{eff:.0f}%</td><td>{r['cells_per_min']:.1f}</td>"
            f"<td>{spd:.2f}×</td></tr>")

    verdict_html = ""
    if verdict:
        faster = verdict["speedup"] >= 1.0
        verdict_html = (
            f'<p class="verdict">On a per-core, one-cell-at-a-time profile the '
            f'<b style="color:{VCACHE_COLOR}">V-Cache CCD</b> '
            f'({verdict["n_vcache_cores"]} cores, median '
            f'{verdict["vcache_median_s"]:.2f}s/cell) ran the cell '
            f'<b>{verdict["speedup"]:.2f}×</b> '
            f'{"faster" if faster else "slower"} than the non-V-Cache CCD '
            f'({verdict["n_other_cores"]} cores, {verdict["other_median_s"]:.2f}'
            f's/cell) — a {abs(verdict["pct_faster"]):.0f}% '
            f'{"win" if faster else "deficit"} for the extra L3 at this graph '
            f'size. Larger N (more memory pressure) typically widens this gap; '
            f'a cell that fits in L2 shrinks it to nothing.</p>')

    p = params
    sub = (f"k={p.get('k')} · T={p.get('T')} · ℓb={p.get('lb')} · "
           f"N={int(p.get('N', 0)):,} · n_probes={p.get('n_probes')} · "
           f"lanczos_m={p.get('lanczos_m')}")

    # ── optional section 4: discovered cache hierarchy ──
    cache_section = ""
    if cache_data:
        def _ccd(cpu):
            for c in topology.get("ccds", []):
                if cpu in c["cpus"]:
                    return c["id"]
            return None
        crows, biggest = [], {}
        for cpu in sorted(cache_data):
            knees = detect_cache_levels(cache_data[cpu])
            cid = _ccd(cpu)
            is_v = cid in vids
            l3 = next((c["l3_bytes"] for c in topology.get("ccds", [])
                       if c["id"] == cid), None)
            ks = ", ".join(f"≈{fmt_bytes(k['boundary_bytes'])} "
                           f"(×{k['ratio']:.2f})" for k in knees) or "—"
            crows.append(f"<tr><td>cpu {cpu}{' · V-Cache' if is_v else ''}</td>"
                         f"<td>{fmt_bytes(l3)}</td><td>{ks}</td></tr>")
            if knees:
                biggest[cpu] = max(k["boundary_bytes"] for k in knees)
        summ = ""
        vcpu = next((c for c in biggest if _ccd(c) in vids), None)
        ocpu = next((c for c in biggest if _ccd(c) not in vids), None)
        if vcpu is not None and ocpu is not None:
            summ = (f'<p>The largest working set the workload sustains before '
                    f'per-node time climbs reaches '
                    f'<b style="color:{VCACHE_COLOR}">≈{fmt_bytes(biggest[vcpu])}'
                    f'</b> on the V-Cache core (cpu {vcpu}) vs '
                    f'<b>≈{fmt_bytes(biggest[ocpu])}</b> on the standard core '
                    f'(cpu {ocpu}) — the extra L3 keeps a larger graph resident '
                    f'before it spills.</p>')
        cache_section = f"""
<h2>4 · Effective cache hierarchy (discovered from real runs)</h2>
<p>Pinned to a fixed core, the graph size is swept across a wide range and the
per-node <i>probe</i> time recorded. While the working set fits a cache level
the per-node time is roughly flat; when it outgrows that level it steps up. The
step locations are the effective cache boundaries this workload sees — soft,
because the access pattern is an irregular graph traversal and the prefetchers
help, so they approximate rather than equal the datasheet sizes (overlaid faint
for comparison). L1 is already exceeded by the smallest graph here, so the map
is most meaningful for the L2 → L3 → DRAM regimes.</p>
{cache_map_chart_svg(cache_data, topology)}
{summ}
<table><tr><th>core</th><th>L3 (spec)</th>
<th>detected knees (≈ size, slowdown)</th></tr>
{''.join(crows)}</table>"""

    # ── memory-pressure section + its cards (lead section when present) ──
    mem_section, mem_cards = "", []
    if mem_bench:
        mem_cards, mem_section = _membench_render(mem_bench, topology)

    sec_through = ""
    if summary:
        sec_through = f"""
<h2>1 · Throughput vs worker count — find the optimum</h2>
<p>Each point is the aggregate cells/min from running many real cells (a graph
build + the full d_s(t) SLQ probe) at that pool size, every worker pinned to one
thread. Where the curve peaks is the worker count that finishes the most work
per minute on this machine; where it falls below the dashed ideal-∝W line is the
memory-bandwidth / shared-L3 contention starting to bite.</p>
{throughput_chart_svg(summary)}
<table><tr><th>workers</th><th>cells</th><th>s/cell (med)</th>
<th>core-eff</th><th>cells/min</th><th>speedup</th></tr>
{''.join(w_rows)}</table>"""

    sec_percore = ""
    if percore:
        sec_percore = f"""
<h2>2 · Per-core latency — where the 3D V-Cache helps</h2>
<p>Each cell here was run <b>alone</b>, pinned to one specific logical CPU,
round-robin across the machine. With nothing else competing, a core's median
time isolates its own cache/memory path — so the larger-L3 CCD shows up as a
systematically shorter bar on a memory-bound cell.</p>
{percore_chart_svg(percore, topology)}"""

    sec_scatter = ""
    if per_cell:
        sec_scatter = f"""
<h2>3 · Per-cell spread</h2>
<p>Every cell from the throughput phase, in completion order, coloured by the
CCD it happened to run on. Two cache tiers tend to separate into two horizontal
bands; the vertical scatter within a band is seed-to-seed graph variation.</p>
{scatter_svg(per_cell, topology)}"""

    sections_html = "\n".join(s for s in [mem_section, sec_through, sec_percore,
                                          sec_scatter, cache_section] if s)
    if mem_cards:
        cards = mem_cards + cards

    meta_refresh = (f'<meta http-equiv="refresh" content="{int(refresh_s)}">'
                    if refresh_s else "")
    live_banner = ""
    if refresh_s:
        _st = (mem_bench or {}).get("status", {}) if mem_bench else {}
        live_banner = (
            '<p class="sub" style="font-size:13px">'
            + (f'<span style="color:{GOOD}">●</span> LIVE — auto-refreshing '
               f'every {int(refresh_s)}s'
               if _st.get("running", True) else
               f'<span style="color:{MUTE}">■</span> run finished')
            + '</p>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{meta_refresh}
<title>relational-reality · CPU/memory benchmark</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:{BG}; color:{FG}; margin:0; padding:28px;
         font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size:13px;
         line-height:1.5; }}
  h1 {{ font-size:19px; color:{ACCENT}; margin:0 0 2px; }}
  h2 {{ font-size:14px; color:{ACCENT}; margin:30px 0 10px; font-weight:600;
        border-bottom:1px solid {GRID}; padding-bottom:5px; }}
  .sub {{ color:{MUTE}; margin:0 0 4px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:14px; margin:18px 0 4px; }}
  .card {{ background:{PANEL}; border:1px solid {FRAME}; border-radius:8px;
           padding:12px 16px; min-width:150px; }}
  .cv {{ font-size:26px; color:{FG}; font-weight:700; }}
  .cl {{ color:{ACCENT}; font-size:11px; margin-top:3px; }}
  .cs {{ color:{MUTE}; font-size:10px; margin-top:2px; }}
  table {{ border-collapse:collapse; margin:8px 0 4px; }}
  th,td {{ padding:4px 14px; text-align:right; border-bottom:1px solid #2c2c38;
           font-size:12px; }}
  th {{ color:{ACCENT}; font-weight:normal; text-align:right; }}
  td:first-child, th:first-child {{ text-align:left; }}
  td.mono {{ font-size:11px; color:{MUTE}; }}
  p {{ max-width:920px; color:#bbb; }}
  .verdict {{ background:{PANEL}; border-left:3px solid {VCACHE_COLOR};
              padding:10px 14px; border-radius:4px; }}
  .foot {{ color:{MUTE}; font-size:11px; margin-top:28px;
           border-top:1px solid {GRID}; padding-top:10px; }}
  svg {{ margin:6px 0 2px; }}
</style></head><body>

<h1>Simulating universes — CPU / memory benchmark</h1>
<p class="sub">{_esc(model)} · {n_log} logical / {n_phys} physical cores
{' · SMT on' if topology.get('smt') else ''} · sysfs: {_esc(topology.get('source',''))}</p>
<p class="sub">cell: {_esc(sub)}</p>
<p class="sub">generated {_esc(gen_at)}
{(' · ' + fmt_hms(meta['elapsed_s']) + ' elapsed') if meta.get('elapsed_s') else ''}</p>

<div class="cards">{''.join(cards) if cards else ''}</div>
{live_banner}
{verdict_html}
{sections_html}
<h2>CPU topology (from sysfs)</h2>
<table><tr><th>group</th><th>L3</th><th>cores</th><th>logical CPU ids</th></tr>
{''.join(ccd_rows) if ccd_rows else '<tr><td colspan=4>topology unavailable</td></tr>'}</table>

<p class="foot">Method: real work only — each "cell" is one graph grown from the
Hamiltonian plus its stochastic-Lanczos d_s(t) probe, the same code path the
sweep runs, so the benchmark measures the actual cost of <i>simulating a
universe</i> rather than a synthetic kernel. BLAS and numba are pinned to one
thread per worker, so a W-worker run uses exactly W cores. Throughput uses
random seeds (every cell is distinct work); the per-core profile pins affinity
to read each core in isolation. Numbers are machine- and N-specific: a cell that
fits in L2 scales to all cores, one that spills to DRAM peaks earlier. Generated
by <code>relational-reality</code> bench_report.py.</p>
</body></html>"""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(html)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def _compact_ranges(nums):
    """[0,1,2,3,8,9] -> '0-3, 8-9' for compact CPU-id display."""
    nums = sorted(set(nums))
    if not nums:
        return ""
    out = []
    a = b = nums[0]
    for x in nums[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(f"{a}-{b}" if b > a else f"{a}")
            a = b = x
    out.append(f"{a}-{b}" if b > a else f"{a}")
    return ", ".join(out)


# ════════════════════════════════════════════════════════════════════
#  Synthetic demo data (for tests + `optimal_workers.py --demo`)
# ════════════════════════════════════════════════════════════════════
def synthetic_demo(n_logical=32, vcache_cores=16, seed=0):
    """Fabricate a plausible dual-CCD dataset (one V-Cache CCD, one not) so the
    report can be rendered and eyeballed without running the real benchmark.
    Models: V-Cache cores ~20% faster per cell; throughput rising then bending
    over from bandwidth contention; per-cell noise."""
    import random
    rng = random.Random(seed)
    # Topology: a 9950X3D-like split — two CCDs of `phys/2` physical cores
    # each, 2-way SMT. Logical CPU c has physical core (c % phys); its SMT
    # sibling is c+phys. CCD0 carries the 3D V-Cache (96 MiB L3); CCD1 is
    # standard (32 MiB). So with 32 logical / 16 physical, the V-Cache CCD is
    # logical CPUs {0..7, 16..23}.
    phys = max(1, n_logical // 2)
    half_phys = max(1, min(phys, vcache_cores // 2))   # V-Cache physical cores
    vc_phys = set(range(half_phys))
    vset = {c for c in range(n_logical) if (c % phys) in vc_phys}
    ccds = [
        {"id": 0, "cpus": sorted(vset),
         "l3_bytes": 96 * 1024 ** 2, "is_vcache": True},
        {"id": 1, "cpus": sorted(set(range(n_logical)) - vset),
         "l3_bytes": 32 * 1024 ** 2, "is_vcache": False},
    ]
    cpus = {}
    topology = {"source": "synthetic", "model": "AMD Ryzen 9 9950X3D (demo)",
                "n_logical": n_logical, "n_physical": phys, "smt": True,
                "cpus": cpus, "ccds": ccds, "vcache_ccd_ids": [0]}
    vset = set(ccds[0]["cpus"])

    base_solo = 3.0  # s/cell solo on a standard core
    # throughput sweep
    workers = sorted({1, 4, 8, 16, n_logical})
    summary = []
    for w in workers:
        # contention: per-cell time grows mildly with concurrency (bandwidth)
        contention = 1.0 + 0.5 * (w / n_logical) ** 1.4
        s_per_cell = base_solo * contention
        cpm = w / s_per_cell * 60.0 * 0.97
        summary.append({"workers": w, "workers_eff": w,
                        "cells": w * 4, "wall_s": w * 4 / cpm * 60.0,
                        "cells_per_min": cpm, "s_per_cell": s_per_cell})
    # per-core profile (pinned, solo)
    percore = {}
    for c in range(n_logical):
        solo = base_solo * (0.80 if c in vset else 1.0)
        reps = [solo * rng.uniform(0.95, 1.07) for _ in range(5)]
        reps.sort()
        percore[c] = {"median_s": _median(reps), "p10_s": _pct(reps, 0.1),
                      "p90_s": _pct(reps, 0.9), "n": len(reps),
                      "samples": reps}
    # per-cell records from the (random-scheduled) throughput phase at max W
    per_cell = []
    for i in range(120):
        cpu = rng.randrange(n_logical)
        solo = base_solo * (0.80 if cpu in vset else 1.0)
        per_cell.append({"seed": 1000 + i, "wall_s": solo * 1.45 * rng.uniform(0.9, 1.12),
                         "cpu": cpu, "cpu_end": cpu, "workers": n_logical})
    params = {"k": 8, "T": 0.0, "lb": 0.995, "N": 256000,
              "n_probes": 60, "lanczos_m": 300, "half_window": 10}
    meta = {"elapsed_s": 540, "notes": "synthetic demo data"}

    # cache-hierarchy sweep on one V-Cache core + one standard core. Per-node
    # probe time is flat within a cache level and steps up past it; the V-Cache
    # core's larger L3 (96 MiB) holds a bigger graph before the step than the
    # standard core's 32 MiB. Small fixed overhead inflates per-node at tiny N.
    vc_cpu = ccds[0]["cpus"][0]
    st_cpu = ccds[1]["cpus"][0]
    md = 26
    topology["cpus"] = {
        vc_cpu: {"l1d": 48 * 1024, "l2": 1024 ** 2, "l3": 96 * 1024 ** 2},
        st_cpu: {"l1d": 48 * 1024, "l2": 1024 ** 2, "l3": 32 * 1024 ** 2},
    }

    def _ws(N):
        return N * (md * 4 + 4 + 3 * 8)
    n_grid = [256, 1024, 4096, 16384, 65536, 262144, 524288, 1048576, 2097152]
    L2, base, ovh = 1024 ** 2, 8.0e-6, 0.012
    cache_data = {}
    for cpu, l3 in ((vc_cpu, 96 * 1024 ** 2), (st_cpu, 32 * 1024 ** 2)):
        rows = []
        for N in n_grid:
            ws = _ws(N)
            mult = 1.0 if ws <= L2 else (1.35 if ws <= l3 else 2.1)
            per_node = (base * mult + ovh / N) * rng.uniform(0.97, 1.03)
            probe = per_node * N
            rows.append({"N": N, "working_set_bytes": ws,
                         "rss_bytes": int(ws * 1.6), "build_s": probe * 0.3,
                         "probe_s": probe, "wall_s": probe * 1.3,
                         "per_node_probe_s": per_node, "n": 6})
        cache_data[cpu] = rows
    return topology, summary, per_cell, percore, params, meta, cache_data


def synthetic_mem_demo(n_logical=32, vcache_cores=16, seed=3):
    """Fabricate a live memory-benchmark dataset (dual-CCD, climbing N with the
    concurrency collapse and a V-Cache-vs-standard split) for previewing the
    report layout with `optimal_workers.py --mem-bench --demo`."""
    import random
    rng = random.Random(seed)
    phys = max(1, n_logical // 2)
    half_phys = max(1, min(phys, vcache_cores // 2))
    vc_phys = set(range(half_phys))
    vset = {c for c in range(n_logical) if (c % phys) in vc_phys}
    ccds = [
        {"id": 0, "cpus": sorted(vset), "l3_bytes": 96 * 1024 ** 2,
         "is_vcache": True},
        {"id": 1, "cpus": sorted(set(range(n_logical)) - vset),
         "l3_bytes": 32 * 1024 ** 2, "is_vcache": False},
    ]
    topology = {"source": "synthetic", "model": "AMD Ryzen 9 9950X3D (demo)",
                "n_logical": n_logical, "n_physical": phys, "smt": True,
                "cpus": {}, "ccds": ccds, "vcache_ccd_ids": [0]}
    md = 26

    def ws(N):
        return N * (md * 4 + 4 + 24)
    total, reserve = 64 * 1024 ** 3, 4 * 1024 ** 3
    lo, hi, npts = 1000, 300_000_000, 20
    ratio = (hi / lo) ** (1 / (npts - 1))
    Ns = sorted({int(lo * ratio ** i) for i in range(npts)})
    base, L2, L3s, L3v = 8e-6, 1024 ** 2, 32 * 1024 ** 2, 96 * 1024 ** 2
    points, by_v, by_s, vdelta = [], [], [], []
    rv, rs = ccds[0]["cpus"][0], ccds[1]["cpus"][0]
    for N in Ns:
        w = ws(N)
        need = int(w * 1.3)
        conc = max(1, min(n_logical, (total - reserve) // need))
        cont = 1.0 + 0.4 * (conc / n_logical)
        ms = (1.0 if w <= L2 else (1.35 if w <= L3s else 2.1))
        mv = (1.0 if w <= L2 else (1.35 if w <= L3v else 2.1))
        pn_s = base * ms * cont * rng.uniform(0.97, 1.03)
        pn_v = base * mv * cont * rng.uniform(0.97, 1.03)
        pn = (pn_s + pn_v) / 2
        points.append({"N": N, "working_set_bytes": w, "rss_bytes": int(w * 1.6),
                       "probe_s": pn * N, "wall_s": pn * N * 1.3,
                       "build_s": pn * N * 0.3, "per_node_probe_s": pn,
                       "concurrency": int(conc),
                       "n": int(min(conc, n_logical))})
        by_v.append({"N": N, "working_set_bytes": w, "per_node_probe_s": pn_v,
                     "probe_s": pn_v * N})
        by_s.append({"N": N, "working_set_bytes": w, "per_node_probe_s": pn_s,
                     "probe_s": pn_s * N})
        vdelta.append({"N": N, "working_set_bytes": w, "speedup": pn_s / pn_v})
    cur = Ns[len(Ns) // 2]
    status = {"running": True, "elapsed_s": 312,
              "cells_done": sum(p["n"] for p in points if p["N"] <= cur),
              "peak_used_bytes": int(total * 0.82), "mem_total_bytes": total,
              "mem_available_bytes": int(total * 0.18), "reserve_bytes": reserve,
              "max_N": cur, "max_concurrency": n_logical,
              "workers_cap": n_logical, "batch_size": n_logical,
              "stop_reason": None, "cur_N": cur, "cur_done": 18,
              "cur_total": n_logical, "cur_eta_s": 26,
              "next_N": Ns[len(Ns) // 2 + 1], "next_cell_s": 1.7,
              "next_eta_s": 95}
    mem_bench = {"points": points, "status": status,
                 "by_ccd": {rv: by_v, rs: by_s}, "vcache_delta": vdelta}
    params = {"k": 8, "T": 0.0, "lb": 0.995, "N": Ns[-1], "n_probes": 60,
              "lanczos_m": 300, "half_window": 10}
    return topology, params, mem_bench


if __name__ == "__main__":
    # `python src/bench_report.py [out.html]` → render a synthetic demo report.
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "bench_report_demo.html"
    data = synthetic_demo()
    write_html_report(out, *data)
    print(f"wrote synthetic demo report → {out}")
