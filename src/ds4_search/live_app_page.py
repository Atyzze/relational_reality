"""live_app_page — single-page HTML/CSS/JS for the live dashboard.

Pure presentation, no logic. Served by live_app.py via make_handler(); the
URL base path is baked in at serve time. This page is read-only: it reports
the incoming sweep data and never starts/stops anything (that's Ctrl-C in the
terminal). The chosen filters persist in localStorage across reloads.
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>relational-reality · live spectral-dimension dashboard</title>
<style>
  body { background: #0d0d12; color: #ddd; font-family: ui-monospace, monospace;
         margin: 0; padding: 20px; font-size: 13px; }
  h1 { color: #7ab8e8; font-size: 18px; margin: 0 0 4px; }
  .top-row { display: flex; justify-content: space-between;
             align-items: center; margin-bottom: 14px; }
  .panel { background: #15151b; border: 1px solid #2a2a35;
           border-radius: 4px; padding: 10px 14px; }
  .panel h2 { color: #888; font-size: 10px; text-transform: uppercase;
              margin: 0 0 8px; letter-spacing: 0.5px; font-weight: normal; }
  .terminal { background: #08080c; color: #9cdcfe; padding: 8px 12px;
              border-radius: 4px; font-size: 11px; height: 260px;
              overflow-y: auto; white-space: pre; line-height: 1.45; }
  .live-banner { padding: 8px 14px; border-radius: 4px;
                 background: #1a2a3a; border-left: 3px solid #7ab8e8;
                 margin-bottom: 12px; font-size: 11px; }
  .live-banner.stopped { background: #2a2018; border-left-color: #888;
                         color: #aaa; }
  .live-banner.running { background: #1a2a18; border-left-color: #7ec96e; }
  .progress-bar { height: 4px; background: #15151b; border-radius: 2px;
                  margin-top: 6px; overflow: hidden; }
  .progress-bar .fill { height: 100%; background: #7ec96e;
                        transition: width 0.5s; }
  .filter-bar { background: #15151b; padding: 10px 14px; border-radius: 4px;
                margin-bottom: 12px; display: flex; gap: 24px;
                flex-wrap: wrap; align-items: end; }
  .filter-bar label { display: flex; flex-direction: column; gap: 4px; }
  .filter-bar .lbl { color: #888; font-size: 9px; text-transform: uppercase;
                     letter-spacing: 0.5px; }
  .chip { display: inline-block; padding: 2px 8px; margin-right: 4px;
          background: #1a1a22; color: #888; border: 1px solid #2a2a35;
          border-radius: 2px; cursor: pointer; font-size: 11px; }
  .chip.on { background: #2a4a6a; color: #ddd; border-color: #4a7aaa; }
  table { width: 100%; border-collapse: collapse; font-size: 11px;
          margin-top: 12px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #15151b; }
  th { color: #7ab8e8; cursor: pointer; user-select: none; }
  th:hover { color: #aaccee; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .leaderboards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                  margin-bottom: 12px; }
  .leaderboards .panel { padding: 10px 14px; }
  .leaderboards h2 { color: #7ec96e; }
  .lb-row { font-size: 11px; padding: 2px 0; color: #ccc; }
  svg { background: #08080c; border-radius: 4px; }
  a { color: #7ab8e8; }
</style>
</head><body>

<div class="top-row">
  <h1>relational-reality <span style="color:#444;">·</span> <span style="color:#7ec96e;">live spectral-dimension dashboard</span></h1>
</div>

<div class="live-banner stopped" id="live-banner">Initialising…</div>

<div class="panel" style="margin-bottom: 12px;">
  <h2>Sweep activity <span id="hb-count" style="color:#666;">—</span></h2>
  <div class="terminal" id="terminal">waiting…</div>
</div>

<div class="filter-bar" id="filter-bar"></div>

<div id="plot-container"></div>

<div class="leaderboards">
  <div class="panel">
    <h2>Flattest curves (lower flatness_std = better)</h2>
    <div id="leader-flat"></div>
  </div>
  <div class="panel">
    <h2>Most N-stable groups</h2>
    <div id="leader-nstab"></div>
  </div>
</div>

<table id="cell-table"><thead></thead><tbody></tbody></table>
<p style="font-size:11px;color:#666;margin-top:24px">
  Each row is one (k, T, lb, N, seed) cell. d_s is the spectral dimension.
  Dashed grey curves are the reference lattices (tori). Page polls every 4s.
  <a href="__BASE_PATH__/api/csv" target="_blank">Download consolidated CSV →</a>
</p>

<script>
// When served standalone (default) BASE_PATH is ""; when mounted under a
// prefix every fetch / link prepends it.
const BASE_PATH = "__BASE_PATH__";

function esc(s) {
  if (s == null) return "";
  return String(s).replace(/[<>&"']/g, c =>
    ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'})[c]);
}

// Loop-based min/max. Math.min(...arr) / Math.max(...arr) spread the array as
// function arguments, which throws "RangeError: Maximum call stack size
// exceeded" once the array passes the engine's argument-count limit (~65k in
// V8). The point accumulators in renderPlot (every t/d sample of every visible
// curve) cross that as the sweep grows — ~240 pts × cells — so they must never
// be spread. Reduce-style loops have no such limit.
function arrMin(a){ let m = Infinity;  for (let i=0;i<a.length;i++) if (a[i] < m) m = a[i]; return m; }
function arrMax(a){ let m = -Infinity; for (let i=0;i<a.length;i++) if (a[i] > m) m = a[i]; return m; }

// ─── State ───────────────────────────────────────────────────────
let DATA = {cells: [], toruses: [], data_hash: null};
let STATE = null;
let lastHeartbeatTotal = 0;
let lastDataHash = null;
let sortKey = "flatness_std";
let sortAsc = true;
// Which parameter values are selected for display. Empty set on an axis
// means "show all values of that axis".
let activeFilters = {k: new Set(), T: new Set(), N: new Set(), lb: new Set()};
// Color encoding: which axis maps to plotted curve color.
let colorBy = "lb";
// Single-cell highlight (click a curve/row to dim everything else).
let highlightCellId = null;

// ─── Filter persistence (localStorage) ───────────────────────────
// Remember the chosen filters/sort/color across reloads so the user
// doesn't have to re-select their view every time the page (re)opens.
const LS_KEY = "rr_dashboard_filters_v1";
function saveFilters() {
  try {
    const payload = {
      k: [...activeFilters.k], T: [...activeFilters.T],
      N: [...activeFilters.N], lb: [...activeFilters.lb],
      colorBy, sortKey, sortAsc,
    };
    localStorage.setItem(LS_KEY, JSON.stringify(payload));
  } catch (e) {}
}
function loadFilters() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return;
    const p = JSON.parse(raw);
    for (const ax of ["k", "T", "N", "lb"])
      if (Array.isArray(p[ax])) activeFilters[ax] = new Set(p[ax]);
    if (p.colorBy) colorBy = p.colorBy;
    if (p.sortKey) sortKey = p.sortKey;
    if (typeof p.sortAsc === "boolean") sortAsc = p.sortAsc;
  } catch (e) {}
}

function cellId(c) {
  return `k${c.k}_T${c.T}_lb${c.lb}_N${c.N}_s${c.seed||""}`;
}

const SHAPE_COLOR = {
  monotone_rising: "#7ec96e",
  monotone_falling: "#dcdcaa",
  flat: "#9cdcfe",
  wobble: "#888",
  single_peak: "#e5b06b",
  double_peak: "#c586c0",
  bumpy: "#f48771",
  undetermined: "#444",
};

// ─── Polling ─────────────────────────────────────────────────────
// Re-entrancy guard: as the dataset grows a full render takes a moment;
// without this the 4 s interval stacks polls on top of an in-flight one,
// saturating the main thread so the chart looks frozen while the chips still
// respond. One poll at a time.
let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const results = await Promise.all(
      [pollState(), pollHeartbeat(), maybePollData()]);
    const dataChanged = results[2];
    // Banner is cheap and STATE-driven — refresh every tick.
    safe(renderLiveBanner);
    // Heavy panels (plot/table/leaderboards) are DATA-driven — only rebuild
    // them when the dataset actually changed, so routine polling doesn't
    // re-render a big SVG every 4 s and fight user interaction.
    if (dataChanged) renderDataPanels();
  } catch (e) {
    console.error("poll failed", e);
  } finally {
    polling = false;
  }
}
async function pollState() {
  try {
    const r = await fetch(BASE_PATH + "/api/state", {cache: "no-store"});
    STATE = await r.json();
  } catch (e) {}
}
async function pollHeartbeat() {
  try {
    const r = await fetch(BASE_PATH + "/api/heartbeat", {cache: "no-store"});
    const j = await r.json();
    if (j.total !== lastHeartbeatTotal) {
      const term = document.getElementById("terminal");
      term.textContent = j.lines.join("\n");
      term.scrollTop = term.scrollHeight;
      document.getElementById("hb-count").textContent =
        `(${j.total} lines)`;
      lastHeartbeatTotal = j.total;
    }
  } catch (e) {}
}
async function maybePollData() {
  // Re-fetch /api/data only when the dataset hash changes — avoids
  // re-transferring all curves every tick. Cheap server-side (the
  // CSVs sit in the kernel page cache).
  try {
    const r = await fetch(BASE_PATH + "/api/data", {cache: "no-store"});
    const j = await r.json();
    if (j.data_hash !== lastDataHash) {
      DATA = j;
      lastDataHash = j.data_hash;
      return true;          // dataset changed → caller re-renders panels
    }
  } catch (e) {}
  return false;
}

// ─── Rendering ───────────────────────────────────────────────────
// Each panel renders independently and is isolated: a throw in one (e.g. a
// transient malformed cell) must not abort the others or bubble out of the
// poll loop and freeze the page — previously a single renderAll() throw
// stopped the chart updating until a full reload.
function safe(fn) {
  try { fn(); } catch (e) { console.error((fn && fn.name) || "render", e); }
}
function renderDataPanels() {            // DATA-driven panels (the heavy ones)
  safe(renderFilterBar);
  safe(renderLeaderboards);
  safe(renderPlot);
  safe(renderTable);
}
// Full render used by user interactions (toggles, sort, highlight). It no
// longer bails when STATE is missing: the data panels only need DATA, so
// existing curves draw on first load even before /api/state has answered
// (renderLiveBanner guards STATE itself).
function renderAll() {
  safe(renderLiveBanner);
  renderDataPanels();
}

function renderLiveBanner() {
  if (!STATE) return;          // banner needs STATE; the data panels don't
  const el = document.getElementById("live-banner");
  const s = STATE.sweep_status || {};
  const running = STATE.sweep_running;
  const phase = s.phase;
  el.classList.toggle("running", running);
  el.classList.toggle("stopped", !running);
  if (phase === "calibrating") {
    const n = (s.calibration && s.calibration.n_targets) || "?";
    el.innerHTML = `<b>Calibrating</b> &middot; computing μ for ${n} (k, T, lb) `
      + `point(s) before the sweep starts — this can take a while.`
      + `<div class="progress-bar"><div class="fill" style="width:100%;`
      + `background:#e5b06b;opacity:0.4"></div></div>`;
  } else if (running) {
    const pct = s.n_total ? (100 * (s.n_done || 0) / s.n_total).toFixed(1) : 0;
    el.innerHTML = `<b>Sweep running</b> &middot; ${s.n_done||0}/${s.n_total||"?"} cells `
      + `(${pct}%) &middot; rate ${s.rate_per_min||"—"} cells/min &middot; ETA ${fmtSec(s.eta_sec)}`
      + `<div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>`;
  } else {
    el.innerHTML = `<b>Sweep not running</b> &middot; the grid is complete, or the worker is still spinning up. `
      + `Workers are controlled from the terminal (Ctrl-C to stop).`;
  }
}

// ─── Filter bar ──────────────────────────────────────────────────
// Re-renders on every poll so newly discovered parameter values show
// up as chips. Selection state lives in `activeFilters` (preserved
// across renders and persisted to localStorage).
function renderFilterBar() {
  const bar = document.getElementById("filter-bar");
  const cells = DATA.cells || [];
  if (cells.length === 0) {
    bar.innerHTML = `<div style="color:#666">no cells yet — waiting for the sweep…</div>`;
    return;
  }
  const ks = [...new Set(cells.map(c => c.k))].sort((a,b)=>a-b);
  const Ts = [...new Set(cells.map(c => c.T))].sort((a,b)=>a-b);
  const Ns = [...new Set(cells.map(c => c.N))].sort((a,b)=>a-b);
  const lbs = [...new Set(cells.map(c => c.lb))].sort((a,b)=>a-b);

  function chipSet(label, opts, axis) {
    const set = activeFilters[axis];
    let html = `<label><span class="lbl">${esc(label)}</span><span>`;
    for (const o of opts) {
      const ev = esc(o);
      const v = Number(o);
      const on = set.has(v) ? "on" : "";
      html += `<span class="chip ${on}" data-axis="${esc(axis)}" data-val="${ev}" onclick="toggleFilter(this)">${ev}</span>`;
    }
    return html + "</span></label>";
  }
  function colorByOpt(val, label) {
    const on = (colorBy === val) ? "on" : "";
    return `<span class="chip ${on}" onclick="setColorBy('${val}')">${label}</span>`;
  }

  bar.innerHTML =
    chipSet("k", ks, "k")
    + chipSet("T", Ts, "T")
    + chipSet("N", Ns, "N")
    + chipSet("lb", lbs, "lb")
    + `<label><span class="lbl">color by</span><span>`
      + colorByOpt("lb", "lb")
      + colorByOpt("k", "k")
      + colorByOpt("T", "T")
      + colorByOpt("N", "N")
    + `</span></label>`;
}

function toggleFilter(el) {
  const axis = el.dataset.axis;
  const v = Number(el.dataset.val);
  const set = activeFilters[axis];
  if (set.has(v)) set.delete(v); else set.add(v);
  saveFilters();
  renderAll();
}
function setColorBy(axis) { colorBy = axis; saveFilters(); renderAll(); }

function applyFilters(cells) {
  return cells.filter(c => {
    for (const axis of ["k", "T", "N", "lb"]) {
      const set = activeFilters[axis];
      if (set.size > 0 && !set.has(c[axis])) return false;
    }
    return true;
  });
}

// ─── Leaderboards ────────────────────────────────────────────────
function renderLeaderboards() {
  const visible = applyFilters(DATA.cells || []);
  const flat = [...visible].sort((a,b) =>
    (a.flatness_std ?? 1e9) - (b.flatness_std ?? 1e9)).slice(0, 5);
  document.getElementById("leader-flat").innerHTML =
    flat.length ? flat.map(c =>
      `<div class="lb-row">k=${esc(c.k)} T=${esc(c.T)} lb=${esc(c.lb)} N=${esc(c.N)}: `
      + `<b>flat_σ=${(c.flatness_std||0).toFixed(3)}</b> `
      + `d_s med=${(c.ds_median||0).toFixed(2)} `
      + `<span style="color:${SHAPE_COLOR[c.shape]||'#888'}">(${esc(c.shape)})</span></div>`
    ).join("") : `<div style="color:#666">no cells in current selection</div>`;
  const groups = {};
  for (const c of visible) {
    const key = `k=${c.k}, T=${c.T}, lb=${c.lb}`;
    (groups[key] = groups[key] || []).push(c);
  }
  const stab = Object.entries(groups)
    .filter(([k,v]) => v.length >= 2)
    .map(([k,v]) => {
      const meds = v.map(c => c.ds_median).filter(x => x != null);
      const spread = meds.length >= 2 ?
        arrMax(meds) - arrMin(meds) : null;
      return [k, v, spread];
    })
    .filter(([_, __, sp]) => sp != null)
    .sort((a,b) => a[2] - b[2]).slice(0, 5);
  document.getElementById("leader-nstab").innerHTML =
    stab.length ? stab.map(([k,v,sp]) =>
      `<div class="lb-row">${k}: <b>spread=${sp.toFixed(3)}</b> across ${v.length} N-values</div>`
    ).join("") : `<div style="color:#666">need ≥2 N values per (k, T, lb) — sweep more N values</div>`;
}

// ─── Plot ────────────────────────────────────────────────────────
function renderPlot() {
  const visible = applyFilters(DATA.cells || []);
  const toruses = DATA.toruses || [];          // reference lattices: always shown
  const W = 1200, H = 540;
  const PAD = {l: 60, r: 156, t: 20, b: 50};   // wide right margin: d_s axis
                                               // labels + the colour legend
                                               // both live here, so it must
                                               // be roomy enough not to clip.
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
  let allD = [], allT = [];      // in-window points: drive the d-axis range
  let allT_full = [];            // every finite point: drives the t-axis range
                                 // so the faint out-of-window tails (shown
                                 // below) aren't clipped off the right edge.
  for (const c of [...visible, ...toruses]) {
    for (let i = 0; i < c.t.length; i++) {
      if (c.t[i] == null || c.d[i] == null) continue;
      if (!isFinite(c.t[i]) || !isFinite(c.d[i])) continue;
      allT_full.push(c.t[i]);
      if (c.in_window[i]) { allT.push(c.t[i]); allD.push(c.d[i]); }
    }
  }
  if (!allD.length) {
    document.getElementById("plot-container").innerHTML =
      `<div style="text-align:center;padding:80px;color:#666">no data in selection</div>`;
    return;
  }
  const tMin = Math.max(0.05, arrMin(allT_full));
  const tMax = arrMax(allT_full) * 1.05;
  // d-axis range is set by the in-window data so the fitted region keeps
  // its vertical detail; the faint tails extend a little past it and are
  // allowed to clip rather than rescale the whole plot around an outlier.
  const dMin = Math.max(0, arrMin(allD) - 0.4);
  const dMax = arrMax(allD) + 0.4;
  const xOf = t => PAD.l + plotW * (Math.log10(t) - Math.log10(tMin)) / (Math.log10(tMax) - Math.log10(tMin));
  const yOf = d => PAD.t + plotH - plotH * (d - dMin) / (dMax - dMin);

  let svg = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
  // Clip region = the plot area. Faint out-of-window tails can run past the
  // d-axis range; clipping keeps them inside the axes instead of drawing
  // over the d_s tick labels or the colour legend in the right margin.
  svg += `<defs><clipPath id="plotclip"><rect x="${PAD.l}" y="${PAD.t}" `
       + `width="${plotW}" height="${plotH}"/></clipPath></defs>`;
  svg += `<line x1="${PAD.l}" y1="${PAD.t+plotH}" x2="${W-PAD.r}" y2="${PAD.t+plotH}" stroke="#444"/>`;
  svg += `<line x1="${PAD.l}" y1="${PAD.t}" x2="${PAD.l}" y2="${PAD.t+plotH}" stroke="#444"/>`;
  for (let d = Math.ceil(dMin); d <= Math.floor(dMax); d++) {
    if (d === 0) continue;
    const y = yOf(d);
    svg += `<line x1="${PAD.l}" y1="${y}" x2="${W-PAD.r}" y2="${y}" stroke="#222" stroke-dasharray="2,4"/>`;
    svg += `<text x="${W-PAD.r+4}" y="${y+3}" fill="#444" font-size="9">d_s=${d}</text>`;
    svg += `<text x="${PAD.l-8}" y="${y+3}" fill="#888" font-size="10" text-anchor="end">${d}</text>`;
  }
  for (let p = Math.floor(Math.log10(tMin)); p <= Math.ceil(Math.log10(tMax)); p++) {
    const tv = Math.pow(10, p);
    if (tv < tMin || tv > tMax) continue;
    const x = xOf(tv);
    svg += `<line x1="${x}" y1="${PAD.t+plotH}" x2="${x}" y2="${PAD.t+plotH+4}" stroke="#444"/>`;
    svg += `<text x="${x}" y="${PAD.t+plotH+18}" fill="#888" font-size="10" text-anchor="middle">10^${p}</text>`;
  }
  svg += `<text x="${PAD.l+plotW/2}" y="${PAD.t+plotH+38}" fill="#888" font-size="11" text-anchor="middle">diffusion time t (log)</text>`;
  svg += `<text x="${PAD.l-44}" y="${PAD.t+plotH/2}" fill="#888" font-size="11" transform="rotate(-90 ${PAD.l-44} ${PAD.t+plotH/2})" text-anchor="middle">spectral dimension  d_s(t)</text>`;

  const colorOf = makeColorFn(visible);

  // ── Clipped curve layer (references + cells) ──────────────────────
  svg += `<g clip-path="url(#plotclip)">`;

  // Reference lattices (tori) — dashed grey, labelled by dimension.
  // Drawn full-range like the cells so their large-t behaviour is on the
  // same footing for comparison; the out-of-window portion is fainter.
  for (const tr of toruses) {
    const r = buildPath(tr, xOf, yOf, {fullRange: true});
    if (!r) continue;
    for (const seg of r.outSegs) {
      svg += `<path d="${seg}" fill="none" stroke="#888" stroke-width="1.5"
              stroke-dasharray="1,4" opacity="0.28"/>`;
    }
    if (r.inPath) {
      svg += `<path d="${r.inPath}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>`;
    }
    const lastIdx = tr.in_window.lastIndexOf(true);
    if (lastIdx >= 0) {
      const lx = xOf(tr.t[lastIdx]), ly = yOf(tr.d[lastIdx]);
      svg += `<text x="${lx+4}" y="${ly+3}" fill="#888" font-size="9">${tr.torus_dim}D ref</text>`;
    }
  }

  const isHighlighting = highlightCellId != null;
  for (const c of visible) {
    const r = buildPath(c, xOf, yOf, {fullRange: true});
    if (!r) continue;
    const isMe = (cellId(c) === highlightCellId);
    const dim = isHighlighting && !isMe;
    const color = colorOf(c);
    const baseOp = dim ? 0.12 : (isMe ? 1.0 : 0.85);
    const width = isMe ? 2.6 : 1.3;
    // Faint out-of-window tail first (drawn under the solid in-window
    // stroke). Dotted + low opacity so it reads as "measured but outside
    // the power-law fit window" rather than part of the fitted curve.
    for (const seg of r.outSegs) {
      svg += `<path d="${seg}" fill="none" stroke="${color}"
              stroke-width="${width}" opacity="${(baseOp * 0.35).toFixed(2)}"
              stroke-dasharray="1,3" style="pointer-events:none"/>`;
    }
    if (r.inPath) {
      svg += `<path d="${r.inPath}" fill="none" stroke="${color}" stroke-width="${width}" opacity="${baseOp}"
              data-cellid="${esc(cellId(c))}" onclick="onCurveClick(event)"
              style="cursor:pointer"/>`;
    }
  }

  svg += `</g>`;   // end clipped curve layer (legend below must not clip)

  // Legend sits to the right of the d_s axis tick labels (which occupy
  // ~W-PAD.r .. W-PAD.r+40). Start it past them, leaving room for the
  // swatch + value labels inside the viewBox.
  svg += renderLegend(visible, W - PAD.r + 60, PAD.t, colorOf);
  svg += `</svg>`;
  document.getElementById("plot-container").innerHTML = svg;
}

// Colour mapping is linear by default, but switches to log when the colour
// axis spans ≥1000× (e.g. N = 1e3 … 1e9) — on a linear ramp every N below
// ~1M would collapse into one end. Small-range axes (k, T, lb) stay linear.
function colorScale(vals) {
  const vMin = arrMin(vals), vMax = arrMax(vals);
  const useLog = vMin > 0 && (vMax / vMin) >= 1000;
  const lo = useLog ? Math.log10(vMin) : vMin;
  const hi = useLog ? Math.log10(vMax) : vMax;
  const norm = v => {
    if (hi === lo) return 0.5;
    const x = useLog ? Math.log10(v) : v;
    return (x - lo) / (hi - lo);
  };
  return {vMin, vMax, useLog, norm};
}
function makeColorFn(visible) {
  const vals = visible.map(c => c[colorBy]).filter(x => x != null);
  if (!vals.length) return () => "#888";
  const sc = colorScale(vals);
  return c => (c[colorBy] == null) ? "#888" : lbColor(sc.norm(c[colorBy]));
}

function renderLegend(visible, x, y, colorOf) {
  const vals = visible.map(c => c[colorBy]).filter(v => v != null);
  if (!vals.length) return "";
  const sc = colorScale(vals);
  const vMin = sc.vMin, vMax = sc.vMax;
  const tag = sc.useLog ? " (log)" : "";
  let svg = `<text x="${x}" y="${y}" fill="#888" font-size="10">color: ${esc(colorBy)}${tag}</text>`;
  const N = 16, h = 80;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    svg += `<rect x="${x}" y="${y + 12 + i * (h/N)}" width="14" height="${h/N + 0.5}" fill="${lbColor(t)}"/>`;
  }
  svg += `<text x="${x+18}" y="${y+18}" fill="#aaa" font-size="9">${vMin}</text>`;
  svg += `<text x="${x+18}" y="${y+12+h-2}" fill="#aaa" font-size="9">${vMax}</text>`;
  return svg;
}

function onCurveClick(ev) {
  const id = ev.target.getAttribute("data-cellid");
  highlightCellId = (highlightCellId === id) ? null : id;
  renderAll();
}

function buildPath(c, xOf, yOf, opts) {
  // Default (opts omitted / inWindowOnly): the original behaviour — a
  // single path through in-window points only.
  //
  // When opts.fullRange is set, return {inPath, outSegs}: the in-window
  // stretch as one solid path, plus the out-of-window points grouped into
  // contiguous runs (outSegs) so the caller can stroke them faintly. Each
  // out-of-window run is extended by one in-window point on each side so
  // the faint tail visually connects to the solid curve instead of
  // floating with a gap. This lets the large-t tail past the power-law
  // window be inspected without implying it's part of the fitted region.
  const N = c.t.length;
  const ok = i => c.t[i] != null && c.d[i] != null
                  && isFinite(c.t[i]) && isFinite(c.d[i]);

  if (!opts || !opts.fullRange) {
    let d = "", started = false;
    for (let i = 0; i < N; i++) {
      if (!c.in_window[i] || !ok(i)) continue;
      const x = xOf(c.t[i]), y = yOf(c.d[i]);
      d += (started ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
      started = true;
    }
    return started ? d : null;
  }

  // Full-range mode.
  let inPath = "", inStarted = false;
  for (let i = 0; i < N; i++) {
    if (!c.in_window[i] || !ok(i)) continue;
    const x = xOf(c.t[i]), y = yOf(c.d[i]);
    inPath += (inStarted ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
    inStarted = true;
  }

  const outSegs = [];
  let i = 0;
  while (i < N) {
    if (c.in_window[i] || !ok(i)) { i++; continue; }
    // start of an out-of-window run [runStart, runEnd)
    let runStart = i;
    while (i < N && !c.in_window[i]) i++;
    let runEnd = i;
    // extend by one bracketing in-window sample each side, when present
    let lo = runStart, hi = runEnd - 1;
    if (runStart - 1 >= 0 && ok(runStart - 1)) lo = runStart - 1;
    if (runEnd < N && ok(runEnd)) hi = runEnd;
    let seg = "", segStarted = false;
    for (let j = lo; j <= hi; j++) {
      if (!ok(j)) { segStarted = false; continue; }
      const x = xOf(c.t[j]), y = yOf(c.d[j]);
      seg += (segStarted ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
      segStarted = true;
    }
    if (seg) outSegs.push(seg);
  }
  return { inPath: inStarted ? inPath : null, outSegs };
}
function lbColor(t) {
  const hue = 280 * (1 - t);     // 280° (violet) → 0° (red)
  return `hsl(${hue}, 75%, 55%)`;
}

// ─── Table ───────────────────────────────────────────────────────
function renderTable() {
  const visible = applyFilters(DATA.cells || []);
  const cols = [
    {k:"k", t:"k"}, {k:"T", t:"T"}, {k:"lb", t:"lb"}, {k:"N", t:"N"},
    {k:"shape", t:"shape"},
    {k:"ds_median", t:"d_s med", num:true, fmt:v=>v?.toFixed(2)},
    {k:"flatness_std", t:"flat_σ", num:true, fmt:v=>v?.toFixed(3)},
    {k:"flatness_iqr", t:"flat_iqr", num:true, fmt:v=>v?.toFixed(3)},
    {k:"flatness_range", t:"flat_range", num:true, fmt:v=>v?.toFixed(2)},
    {k:"ds_p10", t:"d_s p10", num:true, fmt:v=>v?.toFixed(2)},
    {k:"ds_p90", t:"d_s p90", num:true, fmt:v=>v?.toFixed(2)},
  ];
  const sorted = [...visible].sort((a,b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av == null) return 1;
    if (bv == null) return -1;
    return sortAsc ? av - bv : bv - av;
  });
  let h = "<tr>";
  for (const c of cols)
    h += `<th class="${c.num?'num':''}" onclick="setSort('${esc(c.k)}')">${esc(c.t)}${sortKey===c.k?(sortAsc?' ▲':' ▼'):''}</th>`;
  h += "</tr>";
  document.querySelector("#cell-table thead").innerHTML = h;
  let b = "";
  for (const r of sorted) {
    const id = cellId(r);
    const rowStyle = (id === highlightCellId)
      ? "background:#1a3a5a;cursor:pointer" : "cursor:pointer";
    b += `<tr style="${rowStyle}" onclick="onRowClick('${esc(id)}')">`;
    for (const c of cols) {
      const v = r[c.k];
      const display = (v == null) ? "—" : (c.fmt ? c.fmt(v) : esc(v));
      const style = (c.k==="shape") ? `color:${SHAPE_COLOR[v]||'#888'}` : "";
      b += `<td class="${c.num?'num':''}" style="${style}">${display}</td>`;
    }
    b += "</tr>";
  }
  document.querySelector("#cell-table tbody").innerHTML = b;
}
function onRowClick(id) {
  highlightCellId = (highlightCellId === id) ? null : id;
  renderAll();
}
function setSort(k) {
  if (sortKey === k) sortAsc = !sortAsc;
  else { sortKey = k; sortAsc = true; }
  saveFilters();
  renderTable();
}

// ─── Helpers ─────────────────────────────────────────────────────
function fmtSec(s) {
  if (s == null) return "—";
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s/60) + "m " + (s%60) + "s";
  if (s < 86400) return Math.floor(s/3600) + "h " + Math.floor((s%3600)/60) + "m";
  return Math.floor(s/86400) + "d " + Math.floor((s%86400)/3600) + "h";
}

// ─── Init ────────────────────────────────────────────────────────
loadFilters();
poll();
setInterval(poll, 4000);
</script>

</body></html>
"""
