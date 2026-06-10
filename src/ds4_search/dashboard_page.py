"""dashboard_page — single-page HTML/CSS/JS for the offline ds4_search
dashboard. Pure presentation. dashboard.py injects the curve data by
replacing the __DATA_JSON__ placeholder."""

HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>ds4_search/static_dashboard.py · lb-sweep flow dashboard</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #0a0a0e; color: #ddd; margin: 0; padding: 20px;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    font-size: 12px;
  }
  h1 { font-size: 16px; color: #7ec9ff; margin: 0 0 4px; font-weight: 600; }
  .status { color: #888; font-size: 11px; margin-bottom: 14px; }
  .build-pill {
    float: right; color: #555; font-size: 10px;
    background: #15151b; border: 1px solid #2a2a35; border-radius: 3px;
    padding: 2px 8px; font-family: ui-monospace, monospace;
    cursor: help;
  }
  .build-pill .h { color: #7ec96e; }
  .build-pill .sweep-h { color: #e5b06b; }
  .filter-bar {
    background: #15151b; border: 1px solid #2a2a35; border-radius: 4px;
    padding: 12px 14px; margin-bottom: 12px; display: flex;
    flex-wrap: wrap; gap: 18px; align-items: flex-start;
  }
  .filter-group { display: flex; flex-direction: column; gap: 4px; }
  .filter-group .lbl { color: #7ec9ff; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.06em; }
  .chips { display: flex; gap: 4px; flex-wrap: wrap; }
  .chip {
    background: #1f1f27; color: #aaa; border: 1px solid #333;
    border-radius: 3px; padding: 3px 8px; cursor: pointer;
    font-size: 11px; user-select: none; transition: all 0.1s;
  }
  .chip.on { background: #2d4a5a; color: #9cdcfe; border-color: #4a7090; }
  .chip:hover { border-color: #555; }
  .range-row { display: flex; align-items: center; gap: 6px;
    font-size: 10px; color: #aaa; }
  .range-row input[type=number] { width: 56px; background: #0a0a0e;
    color: #ddd; border: 1px solid #333; border-radius: 2px;
    padding: 2px 4px; font-family: inherit; font-size: 11px; }
  .clear-btn { background: transparent; color: #888; border: 1px solid #333;
    border-radius: 3px; padding: 4px 10px; cursor: pointer; font-size: 11px;
    font-family: inherit; }
  .clear-btn:hover { color: #ddd; border-color: #555; }
  .plot-container {
    background: #15151b; border: 1px solid #2a2a35; border-radius: 4px;
    padding: 8px; margin-bottom: 12px; min-height: 580px;
  }
  #plot-svg { width: 100%; height: 580px; display: block; }
  .leaderboard {
    background: #15151b; border: 1px solid #2a2a35; border-radius: 4px;
    padding: 10px 14px; margin-bottom: 12px;
  }
  .leaderboard h3 { color: #7ec96e; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 6px;
    font-weight: 600; }
  .leaderboard .row { display: flex; gap: 14px; font-size: 11px;
    color: #ccc; padding: 2px 0; }
  .leaderboard .row .key { color: #888; min-width: 220px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 4px 8px; text-align: right; font-size: 11px;
    border-bottom: 1px solid #232330; }
  th { color: #7ec9ff; cursor: pointer; user-select: none;
    font-weight: 600; background: #15151b; position: sticky; top: 0; }
  th:hover { color: #fff; }
  th.sort-asc::after { content: " ▲"; color: #7ec96e; }
  th.sort-desc::after { content: " ▼"; color: #7ec96e; }
  td.left, th.left { text-align: left; }
  tr:hover { background: #1a1a22; }
  tr.highlighted { background: #2a2a40; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap;
    font-size: 10px; color: #aaa; margin: 6px 4px; }
  .legend .swatch { display: inline-block; width: 18px; height: 3px;
    margin-right: 4px; vertical-align: middle; }
  .table-container { background: #15151b; border: 1px solid #2a2a35;
    border-radius: 4px; max-height: 480px; overflow-y: auto;
    overflow-x: auto; }
  .help { color: #777; font-size: 10px; margin-top: 4px; max-width: 900px;
    line-height: 1.5; }
  .live-status {
    background: #15151b; border: 1px solid #2a2a35; border-radius: 4px;
    padding: 10px 14px; margin-bottom: 12px; display: none;
  }
  .live-status.show { display: block; }
  .live-status.running { border-left: 3px solid #e5b06b; }
  .live-status.done    { border-left: 3px solid #7ec96e; }
  .live-status .heading { color: #e5b06b; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
    margin-bottom: 6px; }
  .live-status.done .heading { color: #7ec96e; }
  .live-status .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 8px 18px; font-size: 11px; color: #ccc;
  }
  .live-status .k { color: #888; }
  .live-status .progress-bar {
    background: #0a0a0e; height: 6px; border-radius: 3px;
    margin-top: 6px; overflow: hidden;
  }
  .live-status .progress-bar .fill {
    background: #7ec96e; height: 100%; transition: width 0.4s ease;
  }
  .live-status.running .progress-bar .fill { background: #e5b06b; }
</style></head>
<body>

<h1><span style="color:#7ec96e;">ds4_search/static_dashboard.py</span> <span style="color:#444;">·</span> lb-sweep flow dashboard <span class="build-pill" id="build-pill" title=""></span></h1>
<div class="status" id="status">Loading…</div>

<div class="live-status" id="live-status"></div>

<div class="filter-bar" id="filter-bar"></div>

<div class="leaderboard" id="leaderboard"></div>

<div class="plot-container">
  <svg id="plot-svg" viewBox="0 0 1200 580"
       preserveAspectRatio="xMidYMid meet"></svg>
  <div class="legend" id="plot-legend"></div>
</div>

<div class="table-container">
  <table id="cell-table">
    <thead><tr id="thead-row"></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<p class="help">
Each row is one (k, T, lb, N, seed) cell. <b>flatness_std</b> is the std
of d_s across the in-window region — the smaller, the more "the spectral
dimension at this scale" actually means something. Sort by it to find
candidates. Click a row to highlight its curve. Click a column header to
re-sort. Filters above are AND-combined; multi-select within a group is OR.
</p>

<script>
const DATA = __DATA_JSON__;
const STATUS_JSON_PATH = __STATUS_PATH__;
const PROGRESS = __PROGRESS_JSON__;
const BUILD = __BUILD_JSON__;
const TABLE_COLUMNS = [
  {key: "tag", label: "cell", left: true, fmt: v => v},
  {key: "shape", label: "shape", left: true,
    fmt: v => `<span style="color:${SHAPE_COLOR[v] || '#aaa'}">${v}</span>`},
  {key: "k", label: "k", fmt: v => v ?? "—"},
  {key: "T", label: "T", fmt: v => v == null ? "—" : v},
  {key: "lb", label: "lb", fmt: v => v == null ? "—" : v},
  {key: "N", label: "N", fmt: v => v},
  {key: "ds_median", label: "ds_med", fmt: v => v?.toFixed?.(2) ?? "—"},
  {key: "flatness_std", label: "flat_σ", fmt: v => v?.toFixed?.(3) ?? "—"},
  {key: "flatness_iqr", label: "flat_iqr", fmt: v => v?.toFixed?.(3) ?? "—"},
  {key: "flatness_range", label: "flat_range", fmt: v => v?.toFixed?.(2) ?? "—"},
  {key: "ds_p10", label: "p10", fmt: v => v?.toFixed?.(2) ?? "—"},
  {key: "ds_p90", label: "p90", fmt: v => v?.toFixed?.(2) ?? "—"},
];

const SHAPE_COLOR = {
  "flat": "#7ec96e",
  "monotone_rising": "#e5b06b",
  "monotone_falling": "#7ec9ff",
  "single_peak": "#dcdcaa",
  "double_peak": "#c586c0",
  "bumpy": "#f48771",
  "undetermined": "#666",
};

// ─── State ────────────────────────────────────────────────────────
const state = {
  filterK: new Set(),    // empty set = all
  filterT: new Set(),
  filterN: new Set(),
  filterShape: new Set(),
  showTorus: true,
  lbMin: null, lbMax: null,
  flatMax: null,            // upper bound on flatness_std
  colorBy: "lb",            // "lb" | "k" | "T" | "N" | "shape"
  sortKey: "flatness_std",
  sortDesc: false,
  highlightTag: null,
};

// ─── Build cells/torus from DATA ──────────────────────────────────
const cells = DATA.cells || [];
const toruses = DATA.toruses || [];
cells.forEach((c, i) => { c.tag = `k${c.k}_T${c.T}_lb${c.lb}_N${c.N}_s${c.seed}`; c.idx = i; });

const allK = [...new Set(cells.map(c => c.k))].sort((a, b) => a - b);
const allT = [...new Set(cells.map(c => c.T))].sort((a, b) => a - b);
const allN = [...new Set(cells.map(c => c.N))].sort((a, b) => a - b);
const allShape = [...new Set(cells.map(c => c.shape))].sort();
const allLb = [...new Set(cells.map(c => c.lb))].sort((a, b) => a - b);
const lbDomain = [allLb[0], allLb[allLb.length - 1]];
state.lbMin = lbDomain[0];
state.lbMax = lbDomain[1];

// ─── Filter pipeline ──────────────────────────────────────────────
function applyFilters() {
  return cells.filter(c => {
    if (state.filterK.size && !state.filterK.has(c.k)) return false;
    if (state.filterT.size && !state.filterT.has(c.T)) return false;
    if (state.filterN.size && !state.filterN.has(c.N)) return false;
    if (state.filterShape.size && !state.filterShape.has(c.shape)) return false;
    if (c.lb < state.lbMin || c.lb > state.lbMax) return false;
    if (state.flatMax != null && (!isFinite(c.flatness_std) ||
        c.flatness_std > state.flatMax)) return false;
    return true;
  });
}

// ─── Color generators ─────────────────────────────────────────────
function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2*l - 1)) * s;
  const x = c * (1 - Math.abs(((h/60) % 2) - 1));
  const m = l - c/2;
  let r, g, b;
  if (h < 60)      [r, g, b] = [c, x, 0];
  else if (h < 120)[r, g, b] = [x, c, 0];
  else if (h < 180)[r, g, b] = [0, c, x];
  else if (h < 240)[r, g, b] = [0, x, c];
  else if (h < 300)[r, g, b] = [x, 0, c];
  else             [r, g, b] = [c, 0, x];
  const hex = v => Math.round((v + m) * 255).toString(16).padStart(2, '0');
  return '#' + hex(r) + hex(g) + hex(b);
}
function colorForCell(c, visible) {
  const axis = state.colorBy;
  if (axis === "shape") return SHAPE_COLOR[c.shape] || "#aaa";
  const vals = visible.map(v => v[axis]).filter(v => v != null);
  if (!vals.length) return "#88ccff";
  const min = Math.min(...vals), max = Math.max(...vals);
  if (max === min) return "#88ccff";
  const t = (c[axis] - min) / (max - min);
  return hslToHex(220 * (1 - t), 0.65, 0.62);
}

// ─── Plot (SVG) ────────────────────────────────────────────────────
function renderPlot(visible) {
  const svg = document.getElementById("plot-svg");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const W = 1200, H = 580, ML = 60, MR = 30, MT = 30, MB = 50;
  const PW = W - ML - MR, PH = H - MT - MB;

  // X = log10(t), Y = d_s
  let tMin = Infinity, tMax = -Infinity;
  let dMin = Infinity, dMax = -Infinity;
  for (const c of visible) {
    for (let i = 0; i < c.t.length; i++) {
      if (c.in_window[i] && isFinite(c.d[i])) {
        const t = c.t[i];
        if (t > 0 && t < tMin) tMin = t;
        if (t > tMax) tMax = t;
        if (c.d[i] < dMin) dMin = c.d[i];
        if (c.d[i] > dMax) dMax = c.d[i];
      }
    }
  }
  if (state.showTorus) {
    for (const tr of toruses) {
      for (let i = 0; i < tr.t.length; i++) {
        if (tr.in_window[i] && isFinite(tr.d[i])) {
          const t = tr.t[i];
          if (t > 0 && t < tMin) tMin = t;
          if (t > tMax) tMax = t;
          if (tr.d[i] < dMin) dMin = tr.d[i];
          if (tr.d[i] > dMax) dMax = tr.d[i];
        }
      }
    }
  }
  if (!isFinite(tMin) || !isFinite(dMin)) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", W/2); t.setAttribute("y", H/2);
    t.setAttribute("fill", "#666"); t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-size", "13"); t.textContent = "no data in selection";
    svg.appendChild(t); return;
  }
  const logTMin = Math.log10(tMin), logTMax = Math.log10(tMax);
  const dLow = Math.min(0.5, dMin - 0.5);
  const dHigh = Math.max(4.5, Math.ceil((dMax + 0.5) * 2) / 2);
  const xOf = t => ML + (Math.log10(t) - logTMin) / (logTMax - logTMin) * PW;
  const yOf = d => MT + (1 - (d - dLow) / (dHigh - dLow)) * PH;

  const ns = "http://www.w3.org/2000/svg";
  const mk = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };

  // Frame
  svg.appendChild(mk("rect", {x: ML, y: MT, width: PW, height: PH,
    fill: "#0d0d12", stroke: "#333"}));

  // Reference lines d=2 and d=4
  for (const [d, lab, col] of [[4, "d_s = 4", "#7ec96e"],
                                [2, "d_s = 2", "#e5b06b"]]) {
    if (d >= dLow && d <= dHigh) {
      const y = yOf(d);
      svg.appendChild(mk("line", {x1: ML, x2: ML+PW, y1: y, y2: y,
        stroke: col, "stroke-width": 1, "stroke-dasharray": "5,4",
        opacity: 0.5}));
      svg.appendChild(mk("text", {x: ML+PW-8, y: y-3, fill: col,
        "font-size": 10, "text-anchor": "end", opacity: 0.85}, lab));
    }
  }
  // Y gridlines
  const yStep = (dHigh - dLow) <= 8 ? 1 : 2;
  for (let d = Math.ceil(dLow); d <= Math.floor(dHigh); d += yStep) {
    const y = yOf(d);
    svg.appendChild(mk("line", {x1: ML, x2: ML+PW, y1: y, y2: y,
      stroke: "#222", "stroke-width": 0.5}));
    svg.appendChild(mk("text", {x: ML-6, y: y+4, fill: "#888",
      "font-size": 10, "text-anchor": "end"}, d));
  }
  // X gridlines (decades)
  for (let lt = Math.ceil(logTMin); lt <= Math.floor(logTMax); lt++) {
    const x = xOf(Math.pow(10, lt));
    if (x < ML || x > ML+PW) continue;
    svg.appendChild(mk("line", {x1: x, x2: x, y1: MT, y2: MT+PH,
      stroke: "#222", "stroke-width": 0.5}));
    const txt = mk("text", {x: x, y: MT+PH+15, fill: "#888",
      "font-size": 10, "text-anchor": "middle"});
    txt.appendChild(document.createTextNode("10"));
    const sup = mk("tspan", {dy: -4, "font-size": 8}, lt);
    txt.appendChild(sup);
    svg.appendChild(txt);
  }
  // Axis titles
  svg.appendChild(mk("text", {x: ML+PW/2, y: MT+PH+38, fill: "#aaa",
    "font-size": 11, "text-anchor": "middle"}, "diffusion time t (log)"));
  svg.appendChild(mk("text", {x: 18, y: MT+PH/2, fill: "#aaa",
    "font-size": 11, "text-anchor": "middle",
    transform: `rotate(-90 18 ${MT+PH/2})`}, "d_s(t)"));

  // Torus refs first (background)
  if (state.showTorus) {
    for (const tr of toruses) {
      const path = buildPath(tr, xOf, yOf, dLow, dHigh, true);
      if (path.in) svg.appendChild(mk("path", {d: path.in, fill: "none",
        stroke: "#888", "stroke-width": 1.5, "stroke-dasharray": "4,3",
        opacity: 0.8}));
    }
  }
  // User curves
  for (const c of visible) {
    const color = colorForCell(c, visible);
    const isHi = (state.highlightTag === c.tag);
    const path = buildPath(c, xOf, yOf, dLow, dHigh, false);
    if (path.in) svg.appendChild(mk("path", {d: path.in, fill: "none",
      stroke: color, "stroke-width": isHi ? 3.0 : 1.6,
      opacity: isHi ? 1.0 : (state.highlightTag ? 0.3 : 0.9),
      "data-tag": c.tag, "class": "curve"}));
  }
}

function buildPath(curve, xOf, yOf, dLow, dHigh, isTorus) {
  const t = curve.t, d = curve.d, w = curve.in_window;
  const inSegs = [], outSegs = [];
  let curIn = [], curOut = [];
  for (let i = 0; i < t.length; i++) {
    if (!isFinite(t[i]) || !isFinite(d[i])) {
      if (curIn.length) { inSegs.push(curIn); curIn = []; }
      if (curOut.length) { outSegs.push(curOut); curOut = []; }
      continue;
    }
    const yClip = Math.max(dLow, Math.min(dHigh, d[i]));
    const x = xOf(t[i]), y = yOf(yClip);
    if (w[i]) {
      if (curOut.length) { outSegs.push(curOut); curOut = []; }
      curIn.push([x, y]);
    } else {
      if (curIn.length) { inSegs.push(curIn); curIn = []; }
      curOut.push([x, y]);
    }
  }
  if (curIn.length) inSegs.push(curIn);
  if (curOut.length) outSegs.push(curOut);
  const toD = segs => segs.map(s => s.length ?
    "M" + s.map(p => p[0].toFixed(1)+","+p[1].toFixed(1)).join(" L") : ""
  ).filter(Boolean).join(" ");
  return {in: toD(inSegs), out: toD(outSegs)};
}

// ─── Filter UI ────────────────────────────────────────────────────
function renderFilterBar() {
  const bar = document.getElementById("filter-bar");
  bar.innerHTML = "";

  const mkChips = (label, allVals, key, asNum) => {
    const grp = document.createElement("div");
    grp.className = "filter-group";
    grp.innerHTML = `<div class="lbl">${label}</div>`;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const v of allVals) {
      const c = document.createElement("div");
      c.className = "chip" + (state[key].has(v) ? " on" : "");
      c.textContent = v;
      c.onclick = () => {
        if (state[key].has(v)) state[key].delete(v);
        else state[key].add(v);
        c.classList.toggle("on");          // sync clicked chip
        rerender();
      };
      chips.appendChild(c);
    }
    grp.appendChild(chips);
    bar.appendChild(grp);
  };

  mkChips("k", allK, "filterK");
  mkChips("T", allT, "filterT");
  mkChips("N", allN, "filterN");
  mkChips("shape", allShape, "filterShape");

  // lb range (numeric inputs — slider would be nicer but inputs work)
  const grpLb = document.createElement("div");
  grpLb.className = "filter-group";
  grpLb.innerHTML = `<div class="lbl">lb range</div>`;
  const row = document.createElement("div");
  row.className = "range-row";
  const inMin = document.createElement("input");
  inMin.type = "number"; inMin.step = "0.01";
  inMin.value = state.lbMin; inMin.min = lbDomain[0]; inMin.max = lbDomain[1];
  const inMax = document.createElement("input");
  inMax.type = "number"; inMax.step = "0.01";
  inMax.value = state.lbMax; inMax.min = lbDomain[0]; inMax.max = lbDomain[1];
  inMin.oninput = () => { state.lbMin = parseFloat(inMin.value); rerender(); };
  inMax.oninput = () => { state.lbMax = parseFloat(inMax.value); rerender(); };
  row.appendChild(inMin);
  row.appendChild(document.createTextNode("–"));
  row.appendChild(inMax);
  grpLb.appendChild(row);
  bar.appendChild(grpLb);

  // flatness max filter
  const grpFlat = document.createElement("div");
  grpFlat.className = "filter-group";
  grpFlat.innerHTML = `<div class="lbl">flatness_std max</div>`;
  const rowFlat = document.createElement("div");
  rowFlat.className = "range-row";
  const inFlat = document.createElement("input");
  inFlat.type = "number"; inFlat.step = "0.1";
  inFlat.placeholder = "—"; inFlat.title = "max allowed flatness_std";
  if (state.flatMax != null) inFlat.value = state.flatMax;
  inFlat.oninput = () => {
    const v = parseFloat(inFlat.value);
    state.flatMax = isFinite(v) ? v : null;
    rerender();
  };
  rowFlat.appendChild(inFlat);
  rowFlat.appendChild(document.createTextNode(" (flatter ≤)"));
  grpFlat.appendChild(rowFlat);
  bar.appendChild(grpFlat);

  // Color-by selector
  const grpCol = document.createElement("div");
  grpCol.className = "filter-group";
  grpCol.innerHTML = `<div class="lbl">color by</div>`;
  const colChips = document.createElement("div");
  colChips.className = "chips";
  for (const opt of ["lb", "k", "T", "N", "shape"]) {
    const c = document.createElement("div");
    c.className = "chip" + (state.colorBy === opt ? " on" : "");
    c.textContent = opt;
    c.onclick = () => {
      state.colorBy = opt;
      // single-select: clear all siblings, mark clicked
      for (const sib of colChips.children) sib.classList.remove("on");
      c.classList.add("on");
      rerender();
    };
    colChips.appendChild(c);
  }
  grpCol.appendChild(colChips);
  bar.appendChild(grpCol);

  // Torus toggle
  const grpTr = document.createElement("div");
  grpTr.className = "filter-group";
  grpTr.innerHTML = `<div class="lbl">torus ref</div>`;
  const trChip = document.createElement("div");
  trChip.className = "chip" + (state.showTorus ? " on" : "");
  trChip.textContent = state.showTorus ? "shown" : "hidden";
  trChip.onclick = () => {
    state.showTorus = !state.showTorus;
    trChip.className = "chip" + (state.showTorus ? " on" : "");
    trChip.textContent = state.showTorus ? "shown" : "hidden";
    rerender();
  };
  grpTr.appendChild(trChip);
  bar.appendChild(grpTr);

  // Clear button
  const clr = document.createElement("button");
  clr.className = "clear-btn";
  clr.textContent = "clear filters";
  clr.onclick = () => {
    state.filterK.clear(); state.filterT.clear();
    state.filterN.clear(); state.filterShape.clear();
    state.lbMin = lbDomain[0]; state.lbMax = lbDomain[1];
    state.flatMax = null; state.highlightTag = null;
    renderFilterBar(); rerender();
  };
  bar.appendChild(clr);
}

// ─── Leaderboard: top-N flattest cells ────────────────────────────
function renderLeaderboard(visible) {
  const lb = document.getElementById("leaderboard");
  lb.innerHTML = "";
  const flat = visible
    .filter(c => isFinite(c.flatness_std))
    .sort((a, b) => a.flatness_std - b.flatness_std)
    .slice(0, 5);
  const stable = computeNStability(visible).slice(0, 5);

  const card = (title, rows, emptyMsg) => {
    const div = document.createElement("div");
    div.style.flex = "1"; div.style.minWidth = "300px";
    div.innerHTML = `<h3>${title}</h3>`;
    if (!rows.length) {
      div.innerHTML += `<div class="row"><span class="key" style="color:#555">${emptyMsg}</span></div>`;
    } else {
      for (const r of rows) {
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = `<span class="key">${r.label}</span><span>${r.value}</span>`;
        div.appendChild(row);
      }
    }
    return div;
  };

  const wrap = document.createElement("div");
  wrap.style.display = "flex"; wrap.style.gap = "20px"; wrap.style.flexWrap = "wrap";
  wrap.appendChild(card(
    "FLATTEST CURVES (lower flatness_std = better)",
    flat.map(c => ({
      label: `k=${c.k} T=${c.T} lb=${c.lb} N=${c.N}`,
      value: `flat_σ=${c.flatness_std.toFixed(3)} ds_med=${c.ds_median.toFixed(2)} (${c.shape})`,
    })),
    "no cells in current selection"));
  wrap.appendChild(card(
    "MOST N-STABLE GROUPS (lower N-spread of ds_median = better)",
    stable.map(g => ({
      label: `k=${g.k} T=${g.T} lb=${g.lb} (${g.n} N values)`,
      value: `ds_med ∈ [${g.ds_med_min.toFixed(2)}, ${g.ds_med_max.toFixed(2)}], spread=${g.spread.toFixed(3)}`,
    })),
    "need ≥2 N values per (k, T, lb) to assess N-stability"));
  lb.appendChild(wrap);
}

function computeNStability(visible) {
  const groups = {};
  for (const c of visible) {
    const key = `${c.k}|${c.T}|${c.lb}`;
    if (!groups[key]) groups[key] = {k: c.k, T: c.T, lb: c.lb, cells: []};
    groups[key].cells.push(c);
  }
  const result = [];
  for (const key in groups) {
    const g = groups[key];
    if (g.cells.length < 2) continue;
    const meds = g.cells.map(c => c.ds_median).filter(isFinite);
    if (meds.length < 2) continue;
    const lo = Math.min(...meds), hi = Math.max(...meds);
    result.push({...g, n: g.cells.length, spread: hi - lo,
                 ds_med_min: lo, ds_med_max: hi});
  }
  result.sort((a, b) => a.spread - b.spread);
  return result;
}

// ─── Table ────────────────────────────────────────────────────────
function renderTable(visible) {
  const tbody = document.getElementById("tbody");
  const thead = document.getElementById("thead-row");
  thead.innerHTML = "";
  for (const col of TABLE_COLUMNS) {
    const th = document.createElement("th");
    th.textContent = col.label;
    if (col.left) th.className = "left";
    if (col.key === state.sortKey)
      th.className += (state.sortDesc ? " sort-desc" : " sort-asc");
    th.onclick = () => {
      if (state.sortKey === col.key) state.sortDesc = !state.sortDesc;
      else { state.sortKey = col.key; state.sortDesc = false; }
      rerender();
    };
    thead.appendChild(th);
  }

  const sorted = visible.slice().sort((a, b) => {
    const av = a[state.sortKey], bv = b[state.sortKey];
    if (typeof av === "string" || typeof bv === "string") {
      const cmp = String(av).localeCompare(String(bv));
      return state.sortDesc ? -cmp : cmp;
    }
    if (av == null || !isFinite(av)) return 1;
    if (bv == null || !isFinite(bv)) return -1;
    return state.sortDesc ? bv - av : av - bv;
  });

  tbody.innerHTML = "";
  for (const c of sorted) {
    const tr = document.createElement("tr");
    if (state.highlightTag === c.tag) tr.className = "highlighted";
    for (const col of TABLE_COLUMNS) {
      const td = document.createElement("td");
      if (col.left) td.className = "left";
      td.innerHTML = col.fmt(c[col.key]);
      tr.appendChild(td);
    }
    tr.onclick = () => {
      state.highlightTag = (state.highlightTag === c.tag) ? null : c.tag;
      rerender();
    };
    tbody.appendChild(tr);
  }
}

// ─── Status & combined rerender ───────────────────────────────────
// Tracks cells.length at page load — used by the live polling code
// to detect when new cells have arrived since the dashboard was
// rendered. The static DATA in the page is fixed; we can only show
// new curves by reloading the HTML.
const CELLS_AT_LOAD = cells.length;

function updateStatus(visible) {
  const s = document.getElementById("status");
  const parts = [];
  // X/Y from grid args takes precedence — that's the user's mental
  // model of "how much of my planned work is done"
  if (PROGRESS.expected) {
    const total = PROGRESS.expected;
    const done = PROGRESS.done_in_grid;
    const pct = (100 * done / total).toFixed(0);
    parts.push(`<b style="color:#7ec96e">${done}/${total}</b> grid cells gathered (${pct}%)`);
    if (PROGRESS.missing_in_grid) {
      parts.push(`<span style="color:#e5b06b">${PROGRESS.missing_in_grid} missing</span>`);
    }
    if (PROGRESS.auto_sweep_spawned) {
      parts.push(`<span style="color:#9cdcfe">sweep spawned (PID ${PROGRESS.sweep_pid})</span>`);
    }
  }
  parts.push(`${cells.length} cell(s) loaded`);
  parts.push(`${visible.length} matching filters`);
  parts.push(`${toruses.length} torus ref(s)`);
  s.innerHTML = parts.join(" | ");
}

function rerender() {
  const visible = applyFilters();
  updateStatus(visible);
  renderPlot(visible);
  renderLeaderboard(visible);
  renderTable(visible);
}

// ─── Build pill (visible version stamp) ───────────────────────────
// Populated once at load from the embedded BUILD constant. Updates
// when a sweep status arrives carrying its own script_hash, so when
// hot-reload restarts the sweep with new code, the pill flips
// colour and you can verify the version actually swapped.
function renderBuildPill() {
  const el = document.getElementById("build-pill");
  if (!el) return;
  const dh = BUILD.script_hash || "?";
  const sweep = window._latestSweepVersion || null;
  const sh = sweep ? sweep.script_hash : null;
  let html = `dash <span class="h">${dh}</span> · ${BUILD.rendered_at}`;
  if (sh) {
    html += ` &nbsp;|&nbsp; sweep <span class="sweep-h">${sh}</span>`;
    if (sweep.started_at_iso) {
      html += ` · ${sweep.started_at_iso}`;
    }
  }
  el.innerHTML = html;
  el.title = `dashboard: ${BUILD.script_path} hash=${dh} rendered=${BUILD.rendered_at}`
    + (sweep ? `\nsweep:     ${sweep.script_path||"ds4_search/sweep_runner.py"} hash=${sh} started=${sweep.started_at_iso||"?"}` : "");
}

renderFilterBar();
rerender();
renderBuildPill();

// ─── Live status polling ──────────────────────────────────────────
// The sweep script writes a small JSON to STATUS_JSON_PATH after
// each cell completes. We poll it from the dashboard while the page
// is open. If the file is missing or stale, the banner stays hidden.
// This is purely additive — the dashboard works fine without any
// status JSON present.
function fmtSec(s) {
  if (s == null || !isFinite(s)) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
  return `${Math.floor(s/3600)}h ${Math.round((s%3600)/60)}m`;
}
function fmtKey(o) {
  if (!o) return "—";
  return `k=${o.k} T=${o.T} lb=${o.lb} N=${o.N}`;
}

async function pollStatus() {
  const banner = document.getElementById("live-status");
  try {
    const r = await fetch(STATUS_JSON_PATH + "?_=" + Date.now(),
                          {cache: "no-store"});
    if (!r.ok) { banner.classList.remove("show"); return; }
    const s = await r.json();

    // Pick up sweep version info if present. When hot-reload restarts
    // the sweep, this hash changes and we re-render the pill so the
    // user sees the version flip — confirms the new code is active.
    if (s.build) {
      const prev = window._latestSweepVersion;
      window._latestSweepVersion = s.build;
      const changed = !prev || prev.script_hash !== s.build.script_hash;
      if (changed) renderBuildPill();
    }

    // Stale check — if the status hasn't been updated in 10× the
    // typical cell wall_s (or 5 min as a safety floor), treat as
    // stale so the banner doesn't lie about a long-dead sweep.
    const ageSec = (Date.now() / 1000) - (s.now || 0);
    const cellWall = s.last_cell?.wall_s || 30;
    const staleThreshold = Math.max(300, cellWall * 10);
    const isStale = (s.phase === "running" && ageSec > staleThreshold);
    const isTerminal = (s.phase === "done" || s.phase === "interrupted");

    banner.classList.remove("running", "done");
    banner.classList.add("show", isTerminal ? "done" : "running");
    const pct = s.n_total ? (100 * s.n_done / s.n_total).toFixed(1) : 0;
    let phaseLbl;
    if (isStale) {
      phaseLbl = "stale (last update " + fmtSec(ageSec) + " ago)";
    } else if (s.phase === "done") {
      phaseLbl = "completed";
    } else if (s.phase === "interrupted") {
      phaseLbl = "stopped (Ctrl-C) — plots reflect finished cells";
    } else if (s.phase === "restarting") {
      phaseLbl = `restarting <span style="color:#dcdcaa">(source change: ${s.restart_reason || "?"})</span>`;
    } else {
      phaseLbl = "running";
    }
    // "new since page load" — the page renders a static curve set at
    // load time, but the sweep keeps adding cells to disk. We can't
    // render the new curves without re-fetching the HTML, so we just
    // count and offer a refresh button. Comparing s.n_done (sweep's
    // own counter) to (CELLS_AT_LOAD - PROGRESS.done_in_grid) gives
    // the number of cells finished since load — this is independent
    // of how many were already done before the sweep started.
    const expectedAtLoad = (PROGRESS.done_in_grid != null)
      ? PROGRESS.done_in_grid : CELLS_AT_LOAD;
    const newSinceLoad = Math.max(0, s.n_done - 0); // sweep started fresh, so n_done = new since load if dashboard was rendered before sweep
    const cellsNowDone = (PROGRESS.done_in_grid || 0) + (s.n_done || 0);
    const cellsAddedToView = cellsNowDone - CELLS_AT_LOAD;
    const refreshHint = cellsAddedToView > 0
      ? `<a href="#" onclick="window.location.reload();return false;" style="color:#7ec96e;text-decoration:underline;">${cellsAddedToView} new — refresh to view</a>`
      : "";

    banner.innerHTML = `
      <div class="heading">sweep ${phaseLbl}
        ${refreshHint ? `&nbsp;&middot;&nbsp; ${refreshHint}` : ""}</div>
      <div class="grid">
        <div><span class="k">progress:</span> ${s.n_done}/${s.n_total}
          (${pct}%)${s.n_failed ? ` <span style="color:#f48771">${s.n_failed} failed</span>` : ""}</div>
        <div><span class="k">elapsed:</span> ${fmtSec(s.elapsed_sec)}</div>
        <div><span class="k">core-time:</span> <span title="Sum of every cell's own wall time. Cells run in parallel, so this is total core-hours of compute, NOT elapsed wall-clock (which is roughly this divided by the worker count).">${s.compute_spent_sec != null ? fmtSec(s.compute_spent_sec) : "—"}</span></div>
        <div><span class="k">eta:</span> ${fmtSec(s.eta_sec)}</div>
        <div><span class="k">rate:</span> ${s.rate_per_min ?? "—"} cells/min</div>
        <div><span class="k">rss:</span> ${s.rss_mb ?? "—"} MB
          ${s.rss_peak_mb ? `<span class="k">(peak ${s.rss_peak_mb})</span>` : ""}</div>
        <div><span class="k">last:</span> ${fmtKey(s.last_cell)}${s.last_cell?.shape ?
          ` → <span style="color:${SHAPE_COLOR[s.last_cell.shape] || '#aaa'}">${s.last_cell.shape}</span>` : ""}</div>
        <div><span class="k">next:</span> ${fmtKey(s.next_cell)}</div>
      </div>
      <div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>`;
  } catch (e) {
    // No status JSON / unreachable / parse error → hide banner
    banner.classList.remove("show");
  }
}

// Poll on load and every 3s. When sweep phase=done, slow the poll
// to once every 30s so an idle dashboard doesn't hammer disk I/O.
let pollInterval = 3000;
async function pollLoop() {
  await pollStatus();
  const banner = document.getElementById("live-status");
  pollInterval = banner.classList.contains("done") ? 30000 : 3000;
  setTimeout(pollLoop, pollInterval);
}
pollLoop();
</script>

</body></html>"""
