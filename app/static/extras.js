// Tabs besides the missions: GPS spoofing, degradations, live camera. Uses $, fin, S, setKind from app.js.
"use strict";

const COL = { truth: "#f2f2f2", spoof: "#e0564b", fixed: "#4cb782", vis: "#e8a33d", muted: "#9a9a9a", grid: "#2e2e2e" };
const METHOD_COLORS = { SIFT: "#e8a33d", ORB: "#4cb782", AKAZE: "#d6d6d6" };
const dpr = () => window.devicePixelRatio || 1;

// ---------- navigation ----------
const inited = {};
function showView(name) {
  document.querySelectorAll(".navbtn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.dataset.view !== name));
  $("job-title").classList.toggle("hidden", name !== "missions");
  $("stage").classList.toggle("hidden", name !== "missions");
  if (name === "missions") { resizeCanvases(); S.dirty = true; drawChart(); }
  if (!inited[name] && VIEWS[name]) { inited[name] = true; VIEWS[name].init(); }
  else if (VIEWS[name]) VIEWS[name].redraw();
}
document.querySelectorAll(".navbtn").forEach((b) => (b.onclick = () => showView(b.dataset.view)));
window.addEventListener("resize", () => Object.entries(VIEWS).forEach(([k, v]) => inited[k] && v.redraw()));

// ---------- helpers ----------
function sizeCanvas(c) {
  c.width = Math.max(10, c.clientWidth * dpr());
  c.height = Math.max(10, c.clientHeight * dpr());
  const ctx = c.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  return ctx;
}
function fitImage(c, size, focus) { // map px -> canvas px; optional focus box [x0,y0,x1,y1] in map px
  let [x0, y0, x1, y1] = focus || [0, 0, size[0], size[1]];
  const pad = focus ? 0.12 * Math.max(x1 - x0, y1 - y0) : 0;
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  const s = Math.min(c.width / (x1 - x0), c.height / (y1 - y0));
  const ox = (c.width - (x1 - x0) * s) / 2 - x0 * s, oy = (c.height - (y1 - y0) * s) / 2 - y0 * s;
  return { s, ox, oy, P: (x, y) => [ox + x * s, oy + y * s] };
}
function drawMapImage(ctx, img, size, T, dim = 0.3) {
  if (!img) return;
  ctx.drawImage(img, T.ox, T.oy, size[0] * T.s, size[1] * T.s);
  ctx.fillStyle = `rgba(0,0,0,${dim})`;
  ctx.fillRect(T.ox, T.oy, size[0] * T.s, size[1] * T.s);
}
function line(ctx, pts, T, color, width, dash = []) {
  ctx.strokeStyle = color; ctx.lineWidth = width * dpr(); ctx.setLineDash(dash.map((d) => d * dpr()));
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    if (!p || !fin(p[0])) { pen = false; continue; }
    const [x, y] = T.P(p[0], p[1]);
    pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    pen = true;
  }
  ctx.stroke(); ctx.setLineDash([]);
}
function dot(ctx, x, y, r, fill, stroke = "#111") {
  ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1.5 * dpr();
  ctx.beginPath(); ctx.arc(x, y, r * dpr(), 0, 7); ctx.fill(); ctx.stroke();
}
function poly(ctx, pts, T) {
  ctx.beginPath();
  pts.forEach((p, i) => { const [x, y] = T.P(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath();
}
function loadImage(src) {
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => bad(new Error("не вдалося завантажити " + src)); i.src = src; });
}
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function basemapUrl(kind, mapPath, fallback) {
  return kind === "map" ? fallback : `/api/basemap/${kind}?map=${encodeURIComponent(mapPath)}`;
}

// Line chart: series [{x:[], y:[], color, width, dash, points, markers}] + optional vlines / hline / legend.
function chart(c, { series, xMax, yMax, xLabel, yLabel, vlines = [], hline, xTicks, legend }) {
  const ctx = sizeCanvas(c), r = dpr(), W = c.width, H = c.height;
  const L = 52 * r, R = 14 * r, B = 34 * r, Tp = (26 + (legend ? Math.max(0, legend.length - 1) * 16 : 0)) * r;
  const X = (x) => L + (x / (xMax || 1)) * (W - L - R), Y = (y) => H - B - (Math.min(y, yMax) / yMax) * (H - Tp - B);
  ctx.font = `${11 * r}px Segoe UI, sans-serif`; ctx.fillStyle = COL.muted; ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(v < 10 ? 1 : 0), 8 * r, y + 4 * r);
  }
  const ticks = xTicks || [0, xMax / 4, xMax / 2, (3 * xMax) / 4, xMax].map((v) => [v, Math.round(v)]);
  ticks.forEach(([v, t], i) => {
    const tw = ctx.measureText(String(t)).width;
    ctx.fillText(String(t), i === ticks.length - 1 ? X(v) - tw : i === 0 ? X(v) : X(v) - tw / 2, H - B + 16 * r);
  });
  ctx.fillText(xLabel || "", L, H - 4 * r);
  ctx.fillText(yLabel || "", 8 * r, 14 * r);
  if (hline) {
    ctx.strokeStyle = COL.muted; ctx.setLineDash([3 * r, 4 * r]);
    ctx.beginPath(); ctx.moveTo(L, Y(hline.y)); ctx.lineTo(W - R, Y(hline.y)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText(hline.label, L + 6 * r, Y(hline.y) - 5 * r);
  }
  for (const v of vlines) {
    ctx.strokeStyle = v.color; ctx.lineWidth = (v.width || 1.5) * r;
    ctx.beginPath(); ctx.moveTo(X(v.x), Tp); ctx.lineTo(X(v.x), H - B); ctx.stroke();
    if (v.label) { ctx.fillStyle = v.color; ctx.fillText(v.label, X(v.x) + 5 * r, Tp + 12 * r); }
  }
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = (s.width || 2) * r; ctx.setLineDash((s.dash || []).map((d) => d * r));
    if (!s.points) {
      ctx.beginPath();
      let pen = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (!fin(y)) { pen = false; return; }
        pen ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y));
        pen = true;
      });
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (s.points || s.markers) s.x.forEach((x, i) => fin(s.y[i]) && dot(ctx, X(x), Y(s.y[i]), s.points ? 2.6 : 3.4, s.color, s.points ? s.color : "#111"));
  }
  if (legend) {
    ctx.font = `${12 * r}px Segoe UI, sans-serif`;
    let x = L + 160 * r;  // one row of legend entries above the plot, right of the y label
    legend.forEach(([label, color]) => {
      ctx.fillStyle = color; ctx.fillRect(x, 10 * r, 14 * r, 4 * r);
      ctx.fillStyle = "#cfcfcf"; ctx.fillText(label, x + 20 * r, 15 * r);
      x += ctx.measureText(label).width + 44 * r;
    });
  }
}

// ---------- GPS spoofing ----------
const SP = { sources: [], img: null, data: null };
const SOURCE_HINTS = [
  ["вікно", "Кадр шукається у вікні навколо справжньої позиції: верхня межа можливого, а не чесний політ."],
  ["з цього польоту", "Без підказки позиції, але масштаб кадру виміряний на цьому ж польоті."],
  ["без підказки", "Чесний режим: без підказки позиції й масштабу. Прив'язок мало — це поточна межа методу."],
  ["Мої місії", "Прив'язки з вашої місії: метод шукав кадри на мапі без GPS."],
];
const spoofView = {
  async init() {
    const run = debounce(() => spoofView.run(), 60);
    for (const id of ["sp-start", "sp-rate", "sp-max", "sp-bearing", "sp-threshold", "sp-confirm"]) $(id).oninput = run;
    $("sp-source").onchange = () => spoofView.pick();
    $("sp-basemap").onchange = () => spoofView.loadMap();
    $("sp-refresh").onclick = () => spoofView.load();
    await spoofView.load();
  },
  async load() {
    const prev = $("sp-source").value;
    const r = await getJSON("/api/spoof/sources");
    SP.sources = r.sources;
    $("sp-unavailable").textContent = r.hint || "";
    const sel = $("sp-source");
    sel.innerHTML = "";
    const groups = {};
    for (const s of r.sources) {
      if (!groups[s.group]) { groups[s.group] = document.createElement("optgroup"); groups[s.group].label = s.group; sel.appendChild(groups[s.group]); }
      groups[s.group].appendChild(new Option(`${s.label} — прив'язано ${s.fixes_pct}% кадрів`, s.id));
    }
    if (r.sources.some((s) => s.id === prev)) sel.value = prev;
    if (r.sources.length) await spoofView.pick();
  },
  src() { return SP.sources.find((s) => s.id === $("sp-source").value); },
  async pick() {
    const s = spoofView.src();
    setKind($("sp-kind"), s.kind);
    $("sp-source-hint").textContent = SOURCE_HINTS.find(([k]) => s.group.includes(k))?.[1] || "";
    $("sp-start").max = Math.max(1, s.n_frames - 10);
    if (+$("sp-start").value > +$("sp-start").max) $("sp-start").value = Math.round(s.n_frames / 3);
    await spoofView.loadMap();
    await spoofView.run();
  },
  async loadMap() {
    const s = spoofView.src(), own = `/api/spoof/map?source=${encodeURIComponent(s.id)}`;
    try {
      SP.img = await loadImage(basemapUrl($("sp-basemap").value, s.map, own));
    } catch {
      $("sp-unavailable").textContent = "Підкладка недоступна (немає інтернету?)";
      $("sp-basemap").value = "map";
      SP.img = await loadImage(own);
    }
    spoofView.redraw();
  },
  async run() {
    const v = (id) => $(id).value, s = spoofView.src();
    if (!s) return;
    $("sp-start-v").textContent = v("sp-start"); $("sp-rate-v").textContent = v("sp-rate");
    $("sp-max-v").textContent = v("sp-max"); $("sp-bearing-v").textContent = v("sp-bearing");
    $("sp-threshold-v").textContent = v("sp-threshold"); $("sp-confirm-v").textContent = v("sp-confirm");
    const q = new URLSearchParams({ source: s.id, start: v("sp-start"), rate: v("sp-rate"), max_m: v("sp-max"),
      bearing: v("sp-bearing"), threshold: v("sp-threshold"), confirm: v("sp-confirm") });
    SP.data = await getJSON(`/api/spoof/run?${q}`);
    spoofView.redraw();
  },
  redraw() {
    const d = SP.data;
    if (!d) return;
    const c = $("sp-map"), ctx = sizeCanvas(c), size = d.map_size;
    const T = fitImage(c, size);
    drawMapImage(ctx, SP.img, size, T, 0.3);
    const a = d.stats.alarm;
    line(ctx, d.truth, T, COL.truth, 2);
    line(ctx, d.gps.slice(d.start), T, COL.spoof, 2, [6, 5]);
    if (a !== null) line(ctx, d.position.slice(a), T, COL.fixed, 2.5);
    d.visual.forEach((p) => p && fin(p[0]) && dot(ctx, ...T.P(p[0], p[1]), 3.2, COL.vis));
    const s0 = T.P(...d.gps[d.start]);
    ctx.strokeStyle = COL.spoof; ctx.lineWidth = 2.5 * dpr();
    ctx.beginPath(); ctx.arc(s0[0], s0[1], 10 * dpr(), 0, 7); ctx.stroke();
    if (a !== null) { const p = T.P(...d.gps[a]); dot(ctx, p[0], p[1], 8, COL.spoof, "#fff"); }

    const st = d.stats, n = d.truth.length;
    $("sp-s-alarm").textContent = a === null ? "немає" : `кадр ${a} (+${st.delay_frames})`;
    $("sp-s-drag").textContent = a === null ? "—" : `${st.drag_m.toFixed(0)} м`;
    $("sp-s-gps").textContent = `${st.err_gps_m.toFixed(0)} м`;
    $("sp-s-fixed").textContent = a === null ? "—" : `${st.err_fixed_m.toFixed(1)} м`;
    $("sp-s-vis").textContent = fin(st.vis_err_median_m) ? `${st.vis_err_median_m.toFixed(1)} м · ${st.fixes} кадрів` : "немає прив'язок";
    $("sp-s-false").textContent = st.false_alarm === null ? "немає" : `так, кадр ${st.false_alarm}`;
    $("sp-s-false").style.color = st.false_alarm === null ? "" : "var(--bad)";
    const xs = [...Array(n).keys()];
    const nav = d.err_nav_m.map((e, i) => (a !== null && i >= a ? e : null));
    const yMax = Math.max(60, ...d.err_gps_m.filter(fin)) * 1.08;
    chart($("sp-chart"), {
      xMax: n - 1, yMax, xLabel: "кадр", yLabel: "похибка позиції, м",
      hline: { y: d.threshold, label: `поріг ${d.threshold} м` },
      vlines: [{ x: d.start, color: COL.muted, width: 1, label: "атака" }, ...(a !== null ? [{ x: a, color: COL.spoof, label: "тривога" }] : [])],
      series: [{ x: xs, y: d.err_gps_m, color: COL.spoof }, { x: xs, y: nav, color: COL.fixed, width: 2.5 },
        { x: xs, y: d.residual_m, color: COL.vis, points: true }],
      legend: [["якщо вірити GPS", COL.spoof], ["з виправленням камерою", COL.fixed], ["|GPS − прив'язка камери|", COL.vis]],
    });
  },
};

// ---------- degradations ----------
const ST = { info: null, img: null, res: null, curves: {} };
const AXIS_NAMES = { blur: "Розмиття (рух)", noise: "Шум сенсора", clouds: "Хмари", darkness: "Темрява", altitude: "Висота", tilt: "Нахил камери" };
function levelText(axis, v) {
  return { blur: `${v} px`, noise: `σ = ${v}`, clouds: `${Math.round(100 * v)}% кадру`, darkness: `експозиція ×${v}`,
    altitude: `висота ×${v}`, tilt: `${v}e-4` }[axis] || String(v);
}
const stressView = {
  async init() {
    ST.info = await getJSON("/api/stress/info");
    for (const m of ST.info.methods) $("st-method").add(new Option(m.label, m.id));
    const run = debounce(() => stressView.run(), 120);
    $("st-source").onchange = () => stressView.pickSource();
    $("st-axis").onchange = () => { stressView.fitLevels(); run(); stressView.drawCurves(); };
    for (const id of ["st-level", "st-method", "st-frame"]) $(id).oninput = run;
    $("st-level").addEventListener("input", () => stressView.drawCurves());
    $("st-curve-btn").onclick = () => stressView.buildCurves();
    $("st-upload").onclick = () => stressView.upload();
    await stressView.fillSources();
  },
  async fillSources(select) {
    ST.info = await getJSON("/api/stress/info");
    const sel = $("st-source");
    sel.innerHTML = "";
    for (const s of ST.info.sources) sel.add(new Option(s.label, s.id));
    $("st-error").textContent = ST.info.hint || "";
    if (select) sel.value = select;
    for (const s of ST.info.sources) for (const cv of s.saved || []) {
      const m = ST.info.methods.find((x) => x.label === cv.method);
      if (m && !ST.curves[`${s.id}|${cv.axis}|${m.id}`]) ST.curves[`${s.id}|${cv.axis}|${m.id}`] = { ...cv, saved: true };
    }
    if (ST.info.sources.length) await stressView.pickSource();
    else $("st-upload-box").open = true;
  },
  source() { return ST.info.sources.find((s) => s.id === $("st-source").value); },
  axis() { return ST.info.axes.find((a) => a.name === $("st-axis").value); },
  fitLevels() {
    const a = stressView.axis();
    $("st-level").max = a.levels.length - 1;
    if (+$("st-level").value > a.levels.length - 1) $("st-level").value = a.levels.length - 1;
  },
  async pickSource() {
    const s = stressView.source(), ax = $("st-axis"), prev = ax.value || "clouds";
    setKind($("st-kind"), s.kind === "own" ? "real_nogt" : s.kind);
    if (s.kind === "own") $("st-kind").textContent = "Свій кадр: відповідь — його місце без спотворень";
    ax.innerHTML = "";
    for (const a of ST.info.axes) if (s.geometric || !a.geometric) ax.add(new Option(AXIS_NAMES[a.name] || a.label, a.name));
    ax.value = [...ax.options].some((o) => o.value === prev) ? prev : "clouds";
    $("st-frame-row").classList.toggle("hidden", s.frames <= 1);
    $("st-frame").max = s.frames - 1;
    stressView.fitLevels();
    try { ST.img = await loadImage(`/api/stress/map?source=${encodeURIComponent(s.id)}`); } catch { ST.img = null; }
    ST.res = null;
    await stressView.run();
    stressView.drawCurves();
  },
  async upload() {
    $("st-error").textContent = "";
    const f = $("st-frame-file").files[0];
    if (!f) { $("st-error").textContent = "Оберіть кадр."; return; }
    const fd = new FormData();
    fd.append("frame", f);
    if ($("st-map-file").files[0]) fd.append("map_file", $("st-map-file").files[0]);
    fd.append("method", $("st-method").value || "sift_flann");
    fd.append("frame_scale", $("st-scale").value);
    $("st-upload").disabled = true; $("st-upload").textContent = "Шукаю кадр на мапі…";
    try {
      const r = await getJSON("/api/stress/upload", { method: "POST", body: fd });
      await stressView.fillSources(r.id);
      $("st-upload-box").open = false;
    } catch (e) { $("st-error").textContent = e.message; }
    finally { $("st-upload").disabled = false; $("st-upload").textContent = "Завантажити"; }
  },
  async run() {
    const s = stressView.source(), a = stressView.axis();
    if (!s || !a) return;
    const li = +$("st-level").value;
    $("st-level-v").textContent = levelText(a.name, a.levels[li]);
    $("st-frame-v").textContent = `${+$("st-frame").value + 1} / ${s.frames}`;
    $("st-result").textContent = "Шукаю…"; $("st-result").className = "result";
    const q = new URLSearchParams({ source: s.id, axis: a.name, level_idx: li, method: $("st-method").value, frame: $("st-frame").value });
    try { ST.res = await getJSON(`/api/stress/try?${q}`); } catch (e) { $("st-error").textContent = e.message; return; }
    const r = ST.res;
    $("st-img").src = r.image;
    $("st-result").textContent = r.success ? `Знайдено, похибка ${r.err.toFixed(1)} ${r.units}`
      : r.ok ? `Знайдено не там: ${r.err.toFixed(0)} ${r.units}` : "Не знайдено";
    $("st-result").className = "result " + (r.success ? "ok" : "bad");
    $("st-detail").textContent = `inliers ${r.n_inliers}, ${r.ms.toFixed(0)} мс` + (r.ok ? "" : ` · ${r.reason || "замало відповідностей"}`)
      + ` · успіх: ближче ${ST.info.success_m} ${r.units} до правильного місця`;
    stressView.redrawMap();
  },
  async buildCurves() {
    const s = stressView.source(), a = stressView.axis(), btn = $("st-curve-btn");
    btn.disabled = true;
    for (const m of ST.info.methods) {
      $("st-curve-status").textContent = `рахую ${m.label}: ${s.frames} кадр. × ${a.levels.length} рівнів…`;
      try {
        ST.curves[`${s.id}|${a.name}|${m.id}`] = await getJSON(`/api/stress/curve?${new URLSearchParams({ source: s.id, axis: a.name, method: m.id })}`);
      } catch (e) { $("st-error").textContent = e.message; break; }
      stressView.drawCurves();
    }
    $("st-curve-status").textContent = "";
    btn.disabled = false;
  },
  redraw() { stressView.redrawMap(); stressView.drawCurves(); },
  redrawMap() {
    const r = ST.res;
    if (!ST.img || !r) return;
    const c = $("st-map"), ctx = sizeCanvas(c), size = r.map_size;
    const T = fitImage(c, size);
    drawMapImage(ctx, ST.img, size, T, 0.25);
    const g = T.P(...r.ref);
    ctx.strokeStyle = "#fff"; ctx.setLineDash([4 * dpr(), 3 * dpr()]); ctx.lineWidth = 2 * dpr();
    ctx.beginPath(); ctx.arc(g[0], g[1], 16 * dpr(), 0, 7); ctx.stroke(); ctx.setLineDash([]);
    if (r.corners && r.corners.every((p) => p && fin(p[0]))) {
      ctx.strokeStyle = r.success ? COL.fixed : COL.spoof; ctx.lineWidth = 2.5 * dpr();
      poly(ctx, r.corners, T); ctx.stroke();
    }
  },
  drawCurves() {
    const s = stressView.source(), a = stressView.axis();
    if (!s || !a) return;
    const series = [], legend = [];
    for (const m of ST.info.methods) {
      const cv = ST.curves[`${s.id}|${a.name}|${m.id}`];
      if (!cv) continue;
      const col = METHOD_COLORS[m.label] || "#bbb";
      series.push({ x: cv.levels.map((_, i) => i), y: cv.success.map((v) => 100 * v), color: col, markers: true });
      legend.push([`${m.label} (${cv.frames} кадр.${cv.saved ? ", збережений прогін" : ""})`, col]);
    }
    $("st-curve-btn").textContent = series.length ? "Перерахувати" : "Порахувати для 3 методів";
    chart($("st-chart"), {
      xMax: a.levels.length - 1, yMax: 100, xLabel: "сила спотворення", yLabel: "знайдено кадрів, %",
      xTicks: a.levels.map((v, i) => [i, levelText(a.name, v)]),
      vlines: [{ x: +$("st-level").value, color: "#5a5a5a", width: 6 }], series, legend,
    });
  },
};

// ---------- live camera ----------
const LV = { id: null, img: null, size: null, previewSize: null, running: false, stream: null, trail: [], last: null, n: 0, reg: 0, fps: 0 };
const liveView = {
  info: null,
  async init() {
    liveView.info = await getJSON("/api/live/info");
    for (const m of liveView.info.maps) $("lv-map").add(new Option(m.label, m.id));
    for (const m of liveView.info.methods) $("lv-method").add(new Option(m.label, m.id));
    if (!liveView.info.maps.length) $("lv-error").textContent = "Немає жодної мапи локально (scripts/download_data.py).";
    $("lv-source").onchange = () => {
      const cam = $("lv-source").value === "camera";
      $("lv-camera-row").classList.toggle("hidden", !cam); $("lv-file-row").classList.toggle("hidden", cam);
    };
    $("lv-start").onclick = () => liveView.start();
    $("lv-stop").onclick = () => liveView.stop();
    $("lv-follow").onchange = () => liveView.redraw();
    $("lv-basemap").onchange = () => liveView.loadMap();
    $("lv-map").onchange = async () => {
      if (LV.running) return;
      LV.last = null; LV.trail = []; LV.size = null;
      await liveView.loadMap();
    };
    if (liveView.info.maps.length) await $("lv-map").onchange();
    await liveView.listCameras();
    if (new URLSearchParams(location.hash.slice(1)).get("autostart")) liveView.start();
  },
  mapInfo() { return liveView.info.maps.find((m) => m.id === $("lv-map").value); },
  async loadMap() {
    const m = liveView.mapInfo(), base = `/api/live/preview/${m.id}`;
    $("lv-basemap").classList.toggle("hidden", !m.geo);
    if (!m.geo) $("lv-basemap").value = "map";
    const preview = await loadImage(base);
    LV.previewSize = [preview.naturalWidth, preview.naturalHeight];
    try { LV.img = $("lv-basemap").value === "map" ? preview : await loadImage(basemapUrl($("lv-basemap").value, m.map, base)); }
    catch { $("lv-error").textContent = "Підкладка недоступна (немає інтернету?)"; $("lv-basemap").value = "map"; LV.img = preview; }
    liveView.redraw();
  },
  async listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) { $("lv-camera").add(new Option("камера недоступна в цьому браузері", "")); return; }
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const prev = $("lv-camera").value;
    $("lv-camera").innerHTML = "";
    devs.forEach((d, i) => $("lv-camera").add(new Option(d.label || `Камера ${i + 1}`, d.deviceId)));
    if (!devs.length) $("lv-camera").add(new Option("за замовчуванням", ""));
    if (devs.some((d) => d.deviceId === prev)) $("lv-camera").value = prev;
  },
  async start() {
    $("lv-error").textContent = "";
    $("lv-start").disabled = true; $("lv-start").textContent = "Готую мапу…";
    try {
      const fd = new FormData();
      fd.append("map_id", $("lv-map").value); fd.append("method", $("lv-method").value); fd.append("frame_scale", $("lv-scale").value);
      const body = await getJSON("/api/live/start", { method: "POST", body: fd });
      Object.assign(LV, { id: body.id, size: body.map_size, trail: [], last: null, n: 0, reg: 0, fps: 0 });
      const video = $("lv-video");
      if ($("lv-source").value === "camera") {
        const id = $("lv-camera").value;
        LV.stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: id ? { exact: id } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } } });
        video.srcObject = LV.stream; video.loop = false;
        liveView.listCameras(); // labels are known only after the permission
      } else {
        const f = $("lv-file").files[0];
        if (!f) throw new Error("Оберіть відеофайл.");
        video.srcObject = null; video.src = URL.createObjectURL(f); video.loop = true;
      }
      await video.play();
      $("lv-empty").classList.add("hidden");
      LV.running = true;
      $("lv-start").classList.add("hidden"); $("lv-stop").classList.remove("hidden");
      liveView.loop();
    } catch (e) {
      $("lv-error").textContent = e.name === "NotAllowedError" ? "Браузер не дав доступ до камери." : e.message;
    } finally {
      $("lv-start").disabled = false; $("lv-start").textContent = "Почати";
    }
  },
  stop() {
    LV.running = false;
    if (LV.stream) LV.stream.getTracks().forEach((t) => t.stop());
    LV.stream = null;
    $("lv-video").pause();
    $("lv-start").classList.remove("hidden"); $("lv-stop").classList.add("hidden");
  },
  async loop() {
    const video = $("lv-video"), off = document.createElement("canvas");
    while (LV.running) {
      if (!video.videoWidth) { await new Promise((r) => setTimeout(r, 50)); continue; }
      const k = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
      off.width = Math.round(video.videoWidth * k); off.height = Math.round(video.videoHeight * k);
      off.getContext("2d").drawImage(video, 0, 0, off.width, off.height);
      const blob = await new Promise((r) => off.toBlob(r, "image/jpeg", 0.85));
      const t0 = performance.now();
      let res;
      try {
        res = await getJSON(`/api/live/${LV.id}/frame`, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      } catch (e) { $("lv-error").textContent = e.message; liveView.stop(); return; }
      const dt = (performance.now() - t0) / 1000;
      LV.fps = LV.fps ? 0.8 * LV.fps + 0.2 / dt : 1 / dt;
      LV.n++; LV.last = res;
      if (res.ok) {
        LV.reg++;
        const prev = LV.trail[LV.trail.length - 1], xs = res.corners.map((p) => p[0]);
        const jump = prev && Math.hypot(res.x - prev[0], res.y - prev[1]) > 3 * (Math.max(...xs) - Math.min(...xs));
        if (jump) LV.trail.push(null); // relocalised far away: do not draw a line across the map
        LV.trail.push([res.x, res.y]);
        if (LV.trail.length > 400) LV.trail.shift();
      }
      liveView.stats(); liveView.redraw();
    }
  },
  stats() {
    const r = LV.last;
    $("lv-s-status").textContent = r.ok ? "прив'язано" : "втрачено";
    $("lv-s-status").style.color = r.ok ? "var(--ok)" : "var(--bad)";
    $("lv-s-fps").textContent = LV.fps.toFixed(1);
    $("lv-s-inl").textContent = r.n_inliers;
    $("lv-s-heading").textContent = r.ok && fin(r.heading) ? r.heading.toFixed(0) : "—";
    $("lv-s-pos").textContent = r.ok ? `${r.x.toFixed(0)}, ${r.y.toFixed(0)}` : "—";
    $("lv-s-reg").textContent = `${LV.reg} / ${LV.n}`;
  },
  redraw() {
    const size = LV.size || LV.previewSize;  // map px once the session runs, preview px before
    if (!LV.img || !size) return;
    const c = $("lv-map-canvas"), ctx = sizeCanvas(c), r = LV.last;
    let focus = null;
    if ($("lv-follow").checked && r && r.ok && r.corners) {
      const xs = r.corners.map((p) => p[0]), ys = r.corners.map((p) => p[1]);
      const ext = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.6;
      focus = [r.x - ext, r.y - ext, r.x + ext, r.y + ext];
    }
    const T = fitImage(c, size, focus);
    drawMapImage(ctx, LV.img, size, T, 0.25);
    line(ctx, LV.trail, T, COL.vis, 2.5);
    if (r && r.ok && r.corners && r.corners.every((p) => p && fin(p[0]))) {
      ctx.fillStyle = "rgba(255,255,255,.08)"; poly(ctx, r.corners, T); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr(); poly(ctx, r.corners, T); ctx.stroke();
      const [p0, p1] = [T.P(...r.corners[0]), T.P(...r.corners[1])];
      ctx.strokeStyle = COL.vis; ctx.lineWidth = 4 * dpr(); ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
      dot(ctx, ...T.P(r.x, r.y), 6, COL.spoof, "#fff");
    }
  },
};

const VIEWS = { spoof: spoofView, stress: stressView, live: liveView };
{ // deep link to a tab: #tab=spoof | stress | live
  const tab = new URLSearchParams(location.hash.slice(1)).get("tab");
  if (VIEWS[tab]) showView(tab);
}
