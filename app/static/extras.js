// Tabs besides the missions: GPS spoofing, degradations (stress test), live camera. Uses $ and fin from app.js.
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

// ---------- canvas helpers ----------
function sizeCanvas(c) {
  c.width = Math.max(10, c.clientWidth * dpr());
  c.height = Math.max(10, c.clientHeight * dpr());
  const ctx = c.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  return ctx;
}
function fitImage(c, img, size, focus) { // transform map px -> canvas px; optional focus box [x0,y0,x1,y1] in map px
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
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = src; });
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// Simple line chart: series [{x:[], y:[], color, width, dash, points}] + optional vlines / hline.
function chart(c, { series, xMax, yMax, xLabel, yLabel, vlines = [], hline, xTicks }) {
  const ctx = sizeCanvas(c), r = dpr(), W = c.width, H = c.height;
  const L = 52 * r, R = 14 * r, Tp = 26 * r, B = 34 * r;
  const X = (x) => L + (x / (xMax || 1)) * (W - L - R), Y = (y) => H - B - (Math.min(y, yMax) / yMax) * (H - Tp - B);
  ctx.font = `${11 * r}px Segoe UI, sans-serif`; ctx.fillStyle = COL.muted; ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(v < 10 ? 1 : 0), 8 * r, y + 4 * r);
  }
  (xTicks || [0, xMax / 4, xMax / 2, (3 * xMax) / 4, xMax].map((v) => [v, Math.round(v)])).forEach(([v, t]) =>
    ctx.fillText(String(t), X(v) - 6 * r, H - B + 16 * r));
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
  return { X, Y };
}

// ---------- GPS spoofing ----------
const SP = { info: null, img: null, data: null };
const spoofView = {
  async init() {
    SP.info = await (await fetch("/api/spoof/info")).json();
    if (!SP.info.available) { $("sp-unavailable").textContent = "Немає даних реального польоту: " + SP.info.hint; return; }
    const ms = $("sp-method");
    for (const m of SP.info.methods) ms.add(new Option(`${m.label} — прив'язано ${m.fixes}% кадрів`, m.id));
    $("sp-start").max = SP.info.n_frames - 21;
    SP.img = await loadImage("/api/spoof/map");
    const run = debounce(spoofView.run, 60);
    for (const id of ["sp-method", "sp-start", "sp-rate", "sp-max", "sp-bearing", "sp-threshold", "sp-confirm"]) $(id).oninput = run;
    spoofView.run();
  },
  async run() {
    const v = (id) => $(id).value;
    $("sp-start-v").textContent = v("sp-start"); $("sp-rate-v").textContent = v("sp-rate");
    $("sp-max-v").textContent = v("sp-max"); $("sp-bearing-v").textContent = v("sp-bearing");
    $("sp-threshold-v").textContent = v("sp-threshold"); $("sp-confirm-v").textContent = v("sp-confirm");
    const q = new URLSearchParams({ method: v("sp-method"), start: v("sp-start"), rate: v("sp-rate"), max_m: v("sp-max"),
      bearing: v("sp-bearing"), threshold: v("sp-threshold"), confirm: v("sp-confirm") });
    SP.data = await (await fetch(`/api/spoof/run?${q}`)).json();
    spoofView.redraw();
  },
  redraw() {
    const d = SP.data;
    if (!d) return;
    const c = $("sp-map"), ctx = sizeCanvas(c), size = SP.info.map_size;
    const T = fitImage(c, SP.img, size);
    drawMapImage(ctx, SP.img, size, { ...T, s: T.s * size[0] / size[0] }, 0.35);
    const a = d.stats.alarm;
    line(ctx, d.truth, T, COL.truth, 2);
    line(ctx, d.gps.slice(d.start), T, COL.spoof, 2, [6, 5]);
    if (a !== null) line(ctx, d.position.slice(a), T, COL.fixed, 2.5);
    d.visual.forEach((p) => p && fin(p[0]) && dot(ctx, ...T.P(p[0], p[1]), 3.2, COL.vis));
    const s = T.P(...d.gps[d.start]);
    ctx.strokeStyle = COL.spoof; ctx.lineWidth = 2.5 * dpr();
    ctx.beginPath(); ctx.arc(s[0], s[1], 10 * dpr(), 0, 7); ctx.stroke();
    if (a !== null) { const p = T.P(...d.gps[a]); dot(ctx, p[0], p[1], 8, COL.spoof, "#fff"); }

    const st = d.stats, n = d.truth.length;
    $("sp-s-alarm").textContent = a === null ? "немає" : `кадр ${a} (+${st.delay_frames})`;
    $("sp-s-drag").textContent = a === null ? "—" : `${st.drag_m.toFixed(0)} м`;
    $("sp-s-gps").textContent = `${st.err_gps_m.toFixed(0)} м`;
    $("sp-s-fixed").textContent = a === null ? "—" : `${st.err_fixed_m.toFixed(1)} м`;
    $("sp-s-false").textContent = st.false_alarm ? "так" : "немає";
    const xs = [...Array(n).keys()];
    const nav = d.err_nav_m.map((e, i) => (a !== null && i >= a ? e : null));
    const yMax = Math.max(60, ...d.err_gps_m.filter(fin)) * 1.08;
    chart($("sp-chart"), {
      xMax: n - 1, yMax, xLabel: "кадр", yLabel: "похибка позиції, м",
      hline: { y: d.threshold, label: `поріг ${d.threshold} м` },
      vlines: [{ x: d.start, color: COL.muted, width: 1, label: "атака" }, ...(a !== null ? [{ x: a, color: COL.spoof, label: "тривога" }] : [])],
      series: [{ x: xs, y: d.err_gps_m, color: COL.spoof }, { x: xs, y: nav, color: COL.fixed, width: 2.5 },
        { x: xs, y: d.residual_m, color: COL.vis, points: true }],
    });
  },
};

// ---------- degradations ----------
const ST = { info: null, img: null, res: null };
function levelText(axis, v) {
  return { blur: `${v} px`, noise: `σ = ${v}`, clouds: `${Math.round(100 * v)}% кадру`, darkness: `експозиція ×${v}`,
    altitude: `висота ×${v}`, tilt: `${v}e-4` }[axis] || String(v);
}
const stressView = {
  async init() {
    ST.info = await (await fetch("/api/stress/info")).json();
    const ax = $("st-axis"), ms = $("st-method");
    const names = { blur: "Розмиття (рух)", noise: "Шум сенсора", clouds: "Хмари", darkness: "Темрява", altitude: "Висота", tilt: "Нахил камери" };
    for (const a of ST.info.axes) ax.add(new Option(names[a.name] || a.label, a.name));
    for (const m of ST.info.methods) ms.add(new Option(m.label, m.id));
    ax.value = "clouds";
    $("st-frame").max = ST.info.n_frames - 1;
    if (!ST.info.available) { $("st-unavailable").textContent = ST.info.hint; stressView.redraw(); return; }
    ST.img = await loadImage("/api/stress/map");
    const run = debounce(stressView.run, 120);
    ax.onchange = () => { const a = stressView.axis(); $("st-level").max = a.levels.length - 1; $("st-level").value = Math.min($("st-level").value, a.levels.length - 1); run(); };
    for (const id of ["st-level", "st-method", "st-frame"]) $(id).oninput = run;
    ax.onchange();
  },
  axis() { return ST.info.axes.find((a) => a.name === $("st-axis").value); },
  async run() {
    const a = stressView.axis(), li = +$("st-level").value;
    $("st-level-v").textContent = levelText(a.name, a.levels[li]);
    $("st-frame-v").textContent = `${+$("st-frame").value + 1} / ${ST.info.n_frames}`;
    $("st-result").textContent = "Шукаю…"; $("st-result").className = "result";
    const q = new URLSearchParams({ axis: a.name, level_idx: li, method: $("st-method").value, frame: $("st-frame").value });
    const r = await fetch(`/api/stress/try?${q}`);
    ST.res = await r.json();
    $("st-img").src = ST.res.image;
    const res = ST.res;
    $("st-result").textContent = res.success ? `Знайдено, похибка ${res.err_m.toFixed(1)} м` : res.ok ? "Знайдено неправильно" : "Не знайдено";
    $("st-result").className = "result " + (res.success ? "ok" : "bad");
    $("st-detail").textContent = `inliers ${res.n_inliers}, ${res.ms.toFixed(0)} мс` + (res.ok ? "" : ` · ${res.reason || "замало відповідностей"}`);
    stressView.redraw();
  },
  redraw() {
    if (!ST.info) return;
    if (ST.img) {
      const c = $("st-map"), ctx = sizeCanvas(c), size = ST.info.map_size;
      const T = fitImage(c, ST.img, size);
      drawMapImage(ctx, ST.img, size, T, 0.25);
      const r = ST.res;
      if (r) {
        const g = T.P(...r.gt);
        ctx.strokeStyle = "#fff"; ctx.setLineDash([4 * dpr(), 3 * dpr()]); ctx.lineWidth = 2 * dpr();
        ctx.beginPath(); ctx.arc(g[0], g[1], 16 * dpr(), 0, 7); ctx.stroke(); ctx.setLineDash([]);
        if (r.corners && r.corners.every((p) => p && fin(p[0]))) {
          ctx.strokeStyle = r.success ? COL.fixed : COL.spoof; ctx.lineWidth = 2.5 * dpr();
          poly(ctx, r.corners, T); ctx.stroke();
        }
      }
    }
    const a = stressView.axis();
    if (!a) return;
    const cur = +$("st-level").value;
    const series = ST.info.curves.filter((cv) => cv.axis === a.name).map((cv) => ({
      x: cv.levels.map((_, i) => i), y: cv.success.map((v) => 100 * v), color: METHOD_COLORS[cv.method] || "#bbb", markers: true,
    }));
    const { X } = chart($("st-chart"), {
      xMax: a.levels.length - 1, yMax: 100, xLabel: "сила спотворення", yLabel: "знайдено кадрів, %",
      xTicks: a.levels.map((v, i) => [i, levelText(a.name, v)]),
      vlines: [{ x: cur, color: "#5a5a5a", width: 6 }], series,
    });
    const ctx = $("st-chart").getContext("2d"), r = dpr();
    ctx.font = `${12 * r}px Segoe UI, sans-serif`;
    ST.info.curves.filter((cv) => cv.axis === a.name).forEach((cv, i) => {
      ctx.fillStyle = METHOD_COLORS[cv.method] || "#bbb";
      ctx.fillRect($("st-chart").width - 110 * r, 14 * r + i * 16 * r, 14 * r, 4 * r);
      ctx.fillText(cv.method, $("st-chart").width - 90 * r, 19 * r + i * 16 * r);
    });
  },
};

// ---------- live camera ----------
const LV = { id: null, img: null, size: null, running: false, stream: null, trail: [], last: null, n: 0, reg: 0, fps: 0, view: null };
const liveView = {
  async init() {
    const info = await (await fetch("/api/live/info")).json();
    for (const m of info.maps) $("lv-map").add(new Option(m.label, m.id));
    for (const m of info.methods) $("lv-method").add(new Option(m.label, m.id));
    if (!info.maps.length) $("lv-error").textContent = "Немає жодної мапи локально (scripts/download_data.py).";
    $("lv-source").onchange = () => {
      const cam = $("lv-source").value === "camera";
      $("lv-camera-row").classList.toggle("hidden", !cam); $("lv-file-row").classList.toggle("hidden", cam);
    };
    $("lv-start").onclick = liveView.start;
    $("lv-stop").onclick = liveView.stop;
    $("lv-follow").onchange = liveView.redraw;
    $("lv-map").onchange = async () => { // show the map before the camera starts
      if (LV.running) return;
      LV.last = null; LV.trail = [];
      LV.size = null; LV.img = await loadImage(`/api/live/preview/${$("lv-map").value}`);
      LV.size = [LV.img.naturalWidth, LV.img.naturalHeight];
      liveView.redraw();
    };
    if (info.maps.length) $("lv-map").onchange();
    await liveView.listCameras();
    if (new URLSearchParams(location.hash.slice(1)).get("autostart")) liveView.start();
  },
  async listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) { $("lv-camera").add(new Option("камера недоступна в цьому браузері", "")); return; }
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    $("lv-camera").innerHTML = "";
    devs.forEach((d, i) => $("lv-camera").add(new Option(d.label || `Камера ${i + 1}`, d.deviceId)));
    if (!devs.length) $("lv-camera").add(new Option("за замовчуванням", ""));
  },
  async start() {
    $("lv-error").textContent = "";
    $("lv-start").disabled = true;
    try {
      const fd = new FormData();
      fd.append("map_id", $("lv-map").value); fd.append("method", $("lv-method").value); fd.append("frame_scale", $("lv-scale").value);
      const r = await fetch("/api/live/start", { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      Object.assign(LV, { id: body.id, size: body.map_size, trail: [], last: null, n: 0, reg: 0, fps: 0, view: null });
      LV.img = await loadImage(`/api/live/${LV.id}/map`);
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
      $("lv-start").disabled = false;
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
        const r = await fetch(`/api/live/${LV.id}/frame`, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
        res = await r.json();
        if (!r.ok) throw new Error(res.detail || r.statusText);
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
    if (!LV.img || !LV.size) return;
    const c = $("lv-map-canvas"), ctx = sizeCanvas(c), r = LV.last;
    let focus = null;
    if ($("lv-follow").checked && r && r.ok && r.corners) {
      const xs = r.corners.map((p) => p[0]), ys = r.corners.map((p) => p[1]);
      const ext = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.6;
      focus = [r.x - ext, r.y - ext, r.x + ext, r.y + ext];
    }
    const T = fitImage(c, LV.img, LV.size, focus);
    drawMapImage(ctx, LV.img, LV.size, T, 0.25);
    line(ctx, LV.trail, T, COL.vis, 2.5);
    if (r && r.ok && r.corners && r.corners.every((p) => p && fin(p[0]))) {
      ctx.fillStyle = "rgba(255,255,255,.08)"; poly(ctx, r.corners, T); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr(); poly(ctx, r.corners, T); ctx.stroke();
      const [a, b] = [T.P(...r.corners[0]), T.P(...r.corners[1])];
      ctx.strokeStyle = COL.vis; ctx.lineWidth = 4 * dpr(); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      dot(ctx, ...T.P(r.x, r.y), 6, COL.spoof, "#fff");
    }
  },
};

const VIEWS = { spoof: spoofView, stress: stressView, live: liveView };
{ // deep link to a tab: #tab=spoof | stress | live
  const tab = new URLSearchParams(location.hash.slice(1)).get("tab");
  if (VIEWS[tab]) showView(tab);
}
