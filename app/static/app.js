// Mission Replay frontend: streams a localisation job (SSE) and draws it on a canvas map.
"use strict";

const $ = (id) => document.getElementById(id);
const CORNERS = ["c0x", "c0y", "c1x", "c1y", "c2x", "c2y", "c3x", "c3y"];
const SOURCE_LABELS = {
  global: "пошук по всій мапі", init: "ініціалізація", roi: "трекінг у вікні", roi_retry: "трекінг, ширше вікно",
  reinit: "повторний пошук", tiles: "перебір тайлів", lost: "втрачено",
};

const S = {
  info: null, jobId: null, es: null, status: "", n: 0, names: [], mapSize: null, gt: null,
  frames: [], final: null, metrics: null, cur: -1, live: true, playTimer: null,
  mapImg: null, view: { s: 1, ox: 0, oy: 0 }, target: null, dirty: true, drag: null,
  camShown: -1, camPending: -1, camLoading: false, nextEvent: 0, initK: null,
};

// ---------- helpers ----------
const fin = (v) => v !== null && v !== undefined && Number.isFinite(v);
const timeColor = (k, a = 1) => {
  const t = S.n > 1 ? k / (S.n - 1) : 0;
  return `hsla(${45 - 40 * t}, 82%, ${70 - 12 * t}%, ${a})`; // light amber -> orange -> red
};
const fmt = (v, d = 1) => (fin(v) ? v.toFixed(d) : "—");
const median = (a) => {
  if (!a.length) return NaN;
  const b = [...a].sort((x, y) => x - y), m = b.length >> 1;
  return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
};
function point(k) { // best known position of frame k: final (post-processed) trajectory if ready, else raw fix
  const f = S.final && S.final[k];
  if (f && (f.status === "map" || f.status === "interp") && fin(f.x)) return { ...f, interp: f.status === "interp" };
  const r = S.frames[k];
  if (r && r.status === "map" && fin(r.x)) return { ...r, interp: false };
  return null;
}
const corners = (r) => [0, 1, 2, 3].map((i) => [r[CORNERS[2 * i]], r[CORNERS[2 * i + 1]]]);
const hasCorners = (r) => CORNERS.every((c) => fin(r[c]));

// ---------- setup ----------
async function init() {
  S.info = await (await fetch("/api/info")).json();
  const ms = $("method");
  for (const m of S.info.methods) {
    const o = new Option(m.label + (m.available ? "" : ` (потрібно: ${m.needs.join(", ")})`), m.id);
    o.disabled = !m.available;
    ms.add(o);
  }
  ms.value = "baseline";
  ms.onchange = () => { $("method-hint").textContent = S.info.methods.find((m) => m.id === ms.value)?.note || ""; };
  ms.onchange();

  const ps = $("preset");
  if (S.info.presets.length) {
    for (const p of S.info.presets) ps.add(new Option(p.label, p.id));
    ps.onchange = () => {
      const p = S.info.presets.find((x) => x.id === ps.value);
      $("preset-hint").textContent = [p?.gt ? "Є точний GT: побачите справжній маршрут і похибку." : "",
        p?.geo ? "Мапа з геоприв'язкою: маршрут вийде і в GPS (інтерактивна мапа, GeoJSON)." : ""].join(" ");
    };
    ps.onchange();
  } else {
    const o = new Option("Локальних наборів немає (scripts/download_data.py)", "");
    o.disabled = true; ps.add(o); ps.value = "";
    switchTab("upload");
  }
  $("map-hint").textContent = S.info.default_map ? "(необов'язково — за замовчуванням мапа завдання)" : "(обов'язково)";

  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => switchTab(b.dataset.tab)));
  $("job-form").onsubmit = submitJob;
  $("cancel").onclick = () => S.jobId && fetch(`/api/jobs/${S.jobId}/cancel`, { method: "POST" });
  $("fit-btn").onclick = () => { $("follow").checked = false; fitMap(); };
  $("follow").onchange = () => (S.dirty = true);
  $("show-fp").onchange = () => (S.dirty = true);
  $("slider").oninput = (e) => { stopPlay(); seek(+e.target.value, false); };
  $("play").onclick = togglePlay;
  setupMapInteraction();
  setupChart();
  window.addEventListener("resize", () => { resizeCanvases(); S.dirty = true; drawChart(); });
  resizeCanvases();
  requestAnimationFrame(loop);
  refreshJobs();
  setInterval(refreshJobs, 3000);
  // deep link: #job=<id>[&k=<frame>][&view=fit]
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("job")) {
    openJob(h.get("job"));
    if (h.get("k")) S.initK = +h.get("k");
    if (h.get("view") === "fit") $("follow").checked = false;
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((d) => d.classList.toggle("hidden", d.dataset.tab !== name));
}

async function submitJob(ev) {
  ev.preventDefault();
  $("form-error").textContent = "";
  const fd = new FormData($("job-form"));
  const tab = document.querySelector(".tab.active").dataset.tab;
  if (tab !== "coords") for (const k of ["lat", "lon", "size_km"]) fd.delete(k);
  if (tab === "coords") {
    for (const k of ["preset", "map_file", "frames", "every", "frame_scale"]) fd.delete(k);
    if (!fd.get("lat") || !fd.get("lon")) { $("form-error").textContent = "Вкажіть широту й довготу."; return; }
  } else if (tab === "upload") {
    fd.delete("preset");
    if (!$("frames-input").files.length) { $("form-error").textContent = "Оберіть фото кадрів або відео."; return; }
    if (!fd.get("map_file")?.size) fd.delete("map_file");
  } else {
    for (const k of ["map_file", "frames", "every", "frame_scale"]) fd.delete(k);
    if (!fd.get("preset")) { $("form-error").textContent = "Немає набору — завантажте свої дані."; return; }
  }
  if (!fd.get("max_frames")) fd.delete("max_frames");
  $("start-btn").disabled = true;
  try {
    const r = await fetch("/api/jobs", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    openJob(body.id);
    refreshJobs();
  } catch (e) {
    $("form-error").textContent = e.message;
  } finally {
    $("start-btn").disabled = false;
  }
}

async function refreshJobs() {
  let jobs;
  try { jobs = await (await fetch("/api/jobs")).json(); } catch { return; }
  const ul = $("jobs");
  ul.innerHTML = "";
  if (!jobs.length) { ul.innerHTML = '<li class="muted" style="cursor:default">Ще немає</li>'; return; }
  for (const j of jobs) {
    const li = document.createElement("li");
    li.className = j.id === S.jobId ? "active" : "";
    const prog = j.n_frames ? ` · ${j.n_done}/${j.n_frames}` : "";
    li.innerHTML = `<span class="st ${j.status}">${j.status}</span><b></b><br><span class="muted"></span>`;
    li.querySelector("b").textContent = j.title;
    li.querySelector(".muted").textContent = j.method + prog;
    li.onclick = () => openJob(j.id);
    ul.appendChild(li);
  }
}

// ---------- job stream ----------
function openJob(id) {
  if (S.es) S.es.close();
  stopPlay();
  Object.assign(S, {
    jobId: id, status: "", n: 0, names: [], mapSize: null, gt: null, frames: [], final: null, metrics: null,
    cur: -1, live: true, camShown: -1, camPending: -1, target: null, dirty: true, nextEvent: 0,
  });
  location.hash = `job=${id}`;
  $("cam").removeAttribute("src");
  $("cam-empty").classList.remove("hidden");
  $("cam-empty").textContent = "Очікую кадри…";
  $("downloads").classList.add("hidden");
  $("gps-links").classList.add("hidden");
  $("legend-gt").classList.add("hidden");
  $("stat-err").classList.add("hidden");
  $("cancel").classList.remove("hidden");
  $("stage").textContent = "У черзі…";
  $("follow").checked = true;
  S.mapImg = null;
  loadJobMap(id);
  const es = new EventSource(`/api/jobs/${id}/events`);
  S.es = es;
  es.onmessage = (m) => handle(JSON.parse(m.data));
  es.onerror = () => { if (S.status) es.close(); };
  updateAll();
}

function loadJobMap(id) { // the map of a coordinates job exists only after the tiles are downloaded
  const img = new Image();
  img.onload = () => { if (S.jobId !== id) return; S.mapImg = img; if (!S.mapSize) fitMap(); S.dirty = true; };
  img.src = `/api/jobs/${id}/map?t=${Date.now()}`;
}

function handle(e) {
  if (e.i < S.nextEvent) return; // replayed after an automatic EventSource reconnect
  S.nextEvent = e.i + 1;
  const d = e.data;
  switch (e.type) {
    case "queued":
      $("job-title").textContent = `${d.title} · ${d.method}`;
      break;
    case "stage":
      $("stage").textContent = d.text;
      break;
    case "start":
      S.n = d.n_frames; S.names = d.names; S.mapSize = d.map_size; S.gt = d.gt;
      $("legend-gt").classList.toggle("hidden", !S.gt);
      $("stat-err").classList.toggle("hidden", !S.gt);
      $("stage").textContent = `Локалізація: ${d.n_frames} кадрів (${d.device})`;
      if (!S.mapImg) loadJobMap(S.jobId);
      if (!$("follow").checked) fitMap();
      break;
    case "frame":
      S.frames.push(d);
      if (S.initK !== null && S.frames.length - 1 >= S.initK) { seek(S.initK, false); S.live = false; S.initK = null; }
      else if (S.live) seek(S.frames.length - 1, true);
      else updateTimeline();
      break;
    case "done":
      S.final = d.trajectory; S.metrics = d.metrics;
      showDownloads();
      S.dirty = true;
      break;
    case "gps":
      $("gps-links").classList.remove("hidden");
      showDownloads();
      break;
    case "finished":
      S.status = d.status;
      $("cancel").classList.add("hidden");
      $("stage").textContent = d.status === "done" ? "Готово" : d.status === "cancelled" ? "Зупинено" : "Помилка";
      if (d.status === "error") {
        $("cam-empty").classList.remove("hidden");
        $("cam-empty").textContent = d.error;
        $("form-error").textContent = d.error;
      }
      if (S.es) S.es.close();
      refreshJobs();
      break;
  }
}

function showDownloads() {
  const box = $("downloads");
  box.classList.remove("hidden");
  box.querySelectorAll("a").forEach((a) => (a.href = `/api/jobs/${S.jobId}/files/${a.dataset.file}`));
  const m = S.metrics || {};
  let txt = `прив'язано ${m.n_registered}/${m.n_frames}, інтерпольовано ${m.n_interp}, втрачено ${m.n_lost}, ` +
    `${fmt(m.fps, 1)} кадр/с`;
  if (fin(m.center_err_median)) txt += `, похибка медіана ${fmt(m.center_err_median, 2)} px, ≤5 px: ${fmt(100 * m["success@5px"], 0)}%`;
  $("summary").textContent = txt;
}

// ---------- timeline / playback ----------
function seek(k, live) {
  if (!S.frames.length) return;
  S.cur = Math.max(0, Math.min(k, S.frames.length - 1));
  S.live = live || S.cur === S.frames.length - 1 && S.status === "";
  S.dirty = true;
  requestCamera(S.cur);
  updateAll();
}
function updateTimeline() {
  const sl = $("slider");
  sl.max = Math.max(0, S.frames.length - 1);
  sl.value = Math.max(0, S.cur);
  $("progress").textContent = S.n ? `${S.frames.length} / ${S.n} оброблено` : "";
}
function togglePlay() { S.playTimer ? stopPlay() : startPlay(); }
function startPlay() {
  if (!S.frames.length) return;
  if (S.cur >= S.frames.length - 1) seek(0, false);
  $("play").textContent = "⏸";
  const tick = () => {
    if (S.cur >= S.frames.length - 1) { stopPlay(); S.live = true; return; }
    seek(S.cur + 1, false);
    S.playTimer = setTimeout(tick, 1000 / +$("speed").value);
  };
  S.playTimer = setTimeout(tick, 0);
}
function stopPlay() {
  clearTimeout(S.playTimer);
  S.playTimer = null;
  $("play").textContent = "▶";
}

// ---------- camera ----------
function requestCamera(k) {
  S.camPending = k;
  if (S.camLoading) return;
  loadCamera();
}
function loadCamera() {
  const k = S.camPending;
  if (k < 0 || k === S.camShown) return;
  S.camLoading = true;
  const img = $("cam");
  img.onload = img.onerror = () => {
    S.camLoading = false;
    S.camShown = k;
    $("cam-empty").classList.add("hidden");
    if (S.camPending !== k) loadCamera();
  };
  img.src = `/api/jobs/${S.jobId}/frame/${k}`;
  $("cam-label").textContent = `${k + 1} / ${S.n} · ${S.names[k] || ""}`;
}

// ---------- stats ----------
function updateAll() {
  updateTimeline();
  updateStats();
  drawChart();
}
function updateStats() {
  const k = S.cur;
  if (k < 0) { for (const id of ["s-frame", "s-reg", "s-lost", "s-inl", "s-fps", "s-src", "s-err"]) $(id).textContent = "—"; return; }
  const seen = S.frames.slice(0, k + 1);
  const reg = seen.filter((r) => r.status === "map");
  const r = S.frames[k];
  const tsum = seen.reduce((a, x) => a + (x.t_extract || 0) + (x.t_match || 0) + (x.t_geom || 0), 0);
  $("s-frame").textContent = `${k + 1} / ${S.n || "?"}`;
  $("s-reg").textContent = `${reg.length} (${Math.round((100 * reg.length) / seen.length)}%)`;
  $("s-lost").textContent = seen.length - reg.length;
  const meanInl = reg.reduce((a, x) => a + x.n_inliers, 0) / Math.max(1, reg.length);
  $("s-inl").textContent = `${r.status === "map" ? r.n_inliers : "—"} / ${Math.round(meanInl)}`;
  $("s-fps").textContent = tsum > 0 ? (seen.length / tsum).toFixed(1) : "—";
  $("s-src").textContent = SOURCE_LABELS[r.source] || r.source || "—";
  $("s-src").style.color = r.status === "map" ? "" : "var(--bad)";
  if (S.gt) {
    const errs = [];
    for (let i = 0; i <= k; i++) {
      const p = S.frames[i], g = S.gt[i];
      if (p.status === "map" && g && fin(g[0])) errs.push(Math.hypot(p.x - g[0], p.y - g[1]));
    }
    const g = S.gt[k];
    const cur = r.status === "map" && g && fin(g[0]) ? Math.hypot(r.x - g[0], r.y - g[1]) : NaN;
    $("s-err").textContent = `${fmt(cur, 2)} / ${fmt(median(errs), 2)}`;
  }
}

// ---------- inliers chart ----------
function setupChart() {
  $("chart").addEventListener("click", (e) => {
    if (!S.n) return;
    const rect = e.target.getBoundingClientRect();
    stopPlay();
    seek(Math.floor(((e.clientX - rect.left) / rect.width) * S.n), false);
  });
}
function drawChart() {
  const c = $("chart"), ctx = c.getContext("2d"), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  if (!S.n) return;
  const maxInl = Math.max(50, ...S.frames.map((r) => r.n_inliers || 0));
  const bw = W / S.n;
  S.frames.forEach((r, k) => {
    if (r.status === "map") {
      const h = Math.max(2, ((H - 14) * r.n_inliers) / maxInl);
      ctx.fillStyle = timeColor(k, k <= S.cur ? 0.95 : 0.35);
      ctx.fillRect(k * bw, H - h, Math.max(1, bw - (bw > 3 ? 1 : 0)), h);
    } else {
      ctx.fillStyle = "rgba(224,86,75,.9)";
      ctx.fillRect(k * bw, H - 6, Math.max(1, bw), 6);
    }
  });
  if (S.cur >= 0) {
    ctx.fillStyle = "#fff";
    ctx.fillRect((S.cur + 0.5) * bw - 1, 0, 2, H);
  }
  ctx.fillStyle = "#9a9a9a";
  ctx.font = `${12 * devicePixelRatio}px Segoe UI, sans-serif`;
  ctx.fillText(`inliers (макс. ${maxInl})`, 6, 14 * devicePixelRatio);
}

// ---------- map canvas ----------
function resizeCanvases() {
  for (const id of ["map", "chart"]) {
    const c = $(id);
    c.width = Math.max(10, c.clientWidth * devicePixelRatio);
    c.height = Math.max(10, c.clientHeight * devicePixelRatio);
  }
}
function mapDims() {
  if (S.mapSize) return S.mapSize;
  return S.mapImg ? [S.mapImg.naturalWidth, S.mapImg.naturalHeight] : null;
}
function fitView() {
  const d = mapDims(), c = $("map");
  if (!d) return null;
  const s = Math.min(c.width / d[0], c.height / d[1]) * 0.96;
  return { s, ox: (c.width - d[0] * s) / 2, oy: (c.height - d[1] * s) / 2 };
}
function fitMap() {
  const v = fitView();
  if (v) { S.view = v; S.target = null; S.dirty = true; }
}
function followTarget() {
  const p = point(S.cur) || lastPoint(S.cur);
  const fv = fitView();
  if (!p || !fv) return fv;
  const c = $("map");
  let s = fv.s * 3;
  if (hasCorners(p)) {
    const cs = corners(p), xs = cs.map((q) => q[0]), ys = cs.map((q) => q[1]);
    const ext = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    s = (Math.min(c.width, c.height) * 0.4) / Math.max(ext, 1);
  }
  s = Math.min(Math.max(s, fv.s), fv.s * 14);
  return { s, ox: c.width / 2 - p.x * s, oy: c.height / 2 - p.y * s };
}
function lastPoint(k) {
  for (let i = k; i >= 0; i--) { const p = point(i); if (p) return p; }
  return null;
}
function setupMapInteraction() {
  const c = $("map");
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    $("follow").checked = false;
    const r = c.getBoundingClientRect();
    const mx = (e.clientX - r.left) * devicePixelRatio, my = (e.clientY - r.top) * devicePixelRatio;
    const f = Math.exp(-e.deltaY * 0.0015), v = S.view;
    S.view = { s: v.s * f, ox: mx - (mx - v.ox) * f, oy: my - (my - v.oy) * f };
    S.dirty = true;
  }, { passive: false });
  c.addEventListener("mousedown", (e) => { S.drag = { x: e.clientX, y: e.clientY, v: { ...S.view } }; });
  window.addEventListener("mousemove", (e) => {
    if (!S.drag) return;
    const dx = (e.clientX - S.drag.x) * devicePixelRatio, dy = (e.clientY - S.drag.y) * devicePixelRatio;
    if (Math.abs(dx) + Math.abs(dy) > 3) $("follow").checked = false;
    S.view = { ...S.drag.v, ox: S.drag.v.ox + dx, oy: S.drag.v.oy + dy };
    S.dirty = true;
  });
  window.addEventListener("mouseup", () => (S.drag = null));
}

function loop() {
  if ($("follow").checked && S.cur >= 0 && !S.drag) {
    const t = followTarget();
    if (t) {
      const v = S.view, a = 0.18;
      const nv = { s: v.s + (t.s - v.s) * a, ox: v.ox + (t.ox - v.ox) * a, oy: v.oy + (t.oy - v.oy) * a };
      if (Math.abs(nv.ox - v.ox) + Math.abs(nv.oy - v.oy) + Math.abs(nv.s - v.s) * 100 > 0.3) S.dirty = true;
      S.view = nv;
    }
  }
  if (S.dirty) { S.dirty = false; drawMap(); }
  requestAnimationFrame(loop);
}

function drawMap() {
  const c = $("map"), ctx = c.getContext("2d"), dpr = devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  const d = mapDims();
  if (!S.mapImg || !d) return;
  const { s, ox, oy } = S.view;
  const P = (x, y) => [ox + x * s, oy + y * s];
  ctx.drawImage(S.mapImg, ox, oy, d[0] * s, d[1] * s);
  ctx.fillStyle = "rgba(0,0,0,.28)";
  ctx.fillRect(ox, oy, d[0] * s, d[1] * s);
  if (S.cur < 0) return;
  ctx.lineJoin = ctx.lineCap = "round";

  // ground truth
  if (S.gt) {
    ctx.setLineDash([6 * dpr, 5 * dpr]);
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    let started = false;
    for (let k = 0; k < S.gt.length; k++) {
      const g = S.gt[k];
      if (!g || !fin(g[0])) continue;
      const [x, y] = P(g[0], g[1]);
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // footprints of past frames
  if ($("show-fp").checked) {
    const every = Math.max(1, Math.ceil(S.n / 60));
    ctx.lineWidth = 1 * dpr;
    for (let k = 0; k < S.cur; k += every) {
      const p = point(k);
      if (!p || !hasCorners(p)) continue;
      ctx.strokeStyle = timeColor(k, 0.5);
      polygon(ctx, corners(p).map((q) => P(...q)));
      ctx.stroke();
    }
  }

  // route
  let prev = null, start = null;
  for (let k = 0; k <= S.cur; k++) {
    const p = point(k);
    if (!p) continue;
    const q = P(p.x, p.y);
    if (!start) start = q;
    if (prev) {
      ctx.strokeStyle = timeColor(k);
      ctx.lineWidth = 3 * dpr;
      ctx.setLineDash(p.interp || k - prev.k > 1 ? [4 * dpr, 4 * dpr] : []);
      ctx.beginPath(); ctx.moveTo(prev.q[0], prev.q[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
    }
    prev = { q, k };
  }
  ctx.setLineDash([]);
  for (let k = 0; k <= S.cur; k++) {
    const p = point(k);
    if (!p) continue;
    const [x, y] = P(p.x, p.y);
    ctx.fillStyle = p.interp ? "rgba(255,255,255,.6)" : timeColor(k);
    ctx.beginPath(); ctx.arc(x, y, 2.6 * dpr, 0, 7); ctx.fill();
  }
  if (start) marker(ctx, start, "#3f9a6c", "START");
  const done = S.status === "done" && S.cur === S.frames.length - 1;
  if (done && prev) marker(ctx, prev.q, "#c9473d", "FINISH");

  // current frame
  const r = S.frames[S.cur];
  if (r && r.status === "map" && hasCorners(r)) {
    const cs = corners(r).map((q) => P(...q));
    ctx.fillStyle = "rgba(255,255,255,.08)";
    polygon(ctx, cs); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr;
    polygon(ctx, cs); ctx.stroke();
    ctx.strokeStyle = "#facc15"; ctx.lineWidth = 4 * dpr;
    ctx.beginPath(); ctx.moveTo(...cs[0]); ctx.lineTo(...cs[1]); ctx.stroke();
    const [cx, cy] = P(r.x, r.y);
    const top = [(cs[0][0] + cs[1][0]) / 2, (cs[0][1] + cs[1][1]) / 2];
    const L = Math.hypot(top[0] - cx, top[1] - cy) || 1, len = Math.min(L, 46 * dpr);
    arrow(ctx, cx, cy, cx + ((top[0] - cx) / L) * len, cy + ((top[1] - cy) / L) * len);
    ctx.fillStyle = "#e0564b"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.arc(cx, cy, 6 * dpr, 0, 7); ctx.fill(); ctx.stroke();
  } else if (r) {
    const p = lastPoint(S.cur);
    const [x, y] = p ? P(p.x, p.y) : [c.width / 2, c.height / 2];
    ctx.font = `600 ${14 * dpr}px Segoe UI, sans-serif`;
    const txt = "кадр втрачено — повторний пошук по мапі";
    const w = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(92,36,32,.92)";
    ctx.fillRect(x + 12 * dpr, y - 26 * dpr, w + 16 * dpr, 24 * dpr);
    ctx.fillStyle = "#fff";
    ctx.fillText(txt, x + 20 * dpr, y - 9 * dpr);
  }
}
function polygon(ctx, pts) {
  ctx.beginPath();
  pts.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
  ctx.closePath();
}
function arrow(ctx, x0, y0, x1, y1) {
  const dpr = devicePixelRatio, a = Math.atan2(y1 - y0, x1 - x0), h = 10 * dpr;
  ctx.strokeStyle = ctx.fillStyle = "#facc15";
  ctx.lineWidth = 3 * dpr;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - h * Math.cos(a - 0.45), y1 - h * Math.sin(a - 0.45));
  ctx.lineTo(x1 - h * Math.cos(a + 0.45), y1 - h * Math.sin(a + 0.45));
  ctx.closePath(); ctx.fill();
}
function marker(ctx, [x, y], color, label) {
  const dpr = devicePixelRatio;
  ctx.fillStyle = color; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr;
  ctx.beginPath(); ctx.arc(x, y, 7 * dpr, 0, 7); ctx.fill(); ctx.stroke();
  ctx.font = `700 ${13 * dpr}px Segoe UI, sans-serif`;
  const w = ctx.measureText(label).width;
  ctx.fillStyle = color;
  ctx.fillRect(x + 10 * dpr, y - 22 * dpr, w + 10 * dpr, 18 * dpr);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x + 15 * dpr, y - 8 * dpr);
}

init();
