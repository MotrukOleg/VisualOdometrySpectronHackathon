// Record demo GIFs of the єДрон app (app/) for the README: headless Edge/Chrome + DevTools screencast + ffmpeg.
//
//   python -m app                                   (in another terminal, http://127.0.0.1:8080)
//   node scripts/record_app_gifs.mjs [scenario ...]  (default: all) -> docs/app/<scenario>.gif
//   node scripts/record_app_gifs.mjs --encode [scenario ...]   re-encode the last recording (no browser)
//
// Scenarios: missions, gps, spoof, stress, live. Needs ffmpeg in PATH and the local data the scenarios use
// (task frames, the Lviv area, UAV-VisLoc #24, data/video/IMG_0061.MOV).
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const APP = process.env.APP_URL || "http://127.0.0.1:8080";
const OUT = resolve("docs/app");
const W = 1440, H = 860;          // browser viewport
const GIF_W = 960, GIF_FPS = 8;   // output
const SPEEDUP = { missions: 3, gps: 2, spoof: 1.5, stress: 1.5, live: 1.5 };  // GIF plays faster than real time
const port = 9334;
const VIDEO = resolve(process.env.LIVE_VIDEO || "data/video/IMG_0061.MOV");  // the loop video over the task map

const candidates = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium",
];
const args = process.argv.slice(2);
const encodeOnly = args.includes("--encode");
const names = args.filter((a) => !a.startsWith("--"));
const exe = process.env.BROWSER || candidates.find(existsSync);
if (!exe) throw new Error("no Edge/Chrome found; set BROWSER=<path>");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- DevTools protocol ----------
if (encodeOnly) {
  for (const n of names.length ? names : Object.keys(SPEEDUP)) encode(n);
  process.exit(0);
}
const profile = join(tmpdir(), "edron-gif-profile");
mkdirSync(profile, { recursive: true });
const browser = spawn(exe, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  `--window-size=${W},${H}`, "--autoplay-policy=no-user-gesture-required", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* browser still starting */ }
    await sleep(200);
  }
  throw new Error("browser did not start");
}
const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map(), listeners = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method && listeners.has(m.method)) listeners.get(m.method)(m.params);
});
const send = (method, params = {}) => new Promise((ok, fail) => {
  const i = ++id;
  pending.set(i, (m) => (m.error ? fail(new Error(`${method}: ${m.error.message}`)) : ok(m.result)));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const js = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(`JS: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
};
async function until(expr, timeoutMs, what) {
  for (const t0 = Date.now(); Date.now() - t0 < timeoutMs; await sleep(250)) if (await js(expr)) return;
  throw new Error(`timeout: ${what || expr}`);
}
async function open(url) {
  // mark the old document so we wait for the new one; a hash-only change (/#tab=a -> /#tab=b) keeps the
  // document, so reload in that case
  const sameDoc = await js(`location.href.split('#')[0] === ${JSON.stringify(url.split("#")[0])}`).catch(() => false);
  await js("window.__stale = true").catch(() => {});
  await send("Page.navigate", { url });
  if (sameDoc) await send("Page.reload", { ignoreCache: false });
  await until("!window.__stale && document.readyState === 'complete'", 30000, `page load ${url}`);
  if (dir) await screencast().catch(() => {});  // keep the screencast alive across documents
  await sleep(800);
}
// page-side helpers: set a <select>/<input> and fire its event, click, animate a slider
const PAGE_HELPERS = `
  window.__set = (id, v, ev) => { const el = document.getElementById(id); el.value = v;
    el.dispatchEvent(new Event(ev || (el.tagName === 'SELECT' ? 'change' : 'input'), { bubbles: true })); };
  window.__click = (sel) => document.querySelector(sel).click();
  window.__slide = async (id, from, to, steps, ms) => { for (let i = 0; i <= steps; i++) {
    __set(id, from + (to - from) * i / steps); await new Promise((r) => setTimeout(r, ms)); } };
  true`;
const set = (elId, v) => js(`__set(${JSON.stringify(elId)}, ${JSON.stringify(String(v))})`);
const click = (sel) => js(`__click(${JSON.stringify(sel)})`);
const slide = (elId, from, to, steps, ms) => js(`__slide(${JSON.stringify(elId)}, ${from}, ${to}, ${steps}, ${ms})`);
const optionIds = (elId) => js(`[...document.getElementById(${JSON.stringify(elId)}).options].map((o) => o.value)`);

// ---------- screencast -> GIF ----------
let frames = [], dir = null;
listeners.set("Page.screencastFrame", ({ data, metadata, sessionId }) => {
  send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  if (!dir) return;
  const f = join(dir, `f${String(frames.length).padStart(5, "0")}.jpg`);
  writeFileSync(f, Buffer.from(data, "base64"));
  frames.push({ f, t: metadata.timestamp });
});
const screencast = () => send("Page.startScreencast", { format: "jpeg", quality: 90, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
async function record(name, body) {
  dir = join(tmpdir(), `edron-gif-${name}`);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  frames = [];
  await screencast();
  try { await body(); } finally {
    await sleep(300);
    await send("Page.stopScreencast");
  }
  const d = dir; dir = null;
  const end = Date.now() / 1000;
  if (frames.length < 2) throw new Error(`${name}: no frames captured`);
  console.log(`${name}: ${frames.length} screencast frames, ${(frames.at(-1).t - frames[0].t).toFixed(1)} s`);
  // ffconcat with real durations: the screencast sends a frame only when the page repaints
  let list = "ffconcat version 1.0\n";
  frames.forEach((fr, i) => {
    const next = i + 1 < frames.length ? frames[i + 1].t : Math.min(end, fr.t + 2);
    list += `file '${fr.f.replace(/\\/g, "/")}'\nduration ${Math.max(0.02, next - fr.t).toFixed(3)}\n`;
  });
  list += `file '${frames.at(-1).f.replace(/\\/g, "/")}'\n`;
  writeFileSync(join(d, "list.txt"), list);
  encode(name);
}
function encode(name) {
  const d = join(tmpdir(), `edron-gif-${name}`);
  if (!existsSync(join(d, "list.txt"))) throw new Error(`${name}: nothing recorded yet`);
  mkdirSync(OUT, { recursive: true });
  const out = join(OUT, `${name}.gif`);
  const vf = `setpts=PTS/${SPEEDUP[name] || 1},fps=${GIF_FPS},scale=${GIF_W}:-1:flags=lanczos,split[a][b];` +
    "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle";
  const r = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", join(d, "list.txt"),
    "-vf", vf, "-loop", "0", out], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ffmpeg failed for ${name}`);
  console.log(`${out}: ${(statSync(out).size / 2 ** 20).toFixed(1)} MB`);
}

// ---------- scenarios ----------
async function startMission({ preset, method, speed, maxFrames }) {
  await set("preset", preset);
  await set("method", method);
  await set("speed", speed);
  await js(`document.querySelector('[name=max_frames]').value = ${JSON.stringify(maxFrames ? String(maxFrames) : "")}; true`);
  await sleep(900);
  await click("#start-btn");
  // done when the downloads row appears (the replay may still be pacing behind the computation)
  await until("!document.getElementById('downloads').classList.contains('hidden')", 240000, "mission finished");
}
const SCENARIOS = {
  // 50 task frames, final method: camera on the left, the route growing on the map, stats and inliers chart
  async missions() {
    await open(`${APP}/`);
    await js(PAGE_HELPERS);
    await until("document.getElementById('preset').options.length > 0", 10000);
    await record("missions", async () => {
      await sleep(700);
      await startMission({ preset: "crops", method: "final", speed: 5 });
      await until("+document.getElementById('slider').value >= +document.getElementById('slider').max - 1", 60000, "replay end");
      await sleep(1200);
      await click("#fit-btn");
      await sleep(2500);
    });
  },
  // a geo-referenced area: error vs GT, satellite basemap, then the interactive Leaflet map in lat/lon
  async gps() {
    await open(`${APP}/`);
    await js(PAGE_HELPERS);
    await until("document.getElementById('preset').options.length > 0", 10000);
    await record("gps", async () => {
      await sleep(500);
      await startMission({ preset: "area_lviv", method: "final", speed: 15, maxFrames: 40 });
      await until("+document.getElementById('slider').value >= +document.getElementById('slider').max - 1", 60000, "replay end");
      await click("#fit-btn");
      await sleep(1500);
      await set("basemap", "streets");
      await sleep(3000);
      // the app serves route_map.html as a download, so open the job's copy on disk
      const job = await js("S.jobId");
      await open(pathToFileURL(resolve(`experiments/results/app/${job}/route_map.html`)).href);
      await sleep(4000);
    });
  },
  // GPS spoofing: change the attack and the detector, the alarm and the corrected track update instantly
  async spoof() {
    await open(`${APP}/#tab=spoof`);
    await js(PAGE_HELPERS);
    await until("document.getElementById('sp-source').options.length > 0 && document.getElementById('sp-s-alarm').textContent !== '—'",
      30000, "spoof view");
    await record("spoof", async () => {
      await sleep(1500);
      await slide("sp-rate", 3, 10, 14, 120);
      await sleep(800);
      await slide("sp-bearing", 135, 300, 20, 90);
      await sleep(800);
      await slide("sp-threshold", 15, 45, 15, 120);
      await sleep(1200);
      await slide("sp-threshold", 45, 15, 6, 80);
      // the main real-flight result: islands + fine-tuned DINOv2, checked by XFeat
      const ids = await optionIds("sp-source");
      const islands = ids.find((v) => v.includes("dinoft_track_xfeat"));
      if (islands) {
        await set("sp-source", islands);
        await sleep(3500);
      }
    });
  },
  // degradations: one frame under growing blur / clouds / darkness, where the method finds it on the map
  async stress() {
    await open(`${APP}/#tab=stress`);
    await js(PAGE_HELPERS);
    await until("document.getElementById('st-axis').options.length > 0 && document.getElementById('st-result').textContent !== '—'",
      60000, "stress view");
    const axes = await optionIds("st-axis");
    const pick = ["clouds", "dark", "blur"].map((a) => axes.find((v) => v.includes(a))).filter(Boolean);
    await record("stress", async () => {
      await sleep(1200);
      for (const axis of (pick.length ? pick : axes.slice(0, 3)).slice(0, 3)) {
        await set("st-axis", axis);
        await sleep(1200);
        for (let lv = 0; lv <= 5; lv++) {
          await set("st-level", lv);
          await sleep(1100);
        }
      }
    });
  },
  // live localisation of a video file (same as a webcam, minus the camera permission)
  async live() {
    if (!existsSync(VIDEO)) throw new Error(`no video ${VIDEO}`);
    // headless Edge does not decode the HEVC .MOV from the phone: 20 s of it as H.264 720p
    const mp4 = join(tmpdir(), "edron-live.mp4");
    const r = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-i", VIDEO, "-t", "20", "-vf", "scale=-2:720", "-an",
      "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", mp4], { stdio: "inherit" });
    if (r.status !== 0) throw new Error("ffmpeg could not transcode the live video");
    await open(`${APP}/#tab=live`);
    await js(PAGE_HELPERS);
    await until("document.getElementById('lv-map').options.length > 0", 20000, "live view");
    await set("lv-map", "task");
    await set("lv-method", "track_orb");  // the fastest live method (~4-5 frames/s here)
    await set("lv-scale", "0.5");
    await set("lv-source", "file");
    const { root } = await send("DOM.getDocument", { depth: 1 });
    const { nodeId } = await send("DOM.querySelector", { nodeId: root.nodeId, selector: "#lv-file" });
    await send("DOM.setFileInputFiles", { nodeId, files: [mp4] });
    await js("document.getElementById('lv-file').dispatchEvent(new Event('change', { bubbles: true })); true");
    await sleep(1000);
    await record("live", async () => {
      await sleep(600);
      await click("#lv-start");
      await sleep(18000);
      await click("#lv-stop");
      await sleep(800);
    });
  },
};

try {
  await send("Page.enable");
  await send("DOM.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  for (const n of names.length ? names : Object.keys(SCENARIOS)) {
    if (!SCENARIOS[n]) throw new Error(`unknown scenario ${n}; known: ${Object.keys(SCENARIOS).join(", ")}`);
    console.log(`recording ${n} ...`);
    await SCENARIOS[n]();
  }
} finally {
  ws.close();
  browser.kill();
}
