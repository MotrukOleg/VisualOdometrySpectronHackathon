// Export the reveal.js deck to slides.pdf with headless Edge/Chrome over the DevTools protocol.
// Unlike `msedge --print-to-pdf`, this waits in REAL time until reveal.js has built all print pages
// (.pdf-page), which needs the background video and every figure loaded (~10 s for this deck).
//
//   npx http-server . -p 8000 -c-1      (in another terminal)
//   node export_pdf.mjs [url] [out.pdf]
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] || "http://localhost:8000/index.html?print-pdf";
const out = process.argv[3] || "slides.pdf";
const port = 9333;
const candidates = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium",
];
const exe = process.env.BROWSER || candidates.find(existsSync);
if (!exe) throw new Error("no Edge/Chrome found; set BROWSER=<path>");
const profile = join(tmpdir(), "reveal-pdf-profile");
mkdirSync(profile, { recursive: true });
const browser = spawn(exe, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "--autoplay-policy=no-user-gesture-required", "about:blank"], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* browser still starting */ }
    await sleep(200);
  }
  throw new Error("browser did not start");
}

const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const i = ++id;
  pending.set(i, (m) => (m.error ? reject(new Error(`${method}: ${m.error.message}`)) : resolve(m.result)));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, returnByValue: true })).result.value;

try {
  await send("Page.enable");
  await send("Page.navigate", { url });
  let pages = 0, total = 0;
  for (let t = 0; t < 180; t++) {  // up to 90 s
    await sleep(500);
    [pages, total] = await evaluate(
      "[document.querySelectorAll('.pdf-page').length, window.Reveal && Reveal.getTotalSlides ? Reveal.getTotalSlides() : 0]") || [0, 0];
    if (pages > 0 && pages >= total) break;
  }
  if (!pages) throw new Error("reveal.js print layout did not appear (is the server running?)");
  await evaluate("document.fonts.ready.then(() => true)");
  await sleep(1500);  // let images in the last pages decode
  const { data } = await send("Page.printToPDF", { printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: false, marginTop: 0, marginBottom: 0, marginLeft: 0, marginRight: 0 });
  writeFileSync(out, Buffer.from(data, "base64"));
  console.log(`${out}: ${pages} pages`);
} finally {
  ws.close();
  browser.kill();
}
