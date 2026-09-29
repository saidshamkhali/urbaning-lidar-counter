// Quick screenshots of the viewer: node shot.mjs <out.png> "<query>" [view] [width] [height]
import { chromium } from "playwright";
const [out, query = "", view = "bev", w = "1920", h = "1080"] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
await page.goto(`http://127.0.0.1:5173/?intro=0&${query}`, { waitUntil: "networkidle" });
await page.waitForFunction(() => window.__viewer !== undefined);
await page.evaluate(() => window.__viewer.ready);
await page.evaluate((v) => window.__viewer.setView(v, { instant: true }), view);
const f = new URLSearchParams(query).get("frame");
if (f) await page.evaluate((i) => window.__viewer.setFrame(i), +f);
await page.waitForTimeout(800);
await page.screenshot({ path: out });
await browser.close();
