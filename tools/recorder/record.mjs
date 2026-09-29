// Frame-exact video capture of the web viewer.
//
// Drives the viewer through its `window.__viewer` automation API in headless Chromium,
// screenshots every frame and encodes an MP4 (and optionally a GIF) with ffmpeg.
//
//   node record.mjs --url http://localhost:4173/ --seq 20241126_0024_crossing1_09 \
//        --shots "bev:0-199" --out ../../outputs/video/crossing1_09 --gif
//
// A shot is `<view>:<from>-<to>[@<layers>]`; several shots are separated by commas, e.g.
//   --shots "bev:0-99,orbit:100-199@det,map,trails"
import { chromium } from "playwright";
import { spawnSync } from "node:child_process";
import { mkdirSync, rmSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";

function args() {
  const a = process.argv.slice(2);
  const out = { url: "http://localhost:4173/", width: 1920, height: 1080, fps: 10, gif: false, ui: 1, scale: 1 };
  for (let i = 0; i < a.length; i++) {
    const k = a[i].replace(/^--/, "");
    if (k === "gif") out.gif = true;
    else out[k] = a[++i];
  }
  for (const k of ["width", "height", "fps", "ui", "scale"]) out[k] = Number(out[k]);
  if (!out.seq || !out.shots || !out.out) {
    console.error("usage: node record.mjs --seq <id> --shots <view:a-b,...> --out <dir> [--url] [--gif]");
    process.exit(1);
  }
  return out;
}

function parseShots(spec) {
  return spec.split(",").map((s) => {
    const [head, layers] = s.split("@");
    const [view, range] = head.split(":");
    const [from, to] = range.split("-").map(Number);
    return { view, from, to, layers: layers ? layers.split("+") : null };
  });
}

async function main() {
  const o = args();
  const outDir = resolve(o.out);
  const framesDir = join(outDir, "frames");
  if (existsSync(framesDir)) rmSync(framesDir, { recursive: true });
  mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({ args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: o.width, height: o.height }, deviceScaleFactor: o.scale });
  page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
  const url = `${o.url}?seq=${o.seq}&intro=0&ui=${o.ui}&frame=0`; // frame= disables autoplay
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => window.__viewer !== undefined, null, { timeout: 60000 });
  await page.evaluate(() => window.__viewer.ready);

  let n = 0;
  for (const shot of parseShots(o.shots)) {
    await page.evaluate((v) => window.__viewer.setView(v), shot.view);
    if (shot.layers) {
      await page.evaluate((ls) => {
        const all = ["det", "gt", "compare", "map", "trails", "labels"];
        const obj = Object.fromEntries(all.map((k) => [k, ls.includes(k)]));
        window.__viewer.setLayers(obj);
      }, shot.layers);
    }
    for (let i = shot.from; i <= shot.to; i++) {
      await page.evaluate((f) => window.__viewer.setFrame(f), i);
      await page.screenshot({ path: join(framesDir, `f_${String(n).padStart(5, "0")}.png`) });
      n++;
      if (n % 25 === 0) console.log(`  captured ${n} frames`);
    }
  }
  await browser.close();

  const mp4 = join(outDir, "video.mp4");
  const ff = (argv) => {
    const r = spawnSync("ffmpeg", ["-v", "error", "-y", ...argv], { stdio: "inherit" });
    if (r.status !== 0) throw new Error("ffmpeg failed");
  };
  ff(["-framerate", String(o.fps), "-i", join(framesDir, "f_%05d.png"), "-c:v", "libx264", "-preset", "slow",
    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]);
  console.log("wrote", mp4);
  if (o.gif) {
    const gif = join(outDir, "preview.gif");
    const filt = "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle";
    ff(["-i", mp4, "-vf", filt, gif]);
    console.log("wrote", gif);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
