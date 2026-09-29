// Cinematic 30 fps render of one sequence: the LiDAR data plays in real time (10 Hz, every frame
// held for 3 video frames) while the camera moves smoothly along keyframed poses.
//
//   node cinematic.mjs --seq 20241126_0024_crossing1_09 --out ../../outputs/video/cinematic_09
//        [--url http://127.0.0.1:5173/] [--ui 0] [--from 0 --to 199] [--gif]
import { chromium } from "playwright";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const a = Object.fromEntries(
  process.argv.slice(2).reduce((acc, v, i, arr) => (v.startsWith("--") ? [...acc, [v.slice(2), arr[i + 1]?.startsWith("--") || arr[i + 1] === undefined ? "1" : arr[i + 1]]] : acc), []),
);
const url = a.url ?? "http://127.0.0.1:5173/";
const W = +(a.width ?? 1920), H = +(a.height ?? 1080), FPS = 30, HOLD = 3;
const from = +(a.from ?? 0), to = +(a.to ?? 199);
const ui = a.ui ?? "0";
if (!a.seq || !a.out) throw new Error("--seq and --out are required");

// Camera keyframes over normalised time u ∈ [0, 1] (target in metres, angles in degrees).
const KEYS = [
  { u: 0.0, target: [0, 0, 0], azimuth: -120, elevation: 62, distance: 150 },
  { u: 0.18, target: [0, 0, 0], azimuth: -100, elevation: 38, distance: 95 },
  { u: 0.45, target: [-4, -2, 0], azimuth: -55, elevation: 28, distance: 70 },
  { u: 0.7, target: [2, 0, 0], azimuth: 10, elevation: 34, distance: 80 },
  { u: 1.0, target: [0, 0, 0], azimuth: 60, elevation: 70, distance: 125 },
];
const smooth = (t) => t * t * (3 - 2 * t);
function poseAt(u) {
  let k = 0;
  while (k < KEYS.length - 2 && u > KEYS[k + 1].u) k++;
  const A = KEYS[k], B = KEYS[k + 1];
  const t = smooth(Math.min(1, Math.max(0, (u - A.u) / (B.u - A.u))));
  const lerp = (x, y) => x + (y - x) * t;
  return {
    target: A.target.map((v, i) => lerp(v, B.target[i])),
    azimuth: lerp(A.azimuth, B.azimuth),
    elevation: lerp(A.elevation, B.elevation),
    distance: lerp(A.distance, B.distance),
  };
}

const outDir = resolve(a.out);
const framesDir = join(outDir, "frames");
if (existsSync(framesDir)) rmSync(framesDir, { recursive: true });
mkdirSync(framesDir, { recursive: true });

const browser = await chromium.launch({ args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: W, height: H } });
page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
await page.goto(`${url}?seq=${a.seq}&intro=0&ui=${ui}&frame=${from}`, { waitUntil: "networkidle" });
await page.waitForFunction(() => window.__viewer !== undefined, null, { timeout: 60000 });
await page.evaluate(() => window.__viewer.ready);
if (a.title) await page.evaluate((t) => window.__viewer.setCaption(t), a.title);

const nData = to - from + 1;
const nVideo = nData * HOLD;
let last = -1;
for (let v = 0; v < nVideo; v++) {
  const f = from + Math.floor(v / HOLD);
  if (f !== last) {
    await page.evaluate((i) => window.__viewer.setFrame(i), f);
    last = f;
  }
  if (a.title && v === Math.round(nVideo * 0.22)) await page.evaluate(() => window.__viewer.setCaption(null));
  await page.evaluate((p) => window.__viewer.setCameraPose(p), poseAt(v / (nVideo - 1)));
  await page.screenshot({ path: join(framesDir, `f_${String(v).padStart(5, "0")}.png`) });
  if (v % 60 === 0) console.log(`  ${v}/${nVideo}`);
}
await browser.close();

const ff = (argv) => {
  const r = spawnSync("ffmpeg", ["-v", "error", "-y", ...argv], { stdio: "inherit" });
  if (r.status !== 0) throw new Error("ffmpeg failed");
};
const mp4 = join(outDir, "video.mp4");
ff(["-framerate", String(FPS), "-i", join(framesDir, "f_%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
  "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]);
console.log("wrote", mp4);
if (a.gif) {
  const filt = "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle";
  ff(["-i", mp4, "-vf", filt, join(outDir, "preview.gif")]);
  console.log("wrote", join(outDir, "preview.gif"));
}
