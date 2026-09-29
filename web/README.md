# Web viewer

Interactive 3D viewer for the fused infrastructure-LiDAR point clouds, detections, tracks and
counts exported by the Python pipeline. Vite + TypeScript + three.js, no UI framework.

The viewer reads exactly the format described in [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md)
from `web/public/data/` (served at `data/`). That directory is git-ignored: the dataset licence
(CC BY-NC-ND 4.0) does not allow publishing derived point clouds.

## Run

```bash
cd web
npm install
npm run mock      # optional: synthetic sequence for development (~235 MB in public/data/)
npm run dev       # http://127.0.0.1:5173
npm run build     # type-check (tsc --noEmit) + production build into dist/
npm run preview   # serve dist/
```

Real data: run the Python export (`scripts/export_web.py`), which writes `index.json`,
`map_crossing1.json` and one folder per sequence into `web/public/data/`, replacing the mock.
Note that `vite build` copies `public/data/` into `dist/`.

`npm run mock` accepts `-- --frames 200 --az 900 --channels 48` to change the size of the synthetic
sequence. The mock ray-casts four spinning LiDARs against a 4-way intersection (ground, curbs,
buildings, trees, parked cars, moving cars, a bus, a truck, cyclists and pedestrians with a
traffic-light cycle), and derives detections (with misses, false positives, a class confusion and
an ID switch), ground truth, tracks, counts and metrics.

## Views and layers

| View      | Camera                                        | Mouse                                       |
|-----------|-----------------------------------------------|---------------------------------------------|
| BEV       | orthographic top-down (default)               | drag: pan · wheel: zoom · right-drag: rotate |
| 3D Orbit  | perspective orbit                              | drag: orbit · right-drag: pan · wheel: zoom  |
| Follow    | chase camera on the selected (or a moving) track | as orbit; swings back behind the vehicle     |

Click a box to select it (info card), double-click to follow it. Layers: detections, ground truth
(dashed), compare (TP green / FP red / missed GT orange, greedy BEV centre-distance matching per
frame), HD map, trails (last 3 s), labels, sensors and grid. Point colour modes: height,
intensity, sensor (per LiDAR) and focus (static background dimmed).

## Keyboard

| Key              | Action                                  |
|------------------|-----------------------------------------|
| `Space`          | play / pause                            |
| `←` / `→`        | previous / next frame (`Shift`: 10)     |
| `Home`           | first frame                             |
| `1` `2` `3`      | BEV · 3D orbit · follow                 |
| `F`              | follow the selected track (toggle)      |
| `D` `G` `C`      | detections · ground truth · compare     |
| `M` `T` `L`      | map · trails · all labels               |
| `V`              | cycle point colour mode                 |
| `H`              | hide / show the interface               |
| `?`              | keyboard help                           |
| `Esc`            | clear selection                         |

## URL parameters

| Param      | Values                                   | Default |
|------------|------------------------------------------|---------|
| `seq`      | sequence id from `index.json`            | first sequence |
| `frame`    | frame index (0-based)                    | `0` |
| `view`     | `bev` · `orbit` · `follow`               | `bev` |
| `ui`       | `0` hides all panels (caption only) · `1` | `1` |
| `intro`    | `0` skips the fly-in and scan reveal     | `1` |
| `layers`   | e.g. `det,gt,map,trails` (see below)     | `det,map,trails,sensors,grid` |
| `color`    | `height` · `intensity` · `lidar` · `focus` | `height` |
| `size`     | point size multiplier, 0.25–4            | `1` |
| `speed`    | `0.5` · `1` · `2` · `4`                  | `1` |
| `play`     | `1` autoplay · `0` paused                | `1` unless `frame` is given or `ui=0` |
| `select`   | track id to preselect (useful with `view=follow`) | – |
| `caption`  | `0` hides the caption when `ui=0`        | `1` when `ui=0` |

`layers` lists the overlay layers to enable (`det`, `gt`, `compare`, `map`, `trails`, `labels`);
the ambient layers `sensors` and `grid` stay on unless listed with a minus (`-grid`). A list made
only of `+x`/`-x` tokens modifies the defaults instead, e.g. `layers=+gt,-trails`.

Example recording URL: `?seq=<id>&frame=0&view=orbit&ui=0&intro=0&layers=det,map,trails`.

## Recording API

`window.__viewer` is available as soon as the page script runs:

```ts
window.__viewer = {
  ready: Promise<void>,              // data loaded and the first frame (points included) rendered
  introDone: Promise<void>,          // intro camera flight finished (immediately if intro=0)
  setSequence(id: string): Promise<void>,   // loads the sequence, frame 0, rendered
  setFrame(i: number): Promise<void>,       // pauses playback; resolves after frame i's points
                                            // are uploaded and a frame has been rendered
  setView(name: 'bev' | 'orbit' | 'follow', opts?: { instant?: boolean }): Promise<void>,
                                            // resolves when the camera transition has finished
  setLayers(obj: Partial<{ det, gt, compare, map, trails, labels, sensors, grid: boolean }>): void,
  getState(): {
    sequence, frame, nFrames, pointsFrame, nPoints, view, layers, colorMode, pointSize,
    playing, speed, selected, ui, cachedFrames, introRunning,
  },
  play(): void, pause(): void, setSpeed(s: number): void,
  setColorMode(m: 'height' | 'intensity' | 'lidar' | 'focus'): void,
  setPointSize(v: number): void,
  select(trackId: number | null): void,
  setUi(visible: boolean): void,
  setCaption(html: string | null): void,    // custom caption text; null restores the default
}
```

A typical capture loop: open with `ui=0&intro=0`, `await __viewer.ready`, then for each frame
`await __viewer.setFrame(i)` and take a screenshot.

## Structure

```
src/main.ts            app state, render loop, input, URL params, window.__viewer
src/data/types.ts      data-contract types
src/data/loader.ts     JSON loading, derived track paths / counts, point LRU cache + prefetch
src/data/match.ts      per-frame detection ↔ GT matching for compare mode
src/scene/stage.ts     renderer, cameras, bloom composer, CSS2D label layer
src/scene/camera.ts    BEV / orbit / follow controls, camera flights and intro
src/scene/points.ts    point cloud: raw int16/uint8 records decoded in the vertex shader
src/scene/boxes.ts     pooled 3D boxes (glowing edges, footprint, heading chevron, label chip)
src/scene/trails.ts    pooled fading track trails
src/scene/map.ts       HD-map lines and areas
src/scene/environment.ts  ground grid and LiDAR sensor markers
src/scene/lines.ts     in-place rewritable fat-line segment buffers
src/ui/hud.ts          panels (live count, evaluation, layers, info card, status line)
src/ui/timeline.ts     count chart + scrubber
scripts/make-mock-data.mjs  synthetic development data
```

Performance notes: one point buffer is preallocated for the largest frame of the sequence and
each frame's raw bytes are copied into it (no CPU decoding); point files are prefetched 15 frames
ahead into an 80-frame LRU cache; playback waits (instead of desynchronising) when the next frame
has not arrived yet. Boxes, labels and trails are pooled.
