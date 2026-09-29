import {
  Color, HalfFloatType, OrthographicCamera, PerspectiveCamera, Scene, Vector2, WebGLRenderTarget,
  WebGLRenderer, type Camera,
} from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';

/** Renderer, scene, post-processing and the two cameras. */
export class Stage {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly composer: EffectComposer;
  readonly bloom: UnrealBloomPass;
  readonly labels: CSS2DRenderer;
  readonly ortho: OrthographicCamera;
  readonly persp: PerspectiveCamera;
  /** visible height (m) of the orthographic camera at zoom 1 */
  readonly orthoHeight = 100;
  camera: Camera;
  width = 1;
  height = 1;
  pixelRatio = 1;
  private readonly renderPass: RenderPass;
  private readonly lineMaterials = new Set<LineMaterial>();
  private readonly resizeListeners: (() => void)[] = [];

  constructor(readonly container: HTMLElement) {
    this.renderer = new WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    this.renderer.setClearColor(new Color('#04060a'), 1);
    this.renderer.domElement.className = 'gl';
    container.appendChild(this.renderer.domElement);

    this.scene.background = new Color('#04060a');

    this.ortho = new OrthographicCamera(-1, 1, 1, -1, 0.1, 2000);
    this.ortho.up.set(0, 0, 1);
    this.persp = new PerspectiveCamera(40, 1, 0.2, 3000);
    this.persp.up.set(0, 0, 1);
    this.camera = this.ortho;

    const rt = new WebGLRenderTarget(1, 1, { type: HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(this.renderer, rt);
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);
    this.bloom = new UnrealBloomPass(new Vector2(256, 256), 0.62, 0.42, 0.86);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels-layer';
    container.appendChild(this.labels.domElement);

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  setCamera(cam: Camera): void {
    this.camera = cam;
    this.renderPass.camera = cam;
  }

  registerLineMaterial(m: LineMaterial): LineMaterial {
    this.lineMaterials.add(m);
    m.resolution.set(this.width * this.pixelRatio, this.height * this.pixelRatio);
    return m;
  }

  onResize(fn: () => void): void {
    this.resizeListeners.push(fn);
  }

  resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.width = w;
    this.height = h;
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h);
    this.composer.setPixelRatio(this.pixelRatio);
    this.composer.setSize(w, h);
    this.labels.setSize(w, h);
    const aspect = w / h;
    this.persp.aspect = aspect;
    this.persp.updateProjectionMatrix();
    const hh = this.orthoHeight / 2;
    this.ortho.left = -hh * aspect;
    this.ortho.right = hh * aspect;
    this.ortho.top = hh;
    this.ortho.bottom = -hh;
    this.ortho.updateProjectionMatrix();
    for (const m of this.lineMaterials) m.resolution.set(w * this.pixelRatio, h * this.pixelRatio);
    for (const fn of this.resizeListeners) fn();
  }

  /** pixels (device) per metre on the ground for the orthographic camera */
  orthoPxPerMetre(): number {
    return (this.height * this.pixelRatio * this.ortho.zoom) / this.orthoHeight;
  }

  /** device pixels per unit of (1 / view depth) for the perspective camera */
  perspPxFactor(): number {
    return (this.height * this.pixelRatio) / (2 * Math.tan((this.persp.fov * Math.PI) / 360));
  }

  render(): void {
    this.composer.render();
    this.labels.render(this.scene, this.camera);
  }
}
