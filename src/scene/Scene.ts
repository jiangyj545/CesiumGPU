import { Context } from '../renderer/Context';
import { Camera } from './Camera';
import { Globe } from './Globe';
import { DirectionalLight } from './DirectionalLight';
import { GBufferPass } from './passes/GBufferPass';
import { LightingPass } from './passes/LightingPass';
import { TransparentPass } from './passes/TransparentPass';
import {
  Primitive,
  PrimitiveCollection,
  buildOpaquePipeline,
  buildUnlitPipeline,
} from './Primitive';
import { PickPass } from '../picking/PickPass';
import type { PickPassDeps, PickResult } from '../picking/PickPass';
import { OverlayPass } from './passes/OverlayPass';
import type { IPass } from './passes/IPass';
import { EntityVisualizer } from './EntityVisualizer';
import { EntityCollection } from '../dataSource/EntityCollection';
import { EffectManager } from '../effect/EffectManager';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

/**
 * Scene - Orchestrates all rendering passes.
 *
 * Rendering pipeline (render() walks the {@link IPass} array in order):
 *  1. GBufferPass     → writes albedo / normal / worldPos for all opaque prims
 *  2. LightingPass    → reads G-Buffer, outputs lit opaque scene
 *  3. OverlayPass     → entity billboards (before transparent so WBOIT blends
 *     over the billboards)
 *  4. TransparentPass → WBOIT accumulation + composite over the opaque result
 *  5. Effect passes   → EffectManager executes IPass effects (additive/emissive
 *     overlays, drawn last so they are not overwritten)
 */
export class Scene {
  readonly camera: Camera;
  readonly globe: Globe;
  readonly primitives: PrimitiveCollection;

  /** 特效统一挂载点（T3.1）：Transparent composite 之后逐个 execute。 */
  readonly effects: EffectManager = new EffectManager();

  /**
   * Entities in the scene.  Polygon/polyline visuals are rendered as
   * primitives (via {@link EntityVisualizer}); point/label billboards are
   * drawn in screen space by the overlay pass.
   */
  readonly entities: EntityCollection = new EntityCollection();

  /** Directional lights in the scene (max 8). */
  readonly lights: DirectionalLight[] = [];

  /**
   * Toggles per-primitive frustum culling (G-Buffer opaque loop, transparent
   * loop and the pick view). On by default; disable for debugging.
   */
  debugFrustumCulling: boolean = true;

  /**
   * Per-frame hooks invoked with the shared command encoder after all scene
   * render passes and before submit.  Effects (e.g. particle systems) use
   * these to append their own compute and render passes to the same command
   * buffer as the rest of the frame.
   */
  readonly frameHooks: Array<
    (encoder: GPUCommandEncoder, swapChainView: GPUTextureView) => void
  > = [];

  private _context: Context;
  // GPU resources are created lazily in initialize(), after context.initialize().
  private _gbufferPass: GBufferPass | null = null;
  private _lightingPass: LightingPass | null = null;
  private _transparentPass: TransparentPass | null = null;

  /**
   * Frame pipeline: render()/resize()/destroy() walk this array in order
   * (gbuffer → lighting → overlay → transparent).  Populated in initialize().
   */
  private _renderPasses: IPass[] = [];

  private _opaquePipeline: GPURenderPipeline | null = null;
  private _unlitPipeline: GPURenderPipeline | null = null;
  private _cameraUniformBuffer: GPUBuffer | null = null;
  private _cameraBindGroup: GPUBindGroup | null = null;

  /** GPU picking (lazily created on the first {@link pick} call). */
  private _pickPass: PickPass | null = null;

  /** Screen-space entity billboards (points, labels). */
  private _overlayPass: OverlayPass | null = null;
  private _entityVisualizer: EntityVisualizer | null = null;

  private _prepared = false;
  private _initPromise: Promise<void> | null = null;
  private _width  = 1;
  private _height = 1;

  constructor(context: Context) {
    this._context = context;

    // None of these touch context.device — safe to call before initialize().
    this.camera     = new Camera();
    this.globe      = new Globe(context);
    this.primitives = new PrimitiveCollection();

    // Default sun light
    this.lights.push(new DirectionalLight({
      direction:        [0.5, -0.3, -0.8],
      color:            [1.0, 0.98, 0.9],
      intensity:        1.2,
      ambientIntensity: 0.08,
    }));
  }

  /**
   * Initialise all render pipelines.  Must be called (and awaited) after
   * context.initialize() so that context.device is available.
   */
  async initialize(): Promise<void> {
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      // Read canvas size now — context.initialize() + _resizeCanvas() have already run.
      this._width  = this._context.drawingBufferWidth  || 1;
      this._height = this._context.drawingBufferHeight || 1;

      // Shared camera UBO for the G-Buffer pass
      this._cameraUniformBuffer = this._context.device.createBuffer({
        label: 'Scene.CameraUBO',
        size:  CAMERA_UBO_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // Build globe geometry
      const globePrimitive = this.globe.initialize();

      // Build opaque render pipeline (shared)
      this._opaquePipeline = buildOpaquePipeline(this._context, 'depth32float');

      // Unlit G-Buffer pipeline variant (emissive effect primitives).
      // Shares bind group layouts with the opaque pipeline so per-primitive
      // bind groups stay valid when switching pipelines mid-pass.
      this._unlitPipeline = buildUnlitPipeline(this._context, 'depth32float', this._opaquePipeline);

      // Build per-primitive bind groups
      globePrimitive.buildOpaqueBindGroup(this._opaquePipeline);

      // Give the globe the shared pipeline so imagery-tile primitives can
      // build their per-tile bind groups (see Globe.update / Globe.drawTiles).
      this.globe.setOpaquePipeline(this._opaquePipeline);

      // Camera bind group for G-Buffer pass
      this._cameraBindGroup = this._context.device.createBindGroup({
        label:  'Scene.CameraBG',
        layout: this._opaquePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this._cameraUniformBuffer } }],
      });

      // G-Buffer pass: rasterises opaque geometry into its own targets.
      // Draw inputs are injected now that pipelines/bind groups exist.
      this._gbufferPass = new GBufferPass(this._context, this._width, this._height, {
        globe:                this.globe,
        primitives:           this.primitives,
        cameraBindGroup:      this._cameraBindGroup,
        opaquePipeline:       this._opaquePipeline,
        unlitPipeline:        this._unlitPipeline,
        frustumCullingEnabled: () => this.debugFrustumCulling,
      });
      this._gbufferPass.prepare();

      // Lighting pass (creates GPU buffers in constructor — device is now available)
      this._lightingPass = new LightingPass(
        this._context,
        this._gbufferPass,
        this._context.preferredFormat
      );
      this._lightingPass.prepare();

      // Transparent pass (accumulation + composite encapsulated in execute)
      this._transparentPass = new TransparentPass(
        this._context,
        this._gbufferPass,
        this.primitives,
        this._width,
        this._height,
        this._context.preferredFormat,
        () => this.debugFrustumCulling
      );
      this._transparentPass.prepare();

      // Entity layer: screen-space billboards + polygon/polyline bridging.
      // Created last so host pipelines are ready for any pre-added entities.
      this._overlayPass = new OverlayPass(
        this._context,
        this.entities,
        this._gbufferPass,
        this._context.preferredFormat
      );
      this._overlayPass.prepare();
      this._entityVisualizer = new EntityVisualizer(this._context, this.entities, this);

      // Frame pipeline order — overlay stays before transparent so WBOIT
      // blends over the billboards (behaviour preserved from the previous
      // hand-rolled render sequence).
      this._renderPasses = [
        this._gbufferPass,
        this._lightingPass,
        this._overlayPass,
        this._transparentPass,
      ];

      this._prepared = true;
    })();

    return this._initPromise;
  }

  /**
   * Add a transparent primitive to the scene.
   */
  addTransparentPrimitive(primitive: Primitive): void {
    if (!this._transparentPass || !this._transparentPass.attachPrimitive(primitive)) {
      console.warn('Scene.addTransparentPrimitive: scene not yet prepared');
      return;
    }
    this.primitives.add(primitive);
  }

  /**
   * Add an opaque primitive to the scene.
   */
  addPrimitive(primitive: Primitive): void {
    if (this._opaquePipeline) {
      primitive.buildOpaqueBindGroup(this._opaquePipeline);
    }
    this.primitives.add(primitive);
  }

  /**
   * Handle canvas resize.
   */
  resize(width: number, height: number): void {
    this._width  = width;
    this._height = height;
    this._context.resize(width, height);
    // Pass order guarantees GBufferPass recreates its textures before
    // LightingPass rebuilds its bind group against the new views.
    for (const pass of this._renderPasses) pass.resize?.(width, height);
    this.effects.resize(width, height);
    this.camera.setAspectRatio(width / height);
  }

  /**
   * GPU-pick the object under canvas CSS-pixel coordinates (top-left
   * origin — the same space as `ScreenSpaceEventHandler` positions;
   * converted to pick-texture texels internally).  At most one readback
   * per rendered frame; results are cached until the camera or the scene
   * content changes.
   *
   * @returns null when the scene is not initialised yet, otherwise a
   * `PickResult` (`id === 0` means background / no hit).
   */
  pick(x: number, y: number): Promise<PickResult | null> {
    if (!this._prepared) return Promise.resolve(null);
    // createPickPass() returns the shared lazy instance (idempotent).
    return this.createPickPass().pick(x, y);
  }

  /**
   * Return the shared GPU picking pass, creating it (with explicitly injected
   * dependencies — PickPass has no Scene import) on first use. Callers that
   * need direct access beyond {@link pick} (e.g. `verifyIdPrecision`, debug
   * counters, manual `register`) use this factory instead of constructing
   * PickPass themselves.
   */
  createPickPass(): PickPass {
    if (!this._pickPass) this._pickPass = new PickPass(this._buildPickDeps());
    return this._pickPass;
  }

  /**
   * Render one frame.
   */
  render(): void {
    // No point encoding frames against a lost device (validation errors only).
    if (this._context.deviceLost) return;

    if (!this._prepared || !this._lightingPass) return;

    const ctx     = this._context;
    const device  = ctx.device;
    const encoder = device.createCommandEncoder({ label: 'Frame Encoder' });

    // ── Per-frame CPU prep: camera uniforms, lighting state, tile streaming ─
    this._uploadCameraUBO();
    this._lightingPass.updateLights(this.lights);

    // Imagery tile streaming: quadtree selection + per-frame texture uploads.
    // No-op while globe.layers is empty (legacy single-texture path).
    this.globe.update(this.camera);

    // ── Pass loop: gbuffer → lighting → overlay → transparent ─────────────
    const swapChainView = ctx.getCurrentTextureView();
    const time = performance.now() / 1000;
    for (const pass of this._renderPasses) {
      pass.updateCamera?.(this.camera);
      pass.execute(encoder, swapChainView, time);
    }

    // ── Effects (additive / emissive overlays, drawn last) ────────────────
    this.effects.execute(encoder, swapChainView, time);

    // ── Frame hooks (extra compute / render passes on the same encoder) ───
    for (const hook of this.frameHooks) hook(encoder, swapChainView);

    // ── Submit ────────────────────────────────────────────────────────────
    ctx.submit([encoder.finish()]);
  }

  destroy(): void {
    this._pickPass?.destroy();
    this._pickPass = null;
    this._entityVisualizer?.destroy();
    this._entityVisualizer = null;
    for (const pass of this._renderPasses) pass.destroy();
    this._renderPasses = [];
    this._gbufferPass     = null;
    this._lightingPass    = null;
    this._overlayPass     = null;
    this._transparentPass = null;
    this.globe.destroy();
    this.primitives.removeAll();
    this._cameraUniformBuffer?.destroy();
    this.effects.destroy();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _uploadCameraUBO(): void {
    if (!this._cameraUniformBuffer) return;
    const cam = this.camera;
    const data = new Float32Array(CAMERA_UBO_SIZE / 4);

    // Layout lives in src/renderer/CameraUniforms.ts (single source of truth).
    writeCameraUniforms(data, {
      viewMatrix:           cam.viewMatrix.values,
      projectionMatrix:     cam.projectionMatrix.values,
      viewProjectionMatrix: cam.viewProjectionMatrix.values,
      position:             cam.position,
      viewportWidth:        this._width,
      viewportHeight:       this._height,
      near:                 cam.frustum.near,
      far:                  cam.frustum.far,
    });

    this._context.writeBuffer(this._cameraUniformBuffer, 0, data);
  }

  // ── Accessors for subsystems (GPU picking, debugging) ──────────────────────

  /** The G-Buffer pass (null before initialize()). */
  get gbufferPass(): GBufferPass | null { return this._gbufferPass; }

  /** The shared opaque render pipeline used by the G-Buffer pass (null before initialize()). */
  getOpaquePipeline(): GPURenderPipeline | null { return this._opaquePipeline; }

  /**
   * The G-Buffer world-position texture used for the GPU-pick 1×1 readback
   * (null before initialize()).
   */
  getWorldPosTexture(): GPUTexture | null {
    return this._gbufferPass?.getWorldPosTexture() ?? null;
  }

  /** The WebGPU context (null-safe only after Viewer.initialize()). */
  get context(): Context { return this._context; }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Wire the explicit dependency object handed to PickPass. Getters are polled
   * lazily so a PickPass created after initialize() keeps working across
   * resize / destroy cycles of the underlying GPU resources.
   */
  private _buildPickDeps(): PickPassDeps {
    return {
      context:    this._context,
      camera:     this.camera,
      primitives: this.primitives,
      globe:      this.globe,
      getOpaquePipeline: () => this.getOpaquePipeline(),
      getWorldPosTexture: () => this.getWorldPosTexture(),
      frustumCullingEnabled: () => this.debugFrustumCulling,
      addFrameHook: (hook) => { this.frameHooks.push(hook); },
      removeFrameHook: (hook) => {
        const idx = this.frameHooks.indexOf(hook);
        if (idx !== -1) this.frameHooks.splice(idx, 1);
      },
    };
  }
}

export default Scene;
