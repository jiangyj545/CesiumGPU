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
import type { PickResult } from '../picking/PickPass';
import { OverlayPass } from './passes/OverlayPass';
import { EntityVisualizer } from './EntityVisualizer';
import { EntityCollection } from '../dataSource/EntityCollection';
import { EffectManager } from '../effect/EffectManager';
import { GBUFFER_SHADER } from '../shaders/GBufferShader';

/**
 * Scene - Orchestrates all rendering passes.
 *
 * Rendering pipeline:
 *  1. G-Buffer pass  → writes albedo / normal / worldPos for all opaque prims
 *  2. Lighting pass  → reads G-Buffer, outputs lit opaque scene
 *  3. Transparent accumulation pass → WBOIT accumulate
 *  4. Transparent composite pass    → blend over opaque result
 *  5. Effect passes  → EffectManager executes IPass effects (additive/emissive
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
    if (this._prepared) return;

    // Read canvas size now — context.initialize() + _resizeCanvas() have already run.
    this._width  = this._context.drawingBufferWidth  || 1;
    this._height = this._context.drawingBufferHeight || 1;

    // G-Buffer pass (creates GPU textures lazily on first use)
    this._gbufferPass = new GBufferPass(this._context, this._width, this._height);

    // Lighting pass (creates GPU buffers in constructor — device is now available)
    this._lightingPass = new LightingPass(
      this._context,
      this._gbufferPass,
      this._context.preferredFormat
    );

    // Transparent pass
    this._transparentPass = new TransparentPass(
      this._context,
      this._width,
      this._height,
      this._context.preferredFormat
    );

    // Shared camera UBO for the G-Buffer pass
    this._cameraUniformBuffer = this._context.device.createBuffer({
      label: 'Scene.CameraUBO',
      size:  256,
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

    // Lighting pass
    this._lightingPass.prepare();

    // Transparent pass
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

    this._prepared = true;
  }

  /**
   * Add a transparent primitive to the scene.
   */
  addTransparentPrimitive(primitive: Primitive): void {
    if (
      !this._transparentPass ||
      !this._transparentPass.accumulationPipeline ||
      !this._transparentPass.cameraBindGroupLayout
    ) {
      console.warn('Scene.addTransparentPrimitive: scene not yet prepared');
      return;
    }
    primitive.buildTransparentBindGroups(
      this._transparentPass.accumulationPipeline,
      this._transparentPass.cameraUniformBuffer,
      this._transparentPass.cameraBindGroupLayout
    );
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
    this._gbufferPass?.resize(width, height);
    this._transparentPass?.resize(width, height);
    this._lightingPass?.rebuildGBufferBindGroup();
    this._overlayPass?.resize(width, height);
    this.effects.resize(width, height);
    this.camera.setAspectRatio(width / height);
  }

  /**
   * GPU-pick the object under canvas device-pixel coordinates (top-left
   * origin).  At most one readback per rendered frame; results are cached
   * until the camera or the scene content changes.
   *
   * @returns null when the scene is not initialised yet, otherwise a
   * `PickResult` (`id === 0` means background / no hit).
   */
  pick(x: number, y: number): Promise<PickResult | null> {
    if (!this._prepared) return Promise.resolve(null);
    if (!this._pickPass) this._pickPass = new PickPass(this);
    return this._pickPass.pick(x, y);
  }

  /**
   * Render one frame.
   */
  render(): void {
    if (
      !this._prepared ||
      !this._opaquePipeline ||
      !this._cameraBindGroup ||
      !this._gbufferPass ||
      !this._lightingPass ||
      !this._transparentPass
    ) return;

    const ctx     = this._context;
    const device  = ctx.device;
    const encoder = device.createCommandEncoder({ label: 'Frame Encoder' });

    // ── Upload camera uniforms ─────────────────────────────────────────────
    this._uploadCameraUBO();
    this._lightingPass.updateCamera(this.camera);
    this._lightingPass.updateLights(this.lights);
    this._transparentPass.updateCamera(this.camera);

    // Imagery tile streaming: quadtree selection + per-frame texture uploads.
    // No-op while globe.layers is empty (legacy single-texture path).
    this.globe.update(this.camera);

    // ── 1. G-Buffer pass (opaque geometry) ────────────────────────────────
    {
      const pass = this._gbufferPass.beginRenderPass(encoder);
      pass.setPipeline(this._opaquePipeline);

      // Globe (base sphere + per-tile imagery primitives)
      const gp = this.globe.primitive;
      if (gp && this.globe.show) {
        gp.drawOpaque(pass, this._cameraBindGroup);
        this.globe.drawTiles(pass, this._cameraBindGroup);
      }

      // Other opaque primitives (switch to the unlit pipeline for emissive
      // geometry — bind groups are layout-compatible with both pipelines)
      const opaquePipeline = this._opaquePipeline;
      const unlitPipeline  = this._unlitPipeline;
      let currentPipeline  = opaquePipeline;
      for (const prim of this.primitives.opaque) {
        const wanted = prim.unlit && unlitPipeline ? unlitPipeline : opaquePipeline;
        if (wanted !== currentPipeline) {
          pass.setPipeline(wanted);
          currentPipeline = wanted;
        }
        prim.drawOpaque(pass, this._cameraBindGroup);
      }

      pass.end();
    }

    // ── 2. Lighting pass (deferred shading, outputs to swap-chain) ────────
    const swapChainView = ctx.getCurrentTextureView();
    this._lightingPass.execute(encoder, swapChainView);

    // ── 2.5 Entity overlay (screen-space points/labels, depth-tested) ─────
    // Drawn before the transparent composite so WBOIT blends over billboards.
    this._overlayPass?.updateCamera(this.camera);
    this._overlayPass?.execute(encoder, swapChainView);

    // ── 3. Transparent accumulation pass ──────────────────────────────────
    const transparents = this.primitives.transparent;
    if (transparents.length > 0 && this._transparentPass.accumulationPipeline) {
      const depthView = this._gbufferPass.textures.depthView;
      const accumPass = this._transparentPass.beginAccumulationPass(encoder, depthView);
      accumPass.setPipeline(this._transparentPass.accumulationPipeline);

      for (const prim of transparents) {
        prim.drawTransparent(accumPass);
      }
      accumPass.end();

      // ── 4. Transparent composite pass ─────────────────────────────────
      this._transparentPass.executeComposite(encoder, swapChainView);
    }

    // ── 5. Effects (additive / emissive overlays, drawn last) ─────────────
    this.effects.execute(encoder, swapChainView, performance.now() / 1000);

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
    this._overlayPass?.destroy();
    this._overlayPass = null;
    this.globe.destroy();
    this.primitives.removeAll();
    this._gbufferPass?.destroy();
    this._lightingPass?.destroy();
    this._transparentPass?.destroy();
    this._cameraUniformBuffer?.destroy();
    this.effects.destroy();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _uploadCameraUBO(): void {
    if (!this._cameraUniformBuffer) return;
    const cam = this.camera;
    const data = new Float32Array(64); // 256 bytes / 4

    data.set(cam.viewMatrix.values,            0);   // viewMatrix (offset 0)
    data.set(cam.projectionMatrix.values,     16);   // projMatrix (offset 64)
    data.set(cam.viewProjectionMatrix.values, 32);   // viewProjMatrix (offset 128)
    data[48] = cam.position.x;                       // cameraPosition.x
    data[49] = cam.position.y;
    data[50] = cam.position.z;
    data[51] = 1.0;
    data[52] = this._width;                          // viewportSize
    data[53] = this._height;
    data[54] = cam.frustum.near;                     // nearFar
    data[55] = cam.frustum.far;

    this._context.writeBuffer(this._cameraUniformBuffer, 0, data);
  }

  // ── Accessors for subsystems (GPU picking, debugging) ──────────────────────

  /** The G-Buffer pass (null before initialize()). */
  get gbufferPass(): GBufferPass | null { return this._gbufferPass; }

  /** The shared opaque render pipeline used by the G-Buffer pass. */
  get opaquePipeline(): GPURenderPipeline | null { return this._opaquePipeline; }

  /** The WebGPU context (null-safe only after Viewer.initialize()). */
  get context(): Context { return this._context; }
}

// Re-export for convenience
export { GBUFFER_SHADER };

export default Scene;
