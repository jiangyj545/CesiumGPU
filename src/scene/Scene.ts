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
import { PostProcessManager } from '../postprocessing/PostProcessManager';
import { RenderTexture } from '../renderer/RenderTexture';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';
import { ShadowMap } from '../shadow/ShadowMap';

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
   * P2 独立后处理管线（Bloom/LensFlare/HDR ToneMapping）。
   * 惰性创建（首次访问时 device 已就绪）。enabled=false（默认）时
   * render() 走旧 swapChain 直渲路径，本对象不被触碰。
   */
  get postProcess(): PostProcessManager {
    if (!this._postProcess) {
      this._postProcess = new PostProcessManager(
        this._context.device,
        this._context.preferredFormat,
        this._width,
        this._height,
      );
    }
    return this._postProcess;
  }

  /**
   * Entities in the scene.  Polygon/polyline visuals are rendered as
   * primitives (via {@link EntityVisualizer}); point/label billboards are
   * drawn in screen space by the overlay pass.
   */
  readonly entities: EntityCollection = new EntityCollection();

  /** Directional lights in the scene (max 8). */
  readonly lights: DirectionalLight[] = [];

  /**
   * A1 CSSM 级联阴影管理器。默认 `enabled = false`（全链路零开销、零回归）。
   * 开启：`scene.shadow.enabled = true` 后用 `scene.shadow.addCaster(primitive)`
   * 注册投射体（Primitive 系）。深度 pass 在 Lighting 之前同 encoder 执行，
   * LightingPass 以组 2 采样级联阴影图。
   */
  readonly shadow: ShadowMap;

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
  private _postProcess: PostProcessManager | null = null;
  /**
   * 场景离屏颜色 RT（postProcess.enabled 时作为 pass 循环的输出目标）。
   * 格式跟随交换链（preferredFormat）：Overlay/Transparent/特效管线的
   * 目标格式锁定为 preferredFormat，写入 rgba16float 会触发 WebGPU 校验
   * 错误 —— 故 HDR 范围受 8bit 限制（已知限制，见 P2 报告）。
   */
  private _sceneColorRT: RenderTexture | null = null;
  /**
   * B1 折射用场景颜色历史：postProcess.enabled 时每帧把 _sceneColorRT
   * copyTexture 进来（effects.execute 之前入队 → 特效采样到的是无特效的
   * 底层场景色）。格式跟随交换链（与 _sceneColorRT 一致——copyTextureToTexture
   * 要求源/目标同格式）。postProcess 未开启时保持 null（折射安全降级）。
   * usage = RENDER_ATTACHMENT|TEXTURE_BINDING|COPY_DST（RenderTexture 固定
   * 前两项，extraUsage 传 COPY_DST）。
   */
  private _sceneColorHistory: RenderTexture | null = null;
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
    this.shadow = new ShadowMap(context);
    this.camera     = new Camera();
    this.globe      = new Globe(context);
    // Terrain-aware picking: pickGlobe snaps hits via globe.terrainLayer.
    this.camera.globe = this.globe;
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
    // P2：仅在后处理已被创建时转发（未启用不分配任何资源）
    this._postProcess?.resize(width, height);
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

    // ── A1 CSSM 阴影挂点 ──────────────────────────────────────────────────
    // 深度 pass 必须先于 Lighting（receiver 同帧同 encoder 采样阴影图），
    // frameHooks 在全部 pass 之后执行、来不及喂 Lighting —— 故在此以
    // 最小挂点接入（渲染顺序不动；关阴影时 execute/receiverBinding 均为
    // no-op/null，LightingPass 保持原双组管线逐字节回归安全）。
    this.shadow.updateFrame(this.camera, this.lights);
    this.shadow.execute(encoder, () => this.getOpaquePipeline());
    this._lightingPass.setShadowSource(this.shadow.receiverBinding());

    // ── Pass loop: gbuffer → lighting → overlay → transparent ─────────────
    const swapChainView = ctx.getCurrentTextureView();
    const time = performance.now() / 1000;

    // ── P2 后处理 target 分叉（唯一回归敏感点）─────────────────────────────
    // enabled=false：target === swapChainView，后续所有调用与旧路径完全一致；
    // enabled=true ：场景写离屏 _sceneColorRT，frameHooks 后由后处理链输出。
    let target = swapChainView;
    if (this.postProcess.enabled) {
      if (!this._sceneColorRT) {
        // extraUsage COPY_SRC：折射需要把本 RT 拷入 history（原构造无 COPY_SRC，
        // copyTextureToTexture 会触发 usage 校验错误）
        this._sceneColorRT = new RenderTexture(
          ctx.device, ctx.preferredFormat, 'Scene.SceneColorRT', GPUTextureUsage.COPY_SRC);
      }
      this._sceneColorRT.ensure(this._width, this._height);
      target = this._sceneColorRT.view!;
      // LightingShader 双路径同步：HDR 直通 vs 旧 Reinhard+gamma
      this._lightingPass.setHdrOutput(true);

      // ── B1 场景颜色历史（折射数据源）────────────────────────────────
      // 在 effects.execute 之前入队 copy：特效（OceanFFTWater）采样到的是
      // 本帧无特效的底层场景色。ensure 同步 resize；未开启 postProcess 时
      // _sceneColorHistory 恒 null，特效侧折射自动降级为不混色。
      if (!this._sceneColorHistory) {
        this._sceneColorHistory = new RenderTexture(
          ctx.device, ctx.preferredFormat, 'Scene.SceneColorHistory', GPUTextureUsage.COPY_DST);
      }
      this._sceneColorHistory.ensure(this._width, this._height);
      const histSrc = this._sceneColorRT.texture;
      const histDst = this._sceneColorHistory.texture;
      if (histSrc && histDst) {
        encoder.copyTextureToTexture(
          { texture: histSrc },
          { texture: histDst },
          [this._width, this._height, 1],
        );
      }
    } else {
      this._lightingPass.setHdrOutput(false);
    }

    for (const pass of this._renderPasses) {
      pass.updateCamera?.(this.camera);
      pass.execute(encoder, target, time);
    }

    // ── Effects (additive / emissive overlays, drawn last) ────────────────
    // 与 _renderPasses 对齐：execute 前注入相机，供依赖相机矩阵的特效
    // （如 OceanFFTWater）写 camera UBO；不依赖的特效不受影响（可选协议）。
    for (const effect of this.effects.list) effect.updateCamera?.(this.camera);
    this.effects.execute(encoder, target, time);

    // ── Frame hooks (extra compute / render passes on the same encoder) ───
    for (const hook of this.frameHooks) hook(encoder, target);

    // ── P2 后处理链：sceneColor → bloom/flare/toneMapping → swapChain ──────
    if (this.postProcess.enabled) {
      this.postProcess.execute(encoder, target, swapChainView, time, this.camera);
      this._sceneColorRT!.onFrameEnd();
    }

    // ── Submit ────────────────────────────────────────────────────────────
    ctx.submit([encoder.finish()]);
  }

  destroy(): void {
    this.shadow.destroy();
    this._pickPass?.destroy();
    this._pickPass = null;
    this._postProcess?.destroy();
    this._postProcess = null;
    this._sceneColorRT?.destroy();
    this._sceneColorRT = null;
    this._sceneColorHistory?.destroy();
    this._sceneColorHistory = null;
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

  /**
   * The G-Buffer depth texture (depth32float, COPY_SRC) used by the synthetic
   * sensor channel for full-frame metric-depth readback — null before
   * initialize().
   */
  getDepthTexture(): GPUTexture | null {
    return this._gbufferPass?.getDepthTexture() ?? null;
  }

  /** The WebGPU context (null-safe only after Viewer.initialize()). */
  get context(): Context { return this._context; }

  /**
   * 场景颜色历史视图（B1 折射数据源）。仅 postProcess.enabled 时非 null；
   * 特效以惰性 getter 读取（每帧重查，resize/开关自动跟随）。
   */
  get sceneColorHistory(): GPUTextureView | null {
    if (!this.postProcess.enabled) return null;
    return this._sceneColorHistory?.view ?? null;
  }

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
