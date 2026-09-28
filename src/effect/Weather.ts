import { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import type { Scene } from '../scene/Scene';
import type { IPass } from '../scene/passes/IPass';
import { WEATHER_SHADER } from '../shaders/WeatherShader';

/**
 * Weather - camera-anchored rain / snow (T3.5, aligns EV-Globe EV_Rain / EV_Snow).
 *
 * Implementation: a static per-instance seed point cloud animated entirely in
 * the vertex shader (`p = boxOrigin + mod(seed*boxSize + velocity*t, boxSize)`).
 * The volume is a camera-space box translated every frame to
 * `camPos + forward * depth/2`, so rain/snow follow the camera without any
 * CPU per-particle work.  Occlusion is the GPU depth test against the G-Buffer
 * depth (particles behind terrain are never rasterised).
 *
 * - rain : 1 instance = 1 elongated screen-facing quad along the fall velocity
 *          (length = speed * 0.06 s), additive blend, alpha 0.35
 * - snow : 1 instance = 1 soft round billboard with sinusoidal sway,
 *          standard alpha blend, alpha 0.8
 *
 * All sizes are given in metres and converted to the engine's normalised ECEF
 * space (1 unit = {@link Camera.EARTH_SCALE} metres) internally.
 *
 * Usage:
 * ```ts
 * const weather = new Weather(viewer.context, { type: 'rain', intensity: 0.6, scene: viewer.scene });
 * weather.setCamera(viewer.camera);
 * // depth view is re-fetched from scene.gbufferPass every frame (resize-safe);
 * // without a scene, provide it manually: weather.setDepthView(...), re-set on resize.
 * viewer.scene.effects.add(weather); // show = true
 * weather.setIntensity(0.3);         // density 0..1
 * weather.show = false;              // stop
 * weather.destroy();
 * ```
 */

export type WeatherType = 'rain' | 'snow';

/** Weather volume dimensions in metres, centred in front of the camera. */
export interface WeatherArea {
  /** Extent along the camera right axis. */
  width: number;
  /** Extent along the camera up axis. */
  height: number;
  /** Extent along the camera forward axis. */
  depth: number;
}

export interface WeatherOptions {
  /** Precipitation kind. */
  type: WeatherType;
  /** Density 0..1; instance count = intensity * maxCount. Default 1. */
  intensity?: number;
  /** Volume size in metres. Default 800 x 600 x 800 (width/height/depth). */
  area?: Partial<WeatherArea>;
  /**
   * Visual size multiplier for streak length / thickness / flake size.
   * 1 = physically scaled (rain streak = 9 m/s * 0.06 s = 0.54 m); the
   * default engine near plane is far larger than real rain, so demos usually
   * raise this.  Default 1.
   */
  sizeScale?: number;
  /** Horizontal wind drift in m/s. Default 2 (rain) / 0.8 (snow). */
  wind?: number;
  /**
   * Owning scene. When provided the G-Buffer depth view is re-fetched from
   * `scene.gbufferPass.getDepthView()` every frame, so canvas resize
   * (which destroys and recreates the G-Buffer textures) needs no manual
   * re-binding — same pattern as ParticleSystem.
   */
  scene?: Scene;
}

/** Fall speed, m/s. */
const RAIN_FALL_SPEED = 9.0;
const SNOW_FALL_SPEED = 1.5;
/** Streak duration, s: length = speed * 0.06. */
const RAIN_STREAK_SECONDS = 0.06;
/** Rain streak cross-section width, m (before sizeScale). */
const RAIN_THICKNESS_M = 0.06;
/** Snow flake diameter, m (before sizeScale). */
const SNOW_FLAKE_M = 0.1;
/** Snow sway amplitude, m (before sizeScale). */
const SNOW_SWAY_M = 3.0;
/** Straight alpha. */
const RAIN_ALPHA = 0.35;
const SNOW_ALPHA = 0.8;
/** Max instance count = intensity 1.0 densities (per spec). */
const RAIN_MAX_COUNT = 20000;
const SNOW_MAX_COUNT = 8000;

/** WeatherParams uniform: mat4x4 + 8 x vec4 = 192 bytes. */
const PARAMS_FLOATS = 48;
const PARAMS_BYTES = PARAMS_FLOATS * 4;

/** Instance seed: vec4 (position seed xyz + phase w). */
const SEED_FLOATS = 4;

export class Weather implements IPass {
  readonly label: string;

  /** Visibility switch consumed by EffectManager. */
  show = true;

  private readonly _context: Context;
  private readonly _type: WeatherType;
  private readonly _areaM: WeatherArea;
  private readonly _sizeScale: number;
  private readonly _windMs: number;
  private readonly _maxCount: number;

  private _intensity: number;
  private readonly _scene: Scene | null;
  private _camera: Camera | null = null;
  private _depthView: GPUTextureView | null = null;

  private readonly _paramsBuffer: GPUBuffer;
  private readonly _seedBuffer: GPUBuffer;
  private readonly _quadBuffer: GPUBuffer;
  /** 渲染管线共享 BGL（setTargetFormat 重建管线时 bind group 原样复用）。 */
  private readonly _bgl: GPUBindGroupLayout;
  /** T5 真 HDR：场景目标格式（null = 未下发跟随 preferredFormat 旧行为）。 */
  private _targetFormat: GPUTextureFormat | null = null;
  private _pipeline: GPURenderPipeline;
  private readonly _bindGroup: GPUBindGroup;

  private readonly _params = new Float32Array(PARAMS_FLOATS);
  private _warnedNoDepth = false;
  private _destroyed = false;

  constructor(context: Context, options: WeatherOptions) {
    if (!context.initialized) {
      throw new Error('Weather: Context.initialize() must complete first.');
    }

    this._context   = context;
    this._type      = options.type;
    this._intensity = clamp01(options.intensity ?? 1.0);
    this._areaM = {
      width:  options.area?.width  ?? 800,
      height: options.area?.height ?? 600,
      depth:  options.area?.depth  ?? 800,
    };
    this._sizeScale = Math.max(options.sizeScale ?? 1.0, 0.01);
    this._windMs    = options.wind ?? (this._type === 'rain' ? 2.0 : 0.8);
    this._maxCount  = this._type === 'rain' ? RAIN_MAX_COUNT : SNOW_MAX_COUNT;
    this._scene     = options.scene ?? null;
    this.label      = `Weather(${this._type})`;

    if (!Number.isFinite(this._areaM.width) || this._areaM.width  <= 0 ||
        !Number.isFinite(this._areaM.height) || this._areaM.height <= 0 ||
        !Number.isFinite(this._areaM.depth)  || this._areaM.depth  <= 0) {
      throw new Error('Weather: area width/height/depth must be positive metres.');
    }

    const device = context.device;

    // ── Uniform + geometry buffers ─────────────────────────────────────────
    this._paramsBuffer = device.createBuffer({
      label: `${this.label}.Params`,
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Static seed cloud, pre-generated once; animation is shader-side only.
    this._seedBuffer = device.createBuffer({
      label: `${this.label}.Seeds`,
      size: this._maxCount * SEED_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(this._seedBuffer, 0, this._createSeedData());

    // 4-vertex quad (triangle strip), corner in [-1,1].
    this._quadBuffer = device.createBuffer({
      label: `${this.label}.Quad`,
      size: 32,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(
      this._quadBuffer, 0,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]),
    );

    // ── Render pipeline (instanced quads; blending by weather type) ────────
    this._bgl = device.createBindGroupLayout({
      label: `${this.label}.BGL`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this._pipeline = this._buildRenderPipeline();
    this._bindGroup = device.createBindGroup({
      layout: this._bgl,
      entries: [{ binding: 0, resource: { buffer: this._paramsBuffer } }],
    });
  }

  /** Density 0..1 (instance count = intensity * maxCount). */
  get intensity(): number {
    return this._intensity;
  }

  /** Weather kind ('rain' | 'snow'). */
  get type(): WeatherType {
    return this._type;
  }

  setIntensity(v: number): void {
    this._intensity = clamp01(v);
  }

  /** Provide the camera the volume follows (required before execute). */
  setCamera(camera: Camera): void {
    this._camera = camera;
  }

  /**
   * Provide the G-Buffer depth view used for terrain occlusion.  Re-call after
   * canvas resize (the G-Buffer recreates its textures): e.g. every frame in a
   * `preRender` hook read `scene.gbufferPass.textures.depthView`.  Without a
   * depth view the draw is skipped (a one-time warning is logged).
   */
  setDepthView(view: GPUTextureView | null): void {
    this._depthView = view;
    if (view !== null) this._warnedNoDepth = false;
  }

  /**
   * T5 真 HDR: target format delivery (optional EffectManager protocol;
   * Scene.render calls it every frame with the current render target format).
   * On change the render pipeline is rebuilt (buffers / BGL / bind group are
   * format-independent and reused as-is); same format returns early (idempotent).
   * Call paths without this protocol keep the previous preferredFormat behavior.
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._destroyed || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._pipeline = this._buildRenderPipeline();
  }

  /** IPass: GPU resources are built in the constructor; nothing to do. */
  prepare(): void {
    /* resources are created eagerly in the constructor */
  }

  /**
   * IPass: canvas resize (the G-Buffer already recreated its depth texture in
   * Scene.resize before this runs) — drop the cached depth view so a destroyed
   * view is never re-bound. With `scene` held, execute() re-fetches anyway.
   */
  resize(): void {
    this._depthView = null;
  }

  /**
   * IPass: refresh camera-dependent uniforms and draw the instanced particles
   * composited over `target` (loadOp load), depth-tested against the G-Buffer.
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed || !this._camera || !this.show) return;

    const count = Math.round(this._maxCount * this._intensity);
    if (count <= 0) return;

    const cam = this._camera;
    // Accessing viewProjectionMatrix flushes lazy camera matrix updates.
    const viewProj = cam.viewProjectionMatrix.values;

    // ── Camera basis (normalised ECEF) ─────────────────────────────────────
    const px = cam.position.x, py = cam.position.y, pz = cam.position.z;
    let fx = cam.direction.x, fy = cam.direction.y, fz = cam.direction.z;
    let fm = Math.hypot(fx, fy, fz);
    if (fm < 1e-12) { fx = 0; fy = 0; fz = -1; fm = 1; }
    fx /= fm; fy /= fm; fz /= fm;

    let rx = fy * cam.up.z - fz * cam.up.y;  // cross(forward, up)
    let ry = fz * cam.up.x - fx * cam.up.z;
    let rz = fx * cam.up.y - fy * cam.up.x;
    let rm = Math.hypot(rx, ry, rz);
    if (rm < 1e-12) { rx = 1; ry = 0; rz = 0; rm = 1; } // degenerate: nadir + parallel up
    rx /= rm; ry /= rm; rz /= rm;

    const ux = ry * fz - rz * fy;            // cross(right, forward)
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    // ── Box in front of the camera ─────────────────────────────────────────
    const m2u = 1.0 / Camera.EARTH_SCALE;
    const bw = this._areaM.width  * m2u;
    const bh = this._areaM.height * m2u;
    const bd = this._areaM.depth  * m2u;
    const halfDepth = bd * 0.5;
    const ox = px + fx * halfDepth;
    const oy = py + fy * halfDepth;
    const oz = pz + fz * halfDepth;

    // ── Motion / style, normalised units ───────────────────────────────────
    const isRain      = this._type === 'rain';
    const fallSpeed   = (isRain ? RAIN_FALL_SPEED : SNOW_FALL_SPEED) * m2u;
    const windSpeed   = this._windMs * m2u;
    const streakLen   = RAIN_FALL_SPEED * RAIN_STREAK_SECONDS * this._sizeScale * m2u;
    const elementSize = (isRain ? RAIN_THICKNESS_M : SNOW_FLAKE_M) * this._sizeScale * m2u;
    const swayAmp     = isRain ? 0.0 : SNOW_SWAY_M * this._sizeScale * m2u;
    const alpha       = isRain ? RAIN_ALPHA : SNOW_ALPHA;

    const u = this._params;
    u.set(viewProj, 0);
    u[16] = px; u[17] = py; u[18] = pz; u[19] = 0;
    u[20] = rx; u[21] = ry; u[22] = rz; u[23] = 0;
    u[24] = ux; u[25] = uy; u[26] = uz; u[27] = 0;
    u[28] = fx; u[29] = fy; u[30] = fz; u[31] = 0;
    u[32] = ox; u[33] = oy; u[34] = oz; u[35] = 0;
    u[36] = bw; u[37] = bh; u[38] = bd; u[39] = time;
    u[40] = fallSpeed; u[41] = swayAmp; u[42] = isRain ? streakLen : 0; u[43] = windSpeed;
    u[44] = elementSize; u[45] = alpha; u[46] = 0; u[47] = 0;
    this._context.writeBuffer(this._paramsBuffer, 0, u);

    // ── Draw (composite over the scene, depth-tested) ──────────────────────
    // Depth view is re-fetched every frame when a scene is held: the G-Buffer
    // destroys and recreates its textures on canvas resize, so a cached view
    // would go stale and invalidate the whole command buffer (same pattern as
    // ParticleSystem). Falls back to the setDepthView cache otherwise.
    const depthView = this._scene
      ? (this._scene.gbufferPass?.getDepthView() ?? null)
      : this._depthView;
    if (!depthView) {
      if (!this._warnedNoDepth) {
        this._warnedNoDepth = true;
        console.warn(`${this.label}: no depth view (scene or setDepthView) - skipping draw.`);
      }
      return;
    }

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view:    target,
        loadOp:  'load',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        // Read-only: the G-Buffer depth was already written this frame (and
        // consumed read-only by the transparent pass); a writable re-use in
        // the same encoder would invalidate the whole command buffer.
        depthReadOnly: true,
      },
    });
    pass.setPipeline(this._pipeline);
    pass.setVertexBuffer(0, this._quadBuffer);
    pass.setVertexBuffer(1, this._seedBuffer);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(4, count);
    pass.end();
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._paramsBuffer.destroy();
    this._seedBuffer.destroy();
    this._quadBuffer.destroy();
  }

  /**
   * Build / rebuild the render pipeline (shared by the constructor and
   * setTargetFormat; the descriptor matches the pre-extraction one field for
   * field, with the target format taken from _targetFormat, falling back to
   * the context preferredFormat while null). Instanced quads with per-type
   * blending and the read-only G-Buffer depth test are unchanged.
   */
  private _buildRenderPipeline(): GPURenderPipeline {
    const device = this._context.device;
    const module = this._context.createShaderModule(WEATHER_SHADER, `${this.label}.Module`);
    const isRain = this._type === 'rain';
    return device.createRenderPipeline({
      label: `${this.label}.Pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
      vertex: {
        module,
        entryPoint: isRain ? 'vs_rain' : 'vs_snow',
        buffers: [
          {
            // Per-vertex quad corner.
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          {
            // Per-instance seed vec4.
            arrayStride: 16,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: isRain ? 'fs_rain' : 'fs_snow',
        targets: [{
          format: this._targetFormat ?? this._context.preferredFormat,
          blend: isRain
            ? { // additive: pale streaks glow over the lit scene
                color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
              }
            : { // standard alpha for soft white flakes
                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      // Depth-test against the G-Buffer depth; never write it.
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });
  }

  /**
   * Per-instance seeds: xyz = uniform position seed in [0,1) per box axis,
   * w = random phase in [0,1) for the snow sway sin/cos.
   */
  private _createSeedData(): Float32Array<ArrayBuffer> {
    const data = new Float32Array(this._maxCount * SEED_FLOATS);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random();
    }
    return data;
  }
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export default Weather;
