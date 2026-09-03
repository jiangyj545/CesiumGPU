import type { Effect } from '../effect/EffectManager';
import type { Context } from '../renderer/Context';
import type { Camera } from '../scene/Camera';
import { CesiumMath } from '../math/CesiumMath';
import { Cartesian3 } from '../math/Cartesian3';
import type { Matrix4 } from '../math/Matrix4';
import { VIDEO_OVERLAY_SHADER } from '../shaders/VideoOverlayShader';
import {
  EARTH_SCALE,
  buildFootprintGrid,
  buildFrustumLineVertices,
  buildTextureMatrix,
  computeFootprint,
} from './GimbalMath';
import type { Footprint, GimbalPose } from './GimbalMath';
import { DeviceOsdParser } from './Telemetry';
import type { DeviceOsdParseOptions, GimbalState } from './Telemetry';
import { VideoTexture, createVideoSampler } from './VideoTexture';
import type { VideoFrameSurface } from './VideoSource';

/**
 * VideoOverlay - 云台视频流叠加到三维球（数学核心见 `GimbalMath`）。
 *
 * **职责**：把 `VideoFrameSurface`（`<video>` / WebCodecs / 程序化画布）的
 * 最新帧贴到由云台遥测（`GimbalState`）推算的地表足迹上，同时绘制视锥线框。
 * 几何（足迹网格 / 纹理矩阵 / 线框）全部复用 `GimbalMath` 纯函数，在
 * 遥测更新后按 `minGeoIntervalMs` 节流重建；每帧仅做纹理上传 + uniform
 * 上传 + 两次 draw，零每帧分配。
 *
 * **挂载**：通过 `EffectManager`（`scene.effects.add(overlay)`），在
 * Transparent composite 之后画到 swap-chain 上（alpha blend，无深度测试）；
 * 地球远侧剔除由 shader 内射线-球求交 discard 完成。相机矩阵通过
 * `setCamera` 提供。
 *
 * **状态机**：
 * - `empty`    —— 尚无遥测（`update` / `updateFromOsd` 从未成功）
 * - `seeking`  —— 有遥测但视频源尚未送来任何一帧（网格画纯 tint*0.3 占位）
 * - `live`     —— 遥测 + 视频帧 + 足迹有效
 * - `degraded` —— 足迹射线未命中椭球（console.warn 一次；跳过网格，
 *                 近平面线框仍绘制；恢复有效时重置警告标志）
 *
 * **与 DEM 的关系**：`heightOffsetMeters` 只是把网格沿径向抬离椭球面
 * （`projectToSurface` 的 `k = s * (1 + h / EARTH_SCALE)`）。接入 DEM 时
 * 仅需替换 `GimbalMath.projectToSurface` 的采样源，本类接口不变。
 *
 * ```ts
 * const overlay = new VideoOverlay(viewer.context, {
 *   source: new HtmlVideoSource(videoEl),
 *   showFrustum: true,
 * });
 * overlay.setCamera(viewer.camera);
 * overlay.updateFromOsd(osdJson);
 * viewer.scene.effects.add(overlay);
 * ```
 */
export interface VideoOverlayOptions {
  /** 视频帧来源（引擎只消费像素，解码归实现方）。 */
  source: VideoFrameSurface;
  /** 无变焦时的垂直视场角（度），透传 `DeviceOsdParser`。默认 45。 */
  fovYDeg?: number;
  /** 视口宽高比，透传 `DeviceOsdParser`。默认 16/9。 */
  aspect?: number;
  /** 足迹网格每边细分段数。默认 24（顶点 (n+1)²，索引 6n²）。 */
  subdivisions?: number;
  /** 叠加不透明度 [0,1]。默认 1。 */
  opacity?: number;
  /** 视频色调乘子（rgb）。默认 [1,1,1]。 */
  tint?: [number, number, number];
  /** 有效视场角偏置（度），叠加到遥测 fovY 上：fovY = state.fovYRad + bias。默认 0。 */
  fovBiasDeg?: number;
  /** 网格沿径向抬离椭球面的高度（米），DEM 预留。默认 0。 */
  heightOffsetMeters?: number;
  /** 是否绘制视锥线框。默认 true。 */
  showFrustum?: boolean;
  /** 视锥线框颜色（rgb + a）。默认 [0.22, 0.88, 1.0, 0.55]。 */
  frustumColor?: [number, number, number, number];
  /** 两次几何重建之间的最小间隔（毫秒），抑制高频遥测抖动。默认 60。 */
  minGeoIntervalMs?: number;
  /** 由 zoom_factor 计算 fovY（度）的回调，透传 `DeviceOsdParser`；提供时优先于 fovYDeg。 */
  fovFromZoom?: (zoom: number) => number;
}

/** 叠加器的运行状态。 */
export type VideoOverlayState = 'empty' | 'seeking' | 'live' | 'degraded';

const UNIFORM_BYTES = 192; // viewProj + texMatrix + color + params + lineColor + camPos
const FLOATS_PER_POS = 3;
/** buildFrustumLineVertices 实际上限 48 floats（近平面 24 + 位置→4 地面角 24），留余量。 */
const LINE_FLOATS_CAP = 200;
/** 纹理投影近/远平面（米）。uv = xy / w 与 near/far 无关，仅要求 far > near > 0。 */
const TEXTURE_NEAR_METERS = 1.0;
const TEXTURE_FAR_METERS = EARTH_SCALE * 4;
/** 视锥线框近平面矩形与云台的距离（米）。 */
const FRUSTUM_NEAR_METERS = 1.0;

const DEFAULT_FOV_Y_DEG = 45;
const DEFAULT_ASPECT = 16 / 9;
const DEFAULT_SUBDIVISIONS = 24;
const DEFAULT_OPACITY = 1;
const DEFAULT_TINT: [number, number, number] = [1, 1, 1];
const DEFAULT_FOV_BIAS_DEG = 0;
const DEFAULT_HEIGHT_OFFSET_METERS = 0;
const DEFAULT_SHOW_FRUSTUM = true;
const DEFAULT_FRUSTUM_COLOR: [number, number, number, number] = [0.22, 0.88, 1.0, 0.55];
const DEFAULT_MIN_GEO_INTERVAL_MS = 60;

export class VideoOverlay implements Effect {
  readonly label = 'VideoOverlay';

  /** 可见性开关（Effect 类型要求）。 */
  show = true;

  private _context: Context;
  private _source: VideoFrameSurface;
  private _camera: Camera | null = null;

  // ── 配置 ──
  private _fovYDeg: number;
  private _aspect: number;
  private readonly _subdivisions: number;
  private _opacity: number;
  private readonly _tint: [number, number, number];
  private readonly _fovBiasRad: number;
  private _heightOffsetMeters: number;
  private _showFrustum: boolean;
  private readonly _frustumColor: [number, number, number, number];
  private readonly _minGeoIntervalSec: number;
  /** 持久化解析选项（避免每条 OSD 报文分配）。 */
  private readonly _parseOptions: DeviceOsdParseOptions;

  // ── 遥测与位姿 ──
  /** 实例级状态副本缓冲（调用方可能复用缓冲，如 TelemetryBuffer.sample 的 scratch）。 */
  private readonly _telemetry: GimbalState;
  /** 首次 update 后指向 _telemetry；null = 尚无遥测（状态 'empty'）。 */
  private _state: GimbalState | null = null;
  private readonly _pose: GimbalPose;
  private readonly _positionMeters = new Cartesian3(); // fromRadians 出参（米制 ECEF）
  private _geoDirty = false;

  // ── 几何（实例级 scratch，零每帧分配） ──
  private readonly _footprint: Footprint = {
    corners: [new Cartesian3(), new Cartesian3(), new Cartesian3(), new Cartesian3()],
    center: new Cartesian3(),
    valid: false,
    centerRangeMeters: 0.0,
  };
  private readonly _texMatrix: Matrix4;
  // TypedArrays 不写显式标注：TS 5.7+ 裸 Float32Array 标注会展宽为
  // <ArrayBufferLike>，不满足 writeBuffer 的 BufferSource（<ArrayBuffer>）。
  private _positions = new Float32Array(0);
  private _indices = new Uint32Array(0);
  private _lineVerts = new Float32Array(LINE_FLOATS_CAP);
  private _indexCount = 0;
  private _lineFloatCount = 0;
  private _geometryBuilt = false;
  private _lastGeoTime = -Infinity;
  private _warnedDegraded = false;
  private _everHadFrame = false;

  // ── GPU 资源 ──
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _lineVertexBuffer: GPUBuffer | null = null;
  private _uniformBuffer: GPUBuffer | null = null;
  private _videoTexture: VideoTexture | null = null;
  private _sampler: GPUSampler | null = null;
  private _bgl: GPUBindGroupLayout | null = null;
  private _meshPipeline: GPURenderPipeline | null = null;
  private _linePipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  /** 当前 bindGroup 绑定的纹理；VideoTexture 尺寸变化重建纹理后需重建 bindGroup。 */
  private _bgTexture: GPUTexture | null = null;
  /** 每帧上传的 uniform 暂存（48 floats = 192B）。 */
  private readonly _uniformData = new Float32Array(UNIFORM_BYTES / 4);
  private _destroyed = false;

  constructor(context: Context, options: VideoOverlayOptions) {
    this._context = context;
    this._source = options.source;
    this._fovYDeg = options.fovYDeg ?? DEFAULT_FOV_Y_DEG;
    this._aspect = options.aspect ?? DEFAULT_ASPECT;
    this._subdivisions = Math.max(1, Math.round(options.subdivisions ?? DEFAULT_SUBDIVISIONS));
    this._opacity = Math.min(1, Math.max(0, options.opacity ?? DEFAULT_OPACITY));
    const tint = options.tint ?? DEFAULT_TINT;
    this._tint = [tint[0], tint[1], tint[2]];
    this._fovBiasRad = CesiumMath.toRadians(options.fovBiasDeg ?? DEFAULT_FOV_BIAS_DEG);
    this._heightOffsetMeters = options.heightOffsetMeters ?? DEFAULT_HEIGHT_OFFSET_METERS;
    this._showFrustum = options.showFrustum ?? DEFAULT_SHOW_FRUSTUM;
    const lc = options.frustumColor ?? DEFAULT_FRUSTUM_COLOR;
    this._frustumColor = [lc[0], lc[1], lc[2], lc[3]];
    this._minGeoIntervalSec = Math.max(0, options.minGeoIntervalMs ?? DEFAULT_MIN_GEO_INTERVAL_MS) / 1000;

    this._parseOptions = {
      fovYDeg: this._fovYDeg,
      aspect: this._aspect,
      fovFromZoom: options.fovFromZoom,
    };

    this._telemetry = {
      timestampMs: 0,
      lonRad: 0,
      latRad: 0,
      heightMeters: 0,
      headingRad: 0,
      pitchRad: 0,
      rollRad: 0,
      fovYRad: CesiumMath.toRadians(this._fovYDeg),
      aspect: this._aspect,
      zoomFactor: 1,
    };
    this._pose = {
      position: new Cartesian3(),
      heading: 0,
      pitch: 0,
      roll: 0,
      fovY: this._telemetry.fovYRad + this._fovBiasRad,
      aspect: this._aspect,
    };
    this._texMatrix = buildTextureMatrix(
      {
        position: new Cartesian3(0, 0, 1.1), // 中性占位（execute 首帧重建前保持有效）
        heading: 0,
        pitch: -Math.PI / 2,
        roll: 0,
        fovY: Math.PI / 4,
        aspect: 1,
      },
      TEXTURE_NEAR_METERS,
      TEXTURE_FAR_METERS,
    );
  }

  // ── 公开 API ────────────────────────────────────────────────────────────────

  /** 提供用于投影的主相机（viewProj / position 每帧重取）。挂载后调用一次即可。 */
  setCamera(camera: Camera): void {
    this._camera = camera;
  }

  /** 推入一帧遥测状态（内部拷贝；调用方缓冲可安全复用）。置几何脏标记。 */
  update(state: GimbalState): void {
    const s = this._telemetry;
    s.timestampMs = state.timestampMs;
    s.lonRad = state.lonRad;
    s.latRad = state.latRad;
    s.heightMeters = state.heightMeters;
    s.headingRad = state.headingRad;
    s.pitchRad = state.pitchRad;
    s.rollRad = state.rollRad;
    s.fovYRad = state.fovYRad;
    s.aspect = state.aspect;
    s.zoomFactor = state.zoomFactor;
    this._state = s;
    this._applyStateToPose();
    this._geoDirty = true;
  }

  /**
   * 解析 DJI device_osd 报文并推入遥测。
   * 校验失败（biz_code 不符 / 经纬高缺失等）返回 false 且不改变状态。
   */
  updateFromOsd(raw: unknown): boolean {
    const state = DeviceOsdParser.parse(raw, this._parseOptions);
    if (state === null) return false;
    this.update(state);
    return true;
  }

  /** 设置无变焦基准视场角（度）：影响后续 OSD 解析；有遥测时有效 fovY = state.fovYRad + fovBias。 */
  setFovYDeg(deg: number): void {
    this._fovYDeg = deg;
    this._parseOptions.fovYDeg = deg;
    this._geoDirty = true;
  }

  /** 设置叠加不透明度 [0,1]（uniform 级，无几何重建）。 */
  setOpacity(opacity: number): void {
    this._opacity = Math.min(1, Math.max(0, opacity));
  }

  /** 设置是否绘制视锥线框（draw 级开关）。 */
  setShowFrustum(show: boolean): void {
    this._showFrustum = show;
  }

  /** 设置网格径向抬升高度（米），影响足迹网格贴地，置几何脏标记。 */
  setHeightOffsetMeters(meters: number): void {
    this._heightOffsetMeters = meters;
    this._geoDirty = true;
  }

  /** 最近一次重建的足迹；尚无遥测时为 null。 */
  get footprint(): Footprint | null {
    return this._geometryBuilt ? this._footprint : null;
  }

  /** 当前状态：无遥测 empty / 无视频帧 seeking / 足迹无效 degraded / 正常 live。 */
  get state(): VideoOverlayState {
    if (this._state === null) return 'empty';
    if (this._geometryBuilt && !this._footprint.valid) return 'degraded';
    if (!this._everHadFrame) return 'seeking';
    return 'live';
  }

  // ── IPass ───────────────────────────────────────────────────────────────────

  /** 幂等初始化（缓冲 / 管线 / bindGroup / 视频纹理）。 */
  prepare(): void {
    if (!this._destroyed) this._prepare();
  }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed || !this.show) return;
    this._prepare();

    const camera        = this._camera;
    const vertexBuffer  = this._vertexBuffer;
    const indexBuffer   = this._indexBuffer;
    const lineBuffer    = this._lineVertexBuffer;
    const uniformBuffer = this._uniformBuffer;
    const videoTexture  = this._videoTexture;
    const bindGroup     = this._bindGroup;
    const meshPipeline  = this._meshPipeline;
    const linePipeline  = this._linePipeline;
    if (
      this._state === null || camera === null || vertexBuffer === null ||
      indexBuffer === null || lineBuffer === null || uniformBuffer === null ||
      videoTexture === null || bindGroup === null ||
      meshPipeline === null || linePipeline === null
    ) {
      return; // 'empty'（无遥测）或未 setCamera —— IPass 契约内静默跳过
    }

    // 1) 尝试上传最新视频帧（VideoTexture 尺寸不匹配时内部自动重建纹理）。
    if (videoTexture.update(this._source)) this._everHadFrame = true;

    // 2) 纹理被重建（尺寸变化）→ 旧 view 已销毁，需重建 bindGroup。
    const tex = videoTexture.texture;
    const bgl = this._bgl;
    const sampler = this._sampler;
    if (tex !== null && tex !== this._bgTexture && bgl !== null && sampler !== null) {
      const view = videoTexture.view;
      if (view !== null) {
        this._bindGroup = this._context.device.createBindGroup({
          layout: bgl,
          entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: view },
            { binding: 2, resource: sampler },
          ],
        });
        this._bgTexture = tex;
      }
    }

    // 3) 节流重建几何（time 为秒，阈值毫秒 → 秒换算）。
    if (this._geoDirty && time - this._lastGeoTime >= this._minGeoIntervalSec) {
      this._rebuildGeometry(time);
    }

    const valid = this._geometryBuilt && this._footprint.valid;
    const seeking = !this._everHadFrame;

    // 4) uniform（实例级 Float32Array 复用，零分配；偏移对齐 shader 192B 布局）。
    const d = this._uniformData;
    d.set(camera.viewProjectionMatrix.values, 0);          // viewProj @0
    d.set(this._texMatrix.values, 16);                     // texMatrix @64
    const tint = this._tint;
    d[32] = tint[0];                                       // color @128
    d[33] = tint[1];
    d[34] = tint[2];
    d[35] = this._opacity;
    d[36] = time;                                          // params @144
    d[37] = seeking ? 1 : 0;
    d[38] = 0;
    d[39] = 0;
    const lc = this._frustumColor;
    d[40] = lc[0];                                         // lineColor @160
    d[41] = lc[1];
    d[42] = lc[2];
    d[43] = lc[3];
    const cp = camera.position;                            // camPos @176
    d[44] = cp.x;
    d[45] = cp.y;
    d[46] = cp.z;
    d[47] = 0;
    this._context.writeBuffer(uniformBuffer, 0, d);

    // 5) 单个 render pass 两次 draw：网格（triangle-list）+ 视锥线框（line-list）。
    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,
        loadOp: 'load',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setBindGroup(0, bindGroup);
    if (valid && this._indexCount > 0) {
      pass.setPipeline(meshPipeline);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, 'uint32');
      pass.drawIndexed(this._indexCount);
    }
    if (this._showFrustum && this._lineFloatCount > 0) {
      pass.setPipeline(linePipeline);
      pass.setVertexBuffer(0, lineBuffer);
      pass.draw(this._lineFloatCount / FLOATS_PER_POS);
    }
    pass.end();
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._lineVertexBuffer?.destroy();
    this._uniformBuffer?.destroy();
    this._videoTexture?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._lineVertexBuffer = null;
    this._uniformBuffer = null;
    this._videoTexture = null;
    this._sampler = null;
    this._bgl = null;
    this._meshPipeline = null;
    this._linePipeline = null;
    this._bindGroup = null;
    this._bgTexture = null;
  }

  // ── Internals ───────────────────────────────────────────────────────────────

  /** 遥测 → GimbalPose：米制 ECEF / EARTH_SCALE 归一化（引擎约定），fovY += bias。 */
  private _applyStateToPose(): void {
    const s = this._telemetry;
    const pose = this._pose;
    // Cartesian3.fromRadians 返回米制 ECEF；归一化空间 = 米 / EARTH_SCALE。
    Cartesian3.fromRadians(s.lonRad, s.latRad, s.heightMeters, this._positionMeters);
    Cartesian3.multiplyByScalar(this._positionMeters, 1 / EARTH_SCALE, pose.position);
    pose.heading = s.headingRad;
    pose.pitch = s.pitchRad;
    pose.roll = s.rollRad;
    pose.fovY = s.fovYRad + this._fovBiasRad;
    pose.aspect = s.aspect > 0 ? s.aspect : this._aspect;
  }

  /**
   * 足迹 + 网格 + 纹理矩阵 + 线框的节流重建。索引拓扑固定（构造期一次
   * 写入后内容恒定，重建时无需重传）；顶点/线框按实际数量部分写入。
   */
  private _rebuildGeometry(timeSeconds: number): void {
    const pose = this._pose;
    const fp = computeFootprint(pose, this._footprint);
    buildTextureMatrix(pose, TEXTURE_NEAR_METERS, TEXTURE_FAR_METERS, this._texMatrix);

    const grid = buildFootprintGrid(
      fp,
      this._subdivisions,
      this._heightOffsetMeters,
      this._positions,
      this._indices,
    );
    if (grid.indexCount > 0) {
      // 传全量数组（buffer 尺寸与数组一致）；draw 数量由 _indexCount 控制。
      if (this._vertexBuffer !== null) {
        this._context.writeBuffer(this._vertexBuffer, 0, this._positions);
      }
      if (this._indexBuffer !== null) {
        this._context.writeBuffer(this._indexBuffer, 0, this._indices);
      }
    }
    this._indexCount = grid.indexCount;

    const lineFloats = buildFrustumLineVertices(pose, fp, FRUSTUM_NEAR_METERS, this._lineVerts);
    if (lineFloats > 0 && this._lineVertexBuffer !== null) {
      // 线框 buffer 按 LINE_FLOATS_CAP 预留，实际绘制数 = lineFloats / 3。
      this._context.writeBuffer(this._lineVertexBuffer, 0, this._lineVerts);
    }
    this._lineFloatCount = lineFloats;

    // degraded 警告只发一次；足迹恢复有效时重置标志。
    if (!fp.valid) {
      if (!this._warnedDegraded) {
        this._warnedDegraded = true;
        console.warn('VideoOverlay: gimbal footprint invalid (corner rays miss the ellipsoid); mesh paused, near-plane frustum lines only');
      }
    } else {
      this._warnedDegraded = false;
    }

    this._geometryBuilt = true;
    this._geoDirty = false;
    this._lastGeoTime = timeSeconds;
  }

  /** 幂等初始化 GPU 资源（只跑一次；纹理重建走 execute 的 bindGroup 重建路径）。 */
  private _prepare(): void {
    if (this._meshPipeline !== null) return;
    const device = this._context.device;
    const n = this._subdivisions;

    this._positions = new Float32Array(FLOATS_PER_POS * (n + 1) * (n + 1));
    this._indices = new Uint32Array(6 * n * n);
    this._lineVerts = new Float32Array(LINE_FLOATS_CAP);

    this._vertexBuffer = device.createBuffer({
      label: 'VideoOverlay.Vertices',
      size: this._positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this._indexBuffer = device.createBuffer({
      label: 'VideoOverlay.Indices',
      size: this._indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this._lineVertexBuffer = device.createBuffer({
      label: 'VideoOverlay.FrustumLines',
      size: this._lineVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this._uniformBuffer = device.createBuffer({
      label: 'VideoOverlay.Uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._videoTexture = new VideoTexture(device, this._source.width, this._source.height);
    this._sampler = createVideoSampler(device);

    const module = this._context.createShaderModule(VIDEO_OVERLAY_SHADER, 'VideoOverlay.Module');
    const bgl = device.createBindGroupLayout({
      label: 'VideoOverlay.BGL',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });
    this._bgl = bgl;

    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    this._meshPipeline = device.createRenderPipeline({
      label: 'VideoOverlay.Mesh',
      layout,
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 12, // float32x3
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: this._context.preferredFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' }, // cullMode 默认 none（网格可两面观察）
    });
    this._linePipeline = device.createRenderPipeline({
      label: 'VideoOverlay.FrustumLines',
      layout,
      vertex: {
        module,
        entryPoint: 'vs_line',
        buffers: [{
          arrayStride: 12,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_line',
        targets: [{
          format: this._context.preferredFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'line-list' },
    });

    const view = this._videoTexture.view;
    if (view === null) {
      throw new Error('VideoOverlay: VideoTexture view is null after construction');
    }
    this._bgTexture = this._videoTexture.texture;
    this._bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuffer } },
        { binding: 1, resource: view },
        { binding: 2, resource: this._sampler },
      ],
    });
  }
}

export default VideoOverlay;
