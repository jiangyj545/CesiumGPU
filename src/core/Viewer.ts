import { Context } from '../renderer/Context';
import { Scene } from '../scene/Scene';
import { Cartesian3 } from '../math/Cartesian3';
import { Camera } from '../scene/Camera';
import { Globe } from '../scene/Globe';
import { Color } from '../math/Color';
import { ScreenSpaceEventHandler } from './ScreenSpaceEventHandler';
import type { PinchPosition, WheelPosition, ScreenSpacePosition } from './ScreenSpaceEventHandler';
import type { PickResult } from '../picking/PickPass';
import { VideoRecorder } from '../media/VideoRecorder';
import type { RecordingOptions } from '../media/VideoRecorder';
import { Entity } from '../dataSource/Entity';
import { resolveProperty } from '../dataSource/CallbackProperty';
import { EntityTracker } from '../scene/EntityTracker';
import { BoundingSphere } from '../scene/BoundingSphere';
import { HeadingPitchRange } from '../scene/HeadingPitchRange';
import type { Tileset3D } from '../tileset/Tileset3D';
import type { TerrainLayer } from '../terrain/TerrainLayer';
import { RenderScheduler } from './RenderScheduler';

export interface ViewerOptions {
  /** GPU power preference. Default: 'high-performance' */
  powerPreference?: GPUPowerPreference;
  /** Initial background color (space). */
  backgroundColor?: Color;
  /**
   * P2-5 按需渲染（Cesium requestRenderMode 对齐）：场景静止时跳过
   * Scene.render 省 CPU。`scene.requestRender()` / 相机 / 实体 / 图元 /
   * 瓦片 / 特效 / resize 变化自动触发渲染。默认 false（每帧渲染）。
   */
  requestRenderMode?: boolean;
  /**
   * 静止超过该秒数自动补渲一帧（P2-5，Cesium maximumRenderTimeChange
   * 简化语义）。默认 Infinity（永不，对齐 hexapod mapController）。
   */
  maximumRenderTimeChange?: number;
}

/**
 * Viewer - The main CesiumGPU viewer class.
 *
 * Usage:
 * ```typescript
 * const viewer = new Viewer('cesiumContainer');
 * await viewer.initialize();
 * viewer.camera.setView({
 *   destination: Cartesian3.fromDegrees(116.4, 39.9, 10_000_000),
 * });
 * ```
 *
 * Matches Cesium's Viewer API.
 */
export class Viewer {
  private _container: HTMLElement;
  private _canvas: HTMLCanvasElement;
  private _context: Context;
  private _scene: Scene;
  private _handler: ScreenSpaceEventHandler | null = null;

  /**
   * Optional callback invoked at the beginning of every render frame, before
   * the scene is rendered.  Use this to update lights, primitives, or the clock
   * based on the current simulation time.
   *
   * ```typescript
   * viewer.preRender = () => {
   *   clock.tick();
   *   scene.lights[0].direction = SunPosition.computeSunDirection(clock.currentTime);
   * };
   * ```
   */
  preRender: (() => void) | null = null;

  /**
   * Optional GPU-pick callback invoked for every LEFT_CLICK once registered.
   * Receives the pick result (`null` = scene not ready) and the click
   * position in canvas CSS pixels.  Without a registered callback no picking
   * happens on click.
   *
   * ```typescript
   * viewer.onPick = (result) => console.log(result?.id, result?.worldPosition);
   * ```
   */
  onPick: ((result: PickResult | null, position: ScreenSpacePosition) => void) | null = null;

  /**
   * When true (drawing tools active, e.g. DrawHandler), camera drag handlers
   * are suppressed so pointer input drives drawing instead of orbiting.
   * Wheel / pinch zoom stay active.
   */
  drawMode = false;

  private _animFrameId: number | null = null;
  private _fps = 0;
  private _frameCount = 0;
  private _fpsInterval = 0;
  private _fpsEl: HTMLElement | null = null;
  private _lastFrameTime: number | null = null;

  // Mouse orbit state (left button)
  private _isDragging = false;
  private _lastMouseX = 0;
  private _lastMouseY = 0;

  // Mouse zoom state (right button)
  private _isRightDragging = false;
  private _lastRightMouseY = 0;

  // Mouse pitch/rotation state (middle button) — orbits around the globe point
  // that was under the cursor when the button was pressed.
  private _isMiddleDragging = false;
  private _lastMiddleMouseX = 0;
  private _lastMiddleMouseY = 0;
  private _middlePivot: Cartesian3 | null = null;

  private _destroyed = false;
  private _initPromise: Promise<void> | null = null;
  private _resizeObserver: ResizeObserver | null = null;

  /** Tracked-entity follow logic (`trackedEntity` getter/setter below). */
  private _tracker: EntityTracker;

  // 4K 视频输出（惰性创建，见 captureStream/startRecording/stopRecording）
  private _recorder: VideoRecorder | null = null;
  /** 录制期间挂起容器自适应 resize（否则 ResizeObserver 会改回录制分辨率）。 */
  private _autoResizePaused = false;

  // ── P2-5 按需渲染 ─────────────────────────────────────────────────────────
  private _requestRenderMode = false;
  private _scheduler = new RenderScheduler(Number.POSITIVE_INFINITY);
  private _lastCameraFp = '';
  private _lastSceneFp = '';

  /** 显式请求渲染下一帧（等价 Cesium scene.requestRender；任何时候可用）。 */
  requestRender(): void {
    this._scheduler.requestRender();
  }

  /** 是否启用按需渲染（P2-5）。 */
  get requestRenderMode(): boolean {
    return this._requestRenderMode;
  }

  constructor(containerOrId: string | HTMLElement, options: ViewerOptions = {}) {
    if (typeof containerOrId === 'string') {
      const el = document.getElementById(containerOrId);
      if (!el) throw new Error(`Container element not found: ${containerOrId}`);
      this._container = el;
    } else {
      this._container = containerOrId;
    }

    // Create canvas
    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'display:block;width:100%;height:100%;';
    this._container.appendChild(this._canvas);

    this._context = new Context(this._canvas);
    this._scene   = new Scene(this._context);
    this._tracker = new EntityTracker(this._scene.camera);
    // P2-5 按需渲染：调度器 + scene.requestRender 钩子。
    this._requestRenderMode = options.requestRenderMode ?? false;
    this._scheduler = new RenderScheduler(options.maximumRenderTimeChange ?? Number.POSITIVE_INFINITY);
    this._scene.setRequestRenderHook(() => this._scheduler.requestRender());

    // Look for FPS display element
    this._fpsEl = document.getElementById('fps');
  }

  // ── Accessors (Cesium-compatible) ─────────────────────────────────────────

  get scene(): Scene  { return this._scene; }
  get camera(): Camera { return this._scene.camera; }
  get globe(): Globe   { return this._scene.globe; }
  get canvas(): HTMLCanvasElement { return this._canvas; }
  /** WebGPU device context. Valid after {@link initialize}. */
  get context(): Context { return this._context; }
  /** The canvas input abstraction (T1.2). Null before {@link initialize}. */
  get screenSpaceEventHandler(): ScreenSpaceEventHandler | null { return this._handler; }

  // ── Initialization ────────────────────────────────────────────────────────

  /**
   * Initialize WebGPU and prepare all render resources.
   * Must be awaited before the viewer is usable.
   */
  async initialize(): Promise<void> {
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      await this._context.initialize();

      // Stop the render loop and surface a message as soon as the device is lost.
      this._context.onDeviceLost = () => {
        if (this._destroyed) return; // deliberate destroy() also resolves device.lost
        this._stopRenderLoop();
        this._showDeviceLostMessage();
      };

      this._resizeCanvas();
      await this._scene.initialize();

      // Default camera position: 3× Earth radius above equator
      this._scene.camera.position.set(0, 0, 3.0);
      this._scene.camera.setAspectRatio(this._canvas.width / this._canvas.height);

      this._setupInputHandlers();
      this._startRenderLoop();
    })();

    return this._initPromise;
  }

  /**
   * Stop rendering and release all GPU resources.
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    if (this._animFrameId !== null) cancelAnimationFrame(this._animFrameId);
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this._handler?.destroy();
    this._handler = null;
    this._scene.destroy();
    this._context.destroy();
    this._canvas.remove();
  }

  // ── 4K 视频输出（委托 VideoRecorder） ─────────────────────────────────────

  /**
   * 采集画布为 MediaStream。
   * @param fps 采集帧率，默认 60。
   */
  captureStream(fps = 60): MediaStream {
    return this._getRecorder().captureStream(fps);
  }

  /**
   * 开始录制：保存旧尺寸 → resize 到目标分辨率（如 4K）→ 等一帧 → start。
   * 录制期间挂起 ResizeObserver 自适应 resize。
   * @throws 已在录制中、环境不支持或 mime 候选链全失败时 reject 带原因。
   */
  async startRecording(opts?: RecordingOptions): Promise<void> {
    this._autoResizePaused = true;
    try {
      await this._getRecorder().startRecording(opts);
    } catch (err) {
      this._autoResizePaused = false; // 失败不留挂起态，原因照原样上抛（Invariant 1）
      throw err;
    }
  }

  /**
   * 停止录制并 resolve 视频 Blob；同时恢复原尺寸并解除 resize 挂起。
   * @throws 非 recording 态调用时 reject 带原因。
   */
  async stopRecording(): Promise<Blob> {
    const blob = await this._getRecorder().stopRecording();
    this._autoResizePaused = false;
    this._resizeCanvas(); // 与容器重新对齐一次（录制期间容器可能变化）
    return blob;
  }

  /** 当前录制状态：'idle' | 'recording'。 */
  get recordingState(): 'idle' | 'recording' {
    return this._recorder ? this._recorder.state : 'idle';
  }

  /** 惰性创建 VideoRecorder（Viewer 满足其最小宿主接口）。 */
  private _getRecorder(): VideoRecorder {
    if (!this._recorder) this._recorder = new VideoRecorder(this);
    return this._recorder;
  }


  // ── Tracked entity / zoomTo ───────────────────────────────────────────────

  /**
   * The live terrain layer backing the globe (read-only), or `undefined`
   * while no terrain provider is attached — direct drop-in for
   * `sampleTerrainMostDetailed(viewer.terrainLayer, positions)`
   * (hexapod measurement.js / waypointEditor.js call shape).
   */
  get terrainLayer(): TerrainLayer | undefined {
    return this._scene.globe.terrainLayer ?? undefined;
  }

  /**
   * The entity the camera follows, mirroring Cesium's `Viewer.trackedEntity`.
   *
   * Setting an entity captures the current camera→entity offset (distance +
   * heading / pitch) and the render loop then translates the camera in
   * lockstep with the entity every frame (`EntityTracker.update`). User
   * orbit / zoom while tracking adjust the offset freely — following resumes
   * from the new relative placement on the next frame. Assign `undefined` to
   * release (call `camera.lookAtTransform(Matrix4.IDENTITY)` afterwards for
   * Cesium call-site parity; it is a no-op unlock in this engine).
   */
  get trackedEntity(): Entity | undefined {
    return this._tracker.entity;
  }

  set trackedEntity(entity: Entity | undefined) {
    if (entity) this._tracker.track(entity);
    else this._tracker.release();
  }

  /**
   * Fly the camera to frame a target (Cesium `Viewer.zoomTo`).
   *
   * - `Entity` → its (resolved) position; `Cartesian3` (metres ECEF) → that
   *   point; `Cartesian3[]` (normalised ECEF) → bounding sphere of the points;
   *   `Tileset3D` → its root tile's bounding sphere.
   * - `headingPitchRange` defaults to heading 0 / pitch −90° with a range
   *   sized from the bounding sphere (entities: 50 m).
   */
  zoomTo(
    target: Entity | Cartesian3 | Cartesian3[] | Tileset3D,
    headingPitchRange?: HeadingPitchRange
  ): void {
    const sphere = Viewer.boundingSphereForTarget(target);
    if (!sphere) return;
    const offset =
      headingPitchRange ??
      new HeadingPitchRange(0, -Math.PI / 2, Math.max(sphere.radius * 2.5, 50));
    this._scene.camera.flyToBoundingSphere(sphere, { offset });
  }

  /** Resolve the bounding sphere for a `zoomTo` target (null = nothing to do). */
  private static boundingSphereForTarget(
    target: Entity | Cartesian3 | Cartesian3[] | Tileset3D
  ): BoundingSphere | null {
    if (target instanceof Entity) {
      const pos = resolveProperty(target.position);
      if (!pos) return null;
      return new BoundingSphere(pos, 0);
    }
    if (Array.isArray(target)) return BoundingSphere.fromPoints(target);
    if ('uri' in target || 'readyPromise' in target || 'root' in target) {
      // Tileset3D: root tile world bounding sphere (center normalised ECEF,
      // radius in normalised units → convert to metres).
      const sphere = (target as Tileset3D).root?.sphere;
      if (!sphere) return null;
      return new BoundingSphere(sphere.center, sphere.radius * 6378137.0);
    }
    // Plain metre-ECEF Cartesian3 → normalised.
    return new BoundingSphere(
      Cartesian3.multiplyByScalar(target, 1 / 6378137.0, new Cartesian3()),
      0
    );
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _resizeCanvas(): void {
    if (this._autoResizePaused) return; // 录制 4K 期间禁止把画布改回容器尺寸
    const w = this._container.clientWidth  || window.innerWidth;
    const h = this._container.clientHeight || window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    this._canvas.width  = Math.round(w * dpr);
    this._canvas.height = Math.round(h * dpr);
    this._scene.resize(this._canvas.width, this._canvas.height);
    this._scheduler.markDirty(); // P2-5：resize 即脏
  }

  /** Cancel the pending animation frame so no further frames are rendered. */
  private _stopRenderLoop(): void {
    if (this._animFrameId !== null) {
      cancelAnimationFrame(this._animFrameId);
      this._animFrameId = null;
    }
  }

  /** Show a device-lost overlay in the container (created once). */
  private _showDeviceLostMessage(): void {
    if (this._container.querySelector('[data-gpu-lost]')) return;
    const el = document.createElement('div');
    el.setAttribute('data-gpu-lost', '');
    el.style.cssText =
      'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
      'background:rgba(0,0,0,0.75);color:#fff;font:14px/1.5 system-ui,sans-serif;' +
      'text-align:center;padding:16px;z-index:10;';
    el.textContent = 'WebGPU device lost — rendering stopped. Reload the page to recover.';
    this._container.appendChild(el);
  }

  private _startRenderLoop(): void {
    this._fpsInterval = performance.now();

    const loop = (now: number) => {
      if (this._destroyed) return;

      // Stop the loop once the GPU device is lost (further frames would only
      // keep failing on getCurrentTexture / command submission).
      if (this._context.deviceLost) {
        console.error('[Viewer] WebGPU device lost — rendering stopped. Reload the page to recover.');
        this._stopRenderLoop();
        this._showDeviceLostMessage();
        return;
      }

      this._animFrameId = requestAnimationFrame(loop);

      // Pre-render hook (clock tick, sun direction update, etc.)
      this.preRender?.();

      // Camera flight (T2.6): advance the active flyTo by the frame delta time.
      // 飞行用宽松上限（1s/帧 vs 渲染 dt 的 0.1s）：动画应走墙钟——慢机/
      // 标签页冻结解冻后 3s 飞行被 0.1s 上限拉长成 15s+ 墙钟（96 页
      // 「相机 15 秒还悬在半空」的放大器），1s/帧保证低速下也在墙钟内
      // 走完、解冻后 3 帧内到位。
      const dtFlight = this._lastFrameTime === null ? 0 : Math.min((now - this._lastFrameTime) / 1000, 1.0);
      //（渲染侧其他 dt 消费者仍用 0.1s 上限，此处不再重复计算）
      this._lastFrameTime = now;
      this._scene.camera.updateFlight(dtFlight);

      // trackedEntity follow: translate the camera by the entity's
      // frame-to-frame position delta (see EntityTracker).
      this._tracker.update();

      // P2-5 按需渲染：静止场景跳过 Scene.render（相机/实体/图元/瓦片/
      // 特效/动态 Property 任一变化 → RenderScheduler 置脏）。
      if (this._requestRenderMode) {
        this._pollDirtySources(now / 1000);
        if (!this._scheduler.shouldRender(now / 1000)) return;
      }

      this._scene.render();

      // FPS counter
      this._frameCount++;
      if (now - this._fpsInterval >= 500) {
        this._fps = Math.round((this._frameCount * 1000) / (now - this._fpsInterval));
        this._frameCount = 0;
        this._fpsInterval = now;
        if (this._fpsEl) this._fpsEl.textContent = `${this._fps} FPS`;
      }
    };

    this._animFrameId = requestAnimationFrame(loop);
  }

  /**
   * P2-5 脏源轮询：相机指纹（拖拽/飞行/滚轮）、场景内容指纹（图元与
   * 实体增删、瓦片/地形到达）、动态实体变更（CallbackProperty 每帧变
   * 化经 updateDynamicEntities 检出）、活动特效（视为持续动画）。
   */
  private _pollDirtySources(timeSec: number): void {
    if (this._scene.updateDynamicEntities(timeSec)) this._scheduler.markDirty();

    const cam = this._scene.camera;
    const camFp = [
      cam.position.x, cam.position.y, cam.position.z,
      cam.direction.x, cam.direction.y, cam.direction.z,
      cam.up.x, cam.up.y, cam.up.z,
      cam.frustum.fov, cam.inFlight ? 1 : 0,
    ].join(',');
    if (camFp !== this._lastCameraFp) {
      this._lastCameraFp = camFp;
      this._scheduler.markDirty();
    }

    const globe = this._scene.globe;
    const sceneFp = [
      this._scene.primitives.opaque.length,
      this._scene.primitives.transparent.length,
      this._scene.entities.values.length,
      globe.tilePrimitiveCount,
      globe.terrainLayer ? globe.terrainLayer.meshCount : 0,
      this._scene.effects.list.length,
    ].join(',');
    if (sceneFp !== this._lastSceneFp) {
      this._lastSceneFp = sceneFp;
      this._scheduler.markDirty();
    }

    // 特效（粒子/天气/水面等）有逐帧动画：挂载即持续脏。
    if (this._scene.effects.list.length > 0) this._scheduler.markDirty();
  }

  private _setupInputHandlers(): void {
    // Resize observer
    this._resizeObserver = new ResizeObserver(() => this._resizeCanvas());
    this._resizeObserver.observe(this._container);

    // All raw DOM input goes through the event abstraction layer (T1.2).
    const handler = new ScreenSpaceEventHandler(this._canvas);
    this._handler = handler;
    const camera = this._scene.camera;

    // ── Left button / one-finger drag → orbit (Cesium-compatible) ──────────
    handler.setInputAction((pos) => {
      if (this.drawMode) return; // drawing tools own pointer input (T2.5)
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      this._isDragging = true;
      this._lastMouseX = pos.x;
      this._lastMouseY = pos.y;
    }, 'LEFT_DOWN');

    handler.setInputAction(() => {
      this._isDragging = false;
    }, 'LEFT_UP');

    // ── Mouse move: orbit / middle pivot tilt / right-drag zoom ────────────
    handler.setInputAction((pos) => {
      if (this._isDragging) {
        const dx = pos.x - this._lastMouseX;
        const dy = pos.y - this._lastMouseY;
        this._lastMouseX = pos.x;
        this._lastMouseY = pos.y;

        // Orbit sensitivity follows *altitude* (Cesium-like finger-tracking):
        // 旧 r·0.0003 rad/px 恒 ≈1913 m/px 与高度无关——300m 低空拖 12 步白膜
        // 出画（97 页「一拖动就看不到白膜」根因，2026-09-29）。
        const sensitivity = Camera.orbitSensitivity(
          camera.position,
          camera.frustum.fov,
          this._canvas.clientHeight,
        );
        // dy sign matches Cesium: drag DOWN → camera orbits northward (positive deltaLat)
        camera.rotate(-dx * sensitivity, dy * sensitivity);
      }

      if (this._isMiddleDragging && this._middlePivot) {
        const dx = pos.x - this._lastMiddleMouseX;
        const dy = pos.y - this._lastMiddleMouseY;
        this._lastMiddleMouseX = pos.x;
        this._lastMiddleMouseY = pos.y;
        // dx > 0 (right) → negative heading  (Cesium-compatible)
        // dy > 0 (down)  → positive pitch delta (tilt away from horizon)
        camera.orbitAroundPivot(this._middlePivot, -dx * 0.005, -dy * 0.005);
      }

      if (this._isRightDragging) {
        const dy = pos.y - this._lastRightMouseY;
        this._lastRightMouseY = pos.y;
        // Right-drag down → zoom in (matches Cesium: drag toward globe = move closer).
        // zoom 吃 ln 缩放因子：拖 100px ≈ ×1.35
        camera.zoom(-dy * 0.003);
      }
    }, 'MOUSE_MOVE');

    // ── Middle button: pitch/heading orbit around the hit point on the globe
    handler.setInputAction((pos) => {
      if (this.drawMode) return; // drawing tools own pointer input (T2.5)
      const ndcX = (pos.x / this._canvas.clientWidth) * 2 - 1;
      const ndcY = 1 - (pos.y / this._canvas.clientHeight) * 2;
      // Pick the globe at the click position; fall back to screen centre
      //（水下/朝天视角拾取不到地表——2026-09-30 用户「中键不能拖动视角」
      // 水下实锤：再兜底视线前方 10 m 的 look-at 点，任何机位中键都能转）
      const lookAhead = new Cartesian3(
        camera.position.x + camera.direction.x * (10 / Camera.EARTH_SCALE),
        camera.position.y + camera.direction.y * (10 / Camera.EARTH_SCALE),
        camera.position.z + camera.direction.z * (10 / Camera.EARTH_SCALE),
      );
      this._middlePivot = camera.pickGlobe(ndcX, ndcY) ?? camera.pickGlobe(0, 0) ?? lookAhead;
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      this._isMiddleDragging = true;
      this._lastMiddleMouseX = pos.x;
      this._lastMiddleMouseY = pos.y;
    }, 'MIDDLE_DOWN');

    handler.setInputAction(() => {
      this._isMiddleDragging = false;
      this._middlePivot = null;
    }, 'MIDDLE_UP');

    // ── Right button drag → zoom ───────────────────────────────────────────
    handler.setInputAction((pos) => {
      if (this.drawMode) return; // drawing tools own pointer input (T2.5)
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      this._isRightDragging = true;
      this._lastRightMouseY = pos.y;
    }, 'RIGHT_DOWN');

    handler.setInputAction(() => {
      this._isRightDragging = false;
    }, 'RIGHT_UP');

    // ── Scroll wheel → proportional zoom (Cesium-compatible) ──────────────
    handler.setInputAction((pos: WheelPosition) => {
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      // Scroll up (deltaY < 0) → zoom in; scroll down → zoom out.
      // zoom 吃 ln 缩放因子：一格(±100) ≈ ×0.8 / ×1.25，锚定构图不滑出画面
      camera.zoom(pos.deltaY * 0.0022);
    }, 'WHEEL');

    // ── Two-finger pinch → proportional zoom ───────────────────────────────
    handler.setInputAction((pos: PinchPosition) => {
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      // Pinch apart → zoom in (negative delta), pinch together → zoom out
      camera.zoom((pos.lastDist - pos.dist) * 0.004);
    }, 'PINCH');

    // ── GPU pick on left click (only when a consumer registered onPick) ─────
    handler.setInputAction((pos: ScreenSpacePosition) => {
      if (!this.onPick) return;
      // Handler reports CSS pixels; scene.pick converts them to texels.
      this._scene
        .pick(pos.x, pos.y)
        .then((result) => this.onPick?.(result, pos))
        .catch((err: unknown) => console.error('[Viewer] GPU pick failed:', err));
    }, 'LEFT_CLICK');
  }
}

// Allow `camera.position.set(x, y, z)` by augmenting Cartesian3
declare module '../math/Cartesian3' {
  interface Cartesian3 {
    set(x: number, y: number, z: number): void;
  }
}

Cartesian3.prototype.set = function(x: number, y: number, z: number): void {
  this.x = x; this.y = y; this.z = z;
};

export default Viewer;
