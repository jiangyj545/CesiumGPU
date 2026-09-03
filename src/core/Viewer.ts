import { Context } from '../renderer/Context';
import { Scene } from '../scene/Scene';
import { Cartesian3 } from '../math/Cartesian3';
import { Camera } from '../scene/Camera';
import { Globe } from '../scene/Globe';
import { Color } from '../math/Color';
import { ScreenSpaceEventHandler } from './ScreenSpaceEventHandler';
import type { PinchPosition, WheelPosition, ScreenSpacePosition } from './ScreenSpaceEventHandler';
import type { PickResult } from '../picking/PickPass';

export interface ViewerOptions {
  /** GPU power preference. Default: 'high-performance' */
  powerPreference?: GPUPowerPreference;
  /** Initial background color (space). */
  backgroundColor?: Color;
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

  private _initialized = false;
  private _destroyed = false;

  constructor(containerOrId: string | HTMLElement, _options: ViewerOptions = {}) {
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
    if (this._initialized) return;

    await this._context.initialize();
    this._resizeCanvas();
    await this._scene.initialize();
    this._initialized = true;

    // Default camera position: 3× Earth radius above equator
    this._scene.camera.position.set(0, 0, 3.0);
    this._scene.camera.setAspectRatio(this._canvas.width / this._canvas.height);

    this._setupInputHandlers();
    this._startRenderLoop();
  }

  /**
   * Stop rendering and release all GPU resources.
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    if (this._animFrameId !== null) cancelAnimationFrame(this._animFrameId);
    this._handler?.destroy();
    this._handler = null;
    this._scene.destroy();
    this._context.destroy();
    this._canvas.remove();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _resizeCanvas(): void {
    const w = this._container.clientWidth  || window.innerWidth;
    const h = this._container.clientHeight || window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    this._canvas.width  = Math.round(w * dpr);
    this._canvas.height = Math.round(h * dpr);
    this._scene.resize(this._canvas.width, this._canvas.height);
  }

  private _startRenderLoop(): void {
    this._fpsInterval = performance.now();

    const loop = (now: number) => {
      if (this._destroyed) return;
      this._animFrameId = requestAnimationFrame(loop);

      // Pre-render hook (clock tick, sun direction update, etc.)
      this.preRender?.();

      // Camera flight (T2.6): advance the active flyTo by the frame delta time
      const dt = this._lastFrameTime === null ? 0 : Math.min((now - this._lastFrameTime) / 1000, 0.1);
      this._lastFrameTime = now;
      this._scene.camera.updateFlight(dt);

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

  private _setupInputHandlers(): void {
    // Resize observer
    const ro = new ResizeObserver(() => this._resizeCanvas());
    ro.observe(this._container);

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
      // Orbit sensitivity scales with camera distance (Cesium behaviour)
      const r = Cartesian3.magnitude(camera.position);

      if (this._isDragging) {
        const dx = pos.x - this._lastMouseX;
        const dy = pos.y - this._lastMouseY;
        this._lastMouseX = pos.x;
        this._lastMouseY = pos.y;

        const sensitivity = r * 0.0003;
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
        // Right-drag down → zoom in (matches Cesium: drag toward globe = move closer)
        camera.zoom(-dy * r * 0.002);
      }
    }, 'MOUSE_MOVE');

    // ── Middle button: pitch/heading orbit around the hit point on the globe
    handler.setInputAction((pos) => {
      if (this.drawMode) return; // drawing tools own pointer input (T2.5)
      const ndcX = (pos.x / this._canvas.clientWidth) * 2 - 1;
      const ndcY = 1 - (pos.y / this._canvas.clientHeight) * 2;
      // Pick the globe at the click position; fall back to screen centre
      this._middlePivot = camera.pickGlobe(ndcX, ndcY) ?? camera.pickGlobe(0, 0);
      if (this._middlePivot) {
        camera.cancelFlight(); // user input interrupts flyTo (T2.6)
        this._isMiddleDragging = true;
        this._lastMiddleMouseX = pos.x;
        this._lastMiddleMouseY = pos.y;
      }
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
      const r = Cartesian3.magnitude(camera.position);
      // Scroll up (deltaY < 0) → zoom in; scroll down → zoom out
      camera.zoom(pos.deltaY * r * 0.0002);
    }, 'WHEEL');

    // ── Two-finger pinch → proportional zoom ───────────────────────────────
    handler.setInputAction((pos: PinchPosition) => {
      camera.cancelFlight(); // user input interrupts flyTo (T2.6)
      const r = Cartesian3.magnitude(camera.position);
      // Pinch apart → zoom in (negative delta), pinch together → zoom out
      camera.zoom((pos.lastDist - pos.dist) * r * 0.002);
    }, 'PINCH');

    // ── GPU pick on left click (only when a consumer registered onPick) ─────
    handler.setInputAction((pos: ScreenSpacePosition) => {
      if (!this.onPick) return;
      // Handler reports CSS pixels; picking uses canvas device pixels.
      const sx = this._canvas.width  / Math.max(1, this._canvas.clientWidth);
      const sy = this._canvas.height / Math.max(1, this._canvas.clientHeight);
      this._scene
        .pick(pos.x * sx, pos.y * sy)
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
