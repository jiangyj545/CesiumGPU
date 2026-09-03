/**
 * ScreenSpaceEventHandler - DOM input abstraction for the viewer canvas.
 *
 * Semantically mirrors Cesium's `ScreenSpaceEventHandler`: instead of attaching
 * raw DOM listeners, consumers register position-based callbacks per event
 * type.  Mouse, wheel and touch input are normalised into a single stream of
 * canvas-space positions:
 *
 * - Single touch → LEFT_* / MOUSE_MOVE (same stream as the mouse)
 * - Two touches  → PINCH
 * - Mouse wheel  → WHEEL (payload includes `deltaY`)
 * - A press/release pair with < 4 px of travel inside 500 ms becomes a click
 * - Two clicks inside 500 ms additionally emit LEFT_DOUBLE_CLICK
 */

/** Input event types understood by {@link ScreenSpaceEventHandler}. */
export type ScreenSpaceEventType =
  | 'LEFT_CLICK'
  | 'LEFT_DOUBLE_CLICK'
  | 'RIGHT_CLICK'
  | 'MOUSE_MOVE'
  | 'LEFT_DOWN'
  | 'LEFT_UP'
  | 'RIGHT_DOWN'
  | 'RIGHT_UP'
  | 'MIDDLE_DOWN'
  | 'MIDDLE_UP'
  | 'WHEEL'
  | 'PINCH';

/** A position in canvas (CSS) pixel coordinates, origin at the top-left corner. */
export interface ScreenSpacePosition {
  x: number;
  y: number;
}

/** Payload delivered for {@link ScreenSpaceEventType} `WHEEL` actions. */
export interface WheelPosition extends ScreenSpacePosition {
  deltaY: number;
}

/** Payload delivered for {@link ScreenSpaceEventType} `PINCH` actions. */
export interface PinchPosition extends ScreenSpacePosition {
  /** Current distance between the two touches, in CSS pixels. */
  dist: number;
  /** Distance between the two touches at the previous PINCH event. */
  lastDist: number;
}

/** Maximum pointer travel (px) between DOWN and UP for it to count as a click. */
const CLICK_MAX_DIST = 4;
/** Maximum duration (ms) of a DOWN/UP pair for it to count as a click. */
const CLICK_MAX_DURATION = 500;
/** Maximum interval (ms) between two clicks for a double click. */
const DOUBLE_CLICK_MAX_INTERVAL = 500;

type AnyAction = (position: ScreenSpacePosition | WheelPosition | PinchPosition) => void;

interface ListenerEntry {
  target: EventTarget;
  type: string;
  fn: EventListener;
  options?: AddEventListenerOptions;
}

/** Where and when a press started, used for click detection. */
interface PressRecord {
  pos: ScreenSpacePosition;
  time: number;
}

/**
 * Registers per-event-type callbacks for mouse, wheel and touch input on a
 * canvas element.  All coordinates are canvas-space CSS pixels.
 *
 * ```typescript
 * const handler = new ScreenSpaceEventHandler(canvas);
 * handler.setInputAction((pos) => console.log(pos), 'LEFT_CLICK');
 * handler.setInputAction((pos) => console.log(pos.deltaY), 'WHEEL');
 * ```
 */
export class ScreenSpaceEventHandler {
  private readonly _canvas: HTMLCanvasElement;
  private readonly _actions = new Map<ScreenSpaceEventType, AnyAction>();
  private readonly _listeners: ListenerEntry[] = [];

  // Click detection state
  private _leftDown: PressRecord | null = null;
  private _rightDown: PressRecord | null = null;
  private _lastClickTime = 0;

  // Touch state
  private _touchActive = false;
  private _pinchLastDist = 0;

  constructor(canvas: HTMLCanvasElement) {
    this._canvas = canvas;
    this._installListeners();
  }

  /**
   * Register (or replace) the action invoked when `type` occurs.
   * MOUSE_MOVE / WHEEL / PINCH fire continuously; *_CLICK fire on taps.
   */
  setInputAction(action: (position: ScreenSpacePosition) => void, type: ScreenSpaceEventType): void;
  setInputAction(action: (position: WheelPosition) => void, type: 'WHEEL'): void;
  setInputAction(action: (position: PinchPosition) => void, type: 'PINCH'): void;
  // Implementation signature: `never` accepts every action shape (contravariance).
  setInputAction(action: (position: never) => void, type: ScreenSpaceEventType): void {
    this._actions.set(type, action as AnyAction);
  }

  /** Remove the action registered for `type`. */
  removeInputAction(type: ScreenSpaceEventType): void {
    this._actions.delete(type);
  }

  /** Detach every DOM listener and drop all registered actions. */
  destroy(): void {
    for (const { target, type, fn, options } of this._listeners) {
      target.removeEventListener(type, fn, options);
    }
    this._listeners.length = 0;
    this._actions.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _installListeners(): void {
    const canvas = this._canvas;

    // Mouse presses start on the canvas; move/up are tracked on `window` so a
    // drag that leaves the canvas keeps working (Cesium behaviour).
    this._listen(canvas, 'mousedown', (e: MouseEvent) => this._onMouseDown(e));
    this._listen(window, 'mousemove', (e: MouseEvent) => this._onMouseMove(e), { passive: true });
    this._listen(window, 'mouseup',   (e: MouseEvent) => this._onMouseUp(e),   { passive: true });

    // Right click must not open the browser context menu over the canvas.
    this._listen(canvas, 'contextmenu', (e: Event) => e.preventDefault());

    // Wheel zooms the camera, so the page must not scroll instead.
    this._listen(canvas, 'wheel', (e: WheelEvent) => this._onWheel(e), { passive: false });

    // Touch: one finger maps to the LEFT mouse stream, two fingers to PINCH.
    this._listen(canvas, 'touchstart',  (e: TouchEvent) => this._onTouchStart(e), { passive: true });
    this._listen(canvas, 'touchmove',   (e: TouchEvent) => this._onTouchMove(e),  { passive: false });
    this._listen(canvas, 'touchend',    (e: TouchEvent) => this._onTouchEnd(e),   { passive: true });
    this._listen(canvas, 'touchcancel', (e: TouchEvent) => this._onTouchEnd(e),   { passive: true });
  }

  private _listen<T extends Event>(
    target: EventTarget,
    type: string,
    fn: (event: T) => void,
    options?: AddEventListenerOptions,
  ): void {
    const listener = fn as EventListener;
    target.addEventListener(type, listener, options);
    this._listeners.push({ target, type, fn: listener, options });
  }

  private _emit(type: ScreenSpaceEventType, position: ScreenSpacePosition | WheelPosition | PinchPosition): void {
    this._actions.get(type)?.(position);
  }

  private _toCanvas(clientX: number, clientY: number): ScreenSpacePosition {
    const rect = this._canvas.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  // ── Mouse ─────────────────────────────────────────────────────────────────

  private _onMouseDown(e: MouseEvent): void {
    if (this._touchActive) return; // Ignore synthetic mouse events during touch
    const pos = this._toCanvas(e.clientX, e.clientY);

    if (e.button === 0) {
      this._leftDown = { pos, time: performance.now() };
      this._emit('LEFT_DOWN', pos);
    } else if (e.button === 1) {
      // Prevent the middle-button auto-scroll indicator
      e.preventDefault();
      this._emit('MIDDLE_DOWN', pos);
    } else if (e.button === 2) {
      this._rightDown = { pos, time: performance.now() };
      this._emit('RIGHT_DOWN', pos);
    }
  }

  private _onMouseMove(e: MouseEvent): void {
    if (this._touchActive) return;
    this._emit('MOUSE_MOVE', this._toCanvas(e.clientX, e.clientY));
  }

  private _onMouseUp(e: MouseEvent): void {
    if (this._touchActive) return;
    const pos = this._toCanvas(e.clientX, e.clientY);

    if (e.button === 0) {
      this._endLeftPress(this._leftDown, pos);
      this._leftDown = null;
    } else if (e.button === 1) {
      this._emit('MIDDLE_UP', pos);
    } else if (e.button === 2) {
      this._endRightPress(this._rightDown, pos);
      this._rightDown = null;
    }
  }

  private _onWheel(e: WheelEvent): void {
    // The canvas consumes the wheel; the page must not scroll.
    e.preventDefault();
    const pos = this._toCanvas(e.clientX, e.clientY);
    this._emit('WHEEL', { ...pos, deltaY: e.deltaY });
  }

  // ── Touch ─────────────────────────────────────────────────────────────────

  private _onTouchStart(e: TouchEvent): void {
    if (e.touches.length === 1) {
      this._touchActive = true;
      const t = e.touches[0];
      const pos = this._toCanvas(t.clientX, t.clientY);
      this._leftDown = { pos, time: performance.now() };
      this._emit('LEFT_DOWN', pos);
    } else if (e.touches.length >= 2) {
      // Second finger down: cancel any pending click and end the one-finger drag.
      this._leftDown = null;
      const t = e.touches[0];
      this._emit('LEFT_UP', this._toCanvas(t.clientX, t.clientY));
      this._pinchLastDist = this._touchDistance(e.touches);
    }
  }

  private _onTouchMove(e: TouchEvent): void {
    // The page must not scroll/zoom while gesturing on the canvas.
    e.preventDefault();

    if (e.touches.length === 1) {
      const t = e.touches[0];
      this._emit('MOUSE_MOVE', this._toCanvas(t.clientX, t.clientY));
    } else if (e.touches.length >= 2) {
      const dist = this._touchDistance(e.touches);
      const mid = this._touchMidpoint(e.touches);
      this._emit('PINCH', { x: mid.x, y: mid.y, dist, lastDist: this._pinchLastDist });
      this._pinchLastDist = dist;
    }
  }

  private _onTouchEnd(e: TouchEvent): void {
    if (e.touches.length === 0) {
      this._touchActive = false;
      const t = e.changedTouches[0];
      const pos = this._toCanvas(t.clientX, t.clientY);
      this._endLeftPress(this._leftDown, pos);
      this._leftDown = null;
    }
    // 2 → 1 fingers: stay in pinch mode; the remaining finger is ignored until
    // every finger is lifted (matches the previous viewer behaviour).
  }

  // ── Click detection ───────────────────────────────────────────────────────

  /** Emit LEFT_UP plus, when the press was short and stationary, click events. */
  private _endLeftPress(down: PressRecord | null, pos: ScreenSpacePosition): void {
    this._emit('LEFT_UP', pos);
    if (!down || !this._isClick(down, pos)) return;

    const now = performance.now();
    this._emit('LEFT_CLICK', pos);
    if (now - this._lastClickTime < DOUBLE_CLICK_MAX_INTERVAL) {
      this._emit('LEFT_DOUBLE_CLICK', pos);
    }
    this._lastClickTime = now;
  }

  /** Emit RIGHT_UP plus RIGHT_CLICK when the press was short and stationary. */
  private _endRightPress(down: PressRecord | null, pos: ScreenSpacePosition): void {
    this._emit('RIGHT_UP', pos);
    if (!down || !this._isClick(down, pos)) return;
    this._emit('RIGHT_CLICK', pos);
  }

  private _isClick(down: PressRecord, up: ScreenSpacePosition): boolean {
    const dx = up.x - down.pos.x;
    const dy = up.y - down.pos.y;
    return dx * dx + dy * dy < CLICK_MAX_DIST * CLICK_MAX_DIST
      && performance.now() - down.time < CLICK_MAX_DURATION;
  }

  private _touchDistance(touches: TouchList): number {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private _touchMidpoint(touches: TouchList): ScreenSpacePosition {
    const x = (touches[0].clientX + touches[1].clientX) / 2;
    const y = (touches[0].clientY + touches[1].clientY) / 2;
    return this._toCanvas(x, y);
  }
}
