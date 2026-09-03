/**
 * Event - a minimal publish/subscribe event, mirroring the subset of
 * Cesium's `Event` API used by the data-source layer.
 */
export class Event<Args extends unknown[] = unknown[]> {
  private _listeners: Array<(...args: Args) => void> = [];

  /** Number of currently registered listeners. */
  get numberOfListeners(): number {
    return this._listeners.length;
  }

  /**
   * Register a listener. Listeners are invoked in registration order when
   * {@link Event.raiseEvent} fires.
   */
  addListener(listener: (...args: Args) => void): void {
    this._listeners.push(listener);
  }

  /**
   * Unregister a previously added listener.
   * @returns true if the listener was found and removed.
   */
  removeCallback(listener: (...args: Args) => void): boolean {
    const index = this._listeners.indexOf(listener);
    if (index === -1) return false;
    this._listeners.splice(index, 1);
    return true;
  }

  /** Raise the event, invoking all registered listeners with the given arguments. */
  raiseEvent(...args: Args): void {
    // Copy so listeners may safely add/remove listeners while raising.
    const listeners = this._listeners.slice();
    for (const listener of listeners) {
      listener(...args);
    }
  }
}

export default Event;
