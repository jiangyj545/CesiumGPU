import { ImageryLayer } from './ImageryLayer';
import { ImageryProvider } from './ImageryProvider';

/**
 * ImageryLayerCollection - Ordered set of {@link ImageryLayer}s composited
 * onto the globe (matches Cesium's ImageryLayerCollection, simplified).
 *
 * Index 0 is the bottom of the stack; higher indices composite on top.
 * The current opaque render path surfaces the **topmost visible** layer
 * (see {@link ImageryLayerCollection.activeLayer}); per-layer alpha blending
 * is reserved for a later compositing pass.
 */
export class ImageryLayerCollection {
  private _layers: ImageryLayer[] = [];

  /** Number of layers in the collection. */
  get length(): number {
    return this._layers.length;
  }

  /**
   * Create a layer from a provider and push it on top of the stack.
   * An existing {@link ImageryLayer} may be passed instead to reuse its
   * loaded tiles and settings.
   */
  addImageryProvider(
    provider: ImageryProvider | ImageryLayer,
    options: Omit<ConstructorParameters<typeof ImageryLayer>[0], 'provider'> = {}
  ): ImageryLayer {
    const layer =
      provider instanceof ImageryLayer
        ? provider
        : new ImageryLayer({ ...options, provider: provider as ImageryProvider });
    this._layers.push(layer);
    return layer;
  }

  /** Remove a layer (and release its GPU textures). Returns false when absent. */
  remove(layer: ImageryLayer): boolean {
    const index = this._layers.indexOf(layer);
    if (index === -1) return false;
    this._layers.splice(index, 1);
    layer.destroy();
    return true;
  }

  /** Remove and destroy every layer. */
  removeAll(): void {
    for (const layer of this._layers) layer.destroy();
    this._layers = [];
  }

  /** Layer at `index` (0 = bottom of the stack). */
  get(index: number): ImageryLayer {
    return this._layers[index];
  }

  /** Index of a layer, or -1 when absent. */
  indexOf(layer: ImageryLayer): number {
    return this._layers.indexOf(layer);
  }

  /** Move a layer to the top of the stack (renders over everything else). */
  raiseToTop(layer: ImageryLayer): void {
    const index = this._layers.indexOf(layer);
    if (index === -1 || index === this._layers.length - 1) return;
    this._layers.splice(index, 1);
    this._layers.push(layer);
  }

  /**
   * The layer the globe currently renders: the topmost visible layer,
   * or `null` when no layer is visible.
   */
  get activeLayer(): ImageryLayer | null {
    for (let i = this._layers.length - 1; i >= 0; i--) {
      if (this._layers[i].show) return this._layers[i];
    }
    return null;
  }

  [Symbol.iterator](): Iterator<ImageryLayer> {
    return this._layers[Symbol.iterator]();
  }

  /** Destroy every layer (GPU textures). */
  destroy(): void {
    this.removeAll();
  }
}

export default ImageryLayerCollection;
