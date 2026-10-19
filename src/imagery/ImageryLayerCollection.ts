import { ImageryLayer } from './ImageryLayer';
import { ImageryProvider } from './ImageryProvider';
import { CompositeImageryProvider } from './CompositeImageryProvider';

/**
 * ImageryLayerCollection - Ordered set of {@link ImageryLayer}s composited
 * onto the globe (matches Cesium's ImageryLayerCollection, simplified).
 *
 * Index 0 is the bottom of the stack; higher indices composite on top.
 *
 * Render path ({@link ImageryLayerCollection.renderLayer}): a **single**
 * visible layer renders as-is (the historical activeLayer path, zero change);
 * **two or more** visible layers render through a synthetic composite layer
 * whose provider merges every child per tile (see CompositeImageryProvider —
 * P3 fix for the "base + labels only shows labels" defect).
 */
export class ImageryLayerCollection {
  private _layers: ImageryLayer[] = [];
  private _compositeLayer: ImageryLayer | null = null;
  private _compositeSignature = '';

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
    this._compositeSignature = ''; // 可见集合可能变化：合成层失效
    return layer;
  }

  /**
   * Remove a layer from the stack. Returns false when absent.
   * @param destroy - release the layer's GPU textures (default `true`,
   *   CesiumJS semantics). Pass `false` when the layer will be re-added
   *   later (e.g. stack reordering) — a destroyed layer cannot render again.
   */
  remove(layer: ImageryLayer, destroy = true): boolean {
    const index = this._layers.indexOf(layer);
    if (index === -1) return false;
    this._layers.splice(index, 1);
    this._compositeSignature = '';
    if (destroy) layer.destroy();
    return true;
  }

  /** Remove and destroy every layer. */
  removeAll(): void {
    for (const layer of this._layers) layer.destroy();
    this._layers = [];
    this._destroyComposite();
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
    this._compositeSignature = ''; // 栈序变化：合成顺序失效
  }

  /**
   * The layer the globe currently renders: the topmost visible layer,
   * or `null` when no layer is visible.
   */

  private _destroyComposite(): void {
    this._compositeLayer?.destroy();
    this._compositeLayer = null;
    this._compositeSignature = '';
  }

  /** 可见层签名（身份×顺序）：变化即重建合成层（覆盖 show 切换）。 */
  private _visibleSignature(visible: ImageryLayer[]): string {
    return visible.map((l) => this._layers.indexOf(l)).join(',');
  }

  /**
   * The layer the globe renders. Resolution:
   * - 0 visible → null;
   * - 1 visible → that layer itself（原 activeLayer 语义，零改动路径）;
   * - ≥2 visible → a composite layer rebuilt whenever the visible set or
   *   order changes（add / remove / raiseToTop / show 翻转——签名比对捕获）.
   */
  get renderLayer(): ImageryLayer | null {
    const visible = this._layers.filter((l) => l.show);
    if (visible.length === 0) {
      this._destroyComposite();
      return null;
    }
    if (visible.length === 1) {
      this._destroyComposite();
      return visible[0];
    }
    const signature = this._visibleSignature(visible);
    if (!this._compositeLayer || signature !== this._compositeSignature) {
      this._destroyComposite();
      this._compositeLayer = new ImageryLayer({
        provider: new CompositeImageryProvider(visible),
      });
      this._compositeSignature = signature;
    }
    return this._compositeLayer;
  }

  /**
   * Member layers of the active composite (bottom→top); empty on the
   * single-layer path. The globe skips per-member update() for these —
   * the composite drives their providers directly（避免双层取瓦片）.
   */
  compositedMembers(): readonly ImageryLayer[] {
    const visible = this._layers.filter((l) => l.show);
    if (visible.length < 2) return [];
    const signature = this._visibleSignature(visible);
    return signature === this._compositeSignature && this._compositeLayer
      ? (this._compositeLayer.provider as CompositeImageryProvider).layers
      : visible;
  }
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
    this._destroyComposite();
  }
}

export default ImageryLayerCollection;
