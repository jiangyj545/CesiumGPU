/**
 * GoogleTilesProvider – Google Photorealistic 3D Tiles 数据源封装。
 *
 * Google 3D Tiles API 的根清单：
 *   https://tile.googleapis.com/v1/3dtiles/root.json?key=<API_KEY>
 * 返回的 tileset.json 里所有子请求 URI（tile json / glb / 纹理）都是
 * tile.googleapis.com 绝对地址且**不带 key**，每个请求必须自动补
 * `key=<API_KEY>`（缺 key 直接 401/403）。
 *
 * 实现方式：内部组合一个 `Tileset3D`，用 `urlDecorator` 钩子拦截全部
 * 请求 URL（含 tileset.json 本身）做 query 注入；ready/update/stats/
 * destroy 全部透传。注意 Google 的 URI 通常已带 `session=` 等 query，
 * 注入时按已有 `?` 选择 `&` 拼接。
 *
 * 运行前置：有效 Google Maps API key（启用 Map Tiles API）+ 外网连通
 * （Google 官方域允许浏览器直连 CORS）。
 */

import { Tileset3D } from './Tileset3D';
import type { Tileset3DStats } from './Tileset3D';
import type { Scene } from '../scene/Scene';
import type { Camera } from '../scene/Camera';

const GOOGLE_ROOT_URL = 'https://tile.googleapis.com/v1/3dtiles/root.json';

export interface GoogleTiles3DOptions {
  /** 细分屏幕误差阈值（像素）。默认 16。 */
  maximumSseError?: number;
  /** 最大树深。默认 Infinity。 */
  maximumLevel?: number;
  /** LRU 容量。默认 64。 */
  cacheCapacity?: number;
  /** 并发请求上限。默认 8。 */
  requestConcurrency?: number;
}

/**
 * GoogleTiles3D – Google 摄影实景 3DTiles（透传 Tileset3D 语义）。
 *
 * ```typescript
 * const google = new GoogleTiles3D(apiKey, viewer.scene);
 * await google.ready();
 * viewer.preRender = () => google.update(viewer.camera, viewer.canvas.height);
 * ```
 */
export class GoogleTiles3D {
  private readonly _tileset: Tileset3D;
  private readonly _apiKey: string;

  constructor(apiKey: string, scene: Scene, options?: GoogleTiles3DOptions) {
    if (!apiKey || apiKey.trim().length === 0) {
      throw new Error('GoogleTiles3D: apiKey 不能为空');
    }
    this._apiKey = apiKey.trim();

    this._tileset = new Tileset3D({
      url: `${GOOGLE_ROOT_URL}?key=${encodeURIComponent(this._apiKey)}`,
      scene,
      maximumSseError: options?.maximumSseError,
      maximumLevel: options?.maximumLevel,
      cacheCapacity: options?.cacheCapacity,
      requestConcurrency: options?.requestConcurrency,
      // 每请求 key 注入：已有 query 用 &，否则用 ?（Google URI 常带 session=）
      urlDecorator: (url: string) => this._injectKey(url),
    });
  }

  /** 根清单解析完成（网络失败/403 会 reject——调用方展示明确错误）。 */
  ready(): Promise<void> {
    return this._tileset.ready();
  }

  /** 每帧调度（同 Tileset3D.update）。 */
  update(camera: Camera, viewportHeight: number = 1080): void {
    this._tileset.update(camera, viewportHeight);
  }

  /** 调度统计透传。 */
  get stats(): Tileset3DStats {
    return this._tileset.stats;
  }

  /** 树根（ready 后非 null）。 */
  get root() {
    return this._tileset.root;
  }

  /** 卸载全部瓦片资源。 */
  destroy(): void {
    this._tileset.destroy();
  }

  /** key 注入（幂等：已带本 key 的 URL 原样返回）。 */
  private _injectKey(url: string): string {
    if (url.includes(`key=${this._apiKey}`)) return url;
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}key=${encodeURIComponent(this._apiKey)}`;
  }
}
