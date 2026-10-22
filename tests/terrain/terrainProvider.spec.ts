import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  HttpTerrainProviderBase,
  TerrainMeshData,
} from '../../src/terrain/TerrainProvider';
import { LocalQuantizedMeshProvider } from '../../src/terrain/LocalQuantizedMeshProvider';
import { TerrariumProvider } from '../../src/terrain/TerrariumProvider';

/** decodeTile 直接返回合成网格的桩实现（绕开网络解码）。 */
class StubProvider extends HttpTerrainProviderBase {
  decodeCalls = 0;

  constructor(maximumLevel = 10) {
    super({
      url: 'http://mock/{z}/{x}/{y}',
      tilingScheme: 'geographic',
      maximumLevel,
    });
  }

  protected override async decodeTile(): Promise<TerrainMeshData | null> {
    this.decodeCalls++;
    return {
      positions: new Float32Array([1, 0, 0]),
      indices: new Uint16Array([0, 0, 0]),
      minHeight: 0,
      maxHeight: 0,
      center: [0, 0, 0],
    };
  }
}

/** 带 {s} 子域的桩，验证模板轮转。 */
class SubdomainProvider extends HttpTerrainProviderBase {
  constructor() {
    super({
      url: 'https://{s}.example.com/{z}/{x}/{y}',
      tilingScheme: 'geographic',
      maximumLevel: 5,
      subdomains: ['a', 'b', 'c'],
    });
  }

  protected override async decodeTile(): Promise<TerrainMeshData | null> {
    return null;
  }
}

function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('terrainProvider', () => {
  describe('URL 模板替换', () => {
    it('LocalQuantizedMeshProvider 默认模板与参数', () => {
      const provider = new LocalQuantizedMeshProvider();
      expect(provider.url).toBe(
        'http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/{z}/{x}/{y}.terrain'
      );
      expect(provider.resolveUrl(3, 2, 1)).toBe(
        'http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/3/2/1.terrain'
      );
      expect(provider.tilingScheme).toBe('geographic');
      // 12：7777 terrain_hubei 的 CTB 构建范围（max_zoom=12）；请求 z13+ 会触发
      // 服务端 429 限流 → 地形瓦片 30s 重试循环 → 可见的精细度周期闪烁。
      expect(provider.maximumLevel).toBe(12);
      expect(provider.ready).toBe(true);
    });

    it('TerrariumProvider 默认模板与参数', () => {
      const provider = new TerrariumProvider();
      expect(provider.url).toBe(
        'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
      );
      expect(provider.resolveUrl(1, 0, 1)).toBe(
        'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/1/0/1.png'
      );
      expect(provider.tilingScheme).toBe('webmercator');
      expect(provider.maximumLevel).toBe(15);
    });

    it('{s} 子域按 (x+y+z) 确定性轮转', () => {
      const provider = new SubdomainProvider();
      expect(provider.resolveUrl(0, 0, 0)).toBe(
        'https://a.example.com/0/0/0'
      );
      expect(provider.resolveUrl(0, 1, 1)).toBe(
        'https://c.example.com/0/1/1'
      );
    });
  });

  describe('失败缓存', () => {
    it('网络失败后同瓦片会话内不再重试', async () => {
      const fetchMock = vi.fn(async () => {
        throw new TypeError('network down');
      });
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      expect(await provider.requestTileGeometry(1, 2, 3)).toBeNull();
      expect(await provider.requestTileGeometry(1, 2, 3)).toBeNull();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(provider.getTileDataAvailable(1, 2, 3)).toBe(false);
    });

    it('HTTP 404 视为无数据并缓存', async () => {
      const fetchMock = vi.fn(
        async () => new Response('nope', { status: 404 })
      );
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      expect(await provider.requestTileGeometry(2, 0, 1)).toBeNull();
      expect(await provider.requestTileGeometry(2, 0, 1)).toBeNull();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(provider.decodeCalls).toBe(0);
    });

    it('成功路径不进失败缓存，不同瓦片不受影响', async () => {
      const fetchMock = vi.fn(
        async () => new Response(new ArrayBuffer(8), { status: 200 })
      );
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      const mesh = await provider.requestTileGeometry(0, 0, 0);
      expect(mesh).not.toBeNull();
      expect(mesh?.positions).toBeInstanceOf(Float32Array);
      expect(provider.getTileDataAvailable(0, 0, 0)).toBe(true);

      expect(await provider.requestTileGeometry(0, 1, 0)).not.toBeNull();
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it('超过 maximumLevel 直接返回 null 且不发请求', async () => {
      const fetchMock = vi.fn(
        async () => new Response(new ArrayBuffer(8), { status: 200 })
      );
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider(10);

      expect(await provider.requestTileGeometry(11, 0, 0)).toBeNull();
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  describe('并发上限', () => {
    it('在飞请求最多 4 个，释放后队列继续', async () => {
      const gates = new Map<
        string,
        { promise: Promise<Response>; resolve: (v: Response) => void }
      >();
      const fetchMock = vi.fn((url: string) => {
        // 前 4 个请求挂起以占满并发槽；放行后的新请求立即成功
        if (gates.size < 4) {
          const gate = deferred<Response>();
          gates.set(url, gate);
          return gate.promise;
        }
        return Promise.resolve(new Response(new ArrayBuffer(8), { status: 200 }));
      });
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      const requests = Array.from({ length: 6 }, (_, i) =>
        provider.requestTileGeometry(0, i, 0)
      );

      // 前 4 个立即发出，后 2 个排队
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
      expect(gates.size).toBe(4);

      // 释放全部在飞请求 → 队列中的 2 个被放行并立即成功
      for (const gate of gates.values()) {
        gate.resolve(new Response(new ArrayBuffer(8), { status: 200 }));
      }
      const results = await Promise.all(requests);
      expect(fetchMock).toHaveBeenCalledTimes(6);
      expect(results).toHaveLength(6);
      expect(results.every((mesh) => mesh !== null)).toBe(true);
    });
  });

  describe('destroy', () => {
    it('destroy 后新请求直接返回 null，且幂等', async () => {
      const fetchMock = vi.fn(
        async () => new Response(new ArrayBuffer(8), { status: 200 })
      );
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      provider.destroy();
      provider.destroy(); // 幂等
      expect(await provider.requestTileGeometry(0, 0, 0)).toBeNull();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('destroy 中止在途请求并放行排队请求（全部 null）', async () => {
      const signals: AbortSignal[] = [];
      const fetchMock = vi.fn(
        (_url: string, init?: { signal?: AbortSignal }) =>
          new Promise<Response>((_resolve, reject) => {
            if (init?.signal) {
              signals.push(init.signal);
              init.signal.addEventListener('abort', () =>
                reject(new DOMException('Aborted', 'AbortError'))
              );
            }
          })
      );
      vi.stubGlobal('fetch', fetchMock);
      const provider = new StubProvider();

      const requests = Array.from({ length: 6 }, (_, i) =>
        provider.requestTileGeometry(0, i, 0)
      );
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));

      provider.destroy();
      const results = await Promise.all(requests);
      expect(results.every((mesh) => mesh === null)).toBe(true);
      // 4 个在飞请求全部收到中止信号
      expect(signals).toHaveLength(4);
      expect(signals.every((s) => s.aborted)).toBe(true);
      // destroy 后不再发起新的 fetch
      expect(fetchMock).toHaveBeenCalledTimes(4);
    });
  });
});
